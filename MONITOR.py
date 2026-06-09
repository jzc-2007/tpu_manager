import os, sys, subprocess
import json
import yaml
import time
import re
import ast
import argparse
import uuid
import shlex
from collections import deque
import datetime
import utils.data_io as data_io
from utils.constants import (
    FAIL,
    GAOCHAO,
    GOOD,
    INFO,
    LOG,
    MADE,
    WARNING,
    RESUME_QUEUE_ALLOWED_ZONE_LABEL,
    is_resume_queue_allowed_zone,
)

USER = 'sqa'
SAME_TYPE_ONLY = False

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
TYPE_RE = re.compile(r"^(v[0-9a-z]+-\d+)")
ZONE_RE = re.compile(r"(us|asia|europe|australia|northamerica|southamerica)-[a-z0-9-]+-[a-z]")

_tpu = 'python /home/jzc/zhichengjiang/working/xibo_tpu_manager/tpu.py'
_tou = 'python /kmh-nfs-ssd-us-mount/code/qiao/work/tpu_dls/wrap_master.py'
TOU_MAX_CACHE_AGE_SECONDS = 60
TOU_CACHE_AGE_RE = re.compile(r"audit was\s+([0-9.]+)s ago")
TPU_PROJECT = "he-vision-group"
TPU_ENV_CHECK_TIMEOUT_SECONDS = 300
TPU_JAX_CHECK_REMOTE_CMD = "python -c \"import jax; print('JAX_OK', jax.__version__)\""
_ASIA_RESERVED_ZONE = 'asia-northeast1-b'


class TouCacheStale(Exception):
    """Raised to abort the current monitor loop when `tou` cache is stale."""

    pass


def get_abs_time_str():
    """Return current local time in the timestamp format used by monitor logs."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def convert_utcstr_to_edtstr(utc_str):
    utc_time = datetime.datetime.strptime(utc_str, "%Y-%m-%d %H:%M:%S")
    edt_time = utc_time - datetime.timedelta(hours=4)
    return edt_time.strftime("%Y-%m-%d %H:%M:%S")


def convert_utcstr_to_chnstr(utc_str):
    utc_time = datetime.datetime.strptime(utc_str, "%Y-%m-%d %H:%M:%S")
    chn_time = utc_time + datetime.timedelta(hours=8)
    return chn_time.strftime("%Y-%m-%d %H:%M:%S")

def read_sqa():
    """读取 {USER}.json，返回包含 running / finished / resume_next_round 三个列表的 dict。"""
    with open(f'/kmh-nfs-ssd-us-mount/code/qiao/work/tpu_manager/{USER}.json', 'r') as file:
        x = file.read()
        x = json.loads(x)
    x.setdefault('running', [])
    x.setdefault('finished', [])
    x.setdefault('resume_next_round', [])
    return x

def _write_sqa_content(x):
    """将 sqa dict 写回 sqa.json（内部使用，调用前请先填充三个列表字段）。"""
    x.setdefault('running', [])
    x.setdefault('finished', [])
    x.setdefault('resume_next_round', [])
    with open(f'/kmh-nfs-ssd-us-mount/code/qiao/work/tpu_manager/{USER}.json', 'w') as file:
        json.dump(x, file)
        file.write('\n')

def write_sqa(window_id):
    """将 window_id 加入 sqa.running（表示 MONITOR 正在处理该窗口，防止重入）。"""
    x = read_sqa()
    if window_id not in x['running']:
        x['running'].append(window_id)
    _write_sqa_content(x)
    return

def remove_sqa(window_id):
    """将 window_id 从 sqa.running 移除（resume 失败/跳过时调用，解除锁定）。"""
    x = read_sqa()
    if window_id in x['running']:
        x['running'].remove(window_id)
    _write_sqa_content(x)
    return

def finish_sqa(window_id):
    """将 window_id 从 running 移入 finished（resume 成功后调用）。"""
    x = read_sqa()
    if window_id in x['running']:
        x['running'].remove(window_id)
    if window_id not in x['finished']:
        x['finished'].append(window_id)
    _write_sqa_content(x)
    return

def add_resume_next_round(window_id):
    """将 window_id 加入 resume_next_round 缓冲区：TPU 仍存在但空闲，下一轮再 resume。"""
    x = read_sqa()
    if window_id not in x['resume_next_round']:
        x['resume_next_round'].append(window_id)
    _write_sqa_content(x)
    return

def remove_resume_next_round(window_id):
    """将 window_id 从 resume_next_round 缓冲区移除。"""
    x = read_sqa()
    if window_id in x['resume_next_round']:
        x['resume_next_round'].remove(window_id)
    _write_sqa_content(x)
    return

# ---------------------------------------------------------------------------
# Queue: submit new jobs to run on idle TPUs
# ---------------------------------------------------------------------------
QUEUE_FILE = '/kmh-nfs-ssd-us-mount/code/qiao/work/tpu_manager/queue.json'

def _read_queue():
    """读取 queue.json，返回含 pending / running / done 三个列表的 dict。"""
    if not os.path.exists(QUEUE_FILE):
        return {"pending": [], "running": [], "done": []}
    with open(QUEUE_FILE, 'r') as f:
        q = json.load(f)
    q.setdefault('pending', [])
    q.setdefault('running', [])
    q.setdefault('done', [])
    return q

def _write_queue(q):
    """将 queue dict 写回 queue.json。"""
    with open(QUEUE_FILE, 'w') as f:
        json.dump(q, f, indent=2)
        f.write('\n')

def _parse_queue_priority(priority):
    """Queue priority is intentionally simple: any integer, larger runs first."""
    try:
        return int(priority)
    except (TypeError, ValueError):
        return 0

def _queue_entry_priority(entry):
    return _parse_queue_priority(entry.get('priority', 0))

def _queue_sort_key(entry):
    # Higher priority wins; queued_at/id keep same-priority jobs stable and readable.
    return (-_queue_entry_priority(entry), str(entry.get('queued_at', '')), str(entry.get('id', '')))

def _ensure_queue_entry_metadata(entry, refresh_notes=False):
    """Backfill fields used for queue.json readability without breaking old entries."""
    changed = False
    if 'priority' not in entry:
        entry['priority'] = 0
        changed = True
    else:
        parsed_priority = _parse_queue_priority(entry.get('priority'))
        if entry.get('priority') != parsed_priority:
            entry['priority'] = parsed_priority
            changed = True

    if 'wandb_notes' not in entry or entry.get('wandb_notes') is None or refresh_notes:
        entry['wandb_notes'] = _get_queue_job_wandb_notes(entry.get('dir', ''))
        changed = True
    return changed

def _ensure_queue_metadata(q, refresh_notes=False, sections=('pending', 'running')):
    changed = False
    for section in sections:
        for entry in q.get(section, []):
            changed = _ensure_queue_entry_metadata(entry, refresh_notes=refresh_notes) or changed
    return changed

def queue_add_job(dir_no, preferred_type=None, preferred_zone=None, priority=0):
    """将一个新 job 加入 pending 队列并打印当前队列状态。"""
    q = _read_queue()
    priority = _parse_queue_priority(priority)
    wandb_notes = _get_queue_job_wandb_notes(dir_no)
    entry = {
        "id": uuid.uuid4().hex[:8],
        "dir": str(dir_no),
        "queued_at": get_abs_time_str(),
        "priority": priority,
        "wandb_notes": wandb_notes,
    }
    if preferred_type:
        entry["preferred_type"] = preferred_type
    if preferred_zone:
        entry["preferred_zone"] = preferred_zone
    q['pending'].append(entry)
    _write_queue(q)
    print(
        f"[queue] 已加入队列: dir={dir_no} priority={priority}"
        + (f" type={preferred_type}" if preferred_type else "")
        + (f" zone={preferred_zone}" if preferred_zone else "")
        + (f" wandb_notes={wandb_notes}" if wandb_notes else "")
    )
    print(f"[queue] 当前待跑队列 ({len(q['pending'])} 个): {[e['dir'] for e in q['pending']]}")

def _queue_start_job(job_id, tpu_name, alias):
    """将 job_id 对应的 entry 从 pending 移入 running。"""
    q = _read_queue()
    entry = next((e for e in q['pending'] if e['id'] == job_id), None)
    if entry is None:
        return
    q['pending'].remove(entry)
    running_entry = dict(entry)
    running_entry['tpu'] = tpu_name
    running_entry['alias'] = alias
    running_entry['started_at'] = get_abs_time_str()
    q['running'].append(running_entry)
    _write_queue(q)

def _queue_finish_job(job_id):
    """将 job_id 对应的 running entry 移入 done。"""
    q = _read_queue()
    entry = next((e for e in q['running'] if e['id'] == job_id), None)
    if entry is None:
        return
    q['running'].remove(entry)
    done_entry = dict(entry)
    done_entry['finished_at'] = get_abs_time_str()
    q['done'].append(done_entry)
    _write_queue(q)

def _queue_fail_job(job_id):
    """将 job_id 对应的 running entry 移回 pending 队列头（等待重试）。"""
    q = _read_queue()
    entry = next((e for e in q['running'] if e['id'] == job_id), None)
    if entry is None:
        return
    q['running'].remove(entry)
    pending_entry = {k: v for k, v in entry.items() if k not in ('tpu', 'alias', 'started_at')}
    q['pending'].insert(0, pending_entry)
    _write_queue(q)

def queue_show():
    """打印当前队列状态（pending / running / done）。"""
    q = _read_queue()
    if _ensure_queue_metadata(q):
        _write_queue(q)
    print(f"=== Queue Status ===")
    print(f"Pending ({len(q['pending'])}):")
    for e in sorted(q['pending'], key=_queue_sort_key):
        print(
            f"  dir={e['dir']}  priority={_queue_entry_priority(e)}  queued_at={e['queued_at']}"
            + (f"  type={e.get('preferred_type','')}" if e.get('preferred_type') else "")
            + (f"  zone={e.get('preferred_zone','')}" if e.get('preferred_zone') else "")
            + (f"  notes={e.get('wandb_notes','')}" if e.get('wandb_notes') else "")
        )
    print(f"Running ({len(q['running'])}):")
    for e in q['running']:
        print(
            f"  dir={e['dir']}  priority={_queue_entry_priority(e)}  alias={e.get('alias','')}  "
            f"tpu={e.get('tpu','')}  started_at={e.get('started_at','')}"
            + (f"  notes={e.get('wandb_notes','')}" if e.get('wandb_notes') else "")
        )
    print(f"Done ({len(q['done'])}):")
    for e in q['done'][-10:]:
        print(
            f"  dir={e['dir']}  priority={_queue_entry_priority(e)}  alias={e.get('alias','')}  "
            f"finished_at={e.get('finished_at','')}"
            + (f"  notes={e.get('wandb_notes','')}" if e.get('wandb_notes') else "")
        )

def add_MONITOR_log(log):
    """将日志同时追加写入 output.log 并打印到 stdout（带 UTC 时间戳）。"""
    with open(f'/kmh-nfs-ssd-us-mount/code/qiao/work/tpu_manager/output.log', 'a') as file:
        file.write(f"{get_abs_time_str()}: {log}\n")
    print(f"{get_abs_time_str()}: {log}")
    return

def _append_resume_file_log(window_id, command_name, command, result):
    """将某次 resume/fmd 命令的完整执行结果（cmd、returncode、stdout、stderr）追加写入
    logs/<window_id>/<command_name>.txt，便于事后排查。"""
    log_dir = os.path.join("logs", str(window_id))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{command_name}.txt")

    with open(log_path, "a") as log_file:
        log_file.write(f"[{get_abs_time_str()}] {command_name}\n")
        log_file.write(f"cmd: {command}\n")
        log_file.write(f"returncode: {result.returncode}\n")
        log_file.write("stdout:\n")
        log_file.write((result.stdout or "") + "\n")
        log_file.write("stderr:\n")
        log_file.write((result.stderr or "") + "\n")
        log_file.write("-" * 80 + "\n")

def _append_queue_file_log(job_id, dir_no, command_name, command, result):
    """将 queue 子任务命令执行结果落盘，便于排查 returncode=0 但实际失败的情况。"""
    log_dir = os.path.join("logs", "queue")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"job_{job_id}_dir_{dir_no}.txt")

    with open(log_path, "a") as log_file:
        log_file.write(f"[{get_abs_time_str()}] {command_name}\n")
        log_file.write(f"cmd: {command}\n")
        log_file.write(f"returncode: {result.returncode}\n")
        log_file.write("stdout:\n")
        log_file.write((result.stdout or "") + "\n")
        log_file.write("stderr:\n")
        log_file.write((result.stderr or "") + "\n")
        log_file.write("-" * 80 + "\n")
    return log_path

def _tail_text(text, max_lines=20, max_chars=4000):
    """截取文本尾部若干行，避免把超长输出全部打到主日志。"""
    if not text:
        return ""
    lines = text.splitlines()
    tail = "\n".join(lines[-max_lines:])
    if len(tail) > max_chars:
        tail = tail[-max_chars:]
    return tail

def _run_output_looks_failed(stdout_text, stderr_text):
    """tpu run 可能吞掉异常并以 0 退出，这里基于输出做兜底失败判断。"""
    merged = _strip_ansi(f"{stdout_text or ''}\n{stderr_text or ''}").lower()
    failure_markers = (
        "[fail]",
        "traceback (most recent call last)",
        "exception:",
        "error:",
        "failed to query status",
        "this may indicate that this tpu is deleted",
        "quiting...",
        "quitting...",
    )
    if any(marker in merged for marker in failure_markers):
        return True

    # The xibo backend sometimes prints "TPU ... is reserved by zak" and exits
    # with code 0. Do not match the healthy "is not reserved by others" line.
    return re.search(r"\bis (?:already )?reserved by (?!others\b)[^\n]+", merged) is not None

def _command_result_looks_failed(result):
    """Return True for commands that failed by exit code or by xibo log markers."""
    return result.returncode != 0 or _run_output_looks_failed(result.stdout, result.stderr)

def _command_created_job(stdout_text, stderr_text):
    """Return True once the external tpu command has created a tmux window.

    Some backend commands print later spreadsheet/sheet errors after the job is
    already launched. Those must be treated as success by this monitor, otherwise
    the same queue entry or failed window gets dispatched again next loop.
    """
    merged = _strip_ansi(f"{stdout_text or ''}\n{stderr_text or ''}").lower()
    return (
        "successfully created job in tmux window" in merged
        or re.search(r"\b(?:run|resume|rerun) job .* with new windows id \d+", merged) is not None
    )

def _build_tpu_jax_check_cmd(tpu_name, zone):
    """Build a cheap remote sanity check matching the Python used by staging.sh."""
    return [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        tpu_name,
        f"--zone={zone}",
        f"--project={TPU_PROJECT}",
        "--worker=all",
        "--command",
        TPU_JAX_CHECK_REMOTE_CMD,
    ]

def _run_tpu_jax_check(tpu_name, zone):
    """Verify all TPU workers can import jax before creating a tmux job."""
    cmd = _build_tpu_jax_check_cmd(tpu_name, zone)
    cmd_text = shlex.join(cmd)
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=TPU_ENV_CHECK_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        result = _timeout_result(cmd_text, exc)
    return cmd_text, result

def _build_tmd_inline(tpu_name=None, zone=None):
    """Build a shell `tmd` wrapper with xibo's required --force argument order."""
    del tpu_name, zone  # Kept in the signature so callers log/pass the chosen TPU.
    return (
        r'tmd() { '
        r'local force=0; local args=(); '
        r'for arg in "$@"; do '
        r'if [ "$arg" = "--force" ]; then force=1; else args+=("$arg"); fi; '
        r'done; '
        r'while true; do '
        r'if [ "$force" = "1" ]; then '
        r'output=$(' + f'{_tpu}' + r' mount-disk "${args[@]}" --force 2>&1 | tee /dev/tty); '
        r'else '
        r'output=$(' + f'{_tpu}' + r' mount-disk "${args[@]}" 2>&1 | tee /dev/tty); '
        r'fi; '
        r'if [ "$force" = "0" ] && echo "$output" | grep -q "another process is already mounting"; then '
        r'sleep 10; echo "sleep 10 and retrying..."; '
        r'else break; fi; '
        r'done; '
        r'}'
    )

def _tpu_jax_check_passed(result):
    """Require an explicit JAX_OK marker; gcloud can be noisy even on success."""
    merged = _strip_ansi(f"{result.stdout or ''}\n{result.stderr or ''}").lower()
    return (
        result.returncode == 0
        and "jax_ok" in merged
        and "no module named 'jax'" not in merged
        and "modulenotfounderror" not in merged
    )

def _timeout_result(command, exc):
    """Convert TimeoutExpired into a CompletedProcess-like object for logging."""
    stdout = exc.stdout or getattr(exc, 'output', '') or ''
    stderr = exc.stderr or ''
    if isinstance(stdout, bytes):
        stdout = stdout.decode(errors='replace')
    if isinstance(stderr, bytes):
        stderr = stderr.decode(errors='replace')
    return subprocess.CompletedProcess(command, returncode=124, stdout=stdout, stderr=stderr)

def _tou_cache_age_seconds(result):
    """Parse wrap_master.py cache age from stderr/stdout."""
    merged = _strip_ansi(f"{result.stderr or ''}\n{result.stdout or ''}")
    match = TOU_CACHE_AGE_RE.search(merged)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None

def _restart_yizhitou(reason):
    """Restart the tmux session that continuously refreshes the tou cache."""
    add_MONITOR_log(f'{WARNING} tou cache 不可信：{reason}，重启 yizhitou 并中断本轮\n')
    commands = [
        ['tmux', 'kill-session', '-t', 'yizhitou'],
        ['tmux', 'new-session', '-d', '-s', 'yizhitou'],
        ['sleep', '5'],
        ['tmux', 'send-keys', '-t', 'yizhitou', 'yizhitou', 'C-m'],
    ]
    for cmd in commands:
        if cmd[:2] == ['tmux', 'kill-session']:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

def _run_tou_checked(context):
    """Run `tou` and abort the current loop if its cache is too old.

    `wrap_master.py` prints cache age like "audit was 124s ago". If `yizhitou`
    has been closed, that age keeps growing while `tou` still returns stale
    IDLE/BUSY data. Selection must not proceed in that state.
    """
    result = subprocess.run(
        _tou,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    age = _tou_cache_age_seconds(result)
    if result.returncode != 0 and "no cache" in _strip_ansi(result.stderr or result.stdout).lower():
        _restart_yizhitou(f'{context}: tou 没有可用 cache')
        raise TouCacheStale(context)
    if age is not None and age > TOU_MAX_CACHE_AGE_SECONDS:
        _restart_yizhitou(
            f'{context}: tou delay={age:.0f}s > {TOU_MAX_CACHE_AGE_SECONDS}s'
        )
        raise TouCacheStale(context)
    return result

def show_MONITOR_log(timezone = 'us'):
    """从 data.json 读取历史 MONITOR 日志并按指定时区（'us'=EDT / 'cn'=CHN）打印。"""
    data = data_io.read_data()
    for log in data["MONITOR_logs"]:
        cur_time = log["time"]
        msg = log["msg"]
        show_time = None
        if timezone == 'us':
            show_time = convert_utcstr_to_edtstr(cur_time)
        elif timezone == 'cn':
            show_time = convert_utcstr_to_chnstr(cur_time)
        else:
            show_time = cur_time
        print(f"{LOG} {show_time}: {msg}")
    
def avilable_aliases(tpu_type: str, zone):
    """根据 TPU 类型和 zone 返回该区域可用的临时 alias 列表（如 v6e-64-tmp51 … v6e-64-tmp58）。"""
    if zone.startswith('us-central1'):
        return [tpu_type + '-tmp' + str(i) for i in range(2, 9)]
    elif zone.startswith('us-east5'):
        if tpu_type.startswith('v5p'):
            return [tpu_type + '-tmp' + str(i) for i in range(201, 209)]
        if tpu_type.startswith('v6e'):
            return [tpu_type + '-tmp' + str(i) for i in range(51, 59)]
    elif zone.startswith('asia-northeast1-b'):
        assert tpu_type.startswith('v6e')
        return [tpu_type + '-tmp' + str(i) for i in range(201, 209)]
    else:
        raise ValueError(f"Invalid zone: {zone}")

def check_job_status(job):
    """通过 gcloud 查询 job 对应 TPU 的状态，返回 'preempted' / 'deleted' / None。
    None 表示 TPU 仍然存活（非抢占）。"""
    if job["log_dir"] == '' or job["log_dir"] is None:
        return None
    tpu = job["tpu"]
    if tpu == '':
        print(f"{MADE} check_job_status: tpu is empty")
        return None

    log_dir = job["log_dir"]
    zone = _extract_log_dir_zone(log_dir)
    if zone is None:
        print(f"{MADE} check_job_status: cannot parse zone from log_dir {log_dir}")
        return None

    cmd = f"gcloud compute tpus tpu-vm describe {tpu} --zone={zone} --format='value(state)'"
    try:
        state = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip().lower()
    except subprocess.CalledProcessError:
        return 'deleted'

    if state == 'preempted':
        return 'preempted'

    return None

def _strip_ansi(text):
    """
    去掉颜色代码
    """
    return ANSI_ESCAPE_RE.sub('', text or '')

def _extract_tpu_type(name):
    """
    Extract TPU type from TPU name
    Example:
        kmh-tpuvm-v6e-64-spot-sqa24jgcg -> v6e-64
        kmh-tpuvm-v5p-64-spot-llqnomzp1 -> v5p-64
        kmh-tpuvm-v6e-32-sqa-3210 -> v6e-32
        kmh-tpuvm-v5e-32-spot-gzy24jgcg -> v5e-32
        kmh-tpuvm-v6e-64-spot-sqa24jgcg -> v6e-64
        kmh-tpuvm-v5p-64-spot-llqnomzp1 -> v5p-64
        kmh-tpuvm-v6e-32-sqa-3210 -> v6e-32
    """
    if not name:
        return None
    norm_name = name.strip()
    if norm_name.startswith('kmh-tpuvm-'):
        norm_name = norm_name[len('kmh-tpuvm-'):]
    match = TYPE_RE.match(norm_name)
    if match is None:
        # try special case: kmh-tpuvm-v6e-spot-xtiange-0423-2144-05a8a4-asia-northeast1-b, this is v6e-8
        if norm_name.startswith('v6e-spot-xtiange') and norm_name.endswith('asia-northeast1-b'):
            return 'v6e-8'
        else:
            return None
    return match.group(1)

def _extract_zone(tpu_name):
    """
    Extract zone from text
    Example:
        us-central1-a -> us-central1-a
        us-central1-b -> us-central1-b
        us-central2-b -> us-central2-b
        us-east1-d -> us-east1-d
        us-east5-b -> us-east5-b
        asia-northeast1-b -> asia-northeast1-b
    """
    # we need to look into data.json, to find the zone of the tpu
    data = data_io.read_data()
    for zone, tpu_list in data['all_tpus'].items():
        if tpu_name in tpu_list:
            return zone
    return None

def _extract_log_dir_zone(log_dir):
    """Extract the real TPU zone from a log_dir path.

    TPU names can contain strings such as `us-central1-spot-zongyili-...`,
    which match the loose zone regex but are not zones. Prefer the last
    allowed zone in the log path, usually the `_<zone>__...` suffix.
    """
    zones = [m.group(0) for m in ZONE_RE.finditer(log_dir or "")]
    if not zones:
        return None
    allowed_zones = [z for z in zones if is_resume_queue_allowed_zone(z)]
    if allowed_zones:
        return allowed_zones[-1]
    return zones[-1]

def _get_job_type_zone(job):
    """从 job dict 中提取 (tpu_type, zone)，类型来自 TPU 名称，zone 来自 log_dir 路径。"""
    old_tpu = job["tpu"]
    target_type = _extract_tpu_type(old_tpu)
    target_zone = _extract_log_dir_zone(job.get("log_dir", ""))
    return target_type, target_zone

def _zone_region(zone):
    """
    Convert zone to region-like key.
    Example:
      us-central1-a / us-central1-b -> us-central1
      us-east5-a / us-east5-b       -> us-east5
    """
    if not zone:
        return None
    parts = zone.split('-')
    if len(parts) < 3:
        return zone
    return '-'.join(parts[:-1])

def _parse_idle_tpus_from_tou(stdout):
    """解析 tou 输出，返回所有 [IDLE] TPU 的 (tpu_name, zone) 列表。"""
    idle_tpus = []
    clean_stdout = _strip_ansi(stdout)
    for line in clean_stdout.splitlines():
        match = re.search(r"\[IDLE\]\s+([^\s]+)\s+\(([^)]+)\)", line)
        if not match:
            continue
        tpu_name, zone = match.group(1), match.group(2)
        idle_tpus.append((tpu_name.strip(), zone.strip()))
    return idle_tpus

def _parse_all_tpu_names_from_tou(stdout):
    """解析 tou 输出中出现的所有 TPU 名称（含 IDLE/BUSY）。"""
    names = set()
    clean_stdout = _strip_ansi(stdout)
    for line in clean_stdout.splitlines():
        match = re.search(r"\[(?:IDLE|BUSY)\]\s+([^\s]+)\s+\(([^)]+)\)", line)
        if not match:
            continue
        names.add(match.group(1).strip())
    return names

def _get_tpu_usage_from_tou(stdout, tpu_name):
    """在 tou 输出中查找指定 TPU 的占用状态。
    返回 ('idle', []) / ('busy', [user, ...]) / ('unknown', [])。"""
    clean_stdout = _strip_ansi(stdout)
    for line in clean_stdout.splitlines():
        idle_match = re.search(r"\[IDLE\]\s+([^\s]+)\s+\(([^)]+)\)", line)
        if idle_match and idle_match.group(1).strip() == tpu_name:
            return 'idle', []

        busy_match = re.search(r"\[BUSY\]\s+([^\s]+)\s+\(([^)]+)\)\s+users=(\[[^\]]*\])", line)
        if busy_match and busy_match.group(1).strip() == tpu_name:
            users_raw = busy_match.group(3)
            try:
                users = ast.literal_eval(users_raw)
                if not isinstance(users, list):
                    users = [str(users)]
            except Exception:
                users = []
            users = [str(u).strip() for u in users if str(u).strip()]
            return 'busy', users

    return 'unknown', []

def _has_retrying_ssh_error_in_recent_logs(job, recent_lines=40):
    """检查 output.log 末尾是否包含明确需要 resume/rerun 的错误 marker。"""
    log_dir = job.get('log_dir')
    if not log_dir:
        return False

    log_path = os.path.join(log_dir, 'output.log')
    if not os.path.isfile(log_path):
        return False

    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            tail_lines = list(deque(f, maxlen=recent_lines))
    except Exception:
        return False

    tail_text = ''.join(tail_lines)
    tail_lower = tail_text.lower()
    markers = [
        "retrying: ssh command error",
        "unable to initialize backend 'tpu': aborted",
        "fatal python error: aborted",
        "jax distributed service detected fatal errors",
        "coordinationservice/heartbeat",
        "deadline_exceeded",
        "python: can't open file '/home/sqa/main.py'",
        "cp: cannot stat",
        "no module named 'jax'",
        "modulenotfounderror",
    ]
    return any(marker in tail_lower for marker in markers)

def _tmux_windows_for_session(session_name):
    """Return existing tmux window ids for a session.

    Missing sessions return an empty set. We use this as a guard against stale
    `data.json` rows: a status=running job only blocks a TPU if its window still
    exists locally.
    """
    if not session_name:
        return set()
    try:
        result = subprocess.run(
            ['tmux', 'list-windows', '-t', str(session_name), '-F', '#{window_index}'],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except Exception:
        return set()
    if result.returncode != 0:
        return set()

    windows = set()
    for line in result.stdout.splitlines():
        try:
            windows.add(int(line.strip()))
        except (TypeError, ValueError):
            continue
    return windows

def _iter_user_jobs(data):
    """Yield (user, user_data, job) for every job in the external xibo data."""
    for user, user_data in (data.get('users') or {}).items():
        for job in user_data.get('job_data', []) or []:
            yield user, user_data, job

def _job_window_id(job):
    try:
        return int(job.get('windows_id'))
    except (TypeError, ValueError):
        return None

def _job_looks_active_for_tpu_selection(job):
    """Return True when a job should make its TPU unavailable for new work.

    `tou` can briefly report a just-launched TPU as IDLE before worker-side
    processes are visible. The durable signal during that gap is data.json plus
    an existing tmux window. Error jobs without recent SSH-retry lines are the
    same "误报 error" cases that mainloop already treats as still running.
    """
    status = str(job.get('status') or '').lower()
    if status == 'running':
        return True
    if status == 'error':
        return not _has_retrying_ssh_error_in_recent_logs(job, recent_lines=40)
    return False

def _active_tpu_occupants_by_tpu(data=None):
    """Map TPU name -> jobs whose existing windows make that TPU unavailable."""
    data = data or data_io.read_data()
    tmux_windows_by_user = {
        user: _tmux_windows_for_session(user_data.get('tmux_name'))
        for user, user_data in (data.get('users') or {}).items()
    }

    occupants = {}
    for user, _user_data, job in _iter_user_jobs(data):
        tpu = str(job.get('tpu') or '').strip()
        if not tpu:
            continue
        window_id = _job_window_id(job)
        if window_id is None or window_id not in tmux_windows_by_user.get(user, set()):
            continue
        if not _job_looks_active_for_tpu_selection(job):
            continue
        occupants.setdefault(tpu, []).append({
            'user': user,
            'window': window_id,
            'status': str(job.get('status') or ''),
        })
    return occupants

def _format_tpu_occupants(occupants, ignore_windows=None):
    """Compact log string for active TPU occupants."""
    ignore_windows = {int(w) for w in (ignore_windows or set())}
    parts = []
    for item in occupants or []:
        if item['window'] in ignore_windows:
            continue
        parts.append(f"{item['user']}:{item['window']}({item['status']})")
    return ", ".join(parts)

# Regions that are allowed to cross-resume with each other.
# "asia-northeast1-b" is the only zone in that region, so we use the region prefix.
_CROSS_REGION_ALLOWED = {'us-central1', 'us-east5', 'asia-northeast1'}

_TPU_GEN_SIZE_RE = re.compile(r'^(v\d+[a-z]*)-(\d+)$')

def _parse_tpu_gen_size(tpu_type):
    """Parse 'v6e-64' -> ('v6e', 64), 'v5p-128' -> ('v5p', 128), etc."""
    if not tpu_type:
        return None, None
    m = _TPU_GEN_SIZE_RE.match(tpu_type)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))

def _tpu_generation_rank(gen):
    """Return a numeric generation rank: v6e > v5p for same chip count."""
    if not gen:
        return -1
    match = re.match(r"v(\d+)", str(gen))
    if not match:
        return -1
    return int(match.group(1))

def _tpu_size_sort_key(tpu_name, zone, gpt_b_only=False):
    """Sort TPU candidates by effective capacity.

    Non-Asia TPUs are preferred first to preserve asia-northeast1-b capacity.
    Inside that group, size is the primary signal and generation is the tie
    breaker. GPT-B/DeiT jobs invert the size direction.
    """
    gen, size = _parse_tpu_gen_size(_extract_tpu_type(tpu_name))
    gen_rank = _tpu_generation_rank(gen)
    size = size if size is not None else -1
    zone_rank = _zone_priority(zone)
    if gpt_b_only:
        return (zone_rank, size, gen_rank, tpu_name)
    return (zone_rank, -size, -gen_rank, tpu_name)

def _is_v5_generation(gen):
    """Return True for TPU generations that are operationally treated as v5."""
    return str(gen or '').startswith('v5')

def _is_type_allowed(candidate_type, low_cost, gpt_b_only=False, v5_only=False):
    """Return True if candidate_type is allowed for a from-scratch launch.

    Rules:
      - high-cost (low_cost=False): exactly v6e-64 or v5/v5p-128.
      - low-cost  (low_cost=True):  exactly v6e-32 or v5/v5p-64.
      - GPT-B notes (gpt_b_only=True): v6e <= 16, v5p <= 32.
      - v5_only=True: additionally reject every non-v5 generation.

    Checkpoint resumes do not use this cost-class choice. They require the
    original TPU type exactly, handled by _same_resume_type_allowed().
    """
    gen, size = _parse_tpu_gen_size(candidate_type)
    if gen is None:
        return False
    is_v5 = _is_v5_generation(gen)
    if v5_only and not is_v5:
        return False
    if gpt_b_only:
        if gen == 'v6e' and size <= 16:
            return True
        if is_v5 and size <= 32:
            return True
        return False
    if low_cost:
        if gen == 'v6e' and size == 32:
            return True
        if is_v5 and size == 64:
            return True
        return False
    if gen == 'v6e' and size == 64:
        return True
    if is_v5 and size == 128:
        return True
    return False

def _same_resume_type_allowed(candidate_type, target_type, v5_only=False):
    """Hard constraints for checkpoint resumes, which otherwise require exact type."""
    if candidate_type != target_type:
        return False
    gen, _ = _parse_tpu_gen_size(candidate_type)
    if gen is None:
        return False
    if v5_only and not _is_v5_generation(gen):
        return False
    return True

def _is_low_cost_job(job):
    """Return True if the job qualifies for low-cost (smaller) TPUs.
    Triggered by tokens in spreadsheet notes."""
    notes = _get_job_notes(job)
    notes_l = str(notes).lower()
    allowed_tokens = ['jit', 'mae-b', 'unify-base', 'llava-1.5 reproduction']
    return any(token in notes_l for token in allowed_tokens)

def _is_llava15_reproduction_job(job):
    """LLaVA-1.5 reproduction jobs OOM on v6, so they must run only on v5."""
    return 'llava-1.5 reproduction' in str(_get_job_notes(job)).lower()

def _is_gpt_b_job(job):
    """Return True if spreadsheet/wandb notes indicate GPT-B workload."""
    notes = _get_job_notes(job)
    return ('gpt-b' in str(notes).lower() or 'deit' in str(notes).lower())

def _get_job_notes(job):
    """Return all note-like text available on a job dict."""
    extra = job.get('extra_msgs') or {}
    chunks = [
        job.get('wandb_notes'),
        job.get('notes'),
        job.get('user_note'),
        job.get('job_tags'),
        job.get('extra_configs'),
        extra.get('wandb_notes'),
        extra.get('spreadsheet_notes'),
        extra.get('notes'),
        extra.get('user_note'),
        extra.get('job_tags'),
        extra.get('extra_configs'),
    ]
    return "\n".join(str(x) for x in chunks if x is not None)

def _jobs_by_window(data):
    return {
        int(job['windows_id']): job
        for job in data.get('users', {}).get(USER, {}).get('job_data', [])
        if job.get('windows_id') is not None
    }

def _job_child_window(job):
    child = (job.get('extra_msgs') or {}).get('child')
    if child is None or child == '':
        return None
    try:
        return int(child)
    except (TypeError, ValueError):
        return None

def _child_window_for(window_id, jobs_by_window):
    job = jobs_by_window.get(int(window_id))
    if job is None:
        return None
    return _job_child_window(job)

def _father_window_for(window_id, jobs_by_window):
    job = jobs_by_window.get(int(window_id))
    if job is None:
        return None
    father = (job.get('extra_msgs') or {}).get('father')
    if father is None or father == '':
        return None
    try:
        return int(father)
    except (TypeError, ValueError):
        return None

def _is_ancestor_window(ancestor_window, window_id, jobs_by_window):
    """Return True if ancestor_window is on window_id's father chain."""
    try:
        ancestor_window = int(ancestor_window)
        current = int(window_id)
    except (TypeError, ValueError):
        return False

    seen = set()
    while current not in seen:
        seen.add(current)
        father = _father_window_for(current, jobs_by_window)
        if father is None:
            return False
        if father == ancestor_window:
            return True
        current = father
    return False

def _resume_superseded_by(window_id, actual_window, jobs_by_window):
    """Return newer child id if this failed window is no longer the chain leaf."""
    own_child = _child_window_for(window_id, jobs_by_window)
    if own_child is not None:
        return own_child

    if actual_window is None or int(actual_window) == int(window_id):
        return None

    actual_child = _child_window_for(actual_window, jobs_by_window)
    if actual_child is not None and int(actual_child) != int(window_id):
        # If actual_child is on window_id's own ancestry, it is the path from
        # the saving ancestor to this failed leaf, not a newer sibling branch.
        if _is_ancestor_window(actual_child, window_id, jobs_by_window):
            return None
        return actual_child
    return None

def _get_queue_job_wandb_notes(dir_no):
    """从 USER 的 working_dir[dir_no] 读取 wandb_notes。

    默认读取 configs/remote_run_config.yml；若其中 finetune=True，改读
    configs/finetune_config.yml。失败时返回空字符串。
    """
    try:
        data = data_io.read_data()
        user_data = data.get('users', {}).get(USER, {})
        working_dir = (user_data.get('working_dir') or {}).get(str(dir_no))
        if not working_dir:
            return ''

        remote_cfg = os.path.join(working_dir, 'configs', 'remote_run_config.yml')
        if not os.path.exists(remote_cfg):
            return ''

        with open(remote_cfg, 'r', encoding='utf-8', errors='ignore') as f:
            cfg = yaml.safe_load(f) or {}

        target_cfg = cfg
        if bool(cfg.get('finetune', False)):
            finetune_cfg = os.path.join(working_dir, 'configs', 'finetune_config.yml')
            if os.path.exists(finetune_cfg):
                with open(finetune_cfg, 'r', encoding='utf-8', errors='ignore') as f:
                    target_cfg = yaml.safe_load(f) or {}

        notes = target_cfg.get('wandb_notes')
        if notes is None and isinstance(target_cfg.get('logging'), dict):
            notes = target_cfg.get('logging', {}).get('wandb_notes')
        return str(notes or '')
    except Exception:
        return ''

def _zone_priority(zone):
    """Return sort key for zone priority during TPU selection (lower = higher priority).

    Priority order:
      0 – us-east5-* / us-central1-* (prefer first)
      1 – other allowed non-Asia zones
      2 – asia-northeast1-b          (preserve for LAION unless needed)
    """
    if zone == _ASIA_RESERVED_ZONE:
        return 2
    if zone and (zone.startswith('us-east5') or zone.startswith('us-central1')):
        return 0
    return 1

_FS_SCRIPT = "/kmh-nfs-ssd-us-mount/code/qiao/work/tpu_manager/find_saving_window.py"
_NO_SAVING_ROOT_RE = re.compile(r"no 'saving' found anywhere in chain; reached root window (\d+)")

def _get_saving_window(window_id):
    """
    Run find_saving_window.py to walk up the father chain and find the most
    recent ancestor whose output.log contains a "saving" line.

    Returns (actual_window_id, action) where action is 'resume' or 'rerun':
      - 'resume': actual_window_id has saving; resume it normally (load_ckpt=True).
      - 'rerun':  no saving anywhere in the chain; actual_window_id is the root
                  (earliest) window — rerun it from scratch (load_ckpt=False).
    Falls back to (window_id, 'resume') on unexpected errors.
    """
    try:
        result = subprocess.run(
            ["python3", _FS_SCRIPT, str(window_id)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            actual = int(result.stdout.strip())
            if actual != window_id:
                add_MONITOR_log(
                    f"{INFO} _get_saving_window: 最近的 window {window_id} 没有saving，"
                    f"找到有saving的祖先 window {actual}，改为resume它\n"
                )
            return actual, 'resume'
        else:
            m = _NO_SAVING_ROOT_RE.search(result.stderr)
            if m:
                root_id = int(m.group(1))
                add_MONITOR_log(
                    f"{WARNING} _get_saving_window: 整条链上都没有saving，"
                    f"根窗口是 {root_id}，改为rerun它\n"
                )
                return root_id, 'rerun'
            add_MONITOR_log(
                f"{WARNING} _get_saving_window: fs {window_id} 失败: "
                f"{result.stderr.strip()}, 保持原窗口 {window_id}\n"
            )
            return window_id, 'resume'
    except Exception as e:
        add_MONITOR_log(
            f"{WARNING} _get_saving_window: fs {window_id} 异常: {e}, 保持原窗口 {window_id}\n"
        )
        return window_id, 'resume'


def _get_tpu_existing_alias(tpu_name):
    """已注册 TPU 的反查：在 tpu_aliases 中找 value == tpu_name 的 key。
    找不到则返回 None。"""
    data = data_io.read_data()
    for alias, full_name in data.get('tpu_aliases', {}).items():
        if full_name == tpu_name:
            return alias
    return None

def _pick_idle_tpu(
    idle_tpus,
    target_type,
    target_zone,
    low_cost=False,
    same_type_only=False,
    excluded_tpus=None,
    gpt_b_only=False,
    v5_only=False,
):
    """
    Pick an idle TPU that is allowed to resume the job.

    Type rules:
      - same_type_only=True: exact target_type, across allowed zones if needed.
      - otherwise see _is_type_allowed for checkpoint-free reruns.
      - LLaVA-1.5 reproduction keeps its v5-only hard constraint.

    Region rules:
      - Pass 1: same type + same zone.
      - Pass 2: same region; exact type when same_type_only, otherwise any allowed type.
      - Pass 3: cross-region fallback, but only when both the job's region AND the
                candidate's region are in _CROSS_REGION_ALLOWED.

    Within each pass, unowned TPUs are tried before owned ones. Inside each
    ownership group, normal jobs try larger cards first; GPT-B/DeiT jobs try
    smaller cards first. For equal chip count, newer generation wins.

    Return:
      (new_tpu_name, new_zone, pick_mode, already_owned)
      pick_mode in {
          "exact",
          "same_type_same_region",
          "same_type_cross_region",
          "cross_type_same_region",
          "cross_region",
      }
      or (None, None, None, None) if not found.
    """
    data = data_io.read_data()
    all_tpus = data.get('all_tpus', {})
    blocked_tpus = set(_active_tpu_occupants_by_tpu(data))
    target_region = _zone_region(target_zone)

    def _is_owned(tpu_name, zone):
        return tpu_name in all_tpus.get(zone, [])

    excluded_tpus = excluded_tpus or set()

    # Sort all candidates by capacity; then partition into unowned/owned.
    sorted_idle = sorted(
        [
            (n, z)
            for n, z in idle_tpus
            if n not in excluded_tpus
            and n not in blocked_tpus
            and is_resume_queue_allowed_zone(z)
        ],
        key=lambda t: _tpu_size_sort_key(t[0], t[1], gpt_b_only=gpt_b_only)
    )
    unowned = [(n, z) for n, z in sorted_idle if not _is_owned(n, z)]
    owned   = [(n, z) for n, z in sorted_idle if _is_owned(n, z)]

    # Pass 1: same type + same zone. Checkpoint resume intentionally does not
    # apply low/high-cost class filters here; the old checkpoint fixes topology.
    for candidates, is_already_owned in [(unowned, False), (owned, True)]:
        for tpu_name, zone in candidates:
            tpu_type = _extract_tpu_type(tpu_name)
            if (
                zone == target_zone
                and _same_resume_type_allowed(tpu_type, target_type, v5_only=v5_only)
            ):
                return tpu_name, zone, "exact", is_already_owned

    if same_type_only:
        # Pass 2: same type + same region.
        for candidates, is_already_owned in [(unowned, False), (owned, True)]:
            for tpu_name, zone in candidates:
                if _zone_region(zone) != target_region:
                    continue
                tpu_type = _extract_tpu_type(tpu_name)
                if _same_resume_type_allowed(tpu_type, target_type, v5_only=v5_only):
                    return tpu_name, zone, "same_type_same_region", is_already_owned

        # Pass 3: same type + cross-region.
        if target_region in _CROSS_REGION_ALLOWED:
            for candidates, is_already_owned in [(unowned, False), (owned, True)]:
                for tpu_name, zone in candidates:
                    candidate_region = _zone_region(zone)
                    if candidate_region == target_region:
                        continue  # already covered in Pass 2
                    if candidate_region not in _CROSS_REGION_ALLOWED:
                        continue
                    tpu_type = _extract_tpu_type(tpu_name)
                    if _same_resume_type_allowed(tpu_type, target_type, v5_only=v5_only):
                        return tpu_name, zone, "same_type_cross_region", is_already_owned
        return None, None, None, None

    # Pass 2: any allowed type, same region — unowned first, owned as fallback.
    for tpu_name, zone in unowned:
        if _zone_region(zone) != target_region:
            continue
        if _is_type_allowed(_extract_tpu_type(tpu_name), low_cost, gpt_b_only=gpt_b_only, v5_only=v5_only):
            return tpu_name, zone, "cross_type_same_region", False
    for tpu_name, zone in owned:
        if _zone_region(zone) != target_region:
            continue
        if _is_type_allowed(_extract_tpu_type(tpu_name), low_cost, gpt_b_only=gpt_b_only, v5_only=v5_only):
            return tpu_name, zone, "cross_type_same_region", True

    # Pass 3: any allowed type, cross-region — unowned first, owned as fallback.
    if target_region in _CROSS_REGION_ALLOWED:
        for candidates, is_already_owned in [(unowned, False), (owned, True)]:
            for tpu_name, zone in candidates:
                tpu_type = _extract_tpu_type(tpu_name)
                candidate_region = _zone_region(zone)
                if candidate_region == target_region:
                    continue  # already covered in Pass 2
                if candidate_region not in _CROSS_REGION_ALLOWED:
                    continue
                if _is_type_allowed(tpu_type, low_cost, gpt_b_only=gpt_b_only, v5_only=v5_only):
                    return tpu_name, zone, "cross_region", is_already_owned

    return None, None, None, None

def _pick_any_idle_tpu(
    idle_tpus,
    preferred_type=None,
    preferred_zone=None,
    excluded_tpus=None,
    low_cost=False,
    gpt_b_only=False,
    v5_only=False,
):
    """为新 job 选一张空闲 TPU（不要求 resume 同型号）。

    优先级（每级内 unowned 先于 owned；普通 job 大卡优先，GPT-B/DeiT 小卡优先）：
      Pass 1: preferred_type + preferred_zone（若均指定）。
      Pass 2: preferred_type 任意 zone（若只指定了 type）。
      Pass 3: 任意通过 _is_type_allowed 的卡（high exact v6e-64/v5-128,
              low exact v6e-32/v5-64）。

    返回 (tpu_name, zone, already_owned) 或 (None, None, False)。
    """
    data = data_io.read_data()
    all_tpus = data.get('all_tpus', {})
    blocked_tpus = set(_active_tpu_occupants_by_tpu(data))

    def _is_owned(tpu_name, zone):
        return tpu_name in all_tpus.get(zone, [])

    excluded_tpus = excluded_tpus or set()
    filtered_idle = sorted(
        [
            (n, z)
            for n, z in idle_tpus
            if n not in excluded_tpus
            and n not in blocked_tpus
            and is_resume_queue_allowed_zone(z)
        ],
        key=lambda t: _tpu_size_sort_key(t[0], t[1], gpt_b_only=gpt_b_only)
    )

    unowned = [(n, z) for n, z in filtered_idle if not _is_owned(n, z)]
    owned   = [(n, z) for n, z in filtered_idle if _is_owned(n, z)]

    if preferred_type and preferred_zone:
        for candidates, flag in [(unowned, False), (owned, True)]:
            for tpu_name, zone in candidates:
                tpu_type = _extract_tpu_type(tpu_name)
                if (
                    tpu_type == preferred_type
                    and zone == preferred_zone
                    and _is_type_allowed(tpu_type, low_cost=low_cost, gpt_b_only=gpt_b_only, v5_only=v5_only)
                ):
                    return tpu_name, zone, flag

    if preferred_type:
        for candidates, flag in [(unowned, False), (owned, True)]:
            for tpu_name, zone in candidates:
                tpu_type = _extract_tpu_type(tpu_name)
                if (
                    tpu_type == preferred_type
                    and _is_type_allowed(tpu_type, low_cost=low_cost, gpt_b_only=gpt_b_only, v5_only=v5_only)
                ):
                    return tpu_name, zone, flag

    for candidates, flag in [(unowned, False), (owned, True)]:
        for tpu_name, zone in candidates:
            if _is_type_allowed(_extract_tpu_type(tpu_name), low_cost=low_cost, gpt_b_only=gpt_b_only, v5_only=v5_only):
                return tpu_name, zone, flag

    return None, None, False

def _pick_new_alias(target_type, zone, tou_stdout=None, reserved_aliases=None, reserved_tpus=None):
    """从该 zone 的候选 alias 列表中，找出全名不出现在 tou 输出里的第一个 alias
    （即当前没有被任何人用到的 alias），用于 fmd 放卡。返回 alias 字符串或 None。"""
    available_aliases = avilable_aliases(target_type, zone)
    if tou_stdout is None:
        tou_result = _run_tou_checked(f'pick alias {target_type}/{zone}')
        if tou_result.returncode != 0:
            return None
        tou_stdout = tou_result.stdout

    reserved_aliases = reserved_aliases or set()
    data = data_io.read_data()
    active_tpus = set(_active_tpu_occupants_by_tpu(data))
    reserved_tpus = set(reserved_tpus or set()) | active_tpus
    tpus_in_tou = _parse_all_tpu_names_from_tou(tou_stdout)
    alias_map = data.get('tpu_aliases', {})

    for a in available_aliases:
        if a in reserved_aliases:
            continue
        full_name = alias_map.get(a)
        if not full_name:
            continue
        if full_name in reserved_tpus:
            continue
        if full_name not in tpus_in_tou:
            return a
    return None

def _queue_reserved_aliases_tpus():
    """返回 queue running 中已占用的 alias/tpu 集合。"""
    q = _read_queue()
    running = q.get('running', [])
    aliases = {str(e.get('alias')).strip() for e in running if e.get('alias')}
    tpus = {str(e.get('tpu')).strip() for e in running if e.get('tpu')}
    return aliases, tpus

def _run_queued_jobs(excluded_tpus=None):
    """处理 queue.json 中的 pending 任务：找空卡 -> ftmd -> tpu run。
    每轮 mainloop 结束后调用一次。"""
    # Queue 里的 dir_no 是 sqa 的 working_dir 编号；只有 sqa 能正确消费。
    # 其他 user（如 bird）跳过 queue，避免 hijack sqa 的 job 还跑失败。
    if USER != "sqa":
        q = _read_queue()
        n_pending = len(q.get("pending", []))
        if n_pending > 0:
            add_MONITOR_log(
                f"{INFO} queue: 跳过 ({n_pending} pending) because USER={USER}, only sqa consumes queue"
            )
        return
    q = _read_queue()
    if not q['pending']:
        return
    if _ensure_queue_metadata(q):
        _write_queue(q)

    excluded_tpus = set(excluded_tpus or set())

    add_MONITOR_log(f"{INFO} queue: 发现 {len(q['pending'])} 个待跑任务，开始处理")

    loop_reserved_aliases = set()
    loop_reserved_tpus = set(excluded_tpus)

    for entry in sorted(list(q['pending']), key=_queue_sort_key):
        job_id = entry['id']
        dir_no = entry['dir']
        priority = _queue_entry_priority(entry)
        preferred_type = entry.get('preferred_type')
        preferred_zone = entry.get('preferred_zone')
        notes = entry.get('wandb_notes') or _get_queue_job_wandb_notes(dir_no)
        queue_job = {'extra_msgs': {'spreadsheet_notes': notes}}
        low_cost = _is_low_cost_job(queue_job)
        gpt_b_only = _is_gpt_b_job(queue_job)
        v5_only = _is_llava15_reproduction_job(queue_job)
        add_MONITOR_log(f"{INFO} queue: 尝试 dir={dir_no} priority={priority}")
        if preferred_type or preferred_zone:
            add_MONITOR_log(
                f"{INFO} queue: dir={dir_no} 使用队列偏好"
                + (f" type={preferred_type}" if preferred_type else "")
                + (f" zone={preferred_zone}" if preferred_zone else "")
            )
        if notes:
            add_MONITOR_log(f"{INFO} queue: dir={dir_no} wandb_notes={notes}")
        if gpt_b_only:
            add_MONITOR_log(f'{INFO} queue: dir={dir_no} 命中 GPT-B notes，选卡仅允许 v5p<=32 / v6e<=16')
        if v5_only:
            add_MONITOR_log(f'{INFO} queue: dir={dir_no} 命中 llava-1.5 reproduction，v6 会 OOM，选卡仅允许 v5/v5p-64\n')
        elif low_cost and not gpt_b_only:
            add_MONITOR_log(f'{INFO} queue: dir={dir_no} 命中 low-cost notes，选卡仅允许 v6e-32 / v5/v5p-64')

        tou_result = _run_tou_checked(f'queue dir={dir_no}')
        if tou_result.returncode != 0:
            add_MONITOR_log(f'{MADE} queue: _tou 失败，跳过本轮 queue 处理\n')
            break

        idle_tpus = _parse_idle_tpus_from_tou(tou_result.stdout)
        queue_reserved_aliases, queue_reserved_tpus = _queue_reserved_aliases_tpus()
        reserved_aliases = queue_reserved_aliases | loop_reserved_aliases
        reserved_tpus = queue_reserved_tpus | loop_reserved_tpus

        excluded_tpus_for_pick = set(reserved_tpus)
        new_tpu_name, new_tpu_zone, tpu_already_owned, new_alias, fmd_cmd = None, None, None, None, None

        while True:
            picked_tpu, picked_zone, picked_owned = _pick_any_idle_tpu(
                idle_tpus,
                preferred_type,
                preferred_zone,
                excluded_tpus=excluded_tpus_for_pick,
                low_cost=low_cost,
                gpt_b_only=gpt_b_only,
                v5_only=v5_only,
            )
            if not picked_tpu:
                break

            picked_type = _extract_tpu_type(picked_tpu)
            if not picked_type or not picked_zone:
                add_MONITOR_log(f'{MADE} queue: 卡 {picked_tpu} 无法解析类型/zone，跳过 dir={dir_no}\n')
                excluded_tpus_for_pick.add(picked_tpu)
                continue

            _tmd_inline = _build_tmd_inline(picked_tpu, picked_zone)
            if picked_owned:
                picked_alias = _get_tpu_existing_alias(picked_tpu)
                if not picked_alias:
                    add_MONITOR_log(f'{MADE} queue: 卡 {picked_tpu} 已注册但找不到 alias，跳过 dir={dir_no}\n')
                    excluded_tpus_for_pick.add(picked_tpu)
                    continue
                if picked_alias in reserved_aliases:
                    add_MONITOR_log(f'{WARNING} queue: alias={picked_alias} 已被 queue 占用，跳过卡 {picked_tpu} for dir={dir_no}\n')
                    excluded_tpus_for_pick.add(picked_tpu)
                    continue
                new_tpu_name, new_tpu_zone, tpu_already_owned, new_alias = picked_tpu, picked_zone, True, picked_alias
                add_MONITOR_log(f'{INFO} queue: 卡 {new_tpu_name} 已注册，alias={new_alias}，跳过 fang 直接 zhan+tmd，准备跑 dir={dir_no}\n')
                fmd_cmd = f'{_tmd_inline} && {_tpu} zhan {new_tpu_name} {USER} && tmd {new_alias}'
                add_MONITOR_log(f'{INFO} queue: 运行 zhan+mount-disk (已注册卡): tpu zhan {new_tpu_name} {USER} && tmd {new_alias}\n')
                break

            picked_alias = _pick_new_alias(
                picked_type,
                picked_zone,
                tou_stdout=tou_result.stdout,
                reserved_aliases=reserved_aliases,
                reserved_tpus=reserved_tpus | {picked_tpu},
            )
            if not picked_alias:
                add_MONITOR_log(f'{WARNING} queue: 卡 {picked_tpu} 未找到可用 alias，尝试下一张卡 for dir={dir_no}\n')
                excluded_tpus_for_pick.add(picked_tpu)
                continue

            new_tpu_name, new_tpu_zone, tpu_already_owned, new_alias = picked_tpu, picked_zone, False, picked_alias
            add_MONITOR_log(f'{INFO} queue: 准备跑 dir={dir_no}，卡={new_tpu_name} alias={new_alias}\n')
            fmd_cmd = f'{_tmd_inline} && {_tpu} zhan {new_tpu_name} {USER} && {_tpu} fang {new_tpu_name} {new_alias} && tmd {new_alias}'
            add_MONITOR_log(f'{INFO} queue: 运行 ftmd: tpu zhan {new_tpu_name} {USER} && tpu fang {new_tpu_name} {new_alias} && tmd {new_alias}\n')
            break

        if not new_tpu_name:
            add_MONITOR_log(f'{MADE} queue: 找不到可用的 IDLE 卡 for dir={dir_no}，允许区域={RESUME_QUEUE_ALLOWED_ZONE_LABEL}，看下一个\n')
            continue

        # mark as running before executing to prevent double-dispatch
        _queue_start_job(job_id, new_tpu_name, new_alias)
        loop_reserved_aliases.add(new_alias)
        loop_reserved_tpus.add(new_tpu_name)

        try:
            fmd_result = subprocess.run(
                fmd_cmd,
                shell=True,
                executable='/bin/bash',
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=900
            )
        except subprocess.TimeoutExpired:
            add_MONITOR_log(f'{MADE} queue: ftmd 超时 for dir={dir_no}，移回 pending\n')
            _queue_fail_job(job_id)
            continue

        if fmd_result.returncode == 124:
            add_MONITOR_log(f'{MADE} queue: ftmd 超时(124) for dir={dir_no}，移回 pending\n')
            _queue_fail_job(job_id)
            continue

        fmd_log_path = _append_queue_file_log(job_id, dir_no, "ftmd", fmd_cmd, fmd_result)
        if _command_result_looks_failed(fmd_result):
            add_MONITOR_log(f'{MADE} queue: ftmd 失败 for dir={dir_no}，移回 pending。完整输出见 {fmd_log_path}\n')
            _queue_fail_job(job_id)
            continue

        env_cmd, env_result = _run_tpu_jax_check(new_tpu_name, new_tpu_zone)
        env_log_path = _append_queue_file_log(job_id, dir_no, "jax_check", env_cmd, env_result)
        if not _tpu_jax_check_passed(env_result):
            clean_stdout = _strip_ansi(env_result.stdout or "").strip()
            clean_stderr = _strip_ansi(env_result.stderr or "").strip()
            add_MONITOR_log(
                f'{MADE} queue: 卡 {new_tpu_name} JAX 环境检查失败，不创建 dir={dir_no} 的 tmux job，'
                f'移回 pending。完整输出见 {env_log_path}\n'
            )
            if clean_stdout:
                add_MONITOR_log(f'{INFO} queue: jax_check stdout尾部(dir={dir_no}):\n{_tail_text(clean_stdout)}\n')
            if clean_stderr:
                add_MONITOR_log(f'{INFO} queue: jax_check stderr尾部(dir={dir_no}):\n{_tail_text(clean_stderr)}\n')
            _queue_fail_job(job_id)
            continue

        add_MONITOR_log(f'{INFO} queue: 卡 {new_tpu_name} JAX 环境检查通过\n')
        add_MONITOR_log(f'{GAOCHAO} queue: ftmd 完成，准备 tpu run {new_alias} {USER} dir={dir_no}\n')

        run_cmd = f'yes | {_tpu} run {new_alias} {USER} dir={dir_no} --monitor=False'
        add_MONITOR_log(f'{INFO} queue: 运行 {run_cmd}\n')
        try:
            run_result = subprocess.run(
                run_cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=900
            )
        except subprocess.TimeoutExpired as e:
            run_result = _timeout_result(run_cmd, e)

        run_log_path = _append_queue_file_log(job_id, dir_no, "tpu_run", run_cmd, run_result)
        clean_stdout = _strip_ansi(run_result.stdout or "").strip()
        clean_stderr = _strip_ansi(run_result.stderr or "").strip()
        if clean_stdout:
            add_MONITOR_log(f'{INFO} queue: tpu run stdout尾部(dir={dir_no}):\n{_tail_text(clean_stdout)}\n')
        if clean_stderr:
            add_MONITOR_log(f'{INFO} queue: tpu run stderr尾部(dir={dir_no}):\n{_tail_text(clean_stderr)}\n')

        if _command_created_job(clean_stdout, clean_stderr):
            add_MONITOR_log(
                f'{GAOCHAO} queue: dir={dir_no} 已创建 tmux job，'
                f'即使后续输出有sheet/alias错误也视为已启动。完整输出见 {run_log_path}'
            )
            _queue_finish_job(job_id)
            subprocess.run('sleep 2', shell=True)
            continue

        if run_result.returncode == 124:
            add_MONITOR_log(f'{MADE} queue: tpu run 超时且没看到创建job标记 for dir={dir_no}，移回 pending\n')
            _queue_fail_job(job_id)
            continue

        if run_result.returncode != 0:
            add_MONITOR_log(
                f'{MADE} queue: tpu run 失败 for dir={dir_no}，移回 pending。'
                f'完整输出见 {run_log_path}\n'
            )
            _queue_fail_job(job_id)
            continue

        if _run_output_looks_failed(clean_stdout, clean_stderr):
            add_MONITOR_log(
                f'{MADE} queue: tpu run 输出含失败标记(尽管returncode=0) for dir={dir_no}，移回 pending。'
                f'完整输出见 {run_log_path}\n'
            )
            _queue_fail_job(job_id)
            continue

        add_MONITOR_log(
            f'{MADE} queue: tpu run returncode=0 但没看到创建 tmux job 标记 '
            f'for dir={dir_no}，不视为成功，移回 pending。完整输出见 {run_log_path}\n'
        )
        _queue_fail_job(job_id)
        subprocess.run('sleep 2', shell=True)


def mainloop():
    error_jobs = {'deleted': [], 'tpu_still_exists': [], 'resume_next_round': []}
    resumed_tpus_this_round = set()
    data = data_io.read_data()
    jobs_by_window = _jobs_by_window(data)
    active_tpu_occupants = _active_tpu_occupants_by_tpu(data)
    sqa_session_name = data['users'][USER]['tmux_name']
    sqa = read_sqa()
    resume_next_round_set = set(sqa.get('resume_next_round', []))
    check_result = subprocess.run(f'python /home/jzc/zhichengjiang/working/xibo_tpu_manager/tpu.py check {USER}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    add_MONITOR_log(f"{INFO} 我来看看job活着没")
    for job in data["users"][USER]["job_data"]:
        if job['windows_id'] in sqa['running'] or job['windows_id'] in sqa['finished']: continue # have tried this before
        if job['status'] in ['finished', 'rerunned', 'resumed', 'killed']:
            if job['windows_id'] in resume_next_round_set:
                remove_resume_next_round(job['windows_id'])
            continue
        if f'Window {job["windows_id"]}' not in check_result.stdout:
            if job['windows_id'] in resume_next_round_set:
                remove_resume_next_round(job['windows_id'])
            continue
        if job['status'] != 'error':
            if job['windows_id'] in resume_next_round_set:
                remove_resume_next_round(job['windows_id'])
            continue
        child_window = _job_child_window(job)
        if child_window is not None:
            add_MONITOR_log(
                f"{INFO} window {job['windows_id']} 已有 child {child_window}，"
                f"这是旧 error window，本地标记为已处理以免重复resume\n"
            )
            finish_sqa(job['windows_id'])
            continue
        if job['windows_id'] in resume_next_round_set:
            error_jobs['resume_next_round'].append(job)
            continue
        error_type = check_job_status(job)
        if error_type == 'preempted':
            add_MONITOR_log(
                f"{INFO} window {job['windows_id']} 的卡是 preempted；跳过，等外部脚本删卡后下一轮按 deleted 处理"
            )
            continue
        if error_type in error_jobs:
            error_jobs[error_type].append(job)
        elif error_type is None:
            if _has_retrying_ssh_error_in_recent_logs(job, recent_lines=40):
                error_jobs['tpu_still_exists'].append(job)
            else:
                add_MONITOR_log(
                    f"{WARNING} window {job['windows_id']} 卡还在，但output.log最近40行没看到 `Retrying: SSH command error`，判定为误报error，先跳过"
                )

    if len(error_jobs['deleted']) != 0:
        error_windows_list = [(job['user'], job['windows_id']) for job in error_jobs['deleted']]
        add_MONITOR_log(f"{INFO} 我找到了 {len(error_jobs['deleted'])} 个被删掉的卡, 窗口列表是: {error_windows_list}")

    if len(error_jobs['resume_next_round']) != 0:
        error_windows_list = [(job['user'], job['windows_id']) for job in error_jobs['resume_next_round']]
        add_MONITOR_log(f"{INFO} 我找到了 {len(error_jobs['resume_next_round'])} 个下一轮直冲resume的job, 窗口列表是: {error_windows_list}")
    
    all_good = all(len(error_jobs[error_type]) == 0 for error_type in error_jobs)

    if all_good:
        add_MONITOR_log(f"{GAOCHAO} 好像都没问题，睡大觉")
        
    if not all_good:
        resume_targets_this_round = set()

        for job in error_jobs["resume_next_round"]:
            # check whether the window still exists, and the tpu is still there

            _window = job['windows_id']
            _old_tpu = job['tpu']
            old_zone = _extract_zone(_old_tpu) or _get_job_type_zone(job)[1]
            if not is_resume_queue_allowed_zone(old_zone):
                add_MONITOR_log(
                    f'{INFO} window {_window} 旧卡 {_old_tpu}/{old_zone} '
                    f'不在允许区域 ({RESUME_QUEUE_ALLOWED_ZONE_LABEL})，不走buffer直接resume，改为找允许区域空卡\n'
                )
                remove_resume_next_round(_window)
                error_jobs['deleted'].append(job)
                continue
            if f'Window {_window}' not in check_result.stdout:
                add_MONITOR_log(f'{WARNING} window {_window} 已不存在，移出 resume_next_round')
                remove_resume_next_round(_window)
                continue

            tou_result = _run_tou_checked(f'resume_next_round window={_window}')
            if tou_result.returncode != 0:
                add_MONITOR_log(f'{MADE} _tou failed, skip resume_next_round for window {_window}\n')
                continue

            usage, busy_users = _get_tpu_usage_from_tou(tou_result.stdout, _old_tpu)
            if usage != 'idle':
                if usage == 'busy':
                    add_MONITOR_log(
                        f'{INFO} window {_window} 对应的卡 {_old_tpu} 当前 busy users={busy_users}，移出 resume_next_round'
                    )
                else:
                    add_MONITOR_log(
                        f'{INFO} window {_window} 对应的卡 {_old_tpu} 在 tou 中不可见/不确定，移出 resume_next_round'
                    )
                remove_resume_next_round(_window)
                continue

            blocking_desc = _format_tpu_occupants(
                active_tpu_occupants.get(_old_tpu),
                ignore_windows={_window},
            )
            if blocking_desc:
                add_MONITOR_log(
                    f'{INFO} window {_window} 对应的卡 {_old_tpu} 在 tou 中是 idle，'
                    f'但 data/tmux 里已有活跃窗口 {blocking_desc}，移出 resume_next_round'
                )
                remove_resume_next_round(_window)
                continue

            actual_window, resume_action = _get_saving_window(_window)
            superseded_by = _resume_superseded_by(_window, actual_window, jobs_by_window)
            if superseded_by is not None:
                add_MONITOR_log(
                    f'{INFO} window {_window} 已被 child {superseded_by} 覆盖，'
                    f'不再从 actual_window={actual_window} 重复resume\n'
                )
                remove_resume_next_round(_window)
                finish_sqa(_window)
                continue
            if resume_action == 'rerun':
                old_tpu_type = _extract_tpu_type(_old_tpu)
                old_type_allowed = _is_type_allowed(
                    old_tpu_type,
                    low_cost=_is_low_cost_job(job),
                    gpt_b_only=_is_gpt_b_job(job),
                    v5_only=_is_llava15_reproduction_job(job),
                )
                if not old_type_allowed:
                    add_MONITOR_log(
                        f'{INFO} window {_window} 没有完整checkpoint，需要从头rerun；'
                        f'旧卡 {_old_tpu} type={old_tpu_type} 不符合当前job cost class，'
                        f'移出buffer改走正常选卡\n'
                    )
                    remove_resume_next_round(_window)
                    error_jobs['deleted'].append(job)
                    continue
            target_key = f"{resume_action}:{actual_window}"
            if target_key in resume_targets_this_round:
                add_MONITOR_log(
                    f'{WARNING} window {_window} 和本轮前面的任务指向同一个 actual_window={actual_window}，'
                    f'本轮跳过，避免同一轮重复创建child\n'
                )
                continue
            resume_targets_this_round.add(target_key)
            write_sqa(_window)
            try:
                add_MONITOR_log(f'{INFO} 我在试着 resume window {_window}. 这是上一轮我就看见的，buffer命中，直接resume\n')
                if resume_action == 'rerun':
                    resume_cmd = f'{_tpu} rerun {USER} window={actual_window} tpu={_old_tpu}'
                    add_MONITOR_log(f'{INFO} 整条链无saving，改为rerun根窗口 (actual_window={actual_window}): {resume_cmd}\n')
                else:
                    resume_cmd = f'{_tpu} resume {USER} window={actual_window} tpu={_old_tpu}'
                    add_MONITOR_log(f'{INFO} 运行 resume 命令 (actual_window={actual_window}): {resume_cmd}\n')
                resume_result = subprocess.run(
                    resume_cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                _append_resume_file_log(_window, "resume_next_round", resume_cmd, resume_result)
                if _command_created_job(resume_result.stdout, resume_result.stderr):
                    remove_resume_next_round(_window)
                    resumed_tpus_this_round.add(_old_tpu)
                    add_MONITOR_log(
                        f'{GAOCHAO} buffer {"rerun" if resume_action == "rerun" else "resume"} 已创建新tmux job，'
                        f'window {_window} 标记 finished\n'
                    )
                    finish_sqa(_window)
                    if actual_window != _window:
                        add_MONITOR_log(f'{INFO} 关掉无saving的窗口 {_window}\n')
                        os.system(f"tmux kill-window -t {sqa_session_name}:{_window}")
                    continue
                if resume_result.returncode != 0:
                    add_MONITOR_log(f'{MADE} buffer {"rerun" if resume_action == "rerun" else "resume"} failed, skip finish for window {_window}\n')
                    remove_sqa(_window)
                    continue
                if _run_output_looks_failed(resume_result.stdout, resume_result.stderr):
                    add_MONITOR_log(f'{MADE} buffer {"rerun" if resume_action == "rerun" else "resume"} 输出含失败/占用标记，skip finish for window {_window}\n')
                    remove_sqa(_window)
                    continue
                remove_resume_next_round(_window)
                add_MONITOR_log(
                    f'{MADE} buffer {"rerun" if resume_action == "rerun" else "resume"} returncode=0 '
                    f'但没看到创建 tmux job 标记，skip finish for window {_window}\n'
                )
                remove_sqa(_window)
            except Exception as e:
                if isinstance(e, TouCacheStale):
                    remove_sqa(_window)
                    raise
                add_MONITOR_log(f"{MADE} buffer resume 失败了: {e}")
                remove_sqa(_window)
            subprocess.run('sleep 2', shell=True)

        force_deleted_jobs = []
        for job in error_jobs["tpu_still_exists"]:
            _window = job['windows_id']
            _old_tpu = job['tpu']
            try:
                tou_result = _run_tou_checked(f'check old tpu usage window={_window}')
                if tou_result.returncode != 0:
                    add_MONITOR_log(f'{MADE} _tou failed, skip checking tpu usage for window {_window}\n')
                    continue

                usage, busy_users = _get_tpu_usage_from_tou(tou_result.stdout, _old_tpu)
                if usage == 'idle':
                    blocking_desc = _format_tpu_occupants(
                        active_tpu_occupants.get(_old_tpu),
                        ignore_windows={_window},
                    )
                    if blocking_desc:
                        add_MONITOR_log(
                            f'{INFO} window {_window} 对应的卡 {_old_tpu} 在 tou 中是 idle，'
                            f'但 data/tmux 里已有活跃窗口 {blocking_desc}，按卡被占用处理\n'
                        )
                        force_deleted_jobs.append(job)
                        continue
                    old_zone = _extract_zone(_old_tpu) or _get_job_type_zone(job)[1]
                    if not is_resume_queue_allowed_zone(old_zone):
                        add_MONITOR_log(
                            f'{INFO} window {_window} 旧卡 {_old_tpu}/{old_zone} '
                            f'不在允许区域 ({RESUME_QUEUE_ALLOWED_ZONE_LABEL})，按卡没了处理以便换到允许区域\n'
                        )
                        force_deleted_jobs.append(job)
                        continue
                    add_resume_next_round(_window)
                    add_MONITOR_log(f'{INFO} window {_window} 对应的卡 {_old_tpu} 现在是空着的，已加入下一轮resume buffer\n')
                elif usage == 'busy':
                    add_MONITOR_log(f'{INFO} window {_window} 对应的卡 {_old_tpu} 被占用了 users={busy_users}，按卡没了处理\n')
                    force_deleted_jobs.append(job)
                else:
                    add_MONITOR_log(f'{WARNING} window {_window} 对应的卡 {_old_tpu} 在tou里状态不确定，先跳过\n')
            except Exception as e:
                if isinstance(e, TouCacheStale):
                    raise
                add_MONITOR_log(f"{MADE} 检查tpu使用状态失败了: {e}")

        error_jobs["deleted"].extend(force_deleted_jobs)
        used_tpus_this_round = set()
        for job in error_jobs["deleted"]:
            
            _window = job['windows_id']
            _old_tpu = job['tpu']
            write_sqa(_window)

            try:
                add_MONITOR_log(f'{INFO} 我在试着 resume window {_window}. 这卡没了\n')
                target_type, target_zone = _get_job_type_zone(job)
                add_MONITOR_log(f'{INFO} 这个老登的卡型号是 {target_type}, 所在区域是 {target_zone}\n')
                if not target_type or not target_zone:
                    add_MONITOR_log(f'{MADE} 我无法确定这个老登的卡型号和所在区域, 跳过这个老登\n')
                    remove_sqa(_window)
                    continue

                actual_window, resume_action = _get_saving_window(_window)
                superseded_by = _resume_superseded_by(_window, actual_window, jobs_by_window)
                if superseded_by is not None:
                    add_MONITOR_log(
                        f'{INFO} window {_window} 已被 child {superseded_by} 覆盖，'
                        f'不再从 actual_window={actual_window} 重复resume\n'
                    )
                    finish_sqa(_window)
                    continue
                target_key = f"{resume_action}:{actual_window}"
                if target_key in resume_targets_this_round:
                    add_MONITOR_log(
                        f'{WARNING} window {_window} 和本轮前面的任务指向同一个 actual_window={actual_window}，'
                        f'本轮跳过，避免同一轮重复创建child\n'
                    )
                    remove_sqa(_window)
                    continue
                resume_targets_this_round.add(target_key)

                tou_result = _run_tou_checked(f'pick resume tpu window={_window}')

                if tou_result.returncode != 0:
                    add_MONITOR_log(f'{MADE} _tou failed, skip this job\n')
                    remove_sqa(_window)
                    continue

                idle_tpus = _parse_idle_tpus_from_tou(tou_result.stdout)
                gpt_b_only = _is_gpt_b_job(job)
                v5_only = _is_llava15_reproduction_job(job)
                if gpt_b_only:
                    add_MONITOR_log(f'{INFO} window {_window} 命中 GPT-B notes，resume 仅允许 v5p<=32 / v6e<=16\n')
                if v5_only:
                    add_MONITOR_log(f'{INFO} window {_window} 命中 llava-1.5 reproduction，v6 会 OOM，resume/rerun 仅允许 v5\n')
                same_type_resume = resume_action == 'resume'
                same_type_pick = same_type_resume or SAME_TYPE_ONLY
                if same_type_resume:
                    add_MONITOR_log(
                        f'{INFO} window {_window} 有完整checkpoint，resume只允许同type={target_type}；'
                        f'可以跨允许区域找同type空卡\n'
                    )
                else:
                    add_MONITOR_log(
                        f'{INFO} window {_window} 整条链没有完整checkpoint，本次从头rerun，'
                        f'允许按job cost class重新选择卡type\n'
                    )
                new_tpu_name, new_tpu_zone, pick_mode, tpu_already_owned = _pick_idle_tpu(
                    idle_tpus,
                    target_type,
                    target_zone,
                    low_cost=_is_low_cost_job(job),
                    same_type_only=same_type_pick,
                    excluded_tpus=used_tpus_this_round,
                    gpt_b_only=gpt_b_only,
                    v5_only=v5_only,
                )
                if not new_tpu_name:
                    add_MONITOR_log(
                        f'{MADE} 我找不到可用的 IDLE 卡(本轮已排除: {sorted(list(used_tpus_this_round))}), '
                        f'型号是 {target_type}, 所在区域是 {target_zone}, 允许区域={RESUME_QUEUE_ALLOWED_ZONE_LABEL}\n'
                    )
                    remove_sqa(_window)
                    continue

                used_tpus_this_round.add(new_tpu_name)
                resumed_tpus_this_round.add(new_tpu_name)

                new_tpu_type = _extract_tpu_type(new_tpu_name)
                if not new_tpu_type or not new_tpu_zone:
                    add_MONITOR_log(f'{MADE} 我找到了卡 {new_tpu_name}，但无法确定类型/区域，跳过这个老登\n')
                    remove_sqa(_window)
                    continue

                if pick_mode == "cross_type_same_region":
                    add_MONITOR_log(
                        f'{INFO} 未找到同type同zone空卡，启用同区跨类型兜底: {target_type}/{target_zone} -> {new_tpu_type}/{new_tpu_zone}'
                    )
                elif pick_mode == "same_type_same_region":
                    add_MONITOR_log(
                        f'{INFO} 未找到同type同zone空卡，启用同区同type兜底: {target_type}/{target_zone} -> {new_tpu_type}/{new_tpu_zone}'
                    )
                elif pick_mode == "same_type_cross_region":
                    add_MONITOR_log(
                        f'{INFO} 未找到同区同type空卡，启用跨区同type兜底: {target_type}/{target_zone} -> {new_tpu_type}/{new_tpu_zone}'
                    )
                elif pick_mode == "cross_region":
                    add_MONITOR_log(
                        f'{INFO} 未找到同区空卡，启用跨区兜底: {target_type}/{target_zone} -> {new_tpu_type}/{new_tpu_zone}'
                    )
                _tmd_inline = _build_tmd_inline(new_tpu_name, new_tpu_zone)
                if tpu_already_owned:
                    new_alias = _get_tpu_existing_alias(new_tpu_name)
                    if not new_alias:
                        add_MONITOR_log(f'{MADE} 卡 {new_tpu_name} 已注册但找不到 alias，跳过这个老登\n')
                        remove_sqa(_window)
                        continue
                    add_MONITOR_log(f'{INFO} 卡 {new_tpu_name} 已注册，alias={new_alias}，跳过 fang 直接 zhan+tmd\n')
                    fmd_cmd = f'{_tmd_inline} && {_tpu} zhan {new_tpu_name} {USER} && tmd {new_alias}'
                    add_MONITOR_log(f'{INFO} 运行 zhan+mount-disk (已注册卡): tpu zhan {new_tpu_name} {USER} && tmd {new_alias}\n')
                else:
                    new_alias = _pick_new_alias(new_tpu_type, new_tpu_zone)
                    if not new_alias:
                        add_MONITOR_log(f'{MADE} 我找不到可用的 alias, 型号是 {new_tpu_type}, 跳过这个老登\n')
                        remove_sqa(_window)
                        continue
                    add_MONITOR_log(f'{INFO} 我找到了可用的 alias: {new_alias}, 和卡: {new_tpu_name}\n')
                    fmd_cmd = f'{_tmd_inline} && {_tpu} zhan {new_tpu_name} {USER} && {_tpu} fang {new_tpu_name} {new_alias} && tmd {new_alias}'
                    add_MONITOR_log(f'{INFO} 运行放他妈的命令: tpu zhan {new_tpu_name} {USER} && tpu fang {new_tpu_name} {new_alias} && tmd {new_alias}\n')
                fmd_result = subprocess.run(
                    fmd_cmd,
                    shell=True,
                    executable='/bin/bash',
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=900
                )
                if fmd_result.returncode == 124:
                    add_MONITOR_log(f'{MADE} 放+mount-disk超时了！去看看 window {_window} 的log吧。我先跳过\n')
                    remove_sqa(_window)
                    continue
                _append_resume_file_log(_window, "ftmd", fmd_cmd, fmd_result)
                if _command_result_looks_failed(fmd_result):
                    add_MONITOR_log(f'{MADE} 放+mount-disk失败了！去看看 window {_window} 的log吧。我先跳过\n')
                    remove_sqa(_window)
                    continue
                env_cmd, env_result = _run_tpu_jax_check(new_tpu_name, new_tpu_zone)
                _append_resume_file_log(_window, "jax_check", env_cmd, env_result)
                if not _tpu_jax_check_passed(env_result):
                    clean_stdout = _strip_ansi(env_result.stdout or "").strip()
                    clean_stderr = _strip_ansi(env_result.stderr or "").strip()
                    add_MONITOR_log(
                        f'{MADE} 卡 {new_tpu_name} JAX 环境检查失败，不创建 resume job。'
                        f'去看 logs/{_window}/jax_check.txt\n'
                    )
                    if clean_stdout:
                        add_MONITOR_log(f'{INFO} jax_check stdout尾部(window={_window}):\n{_tail_text(clean_stdout)}\n')
                    if clean_stderr:
                        add_MONITOR_log(f'{INFO} jax_check stderr尾部(window={_window}):\n{_tail_text(clean_stderr)}\n')
                    remove_sqa(_window)
                    continue
                add_MONITOR_log(f'{INFO} 卡 {new_tpu_name} JAX 环境检查通过\n')
                add_MONITOR_log(f'{GAOCHAO} 放完了，哈哈')
                if resume_action == 'rerun':
                    resume_cmd = f'{_tpu} rerun {USER} window={actual_window} tpu={new_tpu_name}'
                    add_MONITOR_log(f'{INFO} 整条链无saving，改为rerun根窗口 (actual_window={actual_window}): {resume_cmd}\n')
                else:
                    resume_cmd = f'{_tpu} resume {USER} window={actual_window} tpu={new_tpu_name}'
                    add_MONITOR_log(f'{INFO} 运行 resume 命令 (actual_window={actual_window}): {resume_cmd}\n')
                try:
                    resume_result = subprocess.run(
                        resume_cmd,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=900
                    )
                except subprocess.TimeoutExpired as e:
                    resume_result = _timeout_result(resume_cmd, e)
                _append_resume_file_log(_window, "resume", resume_cmd, resume_result)
                if _command_created_job(resume_result.stdout, resume_result.stderr):
                    add_MONITOR_log(
                        f'{GAOCHAO} {"rerun" if resume_action == "rerun" else "resume"} 已创建新tmux job，'
                        f'window {_window} 标记 finished\n'
                    )
                    finish_sqa(_window)
                    if actual_window != _window:
                        add_MONITOR_log(f'{INFO} 关掉无saving的窗口 {_window}\n')
                        os.system(f"tmux kill-window -t {sqa_session_name}:{_window}")
                    continue
                if resume_result.returncode == 124:
                    add_MONITOR_log(f'{MADE} {"rerun" if resume_action == "rerun" else "resume"} 超时了！去看看 window {_window} 的log吧。我先跳过\n')
                    remove_sqa(_window)
                    continue
                fail = (resume_result.returncode != 0)
                fail = fail or _run_output_looks_failed(resume_result.stdout, resume_result.stderr)
                if fail:
                    add_MONITOR_log(f'{MADE} {"rerun" if resume_action == "rerun" else "resume"} 失败了！去看看 window {_window} 的log吧。我先跳过\n')
                    remove_sqa(_window)
                    continue
                add_MONITOR_log(
                    f'{MADE} {"rerun" if resume_action == "rerun" else "resume"} returncode=0 '
                    f'但没看到创建 tmux job 标记，skip finish for window {_window}\n'
                )
                remove_sqa(_window)
            except subprocess.TimeoutExpired as e:
                add_MONITOR_log(f"{MADE} 我失败了(timeout): {e}\n")
                remove_sqa(_window)
            except Exception as e:
                if isinstance(e, TouCacheStale):
                    remove_sqa(_window)
                    raise
                add_MONITOR_log(f"{MADE} 我失败了: {e}")
                remove_sqa(_window)
            subprocess.run('sleep 2', shell=True)

    _run_queued_jobs(excluded_tpus=resumed_tpus_this_round)


if __name__ == "__main__":
    # ---------------------------------------------------------------------------
    # Subcommands: queue / queue-list / queue-clear
    # Usage examples:
    #   python MONITOR.py queue dir=123
    #   python MONITOR.py queue dir=123 type=v6e-64
    #   python MONITOR.py queue-list
    #   python MONITOR.py queue-clear
    # ---------------------------------------------------------------------------
    _subcmd = sys.argv[1] if len(sys.argv) > 1 else None

    if _subcmd == 'queue':
        _params = {}
        for _arg in sys.argv[2:]:
            if '=' in _arg:
                _k, _v = _arg.split('=', 1)
                _params[_k.strip()] = _v.strip()
        if 'dir' not in _params:
            print("Usage: queue dir=<dir_no> [type=<tpu_type>] [zone=<zone>] [priority=<int>]")
            sys.exit(1)
        queue_add_job(
            dir_no=_params['dir'],
            preferred_type=_params.get('type'),
            preferred_zone=_params.get('zone'),
            priority=_params.get('priority', _params.get('p', 0)),
        )
        sys.exit(0)

    if _subcmd == 'queue-list':
        queue_show()
        sys.exit(0)

    if _subcmd == 'queue-clear':
        _q = _read_queue()
        _cleared = len(_q['pending'])
        _q['pending'] = []
        _write_queue(_q)
        print(f"[queue] 已清空 {_cleared} 个 pending 任务。")
        sys.exit(0)

    # ---------------------------------------------------------------------------
    # Default: run the MONITOR loop
    # ---------------------------------------------------------------------------
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--user', type=str, default='sqa', help='User name (default: sqa)')
    _parser.add_argument(
        '--same-type-only',
        action='store_true',
        help='Also force checkpoint-free reruns to use the original TPU type. Checkpoint resumes are always same-type.',
    )
    _args = _parser.parse_args()
    USER = _args.user
    SAME_TYPE_ONLY = _args.same_type_only

    num_loops = 0
    last_test_time = time.time()
    last_clean_time = time.time()
    add_MONITOR_log(f"\n{GOOD} 哈哈，我上线了。我是何恺明的狗 (给{USER}口的)")

    if data_io.check_code_lock():
        print(f"{MADE} Code is locked for developing, please unlock it first.")
        sys.exit(1)
    try:
        while True:
            # data = data_io.read_data()
            # checking_freq, test_freq, clean_freq = data["MONITOR_config"]["checking_freq"], data["MONITOR_config"]["test_freq"], data["MONITOR_config"]["clean_freq"]

            num_loops += 1
            last_time = time.time()
            try:
                mainloop()
            except TouCacheStale as e:
                time_used = time.time() - last_time
                add_MONITOR_log(
                    f"\n{WARNING} 第 {num_loops} 轮因为 tou cache 过旧而中断，"
                    f"已尝试重启 yizhitou。用时 {time_used:.2f} 秒。现在睡觉 5 分钟."
                )
            else:
                time_used = time.time() - last_time # in seconds
                add_MONITOR_log(f"\n{GAOCHAO} 我看完了。现在是第 {num_loops} 轮，用时 {time_used:.2f} 秒。现在的时间是 {convert_utcstr_to_edtstr(get_abs_time_str())}. 现在睡觉 5 分钟.")
            subprocess.run('sleep 300', shell=True)

    except KeyboardInterrupt:
        print("KeyboardInterrupt, exiting...")
        add_MONITOR_log(f"{FAIL} KeyboardInterrupt, exiting")
        sys.exit(1)
