import os, sys, subprocess
import json
import time
import multiprocessing
import re
import ast
import argparse
from collections import deque
import utils.users as users
import utils.data_io as data_io
import utils.operate as operate
import utils.unit_tests as unit_tests
import utils.jobs as jobs
import utils.clean as clean
from utils.helpers import *

USER = 'sqa'

running_processes = []
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
TYPE_RE = re.compile(r"^(v[0-9a-z]+-\d+)")
ZONE_RE = re.compile(r"(us|asia|europe|australia|northamerica|southamerica)-[a-z0-9-]+-[a-z]")

_tpu = 'python /home/jzc/zhichengjiang/working/xibo_tpu_manager/tpu.py'
_tou = 'python /kmh-nfs-ssd-us-mount/code/qiao/work/tpu_dls/wrap_master.py'

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

def add_MONITOR_log(log):
    """将日志同时追加写入 output.log 并打印到 stdout（带 UTC 时间戳）。"""
    # data = data_io.read_and_lock_data()
    # try:
    #     data["MONITOR_logs"].append({
    #         "time": get_abs_time_str(),
    #         "msg": log
    #     })
    #     data_io.write_and_unlock_data(data)
    # except Exception as e:
    #     print(f"{FAIL} add_MONITOR_log: Failed to add log {log}: {e}")
    #     data_io.release_lock_data()
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
    zone_match = ZONE_RE.search(log_dir)
    if zone_match is None:
        print(f"{MADE} check_job_status: cannot parse zone from log_dir {log_dir}")
        return None
    zone = zone_match.group(0)

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
    return match.group(1) if match else None

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

def _get_job_type_zone(job):
    """从 job dict 中提取 (tpu_type, zone)，类型来自 TPU 名称，zone 来自 log_dir 路径。"""
    old_tpu = job["tpu"]
    target_type = _extract_tpu_type(old_tpu)
    log_dir = job.get("log_dir", "")
    zone_match = ZONE_RE.search(log_dir or "")
    target_zone = zone_match.group(0) if zone_match else None
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
    """检查 job 的 output.log 末尾 recent_lines 行中是否包含 SSH 重试错误，
    用于区分「TPU 还活着但 SSH 卡住」和「TPU 已消失」两种 error 情形。"""
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
    return "Retrying: SSH command error" in tail_text

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

def _is_type_allowed(candidate_type, is_jit):
    """Return True if candidate_type is allowed to resume the job.

    Rules:
      - Always allowed: v6e-64 or larger, v5p-128 or larger.
      - jit-only: v6e-32, v5p-64.
    """
    gen, size = _parse_tpu_gen_size(candidate_type)
    if gen is None:
        return False
    if gen == 'v6e' and size >= 64:
        return True
    if gen == 'v5p' and size >= 128:
        return True
    if is_jit:
        if gen == 'v6e' and size == 32:
            return True
        if gen == 'v5p' and size == 64:
            return True
    return False

def _is_jit_job(job):
    """Return True if the job's wandb notes contain the token 'jit'."""
    notes = (job.get('extra_msgs') or {}).get('spreadsheet_notes') or ''
    return 'jit' in str(notes).lower().split()

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


def _pick_idle_tpu(idle_tpus, target_type, target_zone, is_jit=False):
    """
    Pick an idle TPU that is allowed to resume the job.

    Type rules (see _is_type_allowed):
      - Always allowed: v6e-64+, v5p-128+.
      - jit-only:       v6e-32, v5p-64.

    Region rules:
      - Pass 1: same type + same zone (type must pass rules).
      - Pass 2: allowed type + same region.
      - Pass 3: allowed type + cross-region, but only when both the job's
                region AND the candidate's region are in _CROSS_REGION_ALLOWED
                (us-central1, us-east5, asia-northeast1).

    Return:
      (new_tpu_name, new_zone, pick_mode)
      pick_mode in {"exact", "cross_type_same_region", "cross_region"}
      or (None, None, None) if not found.
    """
    data = data_io.read_data()
    all_tpus = data.get('all_tpus', {})
    target_region = _zone_region(target_zone)

    def _is_owned(tpu_name, zone):
        return tpu_name in all_tpus.get(zone, [])

    # Pass 1: same type + same zone (type still subject to allow rules).
    for tpu_name, zone in idle_tpus:
        if _is_owned(tpu_name, zone):
            continue
        tpu_type = _extract_tpu_type(tpu_name)
        if tpu_type == target_type and zone == target_zone and _is_type_allowed(tpu_type, is_jit):
            return tpu_name, zone, "exact"

    # Pass 2: any allowed type, same region.
    for tpu_name, zone in idle_tpus:
        if _is_owned(tpu_name, zone):
            continue
        tpu_type = _extract_tpu_type(tpu_name)
        if _zone_region(zone) != target_region:
            continue
        if _is_type_allowed(tpu_type, is_jit):
            return tpu_name, zone, "cross_type_same_region"

    # Pass 3: any allowed type, cross-region (both sides must be in allowed set).
    if target_region in _CROSS_REGION_ALLOWED:
        for tpu_name, zone in idle_tpus:
            if _is_owned(tpu_name, zone):
                continue
            tpu_type = _extract_tpu_type(tpu_name)
            candidate_region = _zone_region(zone)
            if candidate_region == target_region:
                continue  # already covered in Pass 2
            if candidate_region not in _CROSS_REGION_ALLOWED:
                continue
            if _is_type_allowed(tpu_type, is_jit):
                return tpu_name, zone, "cross_region"

    return None, None, None

def _pick_new_alias(target_type, zone):
    """从该 zone 的候选 alias 列表中，找出全名不出现在 tou 输出里的第一个 alias
    （即当前没有被任何人用到的 alias），用于 fmd 放卡。返回 alias 字符串或 None。"""
    available_aliases = avilable_aliases(target_type, zone)
    # do tou
    tou_result = subprocess.run(
        _tou,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if tou_result.returncode != 0:
        return None
    # for each alias, find its tpu full name, and look whether it is in output of tou
    clean_stdout = _strip_ansi(tou_result.stdout)
    for a in available_aliases:
        data = data_io.read_data()
        full_name = data['tpu_aliases'][a]
        if full_name not in clean_stdout:
            return a
    return None

# def restart_worker(ka, result_queue):
#     sys.stdout = open(os.devnull, 'w')
#     try:
#         print(f"{INFO} restart_worker: Restarting TPU {ka}...")
#         result = operate.restart(ka)
#         if result == 'success':
#             print(f"{GOOD} restart_worker: Restart TPU {ka} done")
#             add_MONITOR_log(f"{GOOD} restart_worker: Restart TPU {ka} done")
#         else:
#             raise Exception(f"Restart TPU {ka} failed, please contact the admin")
#         result_queue.put(result)
#     except Exception as e:
#         print(f"{FAIL} restart_worker: Failed to restart TPU {ka}: {e}")
#         add_MONITOR_log(f"{FAIL} restart_worker: Failed to restart TPU {ka}: {e}")
#         result_queue.put(e)

def kill_resume(job):
    ka = job["tpu"]
    # print(f"{INFO} kill_resume: Killing jobs  TPU {ka}...")
    operate.kill_jobs_tpu(ka)
    print(f"{INFO} resume job...")
    jobs.resume_rerun_job(job, load_ckpt=True)

def kill_rerun(job):
    ka = job["tpu"]
    # print(f"{INFO} kill_rerun: Killing jobs  TPU {ka}...")
    operate.kill_jobs_tpu(ka)
    print(f"{INFO} rerun job...")
    jobs.resume_rerun_job(job, load_ckpt=False)

# def restart_rerun(job, timeout=900):
#     ka = job["tpu"]
#     print(f"{INFO} restart_rerun: Restarting TPU {ka}...")
#     result_queue = multiprocessing.Queue()
#     process = multiprocessing.Process(target=restart_worker, args=(ka, result_queue))
#     running_processes.append(process)
#     process.start()
#     process.join(timeout)
#     if process.is_alive():
#         print(f"Restart TPU {ka} timeout, killing the process")
#         process.terminate()
#         process.join()
#         running_processes.remove(process)
#         print(f"{WARNING} restart_rerun: Restart TPU {ka} failed, process killed")
#     else:
#         if not result_queue.empty():
#             result = result_queue.get()
#             if isinstance(result, Exception):
#                 print(f"{FAIL} restart_rerun: Restart TPU {ka} failed: {result}")
#                 add_MONITOR_log(f"{FAIL} restart_rerun: Restart TPU {ka} failed: {result}")
#             else:
#                 print(f"{GOOD} Restart TPU {ka} success: {result}, start rerun job")
#                 jobs.resume_rerun_job(job, load_ckpt=False)
#         else:
#             print(f"{FAIL} restart_rerun: Restart TPU {ka} failed, no result returned")
#             add_MONITOR_log(f"{FAIL} restart_rerun: Restart TPU {ka} failed, no result returned")


# def reapply_resume(job, timeout=900):
#     ka = job["tpu"]
#     add_MONITOR_log(f"{INFO} reapply_resume: Reapply TPU {ka}...")
#     result_queue = multiprocessing.Queue()
#     process = multiprocessing.Process(target=reapply_worker, args=(ka, result_queue))
#     running_processes.append(process)
#     process.start()
#     process.join(timeout)
#     if process.is_alive():
#         print(f"Reapply TPU {ka} timeout, killing the process")
#         process.terminate()
#         process.join()
#         running_processes.remove(process)
#         print(f"{WARNING} reapply_resume: Reapply TPU {ka} failed, process killed")
#     else:
#         if not result_queue.empty():
#             result = result_queue.get()
#             if isinstance(result, Exception):
#                 print(f"{FAIL} reapply_resume: Reapply TPU {ka} failed: {result}")
#                 add_MONITOR_log(f"{FAIL} reapply_resume: Reapply TPU {ka} failed: {result}")
#             else:
#                 print(f"{GOOD} Reapply TPU {ka} success: {result}, start resume job")
#                 jobs.resume_rerun_job(job, load_ckpt=True)
#         else:
#             print(f"{FAIL} reapply_resume: Reapply TPU {ka} failed, no result returned")
#             add_MONITOR_log(f"{FAIL} reapply_resume: Reapply TPU {ka} failed, no result returned")

def mainloop():
    error_jobs = {'preempted': [], 'deleted': [], 'tpu_still_exists': [], 'resume_next_round': []}
    data = data_io.read_data()
    sqa_session_name = data['users'][USER]['tmux_name']
    sqa = read_sqa()
    resume_next_round_set = set(sqa.get('resume_next_round', []))
    check_result = subprocess.run(f'python /home/jzc/zhichengjiang/working/xibo_tpu_manager/tpu.py check {USER}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    add_MONITOR_log(f"{INFO} 我来看看job活着没")
    for job in data["users"][USER]["job_data"]:
        if job['windows_id'] in sqa['running'] or job['windows_id'] in sqa['finished']: continue # have tried this before
        if job['status'] in ['finished', 'rerunned', 'resumed', 'killed'] or not job['monitor']:
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
        if job['windows_id'] in resume_next_round_set:
            error_jobs['resume_next_round'].append(job)
            continue
        error_type = check_job_status(job)
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
    
    # if len(error_jobs['preempted']) != 0:
    #     error_windows_list = [(job['user'], job['windows_id']) for job in error_jobs['preempted']]
    #     print(f"{INFO} mainloop: Found {len(error_jobs['preempted'])} preempted jobs, windows list: {error_windows_list}")
    #     add_MONITOR_log(f"{INFO} mainloop: Found {len(error_jobs['preempted'])} preempted jobs, windows list: {error_windows_list}")
    
    # if len(error_jobs['grpc']) != 0:
    #     error_windows_list = [(job['user'], job['windows_id']) for job in error_jobs['grpc']]
    #     print(f"{INFO} mainloop: Found {len(error_jobs['grpc'])} grpc jobs, windows list: {error_windows_list}")
    #     add_MONITOR_log(
    #         f"{INFO} mainloop: Found {len(error_jobs['grpc'])} grpc jobs, windows list: {error_windows_list}"
    #     )
    
    all_good = all(len(error_jobs[error_type]) == 0 for error_type in error_jobs)

    if all_good:
        add_MONITOR_log(f"{GAOCHAO} 好像都没问题，睡大觉")
        
    if not all_good:
        for job in error_jobs["resume_next_round"]:
            # check whether the window still exists, and the tpu is still there

            _window = job['windows_id']
            _old_tpu = job['tpu']

            if f'Window {_window}' not in check_result.stdout:
                add_MONITOR_log(f'{WARNING} window {_window} 已不存在，移出 resume_next_round')
                remove_resume_next_round(_window)
                continue

            tou_result = subprocess.run(
                _tou,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
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

            actual_window, resume_action = _get_saving_window(_window)
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
                if resume_result.returncode != 0:
                    add_MONITOR_log(f'{MADE} buffer {"rerun" if resume_action == "rerun" else "resume"} failed, skip finish for window {_window}\n')
                    remove_sqa(_window)
                    continue
                remove_resume_next_round(_window)
                add_MONITOR_log(f'{GAOCHAO} buffer {"rerun" if resume_action == "rerun" else "resume"} 上了，siuuuuuuuuuuuuuuu')
                finish_sqa(_window)
                if actual_window != _window:
                    add_MONITOR_log(f'{INFO} 关掉无saving的窗口 {_window}\n')
                    os.system(f"tmux kill-window -t {sqa_session_name}:{_window}")
            except Exception as e:
                add_MONITOR_log(f"{MADE} buffer resume 失败了: {e}")
                remove_sqa(_window)
            subprocess.run('sleep 2', shell=True)

        force_deleted_jobs = []
        for job in error_jobs["tpu_still_exists"]:
            _window = job['windows_id']
            _old_tpu = job['tpu']
            try:
                tou_result = subprocess.run(
                    _tou,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                if tou_result.returncode != 0:
                    add_MONITOR_log(f'{MADE} _tou failed, skip checking tpu usage for window {_window}\n')
                    continue

                usage, busy_users = _get_tpu_usage_from_tou(tou_result.stdout, _old_tpu)
                if usage == 'idle':
                    add_resume_next_round(_window)
                    add_MONITOR_log(f'{INFO} window {_window} 对应的卡 {_old_tpu} 现在是空着的，已加入下一轮resume buffer\n')
                elif usage == 'busy':
                    add_MONITOR_log(f'{INFO} window {_window} 对应的卡 {_old_tpu} 被占用了 users={busy_users}，按卡没了处理\n')
                    force_deleted_jobs.append(job)
                else:
                    add_MONITOR_log(f'{WARNING} window {_window} 对应的卡 {_old_tpu} 在tou里状态不确定，先跳过\n')
            except Exception as e:
                add_MONITOR_log(f"{MADE} 检查tpu使用状态失败了: {e}")

        error_jobs["deleted"].extend(force_deleted_jobs)
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

                tou_result = subprocess.run(
                    _tou,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                if tou_result.returncode != 0:
                    add_MONITOR_log(f'{MADE} _tou failed, skip this job\n')
                    continue

                idle_tpus = _parse_idle_tpus_from_tou(tou_result.stdout)
                new_tpu_name, new_tpu_zone, pick_mode = _pick_idle_tpu(idle_tpus, target_type, target_zone, is_jit=_is_jit_job(job))
                if not new_tpu_name:
                    add_MONITOR_log(f'{MADE} 我找不到可用的 IDLE 卡, 型号是 {target_type}, 所在区域是 {target_zone}\n')
                    remove_sqa(_window)
                    continue

                new_tpu_type = _extract_tpu_type(new_tpu_name)
                if not new_tpu_type or not new_tpu_zone:
                    add_MONITOR_log(f'{MADE} 我找到了卡 {new_tpu_name}，但无法确定类型/区域，跳过这个老登\n')
                    remove_sqa(_window)
                    continue

                if pick_mode == "cross_type_same_region":
                    add_MONITOR_log(
                        f'{INFO} 未找到同type同zone空卡，启用同区跨类型兜底: {target_type}/{target_zone} -> {new_tpu_type}/{new_tpu_zone}'
                    )
                elif pick_mode == "cross_region":
                    add_MONITOR_log(
                        f'{INFO} 未找到同区空卡，启用跨区兜底: {target_type}/{target_zone} -> {new_tpu_type}/{new_tpu_zone}'
                    )

                new_alias = _pick_new_alias(new_tpu_type, new_tpu_zone)
                if not new_alias:
                    add_MONITOR_log(f'{MADE} 我找不到可用的 alias, 型号是 {new_tpu_type}, 跳过这个老登\n')
                    remove_sqa(_window)
                    continue
                
                add_MONITOR_log(f'{INFO} 我找到了可用的 alias: {new_alias}, 和卡: {new_tpu_name}\n')
                # fmd_cmd = f'{_tpu} fmd {USER} {new_tpu_name} {new_alias}'
                _tmd_inline = r'tmd() { local force=0; local args=(); for arg in "$@"; do if [ "$arg" = "--force" ]; then force=1; else args+=("$arg"); fi; done; while true; do if [ "$force" = "1" ]; then output=$(' + f'{_tpu}' + r' mount-disk --force "${args[@]}" 2>&1 | tee /dev/tty); else output=$(' + f'{_tpu}' + r' mount-disk "${args[@]}" 2>&1 | tee /dev/tty); fi; if [ "$force" = "0" ] && echo "$output" | grep -q "another process is already mounting"; then sleep 10; echo "sleep 10 and retrying..."; else break; fi; done; }'
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
                fail = (fmd_result.returncode != 0)
                fail = fail or ("is already reserved by" in fmd_result.stdout.lower())
                if fail:
                    add_MONITOR_log(f'{MADE} 放+mount-disk失败了！去看看 window {_window} 的log吧。我先跳过\n')
                    remove_sqa(_window)
                    continue
                add_MONITOR_log(f'{GAOCHAO} 放完了，哈哈')

                actual_window, resume_action = _get_saving_window(_window)
                if resume_action == 'rerun':
                    resume_cmd = f'{_tpu} rerun {USER} window={actual_window} tpu={new_tpu_name}'
                    add_MONITOR_log(f'{INFO} 整条链无saving，改为rerun根窗口 (actual_window={actual_window}): {resume_cmd}\n')
                else:
                    resume_cmd = f'{_tpu} resume {USER} window={actual_window} tpu={new_tpu_name}'
                    add_MONITOR_log(f'{INFO} 运行 resume 命令 (actual_window={actual_window}): {resume_cmd}\n')
                resume_result = subprocess.run(
                    resume_cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=900
                )
                if resume_result.returncode == 124:
                    add_MONITOR_log(f'{MADE} {"rerun" if resume_action == "rerun" else "resume"} 超时了！去看看 window {_window} 的log吧。我先跳过\n')
                    remove_sqa(_window)
                    continue
                _append_resume_file_log(_window, "resume", resume_cmd, resume_result)
                fail = (resume_result.returncode != 0)
                fail = fail or ("is already reserved by" in resume_result.stdout.lower())
                if fail:
                    add_MONITOR_log(f'{MADE} {"rerun" if resume_action == "rerun" else "resume"} 失败了！去看看 window {_window} 的log吧。我先跳过\n')
                    remove_sqa(_window)
                    continue
                add_MONITOR_log(f'{GAOCHAO} {"rerun" if resume_action == "rerun" else "resume"} 上了，siuuuuuuuuuuuuuuu')
                finish_sqa(_window)
                if actual_window != _window:
                    add_MONITOR_log(f'{INFO} 关掉无saving的窗口 {_window}\n')
                    os.system(f"tmux kill-window -t {sqa_session_name}:{_window}")
            except subprocess.TimeoutExpired as e:
                add_MONITOR_log(f"{MADE} 我失败了(timeout): {e}\n")
                remove_sqa(_window)
            except Exception as e:
                add_MONITOR_log(f"{MADE} 我失败了: {e}")
                raise e
                # remove job from sqa.json
                remove_sqa(_window)
            subprocess.run('sleep 2', shell=True)

            # user = job["user"]
            # data = data_io.read_and_lock_data()
            # try:
            #     for jb in data["users"][user]["job_data"]:
            #         if jb["windows_id"] == job["windows_id"]:
            #             jb["status"] = 'error'
            #             jb['error'] = error_type
            #     data_io.write_and_unlock_data(data)
            # except:
            #     print(f"{FAIL} mainloop: Failed to update job {job['windows_id']} for user {user}")
            #     add_MONITOR_log(f"{FAIL} mainloop: Failed to update job {job['windows_id']} for user {user}")
            #     data_io.release_lock_data()

    # if not all_good:
    #     for error_type in error_jobs:
    #         for job in error_jobs[error_type]:
    #             rule = job["rules"][error_type]
    #             try:
    #                 if rule == 'pass':      continue
    #                 elif rule == 'reapply': reapply_resume(job, timeout=1800)
    #                 elif rule == 'resume':  kill_resume(job)
    #                 elif rule == 'rerun':   kill_rerun(job)
    #                 elif rule == 'restart': restart_rerun(job)
    #             except:
    #                 print(f"{FAIL} mainloop: Failed to handle job {job['windows_id']} for user {user}, (error type {error_type}, rule {rule})")
    #                 add_MONITOR_log(f"{FAIL} mainloop: Failed to handle job {job['windows_id']} for user {user}, (error type {error_type}, rule {rule})")
    

if __name__ == "__main__":
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--user', type=str, default='sqa', help='User name (default: sqa)')
    _args = _parser.parse_args()
    USER = _args.user

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
            mainloop()
            time_used = time.time() - last_time # in seconds

            add_MONITOR_log(f"\n{GAOCHAO} 我看完了。现在是第 {num_loops} 轮，用时 {time_used:.2f} 秒。现在的时间是 {convert_utcstr_to_edtstr(get_abs_time_str())}. 现在睡觉 5 分钟.")
            subprocess.run('sleep 300', shell=True)

    except KeyboardInterrupt:
        print("KeyboardInterrupt, exiting...")
        # kill all the processes
        for process in running_processes:
            process.terminate()
            process.join()
        print("All processes killed")
        add_MONITOR_log(f"{FAIL} KeyboardInterrupt, all processes killed")
        sys.exit(1)