# -*- coding: utf-8 -*-
"""
web_ui.py — TPU 任务可视化网站（Flask）
特性要点：
- 顶栏用户切换、Clean/Clean -re、一键刷新(R)
- 每个 user 的表格：Window/DIR/TPU/Tags/Status/Actions
  * 隐藏 DIR 或 TPU 为 unknown 的行（更干净）
  * Tags 与正文字体一致，过长可横向滑动
  * Sticky 表头不再与内容重叠
- 按钮：Resume / Rerun / chooseTPU Resume / chooseTPU Rerun / Kill / 查看Log
  * 点击立即有“已提交”提示 + 行内转圈；后台线程完成后弹出 成功/失败 并刷新
- choose TPU 弹窗（“像 tpu find 一样”的表格）：仅读取 sheet.read_sheet_info()
  * 只用 spreadsheet 的列表（去重且字段完整）
  * 支持 “在所选 TPU 上 Resume/Rerun”
  * 还支持 “申请并 Resume/Rerun”（若 TPU 不存在可直接申请后续上）
- TPU 面板（类似 spreadsheet）：
  * 只显示 read_sheet_info() 的行
  * 状态为空或 NOT FOUND 的行出现 Apply（可设次数/间隔，默认 20/5）
- 日志页（/log/<username>/<window_id>）：
  * 采用你提供的版本：?n= 字符数、?ansi=0 去色、跟随尾部、自动刷新、快捷键
"""

import os, re, sys, json, time, shlex, subprocess, threading, uuid, shutil, html as _html
from types import SimpleNamespace
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
from flask import Flask, jsonify, redirect, render_template_string, request, url_for, Response, session
from functools import wraps

HERE = os.path.dirname(os.path.abspath(__file__))
TPU_PY_PATH = os.environ.get("TPU_PY_PATH", os.path.join(HERE, "tpu.py"))

# ---------- 可选导入（存在则用，不存在走降级方案） ----------
DATA_IO_OK = True
USERS_MODULE_OK = True
SHEET_MODULE_OK = True

try:
    from data_io import read_data  # type: ignore
except Exception:
    DATA_IO_OK = False
    def read_data() -> Dict[str, Any]:
        for p in [os.path.join(HERE, "data.json"), os.path.join(os.path.dirname(HERE), "data.json")]:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
        return {"users": {}, "tpu_aliases": {}, "all_tpus": {}}

try:
    from utils import users as users_mod  # type: ignore
except Exception:
    USERS_MODULE_OK = False
    users_mod = None

try:
    from utils import sheet as sheet_mod  # type: ignore
except Exception:
    SHEET_MODULE_OK = False
    sheet_mod = None

ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s or "")

DEFAULT_SETTINGS = {
    "monitor_after_run": True,
    "monitor_upd_time": 5,
    "monitor_length": 800,
    "monitor_dir": True,
    "monitor_tpu": True,
    "monitor_verbose": False,
    "show_length": 200,
    "time_zone": "us",
    "extra_settings": {},
}

def _user_from_dict(d: Dict[str, Any]) -> SimpleNamespace:
    if USERS_MODULE_OK and hasattr(users_mod, "user_from_dict"):
        return users_mod.user_from_dict(d)  # type: ignore
    ns = SimpleNamespace()
    ns.name = d.get("name") or d.get("username")
    ns.tmux_name = d.get("tmux_name") or ns.name
    ns.job_data = d.get("job_data", [])
    ns.settings = d.get("settings", DEFAULT_SETTINGS.copy())
    ns.windows_offset = d.get("windows_offset", 0)
    ns.spreadsheet_name = d.get("name", ns.name)
    return ns

@lru_cache(maxsize=1)
def list_all_users() -> List[str]:
    data = read_data()
    if "user_list" in data and isinstance(data["user_list"], list) and data["user_list"]:
        return sorted([str(u) for u in data["user_list"]])
    return sorted(list(data.get("users", {}).keys()))

def get_user_obj(username: str) -> Optional[SimpleNamespace]:
    data = read_data()
    u = data.get("users", {}).get(username)
    if not u:
        return None
    return _user_from_dict(u)

def list_tmux_windows(session_name: str) -> List[Tuple[str, str]]:
    try:
        out = subprocess.check_output(
            ["tmux", "list-windows", "-t", session_name],
            text=True, stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError:
        return []
    res = []
    for line in out.splitlines():
        if not line.strip(): continue
        try:
            left = line.split(":")[0].strip()
            name = line.split(":")[1].split(" ")[0].strip()
            res.append((left, name))
        except Exception:
            continue
    return res

def tmux_capture(session: str, window_id: str, last_n: int = 2000) -> str:
    try:
        # First, try to get the history limit for this pane
        # Use -S - to capture from the start of history to current position
        # -E - means to current position (end)
        # This ensures we get all available history, not just visible area
        # We also try to increase history limit if needed by using -S with a large negative number
        # Calculate approximate line count needed (assuming ~80 chars per line)
        if last_n > 0:
            approx_lines = max(50000, (last_n // 80) + 1000)  # Add buffer
        else:
            approx_lines = 50000
        
        # Try to capture with explicit line range
        # -S -<n> means start from n lines before current, -E - means end at current
        # But we want from the very beginning, so use -S - (start of history)
        out = subprocess.check_output(
            ["tmux", "capture-pane", "-t", f"{session}:{window_id}", "-p", "-S", "-", "-E", "-"],
            text=True, stderr=subprocess.STDOUT
        )
        out = out.rstrip()
        
        # If we got content but it's shorter than expected and we need more,
        # try to increase history limit temporarily
        if last_n > 0 and len(out) < last_n:
            # Try to get more by using a larger history buffer
            # First check current history limit
            try:
                hist_limit_cmd = ["tmux", "show-options", "-t", f"{session}:{window_id}", "-wv", "history-limit"]
                current_limit = subprocess.check_output(hist_limit_cmd, text=True, stderr=subprocess.STDOUT).strip()
                current_limit_int = int(current_limit) if current_limit.isdigit() else 2000
                
                # If current limit is too small, try to increase it temporarily
                if current_limit_int < approx_lines:
                    try:
                        subprocess.check_output(
                            ["tmux", "set-option", "-t", f"{session}:{window_id}", "-w", "history-limit", str(approx_lines)],
                            text=True, stderr=subprocess.STDOUT
                        )
                        # Retry capture
                        out = subprocess.check_output(
                            ["tmux", "capture-pane", "-t", f"{session}:{window_id}", "-p", "-S", "-", "-E", "-"],
                            text=True, stderr=subprocess.STDOUT
                        )
                        out = out.rstrip()
                    except Exception:
                        pass  # If we can't increase limit, use what we have
            except Exception:
                pass  # If we can't check/change limit, use what we have
        
        if last_n and len(out) > last_n:
            return out[-last_n:]
        return out
    except subprocess.CalledProcessError:
        return ""

# --------- 状态检测（与 CLI 表达式一致） ---------
RE_ERR_GENERIC = re.compile(r"(Job failed|[eE]rror|FAIL)")
RE_OOM_ALLOCTYPE = re.compile(r"Allocation type")
RE_GRPC = re.compile(r"GRPC [Ee]rror")
RE_FILE_NOT_FOUND = re.compile(r"python: No such file or directory")
RE_DEADLINE = re.compile(r"DEADLINE_EXCEEDED")
RE_COMPILING = re.compile(r"[cC]ompil(ing|ation|e)")
RE_SAMPLING = re.compile(r"[sS]ampling ")
RE_EPOCH1 = re.compile(r"[eE]poch\s([0-9]{1,6})")
RE_EPOCH2 = re.compile(r"ep\s*=\s*([0-9]{1,4}(?:\.[0-9]{1,6})?)")
RE_INITIALIZING = re.compile(r"[iI]nitializing")
RE_STAGING = re.compile(r"[sS]taging")

def parse_status(job_data: Dict[str, Any], last_cut: str, last_full: str) -> Tuple[str, str]:
    st = job_data.get("status")
    if st == "starting": return ("Don't have logdir yet", "warn")
    if st == "error":
        err = job_data.get("error")
        if err == "preempted": return ("Preempted", "error")
        if err == "OOM": return ("OOM", "error")
        return ("Error", "error")
    if st == "killed": return ("Killed", "warn")
    if st in ("resumed", "rerunned"):
        child = (job_data.get("extra_msgs") or {}).get("child")
        msg = job_data.get("error") or st
        return (f"{msg} (child={child})" if child is not None else msg, "warn")
    if st == "finished": return ("Finished", "success")
    # running/starting/None
    if RE_ERR_GENERIC.search(last_cut):
        if RE_OOM_ALLOCTYPE.search(last_full): return ("OOM Error", "error")
        if RE_GRPC.search(last_full): return ("GRPC Error", "error")
        if RE_FILE_NOT_FOUND.search(last_full): return ("File Error", "error")
        if RE_DEADLINE.search(last_full): return ("DEADLINE EXCEEDED", "error")
        return ("Unknown Error", "error")
    if RE_COMPILING.search(last_cut): return ("Compiling", "info")
    if RE_SAMPLING.search(last_cut):
        m1 = RE_EPOCH1.search(last_cut) or RE_EPOCH2.search(last_cut)
        if m1:
            try:
                epf = float(m1.group(1))
                return (f"Sampling (ep={epf:.2f})", "info")
            except Exception:
                return (f"Sampling (ep={m1.group(1)})", "info")
        return ("Sampling", "info")
    m1 = RE_EPOCH1.search(last_cut)
    m2 = RE_EPOCH2.search(last_cut)
    print(f'last_cut: {last_cut}, m1: {m1}, m2: {m2}')
    if m1: return (f"Running (ep={m1.group(1)})", "success")
    if m2:
        try: return (f"Running (ep={float(m2.group(1)):.2f})", "success")
        except Exception: return (f"Running (ep={m2.group(1)})", "success")
    if RE_INITIALIZING.search(last_cut): return ("Initializing", "info")
    if RE_STAGING.search(last_cut): return ("Staging", "info")
    return ("Unknown", "warn")

def summarize_job_row(user_obj: SimpleNamespace, window_id: str, job_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    # 只渲染 job_data 存在的窗口，并且 dir/tpu 不可 unknown
    if not job_data:
        return None

    settings = getattr(user_obj, "settings", {}) or {}
    show_length = int(settings.get("show_length", DEFAULT_SETTINGS["show_length"]))
    monitor_length = int(settings.get("monitor_length", DEFAULT_SETTINGS["monitor_length"]))

    last_full = tmux_capture(user_obj.tmux_name, window_id, last_n=max(monitor_length, show_length))
    last_cut = last_full[-monitor_length:] if last_full else ""

    try:
        dir_ = (job_data.get("job_dir") or "").split("/")[-1] or "(unknown)"
    except Exception:
        dir_ = "(unknown)"
    tpu_full = job_data.get("tpu")
    if tpu_full:
        tpu_show = tpu_full[10:] if len(tpu_full) > 10 else tpu_full
    else:
        tpu_show = "(unknown)"

    # 过滤 unknown
    if dir_ in ("(unknown)", "", None) or tpu_show in ("(unknown)", "", None, "unknown"):
        return None

    s_text, s_cls = parse_status(job_data, last_cut, last_full)

    return {
        "window": str(window_id),
        "dir": dir_,
        "tpu": tpu_show,
        "tags": str(job_data.get("job_tags") or ""),
        "status": s_text,
        "status_class": s_cls,
        "in_data": True,
    }

def build_job_rows(username: str) -> Dict[str, Any]:
    user_obj = get_user_obj(username)
    if not user_obj:
        return {"jobs": [], "session_exists": False, "error": f"user '{username}' not found"}
    windows = list_tmux_windows(user_obj.tmux_name)
    job_map = {}
    for job in getattr(user_obj, "job_data", []):
        try: job_map[str(job.get("windows_id"))] = job
        except Exception: continue
    rows: List[Dict[str, Any]] = []
    for wid, _ in windows:
        row = summarize_job_row(user_obj, wid, job_map.get(str(wid)))
        if row is not None:
            rows.append(row)
    return {"jobs": rows, "session_exists": True}

# ---------- CLI 封装 ----------
def _ensure_tpu_py() -> str:
    if os.path.exists(TPU_PY_PATH): return TPU_PY_PATH
    for c in [TPU_PY_PATH, os.path.join(HERE, "tpu.py"), os.path.join(os.path.dirname(HERE), "tpu.py")]:
        if os.path.exists(c): return c
    exe = shutil.which("tpu")
    return exe or TPU_PY_PATH

def call_cli(*args: str) -> Tuple[int, str]:
    tpu_py = _ensure_tpu_py()
    if os.path.basename(tpu_py) == "tpu":
        cmd = [tpu_py] + list(args)
    else:
        cmd = [sys.executable, tpu_py] + list(args)
    try:
        p = subprocess.run(cmd, text=True, capture_output=True)
        out = (p.stdout or "") + ("\n" + (p.stderr or "") if p.stderr else "")
        return p.returncode, out.strip()
    except Exception as e:
        return 1, f"Failed to run: {' '.join(shlex.quote(a) for a in cmd)}\nError: {e}"

def action_clean(username: str, re_flag: bool = False) -> Tuple[bool, str]:
    args = ["clean", username]
    if re_flag: args.append("-re")
    code, out = call_cli(*args)
    return code == 0, out

# ---------- 异步任务 ----------
OP_TASKS: Dict[str, Dict[str, Any]] = {}
OP_LOCK = threading.Lock()

def _register_task(kind: str, meta: Dict[str, Any]) -> str:
    tid = str(uuid.uuid4())
    with OP_LOCK:
        OP_TASKS[tid] = {"kind": kind, "status": "pending", "msg": "", "meta": meta, "ts": time.time()}
    return tid

def _finish_task(tid: str, ok: bool, msg: str, log: str = ""):
    with OP_LOCK:
        if tid in OP_TASKS:
            OP_TASKS[tid]["status"] = "ok" if ok else "err"
            OP_TASKS[tid]["msg"] = msg
            OP_TASKS[tid]["log"] = log
            OP_TASKS[tid]["te"] = time.time()

def _run_resume(username: str, window_id: str, new_tpu: Optional[str], tid: str):
    args = ["resume", f"window={window_id}", username]
    if new_tpu: args.insert(2, f"tpu={new_tpu}")
    code, out = call_cli(*args)
    _finish_task(tid, code == 0, out, out)

def _run_rerun(username: str, window_id: str, new_tpu: Optional[str], tid: str):
    args = ["rerun", f"window={window_id}", username]
    if new_tpu: args.insert(2, f"tpu={new_tpu}")
    code, out = call_cli(*args)
    _finish_task(tid, code == 0, out, out)

def _run_kill(username: str, window_id: str, tid: str):
    args = ["kill", f"window={window_id}", username]
    code, out = call_cli(*args)
    if code != 0:
        code2, out2 = call_cli("-kj", f"window={window_id}", username)
        if code2 == 0: code, out = code2, out2
    _finish_task(tid, code == 0, out, out)

# Apply 循环（面板用）
def _run_apply(alias: str, times: int = 20, sleep_sec: int = 5, tid: Optional[str] = None):
    ok = False; last = ""
    log_output = ""
    
    for i in range(max(1, times)):
        code, out = call_cli("apply", alias)
        last = out
        log_output += f"Attempt {i+1}/{times}:\n{out}\n\n"
        
        # Store log during process if tid is provided
        if tid:
            with OP_LOCK:
                if tid in OP_TASKS:
                    OP_TASKS[tid]["log"] = log_output
        
        # Check both exit code and output for success indicators
        if code == 0 and "[FAIL]" not in out and "not found" not in out.lower():
            ok = True; break
        time.sleep(max(1, sleep_sec))
    if tid: _finish_task(tid, ok, last, log_output)

def _run_reapply(alias: str, times: int = 20, sleep_sec: int = 5, tid: Optional[str] = None):
    """Reapply: delete then apply"""
    log_output = ""
    
    # First delete the TPU
    code, out = call_cli("delete", alias)
    log_output += f"Delete attempt:\n{out}\n\n"
    if code != 0:
        if tid: _finish_task(tid, False, f"delete failed: {out}", log_output)
        return
    
    # Then apply for the TPU
    ok = False; last = ""
    for i in range(max(1, times)):
        code, out = call_cli("apply", alias)
        last = out
        log_output += f"Apply attempt {i+1}/{times}:\n{out}\n\n"
        
        # Store log during process if tid is provided
        if tid:
            with OP_LOCK:
                if tid in OP_TASKS:
                    OP_TASKS[tid]["log"] = log_output
        
        # Check both exit code and output for success indicators
        if code == 0 and "[FAIL]" not in out and "not found" not in out.lower():
            ok = True; break
        time.sleep(max(1, sleep_sec))
    if tid: _finish_task(tid, ok, last, log_output)

# 申请并 Resume/Rerun（choose器用）
def _run_apply_then_resume(username: str, window_id: str, alias: str, pre_str: Optional[str], tid: str):
    # Update status to "Applying for TPU"
    with OP_LOCK:
        if tid in OP_TASKS:
            OP_TASKS[tid]["status"] = "applying"
            OP_TASKS[tid]["msg"] = f"Applying for TPU {alias}..."
    
    # Apply for the TPU with retry logic
    times = 20
    sleep_sec = 5
    ok = False
    last = ""
    log_output = ""
    
    for i in range(max(1, times)):
        # Update progress
        with OP_LOCK:
            if tid in OP_TASKS:
                OP_TASKS[tid]["msg"] = f"Applying for TPU {alias}... (attempt {i+1}/{times})"
        
        args = ["apply"]
        if pre_str is not None and str(pre_str).lower() in ("false", "0", "no"):
            args += ["-norm", alias]
        else:
            args += [alias]
            
        code, out = call_cli(*args)
        last = out
        log_output += f"Attempt {i+1}/{times}:\n{out}\n\n"
        
        # Store log after each attempt so it's available during progress
        with OP_LOCK:
            if tid in OP_TASKS:
                OP_TASKS[tid]["log"] = log_output
        
        # Check both exit code and output for success indicators
        if code == 0 and "[FAIL]" not in out and "not found" not in out.lower():
            ok = True
            break
        time.sleep(max(1, sleep_sec))
    
    if not ok:
        _finish_task(tid, False, f"apply failed after {times} attempts: {last}", log_output)
        return
    
    # Update status to "TPU applied, resuming job"
    with OP_LOCK:
        if tid in OP_TASKS:
            OP_TASKS[tid]["status"] = "mounting"
            OP_TASKS[tid]["msg"] = f"TPU {alias} applied, resuming job..."
    
    _run_resume(username, window_id, alias, tid)

def _run_apply_then_rerun(username: str, window_id: str, alias: str, pre_str: Optional[str], tid: str):
    # Update status to "Applying for TPU"
    with OP_LOCK:
        if tid in OP_TASKS:
            OP_TASKS[tid]["status"] = "applying"
            OP_TASKS[tid]["msg"] = f"Applying for TPU {alias}..."
    
    # Apply for the TPU with retry logic
    times = 20
    sleep_sec = 5
    ok = False
    last = ""
    log_output = ""
    
    for i in range(max(1, times)):
        # Update progress
        with OP_LOCK:
            if tid in OP_TASKS:
                OP_TASKS[tid]["msg"] = f"Applying for TPU {alias}... (attempt {i+1}/{times})"
        
        args = ["apply"]
        if pre_str is not None and str(pre_str).lower() in ("false", "0", "no"):
            args += ["-norm", alias]
        else:
            args += [alias]
        
        code, out = call_cli(*args)
        last = out
        log_output += f"Attempt {i+1}/{times}:\n{out}\n\n"
        
        # Store log after each attempt so it's available during progress
        with OP_LOCK:
            if tid in OP_TASKS:
                OP_TASKS[tid]["log"] = log_output
        
        # Check both exit code and output for success indicators
        if code == 0 and "[FAIL]" not in out and "not found" not in out.lower():
            ok = True
            break
        time.sleep(max(1, sleep_sec))
    
    if not ok:
        _finish_task(tid, False, f"apply failed after {times} attempts: {last}", log_output)
        return
    
    # Update status to "TPU applied, rerunning job"
    with OP_LOCK:
        if tid in OP_TASKS:
            OP_TASKS[tid]["status"] = "mounting"
            OP_TASKS[tid]["msg"] = f"TPU {alias} applied, rerunning job..."
    
    _run_rerun(username, window_id, alias, tid)

# 重新申请并 Resume/Rerun（choose器用）
def _run_reapply_then_resume(username: str, window_id: str, alias: str, pre_str: Optional[str], tid: str):
    # Update status to "Deleting TPU"
    with OP_LOCK:
        if tid in OP_TASKS:
            OP_TASKS[tid]["status"] = "applying"
            OP_TASKS[tid]["msg"] = f"Deleting TPU {alias}..."
    
    log_output = ""
    
    # First delete the TPU
    code, out = call_cli("delete", alias)
    log_output += f"Delete attempt:\n{out}\n\n"
    if code != 0:
        _finish_task(tid, False, f"delete failed: {out}", log_output)
        return
    
    # Update status to "Applying for TPU"
    with OP_LOCK:
        if tid in OP_TASKS:
            OP_TASKS[tid]["msg"] = f"Applying for TPU {alias}..."
    
    # Apply for the TPU with retry logic
    times = 20
    sleep_sec = 5
    ok = False
    last = ""
    
    for i in range(max(1, times)):
        # Update progress
        with OP_LOCK:
            if tid in OP_TASKS:
                OP_TASKS[tid]["msg"] = f"Applying for TPU {alias}... (attempt {i+1}/{times})"
        
        args = ["apply"]
        if pre_str is not None and str(pre_str).lower() in ("false", "0", "no"):
            args += ["-norm", alias]
        else:
            args += [alias]
        
        code, out = call_cli(*args)
        last = out
        log_output += f"Attempt {i+1}/{times}:\n{out}\n\n"
        
        # Store log after each attempt so it's available during progress
        with OP_LOCK:
            if tid in OP_TASKS:
                OP_TASKS[tid]["log"] = log_output
        
        # Check both exit code and output for success indicators
        if code == 0 and "[FAIL]" not in out and "not found" not in out.lower():
            ok = True
            break
        time.sleep(max(1, sleep_sec))
    
    if not ok:
        _finish_task(tid, False, f"apply failed after {times} attempts: {last}", log_output)
        return
    
    # Update status to "TPU applied, resuming job"
    with OP_LOCK:
        if tid in OP_TASKS:
            OP_TASKS[tid]["status"] = "mounting"
            OP_TASKS[tid]["msg"] = f"TPU {alias} applied, resuming job..."
    
    _run_resume(username, window_id, alias, tid)

def _run_reapply_then_rerun(username: str, window_id: str, alias: str, pre_str: Optional[str], tid: str):
    # Update status to "Deleting TPU"
    with OP_LOCK:
        if tid in OP_TASKS:
            OP_TASKS[tid]["status"] = "applying"
            OP_TASKS[tid]["msg"] = f"Deleting TPU {alias}..."
    
    log_output = ""
    
    # First delete the TPU
    code, out = call_cli("delete", alias)
    log_output += f"Delete attempt:\n{out}\n\n"
    if code != 0:
        _finish_task(tid, False, f"delete failed: {out}", log_output)
        return
    
    # Update status to "Applying for TPU"
    with OP_LOCK:
        if tid in OP_TASKS:
            OP_TASKS[tid]["msg"] = f"Applying for TPU {alias}..."
    
    # Apply for the TPU with retry logic
    times = 20 # 
    sleep_sec = 5 # 间隔5秒
    ok = False
    last = ""
    
    for i in range(max(1, times)):
        # Update progress
        with OP_LOCK:
            if tid in OP_TASKS:
                OP_TASKS[tid]["msg"] = f"Applying for TPU {alias}... (attempt {i+1}/{times})"
        
        args = ["apply"]
        if pre_str is not None and str(pre_str).lower() in ("false", "0", "no"):
            args += ["-norm", alias]
        else:
            args += [alias]
        
        code, out = call_cli(*args)
        last = out
        log_output += f"Attempt {i+1}/{times}:\n{out}\n\n"
        
        # Store log after each attempt so it's available during progress
        with OP_LOCK:
            if tid in OP_TASKS:
                OP_TASKS[tid]["log"] = log_output
        
        # Check both exit code and output for success indicators
        if code == 0 and "[FAIL]" not in out and "not found" not in out.lower():
            ok = True
            break
        time.sleep(max(1, sleep_sec))
    
    if not ok:
        _finish_task(tid, False, f"apply failed after {times} attempts: {last}", log_output)
        return
    
    # Update status to "TPU applied, rerunning job"
    with OP_LOCK:
        if tid in OP_TASKS:
            OP_TASKS[tid]["status"] = "mounting"
            OP_TASKS[tid]["msg"] = f"TPU {alias} applied, rerunning job..."
    
    _run_rerun(username, window_id, alias, tid)

def start_thread(target, *a, **kw):
    th = threading.Thread(target=target, args=a, kwargs=kw, daemon=True)
    th.start()
    return th

# ---------- Spreadsheet（严格使用 read_sheet_info） ----------
def _get_tpu_information_all() -> Dict[str, Dict[str, Any]]:
    """read_sheet_info() -> {full_name: info_dict}"""
    if not SHEET_MODULE_OK:
        return {}
    if hasattr(sheet_mod, 'read_sheet_info'):
        try:
            info = sheet_mod.read_sheet_info()  # type: ignore
            if isinstance(info, dict):
                return {str(k): (v or {}) for k, v in info.items()}
        except Exception:
            pass
    return {}

def fetch_tpu_sheet_rows() -> List[Dict[str, Any]]:
    """扁平化 read_sheet_info() 结果供 UI 使用"""
    info_all = _get_tpu_information_all()
    rows: List[Dict[str, Any]] = []
    for full_name, info in sorted(info_all.items(), key=lambda x: str(x[0])):
        alias = str(info.get('alias') or full_name)
        status = str(info.get('running_status') or '')
        zone   = str(info.get('zone') or '')
        pre    = info.get('pre')
        if isinstance(pre, bool): pre_str = 'true' if pre else 'false'
        else: pre_str = str(pre or '')
        user   = str(info.get('user') or '')
        note   = str(info.get('user_note') or '')
        script_note = str(info.get('script_note') or '')
        version = str(info.get('version') or '')
        # If version is not available, try to extract from alias or full_name
        if not version:
            if alias.startswith('v4') or 'v4' in full_name:
                version = 'v4'
            elif alias.startswith('v5') or 'v5' in full_name:
                version = 'v5'
            elif alias.startswith('v6') or 'v6' in full_name:
                version = 'v6'
        rows.append({
            "alias": alias,
            "full_name": full_name,
            "running_status": status,
            "zone": zone,
            "preemptible": pre_str,
            "user": user,
            "user_note": note,
            "script_note": script_note,
            "version": version,
            "found": True
        })
    return rows

# ---------- Flask ----------
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key-change-this-in-production')

# Import authentication functions
try:
    from utils import autenticate as auth_mod
except Exception:
    auth_mod = None

def get_web_passwords():
    """Get web passwords from passwords.json"""
    if auth_mod and hasattr(auth_mod, 'get_web_passwords'):
        return auth_mod.get_web_passwords()
    passwords_file = os.path.join(HERE, 'passwords.json')
    if not os.path.exists(passwords_file):
        return {}
    try:
        with open(passwords_file, 'r') as f:
            passwords = json.load(f)
            return passwords if isinstance(passwords, dict) else {}
    except Exception:
        return {}

def password_hash(password):
    """Hash password using SHA256"""
    import hashlib
    return hashlib.sha256(password.encode()).hexdigest()

def check_permission(required_permission=None, username=None):
    """
    Check if user has required permission.
    - If required_permission is None, just check if logged in
    - If required_permission is 'admin', check if user has 'admin' permission
    - If required_permission is a username, check if user has that username permission
    - username parameter is for checking access to specific user's data
    """
    if 'permissions' not in session:
        return False
    
    permissions = session['permissions']
    
    # Admin has all access
    if 'admin' in permissions:
        return True
    
    # If checking for specific username access
    if username:
        return username in permissions
    
    # If checking for specific permission
    if required_permission:
        return required_permission in permissions
    
    # Just check if logged in
    return True

def require_auth(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'permissions' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def require_permission(permission=None):
    """Decorator factory to require specific permission"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'permissions' not in session:
                return redirect(url_for('login', next=request.url))
            
            # Check permission
            if permission == 'admin':
                if 'admin' not in session['permissions']:
                    return render_template_string(
                        '<html><body style="padding:24px;font-family:sans-serif;">'
                        '<h2>Access Denied</h2><p>Admin permission required.</p>'
                        '<a href="/">Go back</a></body></html>'
                    ), 403
            elif permission and permission not in session['permissions']:
                return render_template_string(
                    '<html><body style="padding:24px;font-family:sans-serif;">'
                    '<h2>Access Denied</h2><p>Insufficient permissions.</p>'
                    '<a href="/">Go back</a></body></html>'
                ), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def check_user_access(username: str):
    """Check if current user can access the given username"""
    if 'permissions' not in session:
        return False
    permissions = session['permissions']
    if 'admin' in permissions:
        return True
    return username in permissions

BASE_HTML = r"""
<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <title>TPU Web Monitor</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root{
      --bg:#0b1220; --panel:#0f1a33; --card:#101b36; --accent:#6da7ff;
      --txt:#d9e1ff; --muted:#9fb0d1; --ok:#20c997; --warn:#ffc107; --err:#ff6b6b; --info:#66d9ef;
      --border:#203055; --topbar:60px;
    }
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:var(--txt);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Ubuntu,"Helvetica Neue",Arial;}
    a{color:var(--accent);text-decoration:none}
    .topbar{position:sticky;top:0;z-index:50;background:linear-gradient(180deg,var(--panel),rgba(15,26,51,0.7));backdrop-filter:blur(6px);border-bottom:1px solid var(--border);}
    .container{max-width:1200px;margin:0 auto;padding:16px}
    .users{display:flex;flex-wrap:wrap;gap:8px;padding:12px 16px}
    .user-pill{padding:8px 12px;border:1px solid var(--border);border-radius:999px;background:#0c1834;color:var(--txt);opacity:.9}
    .user-pill.active{background:var(--accent);color:#05132b;border-color:transparent;font-weight:700}
    .toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:0 16px 16px 16px}
    .btn{border:1px solid var(--border);background:#0e1a37;color:var(--txt);padding:8px 12px;border-radius:8px;cursor:pointer}
    .btn:hover{filter:brightness(1.2)}
    .btn.ghost{background:transparent}
    .hint{color:var(--muted);font-size:12px;margin-left:6px}
    table{width:100%;border-collapse:separate;border-spacing:0;overflow:hidden;border-radius:12px;border:1px solid var(--border);}
    thead th{position:sticky;top:var(--topbar);background:#0c1630;color:#bcd0ff;text-align:left;padding:10px;border-bottom:1px solid var(--border);font-weight:600;z-index:30}
    tbody td{padding:10px;border-bottom:1px solid var(--border);vertical-align:top}
    tbody tr:hover{background:#0c1630}
    .mono{font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;}
    .status{display:inline-flex;align-items:center;gap:8px}
    .dot{width:10px;height:10px;border-radius:50%}
    .s-success .dot{background:var(--ok)}
    .s-warn .dot{background:var(--warn)}
    .s-error .dot{background:var(--err)}
    .s-info .dot{background:var(--info)}
    .s-unknown .dot{background:#6c757d}
    .s-success{color:var(--ok)} .s-warn{color:var(--warn)} .s-error{color:var(--err)} .s-info{color:var(--info)}
    /* TPU Status Colors */
    td.status-free{color:#20c997 !important} 
    td.status-reserved{color:#ffc107 !important} 
    td.status-running{color:#66d9ef !important} 
    td.status-notfound{color:#ff6b6b !important} 
    td.status-creating{color:#9fb0d1 !important}
    /* TPU Script Note Colors (like job panel) */
    td.script-ready{color:#20c997 !important} 
    td.script-notfound{color:#ff6b6b !important} 
    td.script-preempted{color:#ffc107 !important} 
    td.script-terminated{color:#9fb0d1 !important} 
    td.script-creating{color:#66d9ef !important}
    td.script-unknown{color:#6c757d !important}
    .actions{display:flex;gap:6px;flex-wrap:wrap}
    .input{background:#0d1a36;border:1px solid var(--border);color:var(--txt);padding:6px 8px;border-radius:6px;min-width:120px}
    .nowrap{white-space:nowrap}
    .tags-cell{max-width:280px;overflow-x:auto;white-space:nowrap;border:1px dashed var(--border);padding:6px;border-radius:6px;background:#0c1733}
    .footer{color:var(--muted);font-size:12px;margin-top:16px}
    kbd{padding:2px 6px;border:1px solid var(--border);border-bottom-width:2px;border-radius:4px;background:#0f1a33}
    /* Modal */
    .modal{position:fixed;inset:0;background:rgba(0,0,0,.45);display:none;align-items:center;justify-content:center;z-index:101}
    .modal.show{display:flex}
    .modal-card{position:relative;width:min(1000px,92vw);max-height:85vh;overflow:auto;background:#0f1a33;border:1px solid var(--border);border-radius:12px;padding:16px}
    .run-step table{max-height:300px;overflow-y:auto;display:block}
    .run-step thead{position:sticky;top:0;z-index:10}
    .run-step tbody{display:block}
    .run-step tr{display:table;width:100%;table-layout:fixed}
    .run-step td{display:table-cell}
    .run-step tr.selected{background:#1a2a4a !important;border:2px solid #6da7ff !important}
    .run-step tr:hover{background:#1a2a4a}
    .run-step tr.running{opacity:0.6 !important;background:#2a3a66 !important;color:#9fb0d1 !important}
    .run-step tr.running:hover{background:#3a4a76 !important}
    .run-controls{position:sticky;bottom:0;background:#0f1a33;border-top:1px solid var(--border);padding:12px 0;margin-top:16px;z-index:20}
    /* TPU Status Colors */
    .script-ready{color:#20c997 !important}
    .script-notfound{color:#ff6b6b !important}
    .script-preempted{color:#ffc107 !important}
    .script-terminated{color:#6c757d !important}
    .script-creating{color:#66d9ef !important}
    .script-unknown{color:#9fb0d1 !important}
    .status-free{color:#20c997 !important}
    .modal-close{position:absolute;right:10px;top:8px;width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;border:1px solid var(--border);background:#0e1a37;cursor:pointer}
    .modal-close:hover{filter:brightness(1.2)}
    .row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
    .spinner{width:14px;height:14px;border:2px solid #2a3a66;border-top-color:var(--accent);border-radius:50%;display:inline-block;animation:spin 1s linear infinite}
    @keyframes spin{to{transform:rotate(360deg)}}
    .pill{display:inline-flex;gap:6px;align-items:center;padding:2px 8px;border-radius:999px;border:1px solid var(--border);background:#0d1a36}
    .muted{color:var(--muted)}
  </style>
</head>
<body>
  <div class="topbar" id="topbar">
    <div class="container">
      <div class="users">
        {% for u in users %}
          <a class="user-pill {% if u==cur_user %}active{% endif %}" href="{{ url_for('user_page', username=u) }}">{{ u }}</a>
        {% endfor %}
        <a class="user-pill" href="{{ url_for('tpu_panel') }}">TPU 面板</a>
      </div>
    </div>
  </div>

  <div class="container">
    <div class="toolbar" id="toolbar">
      <button class="btn" onclick="doClean(false)">🧹 Clean</button>
      <button class="btn" onclick="doClean(true)">🧹 Clean -re</button>
      <button class="btn" onclick="openRunModal()">▶️ Run</button>
      <button class="btn ghost" onclick="refreshJobs()">🔄 刷新</button>
      <span class="hint">或按 <kbd>R</kbd> 刷新当前用户</span>
    </div>

    <table id="job-table">
      <tbody id="job-tbody">
        {% for j in rows %}
          <tr>
            <td class="mono nowrap">{{ j.window }}</td>
            <td class="mono">{{ j.dir }}</td>
            <td class="mono nowrap">{{ j.tpu }}</td>
            <td class="tags-cell">{{ j.tags }}</td>
            <td><span class="status s-{{ j.status_class }}"><span class="dot"></span>{{ j.status }}</span></td>
            <td class="actions">
              {% if j.in_data %}
                <button class="btn" onclick="doResume('{{ j.window }}')">Resume</button>
                <button class="btn" onclick="doRerun('{{ j.window }}')">Rerun</button>
                <button class="btn" onclick="openTpuPicker('{{ j.window }}','resume')">chooseTPU Resume</button>
                <button class="btn" onclick="openTpuPicker('{{ j.window }}','rerun')">chooseTPU Rerun</button>
                <button class="btn" onclick="doKill('{{ j.window }}')">Kill</button>
              {% else %}
                <span class="muted">（未在数据中）</span>
              {% endif %}
              <button class="btn" onclick="viewLog('{{ j.window }}')">查看 Log</button>
              <span id="opstate-{{ j.window }}" class="muted"></span>
            </td>
          </tr>
        {% endfor %}
      </tbody>
    </table>

    <!-- Pending Apply+Run Operations -->
    <div id="pending-operations" style="margin-top: 20px; display: none;">
      <h3 style="color: var(--txt); margin-bottom: 10px;">🔄 进行中的操作</h3>
      <table>
        <tbody id="pending-tbody"></tbody>
      </table>
    </div>

    <div class="footer">
      <p>状态解析与 CLI 的 <code>check_jobs_simp/_render_rows_for_job</code> 使用一致的表达式匹配；Msg 已隐藏。日志页支持去色/跟随/自动刷新。</p>
    </div>
  </div>

  <!-- TPU choose弹窗 -->
  <div class="modal" id="tpu-modal">
    <div class="modal-card" id="tpu-modal-card">
      <button class="modal-close" onclick="closePicker()" aria-label="关闭">✕</button>
      <div class="row" style="justify-content:space-between;margin-bottom:8px">
        <div class="row">
          <strong id="picker-title">Choose a TPU</strong>
        </div>
        <div class="row">
          <span id="picker-step" class="pill muted">步骤 1/2</span>
          <span id="picker-target" class="pill muted">无目标</span>
        </div>
      </div>
      
      <!-- Step 1: Select TPU Type -->
      <div id="picker-step1" class="run-step">
      <div class="row" style="margin-bottom:8px">
          <span class="muted">Choose a TPU type:</span>
      </div>
      <table>
          <tbody id="tpu-types-tbody"></tbody>
        </table>
      </div>
      
      <!-- Step 2: Select Specific TPU -->
      <div id="picker-step2" class="run-step" style="display:none">
        <div class="row" style="margin-bottom:8px">
          <span class="muted">Choose a TPU:</span>
        </div>
        <table>
        <tbody id="tpus-tbody"></tbody>
      </table>
      </div>
      
      <div class="run-controls" style="margin-top:10px;justify-content:flex-end">
        <button class="btn" id="picker-prev" onclick="pickerPrevStep()" style="display:none">← 上一步</button>
        <span id="picked-thing" class="pill muted">未choose</span>
        <button class="btn" id="picker-resume" onclick="submitPicker('resume')" style="display:none">Resume</button>
        <button class="btn" id="picker-rerun" onclick="submitPicker('rerun')" style="display:none">Rerun</button>
        <button class="btn" id="picker-apply-resume" onclick="submitPickerApply('resume')" style="display:none">Apply and Resume</button>
        <button class="btn" id="picker-apply-rerun" onclick="submitPickerApply('rerun')" style="display:none">Apply and Rerun</button>
        <button class="btn" id="picker-reapply-resume" onclick="submitPickerReapply('resume')" style="display:none">Reapply and Resume</button>
        <button class="btn" id="picker-reapply-rerun" onclick="submitPickerReapply('rerun')" style="display:none">Reapply and Rerun</button>
      </div>
    </div>
  </div>

  <!-- Run 弹窗 -->
  <div class="modal" id="run-modal">
    <div class="modal-card" id="run-modal-card">
      <button class="modal-close" onclick="closeRunModal()" aria-label="关闭">✕</button>
      <div class="row" style="justify-content:space-between;margin-bottom:8px">
        <div class="row">
          <strong>运行新任务</strong>
        </div>
        <div class="row">
          <span id="run-step" class="pill muted">步骤 1/3</span>
        </div>
      </div>
      
      <!-- Step 1: Select Directory -->
      <div id="run-step1" class="run-step">
        <div class="row" style="margin-bottom:8px">
          <span class="muted">choose目录：</span>
        </div>
        <table>
          <tbody id="dirs-tbody"></tbody>
        </table>
      </div>
      
      <!-- Step 2: Select TPU Type -->
      <div id="run-step2" class="run-step" style="display:none">
        <div class="row" style="margin-bottom:8px">
          <span class="muted">choose TPU type：</span>
        </div>
        <table>
          <tbody id="types-tbody"></tbody>
        </table>
      </div>
      
      <!-- Step 3: Select TPU -->
      <div id="run-step3" class="run-step" style="display:none">
        <div class="row" style="margin-bottom:8px">
          <span class="muted">choose TPU：</span>
        </div>
        <table>
          <tbody id="run-tpus-tbody"></tbody>
        </table>
      </div>
      
      <div class="run-controls">
        <div class="row" style="justify-content:space-between">
          <button class="btn" id="run-prev" onclick="runPrevStep()" style="display:none">← 上一步</button>
          <div class="row">
            <span id="run-selected" class="pill muted">未choose</span>
            <button class="btn" id="run-execute" onclick="executeRun()" style="display:none">▶️ RUN</button>
          </div>
        </div>
      </div>
    </div>
  </div>

<script>
const CUR_USER = {{ cur_user|tojson }};

function setTopbarVar(){
  const tb = document.getElementById('topbar');
  const h = (tb?.offsetHeight||60);
  document.documentElement.style.setProperty('--topbar', (h)+'px');
}
window.addEventListener('load', setTopbarVar);
window.addEventListener('resize', setTopbarVar);

// Modal functions for See Log functionality
function showOperationLogById(tid){
  // Open log modal with just the task ID
  openLogModal(tid, 'Operation');
}

function openLogModal(tid, alias){
  // Create modal if it doesn't exist
  let modal = document.getElementById('log-modal');
  if(!modal){
    modal = document.createElement('div');
    modal.id = 'log-modal';
    modal.className = 'modal';
    modal.innerHTML = `
      <div class="modal-content" style="max-width: 800px; max-height: 600px;">
        <div class="modal-header">
          <h3>Operation Log - ${alias}</h3>
          <button class="btn modal-close-btn" onclick="closeLogModal()" style="float: right;">✕ Close</button>
        </div>
        <div class="modal-body">
          <div id="log-content" style="background: #1a1a1a; color: #fff; padding: 10px; border-radius: 4px; font-family: monospace; white-space: pre-wrap; max-height: 400px; overflow-y: auto;"></div>
        </div>
        <div class="modal-footer">
          <button class="btn" onclick="refreshLog()">🔄 Refresh</button>
          <button class="btn" onclick="closeLogModal()" style="background: #6c757d; color: white; border: 1px solid #6c757d;">Close</button>
        </div>
      </div>
    `;
    document.body.appendChild(modal);
    
    // Add click outside to close functionality
    modal.addEventListener('click', function(e) {
      if (e.target === modal) {
        closeLogModal();
      }
    });
    
    // Add ESC key to close functionality
    const escHandler = function(e) {
      if (e.key === 'Escape' && modal.style.display === 'block') {
        closeLogModal();
      }
    };
    document.addEventListener('keydown', escHandler);
    
    // Store the handler for cleanup
    modal.escHandler = escHandler;
  }
  
  // Update modal content
  modal.querySelector('h3').textContent = `Operation Log - ${alias}`;
  modal.style.setProperty('display', 'flex', 'important');
  modal.classList.add('show');
  
  // Store current tid and alias for refresh
  modal.setAttribute('data-tid', tid);
  modal.setAttribute('data-alias', alias);
  
  // Load initial log content
  refreshLog();
}

function closeLogModal(){
  const modal = document.getElementById('log-modal');
  if(modal) {
    modal.style.display = 'none';
    modal.classList.remove('show');
    
    // Clean up event listeners
    if(modal.escHandler) {
      document.removeEventListener('keydown', modal.escHandler);
      modal.escHandler = null;
    }
  }
}

async function refreshLog(){
  const modal = document.getElementById('log-modal');
  if(!modal) return;
  
  const tid = modal.getAttribute('data-tid');
  const alias = modal.getAttribute('data-alias');
  if(!tid) return;
  
  try{
    const res = await fetch(`{{ url_for('op_status', tid='__ID__') }}`.replace('__ID__', tid));
    const data = await res.json();
    
    const logContent = document.getElementById('log-content');
    if(logContent){
      const timestamp = new Date().toLocaleTimeString();
      const logText = `[${timestamp}] Status: ${data.status}\nMessage: ${data.msg || 'No message'}\n\n--- Log Output ---\n${data.log || 'No log available yet...'}`;
      logContent.textContent = logText;
    }
  } catch(e){
    console.error('Failed to refresh log:', e);
  }
}

async function refreshJobs(){
  const res = await fetch(`{{ url_for('api_jobs', username=cur_user) }}`);
  const data = await res.json();
  const tbody = document.querySelector('#job-tbody');
  tbody.innerHTML = "";
  for (const j of data.jobs){
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="mono nowrap">${j.window}</td>
      <td class="mono">${j.dir||""}</td>
      <td class="mono nowrap">${j.tpu||""}</td>
      <td class="tags-cell">${j.tags||""}</td>
      <td><span class="status s-${j.status_class}"><span class="dot"></span>${j.status}</span></td>
      <td class="actions">
        ${ j.in_data ? `
          <button class="btn" onclick="doResume('${j.window}')">Resume</button>
          <button class="btn" onclick="doRerun('${j.window}')">Rerun</button>
          <button class="btn" onclick="openTpuPicker('${j.window}','resume')">Resume</button>
          <button class="btn" onclick="openTpuPicker('${j.window}','rerun')">Rerun</button>
          <button class="btn" onclick="doKill('${j.window}')">Kill</button>
        ` : `<span class="muted">（未在数据中）</span>` }
        <button class="btn" onclick="viewLog('${j.window}')">See Log</button>
        <span id="opstate-${j.window}" class="muted"></span>
      </td>
    `;
    tbody.appendChild(tr);
  }
  setTopbarVar();
  loadPendingOperations();
}

async function loadPendingOperations(){
  const res = await fetch(`{{ url_for('api_pending_operations', username=cur_user) }}`);
  const data = await res.json();
  const pendingDiv = document.getElementById('pending-operations');
  const tbody = document.getElementById('pending-tbody');
  
  if(data.operations && data.operations.length > 0){
    pendingDiv.style.display = 'block';
    tbody.innerHTML = '';
    
    for(const op of data.operations){
      const tr = document.createElement('tr');
      const statusClass = getOperationStatusClass(op.status);
      tr.innerHTML = `
        <td class="mono">${op.kind}</td>
        <td class="mono">${op.meta.dir || ''}</td>
        <td class="mono">${op.meta.tpu || ''}</td>
        <td class="${statusClass}">${op.msg || ''}</td>
        <td class="muted">${new Date(op.ts * 1000).toLocaleTimeString()}</td>
        <td>
          <span class="spinner"></span>
          <button class="btn" onclick="showOperationLogById('${op.tid}')" data-tid="${op.tid}" style="margin-left:8px; font-size:11px;">📋 See Log</button>
        </td>
      `;
      tbody.appendChild(tr);
    }
    
    // Set up auto-refresh for pending operations
    if(!window.pendingOpsInterval){
      window.pendingOpsInterval = setInterval(loadPendingOperations, 3000); // Refresh every 3 seconds
    }
  } else {
    pendingDiv.style.display = 'none';
    // Clear the interval if no pending operations
    if(window.pendingOpsInterval){
      clearInterval(window.pendingOpsInterval);
      window.pendingOpsInterval = null;
    }
  }
}

function getOperationStatusClass(status){
  switch(status){
    case 'applying': return 's-info';
    case 'mounting': return 's-warn';
    case 'pending': return 's-info';
    case 'ok': return 's-success';
    case 'err': return 's-error';
    default: return 's-unknown';
  }
}

document.addEventListener('keydown', (e)=>{
  if(e.key==='r' || e.key==='R'){ e.preventDefault(); refreshJobs(); }
  if(e.key==='Escape'){ closePicker(); }
});

// --------- 异步操作：提交后立即提示 + 轮询状态 ---------
function toast(msg){ alert(msg); }

async function pollOp(tid, where){
  const el = document.querySelector(where);
  if(el){ el.innerHTML = ` <span class="spinner"></span> 运行中...`; }
  while(true){
    const res = await fetch(`{{ url_for('op_status', tid='__ID__') }}`.replace('__ID__',tid));
    if(!res.ok){ toast('查询状态失败'); return; }
    const data = await res.json();
    if(data.status==='pending'){
      await new Promise(r=>setTimeout(r, 1500));
      continue;
    }
    if(el){ el.textContent = (data.status==='ok'?'✅ 成功':'❌ 失败'); }
    toast((data.status==='ok'?'✅ 成功':'❌ 失败') + '\\n' + (data.msg||'')); 
    refreshJobs();
    break;
  }
}

async function doResume(win, tpu=null){
  toast('已提交 Resume 操作');
  const res = await fetch(`{{ url_for('api_resume_async', username=cur_user) }}`,{
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({window_id: win, tpu})
  });
  const data = await res.json();
  if(!data.ok){ toast('❌ 提交失败\\n'+(data.msg||'')); return; }
  pollOp(data.tid, '#opstate-'+win);
}
async function doRerun(win, tpu=null){
  toast('已提交 Rerun 操作');
  const res = await fetch(`{{ url_for('api_rerun_async', username=cur_user) }}`,{
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({window_id: win, tpu})
  });
  const data = await res.json();
  if(!data.ok){ toast('❌ 提交失败\\n'+(data.msg||'')); return; }
  pollOp(data.tid, '#opstate-'+win);
}
async function doKill(win){
  toast('已提交 Kill 操作');
  const res = await fetch(`{{ url_for('api_kill_async', username=cur_user) }}`,{
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({window_id: win})
  });
  const data = await res.json();
  if(!data.ok){ toast('❌ 提交失败\\n'+(data.msg||'')); return; }
  pollOp(data.tid, '#opstate-'+win);
}

async function doClean(reFlag){
  toast('已提交 Clean 操作');
  const res = await fetch(`{{ url_for('api_clean', username=cur_user) }}?re=${reFlag?1:0}`,{method:'POST'});
  const data = await res.json();
  toast((data.ok?'✅ 成功':'❌ 失败') + '\\n' + (data.msg||''));
  refreshJobs();
}

function viewLog(win){ window.open(`{{ url_for('view_log', username=cur_user, window_id='__W__') }}`.replace('__W__',win),'_blank'); }

// ---------- TPU choose器（多步骤choose） ----------
let pickerFor = null; // {win, action}
let pickedTpu = null;
let pickerState = {
  step: 1,
  selectedType: null,
  selectedTpu: null
};

function openTpuPicker(win, action){
  pickerFor = {win, action};
  pickedTpu = null;
  pickerState = { step: 1, selectedType: null, selectedTpu: null };
  document.getElementById('picker-target').textContent = `目标: #${win} / ${action}`;
  document.getElementById('picked-thing').textContent = '未choose';
  document.getElementById('tpu-modal').classList.add('show');
  loadPickerStep1();
}
function closePicker(){ document.getElementById('tpu-modal').classList.remove('show'); }
document.getElementById('tpu-modal').addEventListener('click', (e)=>{ if(e.target && e.target.id==='tpu-modal') closePicker(); });

// Multi-step picker functions
function loadPickerStep1(){
  // Load TPU types
  fetch(`{{ url_for('api_tpu_types') }}`)
    .then(r => r.json())
    .then(data => {
      const tbody = document.getElementById('tpu-types-tbody');
      tbody.innerHTML = '';
      if(data.types && data.types.length > 0){
        data.types.forEach(type => {
          const tr = document.createElement('tr');
          tr.style.cursor = 'pointer';
          tr.onclick = () => selectPickerType(type);
          tr.innerHTML = `
            <td class="mono">${type}</td>
          `;
          tbody.appendChild(tr);
        });
      }
      updatePickerStep();
    });
}

function loadPickerStep2(){
  // Load specific TPUs based on selected type
  fetch(`{{ url_for('api_tpus_by_type', tpu_type='__TYPE__') }}`.replace('__TYPE__', pickerState.selectedType))
    .then(r => r.json())
    .then(data => {
      const tbody = document.getElementById('tpus-tbody');
      tbody.innerHTML = '';
      
      if(data.tpus && data.tpus.length > 0){
        data.tpus.forEach(tpu => {
          const tr = document.createElement('tr');
          tr.style.cursor = 'pointer';
          tr.onclick = () => selectPickerTpu(tpu);
          
          // Check if TPU should be marked as grey (running/reserved by other users)
          const isGrey = tpu.user && 
                        tpu.user.trim() !== '' && 
                        tpu.user.toLowerCase() !== 'free' && 
                        tpu.user !== '{{ cur_user }}';
          if(isGrey) {
            tr.classList.add('running');
          }
          
          const statusClass = getTpuStatusClass(tpu.script_note);
          tr.innerHTML = `
            <td class="mono">${tpu.alias}</td>
            <td class="${statusClass}">${tpu.script_note || ''}</td>
            <td class="mono">${tpu.user || ''}</td>
            <td class="mono">${tpu.zone || ''}</td>
          `;
          tbody.appendChild(tr);
        });
      }
      updatePickerStep();
    });
}

function selectPickerType(type){
  pickerState.selectedType = type;
  pickerState.step = 2;
  loadPickerStep2();
}

function selectPickerTpu(tpu){
  pickerState.selectedTpu = tpu;
  pickedTpu = tpu;
  document.getElementById('picked-thing').textContent = `已选: ${tpu.alias}`;
  updatePickerStep();
  updatePickerButtons(tpu);
}

function pickerPrevStep(){
  if(pickerState.step > 1) {
    pickerState.step--;
    if(pickerState.step === 1) {
      pickerState.selectedType = null;
      loadPickerStep1();
    }
  }
}

function updatePickerStep(){
  // Hide all steps
  document.getElementById('picker-step1').style.display = 'none';
  document.getElementById('picker-step2').style.display = 'none';
  
  // Show current step
  document.getElementById(`picker-step${pickerState.step}`).style.display = 'block';
  
  // Update step indicator
  document.getElementById('picker-step').textContent = `步骤 ${pickerState.step}/2`;
  
  // Update buttons
  const prevBtn = document.getElementById('picker-prev');
  prevBtn.style.display = pickerState.step > 1 ? 'block' : 'none';
  
  // Hide all action buttons initially
  const allButtons = ['picker-resume', 'picker-rerun', 'picker-apply-resume', 'picker-apply-rerun', 'picker-reapply-resume', 'picker-reapply-rerun'];
  allButtons.forEach(id => {
    document.getElementById(id).style.display = 'none';
  });
}

function updatePickerButtons(tpu){
  if(!tpu) return;
  
  const status = (tpu.script_note || '').toUpperCase();
  const resumeBtn = document.getElementById('picker-resume');
  const rerunBtn = document.getElementById('picker-rerun');
  const applyResumeBtn = document.getElementById('picker-apply-resume');
  const applyRerunBtn = document.getElementById('picker-apply-rerun');
  const reapplyResumeBtn = document.getElementById('picker-reapply-resume');
  const reapplyRerunBtn = document.getElementById('picker-reapply-rerun');
  
  // Hide all buttons first
  [resumeBtn, rerunBtn, applyResumeBtn, applyRerunBtn, reapplyResumeBtn, reapplyRerunBtn].forEach(btn => {
    btn.style.display = 'none';
  });
  
  // Show buttons based on status
  if(status === 'NOT FOUND') {
    // Only apply buttons for NOT FOUND
    applyResumeBtn.style.display = 'block';
    applyRerunBtn.style.display = 'block';
  } else if(status === 'READY') {
    // Only direct buttons for READY
    resumeBtn.style.display = 'block';
    rerunBtn.style.display = 'block';
  } else if(status === 'PREEMPTED') {
    // Only reapply buttons for PREEMPTED
    reapplyResumeBtn.style.display = 'block';
    reapplyRerunBtn.style.display = 'block';
  } else {
    // Default: show both direct and apply buttons
    resumeBtn.style.display = 'block';
    rerunBtn.style.display = 'block';
    applyResumeBtn.style.display = 'block';
    applyRerunBtn.style.display = 'block';
  }
}

function getTpuStatusClass(scriptNote) {
  if (!scriptNote) return 'status-free';
  const status = scriptNote.toUpperCase();
  if (status === 'READY') return 'script-ready';
  if (status === 'NOT FOUND') return 'script-notfound';
  if (status === 'PREEMPTED') return 'script-preempted';
  if (status === 'TERMINATED') return 'script-terminated';
  if (status === 'CREATING') return 'script-creating';
  return 'script-unknown';
}

async function loadTpus(){
  const res = await fetch(`{{ url_for('api_list_tpus') }}`);
  const data = await res.json();
  const tbody = document.getElementById('tpus-tbody');
  tbody.innerHTML = "";
  for(const r of data.rows){
    const tr = document.createElement('tr');
    tr.style.cursor='pointer';
    tr.onclick = ()=>{ pickedTpu = r; document.getElementById('picked-thing').textContent = '已选：'+r.alias; };
    tr.innerHTML = `
      <td class="mono">${r.alias}</td>
      <td>${r.running_status || ''}</td>
      <td class="mono">${r.zone || ''}</td>
      <td class="mono">${(String(r.preemptible).toLowerCase()==='true' || r.preemptible==='1') ? 'preemptible' : 'normal'}</td>
      <td class="mono">${r.user||''}</td>
      <td style="max-width:320px;white-space:nowrap;overflow-x:auto">${r.user_note||''}</td>
    `;
    tbody.appendChild(tr);
  }
  const input = document.getElementById('tpu-search');
  input.oninput = ()=>{
    const q = input.value.toLowerCase();
    for(const tr of tbody.querySelectorAll('tr')){
      const s = tr.textContent.toLowerCase();
      tr.style.display = s.includes(q) ? '' : 'none';
    }
  };
}

function submitPicker(what){
  if(!pickerFor || !pickedTpu){ toast('choose TPU'); return; }
  const alias = pickedTpu.alias || pickedTpu;
  if(what==='resume') doResume(pickerFor.win, alias);
  else doRerun(pickerFor.win, alias);
  closePicker();
}
function submitPickerApply(what){
  if(!pickerFor || !pickedTpu){ toast('choose TPU'); return; }
  const alias = pickedTpu.alias || pickedTpu;
  const pre = pickedTpu.preemptible;
  toast('已提交 申请并'+(what==='resume'?'Resume':'Rerun')+' 操作');
  const url = what==='resume' ? `{{ url_for('api_apply_resume_async', username=cur_user) }}` : `{{ url_for('api_apply_rerun_async', username=cur_user) }}`;
  fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({window_id: pickerFor.win, alias, pre})})
    .then(r=>r.json()).then(data=>{
      if(!data.ok){ toast('❌ 提交失败\\n'+(data.msg||'')); return; }
      // Close the picker modal
      closePicker();
      // Show progress bar
      pollOp(data.tid, '#opstate-'+pickerFor.win);
    });
}

function submitPickerReapply(what){
  if(!pickerFor || !pickedTpu){ toast('choose TPU'); return; }
  const alias = pickedTpu.alias || pickedTpu;
  const pre = pickedTpu.preemptible;
  toast('已提交 重新申请并'+(what==='resume'?'Resume':'Rerun')+' 操作');
  const url = what==='resume' ? `{{ url_for('api_reapply_resume_async', username=cur_user) }}` : `{{ url_for('api_reapply_rerun_async', username=cur_user) }}`;
  fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({window_id: pickerFor.win, alias, pre})})
    .then(r=>r.json()).then(data=>{
      if(!data.ok){ toast('❌ 提交失败\\n'+(data.msg||'')); return; }
      // Close the picker modal
      closePicker();
      // Show progress bar
      pollOp(data.tid, '#opstate-'+pickerFor.win);
    });
}

// ---------- Run Modal Functions ----------
let runState = {
  step: 1,
  selectedDir: null,
  selectedType: null,
  selectedTpu: null
};

function openRunModal(){
  runState = { step: 1, selectedDir: null, selectedType: null, selectedTpu: null };
  document.getElementById('run-modal').classList.add('show');
  loadRunStep1();
}

function closeRunModal(){
  document.getElementById('run-modal').classList.remove('show');
}

function loadRunStep1(){
  // Load directories for current user
  fetch(`{{ url_for('api_user_dirs', username=cur_user) }}`)
    .then(r => r.json())
    .then(data => {
      const tbody = document.getElementById('dirs-tbody');
      tbody.innerHTML = '';
      if(data.dirs && data.dirs.length > 0){
        data.dirs.forEach((dir, index) => {
          const tr = document.createElement('tr');
          tr.style.cursor = 'pointer';
          tr.onclick = () => selectDir(dir, index);
          tr.innerHTML = `
            <td class="mono">${dir.num}</td>
            <td class="mono">${dir.path}</td>
          `;
          tbody.appendChild(tr);
        });
      } else {
        tbody.innerHTML = '<tr><td colspan="2" class="muted">没有找到目录</td></tr>';
      }
    })
    .catch(e => {
      console.error('Failed to load directories:', e);
      document.getElementById('dirs-tbody').innerHTML = '<tr><td colspan="2" class="muted">加载失败</td></tr>';
    });
  
  updateRunStep();
}

function selectDir(dir, index){
  runState.selectedDir = { 
    dir: dir.path, 
    num: dir.num, 
    index 
  };
  
  // Remove previous selection highlighting
  const tbody = document.getElementById('dirs-tbody');
  tbody.querySelectorAll('tr').forEach(row => {
    row.classList.remove('selected');
  });
  
  // Add highlighting to selected row
  event.target.closest('tr').classList.add('selected');
  
  runState.step = 2;
  loadRunStep2();
}

function loadRunStep2(){
  // Load TPU types
  fetch(`{{ url_for('api_tpu_types') }}`)
    .then(r => r.json())
    .then(data => {
      const tbody = document.getElementById('types-tbody');
      tbody.innerHTML = '';
      if(data.types && data.types.length > 0){
        data.types.forEach(type => {
          const tr = document.createElement('tr');
          tr.style.cursor = 'pointer';
          tr.onclick = () => selectType(type);
          tr.innerHTML = `
            <td class="mono">${type}</td>
          `;
          tbody.appendChild(tr);
        });
      } else {
        tbody.innerHTML = '<tr><td class="muted">没有找到 TPU type</td></tr>';
      }
    })
    .catch(e => {
      console.error('Failed to load TPU types:', e);
      document.getElementById('types-tbody').innerHTML = '<tr><td class="muted">加载失败</td></tr>';
    });
  
  updateRunStep();
}

function selectType(type){
  runState.selectedType = type;
  
  // Remove previous selection highlighting
  const tbody = document.getElementById('types-tbody');
  tbody.querySelectorAll('tr').forEach(row => {
    row.classList.remove('selected');
  });
  
  // Add highlighting to selected row
  event.target.closest('tr').classList.add('selected');
  
  runState.step = 3;
  loadRunStep3();
}

function loadRunStep3(){
  // Load TPUs of selected type, including running TPUs but marking them differently
  fetch(`{{ url_for('api_tpus_by_type', tpu_type='__TYPE__') }}`.replace('__TYPE__', runState.selectedType))
    .then(r => r.json())
    .then(data => {
      const tbody = document.getElementById('run-tpus-tbody');
      tbody.innerHTML = '';
      if(data.tpus && data.tpus.length > 0){
        data.tpus.forEach(tpu => {
          const tr = document.createElement('tr');
          tr.style.cursor = 'pointer';
          tr.onclick = () => selectTpu(tpu);
          
          // Check if TPU should be marked as grey (running/reserved by other users)
          // Grey if: has a user that is not 'free' and not the current user
          const isGrey = tpu.user && 
                        tpu.user.trim() !== '' && 
                        tpu.user.toLowerCase() !== 'free' && 
                        tpu.user !== '{{ cur_user }}';
          if(isGrey) {
            tr.classList.add('running');
          }
          
          const statusClass = getTpuStatusClass(tpu.script_note);
          tr.innerHTML = `
            <td class="mono">${tpu.alias}</td>
            <td class="${statusClass}">${tpu.script_note || ''}</td>
            <td class="mono">${tpu.zone || ''}</td>
            <td class="mono">${tpu.user || ''}</td>
            <td>
              ${tpu.script_note === 'NOT FOUND' ? 
                `<button class="btn" onclick="event.stopPropagation(); applyAndRun('${tpu.alias}')">申请并运行</button>` : 
                '<span class="muted">—</span>'
              }
            </td>
          `;
          tbody.appendChild(tr);
        });
      } else {
        tbody.innerHTML = '<tr><td colspan="5" class="muted">没有找到可用的 TPU</td></tr>';
      }
    })
    .catch(e => {
      console.error('Failed to load TPUs:', e);
      document.getElementById('run-tpus-tbody').innerHTML = '<tr><td colspan="5" class="muted">加载失败</td></tr>';
    });
  
  updateRunStep();
}

function selectTpu(tpu){
  // Check if TPU should show confirmation (running/reserved by other users)
  const needsConfirmation = tpu.user && 
                           tpu.user.trim() !== '' && 
                           tpu.user.toLowerCase() !== 'free' && 
                           tpu.user !== '{{ cur_user }}';
  
  if(needsConfirmation) {
    const confirmed = confirm(`警告: TPU ${tpu.alias} 正在被用户 ${tpu.user} 使用。\n\n确定要继续使用这个 TPU 吗？\n\n点击"确定"继续，点击"取消"choose其他 TPU。`);
    if(!confirmed) {
      return; // User cancelled, don't select this TPU
    }
  }
  
  runState.selectedTpu = tpu;
  
  // Remove previous selection highlighting
  const tbody = document.getElementById('run-tpus-tbody');
  tbody.querySelectorAll('tr').forEach(row => {
    row.classList.remove('selected');
  });
  
  // Add highlighting to selected row
  event.target.closest('tr').classList.add('selected');
  
  updateRunStep();
}

function getTpuStatusClass(scriptNote){
  if(!scriptNote) return '';
  const note = scriptNote.toUpperCase();
  if(note === 'READY') return 'script-ready';
  if(note === 'NOT FOUND') return 'script-notfound';
  if(note === 'PREEMPTED') return 'script-preempted';
  if(note === 'TERMINATED') return 'script-terminated';
  if(note === 'CREATING') return 'script-creating';
  return 'script-unknown';
}

function applyAndRun(alias){
  if(!runState.selectedDir) return;
  toast('已提交 申请并运行 操作');
  fetch(`{{ url_for('api_apply_run_async', username=cur_user) }}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      dir: `dir=${runState.selectedDir.num}`,
      tpu: alias
    })
  })
  .then(r => r.json())
  .then(data => {
    if(!data.ok){ toast('❌ 提交失败\\n'+(data.msg||'')); return; }
    toast('✅ 申请并运行操作已提交');
    closeRunModal();
    refreshJobs();
  });
}

function runPrevStep(){
  if(runState.step > 1){
    runState.step--;
    updateRunStep();
  }
}

function updateRunStep(){
  // Update step indicator
  document.getElementById('run-step').textContent = `步骤 ${runState.step}/3`;
  
  // Show/hide steps
  for(let i = 1; i <= 3; i++){
    const stepEl = document.getElementById(`run-step${i}`);
    stepEl.style.display = i === runState.step ? 'block' : 'none';
  }
  
  // Show/hide prev button
  document.getElementById('run-prev').style.display = runState.step > 1 ? 'block' : 'none';
  
  // Update selected info and execute button
  const selectedEl = document.getElementById('run-selected');
  const executeEl = document.getElementById('run-execute');
  
  if(runState.step === 3 && runState.selectedTpu){
    selectedEl.textContent = `已choose: ${runState.selectedDir.dir} → ${runState.selectedType} → ${runState.selectedTpu.alias}`;
    executeEl.style.display = 'block';
  } else {
    selectedEl.textContent = '未choose';
    executeEl.style.display = 'none';
  }
}

function executeRun(){
  if(!runState.selectedDir || !runState.selectedTpu) return;
  
  toast('已提交运行操作');
  fetch(`{{ url_for('api_run_async', username=cur_user) }}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      dir: `dir=${runState.selectedDir.num}`,
      tpu: runState.selectedTpu.alias
    })
  })
  .then(r => r.json())
  .then(data => {
    if(!data.ok){ toast('❌ 提交失败\\n'+(data.msg||'')); return; }
    toast('✅ 运行操作已提交');
    closeRunModal();
    refreshJobs();
  });
}

// Close modal when clicking outside
document.getElementById('run-modal').addEventListener('click', (e) => {
  if(e.target && e.target.id === 'run-modal') closeRunModal();
});
</script>

</body>
</html>
"""

TPU_PANEL_HTML = r"""
<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <title>TPU 面板</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root{ --bg:#0b1220; --panel:#0f1a33; --txt:#d9e1ff; --border:#203055; --accent:#6da7ff; --topbar:60px; }
    body{margin:0;background:var(--bg);color:var(--txt);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Ubuntu,"Helvetica Neue",Arial;}
    .container{max-width:1200px;margin:0 auto;padding:16px}
    .row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
    table{width:100%;border-collapse:separate;border-spacing:0;border:1px solid var(--border);border-radius:12px;overflow:hidden}
    thead th{position:sticky;top:var(--topbar);background:#0c1630;border-bottom:1px solid var(--border);padding:8px;text-align:left;z-index:30}
    tbody td{border-bottom:1px solid var(--border);padding:8px;vertical-align:top}
    .btn{border:1px solid var(--border);background:#0e1a37;color:var(--txt);padding:6px 10px;border-radius:8px;cursor:pointer}
    .btn:hover{filter:brightness(1.2)}
    .input{background:#0d1a36;border:1px solid var(--border);color:#d9e1ff;padding:6px 8px;border-radius:6px;min-width:120px}
    .mono{font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;}
    .muted{color:#9fb0d1}
    .spinner{width:14px;height:14px;border:2px solid #2a3a66;border-top-color:var(--accent);border-radius:50%;display:inline-block;animation:spin 1s linear infinite}
    @keyframes spin{to{transform:rotate(360deg)}}
    /* TPU Script Note Colors */
    td.script-ready{color:#20c997 !important} 
    td.script-notfound{color:#ff6b6b !important} 
    td.script-preempted{color:#ffc107 !important} 
    td.script-terminated{color:#9fb0d1 !important} 
    td.script-creating{color:#66d9ef !important}
    td.script-unknown{color:#6c757d !important}
    /* TPU Running Status Colors */
    td.status-running{color:#66d9ef !important}
    td.status-free{color:#20c997 !important}
    td.status-reserved{color:#ffc107 !important}
    td.status-notfound{color:#ff6b6b !important}
    td.status-creating{color:#66d9ef !important}
    .progress-bar{width:100%;height:8px;background:#2a3a66;border-radius:4px;overflow:hidden;margin:4px 0}
    .progress-fill{height:100%;background:linear-gradient(90deg,#20c997,#66d9ef);transition:width 0.3s ease}
    .progress-text{font-size:11px;color:var(--muted);margin-top:2px}
    /* Log Modal */
    .modal{position:fixed;inset:0;background:rgba(0,0,0,.7);display:none;align-items:center;justify-content:center;z-index:101}
    .modal.show{display:flex}
    .modal-content{position:relative;width:min(800px,92vw);max-height:85vh;overflow:auto;background:#0f1a33;border:2px solid #6da7ff;border-radius:12px;padding:16px;box-shadow:0 10px 30px rgba(0,0,0,0.5)}
    .modal-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;border-bottom:1px solid var(--border);padding-bottom:8px}
    .modal-body{margin-bottom:16px}
    .modal-footer{display:flex;gap:8px;justify-content:flex-end;border-top:1px solid var(--border);padding-top:8px}
    .modal-close-btn{background:#dc3545 !important;color:white !important;border:1px solid #dc3545 !important;padding:4px 8px !important;font-size:14px !important;cursor:pointer !important}
    .modal-close-btn:hover{background:#c82333 !important;border-color:#bd2130 !important}
    .topbar{position:sticky;top:0;z-index:50;background:linear-gradient(180deg,var(--panel),rgba(15,26,51,0.7));border-bottom:1px solid var(--border);}
    .users{display:flex;gap:8px;padding:12px 16px}
    .user-pill{padding:8px 12px;border:1px solid var(--border);border-radius:999px;background:#0c1834;color:#d9e1ff;opacity:.9}
    .filter-tags{display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap;align-items:center}
    .filter-tag{display:inline-flex;align-items:center;gap:6px;padding:6px 12px;border:1px solid var(--border);border-radius:8px;background:#0e1a37;color:var(--txt)}
    .filter-tag-close{cursor:pointer;padding:2px 4px;border-radius:4px;font-weight:bold}
    .filter-tag-close:hover{background:rgba(255,255,255,0.1)}
    .filter-modal{position:fixed;inset:0;background:rgba(0,0,0,.7);display:none;align-items:center;justify-content:center;z-index:101}
    .filter-modal.show{display:flex}
    .filter-modal-content{background:#0f1a33;border:1px solid var(--border);border-radius:12px;padding:20px;min-width:300px}
    .filter-modal-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px}
    .filter-modal-body{display:flex;flex-direction:column;gap:12px}
    .filter-checkbox{display:flex;align-items:center;gap:8px;cursor:pointer}
    .filter-checkbox input[type="checkbox"]{width:18px;height:18px;cursor:pointer}
    .filter-modal-footer{display:flex;justify-content:flex-end;gap:8px;margin-top:16px;padding-top:16px;border-top:1px solid var(--border)}
  </style>
</head>
<body>
  <div class="topbar" id="topbar2">
    <div class="container">
      <div class="users">
        <a class="user-pill" href="{{ url_for('index') }}">← 返回用户页</a>
        <strong style="margin-left:8px">TPU 面板（以 spreadsheet 为准）</strong>
      </div>
    </div>
  </div>

  <div class="container">
    <div class="filter-tags" id="filter-tags" style="display:none"></div>
    
    <!-- Statistics Display -->
    <div id="statistics-container" style="margin-bottom:16px;background:#0f1a33;border:1px solid var(--border);border-radius:8px;overflow:hidden">
      <div style="display:flex;justify-content:space-between;align-items:center;padding:12px;cursor:pointer;user-select:none;" onclick="toggleStatistics()">
        <strong>统计信息</strong>
        <span id="statistics-toggle-icon" style="font-size:18px;">▼</span>
      </div>
      <div id="statistics-display" style="padding:0 12px 12px 12px;display:block;"></div>
    </div>
    
    <div class="row" style="justify-content:space-between">
      <div class="row">
        <button class="btn" onclick="openFilterModal('type')">type</button>
        <button class="btn" onclick="openFilterModal('zone')">zone</button>
        <button class="btn" onclick="openFilterModal('status')">status</button>
        <button class="btn" onclick="openFilterModal('user')">user</button>
        <button class="btn" onclick="openFilterModal('running_status')">running status</button>
        <button class="btn" onclick="loadPanel(true)">refresh</button>
        <button class="btn" onclick="viewAllTPUStatus()" style="background:var(--accent);color:#05132b">查看所有TPU状态</button>
      </div>
    </div>

    <div class="row muted">Apply Default: 20 times / 5 seconds.</div>

    <!-- Filter Modal -->
    <div class="filter-modal" id="filter-modal">
      <div class="filter-modal-content">
        <div class="filter-modal-header">
          <strong id="filter-modal-title">choose TPU type</strong>
          <button class="btn" onclick="closeFilterModal()" style="padding:4px 8px">✕</button>
        </div>
        <div class="filter-modal-body" id="filter-modal-body">
          <!-- Content will be dynamically generated -->
        </div>
        <div class="filter-modal-footer">
          <button class="btn" onclick="closeFilterModal()">取消</button>
          <button class="btn" onclick="applyFilter()" style="background:var(--accent);color:#05132b">确认</button>
        </div>
      </div>
    </div>

    <!-- TPU Status Modal -->
    <div class="modal" id="tpu-status-modal">
      <div class="modal-content" style="max-width:1200px;width:95vw;">
        <div class="modal-header">
          <strong>所有 TPU 状态</strong>
          <button class="btn modal-close-btn" onclick="closeTPUStatusModal()">✕</button>
        </div>
        <div class="modal-body">
          <div id="tpu-status-loading" style="text-align:center;padding:20px;">
            <div class="spinner" style="width:32px;height:32px;border-width:3px;"></div>
            <div style="margin-top:12px;color:var(--muted);">加载中...</div>
          </div>
          <div id="tpu-status-content" style="display:none;">
            <div style="margin-bottom:16px;display:flex;gap:8px;align-items:center;">
              <input type="text" id="tpu-status-search" class="input" placeholder="搜索 TPU 名称、区域、用户..." style="flex:1;min-width:0;">
              <button class="btn" onclick="filterTPUStatus()">搜索</button>
              <button class="btn" onclick="clearTPUStatusFilter()">清除</button>
            </div>
            <div style="max-height:60vh;overflow-y:auto;border:1px solid var(--border);border-radius:8px;">
              <table style="margin:0;">
                <tbody id="tpu-status-table-body">
                </tbody>
              </table>
            </div>
            <div id="tpu-status-summary" style="margin-top:12px;padding:12px;background:#0e1a37;border:1px solid var(--border);border-radius:8px;">
            </div>
          </div>
          <div id="tpu-status-error" style="display:none;color:#ff6b6b;padding:20px;text-align:center;">
          </div>
        </div>
        <div class="modal-footer">
          <button class="btn" onclick="closeTPUStatusModal()">关闭</button>
        </div>
      </div>
    </div>

    <!-- Steal TPU (Keng) Modal -->
    <div class="modal" id="steal-modal">
      <div class="modal-content" style="max-width:800px;">
        <div class="modal-header">
          <strong>🥷 偷 TPU - 选择坑位</strong>
          <button class="btn modal-close-btn" onclick="closeStealModal()">✕</button>
        </div>
        <div class="modal-body">
          <div id="steal-info" style="margin-bottom:16px;padding:12px;background:#0e1a37;border:1px solid var(--border);border-radius:8px;">
            <div><strong>目标 TPU:</strong> <span id="steal-tpu-name" class="mono"></span></div>
            <div style="margin-top:4px;"><strong>区域:</strong> <span id="steal-tpu-zone" class="mono"></span></div>
          </div>
          <div id="steal-loading" style="text-align:center;padding:20px;">
            <div class="spinner" style="width:24px;height:24px;border-width:3px;"></div>
            <div style="margin-top:8px;color:var(--muted);">加载坑位中...</div>
          </div>
          <div id="steal-content" style="display:none;">
            <div style="margin-bottom:12px;color:var(--muted);">
              找到 <strong id="keng-count">0</strong> 个可用坑位，点击坑位按钮将目标 TPU 放入该坑位：
            </div>
            <div id="kengs-list" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:8px;max-height:50vh;overflow-y:auto;">
            </div>
          </div>
          <div id="steal-error" style="display:none;color:#ff6b6b;padding:20px;text-align:center;">
          </div>
        </div>
        <div class="modal-footer">
          <button class="btn" onclick="closeStealModal()">取消</button>
        </div>
      </div>
    </div>

    <table>
      <tbody id="panel-body"></tbody>
    </table>
  </div>

<script>
// Filter state: store selected filters (empty = no filter)
let selectedVersions = new Set();
let selectedZones = new Set();
let selectedStatuses = new Set();
let selectedUsers = new Set();
let selectedRunningStatuses = new Set();
let currentFilterType = 'type'; // 'type', 'zone', 'status', 'user', or 'running_status'

// Throttle mechanism for spreadsheet reads (30 seconds)
let lastSheetReadTime = 0;
let cachedPanelData = null;
const SHEET_READ_INTERVAL = 30000; // 30 seconds in milliseconds
let pendingRefresh = false; // Flag to indicate if refresh is needed after operation

// Available zones (from constants.py)
const availableZones = [
  'us-central1-a', 'us-central1-b', 'us-central2-b', 
  'us-east1-d', 'us-east5-b', 'asia-northeast1-b'
];

// Available statuses
const availableStatuses = ['READY', 'PREEMPTED', 'CREATING', 'NOT FOUND', 'TERMINATED'];
// Available running statuses
const availableRunningStatuses = ['free', 'reserved', 'running'];

function openFilterModal(type){
  currentFilterType = type;
  const modal = document.getElementById('filter-modal');
  const title = document.getElementById('filter-modal-title');
  const body = document.getElementById('filter-modal-body');
  
  // Set title
  if(type === 'type'){
    title.textContent = 'choose TPU type';
  } else if(type === 'zone'){
    title.textContent = 'choose TPU 地区';
  } else if(type === 'status'){
    title.textContent = 'choose TPU 状态';
  } else if(type === 'user'){
    title.textContent = 'choose TPU user';
  } else if(type === 'running_status'){
    title.textContent = 'choose running status';
  }
  
  // Generate checkboxes based on type
  body.innerHTML = '';
  
  if(type === 'user'){
    // For user, use cached data if available, otherwise fetch
    const loadUserOptions = (data) => {
      const userSet = new Set();
      data.rows.forEach(r => {
        const user = (r.user || '').trim();
        if(user && user.toLowerCase() !== 'free'){
          userSet.add(user);
        }
      });
      const options = Array.from(userSet).sort();
      const selectedSet = selectedUsers;
      
      options.forEach(opt => {
        const label = document.createElement('label');
        label.className = 'filter-checkbox';
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.value = opt;
        checkbox.id = `check-${type}-${opt.replace(/\s+/g, '-')}`;
        checkbox.checked = selectedSet.has(opt);
        label.appendChild(checkbox);
        const span = document.createElement('span');
        span.textContent = opt;
        label.appendChild(span);
        body.appendChild(label);
      });
    };
    
    // Use cached data if available and recent, otherwise fetch
    if(cachedPanelData && (Date.now() - lastSheetReadTime < SHEET_READ_INTERVAL)) {
      loadUserOptions(cachedPanelData);
    } else {
      fetch(`{{ url_for('api_list_tpus') }}`)
        .then(r => r.json())
        .then(data => {
          cachedPanelData = data;
          lastSheetReadTime = Date.now();
          loadUserOptions(data);
        });
    }
  } else {
    let options = [];
    let selectedSet = new Set();
    
    if(type === 'type'){
      options = ['v4', 'v5', 'v6'];
      selectedSet = selectedVersions;
    } else if(type === 'zone'){
      options = availableZones;
      selectedSet = selectedZones;
    } else if(type === 'status'){
      options = availableStatuses;
      selectedSet = selectedStatuses;
    } else if(type === 'running_status'){
      options = availableRunningStatuses;
      selectedSet = selectedRunningStatuses;
    }
    
    options.forEach(opt => {
      const label = document.createElement('label');
      label.className = 'filter-checkbox';
      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.value = opt;
      checkbox.id = `check-${type}-${opt}`;
      checkbox.checked = selectedSet.has(opt);
      label.appendChild(checkbox);
      const span = document.createElement('span');
      span.textContent = opt;
      label.appendChild(span);
      body.appendChild(label);
    });
  }
  
  modal.classList.add('show');
}

function closeFilterModal(){
  document.getElementById('filter-modal').classList.remove('show');
}

function applyFilter(){
  // Get selected values from checkboxes
  const body = document.getElementById('filter-modal-body');
  const checkboxes = body.querySelectorAll('input[type="checkbox"]');
  
  if(currentFilterType === 'type'){
    selectedVersions.clear();
    checkboxes.forEach(cb => {
      if(cb.checked) selectedVersions.add(cb.value);
    });
  } else if(currentFilterType === 'zone'){
    selectedZones.clear();
    checkboxes.forEach(cb => {
      if(cb.checked) selectedZones.add(cb.value);
    });
  } else if(currentFilterType === 'status'){
    selectedStatuses.clear();
    checkboxes.forEach(cb => {
      if(cb.checked) selectedStatuses.add(cb.value);
    });
  } else if(currentFilterType === 'user'){
    selectedUsers.clear();
    checkboxes.forEach(cb => {
      if(cb.checked) selectedUsers.add(cb.value);
    });
  } else if(currentFilterType === 'running_status'){
    selectedRunningStatuses.clear();
    checkboxes.forEach(cb => {
      if(cb.checked) selectedRunningStatuses.add(cb.value);
    });
  }
  
  closeFilterModal();
  updateFilterTags();
  loadPanel(false); // Use throttled version, don't force refresh for filter changes
}

function updateFilterTags(){
  const tagsContainer = document.getElementById('filter-tags');
  tagsContainer.innerHTML = '';
  
  const hasFilters = selectedVersions.size > 0 || selectedZones.size > 0 || selectedStatuses.size > 0 || selectedUsers.size > 0 || selectedRunningStatuses.size > 0;
  
  if(!hasFilters){
    tagsContainer.style.display = 'none';
    return;
  }
  
  tagsContainer.style.display = 'flex';
  
  // Add version tags
  selectedVersions.forEach(v => {
    const tag = document.createElement('div');
    tag.className = 'filter-tag';
    tag.innerHTML = `
      <span>type: ${v}</span>
      <span class="filter-tag-close" onclick="removeFilter('type', '${v}')">✕</span>
    `;
    tagsContainer.appendChild(tag);
  });
  
  // Add zone tags
  selectedZones.forEach(z => {
    const tag = document.createElement('div');
    tag.className = 'filter-tag';
    tag.innerHTML = `
      <span>zone: ${z}</span>
      <span class="filter-tag-close" onclick="removeFilter('zone', '${z}')">✕</span>
    `;
    tagsContainer.appendChild(tag);
  });
  
  // Add status tags
  selectedStatuses.forEach(s => {
    const tag = document.createElement('div');
    tag.className = 'filter-tag';
    tag.innerHTML = `
      <span>status: ${s}</span>
      <span class="filter-tag-close" onclick="removeFilter('status', '${s}')">✕</span>
    `;
    tagsContainer.appendChild(tag);
  });
  
  // Add user tags
  selectedUsers.forEach(u => {
    const tag = document.createElement('div');
    tag.className = 'filter-tag';
    tag.innerHTML = `
      <span>user: ${u}</span>
      <span class="filter-tag-close" onclick="removeFilter('user', '${u}')">✕</span>
    `;
    tagsContainer.appendChild(tag);
  });
  
  // Add running status tags
  selectedRunningStatuses.forEach(rs => {
    const tag = document.createElement('div');
    tag.className = 'filter-tag';
    tag.innerHTML = `
      <span>running status: ${rs}</span>
      <span class="filter-tag-close" onclick="removeFilter('running_status', '${rs}')">✕</span>
    `;
    tagsContainer.appendChild(tag);
  });
}

function removeFilter(type, value){
  if(type === 'type'){
    selectedVersions.delete(value);
  } else if(type === 'zone'){
    selectedZones.delete(value);
  } else if(type === 'status'){
    selectedStatuses.delete(value);
  } else if(type === 'user'){
    selectedUsers.delete(value);
  } else if(type === 'running_status'){
    selectedRunningStatuses.delete(value);
  }
  updateFilterTags();
  loadPanel(false); // Use throttled version, don't force refresh for filter changes
}

function setTopbarVar(){
  const tb = document.getElementById('topbar2');
  document.documentElement.style.setProperty('--topbar', (tb?.offsetHeight||60)+'px');
}
window.addEventListener('load', setTopbarVar);
window.addEventListener('resize', setTopbarVar);

// Close modal when clicking outside
document.getElementById('filter-modal').addEventListener('click', (e) => {
  if(e.target && e.target.id === 'filter-modal') closeFilterModal();
});

// Statistics collapse/expand state
let statisticsExpanded = true;

function toggleStatistics(){
  const statsDisplay = document.getElementById('statistics-display');
  const toggleIcon = document.getElementById('statistics-toggle-icon');
  
  statisticsExpanded = !statisticsExpanded;
  
  if(statisticsExpanded){
    statsDisplay.style.display = 'block';
    toggleIcon.textContent = '▼';
  } else {
    statsDisplay.style.display = 'none';
    toggleIcon.textContent = '▶';
  }
}

// Extract specific TPU type (v4, v5e, v5p, v6e)
function extractTpuType(alias, fullName, version){
  // Try alias first
  const aliasStr = (alias || '').toLowerCase();
  if(aliasStr.match(/^v4[-_]/)) return 'v4';
  if(aliasStr.match(/^v5e[-_]/)) return 'v5e';
  if(aliasStr.match(/^v5p[-_]/)) return 'v5p';
  if(aliasStr.match(/^v5[-_]/)) return 'v5p'; // v5 defaults to v5p
  if(aliasStr.match(/^v6e[-_]/)) return 'v6e';
  if(aliasStr.match(/^v6[-_]/)) return 'v6e'; // v6 defaults to v6e
  
  // Try fullName
  const fullNameStr = (fullName || '').toLowerCase();
  if(fullNameStr.match(/v4[-_]/)) return 'v4';
  if(fullNameStr.match(/v5e[-_]/)) return 'v5e';
  if(fullNameStr.match(/v5p[-_]/)) return 'v5p';
  if(fullNameStr.match(/v5[-_]/)) return 'v5p';
  if(fullNameStr.match(/v6e[-_]/)) return 'v6e';
  if(fullNameStr.match(/v6[-_]/)) return 'v6e';
  
  // Try version field
  if(version){
    const vLower = version.toLowerCase();
    if(vLower === 'v4') return 'v4';
    if(vLower === 'v5e') return 'v5e';
    if(vLower === 'v5p' || vLower === 'v5') return 'v5p';
    if(vLower === 'v6e' || vLower === 'v6') return 'v6e';
  }
  
  return null;
}

// Extract version from alias or fullName (e.g., v4-32 -> v4, v6e-64 -> v6, v5p-64 -> v5)
function extractVersion(alias, fullName, version){
  const tpuType = extractTpuType(alias, fullName, version);
  if(tpuType === 'v4') return 'v4';
  if(tpuType === 'v5e' || tpuType === 'v5p') return 'v5';
  if(tpuType === 'v6e') return 'v6';
  return null;
}

// Extract card count from TPU name (e.g., v4-32 -> 32, v6-64 -> 64)
function extractCardCount(alias, fullName){
  // Try to extract from alias first (e.g., v4-32)
  const aliasMatch = (alias || '').match(/v\d+[ep]?[-_](\d+)/i);
  if(aliasMatch) return parseInt(aliasMatch[1]);
  
  // Try to extract from fullName
  const fullNameMatch = (fullName || '').match(/v\d+[ep]?[-_](\d+)/i);
  if(fullNameMatch) return parseInt(fullNameMatch[1]);
  
  return 0;
}

// A100 conversion factors
const A100_FACTORS = {
  'v4': 275 / 312,
  'v5e': 197 / 312,
  'v5p': 459 / 312,
  'v6e': 918 / 312
};

// Cost per card (USD per hour)
const COST_PER_CARD = {
  'v4': 0.9215,
  'v5e': 0.244926,
  'v5p': 1.428,
  'v6e': 0.680749
};

// Total capacity for each version and zone
const TOTAL_CAPACITY = {
  'v4': 1344,
  'v5': 1024,
  'v6': 1536,
  // Zone-specific capacities
  'v5-us-central1-a': 2208,
  'v6-us-central1-b': 2500,
  'v6-us-east1-d': 64,
  'v4-us-central2-b': 120,
  'v6-europe-west4-a': 64,
};

// Cache for gcloud counts
let cachedGcloudCounts = null;
let lastGcloudFetchTime = 0;
const GCLOUD_FETCH_INTERVAL = 60000; // 1 minute
const GCLOUD_ENABLED = true; // Enabled with timeout protection

// Fetch gcloud counts (with caching and timeout)
async function fetchGcloudCounts(){
  if (!GCLOUD_ENABLED) {
    return {};
  }
  const now = Date.now();
  if(cachedGcloudCounts && (now - lastGcloudFetchTime < GCLOUD_FETCH_INTERVAL)){
    return cachedGcloudCounts;
  }
  
  try {
    // Use AbortController for timeout (3 seconds)
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 3000);
    
    const res = await fetch(`{{ url_for('api_tpu_gcloud_counts') }}`, {
      signal: controller.signal
    });
    clearTimeout(timeoutId);
    
    const data = await res.json();
    if(data.ok && data.counts){
      cachedGcloudCounts = data.counts;
      lastGcloudFetchTime = now;
      return data.counts;
    }
  } catch(e) {
    if(e.name !== 'AbortError') {
      console.error('Failed to fetch gcloud counts:', e);
    }
    // Return cached data or empty object on error
  }
  return cachedGcloudCounts || {};
}

// Calculate and display statistics
async function calculateStatistics(allRows){
  const stats = {}; // {version: {zone: {READY: count, CREATING: count, runningReserved: {type: count, a100: count, cost: number}}}}
  
  allRows.forEach(r => {
    // Extract version and specific type
    const version = extractVersion(r.alias, r.full_name, r.version);
    const tpuType = extractTpuType(r.alias, r.full_name, r.version);
    const zone = (r.zone || '').trim();
    const status = (r.script_note || '').toUpperCase();
    const runningStatus = (r.running_status || '').toLowerCase();
    
    // Only process v4, v5, v6
    if(!version || !['v4', 'v5', 'v6'].includes(version) || !zone) return;
    
    // Extract card count
    const cardCount = extractCardCount(r.alias, r.full_name);
    if(cardCount === 0) return;
    
    // Initialize structure
    if(!stats[version]) stats[version] = {};
    if(!stats[version][zone]) {
      stats[version][zone] = {
        READY: 0, 
        CREATING: 0,
        free: 0,
        running: 0,
        reserved: 0,
        runningReserved: {}
      };
    }
    
    // Count READY and CREATING
    if(status === 'READY') {
      stats[version][zone].READY += cardCount;
      // Count running_status for READY TPUs only
      if(runningStatus === 'free') {
        stats[version][zone].free += cardCount;
      } else if(runningStatus === 'running') {
        stats[version][zone].running += cardCount;
      } else if(runningStatus === 'reserved') {
        stats[version][zone].reserved += cardCount;
      }
    }
    if(status === 'CREATING') stats[version][zone].CREATING += cardCount;
    
    // Count running/reserved in READY status for A100 calculation
    if(status === 'READY' && (runningStatus === 'running' || runningStatus === 'reserved') && tpuType) {
      if(!stats[version][zone].runningReserved[tpuType]) {
        stats[version][zone].runningReserved[tpuType] = 0;
      }
      stats[version][zone].runningReserved[tpuType] += cardCount;
    }
  });
  
  // Fetch gcloud counts (with timeout protection)
  let gcloudCounts = {};
  try {
    // Use Promise.race to add timeout (2 seconds max wait)
    gcloudCounts = await Promise.race([
      fetchGcloudCounts(),
      new Promise((resolve) => setTimeout(() => resolve({}), 2000))
    ]);
  } catch(e) {
    console.error('Error fetching gcloud counts in calculateStatistics:', e);
    gcloudCounts = {};
  }
  
  // Display statistics
  const statsContainer = document.getElementById('statistics-display');
  // Keep current display state (don't force show/hide)
  
  if(Object.keys(stats).length === 0){
    statsContainer.innerHTML = '<div class="muted">No statistics available</div>';
    return;
  }
  
  let html = '<div style="display:flex;flex-direction:column;gap:12px;">';
  const versions = ['v4', 'v5', 'v6'];
  let hasAnyData = false;
  let grandTotalA100 = 0;
  let grandTotalCost = 0;
  
  versions.forEach(v => {
    if(!stats[v]) return;
    const zones = Object.keys(stats[v]).sort();
    zones.forEach(zone => {
      const ready = stats[v][zone].READY;
      const creating = stats[v][zone].CREATING;
      const total = ready + creating;
      // Check for zone-specific capacity first, then fall back to version capacity
      const zoneKey = `${v}-${zone}`;
      const maxCapacity = TOTAL_CAPACITY[zoneKey] || TOTAL_CAPACITY[v] || 1000;
      
      if(total > 0){
        hasAnyData = true;
        
        // Get running_status counts for READY TPUs
        const free = stats[v][zone].free || 0;
        const running = stats[v][zone].running || 0;
        const reserved = stats[v][zone].reserved || 0;
        
        // Calculate A100 equivalent and cost for running/reserved in READY
        let totalA100 = 0;
        let totalCost = 0;
        const runningReserved = stats[v][zone].runningReserved || {};
        
        Object.keys(runningReserved).forEach(tpuType => {
          const count = runningReserved[tpuType];
          const factor = A100_FACTORS[tpuType] || 0;
          const costPerCard = COST_PER_CARD[tpuType] || 0;
          totalA100 += count * factor;
          totalCost += count * costPerCard;
        });
        
        // Add to grand totals
        grandTotalA100 += totalA100;
        grandTotalCost += totalCost;
        
        html += `<div style="padding:12px;background:#0e1a37;border:1px solid var(--border);border-radius:8px;">`;
        html += `<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;flex-wrap:wrap;gap:8px;">`;
        html += `<strong style="font-size:16px;">${v}(${zone})</strong>`;
        html += `<div style="display:flex;gap:16px;font-size:14px;flex-wrap:wrap;">`;
        html += `<span style="color:#20c997">Free ${free} </span>`;
        html += `<span style="color:#ffc107">Reserved ${reserved} </span>`;
        html += `<span style="color:#66d9ef">Running ${running} </span>`;
        html += `<span style="color:#20c997">Ready: ${ready}</span>`;
        html += `<span style="color:#9b59b6">Creating: ${creating}</span>`;
        
        // Calculate "others" usage from gcloud counts
        const gcloudTotal = (gcloudCounts[v] && gcloudCounts[v][zone]) ? gcloudCounts[v][zone] : 0;
        const others = Math.max(0, gcloudTotal - total);
        if(others > 0){
          html += `<span style="color:#9fb0d1">Others: ${others}</span>`;
        }
        
        if(totalA100 > 0){
          html += `<span style="color:#ffc107">Compute:${totalA100.toFixed(2)} A100s</span>`;
          html += `<span style="color:#ff6b6b">Cost: $${totalCost.toFixed(2)}/h</span>`;
        }
        html += `</div></div>`;
        
        // Progress bar - based on free/running/reserved/creating and others
        const totalUsed = free + running + reserved + creating + others;
        const totalPercent = Math.min((totalUsed / maxCapacity) * 100, 100);
        
        // Calculate percentages as portion of total used (for proper stacking)
        const totalForBar = free + running + reserved + creating + others;
        const freePercent = totalForBar > 0 ? (free / totalForBar) * totalPercent : 0;
        const runningPercent = totalForBar > 0 ? (running / totalForBar) * totalPercent : 0;
        const reservedPercent = totalForBar > 0 ? (reserved / totalForBar) * totalPercent : 0;
        const creatingPercent = totalForBar > 0 ? (creating / totalForBar) * totalPercent : 0;
        const othersPercent = totalForBar > 0 ? (others / totalForBar) * totalPercent : 0;
        
        html += `<div style="position:relative;width:100%;height:28px;background:#2a3a66;border-radius:4px;overflow:hidden;border:1px solid var(--border);">`;
        let currentLeft = 0;
        if(free > 0){
          html += `<div style="position:absolute;left:${currentLeft}%;width:${freePercent}%;height:100%;background:#20c997;z-index:2;"></div>`;
          currentLeft += freePercent;
        }
        if(reserved > 0){
          html += `<div style="position:absolute;left:${currentLeft}%;width:${reservedPercent}%;height:100%;background:#ffc107;z-index:2;"></div>`;
          currentLeft += reservedPercent;
        }
        if(running > 0){
          html += `<div style="position:absolute;left:${currentLeft}%;width:${runningPercent}%;height:100%;background:#66d9ef;z-index:2;"></div>`;
          currentLeft += runningPercent;
        }
        if(creating > 0){
          html += `<div style="position:absolute;left:${currentLeft}%;width:${creatingPercent}%;height:100%;background:#9b59b6;z-index:2;"></div>`;
          currentLeft += creatingPercent;
        }
        if(others > 0){
          html += `<div style="position:absolute;left:${currentLeft}%;width:${othersPercent}%;height:100%;background:#9fb0d1;z-index:2;"></div>`;
        }
        html += `<div style="position:absolute;left:0;top:0;width:100%;height:100%;display:flex;align-items:center;justify-content:center;color:var(--txt);font-size:12px;font-weight:600;z-index:3;text-shadow:0 0 2px rgba(0,0,0,0.8);">`;
        html += `${gcloudTotal > 0 ? gcloudTotal : total} / ${maxCapacity} (${totalPercent.toFixed(1)}%)`;
        html += `</div></div>`;
        html += `</div>`;
      }
    });
  });
  
  html += '</div>';
  
  // Add grand total summary
  if(hasAnyData && (grandTotalA100 > 0 || grandTotalCost > 0)){
    html += '<div style="margin-top:12px;padding-top:12px;border-top:2px solid var(--border);">';
    html += '<div style="display:flex;justify-content:space-between;align-items:center;padding:12px;background:#0c1630;border:1px solid var(--border);border-radius:8px;">';
    html += '<strong style="font-size:16px;">Total</strong>';
    html += '<div style="display:flex;gap:24px;font-size:16px;font-weight:600;">';
    if(grandTotalA100 > 0){
      html += `<span style="color:#ffc107">Total Compute: ${grandTotalA100.toFixed(2)} A100s</span>`;
    }
    if(grandTotalCost > 0){
      html += `<span style="color:#ff6b6b">Total Cost: $${grandTotalCost.toFixed(2)}/h</span>`;
    }
    html += '</div></div></div>';
  }
  
  if(!hasAnyData){
    statsContainer.innerHTML = '<div class="muted">No READY or CREATING TPUs found</div>';
  } else {
    statsContainer.innerHTML = html;
  }
}

// Throttled version of loadPanel that respects 30-second interval
async function loadPanel(forceRefresh = false){
  const now = Date.now();
  const timeSinceLastRead = now - lastSheetReadTime;
  
  // If force refresh or more than 30 seconds have passed, read from API
  if(forceRefresh || timeSinceLastRead >= SHEET_READ_INTERVAL || !cachedPanelData){
    try {
      const res = await fetch(`{{ url_for('api_list_tpus') }}`);
      const data = await res.json();
      cachedPanelData = data;
      lastSheetReadTime = now;
      pendingRefresh = false; // Clear pending flag after successful read
    } catch(e) {
      console.error('Failed to load panel data:', e);
      // If fetch fails and we have cached data, use it
      if(!cachedPanelData) {
        alert('加载数据失败，请稍后重试');
        return;
      }
    }
  }
  
  // Use cached data if available
  const data = cachedPanelData;
  if(!data) return;
  
  const tbody = document.getElementById('panel-body');
  tbody.innerHTML = "";
  
  // Apply all filters
  let filteredRows = data.rows;
  
  // Filter by version
  if(selectedVersions.size > 0){
    filteredRows = filteredRows.filter(r => {
      const version = (r.version || '').toLowerCase();
      const alias = (r.alias || '').toLowerCase();
      const fullName = (r.full_name || '').toLowerCase();
      
      for(const v of selectedVersions){
        const vLower = v.toLowerCase();
        if(version === vLower || version.startsWith(vLower)) return true;
        if(alias.startsWith(vLower + '-') || fullName.includes(vLower + '-')) return true;
      }
      return false;
    });
  }
  
  // Filter by zone
  if(selectedZones.size > 0){
    filteredRows = filteredRows.filter(r => {
      const zone = (r.zone || '').toLowerCase();
      for(const z of selectedZones){
        if(zone === z.toLowerCase()) return true;
      }
      return false;
    });
  }
  
  // Filter by status
  if(selectedStatuses.size > 0){
    filteredRows = filteredRows.filter(r => {
      const status = (r.script_note || '').toUpperCase();
      for(const s of selectedStatuses){
        if(status === s.toUpperCase()) return true;
      }
      return false;
    });
  }
  
  // Filter by user
  if(selectedUsers.size > 0){
    filteredRows = filteredRows.filter(r => {
      const user = (r.user || '').trim();
      for(const u of selectedUsers){
        if(user === u) return true;
      }
      return false;
    });
  }
  
  // Filter by running status
  if(selectedRunningStatuses.size > 0){
    filteredRows = filteredRows.filter(r => {
      let runningStatus = (r.running_status || '').toLowerCase();
      if(runningStatus === '没了!') runningStatus = 'not found';
      for(const rs of selectedRunningStatuses){
        if(runningStatus === rs.toLowerCase()) return true;
      }
      return false;
    });
  }
  
  // Calculate and display statistics (async now)
  calculateStatistics(data.rows).catch(e => console.error('Error calculating statistics:', e));
  
  // Fetch all ongoing operations once (optimization: avoid calling for each TPU)
  // This includes both apply/reapply operations and run operations
  let allApplyReapplyOperations = {};  // {alias: [operations]}
  let allRunOperations = [];
  
  try {
    // Fetch all apply/reapply operations in one call
    const applyRes = await fetch(`{{ url_for('api_all_ongoing_operations') }}`);
    const applyData = await applyRes.json();
    allApplyReapplyOperations = (applyData.operations && typeof applyData.operations === 'object') ? applyData.operations : {};
  } catch(e) {
    console.error('Failed to fetch apply/reapply operations:', e);
  }
  
  try {
    // Fetch all run operations in one call
    const runRes = await fetch(`{{ url_for('api_ongoing_apply_run_operations') }}`);
    const runData = await runRes.json();
    allRunOperations = (runData.operations && runData.operations.length > 0) ? runData.operations : [];
  } catch(e) {
    console.error('Failed to fetch run operations:', e);
  }
  
  // Sort rows by script_note: READY -> PREEMPTED -> CREATING -> NOT FOUND -> others
  const statusOrder = {'READY': 1, 'PREEMPTED': 2, 'CREATING': 3, 'NOT FOUND': 4};
  const sortedRows = [...filteredRows].sort((a, b) => {
    const aNote = (a.script_note || '').toUpperCase();
    const bNote = (b.script_note || '').toUpperCase();
    const aOrder = statusOrder[aNote] || 99;
    const bOrder = statusOrder[bNote] || 99;
    return aOrder - bOrder;
  });
  
  for(const r of sortedRows){
    const tr = document.createElement('tr');
    
    // Normalize status
    let status = r.running_status || '';
    if(status === '没了!') status = 'NOT FOUND';
    
    // Get script note and determine if apply/reapply is allowed
    let scriptNote = r.script_note || '';
    const canApply = scriptNote.toUpperCase() === 'NOT FOUND';
    const canReapply = scriptNote.toUpperCase() === 'PREEMPTED';
    
    // Determine status color class
    let statusClass = '';
    if(status.toLowerCase() === 'free') statusClass = 'status-free';
    else if(status.toLowerCase() === 'reserved') statusClass = 'status-reserved';
    else if(status.toLowerCase() === 'running') statusClass = 'status-running';
    else if(status.toUpperCase() === 'NOT FOUND') statusClass = 'status-notfound';
    else if(status.toLowerCase() === 'creating') statusClass = 'status-creating';
    
    // Determine script note color class
    let scriptClass = '';
    if(scriptNote.toUpperCase() === 'READY') scriptClass = 'script-ready';
    else if(scriptNote.toUpperCase() === 'NOT FOUND') scriptClass = 'script-notfound';
    else if(scriptNote.toUpperCase() === 'PREEMPTED') scriptClass = 'script-preempted';
    else if(scriptNote.toUpperCase() === 'TERMINATED') scriptClass = 'script-terminated';
    else if(scriptNote.toUpperCase() === 'CREATING') scriptClass = 'script-creating';
    else if(scriptNote) scriptClass = 'script-unknown';
    
    tr.innerHTML = `
      <td class="mono">${r.alias}</td>
      <td class="${statusClass}">${status}</td>
      <td class="mono">${r.zone||''}</td>
      <td class="mono">${r.user||''}</td>
      <td class="${scriptClass}">${scriptNote}</td>
      <td style="max-width:200px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis" title="${r.user_note||''}">${r.user_note||''}</td>
      <td>
        ${ (canApply || canReapply) ? `
          <div>
            <button class="btn" onclick="${canApply ? 'applyTpu' : 'reapplyTpu'}('${r.alias}')" id="btn-${r.alias}">${canApply ? 'Apply' : 'Reapply'}</button>
            <div id="progress-${r.alias}" style="display:none">
              <div class="progress-bar">
                <div class="progress-fill" id="fill-${r.alias}" style="width:0%"></div>
              </div>
              <div class="progress-text" id="text-${r.alias}">准备中...</div>
              <button class="btn" id="log-btn-${r.alias}" onclick="showOperationLog('${r.alias}')" style="display:none; margin-top:4px; font-size:11px;">📋 See Log</button>
            </div>
            <span id="ap-${r.alias}" class="muted"></span>
          </div>` : '<span class="muted">—</span>' }
      </td>
    `;
    tbody.appendChild(tr);
    
    // Check for ongoing operations for this TPU (pass cached operations)
    checkOngoingOperations(r.alias, allApplyReapplyOperations, allRunOperations);
  }
}

async function checkOngoingOperations(alias, cachedApplyReapplyOperations = null, cachedRunOperations = null){
  try{
    // Use cached apply/reapply operations if provided, otherwise fetch (fallback)
    let applyReapplyOps = cachedApplyReapplyOperations;
    if(!applyReapplyOps || typeof applyReapplyOps !== 'object') {
      // Fallback: fetch for this specific TPU only if cache not available
      try {
        const res = await fetch(`{{ url_for('api_tpu_ongoing_operations', alias='__ALIAS__') }}`.replace('__ALIAS__', alias));
        const data = await res.json();
        applyReapplyOps = {};
        if(data.operations && data.operations.length > 0) {
          applyReapplyOps[alias] = data.operations;
        }
      } catch(e) {
        applyReapplyOps = {};
      }
    }
    
    // Check for apply/reapply operations for this TPU
    if(applyReapplyOps[alias] && applyReapplyOps[alias].length > 0){
      const op = applyReapplyOps[alias][0]; // Get the first ongoing operation
      
      // Hide button and show progress
      document.getElementById('btn-'+alias).style.display = 'none';
      document.getElementById('progress-'+alias).style.display = 'block';
      
      // Set task ID on log button
      const logBtn = document.getElementById('log-btn-'+alias);
      if(logBtn) {
        logBtn.setAttribute('data-tid', op.tid);
        logBtn.style.display = 'block';
      }
      
      // Start polling for this operation
      pollApplyWithProgress(op.tid, alias, 20);
      return;
    }
    
    // Use cached run operations if provided, otherwise fetch (fallback)
    let runOperations = cachedRunOperations;
    if(!runOperations || !Array.isArray(runOperations)) {
      try {
        const runRes = await fetch(`{{ url_for('api_ongoing_apply_run_operations') }}`);
        const runData = await runRes.json();
        runOperations = (runData.operations && runData.operations.length > 0) ? runData.operations : [];
      } catch(e) {
        runOperations = [];
      }
    }
    
    if(runOperations.length > 0){
      // Find run operations that use this TPU
      const runOp = runOperations.find(op => op.meta && op.meta.tpu === alias);
      if(runOp){
        // Hide button and show progress
        document.getElementById('btn-'+alias).style.display = 'none';
        document.getElementById('progress-'+alias).style.display = 'block';
        
        // Set task ID on log button
        const logBtn = document.getElementById('log-btn-'+alias);
        if(logBtn) {
          logBtn.setAttribute('data-tid', runOp.tid);
          logBtn.style.display = 'block';
        }
        
        // Start polling for this operation
        pollApplyWithProgress(runOp.tid, alias, 12);
      }
    }
  } catch(e){
    console.log('Error checking ongoing operations for', alias, e);
  }
}

async function applyTpu(alias){
  // Hide button and show progress
  document.getElementById('btn-'+alias).style.display = 'none';
  document.getElementById('progress-'+alias).style.display = 'block';
  
  // Default values
  const times = 20;
  const wait = 5;
  const pre = 'true';
  
  const res = await fetch(`{{ url_for('api_apply_async') }}`,{
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({alias, times, wait, pre})
  });
  const data = await res.json();
  if(!data.ok){ 
    alert('❌ 提交失败\\n'+(data.msg||'')); 
    // Reset UI on error
    document.getElementById('btn-'+alias).style.display = 'block';
    document.getElementById('progress-'+alias).style.display = 'none';
    return; 
  }
  
  // Set task ID on log button
  const logBtn = document.getElementById('log-btn-'+alias);
  if(logBtn) {
    logBtn.setAttribute('data-tid', data.tid);
    logBtn.style.display = 'block';
  }
  
  pollApplyWithProgress(data.tid, alias, times);
}

async function reapplyTpu(alias){
  // Hide button and show progress
  document.getElementById('btn-'+alias).style.display = 'none';
  document.getElementById('progress-'+alias).style.display = 'block';
  
  // Default values
  const times = 20;
  const wait = 5;
  const pre = 'true';
  
  const res = await fetch(`{{ url_for('api_reapply_async') }}`,{
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({alias, times, wait, pre})
  });
  const data = await res.json();
  if(!data.ok){ 
    alert('❌ 提交失败\\n'+(data.msg||'')); 
    // Reset UI on error
    document.getElementById('btn-'+alias).style.display = 'block';
    document.getElementById('progress-'+alias).style.display = 'none';
    return; 
  }
  
  // Set task ID on log button
  const logBtn = document.getElementById('log-btn-'+alias);
  if(logBtn) {
    logBtn.setAttribute('data-tid', data.tid);
    logBtn.style.display = 'block';
  }
  
  pollApplyWithProgress(data.tid, alias, times);
}

async function pollApplyWithProgress(tid, alias, totalTimes){
  const progressFill = document.getElementById('fill-'+alias);
  const progressText = document.getElementById('text-'+alias);
  
  while(true){
    const res = await fetch(`{{ url_for('op_status', tid='__ID__') }}`.replace('__ID__',tid));
    if(!res.ok){ 
      progressText.textContent = '查询状态失败';
      return; 
    }
    const data = await res.json();
    
    if(data.status==='pending' || data.status==='applying' || data.status==='mounting'){
      // Show See Log button during operations
      const logBtn = document.getElementById('log-btn-'+alias);
      if(logBtn) {
        logBtn.setAttribute('data-tid', tid);
        logBtn.style.display = 'block';
      }
      
      // Extract attempt number from message if available
      let attempt = 1;
      if(data.msg && data.msg.includes('attempt')){
        const match = data.msg.match(/attempt (\d+)\/(\d+)/);
        if(match){
          attempt = parseInt(match[1]);
          totalTimes = parseInt(match[2]);
        }
      }
      
      const progress = Math.min((attempt / totalTimes) * 100, 95);
      progressFill.style.width = progress + '%';
      progressText.textContent = `尝试 ${attempt}/${totalTimes} (${Math.round(progress)}%)`;
      await new Promise(r=>setTimeout(r, 2000)); // Check every 2 seconds
      continue;
    }
    
    // Final result
    if(data.status==='ok'){
      progressFill.style.width = '100%';
      progressFill.style.background = '#20c997';
      progressText.textContent = '✅ 成功';
    } else {
      progressFill.style.background = '#ff6b6b';
      progressText.textContent = '❌ 失败';
    }
    
    // Reset UI after 3 seconds
    setTimeout(() => {
      document.getElementById('btn-'+alias).style.display = 'block';
      document.getElementById('progress-'+alias).style.display = 'none';
      progressFill.style.width = '0%';
      progressFill.style.background = 'linear-gradient(90deg,#20c997,#66d9ef)';
    }, 3000);
    
    // Mark that refresh is needed after operation completes
    pendingRefresh = true;
    
    // Schedule refresh if enough time has passed, or schedule it for later
    const now = Date.now();
    const timeSinceLastRead = now - lastSheetReadTime;
    if(timeSinceLastRead >= SHEET_READ_INTERVAL) {
      // Can refresh immediately
      loadPanel(true);
    } else {
      // Schedule refresh after the remaining time
      const remainingTime = SHEET_READ_INTERVAL - timeSinceLastRead;
      setTimeout(() => {
        if(pendingRefresh) {
          loadPanel(true);
        }
      }, remainingTime);
    }
    break;
  }
}

function showOperationLog(alias){
  // Find the ongoing operation for this TPU
  const logBtn = document.getElementById('log-btn-'+alias);
  if(!logBtn) return;
  
  // Get the task ID from the button's data attribute or find it from ongoing operations
  const tid = logBtn.getAttribute('data-tid');
  if(!tid) {
    alert('No ongoing operation found for this TPU');
    return;
  }
  
  // Open log modal
  openLogModal(tid, alias);
}

function showOperationLogById(tid){
  // Open log modal with just the task ID
  openLogModal(tid, 'Operation');
}

function openLogModal(tid, alias){
  // Create modal if it doesn't exist
  let modal = document.getElementById('log-modal');
  if(!modal){
    modal = document.createElement('div');
    modal.id = 'log-modal';
    modal.className = 'modal';
    modal.innerHTML = `
      <div class="modal-content" style="max-width: 800px; max-height: 600px;">
        <div class="modal-header">
          <h3>Operation Log - ${alias}</h3>
          <button class="btn modal-close-btn" onclick="closeLogModal()" style="float: right;">✕ Close</button>
        </div>
        <div class="modal-body">
          <div id="log-content" style="background: #1a1a1a; color: #fff; padding: 10px; border-radius: 4px; font-family: monospace; white-space: pre-wrap; max-height: 400px; overflow-y: auto;"></div>
        </div>
        <div class="modal-footer">
          <button class="btn" onclick="refreshLog()">🔄 Refresh</button>
          <button class="btn" onclick="closeLogModal()" style="background: #6c757d; color: white; border: 1px solid #6c757d;">Close</button>
        </div>
      </div>
    `;
    document.body.appendChild(modal);
    
    // Add click outside to close functionality
    modal.addEventListener('click', function(e) {
      if (e.target === modal) {
        closeLogModal();
      }
    });
    
    // Add ESC key to close functionality
    const escHandler = function(e) {
      if (e.key === 'Escape' && modal.style.display === 'block') {
        closeLogModal();
      }
    };
    document.addEventListener('keydown', escHandler);
    
    // Store the handler for cleanup
    modal.escHandler = escHandler;
  }
  
  // Update modal content
  modal.querySelector('h3').textContent = `Operation Log - ${alias}`;
  modal.style.setProperty('display', 'flex', 'important');
  modal.classList.add('show');
  
  // Store current tid and alias for refresh
  modal.setAttribute('data-tid', tid);
  modal.setAttribute('data-alias', alias);
  
  // Load initial log content
  refreshLog();
}

function closeLogModal(){
  const modal = document.getElementById('log-modal');
  if(modal) {
    modal.style.display = 'none';
    modal.classList.remove('show');
    
    // Clean up event listeners
    if(modal.escHandler) {
      document.removeEventListener('keydown', modal.escHandler);
      modal.escHandler = null;
    }
  }
}

async function refreshLog(){
  const modal = document.getElementById('log-modal');
  if(!modal) return;
  
  const tid = modal.getAttribute('data-tid');
  const alias = modal.getAttribute('data-alias');
  if(!tid) return;
  
  try{
    const res = await fetch(`{{ url_for('op_status', tid='__ID__') }}`.replace('__ID__', tid));
    const data = await res.json();
    
    const logContent = document.getElementById('log-content');
    if(logContent){
      const timestamp = new Date().toLocaleTimeString();
      const logText = `[${timestamp}] Status: ${data.status}\nMessage: ${data.msg || 'No message'}\n\n--- Log Output ---\n${data.log || 'No log available yet...'}`;
      logContent.textContent = logText;
    }
  } catch(e){
    console.error('Failed to refresh log:', e);
  }
}

async function pollApply(tid, alias){
  const el = document.getElementById('ap-'+alias);
  if(el){ el.innerHTML = ` <span class="spinner"></span> 正在尝试…`; }
  while(true){
    const res = await fetch(`{{ url_for('op_status', tid='__ID__') }}`.replace('__ID__',tid));
    if(!res.ok){ alert('查询状态失败'); return; }
    const data = await res.json();
    if(data.status==='pending'){
      await new Promise(r=>setTimeout(r,1500));
      continue;
    }
    if(el){ el.textContent = (data.status==='ok'?'✅ 成功':'❌ 失败'); }
    alert((data.status==='ok'?'✅ Apply 成功':'❌ Apply 失败') + '\\n' + (data.msg||''));
    // Mark that refresh is needed after operation completes
    pendingRefresh = true;
    // Schedule refresh if enough time has passed, or schedule it for later
    const now = Date.now();
    const timeSinceLastRead = now - lastSheetReadTime;
    if(timeSinceLastRead >= SHEET_READ_INTERVAL) {
      // Can refresh immediately
      loadPanel(true);
    } else {
      // Schedule refresh after the remaining time
      const remainingTime = SHEET_READ_INTERVAL - timeSinceLastRead;
      setTimeout(() => {
        if(pendingRefresh) {
          loadPanel(true);
        }
      }, remainingTime);
    }
    break;
  }
}

// TPU Status Modal Functions
let allTPUStatusData = [];
let filteredTPUStatusData = [];

async function viewAllTPUStatus(){
  const modal = document.getElementById('tpu-status-modal');
  const loading = document.getElementById('tpu-status-loading');
  const content = document.getElementById('tpu-status-content');
  const error = document.getElementById('tpu-status-error');
  
  // Show modal and loading state
  modal.classList.add('show');
  loading.style.display = 'block';
  content.style.display = 'none';
  error.style.display = 'none';
  
  try {
    const res = await fetch(`{{ url_for('api_tpu_status') }}`);
    const data = await res.json();
    
    if(!data.ok){
      throw new Error(data.msg || '加载失败');
    }
    
    allTPUStatusData = data.tpus || [];
    filteredTPUStatusData = [...allTPUStatusData];
    
    // Display data
    renderTPUStatusTable();
    updateTPUStatusSummary();
    
    loading.style.display = 'none';
    content.style.display = 'block';
    
  } catch(e) {
    loading.style.display = 'none';
    error.style.display = 'block';
    error.textContent = `加载失败: ${e.message}`;
  }
}

function renderTPUStatusTable(){
  const tbody = document.getElementById('tpu-status-table-body');
  tbody.innerHTML = '';
  
  if(filteredTPUStatusData.length === 0){
    const tr = document.createElement('tr');
    tr.innerHTML = '<td colspan="6" style="text-align:center;color:var(--muted);padding:20px;">没有数据</td>';
    tbody.appendChild(tr);
    return;
  }
  
  // Sort by status: IDLE first, then BUSY, then TIMEOUT, then BAD
  const statusOrder = {
    'IDLE': 1,
    'BUSY': 2,
    'TIMEOUT': 3,
    'BAD': 4
  };
  
  const sortedData = [...filteredTPUStatusData].sort((a, b) => {
    const orderA = statusOrder[a.status] || 999;
    const orderB = statusOrder[b.status] || 999;
    return orderA - orderB;
  });
  
  sortedData.forEach(tpu => {
    const tr = document.createElement('tr');
    
    // Status color coding
    let statusClass = '';
    if(tpu.status === 'IDLE') statusClass = 'script-ready';
    else if(tpu.status === 'BUSY') statusClass = 'script-creating';
    else if(tpu.status === 'TIMEOUT') statusClass = 'script-preempted';
    else if(tpu.status === 'BAD') statusClass = 'script-notfound';
    
    const usersDisplay = tpu.users && tpu.users.length > 0 
      ? tpu.users.join(', ') 
      : '—';
    
    // Add steal button for IDLE TPUs
    let actionCell = '—';
    if(tpu.status === 'IDLE' && tpu.zone) {
      actionCell = `<button class="btn" onclick="showStealModal('${tpu.name}', '${tpu.zone}')" style="padding:4px 8px;font-size:12px;">🥷</button>`;
    }
    
    tr.innerHTML = `
      <td class="mono">${tpu.group || '—'}</td>
      <td class="mono">${tpu.name || '—'}</td>
      <td class="mono">${tpu.zone || '—'}</td>
      <td class="mono ${statusClass}">${tpu.status || '—'}</td>
      <td class="mono">${usersDisplay}</td>
      <td>${actionCell}</td>
    `;
    
    tbody.appendChild(tr);
  });
}

function updateTPUStatusSummary(){
  const summary = document.getElementById('tpu-status-summary');
  
  const total = filteredTPUStatusData.length;
  const idle = filteredTPUStatusData.filter(t => t.status === 'IDLE').length;
  const busy = filteredTPUStatusData.filter(t => t.status === 'BUSY').length;
  const timeout = filteredTPUStatusData.filter(t => t.status === 'TIMEOUT').length;
  const bad = filteredTPUStatusData.filter(t => t.status === 'BAD').length;
  
  // Group statistics
  const groups = {};
  filteredTPUStatusData.forEach(tpu => {
    const group = tpu.group || 'unknown';
    if(!groups[group]){
      groups[group] = {total: 0, idle: 0, busy: 0, timeout: 0, bad: 0};
    }
    groups[group].total++;
    if(tpu.status === 'IDLE') groups[group].idle++;
    if(tpu.status === 'BUSY') groups[group].busy++;
    if(tpu.status === 'TIMEOUT') groups[group].timeout++;
    if(tpu.status === 'BAD') groups[group].bad++;
  });
  
  let html = '<div style="display:flex;flex-wrap:wrap;gap:16px;justify-content:space-between;">';
  html += '<div><strong>总计:</strong> ';
  html += `<span style="color:var(--txt)">${total}</span> | `;
  html += `<span style="color:#20c997">IDLE: ${idle}</span> | `;
  html += `<span style="color:#66d9ef">BUSY: ${busy}</span>`;
  if(timeout > 0) html += ` | <span style="color:#ffc107">TIMEOUT: ${timeout}</span>`;
  if(bad > 0) html += ` | <span style="color:#ff6b6b">BAD: ${bad}</span>`;
  html += '</div></div>';
  
  // Add group breakdown
  if(Object.keys(groups).length > 0){
    html += '<div style="margin-top:12px;padding-top:12px;border-top:1px solid var(--border);">';
    html += '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:8px;">';
    
    Object.keys(groups).sort().forEach(group => {
      const g = groups[group];
      html += `<div style="padding:8px;background:#0c1630;border:1px solid var(--border);border-radius:6px;">`;
      html += `<div style="font-weight:600;margin-bottom:4px;">[${group}]</div>`;
      html += `<div style="font-size:12px;color:var(--muted);">`;
      html += `总计: ${g.total} | `;
      html += `<span style="color:#20c997">空闲: ${g.idle}</span> | `;
      html += `<span style="color:#66d9ef">繁忙: ${g.busy}</span>`;
      if(g.timeout > 0) html += ` | <span style="color:#ffc107">超时: ${g.timeout}</span>`;
      if(g.bad > 0) html += ` | <span style="color:#ff6b6b">故障: ${g.bad}</span>`;
      html += `</div></div>`;
    });
    
    html += '</div></div>';
  }
  
  summary.innerHTML = html;
}

function filterTPUStatus(){
  const searchInput = document.getElementById('tpu-status-search');
  const query = searchInput.value.toLowerCase().trim();
  
  if(!query){
    filteredTPUStatusData = [...allTPUStatusData];
  } else {
    filteredTPUStatusData = allTPUStatusData.filter(tpu => {
      return (tpu.name || '').toLowerCase().includes(query) ||
             (tpu.zone || '').toLowerCase().includes(query) ||
             (tpu.group || '').toLowerCase().includes(query) ||
             (tpu.status || '').toLowerCase().includes(query) ||
             (tpu.users || []).some(u => u.toLowerCase().includes(query));
    });
  }
  
  renderTPUStatusTable();
  updateTPUStatusSummary();
}

function clearTPUStatusFilter(){
  document.getElementById('tpu-status-search').value = '';
  filteredTPUStatusData = [...allTPUStatusData];
  renderTPUStatusTable();
  updateTPUStatusSummary();
}

function closeTPUStatusModal(){
  const modal = document.getElementById('tpu-status-modal');
  modal.classList.remove('show');
}

// Steal TPU Functions
let currentStealTPU = null;
let currentStealZone = null;

async function showStealModal(tpuName, zone){
  currentStealTPU = tpuName;
  currentStealZone = zone;
  
  const modal = document.getElementById('steal-modal');
  const loading = document.getElementById('steal-loading');
  const content = document.getElementById('steal-content');
  const error = document.getElementById('steal-error');
  
  // Update info
  document.getElementById('steal-tpu-name').textContent = tpuName;
  document.getElementById('steal-tpu-zone').textContent = zone;
  
  // Show modal and loading
  modal.classList.add('show');
  loading.style.display = 'block';
  content.style.display = 'none';
  error.style.display = 'none';
  
  try {
    // Extract type from TPU name (e.g., v5p-128 from kmh-tpuvm-v5p-128-spot-xxx)
    const typeMatch = tpuName.match(/v(\d+)(e|p)?-(\d+)/i);
    let typeFilter = '';
    if(typeMatch){
      const version = typeMatch[1];
      const variant = typeMatch[2] || '';
      const cards = typeMatch[3];
      typeFilter = `v${version}${variant}-${cards}`;
    }
    
    // Fetch kengs from API
    const params = new URLSearchParams();
    if(zone) params.append('zone', zone);
    if(typeFilter) params.append('type', typeFilter);
    
    const res = await fetch(`{{ url_for('api_tpu_keng') }}?${params.toString()}`);
    const data = await res.json();
    
    if(!data.ok){
      throw new Error(data.msg || '获取坑位失败');
    }
    
    // Display kengs
    const kengsList = document.getElementById('kengs-list');
    kengsList.innerHTML = '';
    
    document.getElementById('keng-count').textContent = data.count;
    
    if(data.kengs.length === 0){
      kengsList.innerHTML = '<div style="grid-column:1/-1;text-align:center;color:var(--muted);padding:20px;">没有找到可用的坑位</div>';
    } else {
      data.kengs.forEach(keng => {
        const kengBtn = document.createElement('button');
        kengBtn.className = 'btn';
        kengBtn.style.cssText = 'padding:12px;text-align:left;display:flex;flex-direction:column;gap:4px;';
        kengBtn.innerHTML = `
          <div style="font-weight:600;color:var(--accent);">${keng.alias}</div>
          <div style="font-size:11px;color:var(--muted);">${keng.zone}</div>
          ${keng.user_note ? `<div style="font-size:10px;color:#9fb0d1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${keng.user_note}">${keng.user_note}</div>` : ''}
        `;
        kengBtn.onclick = () => executeSteal(keng.alias);
        kengsList.appendChild(kengBtn);
      });
    }
    
    loading.style.display = 'none';
    content.style.display = 'block';
    
  } catch(e) {
    loading.style.display = 'none';
    error.style.display = 'block';
    error.textContent = `加载失败: ${e.message}`;
  }
}

async function executeSteal(kengAlias){
  if(!currentStealTPU || !kengAlias){
    alert('参数错误');
    return;
  }
  
  // Confirm action
  if(!confirm(`确认要将 ${currentStealTPU} 放入坑位 ${kengAlias} 吗？`)){
    return;
  }
  
  try {
    const res = await fetch(`{{ url_for('api_tpu_fang') }}`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        new_tpu_name: currentStealTPU,
        old_tpu_alias: kengAlias
      })
    });
    
    const data = await res.json();
    
    if(data.ok){
      alert(`✅ 成功！\n\n${data.msg}`);
      closeStealModal();
      // Optionally refresh the TPU status
    } else {
      alert(`❌ 失败！\n\n${data.msg}`);
    }
    
  } catch(e) {
    alert(`❌ 操作失败: ${e.message}`);
  }
}

function closeStealModal(){
  const modal = document.getElementById('steal-modal');
  modal.classList.remove('show');
  currentStealTPU = null;
  currentStealZone = null;
}

// Close modal when clicking outside
document.getElementById('tpu-status-modal').addEventListener('click', (e) => {
  if(e.target && e.target.id === 'tpu-status-modal') closeTPUStatusModal();
});

document.getElementById('steal-modal').addEventListener('click', (e) => {
  if(e.target && e.target.id === 'steal-modal') closeStealModal();
});

// Add Enter key support for search
document.addEventListener('DOMContentLoaded', function(){
  const searchInput = document.getElementById('tpu-status-search');
  if(searchInput){
    searchInput.addEventListener('keypress', function(e){
      if(e.key === 'Enter') filterTPUStatus();
    });
  }
});

window.addEventListener('load', () => loadPanel(true)); // Force initial load

</script>
</body>
</html>
"""

# Login page
LOGIN_HTML = """
<!doctype html>
<html lang="zh">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Login - TPU Manager</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
        }
        .login-container {
            background: white;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            width: 100%;
            max-width: 400px;
        }
        h1 {
            margin: 0 0 30px 0;
            color: #333;
            text-align: center;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            color: #555;
            font-weight: 500;
        }
        input[type="password"] {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 6px;
            font-size: 16px;
            box-sizing: border-box;
            transition: border-color 0.3s;
        }
        input[type="password"]:focus {
            outline: none;
            border-color: #667eea;
        }
        button {
            width: 100%;
            padding: 12px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.3s;
        }
        button:hover {
            background: #5568d3;
        }
        .error {
            color: #e74c3c;
            margin-top: 10px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <h1>TPU Manager</h1>
        <form method="POST" action="{{ url_for('login') }}">
            <input type="hidden" name="next" value="{{ request.args.get('next', '') }}">
            <div class="form-group">
                <label for="password">Password</label>
                <input type="password" id="password" name="password" required autofocus>
            </div>
            {% if error %}
            <div class="error">{{ error }}</div>
            {% endif %}
            <button type="submit">Login</button>
        </form>
    </div>
</body>
</html>
"""

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        password = request.form.get('password', '')
        if not password:
            return render_template_string(LOGIN_HTML, error="Password is required")
        
        passwords = get_web_passwords()
        password_hash_hex = password_hash(password)
        
        if password_hash_hex in passwords:
            session['permissions'] = passwords[password_hash_hex]
            session['logged_in'] = True
            next_url = request.form.get('next') or request.args.get('next') or url_for('index')
            return redirect(next_url)
        else:
            return render_template_string(LOGIN_HTML, error="Invalid password")
    
    return render_template_string(LOGIN_HTML)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route("/")
@require_auth
def index():
    users = list_all_users()
    if not users:
        return render_template_string("<p style='padding:24px'>没在 data.json 里找到任何用户。</p>")
    
    # If user has specific username permissions, filter users
    if 'admin' not in session.get('permissions', []):
        user_permissions = session.get('permissions', [])
        users = [u for u in users if u in user_permissions]
        if not users:
            return render_template_string(
                '<html><body style="padding:24px;font-family:sans-serif;">'
                '<h2>No Access</h2><p>You do not have permission to access any users.</p>'
                '<a href="/logout">Logout</a></body></html>'
            )
    
    return redirect(url_for("user_page", username=users[0]))

@app.route("/user/<username>")
def user_page(username: str):
    users = list_all_users()
    data = build_job_rows(username)
    rows = data.get("jobs", [])
    return render_template_string(BASE_HTML, users=users, cur_user=username, rows=rows)

# TPU 面板
@app.route("/tpus")
@require_auth
def tpu_panel():
    # Check if user has 'tpus' permission or is admin
    permissions = session.get('permissions', [])
    if 'admin' not in permissions and 'tpus' not in permissions:
        # Check if any username permission exists (users with username permission can access TPU panel)
        if not any(p in list_all_users() for p in permissions):
            return render_template_string(
                '<html><body style="padding:24px;font-family:sans-serif;">'
                '<h2>Access Denied</h2><p>TPU panel access requires "tpus" permission or username permission.</p>'
                '<a href="/">Go back</a></body></html>'
            ), 403
    return render_template_string(TPU_PANEL_HTML)

# -------- APIs --------
@app.route("/api/user/<username>/jobs")
@require_auth
def api_jobs(username: str):
    if not check_user_access(username):
        return jsonify({"ok": False, "error": "Access denied"}), 403
    return jsonify(build_job_rows(username))

@app.route("/api/user/<username>/pending-operations")
@require_auth
def api_pending_operations(username: str):
    if not check_user_access(username):
        return jsonify({"ok": False, "error": "Access denied"}), 403
    """Get pending operations for a user"""
    with OP_LOCK:
        pending_ops = []
        for tid, task in OP_TASKS.items():
            # Show user-specific operations and global apply+run operations
            if (task["status"] in ["pending", "applying", "mounting"] and 
                (task.get("meta", {}).get("user") == username or 
                 task["kind"] in ["apply+run", "run", "apply+resume", "apply+rerun", "reapply+resume", "reapply+rerun"])):
                pending_ops.append({
                    "tid": tid,
                    "kind": task["kind"],
                    "status": task["status"],
                    "msg": task["msg"],
                    "meta": task["meta"],
                    "ts": task["ts"]
                })
        return jsonify({"ok": True, "operations": pending_ops})

@app.route("/api/user/<username>/clean", methods=["POST"])
@require_auth
def api_clean(username: str):
    if not check_user_access(username):
        return jsonify({"ok": False, "msg": "Access denied"}), 403
    re_flag = request.args.get("re", "0") in ("1", "true", "True", "yes")
    ok, msg = action_clean(username, re_flag=re_flag)
    return jsonify({"ok": ok, "msg": msg})

@app.route("/api/user/<username>/resume-async", methods=["POST"])
@require_auth
def api_resume_async(username: str):
    if not check_user_access(username):
        return jsonify({"ok": False, "msg": "Access denied"}), 403
    payload = request.get_json(silent=True) or {}
    win = str(payload.get("window_id","")).strip()
    tpu = payload.get("tpu", None)
    if not win: return jsonify({"ok": False, "msg": "缺少 window_id"})
    tid = _register_task("resume", {"user": username, "win": win, "tpu": tpu})
    start_thread(_run_resume, username, win, tpu, tid)
    return jsonify({"ok": True, "tid": tid})

@app.route("/api/user/<username>/rerun-async", methods=["POST"])
@require_auth
def api_rerun_async(username: str):
    if not check_user_access(username):
        return jsonify({"ok": False, "msg": "Access denied"}), 403
    payload = request.get_json(silent=True) or {}
    win = str(payload.get("window_id","")).strip()
    tpu = payload.get("tpu", None)
    if not win: return jsonify({"ok": False, "msg": "缺少 window_id"})
    tid = _register_task("rerun", {"user": username, "win": win, "tpu": tpu})
    start_thread(_run_rerun, username, win, tpu, tid)
    return jsonify({"ok": True, "tid": tid})

@app.route("/api/user/<username>/kill-async", methods=["POST"])
@require_auth
def api_kill_async(username: str):
    if not check_user_access(username):
        return jsonify({"ok": False, "msg": "Access denied"}), 403
    payload = request.get_json(silent=True) or {}
    win = str(payload.get("window_id","")).strip()
    if not win: return jsonify({"ok": False, "msg": "缺少 window_id"})
    tid = _register_task("kill", {"user": username, "win": win})
    start_thread(_run_kill, username, win, tid)
    return jsonify({"ok": True, "tid": tid})

@app.route("/api/op/<tid>")
@require_auth
def op_status(tid: str):
    with OP_LOCK:
        item = OP_TASKS.get(tid)
        if not item: return jsonify({"status":"err","msg":"no such task"})
        return jsonify({
            "status": item["status"], 
            "msg": item.get("msg",""), 
            "kind": item["kind"],
            "log": item.get("log", "")
        })

@app.route("/api/tpus")
@require_auth
def api_list_tpus():
    try:
        rows = fetch_tpu_sheet_rows()
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e), "rows": []})
    return jsonify({"ok": True, "rows": rows})

@app.route("/api/tpu-gcloud-counts")
@require_auth
def api_tpu_gcloud_counts():
    """Get TPU counts from spreadsheet K and L columns, aggregated by zone and type"""
    try:
        # Use spreadsheet module to read from K and L columns
        if not SHEET_MODULE_OK or not hasattr(sheet_mod, 'read_tpu_total_counts_from_sheet'):
            return jsonify({"ok": False, "msg": "Spreadsheet module not available", "counts": {}})
        
        try:
            counts = sheet_mod.read_tpu_total_counts_from_sheet()  # type: ignore
            return jsonify({"ok": True, "counts": counts})
        except Exception as e:
            return jsonify({"ok": False, "msg": str(e), "counts": {}})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e), "counts": {}})

@app.route("/api/tpu/<alias>/ongoing-operations")
@require_auth
def api_tpu_ongoing_operations(alias: str):
    """Get ongoing apply/reapply operations for a specific TPU"""
    with OP_LOCK:
        ongoing_ops = []
        for tid, task in OP_TASKS.items():
            if (task["status"] in ["pending", "applying"] and 
                task["kind"] in ["apply", "reapply"] and 
                task.get("meta", {}).get("alias") == alias):
                ongoing_ops.append({
                    "tid": tid,
                    "status": task["status"],
                    "msg": task["msg"],
                    "ts": task["ts"]
                })
        return jsonify({"ok": True, "operations": ongoing_ops})

@app.route("/api/all-ongoing-operations")
@require_auth
def api_all_ongoing_operations():
    """Get all ongoing apply/reapply operations for all TPUs (optimization: single call instead of N calls)"""
    with OP_LOCK:
        all_ops = {}  # {alias: [operations]}
        for tid, task in OP_TASKS.items():
            if (task["status"] in ["pending", "applying"] and 
                task["kind"] in ["apply", "reapply"]):
                alias = task.get("meta", {}).get("alias")
                if alias:
                    if alias not in all_ops:
                        all_ops[alias] = []
                    all_ops[alias].append({
                        "tid": tid,
                        "status": task["status"],
                        "msg": task["msg"],
                        "ts": task["ts"]
                    })
        return jsonify({"ok": True, "operations": all_ops})

@app.route("/api/ongoing-apply-run-operations")
@require_auth
def api_ongoing_apply_run_operations():
    """Get all ongoing apply+run operations"""
    with OP_LOCK:
        ongoing_ops = []
        for tid, task in OP_TASKS.items():
            if (task["status"] in ["pending", "applying", "mounting"] and 
                task["kind"] in ["apply+run", "run"]):
                ongoing_ops.append({
                    "tid": tid,
                    "kind": task["kind"],
                    "status": task["status"],
                    "msg": task["msg"],
                    "meta": task["meta"],
                    "ts": task["ts"]
                })
        return jsonify({"ok": True, "operations": ongoing_ops})

@app.route("/api/tpu/apply-async", methods=["POST"])
@require_auth
def api_apply_async():
    p = request.get_json(silent=True) or {}
    alias = str(p.get("alias","")).strip()
    times = int(p.get("times", 20))
    wait = int(p.get("wait", 5))
    pre = p.get("pre", None)  # 目前仅记录；真正 norm/pre 的判定在"申请并 …"里使用
    if not alias: return jsonify({"ok": False, "msg": "缺少 alias"})
    tid = _register_task("apply", {"alias": alias, "times": times, "wait": wait, "pre": pre})
    start_thread(_run_apply, alias, times, wait, tid)
    return jsonify({"ok": True, "tid": tid})

@app.route("/api/tpu/reapply-async", methods=["POST"])
@require_auth
def api_reapply_async():
    p = request.get_json(silent=True) or {}
    alias = str(p.get("alias","")).strip()
    times = int(p.get("times",20))
    wait = int(p.get("wait", 5))
    pre = p.get("pre", None)
    if not alias: return jsonify({"ok": False, "msg": "缺少 alias"})
    tid = _register_task("reapply", {"alias": alias, "times": times, "wait": wait, "pre": pre})
    start_thread(_run_reapply, alias, times, wait, tid)
    return jsonify({"ok": True, "tid": tid})

@app.route("/api/tpu-status")
@require_auth
def api_tpu_status():
    """Get all TPU status by running 'tou' command"""
    try:
        # Run the 'tou' command (full path)
        result = subprocess.run(
            ["python", "/kmh-nfs-ssd-us-mount/code/qiao/work/tpu_dls/wrap_master.py"],
            capture_output=True,
            text=True,
            timeout=120,  # 2 minutes timeout
            cwd=HERE
        )
        
        if result.returncode != 0:
            return jsonify({
                "ok": False,
                "msg": f"Command failed: {result.stderr}",
                "tpus": []
            })
        
        # Parse the output
        # First, strip ANSI color codes from the output
        output = strip_ansi(result.stdout)
        tpus = []
        
        # Merge wrapped lines (lines that start with whitespace)
        lines = output.split('\n')
        merged_lines = []
        current_line = ""
        
        for line in lines:
            # If line starts with whitespace and current_line has content, it's a continuation
            if line and line[0] in (' ', '\t') and current_line:
                # Remove leading/trailing whitespace and append
                current_line += line.strip()
            else:
                # Save previous line if it exists
                if current_line:
                    merged_lines.append(current_line)
                current_line = line
        
        # Don't forget the last line
        if current_line:
            merged_lines.append(current_line)
        
        # Parse lines with TPU information
        # Format: [INFO] [group] [STATUS] tpu_name (zone) [users=['user1', 'user2']]
        # Note: zone is optional (TIMEOUT status may not have zone)
        import re
        pattern = r'\[INFO\]\s+\[([^\]]+)\]\s+\[(IDLE|BUSY|TIMEOUT|BAD)\]\s+([^\s\(]+)\s*(?:\(([^)]+)\))?(?:\s+users=\[([^\]]*)\])?'
        
        for line in merged_lines:
            match = re.search(pattern, line)
            if match:
                group = match.group(1)
                status = match.group(2)
                tpu_name = match.group(3)
                zone = match.group(4) if match.group(4) else ""
                users_str = match.group(5) if match.group(5) else ""
                
                # Parse users list
                users = []
                if users_str:
                    # Remove quotes and split by comma
                    users = [u.strip().strip("'\"") for u in users_str.split(',')]
                    users = [u for u in users if u]  # Remove empty strings
                
                tpus.append({
                    "group": group,
                    "status": status,
                    "name": tpu_name,
                    "zone": zone,
                    "users": users
                })
        
        return jsonify({
            "ok": True,
            "tpus": tpus,
            "raw_output": output
        })
        
    except subprocess.TimeoutExpired:
        return jsonify({
            "ok": False,
            "msg": "Command timed out",
            "tpus": []
        })
    except Exception as e:
        return jsonify({
            "ok": False,
            "msg": str(e),
            "tpus": []
        })

@app.route("/api/tpu-keng")
@require_auth
def api_tpu_keng():
    """Get available keng (坑位) for stealing TPUs"""
    try:
        import re
        
        # Get zone and type from query parameters
        zone_filter = request.args.get('zone', '').strip()
        type_filter = request.args.get('type', '').strip()
        
        # Normalize type filter (v6e-32 -> v6-32, v5p-32 -> v5-32)
        if type_filter:
            type_filter = re.sub(r'v(\d+)(e|p)(-\d+)?', r'v\1\3', type_filter)
        
        # Read all TPU information from spreadsheet
        if not SHEET_MODULE_OK or not hasattr(sheet_mod, 'read_sheet_info'):
            return jsonify({"ok": False, "msg": "Sheet module not available", "kengs": []})
        
        tpu_information = sheet_mod.read_sheet_info()  # type: ignore
        
        # Filter for deleted TPUs with tmp pattern
        kengs = []
        tmp_pattern = re.compile(r'^v\d+(e|p)?-\d+-tmp', re.IGNORECASE)
        
        for tpu_name, info in tpu_information.items():
            alias = info.get('alias', '')
            
            # Check if alias matches v*-*-tmp* pattern
            if not tmp_pattern.match(alias):
                continue
            
            # Check if TPU is deleted (script_note is 'not found' or 'preempted', OR running_status is '没了!')
            script_note = info.get('script_note', '').lower()
            running_status = info.get('running_status', '')
            if not (script_note in ['not found', 'preempted'] or running_status == '没了!'):
                continue
            
            # Apply zone filter if specified
            if zone_filter and info.get('zone') != zone_filter:
                continue
            
            # Apply type filter if specified
            if type_filter:
                # Extract type from alias (e.g., v6-32 from v6-32-tmp1, v6e-64 from v6e-64-tmp2)
                alias_match = re.match(r'(v\d+(e|p)?-\d+)', alias, re.IGNORECASE)
                if alias_match:
                    alias_type = alias_match.group(1)
                    # Normalize alias type (v6e-64 -> v6-64, v5p-32 -> v5-32)
                    alias_type = re.sub(r'v(\d+)(e|p)(-\d+)', r'v\1\3', alias_type)
                    
                    # Check if it matches the type filter
                    if type_filter.count('-') == 0:
                        # Filter is like 'v6', match all v6-* types
                        if not alias_type.startswith(type_filter):
                            continue
                    else:
                        # Filter is like 'v6-32', exact match
                        if alias_type != type_filter:
                            continue
                else:
                    # Could not extract type from alias, skip
                    continue
            
            kengs.append({
                "alias": alias,
                "zone": info.get('zone', ''),
                "user": info.get('user', ''),
                "user_note": info.get('user_note', ''),
                "full_name": tpu_name
            })
        
        # Sort by alias
        kengs.sort(key=lambda x: x['alias'])
        
        return jsonify({
            "ok": True,
            "kengs": kengs,
            "count": len(kengs)
        })
        
    except Exception as e:
        return jsonify({
            "ok": False,
            "msg": str(e),
            "kengs": []
        })

@app.route("/api/tpu-fang", methods=["POST"])
@require_auth
def api_tpu_fang():
    """Execute tpu fang command to replace old TPU with new TPU"""
    try:
        p = request.get_json(silent=True) or {}
        new_tpu_name = str(p.get("new_tpu_name", "")).strip()
        old_tpu_alias = str(p.get("old_tpu_alias", "")).strip()
        
        if not new_tpu_name:
            return jsonify({"ok": False, "msg": "缺少 new_tpu_name 参数"})
        if not old_tpu_alias:
            return jsonify({"ok": False, "msg": "缺少 old_tpu_alias 参数"})
        
        # Run tpu fang command
        result = subprocess.run(
            ["python", TPU_PY_PATH, "fang", new_tpu_name, old_tpu_alias],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=HERE
        )
        
        output = result.stdout + result.stderr
        
        # Check if successful (look for success message)
        if result.returncode == 0 and "Successfully fanged" in output:
            return jsonify({
                "ok": True,
                "msg": f"成功将 {new_tpu_name} 放入坑位 {old_tpu_alias}",
                "output": output
            })
        else:
            return jsonify({
                "ok": False,
                "msg": f"Fang 失败: {output}",
                "output": output
            })
        
    except subprocess.TimeoutExpired:
        return jsonify({
            "ok": False,
            "msg": "命令超时"
        })
    except Exception as e:
        return jsonify({
            "ok": False,
            "msg": str(e)
        })

# 申请并 Resume/Rerun
@app.route("/api/user/<username>/apply-resume-async", methods=["POST"])
@require_auth
def api_apply_resume_async(username: str):
    if not check_user_access(username):
        return jsonify({"ok": False, "msg": "Access denied"}), 403
    p = request.get_json(silent=True) or {}
    alias = str(p.get("alias","")).strip()
    win = str(p.get("window_id","")).strip()
    pre_str = p.get("pre")
    if not alias or not win: return jsonify({"ok": False, "msg": "缺少 alias/window_id"})
    tid = _register_task("apply+resume", {"alias": alias, "user": username, "win": win})
    start_thread(_run_apply_then_resume, username, win, alias, pre_str, tid)
    return jsonify({"ok": True, "tid": tid})

@app.route("/api/user/<username>/apply-rerun-async", methods=["POST"])
@require_auth
def api_apply_rerun_async(username: str):
    if not check_user_access(username):
        return jsonify({"ok": False, "msg": "Access denied"}), 403
    p = request.get_json(silent=True) or {}
    alias = str(p.get("alias","")).strip()
    win = str(p.get("window_id","")).strip()
    pre_str = p.get("pre")
    if not alias or not win: return jsonify({"ok": False, "msg": "缺少 alias/window_id"})
    tid = _register_task("apply+rerun", {"alias": alias, "user": username, "win": win})
    start_thread(_run_apply_then_rerun, username, win, alias, pre_str, tid)
    return jsonify({"ok": True, "tid": tid})

# 重新申请并 Resume/Rerun
@app.route("/api/user/<username>/reapply-resume-async", methods=["POST"])
@require_auth
def api_reapply_resume_async(username: str):
    if not check_user_access(username):
        return jsonify({"ok": False, "msg": "Access denied"}), 403
    p = request.get_json(silent=True) or {}
    alias = str(p.get("alias","")).strip()
    win = str(p.get("window_id","")).strip()
    pre_str = p.get("pre")
    if not alias or not win: return jsonify({"ok": False, "msg": "缺少 alias/window_id"})
    tid = _register_task("reapply+resume", {"alias": alias, "user": username, "win": win})
    start_thread(_run_reapply_then_resume, username, win, alias, pre_str, tid)
    return jsonify({"ok": True, "tid": tid})

@app.route("/api/user/<username>/reapply-rerun-async", methods=["POST"])
@require_auth
def api_reapply_rerun_async(username: str):
    if not check_user_access(username):
        return jsonify({"ok": False, "msg": "Access denied"}), 403
    p = request.get_json(silent=True) or {}
    alias = str(p.get("alias","")).strip()
    win = str(p.get("window_id","")).strip()
    pre_str = p.get("pre")
    if not alias or not win: return jsonify({"ok": False, "msg": "缺少 alias/window_id"})
    tid = _register_task("reapply+rerun", {"alias": alias, "user": username, "win": win})
    start_thread(_run_reapply_then_rerun, username, win, alias, pre_str, tid)
    return jsonify({"ok": True, "tid": tid})

# -------- Run Modal APIs --------
@app.route("/api/user/<username>/dirs")
@require_auth
def api_user_dirs(username: str):
    """Get directories for a user (like tpu ls <user>)"""
    try:
        code, out = call_cli("ls", username)
        if code != 0:
            return jsonify({"ok": False, "msg": out, "dirs": []})
        
        # Parse output to extract directories with their numbers
        lines = out.split('\n')
        dirs = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Extract directory number and path (format like "1: /path/to/dir")
                if ':' in line:
                    parts = line.split(':', 1)
                    dir_num = parts[0].strip()
                    dir_path = parts[1].strip()
                    if dir_path:
                        dirs.append({"num": dir_num, "path": dir_path})
        
        return jsonify({"ok": True, "dirs": dirs})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e), "dirs": []})

@app.route("/api/tpu-types")
@require_auth
def api_tpu_types():
    """Get available TPU types"""
    try:
        # Get unique TPU types from the sheet data
        rows = fetch_tpu_sheet_rows()
        types = set()
        for row in rows:
            # Extract type from full_name (e.g., "kmh-tpuvm-v4-32-spot-yiyang" -> "v4-32")
            full_name = row.get('full_name', '')
            if 'v' in full_name:
                parts = full_name.split('-')
                for i, part in enumerate(parts):
                    if part.startswith('v') and i + 1 < len(parts):
                        tpu_type = f"{part}-{parts[i+1]}"
                        types.add(tpu_type)
                        break
        
        return jsonify({"ok": True, "types": sorted(list(types))})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e), "types": []})

@app.route("/api/tpus/<tpu_type>")
@require_auth
def api_tpus_by_type(tpu_type: str):
    """Get TPUs of a specific type, including all TPUs (running/reserved/free)"""
    try:
        rows = fetch_tpu_sheet_rows()
        current_user = request.args.get('user', '')
        
        # Filter TPUs by type only, include all TPUs
        filtered_tpus = []
        for row in rows:
            full_name = row.get('full_name', '')
            if tpu_type in full_name:
                filtered_tpus.append(row)
        
        return jsonify({"ok": True, "tpus": filtered_tpus})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e), "tpus": []})

@app.route("/api/user/<username>/run-async", methods=["POST"])
@require_auth
def api_run_async(username: str):
    if not check_user_access(username):
        return jsonify({"ok": False, "msg": "Access denied"}), 403
    """Run a job on a TPU"""
    p = request.get_json(silent=True) or {}
    dir_path = str(p.get("dir", "")).strip()
    tpu = str(p.get("tpu", "")).strip()
    
    if not dir_path or not tpu:
        return jsonify({"ok": False, "msg": "缺少 dir/tpu"})
    
    tid = _register_task("run", {"user": username, "dir": dir_path, "tpu": tpu})
    start_thread(_run_job, username, dir_path, tpu, tid)
    return jsonify({"ok": True, "tid": tid})

@app.route("/api/user/<username>/apply-run-async", methods=["POST"])
@require_auth
def api_apply_run_async(username: str):
    if not check_user_access(username):
        return jsonify({"ok": False, "msg": "Access denied"}), 403
    """Apply for a TPU and then run a job on it"""
    p = request.get_json(silent=True) or {}
    dir_path = str(p.get("dir", "")).strip()
    tpu = str(p.get("tpu", "")).strip()
    
    if not dir_path or not tpu:
        return jsonify({"ok": False, "msg": "缺少 dir/tpu"})
    
    tid = _register_task("apply+run", {"user": username, "dir": dir_path, "tpu": tpu})
    start_thread(_run_apply_then_job, username, dir_path, tpu, tid)
    return jsonify({"ok": True, "tid": tid})

def _run_job(username: str, dir_path: str, tpu: str, tid: str):
    """Run tpu run <dir> <tpu> <user> with auto mode to avoid interactive prompts"""
    tpu_py = _ensure_tpu_py()
    if os.path.basename(tpu_py) == "tpu":
        cmd = [tpu_py, "run", dir_path, tpu, username, "-f", "-q"]
    else:
        cmd = [sys.executable, tpu_py, "run", dir_path, tpu, username, "-f", "-q"]
    
    try:
        p = subprocess.run(cmd, text=True, capture_output=True)
        out = (p.stdout or "") + ("\n" + (p.stderr or "") if p.stderr else "")
        
        # Append to existing log if any
        with OP_LOCK:
            if tid in OP_TASKS:
                existing_log = OP_TASKS[tid].get("log", "")
                new_log = existing_log + f"\n--- Running Job ---\n{out.strip()}"
                OP_TASKS[tid]["log"] = new_log
        
        _finish_task(tid, p.returncode == 0, out.strip(), out.strip())
    except Exception as e:
        _finish_task(tid, False, f"Failed to run job: {str(e)}", str(e))

def _run_apply_then_job(username: str, dir_path: str, tpu: str, tid: str):
    """Apply for TPU and then run job"""
    # Update status to "Applying for TPU"
    with OP_LOCK:
        if tid in OP_TASKS:
            OP_TASKS[tid]["status"] = "applying"
            OP_TASKS[tid]["msg"] = f"Applying for TPU {tpu}..."
    
    # Apply for the TPU with retry logic (same as _run_apply)
    times = 20
    sleep_sec = 5
    ok = False
    last = ""
    log_output = ""
    
    for i in range(max(1, times)):
        # Update progress
        with OP_LOCK:
            if tid in OP_TASKS:
                OP_TASKS[tid]["msg"] = f"Applying for TPU {tpu}... (attempt {i+1}/{times})"
        
        code, out = call_cli("apply", tpu)
        last = out
        log_output += f"Attempt {i+1}/{times}:\n{out}\n\n"
        
        # Store log after each attempt so it's available during progress
        with OP_LOCK:
            if tid in OP_TASKS:
                OP_TASKS[tid]["log"] = log_output
        
        # Check both exit code and output for success indicators
        if code == 0 and "[FAIL]" not in out and "not found" not in out.lower():
            ok = True
            break
        time.sleep(max(1, sleep_sec))
    
    if not ok:
        _finish_task(tid, False, f"apply failed after {times} attempts: {last}", last)
        return
    
    # Update status to "TPU applied, mounting disk" and store log
    with OP_LOCK:
        if tid in OP_TASKS:
            OP_TASKS[tid]["status"] = "mounting"
            OP_TASKS[tid]["msg"] = f"TPU {tpu} applied, mounting disk..."
            OP_TASKS[tid]["log"] = log_output
    
    # Then run the job
    _run_job(username, dir_path, tpu, tid)

from flask import Response, request
from string import Template
import html as _html
import re

ANSI_RE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
# -------- Log 页面（安全转义 + 长度限制 + 去色/跟随/自动刷新） --------
@app.route("/log/<username>/<window_id>")
@require_auth
def view_log(username: str, window_id: str):
    if not check_user_access(username):
        return render_template_string(
            '<html><body style="padding:24px;font-family:sans-serif;">'
            '<h2>Access Denied</h2><p>You do not have permission to view logs for user "{username}".</p>'
            '<a href="/">Go back</a></body></html>'.format(username=username)
        ), 403
    # --- 参数与边界 ---
    raw_n = request.args.get("n", "500000")
    try:
        n_int = int(raw_n)
    except Exception:
        n_int = 500000
    tail_chars = max(10_000, min(2_000_000, n_int))

    # 去 ANSI：?ansi=0 关闭 ANSI（默认 1 保留原样）
    ansi_flag = request.args.get("ansi", "1")
    strip_ansi_flag = (ansi_flag == "0")

    # 默认抓取日志
    user_obj = get_user_obj(username)
    if not user_obj:
        ok, log = False, f"user '{username}' not found"
    else:
        log = tmux_capture(user_obj.tmux_name, window_id, last_n=tail_chars)
        ok = True

    if not ok:
        log = f"[Error] {log}"

    # 去颜色（可选）
    if strip_ansi_flag:
        log = ANSI_RE.sub("", log or "")

    # HTML 转义（避免破坏页面或注入）
    log_html = _html.escape(log or "")

    html = f"""
    <html>
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width,initial-scale=1">
        <title>Log {username}:{window_id}</title>
        <style>
          body {{
            background:#0b1220; color:#eaf1ff;
            font-family: ui-monospace, Menlo, Monaco, Consolas, monospace; margin:0;
          }}
          pre {{
            white-space: pre-wrap; word-break: break-word;
            font-size:12px; line-height:1.35; padding:16px; margin:0;
            overflow:auto; height: calc(100vh - 64px);
          }}
          .top {{
            position:sticky; top:0; background:#0f1a33; border-bottom:1px solid #203055;
            padding:8px 12px; display:flex; align-items:center; gap:12px; z-index: 1;
          }}
          kbd {{
            padding:2px 6px; border:1px solid #203055; border-bottom-width:2px;
            border-radius:4px; background:#0f1a33; color:#bcd0ff;
          }}
          .btn {{
            border:1px solid #203055; background:#102046; color:#d9e1ff;
            padding:6px 10px; border-radius:6px; cursor:pointer;
          }}
          .input {{
            background:#0d1a36; border:1px solid #203055; color:#d9e1ff;
            padding:6px 8px; border-radius:6px; min-width:100px;
          }}
          label {{ user-select:none; }}
          .sep {{ width:1px; height:20px; background:#203055; margin:0 6px; }}
          .hint {{ color:#9fb0c6; }}
        </style>
      </head>
      <body>
        <div class="top">
          <button class="btn" onclick="doRefresh()">刷新</button>
          <span class="hint">或按 <kbd>L</kbd> 刷新</span>
          
          <div class="sep"></div>
          
          <button class="btn" onclick="window.close()" style="background:#dc3545; color:white; border-color:#dc3545;">关闭窗口</button>
          <span class="hint">或按 <kbd>Esc</kbd> 关闭</span>

          <div class="sep"></div>

          <label>Tail 字符数：</label>
          <input class="input" id="tailN" value="{tail_chars}" style="width:120px">
          <button class="btn" onclick="setN()">重新加载</button>

          <div class="sep"></div>

          <label><input type="checkbox" id="ansiOff"> 去色 ANSI</label>
          <span class="hint">（或 URL 加 <code>?ansi=0</code>）</span>

          <div class="sep"></div>

          <label><input type="checkbox" id="follow" checked> 跟随尾部</label>
          <span class="hint">快捷键 <kbd>F</kbd></span>

          <div class="sep"></div>

          <label>自动刷新(秒)：</label>
          <input class="input" id="interval" placeholder="例如 5" style="width:80px">
          <button class="btn" id="btnAuto">开始</button>

          <div class="sep"></div>

          <span class="hint">Home/End 到顶/到底</span>
        </div>
        <pre id="log">{log_html}</pre>

        <script>
          // 状态
          let follow = true;
          let autoTimer = null;

          // 初始化控件状态
          const elAnsi   = document.getElementById('ansiOff');
          const elFollow = document.getElementById('follow');
          const elLog    = document.getElementById('log');
          const elIntv   = document.getElementById('interval');
          const elBtnAuto= document.getElementById('btnAuto');

          // 从 URL 参数恢复状态
          const urlNow = new URL(window.location.href);
          const ansiQP = urlNow.searchParams.get('ansi');
          if (ansiQP === '0') elAnsi.checked = true;

          // 刷新 / 重新加载
          function doRefresh() {{
            window.location.reload();
          }}
          function setN() {{
            const n = document.getElementById('tailN').value || '400000';
            const url = new URL(window.location.href);
            url.searchParams.set('n', n);
            url.searchParams.set('ansi', elAnsi.checked ? '0' : '1');
            window.location.href = url.toString();
          }}

          // 去色复选：采用 URL 方案触发重新加载
          elAnsi.addEventListener('change', () => {{
            const url = new URL(window.location.href);
            url.searchParams.set('ansi', elAnsi.checked ? '0' : '1');
            window.location.href = url.toString();
          }});

          // 跟随尾部
          function scrollToBottom() {{
            elLog.scrollTop = elLog.scrollHeight;
          }}
          function maybeFollow() {{
            if (follow) scrollToBottom();
          }}
          elFollow.addEventListener('change', () => {{
            follow = elFollow.checked;
            if (follow) scrollToBottom();
          }});
          document.addEventListener('DOMContentLoaded', () => {{
            maybeFollow();
          }});

          // 自动刷新
          function startAuto() {{
            const s = parseFloat(elIntv.value);
            if (!Number.isFinite(s) || s <= 0) {{
              alert('请输入有效的秒数，例如 5');
              return;
            }}
            stopAuto();
            autoTimer = setInterval(doRefresh, Math.max(1000, s * 1000));
            elBtnAuto.textContent = '停止';
          }}
          function stopAuto() {{
            if (autoTimer) {{
              clearInterval(autoTimer);
              autoTimer = null;
            }}
            elBtnAuto.textContent = '开始';
          }}
          elBtnAuto.addEventListener('click', () => {{
            if (autoTimer) stopAuto(); else startAuto();
          }});

          // 键盘快捷键
          document.addEventListener('keydown', function(e) {{
            if (e.key === 'Escape') window.close();
            if (e.key === 'l' || e.key === 'L') doRefresh();
            if (e.key === 'f' || e.key === 'F') {{
              follow = !follow; elFollow.checked = follow; if (follow) scrollToBottom();
            }}
            if (e.key === 'r' || e.key === 'R') {{
              if (autoTimer) stopAuto(); else startAuto();
            }}
            if (e.key === 'End') scrollToBottom();
            if (e.key === 'Home') elLog.scrollTop = 0;
          }});
        </script>
      </body>
    </html>
    """
    return Response(html, mimetype="text/html; charset=utf-8")



if __name__ == "__main__":
    port = int(os.environ.get("WEB_PORT", "7960"))
    host = os.environ.get("WEB_HOST", "0.0.0.0")
    print(f"[web_ui] http://{host}:{port}  (TPU_PY_PATH={TPU_PY_PATH})")
    app.run(host=host, port=port, debug=False)
