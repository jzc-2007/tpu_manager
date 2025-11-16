from __future__ import annotations

from .jobs import Job, run_job_on_tpu
from .constants import *          # INFO, GOOD, FAIL, etc.
from .helpers import *            # get_chn_time_str, etc.
from .data_io import *            # read_and_lock_queue, write_and_unlock_queue, release_lock_queue, read_data
from .sheet import read_sheet_info, write_sheet_info, get_tpu_info_sheet
from .operate import kill_jobs_tpu, check_tpu_status
from .users import user_from_dict
from .logger import get_wandb_notes

import os
import shlex
import subprocess
import time
import random
from typing import Optional


class Task:
    def __init__(
        self,
        job: Job,
        user,
        tpu_info,
        priority_info=None,
        job_info=None,
        other_info=None
    ):
        """
        job: a Job object
        tpu_info: a dict, keys:
            - valid_tpu: list[str] of TPU full names allowed
        priority_info: a dict, keys:
            - permission: two-digit number
              * first digit (for status=='failed'): bit1=other, bit0=self
              * second digit (for status=='finished'): bit1=other, bit0=self
        job_info: a dict, keys:
            - stage_dir: str
        other_info: a dict, keys:
            - task_id: int random identifier
            - queue_time: enqueue time (display)
            - stage_time: staging time (optional)
        """
        self.job = job
        self.user = user
        self.tpu_info = tpu_info
        self.priority_info = priority_info if priority_info is not None else {}
        self.job_info = job_info if job_info is not None else {}
        self.other_info = other_info if other_info is not None else {}

    def to_dict(self):
        return {
            "job": self.job.to_dict() if self.job else None,
            "user": self.user,
            "tpu_info": self.tpu_info,
            "priority_info": self.priority_info,
            "job_info": self.job_info,
            "other_info": self.other_info,
        }

    @classmethod
    def from_dict(cls, data: dict):
        job_data = data.get("job")
        job_obj = Job.from_dict(job_data) if job_data else None
        return cls(
            job=job_obj,
            user=data.get("user"),
            tpu_info=data.get("tpu_info", {}),
            priority_info=data.get("priority_info", {}),
            job_info=data.get("job_info", {}),
            other_info=data.get("other_info", {}),
        )


def _tmux_has_session(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def ack_queue(ack_information):
    """
    ack_information: dict with keys
        - tpu: TPU full name to be acknowledged
        - status: 'finished' / 'failed'
        - window: optional dict with keys 'session' and 'window' to identify the tmux window
    """
    print(f"{INFO} ack_queue: acknowledging task on TPU {ack_information['tpu']} with status {ack_information['status']}")
    queue = read_and_lock_queue()
    try:
        tpu = ack_information["tpu"]
        status = ack_information["status"]
        tpu_information = get_tpu_info_sheet(tpu)

        idx_to_del: Optional[int] = None
        for i, task_dict in enumerate(queue):
            task_obj = Task.from_dict(task_dict)
            # print(f"check_valid: {check_valid(task_obj, {"tpu": tpu, "info": tpu_information, "status": status})}")
            if check_valid(task_obj, {"tpu": tpu, "info": tpu_information, "status": status}):
                run_job_on_tpu(task_obj.job, tpu, quiet = False, ignore_window=ack_information.get("window"))
                idx_to_del = i
                break

        if idx_to_del is not None:
            del queue[idx_to_del]

        write_and_unlock_queue(queue)

    except Exception as e:
        print(f"{FAIL} ack_queue: error {e}")

    finally:
        release_lock_queue()

def run_queued_job(user_obj, args):
    """
    Run the queued job on the specified TPU in the queue.
    First, check if the job's user is the same as the user_obj.
    If not, print a warning and return.
    If yes, check whether the TPU is allowed for the job, and if it is running, print a warning and return.
    If the TPU is allowed and the job is not running, run the job on the TPU, and remove the job from the queue.

    args:
        - tpu: TPU full name to be run on
        - id: the id of the task to be run
        - user_obj: User object
    """
    queue = read_and_loxck_queue()
    try:
        # read the tpu and id from the args, tpu contains 'v?', where ? is a number between 0 and 9
        # id is a pure integer
        tpu = args[0] if 'v' in args[0] else args[1]
        id = args[0] if is_integer(args[0]) else args[1]
        assert is_integer(id), f"run_queued_job: id {id} is not an integer"

        _, _, _, tpu = get_zone_pre_spot(tpu)
        for i, task_dict in enumerate(queue):
            task_obj = Task.from_dict(task_dict)
            if task_obj.user == user_obj.name and str(task_obj.other_info['task_id']) == str(id):
                if tpu in task_obj.tpu_info['valid_tpu']:
                    if check_tpu_status(tpu) == 'ready':
                        run_job_on_tpu(task_obj.job, tpu, quiet=False)
                        del queue[i]
                        write_and_unlock_queue(queue)
                    else:
                        print(f"{FAIL} run_queued_job: TPU {tpu} is not ready")
                    break
    except Exception as e:
        print(f"{FAIL} run_queued_job: error running tasks on TPU {tpu}: {e}")
    finally:
        release_lock_queue()

def dequeue_and_run(task_id, tpu):
    """
    Find the task with other_info.task_id == task_id, remove it from the queue,
    and run it on the specified TPU.
    """
    zone, _, _, tpu = get_zone_pre_spot(tpu)
    queue = read_and_lock_queue()
    try:
        target = str(task_id)
        task_to_run = None
        idx_to_del: Optional[int] = None

        for i, task_dict in enumerate(queue):
            other = task_dict.get("other_info", {}) or {}
            task_id_in_dict = other.get("task_id")
            if task_id_in_dict is None:
                continue
            if str(task_id_in_dict) == target:
                task_to_run = Task.from_dict(task_dict)
                idx_to_del = i
                break

        if task_to_run is not None and idx_to_del is not None:
            del queue[idx_to_del]
            write_and_unlock_queue(queue)
            run_job_on_tpu(task_to_run.job, tpu, quiet=False)
        else:
            print(f"{FAIL} dequeue_and_run: task {task_id} not found in queue")

    except Exception as e:
        print(f"{FAIL} dequeue_and_run: error running task {task_id} on TPU {tpu}: {e}")

    finally:
        release_lock_queue()


def update_staging_info(task_id, stage_dir, stage_time):
    queue = read_and_lock_queue()
    try:
        for task_dict in queue:
            other = task_dict.get("other_info", {})
            # FIX: use 'task_id' (not 'id'); don't overwrite queue_time
            if int(other.get("task_id", 0)) == int(task_id):
                task_dict.setdefault("other_info", {})["stage_time"] = stage_time
                task_dict.setdefault("job_info", {})["stage_dir"] = stage_dir
                task_dict.setdefault("job", {})["stage_dir"] = stage_dir
                break
        write_and_unlock_queue(queue)
    except Exception as e:
        print(f"{FAIL} update_staging_info: error updating staging info for task {task_id}: {e}")
    finally:
        release_lock_queue()


def check_valid(task, information):
    """
    input:
        task: a task in the queue
        information: dict with keys
            - tpu (full name)
            - info: dict of TPU spreadsheet info (must include 'user')
            - status: 'failed' / 'finished'
    output:
        bool: whether this task is valid to run on the TPU
    """
    # --- sanity: stage_dir present ---
    data = read_data()
    stage_dir = getattr(task, "job_info", {}).get("stage_dir") if hasattr(task, "job_info") else None
    if not stage_dir:
        return False

    print(f"stage_dir: {stage_dir}")

    # --- availability: TPU allowed ---
    valid_tpu = getattr(task, "tpu_info", {}).get("valid_tpu", []) if hasattr(task, "tpu_info") else []
    if information.get("tpu") not in valid_tpu:
        return False

    print(f"valid_tpu: {valid_tpu}")

    # --- permission decoding ---
    status = information.get("status")
    perm_raw = getattr(task, "priority_info", {}).get("permission")
    if perm_raw is None:
        return False

    perm_str = str(perm_raw)
    if len(perm_str) < 2:
        return False

    if status == "finished":
        digit_char = perm_str[1]
    elif status == "failed":
        digit_char = perm_str[0]
    else:
        raise NotImplementedError(f"check_valid: unknown status {information['status']}")

    try:
        digit = int(digit_char)
    except ValueError:
        return False


    allow_own = bool(digit & 1)           # bit0
    allow_other = bool((digit >> 1) & 1)  # bit1
    # --- ownership check ---
    user_spreadsheet = data['users'][task.user]['spreadsheet_name']
    info_user = information.get("info", {}).get("user")

    print(f"digit: {digit}, allow_own: {allow_own}, allow_other: {allow_other}")

    own = (user_spreadsheet == info_user) or (info_user == "free")

    if own and not allow_own:
        return False
    if (not own) and not allow_other:
        return False

    return True


def parse_config_args_for_queue(user_obj, args):
    """
    Queue-mode parser (no single TPU selection).
    Returns:
        dir_id, dir_path, valid_tpu, tag, rule, config_args, customized_settings, spreadsheet_notes, priority
    TPU selection possibilities:
        - explicit TPUs: v?, v?-?, v?+?, v?-?+?, v?+?-?, v?-?+?-?, v?+?-?+?
        - zones: us, asia, us-central, us-east, asia-northeast, all
        - tpu types: v4-32, v6e-32, v6e-64, all
    """
    data = read_data()

    # --- defaults ---
    config_args = ""
    tag, rule = None, None
    customized_settings = {}
    dir_id = "1"
    spreadsheet_notes = None
    priority = {"permission": "01"}  # keep as 2-digit string

    # NEW: collect explicit TPU names/aliases & an optional type filter
    explicit_tpus = []     # preserve order; we'll de-dupe later
    tpu_type = None

    # keys that should NOT become --config*
    ignore_keys = ["dir", "user", "id", "tag", "rule", "ssn", "pm", "type"]

    # accept zones/tpu types
    accept_zones = []
    accept_tpu_types = []

    # alias helpers
    alias_map = data.get("tpu_aliases", {})
    all_tpu_tokens = set()
    for alias, full in alias_map.items():
        all_tpu_tokens.add(alias)
        all_tpu_tokens.add(full)

    # --- parse args ---
    for arg in args:
        if "=" in arg:
            key, value = arg.split("=", 1)

            if key == "tag":
                tag = value
            elif key == "rule":
                rule = value
            elif key == "dir":
                dir_id = value
            elif key == "ssn":
                spreadsheet_notes = value
            elif key == "pm":
                v = "".join(ch for ch in str(value).strip() if ch.isdigit()) or "0"
                priority["permission"] = v.zfill(2)
                # if permission is not 01, 00, 10, or 11, acquire password from user

            elif key == "type":  # NEW: TPU type filter
                tpu_type = value.strip()
            elif key not in ignore_keys:
                if key in getattr(user_obj, "config_aliases", {}):
                    config_args += f" --{user_obj.config_aliases[key]}={value}"
                else:
                    assert key.startswith("config"), f"Unknown config key {key}"
                    config_args += f" --{key}={value}"

        # flags
        if arg == "--log-stage":
            customized_settings["log_stage"] = True

        # collect multiple explicit TPUs (aliases or full names)
        if arg in all_tpu_tokens:
            explicit_tpus.append(alias_map.get(arg, arg))

        if arg in TYPE_DICT:
            accept_tpu_types.extend(TYPE_DICT[arg])
        if arg in ZONE_DICT:
            accept_zones.extend(ZONE_DICT[arg])

        # numeric dir id convenience
        if is_integer(arg):
            dir_id = arg
            if dir_id not in user_obj.working_dir:
                print(f"{FAIL} queue: Directory id {dir_id} not found")
                raise ValueError(f"Directory id {dir_id} not found")
            print(f"{INFO} queue: Using directory id {dir_id}")

    

    if not accept_zones:
        accept_zones = ZONE_DICT['all']
    if not accept_tpu_types:
        accept_tpu_types = TYPE_DICT['all']

    # --- resolve directory path ---
    if dir_id not in user_obj.working_dir:
        print(f"{FAIL} queue: Directory id {dir_id} not found in user config")
        raise ValueError(f"Directory id {dir_id} not found")
    dir_path = user_obj.working_dir[dir_id]

    if not os.path.exists(dir_path):
        raise ValueError(f"Directory {dir_path} does not exist")

    # --- spreadsheet notes (from wandb if not provided) ---
    if spreadsheet_notes is None:
        print(f"{INFO} queue: Getting notes from wandb config file...")
        spreadsheet_notes = get_wandb_notes(dir_path)
        print(f"{INFO} queue: Notes from wandb config file: {spreadsheet_notes}")

    if "-ssn" in args or "--ssn" in args:
        if spreadsheet_notes is not None:
            print(f"{WARNING} queue: Notes already set to {spreadsheet_notes}, change it? (y/n)")
            if input().strip().lower() == "y":
                print("Please enter the notes for the job:")
                spreadsheet_notes = input().strip()
        else:
            print("Please enter the notes for the job:")
            spreadsheet_notes = input().strip()

    if (tag is None) and (spreadsheet_notes is not None) and ("-no-tag" not in args):
        tag = spreadsheet_notes

    # --- build valid_tpu ---
    # Start with any explicit TPUs passed (can be many)
    valid_tpu = list(dict.fromkeys(explicit_tpus))  # de-dup preserving order

    # If a type filter was provided, union in all TPUs of that type
    if not valid_tpu:
        valid_tpu = list(filter_tpu_information(read_sheet_info(), zone=accept_zones, type=accept_tpu_types).keys())
        tpu_type_to_log = ','.join(accept_tpu_types) if accept_tpu_types else 'v'+valid_tpu[0].split('v')[-1]+f'(+{len(valid_tpu)-1})'
    else:
        tpu_type_to_log = ','.join(explicit_tpus) if explicit_tpus else 'v'+valid_tpu[0].split('v')[-1]+f'(+{len(explicit_tpus)-1})'

    
    print(f"valid_tpu: {valid_tpu}")
    # If still empty, fall back to whatever filters read_tpu_info_from_type(args) applies
    if not valid_tpu:
        print(f'{FAIL} parse_config_args_for_queue: NO TPU FOUND')

    if rule is not None:
        rule = RULE_DICT[rule]

    return (
        dir_id,
        dir_path,
        valid_tpu,
        tag,
        rule,
        config_args,
        customized_settings,
        spreadsheet_notes,
        priority,
        tpu_type_to_log
    )

def remove_from_queue(number):
    queue = read_and_lock_queue()
    if number == -1:
        queue = []
    else:
        idx_to_del: Optional[int] = None

        for i, task_dict in enumerate(queue):
            if task_dict['other_info']['task_id'] == number:
                idx_to_del = i
                break

        if idx_to_del is not None:
            del queue[idx_to_del]

    write_and_unlock_queue(queue)

def dequeue(user_obj, args):
    """
    Remove the tasks from the queue.
    args:
        - user_obj: User object
        - args: list of task ids to remove, or '--all' to remove all tasks
    """
    for number in args:
        if not is_integer(number) and number != '--all' and number != '*':
            continue
        remove_from_queue(int(number)) if is_integer(number) else remove_from_queue(-1)

def Queue(user_obj, args):
    data = read_data()
    user_obj = user_from_dict(data["users"][user_obj.name])

    (
        dir_id,
        dir_path,
        valid_tpu,
        tag,
        rule,
        config_args,
        customized_settings,
        spreadsheet_notes,
        priority,
        tpu_type
    ) = parse_config_args_for_queue(user_obj, args)

    unique_id = random.randint(100000, 999999)

    job = Job(
        user=user_obj.name,
        job_dir_id=dir_id,
        job_dir=dir_path,
        job_tags=tag,
        extra_configs=config_args,
        rules=rule,
        customized_settings=customized_settings,
    )

    task = Task(
        job=job,
        user=user_obj.name,
        tpu_info={"valid_tpu": valid_tpu},
        other_info={
            "task_id": unique_id,
            "queue_time": f"CHN: {get_chn_time_str()}",
            "tpu_type": tpu_type
        },
        priority_info=priority,
    )

    if not _tmux_has_session("queue"):
        subprocess.run(["tmux", "new-session", "-d", "-s", "queue"], check=True)

    # create window with index unique_id
    subprocess.run(["tmux", "new-window", "-t", f"queue:{unique_id}"], check=True)

    print(f"{INFO} Queue: Staging job in tmux window queue:{unique_id}")
    time.sleep(0.5)

    # verify window exists
    try:
        lw = subprocess.run(
            ["tmux", "list-windows", "-t", "queue"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
    except subprocess.CalledProcessError:
        print(f"{FAIL} Queue: 'tmux list-windows' failed for session 'queue'")
        return

    if not any(line.startswith(f"{unique_id}:") for line in lw):
        print(f"{FAIL} Queue: Failed to create job in tmux window queue:{unique_id}")
        print(f"{FAIL} Queue: Window {unique_id} not found")
        print(f"{FAIL} Queue: This may indicate the window is already present; please check the tmux session")
        return

    # append to queue with lock
    q = read_and_lock_queue()
    try:
        q.append(task.to_dict())
        write_and_unlock_queue(q)
    except Exception:
        release_lock_queue()
        raise

    # make sure the just_staging.sh file exists
    if not os.path.exists(f"{dir_path}/just_staging.sh"):
        print(f"{FAIL} Queue: just_staging.sh file not found in {dir_path}")
        return

    # start staging
    subprocess.run(
        ["tmux", "send-keys", "-t", f"queue:{unique_id}", f"cd {shlex.quote(dir_path)}", "Enter"],
        check=True,
    )
    subprocess.run(
        ["tmux", "send-keys", "-t", f"queue:{unique_id}", f"source just_staging.sh {unique_id}; sleep 600; exit", "Enter"],
        check=True,
    )

    print(f"{GOOD} Queue: Successfully starting staging in tmux window queue:{unique_id}")

def visualize_queue(limit: int = None, truncate_tpus: int = 6, return_rows: bool = False, user = None):
    """
    Pretty-print the current queue:
      - time  (prefer stage_time, else queue_time)
      - user
      - id    (task_id)
      - permission (two-digit string)
      - tpu_list  (valid_tpu; truncated for display)

    Args:
        limit: if set, show at most this many rows (from the front of the queue).
        truncate_tpus: max TPUs to show before collapsing to "... (+N)".
        return_rows: if True, also return the list of row dicts for programmatic use.
        user: optional user filter to display only the tasks for a specific user.

    Returns:
        If return_rows=True, returns the list of dict rows rendered.
    """
    # --- read queue safely (no mutation) ---
    q = read_and_lock_queue()
    try:
        # snapshot for rendering
        queue_snapshot = list(q[:limit]) if limit is not None else list(q)
        if user is not None:
            queue_snapshot = [task for task in queue_snapshot if task.get("user") == user]
    finally:
        # we didn't modify, so just release; do NOT call write_and_unlock_queue
        release_lock_queue()

    def _pick_time(other_info: dict) -> str:
        if not isinstance(other_info, dict):
            return "-"
        return (other_info.get("stage_time") or other_info.get("queue_time") or "-").replace('CHN: ','')

    def _perm_str(priority_info: dict) -> str:
        if not isinstance(priority_info, dict):
            return "00"
        perm = str(priority_info.get("permission", "00"))
        # normalize to 2-digit display, display "3:os, 2:o-, 1:-s, 0:--"
        # For example, "00" -> '----', "31" -> 'os-s'
        displat_dict = {'0':'--', '1':'-s', '2':'o-', '3':'os'}
        return ''.join([displat_dict[p] for p in perm])

    # --- build rows ---
    rows = []
    for task_dict in queue_snapshot:
        other = task_dict.get("other_info", {}) or {}
        tpu_info = task_dict.get("tpu_info", {}) or {}
        priority_info = task_dict.get("priority_info", {}) or {}

        row = {
            "time": _pick_time(other),
            "user": task_dict.get("user", "-"),
            "id":   str(other.get("task_id", "-")),
            "perm": _perm_str(priority_info),
            "tpu_type": other.get("tpu_type", "-"),
            "note": task_dict.get("job",{}).get("job_tags",'-')[:20]
        }
        rows.append(row)

    # --- render as a simple table ---
    headers = ["time", "user", "id", "perm", "tpu_type", "note"]

    # compute column widths
    col_w = {h: len(h) for h in headers}
    for r in rows:
        for h in headers:
            col_w[h] = max(col_w[h], len(str(r[h])))

    def line(char="-"):
        return "+ " + " + ".join(char * col_w[h] for h in headers) + " +"

    def fmt_row(vals):
        return "| " + " | ".join(str(vals[h]).ljust(col_w[h]) for h in headers) + " |"

    if rows:
        print(line("-"))
        print(fmt_row({h: h for h in headers}))
        print(line("="))
        for r in rows:
            print(fmt_row(r))
        print(line("-"))
        print(f"{INFO} visualize_queue: {len(rows)} task(s) shown")
    else:
        print(f"{INFO} visualize_queue: queue is empty")

    if return_rows:
        return rows

def finish_job(window):
    session_name, window_num = window.split(':')
    window_num = int(window_num)
    data = read_and_lock_data()
    try:
        for user in data['users']:
            if data['users'][user]['tmux_name'] == session_name:
                for job in data['users'][user]['job_data']:
                    if job['windows_id'] == window_num:
                        job['status'] = 'finished'
                        job['extra_msgs']['finish_time_abs'] = get_abs_time_str()
                        job['extra_msgs']['finish_time_chn'] = get_chn_time_str()
                        job['extra_msgs']['finish_time_edt'] = get_edt_time_str()
                        break
                break
        write_and_unlock_data(data)
        # set the status to be reserved for this TPU
        tpu = job['tpu']
        if tpu is not None:
            tpu_info = get_tpu_info_sheet(tpu)
            tpu_info['running_status'] = 'free'
            write_sheet_info(tpu_info)
        print(f"{INFO} finish_job: Finished job {window_num} in session {session_name}")
    except:
        release_lock_data()

    ack_queue({'tpu': tpu, 'status': 'finished', 'window':{'session': session_name, 'window': window_num}})

def fail_job(window):
    session_name, window_num = window.split(':')
    window_num = int(window_num)
    data = read_and_lock_data()
    try:
        for user in data['users']:
            if data['users'][user]['tmux_name'] == session_name:
                for job in data['users'][user]['job_data']:
                    if job['windows_id'] == window_num:
                        job['extra_msgs']['fail_time_abs'] = get_abs_time_str()
                        job['extra_msgs']['fail_time_chn'] = get_chn_time_str()
                        job['extra_msgs']['fail_time_edt'] = get_edt_time_str()
                        break
                break
        write_and_unlock_data(data)
        # set the status to be reserved for this TPU
        tpu = job['tpu']
        if tpu is not None:
            tpu_info = get_tpu_info_sheet(tpu)
            tpu_info['running_status'] = 'reserved(error)'
            write_sheet_info(tpu_info)
        print(f"{INFO} fail_job: Job {window_num} in session {session_name} failed")
    except:
        release_lock_data()
        return False

    ack_queue({'tpu': tpu, 'status': 'failed', 'window':{'session': session_name, 'window': window_num}})

def upd_staging_info(unique_id, window, stage_dir):
    """
    Find the task with other_info.task_id == unique_id and update stage_dir.

    Updates:
      - task["job_info"]["stage_dir"] = stage_dir
      - task["job"]["stage_dir"]      = stage_dir
      - task["other_info"]["window"]  = window  (for reference)
    """
    queue = read_and_lock_queue()
    try:
        target = str(unique_id)
        updated = False

        for task_dict in queue:
            other = task_dict.get("other_info", {}) or {}
            task_id = other.get("task_id")
            if task_id is None:
                continue
            if str(task_id) == target:
                task_dict.setdefault("job_info", {})["stage_dir"] = stage_dir
                task_dict.setdefault("job", {})["stage_dir"] = stage_dir
                updated = True
                break

        if updated:
            write_and_unlock_queue(queue)
            print(f"{GOOD} upd_staging_info: stage_dir set for task {unique_id} -> {stage_dir}")
        else:
            # no mutation; just release the lock
            release_lock_queue()
            print(f"{WARNING} upd_staging_info: task {unique_id} not found in queue")

    except Exception as e:
        # on error, ensure lock is released
        release_lock_queue()
        print(f"{FAIL} upd_staging_info: error updating task {unique_id}: {e}")
