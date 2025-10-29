import os, re, time, json, copy
from .helpers import *
from .constants import *
from . import users
from .data_io import read_and_lock_data, write_and_unlock_data, release_lock_data, read_data, read_and_lock_legacy, write_legacy, write_and_unlock_legacy, release_lock_legacy
from .operate import check_tpu_status, apply_and_set_env, kill_jobs_tpu, restart, check_tpu_running
from .sheet import get_tpu_info_sheet, write_sheet_info, read_tpu_info_from_type, find_tpu_from_type
from .logger import get_wandb_notes
from .autenticate import autenticate
from .gs_buckets import check_gs_logdir_exists

# --- ANSI helpers ---
ANSI_RE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

def _strip_ansi(s: str) -> str:
    return ANSI_RE.sub('', s)

def _vis_len(s: str) -> int:
    return len(_strip_ansi(s))

def _ansi_ljust(s: str, width: int) -> str:
    """Pad with spaces so the *visible* length becomes width."""
    pad = max(0, width - _vis_len(s))
    return s + (' ' * pad)

# --- suppression (unchanged semantics, just using helpers) ---
def _suppress_preview(text: str, limit: int) -> str:
    """
    If text (sans ANSI) exceeds 'limit', show prefix and '..(+N)' where N is hidden.
    Example: 'this is a suppressed te..(+2)'
    """
    vis = _strip_ansi(text)
    if len(vis) <= limit:
        return text
    kept = vis[:limit]
    hidden = len(vis) - limit
    return kept + f"..(+{hidden})"

def _print_in_columns(blocks, num_columns=2, gap=" │ "):
    """
    Pretty print multiple text blocks as a table with full borders.

    Each block is a list of lines (may include ANSI color).
    """
    import math

    # safe fallbacks
    from shutil import get_terminal_size
    term_width = get_terminal_size((120, 20)).columns
    col_width = (term_width - (num_columns - 1) * len(gap) - 4) // num_columns  # inner width

    # utility: ANSI-safe pad
    import re
    ANSI = re.compile(r'\x1b\[[0-9;]*m')
    def visible_len(s):
        return len(ANSI.sub('', s))
    def ansi_ljust(s, width):
        return s + ' ' * max(0, width - visible_len(s))

    # horizontal border pieces
    horiz = "─" * col_width
    top    = "┌" + "┬".join([horiz] * num_columns) + "┐"
    mid    = "├" + "┼".join([horiz] * num_columns) + "┤"
    bottom = "└" + "┴".join([horiz] * num_columns) + "┘"

    # iterate rows of blocks
    for i in range(0, len(blocks), num_columns):
        row_blocks = blocks[i:i + num_columns]
        if len(row_blocks) < num_columns:
            row_blocks += [[] for _ in range(num_columns - len(row_blocks))]

        # row height
        h = max(len(b) for b in row_blocks)
        if i == 0:
            print(top)
        for r in range(h):
            line_parts = []
            for b in row_blocks:
                cell = b[r] if r < len(b) else ""
                line_parts.append(ansi_ljust(cell, col_width))
            print("│" + "│".join(line_parts) + "│")
        if i + num_columns < len(blocks):
            print(mid)
        else:
            print(bottom)



def _kv_rows_to_block(rows):
    """
    rows: list of (key, value)
    Returns list[str]; each line will be ANSI-padded to COL_WIDTH.
    """
    if not rows:
        return [_ansi_ljust("(no info)", COL_WIDTH)]

    # choose key column width
    keyw = min(KEY_COL_MAX, max(KEY_COL_MIN, max(len(str(k)) for k, _ in rows)))
    # value preview limit (visible chars)
    val_limit = max(0, COL_WIDTH - keyw - 1)

    lines = []
    for k, v in rows:
        v = "" if v is None else str(v)
        v_show = _suppress_preview(v, val_limit).replace("\n", " ")
        # compose the line; pad ANSI-aware to COL_WIDTH
        line = f"{str(k):<{keyw}}: {v_show}"
        lines.append(_ansi_ljust(line, COL_WIDTH))
    return lines


# ---------- render one job to rows ----------
def _render_rows_for_job(job_data, msg, last_line, last_line_cut, config, user_obj):
    # returns the rows (list of (key,value)) for this job, and may trigger side-effects like write_error_to_job/ack
    rows = []
    # Base guard
    if (job_data.get("status") is None) or ('s' not in config):
        return rows

    # add tag if there is any tag in the job_data
    if job_data.get("job_tags") is not None and 'T' not in config:
        rows.append(("Tags", job_data.get("job_tags")))

    status = job_data["status"]
    # starting (no logdir yet)
    if status == 'starting':
        rows += [("Status", f"{WARNING}Don't have logdir yet{NC}")]
        return rows

    # error
    if status == 'error':
        err = job_data.get("error")
        if err == 'preempted':
            rows += [("Status", f"{RED}Preempted{NC}")]
        elif err == 'OOM':
            rows += [("Status", f"{RED}OOM{NC}")]
        else:
            rows += [("Status", f"{RED}Error{NC}")]
        return rows

    # killed
    if status == 'killed':
        rows += [("Status", f"{YELLOW}Killed{NC}")]
        if 'v' in config:
            rows.append(("msg", msg))
        return rows

    # resumed / rerunned
    if status in ('resumed', 'rerunned'):
        try:
            child = job_data['extra_msgs']['child']
        except Exception:
            rows += [("Status", f"{RED}Failed to get child window id{NC}")]
            child = None
        rows += [
            ("Status", f"{YELLOW}{job_data.get('error')}{NC}"),
            ("State", f'{status}(child={child})')
        ]
        return rows

    # finished
    if status == 'finished':
        rows += [("Status", f"{GREEN}Finished{NC}")]
        return rows

    # running / starting (detailed parsing)
    if status in ('running', 'starting'):
        # Error patterns
        if (re.search(r'Job failed', last_line_cut) or
            re.search(r'[eE]rror', last_line_cut) or
            re.search(r'FAIL', last_line_cut)) and 's' in config:

            if re.search(r'Allocation type', last_line):
                rows += [("Status", f"{RED}OOM Error{NC}")]
                write_error_to_job(user_obj, job_data, 'OOM')

            elif re.search(r'GRPC [Ee]rror', last_line):
                rows += [("Status", f"{RED}GRPC Error{NC}")]
                write_error_to_job(user_obj, job_data, 'grpc')
                ack_MONITOR()

            elif re.search(r'python: No such file or directory', last_line):
                rows += [("Status", f"{RED}File Error{NC}")]
                write_error_to_job(user_obj, job_data, 'file error')
                ack_MONITOR()

            elif re.search(r'DEADLINE_EXCEEDED', last_line):
                rows += [("Status", f"{RED}DEADLINE EXCEEDED{NC}")]
                write_error_to_job(user_obj, job_data, 'deadline exceeded')
                ack_MONITOR()

            else:
                rows += [("Status", f"{RED}Unknown Error{NC}")]
                write_error_to_job(user_obj, job_data, 'unknown')
                ack_MONITOR()
            return rows

        # Activity patterns
        if (re.search(r'[cC]ompiling', last_line_cut) or
            re.search(r'[cC]ompilation', last_line_cut) or
            re.search(r'[cC]ompile', last_line_cut)) and 's' in config:
            rows += [("Status", f"{GREEN}Compiling{NC}")]
            if 'v' in config:
                rows.append(("msg", msg))
            return rows

        if re.search(r'[sS]ampling ', last_line_cut) and 's' in config:
            epoch = None
            m1 = re.search(r'[eE]poch\s([0-9]{1,6})', last_line_cut)
            m2 = re.search(r'ep=([0-9]){1,4}\.([0-9]){1,6}', last_line_cut)
            if m1:
                epoch = m1.group(1)
            elif m2:
                epoch = m2.group(0)[3:]
            rows += [("Status", f"{GREEN}Sampling{NC}")]
            if epoch is not None:
                rows.append(("Epoch", f"{int(float(epoch))}"))
            if 'v' in config:
                rows.append(("msg", msg))
            return rows

        m_epoch1 = re.search(r'[eE]poch\s([0-9]{1,6})', last_line_cut)
        if m_epoch1 and ('s' in config):
            epoch = m_epoch1.group(1)
            rows += [("Status", f"{GREEN}Running{NC}(ep={epoch})")]
            if 'v' in config:
                rows.append(("msg", msg))
            return rows

        m_epoch2 = re.search(r'ep\s*=\s*([0-9]){1,4}\.([0-9]){1,6}', last_line_cut)
        if m_epoch2 and ('s' in config):
            epoch = m_epoch2.group(0).split('=')[1].strip()
            rows += [("Status", f"{GREEN}Running{NC}(ep={float(epoch):.2f})")]
            if 'v' in config:
                rows.append(("msg", msg))
            return rows

        if re.search(r'[iI]nitializing', last_line_cut) and 's' in config:
            rows += [("Status", f"{GREEN}Initializing{NC}")]
            if 'v' in config:
                rows.append(("msg", msg))
            return rows

        if re.search(r'[sS]taging', last_line_cut) and 's' in config:
            rows += [("Status", f"{GREEN}Staging{NC}")]
            if 'v' in config:
                rows.append(("msg", msg))
            return rows

        if 's' in config:
            rows += [("Status", f"{YELLOW}Unknown{NC}")]
            if 'v' in config:
                rows.append(("msg", msg))
            return rows

    return rows

class Job:
    def __init__(
        self,
        user,
        windows_id=None,
        job_dir_id="",
        job_dir=None,
        tpu=None,
        job_tags="",
        log_dir="",
        stage_dir="",
        extra_configs="",
        status=None,
        error=None,
        stage=0,
        monitor=True,
        rules=None,
        extra_msgs=None,
        customized_settings=None,
        start_time=None
    ):
        self.user = user
        self.windows_id = windows_id
        self.job_dir_id = job_dir_id
        self.job_dir = job_dir
        self.tpu = tpu
        self.job_tags = job_tags
        self.log_dir = log_dir
        self.stage_dir = stage_dir
        self.extra_configs = extra_configs
        self.status = status
        self.error = error
        self.stage = stage
        self.monitor = monitor
        self.rules = rules
        self.extra_msgs = extra_msgs if extra_msgs is not None else {}
        self.customized_settings = customized_settings if customized_settings is not None else {}
        self.start_time = start_time if start_time is not None else {}

    def to_dict(self):
        return {
            "user": self.user,
            "windows_id": self.windows_id,
            "job_dir_id": self.job_dir_id,
            "job_dir": self.job_dir,
            "tpu": self.tpu,
            "job_tags": self.job_tags,
            "log_dir": self.log_dir,
            "stage_dir": self.stage_dir,
            "extra_configs": self.extra_configs,
            "status": self.status,
            "error": self.error,
            "stage": self.stage,
            "monitor": self.monitor,
            "rules": self.rules,
            "extra_msgs": self.extra_msgs,
            "customized_settings": self.customized_settings,
            "start_time": self.start_time
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            user=data.get("user"),
            windows_id=data.get("windows_id"),
            job_dir_id=data.get("job_dir_id", ""),
            job_dir=data.get("job_dir"),
            tpu=data.get("tpu"),
            job_tags=data.get("job_tags", ""),
            log_dir=data.get("log_dir", ""),
            stage_dir=data.get("stage_dir", ""),
            extra_configs=data.get("extra_configs", ""),
            status=data.get("status"),
            error=data.get("error"),
            stage=data.get("stage", 0),
            monitor=data.get("monitor", True),
            rules=data.get("rules"),
            extra_msgs=data.get("extra_msgs", {}),
            customized_settings=data.get("customized_settings", {}),
            start_time=data.get("start_time", {})
        )



def check_rules():
    print(f"AVAILABLE RULES:")
    for rule in RULE_DICT:
        print(f"-> {rule}:".ljust(13) + f"{RULE_DICT[rule]}")

def parse_args_resume_rerun(args):
    """
    Parse the arguments for resume and rerun commands.
    """
    windows_id = None
    new_tpu = None
    for arg in args:
        if arg.startswith('tpu=') or arg.startswith('ka='):
            new_tpu = arg.split('=')[1]
        if arg.startswith('window=') or arg.startswith('-w='):
            windows_id = arg.split('=')[1]
    if windows_id is None:
        print(f"{FAIL} No window id provided")
        return None, None
    if not is_integer(windows_id):
        print(f"{FAIL} Window id {windows_id} is not an integer")
        return None, None
    return windows_id, new_tpu

def resume(user_obj, args):
    # Check if the window is in the job data, if it is, then resume the job
    windows_id, new_tpu = parse_args_resume_rerun(args)

    data = read_data()
    for user in data['users']:
        if data['users'][user]['tmux_name'] == user_obj.tmux_name:
            for job in data['users'][user]['job_data']:
                if str(job['windows_id']) == str(windows_id):
                    print(f"{INFO} Resuming job {windows_id} for user {user}")
                    # check the status of the job
                    resume_rerun_job(job, new_tpu, load_ckpt=True)
                    return
    else:
        print(f"{FAIL} resume: Job {windows_id} not found")
        return

def ignore_error(user_obj, args):
    """
    change the status of the job to 'running', and error to None.
    """
    window_id = args[0]
    if not is_integer(window_id):
        print(f"{FAIL} ignore_error: Window id {window_id} is not an integer")
        return
    data = read_and_lock_data()
    try:
        for user in data['users']:
            if data['users'][user]['tmux_name'] == user_obj.tmux_name:
                for job in data['users'][user]['job_data']:
                    if str(job['windows_id']) == str(window_id):
                        print(f"{INFO} ignore_error: Ignoring error for job {window_id} for user {user}")
                        job['status'] = 'running'
                        job['error'] = None
                        job['extra_msgs'].update({'ignore_error': True})
                        break
                else:
                    print(f"{FAIL} ignore_error: Job {window_id} not found")
                    return
                break
        else:
            print(f"{FAIL} ignore_error: User {user_obj.name} not found")
            return
        write_and_unlock_data(data)
    except Exception as e:
        print(f"{FAIL} ignore_error: Failed to ignore error for job {window_id} for user {user_obj.name}, error: {e}")
        release_lock_data()

def rerun(user_obj, args):
    # Check if the window is in the job data, if it is, then rerun the job
    windows_id, new_tpu = parse_args_resume_rerun(args)

    data = read_data()
    for user in data['users']:
        if data['users'][user]['tmux_name'] == user_obj.tmux_name:
            for job in data['users'][user]['job_data']:
                if str(job['windows_id']) == str(windows_id):
                    print(f"{INFO} Rerunning job {windows_id} for user {user}")
                    # check the status of the job
                    resume_rerun_job(job, new_tpu, load_ckpt=False)
                    return
    else:
        print(f"{FAIL} rerun: Job {windows_id} not found")
        return

def resume_rerun_job(job, new_tpu = None, load_ckpt = True):
    """
    Resume/Rerun a job in the tmux session.
    If load_ckpt is True, it will resume the job from the checkpoint.
    If load_ckpt is False, it will rerun the job from the beginning.
    """
    operation = 'resume' if load_ckpt else 'rerun'
    operationing = 'Resuming' if load_ckpt else 'Rerunning'
    if new_tpu is not None:
        print(f"{INFO} {operation}_job: Using new tpu {new_tpu}")
        zone, _, _, new_tpu = get_zone_pre_spot(new_tpu)
        if zone is None:
            print(f"{FAIL} {operation}_job: No zone found for tpu {new_tpu}")
            return
    data = read_and_lock_data()
    try:
        user = data['users'][job["user"]]
        user_obj = users.user_from_dict(user)
        new_stage = int(job['stage']) + 1 if load_ckpt else 0
        print(f"{INFO} {operation}_job: {operationing} job {job['windows_id']} for user {user_obj.name} with new stage {new_stage}")
        if new_stage > 12:
            print(f"{FAIL} {operation}_job: job {job['windows_id']} for user {user_obj.name} has reached max stage, cannot {operation}")
            release_lock_data()
            return
        id = user_obj.windows_offset
        data['users'][user_obj.name]['windows_offset'] = id + 1
        new_job = {
            'user': user_obj.name,
            'windows_id': id,
            'job_dir_id': job["job_dir_id"],
            'job_dir': job["job_dir"],
            'tpu': job["tpu"] if new_tpu is None else new_tpu,
            'job_tags': job["job_tags"],
            'log_dir': None,
            'stage_dir': None,
            'extra_configs': job["extra_configs"],
            'status': None,
            'stage': new_stage,
            'monitor': job["monitor"],
            'rules': job["rules"],
            'error': None,
            'extra_msgs': job["extra_msgs"] | {"father": job["windows_id"]},
            'customized_settings': job.get("customized_settings", {}),
        }

        # remove the key 'child' from the extra_msgs
        if 'child' in new_job['extra_msgs']:
            del new_job['extra_msgs']['child']

        # remove the key fail_time* from the extra_msgs
        for key in list(new_job['extra_msgs'].keys()):
            if key.startswith('fail_time'):
                del new_job['extra_msgs'][key]

        load_ckpt_path = ""

        if load_ckpt:
            assert job["log_dir"] is not None, f"Job {job['windows_id']} for user {user_obj.name} has no log dir"
            if not check_gs_logdir_exists(job["log_dir"]):
                print(f"{WARNING} {operation}_job: Log dir {job['log_dir']} does not exist, rerun instead")
            else:
                load_ckpt_path = job["log_dir"]

        data['users'][user_obj.name]['job_data'].append(new_job)
        user_obj.windows_offset = id + 1
        data['users'][user_obj.name] = user_obj.to_dict()
        # find the current job in the job_data list and set its status to 'resumed'
        for jb in data["users"][user_obj.name]["job_data"]:
            if jb["windows_id"] == job["windows_id"]:
                jb["status"] = 'resumed' if load_ckpt else 'rerunned'
                jb["extra_msgs"].update({"child": id})
        
        session_name = user_obj.tmux_name
        tpu = job["tpu"] if new_tpu is None else new_tpu
        config_args = job["extra_configs"]
        tags = job["job_tags"]
        stage_dir = job["stage_dir"]
        assert stage_dir is not None, f"Job {job['windows_id']} for user {user_obj.name} has no stage dir"
        log_dir = job["log_dir"]
        print(f"{INFO} {operation} job {job['windows_id']} for user {user_obj.name} with new windows id {id}")

        # make sure that the tpu is ready
        if tpu is not None:
            tpu_status = check_tpu_status(tpu)
            if tpu_status == 'preempted':
                print(f"{WARNING} {operation}_job: TPU {tpu} is preempted, trying to reapply...")
                res = apply_and_set_env(tpu, preemptible=True, delete=True)
                if res == 'success':
                    print(f"{GOOD} {operation}_job: Reapply TPU {tpu} done")
                else:
                    print(f"{FAIL} {operation}_job: Reapply TPU {tpu} failed, please contact the admin")
                    release_lock_data()
                    return
            elif tpu_status == 'failed':
                print(f"{FAIL} {operation}_job: Failed to query status")
                print(f"{FAIL} {operation}_job: This may indicate that this TPU is deleted, please contact the admin")
                
            tpu_status = check_tpu_status(tpu)
            assert tpu_status == 'ready', f"TPU {tpu} is not ready, status: {tpu_status}"

        # kill the old job
        kill_jobs_tpu(tpu)

        # zhh: get zone for new script
        zone, _, _, _ = get_zone_pre_spot(tpu)

        # create the tmux window
        os.system(f"tmux new-window -t {session_name}:{id}")
        time.sleep(0.5)
        os.system(f"tmux send-keys -t {session_name}:{id} 'cd {stage_dir}' Enter")
        if load_ckpt_path:
            if job.get("customized_settings", {}).get("log_stage", False):
                os.system(f"tmux send-keys -t {session_name}:{id} 'source staging.sh ka={tpu} zone={zone} {config_args} --config.load_from={load_ckpt_path} --config.stage={new_stage}' Enter")
                new_job['extra_configs'] += f" --config.load_from={load_ckpt_path} --config.stage={new_stage}"
            else:
                os.system(f"tmux send-keys -t {session_name}:{id} 'source staging.sh ka={tpu} zone={zone} {config_args} --config.load_from={load_ckpt_path}' Enter") 
                new_job['extra_configs'] += f" --config.load_from={load_ckpt_path}"
        else:
            os.system(f"tmux send-keys -t {session_name}:{id} 'source staging.sh ka={tpu} zone={zone} {config_args}' Enter")
        
        print(f"{GOOD} {operation}_job: Successfully created job in tmux window {session_name}:{id}")

        print(f"{INFO} {operation}_job: new job {new_job}")

        # update spreadsheet info
        

        write_and_unlock_data(data)
        spreadsheet_notes = new_job.get("extra_msgs", {}).get("spreadsheet_notes", None)
        tpu_info = get_tpu_info_sheet(tpu)
        tpu_info['running_status'] = 'running'
        tpu_info['user'] = user_obj.spreadsheet_name
        tpu_info['user_note'] = spreadsheet_notes
        print(f"{INFO} {operation}_job: Updating spreadsheet info: {tpu_info}")
        write_sheet_info(tpu_info)


    except Exception as e:
        print(f"{FAIL} {operation}_job: Failed to {operation} job {job['windows_id']} for user {user_obj.name}, error: {e}")
        release_lock_data()

    except KeyboardInterrupt:
        print(f"{INFO} {operation}_job: Stopping {operation}...")
        release_lock_data()
        return

def kill_job_or_tpu(user_obj, args):
    """
    Kill a job in the tmux session with specified window id. Need to acquire the lock.
    """
    windows_id = None
    data = read_data()
    all_tpu_list = []
    for alias, tpu_name in data['tpu_aliases'].items():
        all_tpu_list.append(alias)
        all_tpu_list.append(tpu_name)
    
    for arg in args:
        if arg.startswith('window=') or arg.startswith('-w=') or arg.startswith('w='):
            windows_id = arg.split('=')[1]
        if is_integer(arg):
            windows_id = arg
        if arg in all_tpu_list:
            print(f"{INFO} kill_job: Killing all jobs using tpu {arg}")
            kill_jobs_tpu(arg, username=user_obj.name)
            return
        
    if windows_id is None:
        print(f"{FAIL} kill_job:No window id provided")
        return
    if not is_integer(windows_id):
        print(f"{FAIL} kill_job: Window id {windows_id} is not an integer")
        return
    # mark the associated job as killed, and kill the job in the tmux session
    data = read_and_lock_data()
    try:
        for user in data['users']:
            if data['users'][user]['tmux_name'] == user_obj.tmux_name:
                for job in data['users'][user]['job_data']:
                    if str(job['windows_id']) == str(windows_id):
                        print(f"{INFO} kill_job: Killing job {windows_id} for user {user}")
                        # check the status of the job
                        job['status'] = 'killed'
                        # send Ctrl+C to the tmux window
                        session_name = user_obj.tmux_name
                        os.system(f"tmux send-keys -t {session_name}:{windows_id} C-c")
                        time.sleep(0.5)
                        # kill the tmux window
                        os.system(f"tmux kill-window -t {session_name}:{windows_id}")
                        
                        break
                break
        else:
            print(f"{FAIL} kill_job: Job {windows_id} not found")
            return
        write_and_unlock_data(data)
    except Exception as e:
        print(f"{FAIL} kill_job: Failed to kill job {windows_id} for user {user_obj.name}, error: {e}")
        release_lock_data()


def restart_run(user_obj, args):
    tpu = args[0]
    zone, _, _, tpu = get_zone_pre_spot(tpu)
    if zone is None:
        print(f"{FAIL} No zone found for tpu {tpu}")
        return
    try:
        restart(tpu)
    except Exception as e:
        print(f"{FAIL} Failed to restart TPU {tpu}: {e}")
        return
    except KeyboardInterrupt:
        print(f"{INFO} Stopping restart...")
        return
    print(f"{GOOD} Restarted TPU {tpu} successfully")
    run(user_obj, args)

    # read config args

def select_tpu(args, auto = False):
    if not auto:
        print(f"{INFO} select_tpu: Querying all available tpus...")
        find_tpu_from_type(args)
        tpu = input('Please select a tpu, or `q` to quit:')
        if tpu == 'q':
            print(f"{INFO} select_tpu: Quitting...")
            return None
        return tpu
    else:
        print(f"{INFO} select_tpu: Auto selecting tpu...")
        tpu_info = read_tpu_info_from_type(args)
        free_tpu_list = []
        reserved_tpu_list = []
        for tpu, info in tpu_info.items():
            if info['running_status'] == 'free':
                free_tpu_list.append(tpu)
            elif info['running_status'] == 'reserved':
                reserved_tpu_list.append(tpu)
        if len(free_tpu_list) > 0:
            print(f"{INFO} select_tpu: Found free tpus: {free_tpu_list}")
            print(f"{INFO} select_tpu: selecting free tpu {GREEN}{free_tpu_list[0]}{NC}")
            return free_tpu_list[0]
        else:
            print(f"{INFO} select_tpu: {RED}No free tpus found{NC}")
            if len(reserved_tpu_list) > 0:
                print(f"Found {YELLOW}reserved tpus{NC}:")
                for id, tpu in enumerate(reserved_tpu_list):
                    print(f"{YELLOW}T{id}({tpu_info[tpu]['alias']}){NC} -> {tpu_info[tpu]['user']}: {tpu_info[tpu]['user_note']}")
                print("Do you want to use one of them? (T?/n)")
                tpu_selected = input()
                if tpu_selected.startswith('T') and tpu_selected[1:].isdigit():
                    tpu_selected = int(tpu_selected[1:])
                    if tpu_selected < 0 or tpu_selected >= len(reserved_tpu_list):
                        print(f"{FAIL} select_tpu: Invalid tpu selected")
                        return None
                    tpu_selected = reserved_tpu_list[tpu_selected]
                    zone, _, _, tpu_selected = get_zone_pre_spot(tpu_selected)
                    if zone is None:
                        print(f"{FAIL} select_tpu: No zone found for tpu {tpu_selected}")
                        return None
                    print(f"{INFO} select_tpu: Using reserved tpu {tpu_selected}")
                    return tpu_selected
                elif tpu_selected.lower() == 'n':
                    print(f"{INFO} select_tpu: Not using reserved tpus")
                    return None
                else:
                    print(f"{FAIL} select_tpu: Invalid input, please enter T? or n")
                    return None
            else:
                print(f"{FAIL} select_tpu: No free or reserved tpus found")
                return None


def parse_config_args(user_obj, args):
    """
    Parse the config args from the command line arguments to use in `run`.
    Return: dir_id, dir_path, tpu, tag, rule, monitor, config_args, customized_settings, spreadsheet_notes
    """
    data = read_data()

    config_args = ""
    tag, rule, tpu = None, None, None
    monitor = True
    ignore_keys = ['dir', 'user', 'id', 'tag', 'rule', 'monitor', 'ssn']
    customized_settings = {}
    dir_id = '1'
    spreadsheet_notes = None
    all_tpu_list = []
    for alias, tpu_name in data['tpu_aliases'].items():
        all_tpu_list.append(alias)
        all_tpu_list.append(tpu_name)

    for arg in args:
        if '=' in arg:
            key, value = arg.split('=')[0], arg.split('=')[1]
            if key not in ignore_keys:
                if key in user_obj.config_aliases:
                    config_args += f" --{user_obj.config_aliases[key]}={value}"
                else:
                    assert key.startswith('config'), f"Unknown config key {key}"
                    config_args += f" --{key}={value}"
            if key == 'tag':
                tag = value
            if key == 'rule':
                rule = value
            if key == 'monitor':
                if value == 'False' or value == '0' or value == 'false':
                    monitor = False
                elif value == 'True' or value == '1' or value == 'true':
                    monitor = True
                else:
                    raise ValueError(f"Value {value} is not a valid boolean")
            if key == 'dir':
                dir_id = value
            if key == 'ssn':
                spreadsheet_notes = value
                
        if arg == '--log-stage':
            customized_settings['log_stage'] = True

        if arg in all_tpu_list:
            tpu = data['tpu_aliases'].get(arg, arg)
            print(f"{INFO} run: Using tpu {tpu}")

        if arg in ARG_TO_LIST:
            tpu = select_tpu(args, auto = ('-auto' in args or '--auto' in args))

            if tpu is None:
                print(f"{FAIL} run: No tpu selected")
                raise ValueError(f"TPU {tpu} not found")
            
            tpu = data['tpu_aliases'].get(tpu, tpu)
            print(f"{INFO} run: Using tpu {tpu}")

        if is_integer(arg):
            dir_id = arg
            if dir_id not in user_obj.working_dir:
                print(f"{FAIL} run: Directory id {dir_id} not found")
                raise ValueError(f"Directory id {dir_id} not found")
            print(f"{INFO} run: Using directory id {dir_id}")
        

    dir_path = user_obj.working_dir[dir_id]

    if not os.path.exists(dir_path):
        raise ValueError(f"Directory {dir_path} does not exist")

    if spreadsheet_notes is None:
        print(f"{INFO} run: Getting notes from wandb config file...")
        spreadsheet_notes = get_wandb_notes(dir_path)
        print(f"{INFO} run: Notes from wandb config file: {spreadsheet_notes}")

    if '-ssn' in args or '--ssn' in args:
        if spreadsheet_notes is not None:
            print(f"{WARNING} run: Notes already set to {spreadsheet_notes}, do you want to change it? (y/n)")
            res = input()
            if res == 'y' or res == 'Y':
                print("Please Enter the notes for the job:")
                notes = input()
                spreadsheet_notes = notes
        else:
            print("Please Enter the notes for the job:")
            notes = input()
            spreadsheet_notes = notes 

    if (tag is None) and (spreadsheet_notes is not None) and ('-no-tag' not in args):
        tag = spreadsheet_notes


    if tpu is None:
        print(f'{INFO} run: No TPU Specified, use the TPU in ka.sh instead')

    _, pre, spot, tpu = get_zone_pre_spot(tpu)
    preemptible = pre or spot

    if rule is None:
        rule = 'pass' if not preemptible else 'pre'
        
    if rule not in RULE_DICT:
        print(f"Rule {rule} is not valid.")
        rule = 'pass' if not preemptible else 'pre'
        print(f"Using rule {rule} instead")

    return dir_id, dir_path, tpu, tag, rule, monitor, config_args, customized_settings, spreadsheet_notes  

def run(user_obj, args, monitor_job = True):
    # print('args:', args)
    data = read_data()
    user_obj = users.user_from_dict(data['users'][user_obj.name])
    
    dir_id, dir_path, tpu, tag, rule, monitor, config_args, customized_settings, spreadsheet_notes = parse_config_args(user_obj, args)

    # Check the status of the TPU, and reapply if needed
    if tpu is not None:
        print(f"{INFO} Checking the status of TPU {tpu}...")
        tpu_status = check_tpu_status(tpu)

        if tpu_status == 'preempted':
            print(f"{WARNING} run: TPU {tpu} is preempted")
            REAPPLY = False
            if '-apply' in args:
                print(f"{INFO} run: Re-applying preempted TPU {tpu}...")
                REAPPLY = True
            else:
                print(f"DO YOU WANT TO REAPPLY? (y/n)")
                res = input()
                if res == 'y' or res == 'Y':
                    print(f"{INFO} run: Re-applying preempted TPU {tpu}...")
                    REAPPLY = True
                else:
                    print(f"{INFO} run: Quiting... {tpu}")
                    REAPPLY = False
            if not REAPPLY: return
            try:
                apply_and_set_env(tpu, preemptible=True, delete=True)
            except Exception as e:
                print(f"{FAIL} run: Failed to reapply TPU {tpu}: {e}")
                return
            except KeyboardInterrupt:
                print(f"{INFO} run: Stopping reapply...")
                return
            print(f"{GOOD} run: Re-applying TPU {tpu} successfully")

        elif tpu_status == 'ready':
            print(f"{GOOD} run: TPU {tpu} is ready, starting job...")

        elif tpu_status == 'failed':
            print(f"{WARNING} run: Failed to query status")
            print(f"This may indicate that this TPU is deleted, do you want to apply? (y/n)")
            res = input()
            if res == 'y' or res == 'Y':
                print(f"{INFO} run: Re-applying TPU {tpu}...")
                try: apply_and_set_env(tpu, preemptible=True, delete=False)
                except Exception as e:
                    print(f"{FAIL} run: Failed to reapply TPU {tpu}: {e}")
                    return
                except KeyboardInterrupt:
                    print(f"{INFO} run: Stopping reapply...")
                    return
                print(f"{GOOD} run: Applying TPU {tpu} successfully")
            else:
                print(f"{INFO} run: Quiting... {tpu}")
                return

        elif tpu_status == 'restarting' or tpu_status == 'creating' or tpu_status == 'stopping':
            print(f"{WARNING} run: TPU {tpu} is {tpu_status.lower()}")
            print(f"{INFO} run: Quiting... {tpu}")
            return

        else:
            print(f"{WARNING} run: TPU {tpu} is in unknown state {tpu_status}")
            print(f"{INFO} run: Quiting... {tpu}")
            return

    # Check the spreadsheet for the TPU information
    print(f"{INFO} run: Checking the TPU information in the spreadsheet...")
    tpu_info = get_tpu_info_sheet(tpu)
    running_status, running_user, notes = tpu_info['running_status'], tpu_info['user'], tpu_info['user_note']
    if running_user != user_obj.spreadsheet_name and (running_status == 'running' or running_status == 'reserved') and (not '--auto' in args):
        print(f"{WARNING} run: TPU {tpu} is already {RED}{running_status}{NC} by {running_user} in the spreadsheet")
        print(f"{WARNING} run: Notes: {notes}")
        print("DO YOU WANT TO CONTINUE? (y/n)")
        res = input()
        if res != 'y' and res != 'Y':
            print(f"{INFO} run: Quiting... {tpu}")
            return
        print(f"{INFO} run: Continuing...")
    else:
        if running_status == 'free':
            print(f"{GOOD} run: TPU {tpu} is free in the spreadsheet")
        elif running_status == 'reserved':
            print(f"{GOOD} run: TPU {tpu} is reserved (by yourself) in the spreadsheet")
        elif running_status == 'running':
            print(f"{WARNING} run: TPU {tpu} status is {YELLOW}running{NC} (by yourself) in the spreadsheet, notes: {notes}")

    print(f"{INFO} run: Changing the status to running by {user_obj.spreadsheet_name} in the spreadsheet...")
    tpu_info['running_status'] = 'running'
    tpu_info['user'] = user_obj.spreadsheet_name
    if spreadsheet_notes is not None:
        tpu_info['user_note'] = spreadsheet_notes
    else:
        tpu_info['user_note'] = ''
    write_sheet_info(tpu_info)
    # print(f"{GOOD} run: TPU {tpu} information updated in the spreadsheet")

    # Check if there are jobs running in the tpu
    if tpu is not None:
        running = check_tpu_running(tpu)

        print(f"{INFO} run: TPU {tpu} is {running}")

        if running != 'free':
            print(f"{WARNING} run: TPU {tpu} is not free, do you want to continue? (y/n)")
            res = input()
            if res != 'y' and res != 'Y':
                print(f"{INFO} run: Quiting... {tpu}")
                return
            
        for user in data['users']:
            for job in data['users'][user]['job_data']:
                if job['tpu'] == tpu and job['status'] == 'running':
                    print(f"{WARNING} There is a job using tpu {tpu}(maybe dead), by user {user}")
                    res = 'y'
                    if '-f' not in args:
                        print(f"DO YOU WANT TO CONTINUE? (y/n)")
                        res = input()
                    if res != 'y' and res != 'Y':
                        print("Exiting...")
                        return
                    
                    # mark the jobs in this tpu as killed
                    data = read_and_lock_data()
                    for user_ in data['users']:
                        if data['users'][user_]['tmux_name'] == user_obj.tmux_name:
                            for job_ in data['users'][user_]['job_data']:
                                if str(job_['windows_id']) == str(job['windows_id']):
                                    job_['status'] = 'killed'
                                    break
                            break
                    write_and_unlock_data(data)
        
    try:
        kill_jobs_tpu(tpu)
        data = read_and_lock_data()

        session_name = user_obj.tmux_name

        window_id = data['users'][user_obj.name]['windows_offset']
        data['users'][user_obj.name]['windows_offset'] = window_id + 1
        data['users'][user_obj.name]['job_data'].append({
            'user': user_obj.name,
            'windows_id': window_id,
            'job_dir_id': dir_id,
            'job_dir': dir_path,
            'tpu': tpu,
            'job_tags': tag,
            'log_dir': None,
            'stage_dir': None,
            'extra_configs': config_args,
            'status': 'starting',
            'error': None,
            'stage': 0,
            'monitor': monitor,
            'rules': copy.deepcopy(RULE_DICT[rule]),
            'extra_msgs': {
                "spreadsheet_notes": spreadsheet_notes
            },
            'customized_settings': customized_settings
        })
        # make sure that the tpu is ready
        if tpu is not None:
            tpu_status = check_tpu_status(tpu)
            assert tpu_status == 'ready', f"TPU {tpu} is not ready, status: {tpu_status}"

        # create the tmux window
        os.system(f"tmux new-window -t {session_name}:{window_id}")
        print(f"{INFO} run: Creating job in tmux window {session_name}:{window_id}")
        time.sleep(2)
        # check if the window<id> created
        windows = os.popen(f"tmux list-windows -t {session_name}").read().splitlines()
        found_window = False
        for window in windows:
            if window.startswith(f"{window_id}:"):
                found_window = True
                break
        if not found_window:
            print(f"{FAIL} run: Failed to create job in tmux window {session_name}:{window_id}")
            print(f"{FAIL} run: Window {window_id} not found")
            print(f"{FAIL} run: This may indicate that this window is already created, please check the tmux session")
            release_lock_data()
            return
        os.system(f"tmux send-keys -t {session_name}:{window_id} 'cd {dir_path}' Enter")
        if tpu is None:
            raise RuntimeError('zhh does not how to handle this case')
            os.system(f"tmux send-keys -t {session_name}:{window_id} 'source staging.sh {config_args}' Enter")
        else:
            zone, _, _, _ = get_zone_pre_spot(tpu)
            os.system(f"tmux send-keys -t {session_name}:{window_id} 'source staging.sh ka={tpu} zone={zone} {config_args}' Enter") 
        
        print(f"{GOOD} run: Successfully created job in tmux window {session_name}:{window_id}")

        time.sleep(4)

        write_and_unlock_data(data)
    
    except KeyboardInterrupt:
        print(f"\n{INFO} run: Stopping job creation...")
        release_lock_data()
        return

    except BaseException as e:
        print(f"{FAIL} run: Failed to create job in tmux window")
        print(f"Error: {e}")
        release_lock_data()

    time.sleep(3)

    if user_obj.settings['monitor_after_run'] and '-q' not in args and monitor_job:
        monitor_jobs(user_obj, args)

def check_all_jobs(args):
    """
    check the jobs for all the users
    """
    config = '-wts'
    print(f"args: {args}")
    for arg in args:
        if arg.startswith('-') and args not in ['-f', '-q', '-apply']:
            config = arg
        
    data = read_data()
    try:
        for user in data['users']:
            user_obj = users.user_from_dict(data['users'][user])
            print(f"{YELLOW}==============={NC} User {user_obj.name} {YELLOW}==============={NC}")
            check_jobs(user_obj, [], config=config)
    except Exception as e:
        print(f"{RED}[Error] {NC} check_all_jobs: Failed to check jobs")
        print(f"Error: {e}")

def monitor_all_jobs(args):
    """
    monitor the jobs for all the users
    """
    try:
        while True:
            check_all_jobs()
            data = read_data()
            sleep_time = data["monitor_all_check_time"] if "monitor_all_check_time" in data else 20
            time.sleep(sleep_time)
            # clear the screen
            os.system('clear' if os.name == 'posix' else 'cls')
    except KeyboardInterrupt:
        print(f"\n{INFO} Stopping monitor...")
        return

def write_error_to_job(user_obj, job_data, error):
    """
    Write the error to the job data
    """
    data = read_and_lock_data()
    for user in data['users']:
        if data['users'][user]['tmux_name'] == user_obj.tmux_name:
            for job in data['users'][user]['job_data']:
                if str(job['windows_id']) == str(job_data['windows_id']):
                    job['status'] = 'error'
                    job['error'] = error
                    break
            break
    write_and_unlock_data(data)

def is_monitor_config(arg):
    """
    Check if the argument is a monitor config
    """
    if not arg.startswith('-') or len(arg) < 2:
        return False
    for c in arg[1:]:
        if c not in 'wstdv':
            return False
    return True

def check_jobs(user_obj, args, config = None):
    """
    Print the status of all the jobs in the tmux session.
    """
    for arg in args:
        if is_monitor_config(arg):
            config = arg
            break

    if config is None:
        config = 'ws'
        if user_obj.settings.get("monitor_dir", False):
            config += 'd'
        if user_obj.settings.get("monitor_tpu", False):
            config += 't'
        if user_obj.settings.get("monitor_verbose", False):
            config += 'v'
    # print(f'config: {config}')

    if '-nt' in args:
        config += 'T'

    session_name = user_obj.tmux_name
    windows = os.popen(f"tmux list-windows -t {session_name}").read().splitlines()
    for window in windows:
        window_id = window.split(':')[0]
        window_name = window.split(':')[1].split(' ')[0]
        job_data = None
        for job in user_obj.job_data:
            if str(job['windows_id']) == str(window_id):
                job_data = job
                break
        if job_data is None:
            if window_id != '0' and 'w' in config:
                print(f'Window {window_id} (NOT FOUND IN DATA)')
                print('-'*40)
            continue
        else:
            father_job = None
            try:
                father_job = job_data['extra_msgs']['father']
            except Exception as e:
                father_job = None
            if 'w' in config:
                tag_str = ''
                if job_data['job_tags'] is not None:
                    tag_str = f"tag:{job_data['job_tags']}"
                rerun_str = ''
                if father_job is not None:
                    if job_data['stage'] != '0' and job_data['stage'] != 0:
                        rerun_str = f"resume:{father_job}; stage:{job_data['stage']+1}"
                    else:
                        rerun_str = f"rerun:{father_job}; stage:{job_data['stage']+1}"
                
                # print('jzc:',rerun_str)
                if tag_str != '' and rerun_str != '':
                    print(f"Window {window_id} ({tag_str}; {rerun_str})")

                elif tag_str != '':
                    print(f"Window {window_id} ({tag_str})")

                elif rerun_str != '':
                    print(f"Window {window_id} ({rerun_str})")

                else:
                    print(f"Window {window_id}")

            if 'd' in config:
                print(f"DIR: {job_data['job_dir'].split('/')[-1]}")
            if 't' in config:
                print(f"TPU: {job_data['tpu'][10:]}")
        last_line = os.popen(f"tmux capture-pane -t {session_name}:{window_id} -p").read()
        last_line = last_line.rstrip()
        show_length = user_obj.settings['show_length']
        monitor_length = user_obj.settings['monitor_length']
        monitor_verbose = user_obj.settings['monitor_verbose']
        last_line_cut = last_line[-monitor_length:]
        msg = last_line_cut[-show_length:]
            
        if (job_data["status"] is not None) and ('s' in config):
            if job_data["status"] == 'starting':
                print(f"{WARNING} Don't have logdir yet")
            if job_data["status"] == 'error':
                if job_data["error"] == 'preempted': print(f"Status: {RED}Preempted{NC}")
                elif job_data["error"] == 'OOM': print(f"Status: {RED}OOM{NC}")
                else: print(f"Status: {RED}Error{NC}\nmsg: {msg}")
                print('-'*40)
                continue
            elif job_data["status"] == 'killed':
                print(f"Status: {YELLOW}Killed{NC}")
                if 'v' in config: print(f"msg: {msg}")
                print('-'*40)
                continue
            elif job_data["status"] == 'resumed' or job_data["status"] == 'rerunned':
                try: child = job_data['extra_msgs']['child']
                except Exception as e:
                    print(f"{RED}Failed to get child window id{NC}")
                    child = None
                print(f"Status: {YELLOW}{job_data['error']}{NC} ({job_data['status']} in {child})")
                if 'v' in config: print(f"msg: {msg}")
                print('-'*40)
                continue
            elif job_data["status"] == 'finished':
                print(f"Status: {GREEN}Finished{NC}")
                if 'v' in config: print(f"msg: {msg}")
                print('-'*40)
                continue
            elif job_data["status"] == 'running' or job_data["status"] == 'starting':
                if (re.search(r'Job failed', last_line_cut) or re.search(r'[eE]rror', last_line_cut) or re.search(r'FAIL', last_line_cut)) and 's' in config:
                    if re.search(r'Allocation type', last_line):
                        print(f"Status: {RED}OOM Error{NC}\nmsg: {msg}")
                        write_error_to_job(user_obj, job_data, 'OOM')
                    elif re.search(r'GRPC [Ee]rror', last_line):
                        print(f"Status: {RED}GRPC Error{NC}\nmsg: {msg}")
                        write_error_to_job(user_obj, job_data, 'grpc')
                        ack_MONITOR()
                    elif re.search(r'python: No such file or directory', last_line):
                        print(f"Status: {RED}File Error{NC}\nmsg: {msg}")
                        write_error_to_job(user_obj, job_data, 'file error')
                        ack_MONITOR()
                    elif re.search(r'DEADLINE_EXCEEDED', last_line):
                        print(f"Status: {RED}DEADLINE EXCEEDED{NC}\nmsg: {msg}")
                        write_error_to_job(user_obj, job_data, 'deadline exceeded')
                        ack_MONITOR()
                    else:
                        # print(f"{RED}last line cut: {last_line}{NC}")
                        print(f"Status: {RED}Unknown Error{NC}\nmsg: {msg}")
                        write_error_to_job(user_obj, job_data, 'unknown')
                        ack_MONITOR()
                elif (re.search(r'[cC]ompiling', last_line_cut) or re.search(r'[cC]ompilation', last_line_cut) or re.search(r'[cC]ompile', last_line_cut)) and 's' in config:
                    print(f"Status: {GREEN}Compiling{NC}")
                    if 'v' in config: print(f"msg: {msg}")
                elif re.search(r'[sS]ampling ', last_line_cut) and 's' in config:
                    epoch = None
                    if re.search(r'[eE]poch\s([0-9]{1,4})', last_line_cut):
                        epoch = re.search(r'[eE]poch\s([0-9]{1,6})', last_line_cut).group(1)
                    elif re.search(r'ep=([0-9]){1,4}\.([0-9]){1,6}', last_line_cut):
                        epoch = re.search(r'ep=([0-9]){1,4}\.([0-9]){1,6}', last_line_cut).group(0)[3:]
                    if epoch is not None:
                        print(f"Status: {GREEN}Sampling{NC} (in epoch {int(float(epoch))})")
                    else:
                        print(f"Status: {GREEN}Sampling{NC}")
                    if 'v' in config:
                        print(f"msg: {msg}")

                elif re.search(r'[eE]poch\s([0-9]{1,4})', last_line_cut) and ('s' in config):
                    epoch = re.search(r'[eE]poch\s([0-9]{1,6})', last_line_cut).group(1)
                    print(f"Status: {GREEN}Running{NC} (ep={epoch})")
                    if 'v' in config: print(f"msg: {msg}")
                elif re.search(r'ep\s*=\s*([0-9]){1,4}\.([0-9]){1,6}', last_line_cut) and ('s' in config):
                    epoch = re.search(r'ep\s*=\s*([0-9]){1,4}\.([0-9]){1,6}', last_line_cut).group(0).split('=')[1]
                    epoch = epoch.strip()
                    print(f"Status: {GREEN}Running{NC} (ep={float(epoch):.2f})")
                    if 'v' in config: print(f"msg: {msg}")
                elif re.search(r'[iI]nitializing', last_line_cut) and 's' in config:
                    print(f"Status: {GREEN}Initializing{NC}")
                    if 'v' in config: print(f"msg: {msg}")
                elif re.search(r'[sS]taging', last_line_cut) and 's' in config:
                    print(f"Status: {GREEN}Staging{NC}")
                    if 'v' in config: print(f"msg: {msg}")
                elif 's' in config:
                    print(f"Status: {YELLOW}Unknown{NC}\nmsg: {msg}")
                    # print(f"last line: {last_line_cut}")
                    # print(re.search(r'ep=([0-9]){1,4}\.([0-9]){1,6}', last_line_cut))
        print('-'*40)

def check_jobs_simp(user_obj, args, config = None, num_columns = 3):
    """
    Print the status of all the jobs in the tmux session.
    """

    num_columns = 3

    for arg in args:
        if is_monitor_config(arg):
            config = arg
            break

        if arg.startswith('col='):
            num_columns = int(arg.split('=')[1])
            continue
                    
    if config is None:
        config = 'wst'
    if '-nt' in args:
        config += 'T'

    session_name = user_obj.tmux_name
    job_blocks = []

    windows = os.popen(f"tmux list-windows -t {session_name}").read().splitlines()

    for window in windows:
        window_id = window.split(':')[0]
        window_name = window.split(':')[1].split(' ')[0]

        # find job_data by window id
        job_data = None
        for job in user_obj.job_data:
            if str(job.get('windows_id')) == str(window_id):
                job_data = job
                break

        # If job not found in data
        if job_data is None:
            if window_id != '0' and 'w' in config:
                # Make a minimal block for the 2-up view
                rows = [("Window", f"{window_id}"),
                        ("Info",   "NOT FOUND IN DATA")]
                job_blocks.append(_kv_rows_to_block(rows))
            # else: skip silently
            continue
        # ---------- Window meta rows (what you previously printed before status) ----------
        rows_meta = []
        if 'w' in config:
            rows_meta.append(("Window", str(window_id)))

            # Add rerun info if exists
            father_job = None
            try:
                father_job = job_data['extra_msgs']['father']
            except Exception:
                father_job = None

            if father_job is not None:
                if job_data.get('stage') not in ('0', 0):
                    rows_meta.append(("Resume", f"(father={father_job}, stage={job_data['stage']+1})"))
                else:
                    rows_meta.append(("Rerun", f"(father={father_job}, stage={job_data['stage']+1})"))

        if 'd' in config:
            try:
                rows_meta.append(("DIR", job_data['job_dir'].split('/')[-1]))
            except Exception:
                rows_meta.append(("DIR", "(unknown)"))
        if 't' in config:
            try:
                rows_meta.append(("TPU", job_data['tpu'][10:]))
            except Exception:
                rows_meta.append(("TPU", "(unknown)"))

        # ---------- Capture pane & compute msg ----------
        last_line = os.popen(f"tmux capture-pane -t {session_name}:{window_id} -p").read().rstrip()
        show_length = user_obj.settings['show_length']
        monitor_length = user_obj.settings['monitor_length']
        monitor_verbose = user_obj.settings['monitor_verbose']  # kept for parity; not directly used here
        last_line_cut = last_line[-monitor_length:]
        msg = last_line_cut[-show_length:]

        # ---------- Status rows via your helper (does side-effects like write_error_to_job/ack) ----------
        # Assumes your helper implements suppression via _kv_rows_to_block/_suppress_preview:
        rows_status = _render_rows_for_job(job_data, msg, last_line, last_line_cut, config, user_obj)

        # If there was no status to show (e.g., 's' not in config), still add meta-only block if any meta exists
        combined_rows = rows_meta + rows_status
        if combined_rows:
            block = _kv_rows_to_block(combined_rows)  # applies per-column preview + "..(+N)"
            job_blocks.append(block)
    # ---------- Print two columns for two jobs ----------
    # Renders blocks 2-up, with a separator after each row of two
    if job_blocks:
        _print_in_columns(job_blocks, num_columns=num_columns)




def kill_window(user_obj, args):
    data = read_and_lock_data()
    try:
 
        window_num = args[0]
        if not is_integer(window_num):
            raise ValueError(f"Window number {window_num} is not an integer")
        window_num = int(window_num)
        if window_num < 0:
            raise ValueError(f"Window number {window_num} is not valid")

        # Kill tmux window
        session_name = user_obj.tmux_name
        print(f"Killing window {window_num} in session {session_name}")
        os.system(f"tmux kill-window -t {session_name}:{window_num}")
        time.sleep(0.5)

        # Remove job from job data safely
        all_jobs = user_obj.job_data
        new_jobs = [job for job in all_jobs if str(job.get('windows_id')) != str(window_num)]
        if len(new_jobs) < len(all_jobs):
            print(f"{INFO} Removed job with window_id {window_num}")
        else:
            print(f"{WARNING} No job found with window_id {window_num}")

        data['users'][user_obj.name]['job_data'] = new_jobs
        write_and_unlock_data(data)
    except BaseException as e:
        print(f"{RED}[Error] {NC} kill_window: Failed to kill window {window_num} in session {session_name}")
        print(f"Error: {e}")
        release_lock_data()


def run_job_on_tpu(job: Job, tpu, quiet = True, ignore_window = None):
    data = read_and_lock_data()
    try:
        # update logs
        user = job.user
        user_obj = users.user_from_dict(data['users'][user])
        window_id = user_obj.windows_offset
        data['users'][user_obj.name]['windows_offset'] = window_id + 1
        user_obj.windows_offset = window_id + 1
        zone, pre, spot, tpu = get_zone_pre_spot(tpu)
        if not job.rules:
            job.rules = RULE_DICT["pre"] if pre else RULE_DICT["pass"]
        
        job.windows_id = window_id
        job.tpu = tpu
        data['users'][user_obj.name]['job_data'].append(job.to_dict())

        # sanity check
        session_name = user_obj.tmux_name
        assert job.stage_dir is not None, f"run_job_on_tpu: Job don't have stagedir"

        tpu_status = check_tpu_status(tpu)
        assert tpu_status == 'ready', f"run_job_on_tpu: TPU {tpu} is not ready, status: {tpu_status}"

        kill_jobs_tpu(tpu, ignore_window=ignore_window)

        # run the job
        os.system(f"tmux new-window -t {session_name}:{window_id}")
        time.sleep(0.5)
        os.system(f"tmux send-keys -t {session_name}:{window_id} 'cd {job.stage_dir}' Enter")
        os.system(f"tmux send-keys -t {session_name}:{window_id} 'source staging.sh ka={tpu} zone={zone} {job.extra_configs}' Enter")

        if not quiet:
            print(f"{GOOD} run_job_on_tpu: Successfully created job in tmux window {session_name}:{window_id}")

            print(f"{INFO} run_job_on_tpu: new job {job.to_dict()}")

        write_and_unlock_data(data)

        tpu_info = get_tpu_info_sheet(tpu)
        tpu_info['running_status'] = 'running'
        tpu_info['user'] = user_obj.spreadsheet_name
        tpu_info['user_note'] = job.job_tags
        write_sheet_info(tpu_info)

    except Exception as e:
        print(f"{FAIL} run_job_on_tpu: Failed to run job for user {user_obj.name}, error: {e}")

    except KeyboardInterrupt:
        print(f"{INFO} run_job_on_tpu: Stopping ...")

    finally:
        release_lock_data()

def monitor_jobs(user_obj, args):
    config = None
    num_columns = 3
    for arg in args:
        if arg.startswith('col='):
            num_columns = int(arg.split('=')[1])
            break
    if len(args) > 0:
        for arg in args:
            if is_monitor_config(arg):
                config = arg
                break
    if config is None:
        config = 'ws'
        if user_obj.settings.get("monitor_dir", False):
            config += 'd'
        if user_obj.settings.get("monitor_tpu", False):
            config += 't'
        if user_obj.settings.get("monitor_verbose", False):
            config += 'v'
    
    if '-nt' in args:
        config += 'T'
    try:
        while True:
            check_jobs_simp(user_obj, args, config=config, num_columns=num_columns)
            time.sleep(user_obj.settings['monitor_upd_time'])
            # clear the screen
            os.system('clear' if os.name == 'posix' else 'cls')
            # Update user object
            data = read_data()
            user_obj = data['users'][user_obj.name]
            user_obj = users.user_from_dict(user_obj)
    except KeyboardInterrupt:
        print(f"\n{INFO} Stopping monitor...")
        return
    


def upd_log(window, log_dir, stage_dir, ka, start_time):
    data = read_and_lock_data()
    try:
        session_name, window_num = window.split(':')
        window_num = int(window_num)
        print(f"Updating log dir to {log_dir} for window {window_num} in session {session_name}")
        print(f"Updating ka to {ka}")
        # find the job in the job data
        for user in data['users']:
            if data['users'][user]['tmux_name'] == session_name:
                for job in data['users'][user]['job_data']:
                    if job['windows_id'] == window_num:
                        job['log_dir'] = log_dir
                        job['stage_dir'] = stage_dir
                        job['tpu'] = ka
                        job['start_time'] = {
                            'chn': get_chn_time_str(),
                            'edt': get_edt_time_str(),
                            'utc': get_abs_time_str()
                        }
                        job['status'] = 'running'
                        job['error'] = None
                        break
                break
        write_and_unlock_data(data)
    except:
        print(f"{RED}Error: Failed to update log data{NC}")
        release_lock_data()

def add_tag(user_object, job_window_id, tag):
    data = read_and_lock_data()
    try:
        if job_window_id.startswith('window=') or job_window_id.startswith('-w='):
            job_window_id = job_window_id.split('=')[1]
        for job in user_object.job_data:
            if job['windows_id'] == int(job_window_id):
                job['job_tags'] = tag
                data['users'][user_object.name]['job_data'] = user_object.job_data
                write_and_unlock_data(data)
                print(f"Set tag {tag} to window {job_window_id}")
                break
    except:
        print(f"{RED}Error: Failed to set tag {tag} to window {job_window_id}{NC}")
        release_lock_data()

def clear_finished_jobs(user_object):
    data = read_and_lock_data()
    try:
        print(f"{INFO} clear_finished_jobs: Clearing jobs...")
        all_jobs = user_object.job_data
        jobs_to_remove = []

        for job in all_jobs:
            if job['status'] == 'finished':
                print(f"{INFO} clear_finished_jobs: Clearing finished job {job['windows_id']}")
                os.system(f"tmux kill-window -t {user_object.tmux_name}:{job['windows_id']}")
                jobs_to_remove.append(job)

            elif job['status'] == 'resumed' or job['status'] == 'rerunned':
                cur_job = job
                resume_chain = [cur_job]
                try:
                    while cur_job['status'] == 'resumed' or cur_job['status'] == 'rerunned':
                        next_id = cur_job['extra_msgs']['child']
                        next_job = next(jb for jb in all_jobs if jb['windows_id'] == next_id)
                        resume_chain.append(next_job)
                        cur_job = next_job
                except (StopIteration, KeyError):
                    continue 

                if cur_job['status'] == 'finished':
                    for jb in resume_chain:
                        # print(f"{PURPLE}[DEBUG] {NC}clear_finished_jobs: Killing tmux window {user_object.tmux_name}:{jb['windows_id']}")
                        os.system(f"tmux kill-window -t {user_object.tmux_name}:{jb['windows_id']}")
                        jobs_to_remove.append(jb)

        new_jobs = [job for job in all_jobs if job not in jobs_to_remove]
        data['users'][user_object.name]['job_data'] = new_jobs
        write_and_unlock_data(data)
        # write all the deleted jobs to legacy, with is a list of jobs
        legacy = read_and_lock_legacy()
        legacy.extend(jobs_to_remove)
        write_and_unlock_legacy(legacy)

    except:
        print(f"{RED}[Error] {NC}clear_finished_jobs: Failed to clear finished jobs")
        release_lock_data()
        release_lock_legacy()


def clear_error_jobs(user_object, clear_rerun = False):
    data = read_and_lock_data()
    try:
        print(f"{INFO} clear_error_jobs: Clearing jobs...")
        all_jobs = user_object.job_data
        new_jobs = []

        for job in all_jobs:
            if job['status'] in ['error', 'killed']:
                print(f"{INFO} clear_error_jobs: Clearing error job {job['windows_id']}")
                # print(f"{PURPLE}[DEBUG] {NC}clear_error_jobs: Killing tmux window {user_object.tmux_name}:{job['windows_id']}")
                ret = os.system(f"tmux kill-window -t {user_object.tmux_name}:{job['windows_id']}")
                if ret != 0:
                    print(f"{WARNING} clear_error_jobs: Failed to kill tmux window {user_object.tmux_name}:{job['windows_id']}")
            elif job['status'] == 'resumed' or job['status'] == 'rerunned':
                if clear_rerun:
                    print(f"{INFO} clear_error_jobs: Clearing rerun job {job['windows_id']}")
                    # print(f"{PURPLE}[DEBUG] {NC}clear_error_jobs: Killing tmux window {user_object.tmux_name}:{job['windows_id']}")
                    ret = os.system(f"tmux kill-window -t {user_object.tmux_name}:{job['windows_id']}")
                    if ret != 0:
                        print(f"{WARNING} clear_error_jobs: Failed to kill tmux window {user_object.tmux_name}:{job['windows_id']}")
                    continue
                
                cur_job = job
                resume_chain = [cur_job]
                try:
                    while cur_job['status'] == 'resumed' or cur_job['status'] == 'rerunned':
                        next_id = cur_job['extra_msgs']['child']
                        next_job = next(jb for jb in all_jobs if jb['windows_id'] == next_id)
                        resume_chain.append(next_job)
                        cur_job = next_job
                except (StopIteration, KeyError):                    
                    for jb in resume_chain:
                        print(f"{INFO} clear_error_jobs: Clearing error job {jb['windows_id']}")
                        # print(f"{PURPLE}[DEBUG] {NC}clear_error_jobs: Killing tmux window {user_object.tmux_name}:{jb['windows_id']}")
                        ret = os.system(f"tmux kill-window -t {user_object.tmux_name}:{jb['windows_id']}")
                        if ret != 0:
                            print(f"{WARNING} clear_error_jobs: Failed to kill tmux window {user_object.tmux_name}:{jb['windows_id']}")
                    continue
                if cur_job['status'] in ['error', 'killed', 'resumed', 'rerunned']:
                    for jb in resume_chain:
                        print(f"{INFO} clear_error_jobs: Clearing error job {jb['windows_id']}")
                        # print(f"{PURPLE}[DEBUG] {NC}clear_error_jobs: Killing tmux window {user_object.tmux_name}:{jb['windows_id']}")
                        ret = os.system(f"tmux kill-window -t {user_object.tmux_name}:{jb['windows_id']}")
                        if ret != 0:
                            print(f"{WARNING} clear_error_jobs: Failed to kill tmux window {user_object.tmux_name}:{jb['windows_id']}")
                    continue
            else:
                new_jobs.append(job)

        data['users'][user_object.name]['job_data'] = new_jobs
        write_and_unlock_data(data)

        # write all the deleted jobs to LEGACY_PATH, with is a json file of list of jobs
        legacy = read_and_lock_legacy()
        for job in all_jobs:
            if job['status'] in ['error', 'killed'] or (clear_rerun and (job['status'] == 'resumed' or job['status'] == 'rerunned')):
                legacy.append(job)
        write_and_unlock_legacy(legacy)


    except:
        print(f"{RED}[Error] {NC}clear_error_jobs: Failed to clear error jobs")
        release_lock_data()
        release_lock_legacy()

def clear_all_jobs(user_object, args = None):
    clear_rerun = False
    if args is not None:
        if '-re' in args:
            clear_rerun = True
    try:
        clear_finished_jobs(user_object)
    except:
        print(f"{RED}[Error] {NC}clear_all_jobs: Failed to clear finished jobs")
    try:
        clear_error_jobs(user_object, clear_rerun=clear_rerun)
    except:
        print(f"{RED}[Error] {NC}clear_all_jobs: Failed to clear error jobs{NC}")

def clear_zombie_jobs(user_object):
    """
    clear jobs whose window number can't be found in tmux session
    """
    data = read_and_lock_data()
    try:
        print(f"{INFO} clear_zombie_jobs: Clearing zombie jobs...")
        all_jobs = user_object.job_data
        new_jobs = []
        all_windows = os.popen(f"tmux list-windows -t {user_object.tmux_name}").read().splitlines()
        all_windows = [int(w.split(':')[0]) for w in all_windows]
        for job in all_jobs:
            if int(job['windows_id']) not in all_windows:
                print(f"{INFO} clear_zombie_jobs: Clearing zombie job {job['windows_id']}")
            else:
                new_jobs.append(job)
        data['users'][user_object.name]['job_data'] = new_jobs
        write_and_unlock_data(data)
        legacy = read_and_lock_legacy()
        for job in all_jobs:
            if int(job['windows_id']) not in all_windows:
                legacy.append(job)
        write_and_unlock_legacy(legacy)

    except:
        print(f"{RED}[Error] {NC}clear_zombie_jobs: Failed to clear zombie jobs")
        release_lock_data()
        release_lock_legacy()

def ack_MONITOR():
    """
    Acknowledge the monitor command
    """
    data = read_and_lock_data()
    try:
        data["ack_MONITOR"] = True
        write_and_unlock_data(data)
        print(f"{GOOD} ack_MONITOR: Monitor acknowledged")
    except:
        print(f"{RED}[Error] {NC}ack_MONITOR: Failed to acknowledge monitor")
        release_lock_data()
