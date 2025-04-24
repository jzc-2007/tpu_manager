import json
import os, time
from .helpers import DATA_PATH, LOCK_FILE
from .data_io import read_and_lock_data, write_and_unlock_data, release_lock_data, read_data, write_data
from .operate import get_zone_pre, check_env, mount_disk, check_tpu_status, apply_pre
from .jobs import resume_rerun_job
RED, GREEN, YELLOW, PURPLE, NC = "\033[1;31m", "\033[1;32m", "\033[1;33m", "\033[1;34m", "\033[0m"
GOOD, INFO, WARNING, FAIL = f"{GREEN}[GOOD]{NC}", f"{PURPLE}[INFO]{NC}", f"{YELLOW}[WARNING]{NC}", f"{RED}[FAIL]{NC}"
def clear_zombie_windows(user_obj):
    """
    Clean up all the tmux windows that are not in user's joblist, except window 0.
    """
    try:
        output = os.popen(f'tmux list-windows -t {user_obj.tmux_name}').read().strip()
        windows = output.splitlines()
    except Exception as e:
        print(f"{FAIL} clear_zombie_windows: Failed to list tmux windows: {e}")
        return

    valid_windows = set(int(job['windows_id']) for job in user_obj.job_data)

    for window in windows:
        try:
            window_num = int(window.split(':')[0])
        except ValueError:
            continue
        if window_num == 0:
            continue
        if window_num not in valid_windows:
            print(f"{INFO} clear_zombie_windows: Killing window {user_obj.tmux_name}:{window_num}")
            os.system(f'tmux kill-window -t {user_obj.tmux_name}:{window_num}')
            time.sleep(0.5)

def solve_env(tpu):
    zone, pre, tpu = get_zone_pre(tpu)
    if zone is None:
        print(f"{FAIL} solve_env: TPU {tpu} not found")
        return 'failed'
    print(f"{INFO} solve_env: TPU {tpu} is in zone {zone}, preemptible: {pre}")
    print(f"{INFO} solve_env: Checking the status of TPU {tpu}...")
    tpu_status = check_tpu_status(tpu)
    if tpu_status == 'preempted':
        print(f"{INFO} solve_env: TPU {tpu} is preempted, trying to reapply...")
        res = apply_pre(tpu, delete=True)
        if res == 'success':
            print(f"{GOOD} solve_env: Reapply TPU {tpu} done")
            return 'success'
        else:
            print(f"{FAIL} solve_env: Reapply TPU {tpu} failed, please contact the admin")
    print(f"{INFO} solve_env: Trying to solve the environment in TPU {tpu}...")
    print(f"{INFO} solve_env: Checking the environment, this may take some time...")
    state = check_env(tpu)
    if state == 'success':
        print(f"{GOOD} solve_env: Environment in TPU {tpu} is good")
        return 'success'
    elif state == 'file error':
        print(f"{INFO} solve_env: Environment in TPU has file error, trying to mount disk...")
        res = mount_disk(tpu)
        if res == 'success':
            print(f"{GOOD} solve_env: Solving environment in TPU {tpu} done")
            return 'success'
        else:
            print(f"{FAIL} solve_env: Solving environment in TPU {tpu} failed, please contact the admin")
            return 'failed'
    elif state == 'unknown':
        print(f"{FAIL} solve_env: Environment in TPU {tpu} is unknown error, please contact the admin")
        return 'failed'
    
def initialization():
    """
    remove all the job data for all users
    """
    data = read_data()
    for user in data['users']:
        data['users'][user]['job_data'] = []
    write_data(data)
    # kill and recreate all the tmux sessions
    for user in data['users']:
        # kill the tmux session
        print(f"{INFO} initialization: Killing tmux session {data['users'][user]['tmux_name']}")
        os.system(f'tmux kill-session -t {data["users"][user]["tmux_name"]}')
        time.sleep(5)
        print(f"{INFO} initialization: Recreating tmux session {data['users'][user]['tmux_name']}")
        # create the tmux session
        os.system(f'tmux new-session -d -s {data["users"][user]["tmux_name"]}')
    # set windows offset to 1
    data = read_and_lock_data()
    for user in data['users']:
        data['users'][user]['windows_offset'] = 1
    write_and_unlock_data(data)

def change_ip():
    """
    Deal with ip changes of the machine.
    Restart all the tmux sessions, open the window number that the 
    jobs is of status 'running', and resume all of them.
    """
    data = read_data()
    for user in data['users']:
        # kill the tmux session and recreate it
        print(f"{INFO} change-ip: Killing tmux session {data['users'][user]['tmux_name']}")
        os.system(f'tmux kill-session -t {data["users"][user]["tmux_name"]}')
        time.sleep(5)
        print(f"{INFO} change-ip: Recreating tmux session {data['users'][user]['tmux_name']}")
        os.system(f'tmux new-session -d -s {data["users"][user]["tmux_name"]}')
        print(f"{GOOD} change-ip: Recreating tmux session {data['users'][user]['tmux_name']} done")
    
    print(f"{GOOD} change-ip: Recreating tmux sessions done")

    # open windows that has running jobs
    for user in data['users']:
        for job in data['users'][user]['job_data']:
            if job['status'] == 'running':
                os.system(f'tmux new-window -t {data["users"][user]["tmux_name"]}:{job["windows_id"]}')

    # resume all the jobs
    for user in data['users']:
        for job in data['users'][user]['job_data']:
            if job['status'] == 'running':
                print(f"{INFO} change-ip: Resuming job {job['job_name']} in windows{job['windows_id']} for user {user}")
                resume_rerun_job(job, load_ckpt=True)

    print(f"{GOOD} change-ip: Resuming all jobs done")
    print(f"{INFO} change-ip: Please remember to restart the monitor script")
