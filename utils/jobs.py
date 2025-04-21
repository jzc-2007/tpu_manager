import os, re, time, json
from .helpers import is_integer, DATA_PATH
from . import users
from .data_io import read_and_lock_data, write_and_unlock_data, release_lock_data, read_data
from .operate import describe_tpu, apply_pre
RED, GREEN, YELLOW, PURPLE, NC = "\033[1;31m", "\033[1;32m", "\033[1;33m", "\033[1;34m", "\033[0m"
RULE_DICT ={
    'pre':{
        'preempted': 'reapply',
        'grpc': 'rerun',
    },
    'rerun':{
        'preempted': 'pass',
        'grpc': 'rerun',
    },
    'pass':{
        'preempted': 'pass',
        'grpc': 'pass',
    }
}

def run(user_obj, args):
    data = read_and_lock_data()
    user_obj = users.user_from_dict(data['users'][user_obj.name])
    try:
        dir = '1'
        for arg in args:
            if arg.startswith('dir='):
                dir = arg.split('=')[1]
                break
        dir_path = user_obj.working_dir[dir]
        if not os.path.exists(dir_path):
            raise ValueError(f"Directory {dir_path} does not exist")
        # Get the tpu name
        tpu = None
        for arg in args:
            if arg in data['tpu_aliases']:
                tpu = data['tpu_aliases'][arg]
                print(f"Using tpu {tpu}")
                break
        if tpu is None:
            print('No TPU Specified, use the TPU in ka.sh instead')

        # Check the status of the TPU
        if tpu is not None:
            print(f"{PURPLE}[INFO] {NC}Checking the status of TPU {tpu}...")
            tpu_status = describe_tpu(tpu)

            if tpu_status == 'PREEMPTED':
                print(f"{YELLOW}[WARNING]{NC} TPU {tpu} is preempted")
                REAPPLY = False
                if '-apply' in args:
                    print(f"{PURPLE}[INFO] {NC}Re-applying preempted TPU {tpu}...")
                    REAPPLY = True
                else:
                    print(f"DO YOU WANT TO REAPPLY? (y/n)")
                    res = input()
                    if res == 'y' or res == 'Y':
                        print(f"{PURPLE}[INFO] {NC}Re-applying preempted TPU {tpu}...")
                        REAPPLY = True
                    else:
                        print(f"{PURPLE}[INFO] {NC}Quiting... {tpu}")
                        REAPPLY = False
                if not REAPPLY:
                    release_lock_data()
                    return
                else:
                    try:
                        apply_pre(tpu, delete=True)
                    except Exception as e:
                        print(f"{RED}[ERROR]{NC} Failed to reapply TPU {tpu}: {e}")
                        release_lock_data()
                        return
                    print(f"{GREEN}[INFO] {NC}Re-applying TPU {tpu} successfully")

            elif tpu_status == 'READY':
                print(f"{GREEN}[INFO] {NC}TPU {tpu} is ready, starting job...")

            elif tpu_status == 'failed':
                print(f"{YELLOW}[WARNING]{NC} Failed to query status")
                print(f"This may indicate that this TPU is deleted, do you want to apply? (y/n)")
                res = input()
                if res == 'y' or res == 'Y':
                    print(f"{PURPLE}[INFO] {NC}Re-applying TPU {tpu}...")
                    try:
                        apply_pre(tpu, delete=False)
                    except Exception as e:
                        print(f"{RED}[ERROR]{NC} Failed to reapply TPU {tpu}: {e}")
                        release_lock_data()
                        return
                    print(f"{GREEN}[INFO] {NC}Applying TPU {tpu} successfully")
                else:
                    print(f"{PURPLE}[INFO] {NC}Quiting... {tpu}")
                    release_lock_data()

            elif tpu_status == 'RESTARTING' or tpu_status == 'CREATING' or tpu_status == 'STOPPING':
                print(f"{YELLOW}[WARNING]{NC} TPU {tpu} is {tpu_status.lower()}")
                print(f"{PURPLE}[INFO] {NC}Quiting... {tpu}")
                release_lock_data()
                return

            else:
                print(f"{YELLOW}[WARNING]{NC} TPU {tpu} is in unknown state {tpu_status}")
                print(f"{PURPLE}[INFO] {NC}Quiting... {tpu}")
                release_lock_data()
                return




        # Check if there is job running using this tpu
        if tpu is not None:
            for user in data['users']:
                for job in data['users'][user]['job_data']:
                    if job['tpu'] == tpu and job['status'] == 'running':
                        print(f"{YELLOW}[WARNING]{NC} There is a job running using tpu {tpu}, by user {user}")
                        print(f"DO YOU WANT TO CONTINUE? (y/n)")
                        res = input()
                        if res != 'y' and res != 'Y':
                            print("Exiting...")
                            release_lock_data()
                            return
                        # change the status of this job to 'killed'
                        job['status'] = 'killed'
                        data['users'][user]['job_data'] = job

        config_args = ""
        tag, rule = None, None
        preemptible = tpu in data['all_tpus']['preemptible']
        monitor = True
        ignore_keys = ['dir', 'user', 'id', 'tag', 'rule', 'monitor']
        for arg in args:
            #check if contains '='
            if '=' in arg:
                key, value = arg.split('=')
                if key not in ignore_keys:
                    if key in user_obj.config_aliases:
                        config_args += f" --{user_obj.config_aliases[key]}={value}"
                    else:
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
        if rule is None:
            rule = 'pass' if not preemptible else 'pre'
        if rule not in RULE_DICT:
            print(f"Rule {rule} is not valid.")
            rule = 'pass' if not preemptible else 'pre'
            print(f"Using rule {rule} instead")
        
        rule = RULE_DICT[rule]
        

        # kill all the windows that uses the same tpu
        session_name = user_obj.tmux_name
        all_jobs = user_obj.job_data

        # Find a minimum id not in use
        id = user_obj.windows_offset
        data['users'][user_obj.name]['windows_offset'] = id + 1
        all_jobs.append({
            'user': user_obj.name,
            'windows_id': id,
            'job_dir_id': dir,
            'job_dir': dir_path,
            'tpu': tpu,
            'job_tags': tag,
            'log_dir': None,
            'extra_configs': config_args,
            'status': None,
            'error': None,
            'stage': 0,
            'monitor': monitor,
            'rules': rule,
            'extra_msgs': {},
        })
        data['users'][user_obj.name]['job_data'] = all_jobs

        if os.system(f"tmux list-windows -t {session_name} | grep {id}") == 0:
            print(f"Killing tmux window {session_name}:{id}")
            os.system(f"tmux kill-window -t {session_name}:{id}")
            time.sleep(0.5)

        # create the tmux window
        os.system(f"tmux new-window -t {session_name}:{id} -n {tag}")
        time.sleep(0.5)
        os.system(f"tmux send-keys -t {session_name}:{id} 'cd {dir_path}' Enter")
        if tpu is None:
            os.system(f"tmux send-keys -t {session_name}:{id} 'source kill_remote.sh; source staging.sh {config_args}' Enter")
        else:
            os.system(f"tmux send-keys -t {session_name}:{id} 'source kill_remote.sh {tpu}; source staging.sh ka={tpu} {config_args}' Enter") 
        
        print(f"Successfully created job in tmux window {session_name}:{id}")

        write_and_unlock_data(data)

    except BaseException as e:
        print(f"{RED}[Error] {NC} run: Failed to create job in tmux window")
        print(f"Error: {e}")
        release_lock_data()

    time.sleep(3)

    if user_obj.settings['monitor_after_run']:
        monitor_jobs(user_obj, args)

def check_jobs(user_obj, args):
    """
    Print the status of all the jobs in the tmux session.
    """
    # Get the tmux session name
    session_name = user_obj.tmux_name
    # Get all the windows in the tmux session
    windows = os.popen(f"tmux list-windows -t {session_name}").read().splitlines()
    # Do window by window
    for window in windows:
        # Get the window id
        window_id = window.split(':')[0]
        # Get the window name
        window_name = window.split(':')[1].split(' ')[0]
        # Find that in the job_data
        job_data = None
        for job in user_obj.job_data:
            if job['windows_id'] == int(window_id):
                job_data = job
                break
        if job_data is None:
            if window_id != '0':
                print(f'Window {window_id} (NOT FOUND IN DATA)')
                print('-'*40)
            continue
        else:
            print(f'Window {window_id} (tag: {job_data["job_tags"]})')
            print(f"DIR: {job_data['job_dir'].split('/')[-1]}\nTPU: {job_data['tpu']}")
        # Get the window last line
        last_line = os.popen(f"tmux capture-pane -t {session_name}:{window_id} -p").read()
        # remove all the empty spaces in the end
        last_line = last_line.rstrip()
        # Get last user_obj.monitor_length words
        show_length = user_obj.settings['show_length']
        monitor_length = user_obj.settings['monitor_length']
        monitor_verbose = user_obj.settings['monitor_verbose']
        last_line = last_line[-monitor_length:]
        msg = last_line[-show_length:]
            
        if job_data["status"] is not None:
            if job_data["status"] == 'error':
                if job_data["error"] == 'preempted':
                    print(f"Status: {RED}Preempted{NC}")
                    print(f"msg: {msg}")
                else:
                    print(f"Status: {RED}Error{NC}")
                    print(f"msg: {msg}")
                print('-'*40)
                continue
            elif job_data["status"] == 'killed':
                print(f"Status: {RED}Killed{NC}")
                if monitor_verbose:
                    print(f"msg: {msg}")
                print('-'*40)
                continue
            elif job_data["status"] == 'rerunned':
                try:
                    child = job_data['extra_msgs']['child']
                except Exception as e:
                    print(f"{RED}Failed to get child window id{NC}")
                    child = None
                print(f"Status: {YELLOW}Rerunned({job_data['error']}){NC} in window {child}")
                if monitor_verbose:
                    print(f"msg: {msg}")
                print('-'*40)
                continue
            elif job_data["status"] == 'finished':
                print(f"Status: {GREEN}Finished{NC}")
                if monitor_verbose:
                    print(f"msg: {msg}")
                print('-'*40)
                continue

        if re.search(r'[eE]rror', last_line) or re.search(r'ERROR', last_line):
            print(f"Status: {RED}Error{NC}")
            print(f"msg: {msg}")
        elif re.search(r'[cC]ompiling', last_line) or re.search(r'[cC]ompilation', last_line)or re.search(r'[cC]ompile', last_line):
            print(f"Status: {GREEN}Compiling{NC}")
            if monitor_verbose:
                print(f"msg: {msg}")
        elif re.search(r'[sS]ampling ', last_line):
            print(f"Status: {GREEN}Sampling{NC}")
            if monitor_verbose:
                print(f"msg: {msg}")
        elif re.search(r'[eE]poch\s([0-9]{1,4})', last_line):
            epoch = re.search(r'[eE]poch\s([0-9]{1,6})', last_line).group(1)
            print(f"Status: {GREEN}Running{NC} in epoch {epoch}")
            if monitor_verbose:
                print(f"msg: {msg}")
        elif re.search(r'ep=([0-9]){1,4}\.([0-9]){1,6}', last_line):
            epoch = re.search(r'ep=([0-9]){1,4}\.([0-9]){1,6}', last_line).group(0)[3:]
            print(f"Status: {GREEN}Running{NC} in epoch {epoch}")
            if monitor_verbose:
                print(f"msg: {msg}")
        elif re.search(r'[iI]nitializing', last_line):
            print(f"Status: {GREEN}Initializing{NC}")
            if monitor_verbose:
                print(f"msg: {msg}")
        elif re.search(r'[sS]taging', last_line):
            print(f"Status: {GREEN}Staging{NC}")
            if monitor_verbose:
                print(f"msg: {msg}")
        else:
            print(f"{YELLOW}Unknown{NC}")
            print(f"msg: {msg}")
        print('-'*40)



def kill_window(user_obj, args):
    data = read_and_lock_data()
    try:
        window_num = args[0]
        if not is_integer(window_num):
            raise ValueError(f"Window number {window_num} is not an integer")
        window_num = int(window_num)
        if window_num < 0:
            raise ValueError(f"Window number {window_num} is not valid")
        # Get the tmux session name
        session_name = user_obj.tmux_name
        # Get all the windows in the tmux session
        print(f"Killing window {window_num} in session {session_name}")
        os.system(f"tmux kill-window -t {session_name}:{window_num}")
        time.sleep(0.5)
        # remove the job from the job data
        all_jobs = user_obj.job_data
        for job in all_jobs:
            if job['windows_id'] == window_num:
                all_jobs.remove(job)
                break
        data['users'][user_obj.name]['job_data'] = all_jobs
        write_and_unlock_data(data)
    except:
        release_lock_data()


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
                        break
                break
        write_and_unlock_data(data)
    except:
        release_lock_data()
    
def monitor_jobs(user_obj, args):
    while True:
        check_jobs(user_obj, args)
        time.sleep(user_obj.settings['monitor_upd_time'])
        # clear the screen
        os.system('clear' if os.name == 'posix' else 'cls')
        # Update user object
        data = read_data()
        user_obj = data['users'][user_obj.name]
        user_obj = users.user_from_dict(user_obj)


def upd_log(window, log_dir, ka, start_time):
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
                        job['tpu'] = ka
                        job['start_time'] = start_time
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
        all_jobs = user_object.job_data
        for job in all_jobs:
            if job['status'] == 'finished':
                all_jobs.remove(job)
            # delete tmux window
            os.system(f"tmux kill-window -t {user_object.tmux_name}:{job['windows_id']}")
        data['users'][user_object.name]['job_data'] = all_jobs
        write_and_unlock_data(data)
    except:
        print(f"{RED}[Error] {NC}clear_finished_jobs: Failed to clear finished jobs")
        release_lock_data()

def clear_error_jobs(user_object):
    data = read_and_lock_data()
    try:
        print(f"Clearing error jobs...")
        all_jobs = user_object.job_data
        for job in all_jobs:
            if job['status'] == 'error' or job['error'] is not None:
                all_jobs.remove(job)
            # delete tmux window
            os.system(f"tmux kill-window -t {user_object.tmux_name}:{job['windows_id']}")
        data['users'][user_object.name]['job_data'] = all_jobs
        write_and_unlock_data(data)
    except:
        print(f"{RED}[Error] {NC}clear_error_jobs: Failed to clear error jobs")
        release_lock_data()

def clear_all_jobs(user_object):
    print(f"Clearing all jobs...")
    try:
        clear_finished_jobs(user_object)
    except:
        print(f"{RED}[Error] {NC}clear_all_jobs: Failed to clear finished jobs")
    try:
        clear_error_jobs(user_object)
    except:
        print(f"{RED}[Error] {NC}clear_all_jobs: Failed to clear error jobs{NC}")

def clear_zombie_jobs(user_object):
    """
    clear jobs whose window number can't be found in tmux session
    """
    data = read_and_lock_data()
    try:
        print(f"Clearing zombie jobs...")
        all_jobs = user_object.job_data
        for job in all_jobs:
            if os.system(f"tmux list-windows -t {user_object.tmux_name} | grep \" {job['windows_id']}:\"") != 0:
                all_jobs.remove(job)
        data['users'][user_object.name]['job_data'] = all_jobs
        write_and_unlock_data(data)
    except:
        print(f"{RED}[Error] {NC}clear_zombie_jobs: Failed to clear zombie jobs")
        release_lock_data()
