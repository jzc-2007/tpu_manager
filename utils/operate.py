import os, random, time
import subprocess
from .data_io import read_and_lock_data, write_and_unlock_data, release_lock_data, read_data
from .helpers import *
from .constants import *
from .sheet import read_sheet_info, write_sheet_info, get_tpu_info_sheet

def update_tpu_status_for_spreadsheet():
    tpu_information = read_sheet_info()
    for full_name, info in tpu_information.items():
        result = check_tpu_status(full_name, quiet=True)
        previous_note = info['script_note']

        if result == 'ready':
            info['script_note'] = 'READY'
        elif result == 'preempted':
            info['script_note'] = 'PREEMPTED'
        elif result == 'terminated':
            info['script_note'] = 'TERMINATED'
        elif result == 'creating':
            info['script_note'] = 'CREATING'
        elif result == 'failed':
            info['script_note'] = 'NOT FOUND'
        else:
            info['script_note'] = 'UNKNOWN'

        if previous_note != info['script_note']:
            write_sheet_info(info)
 
def kill_jobs_tpu(tpu, username = None, ignore_window = None):
    zone, pre, spot, tpu = get_zone_pre_spot(tpu)
    if zone is None:
        print(f"{FAIL} kill_jobs_tpu: Could not determine zone.")
        return

    print(f"{INFO} kill_jobs_tpu: Killing jobs on TPU {tpu} zone {zone}...")

    try:
        data = read_data()
        print(f"{INFO} kill_jobs_tpu: Sending C-c to all jobs on TPU {tpu}...")
        for user in data["users"]:
            if username is not None and username != user:
                continue
            user_tmux_name = data["users"][user]["tmux_name"]
            for job in data["users"][user]["job_data"]:
                if job["tpu"] == tpu:
                    window = job["windows_id"]
                    if ignore_window is not None and window == ignore_window['window'] and ignore_window['session'] == user_tmux_name:
                        print(f"{INFO} kill_jobs_tpu: Ignoring window {window} for user {user}")
                        continue
                    if window is not None:
                        subprocess.run(f"tmux send-keys -t {user_tmux_name}:{window} C-c", shell=True, check=False)

        time.sleep(3)

        list_cmd = (
            f"gcloud compute tpus tpu-vm ssh {tpu} --zone {zone} --worker=all "
            "--command \"ps -eo pid,ppid,stat,cmd | grep 'main.py' | grep -v 'grep' || true\""
        )
        result = subprocess.run(list_cmd, shell=True, timeout=30, check=False,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        lines = result.stdout.strip().splitlines()

        if not lines:
            print(f"{INFO} No main.py processes found.")
            return 'success'

        pids = set()

        for line in lines:
            parts = line.strip().split(None, 3)
            if len(parts) >= 2:
                pid, ppid = parts[0], parts[1]
                pids.add(pid)
                pids.add(ppid)

        if not pids:
            print(f"{INFO} No processes to kill.")
            return 'success'

        pid_list = " ".join(pids)
        print(f"{INFO} Killing PIDs: {pid_list}")
        
        kill_cmd = (
            f"gcloud compute tpus tpu-vm ssh {tpu} --zone {zone} --worker=all "
            f"--command \"sudo kill -9 {pid_list} || true\""
        )
        subprocess.run(kill_cmd, shell=True, timeout=30, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        print(f"{INFO} Cleaning /dev/accel0 occupation...")
        kill_accel_cmd = (
            f"gcloud compute tpus tpu-vm ssh {tpu} --zone {zone} --worker=all "
            "--command \"pids=$(sudo lsof -w /dev/accel0 | grep 'python' | grep -v 'grep' | awk '{print $2}'); "
            "if [ ! -z \\\"$pids\\\" ]; then sudo kill -9 $pids; fi\""
        )
        subprocess.run(kill_accel_cmd, shell=True, timeout=30, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    except subprocess.TimeoutExpired:
        print(f"{FAIL} kill_jobs_tpu: Timeout.")
        return 'kill timeout'
    except Exception as e:
        print(f"{FAIL} kill_jobs_tpu: {e}")
        return 'kill error'

    print(f"{GOOD} kill_jobs_tpu: Jobs killed successfully.")
    return 'success'

def kill_jobs_tpu_old(tpu):
    zone, pre, spot, tpu = get_zone_pre_spot(tpu)
    if zone is None:
        print("[FAIL] kill_jobs_tpu: Could not determine zone.")
        return

    print(f"{INFO} kill_jobs_tpu: Killing jobs in TPU {tpu} in zone {zone}...")

    cmd1 = (
        f"gcloud compute tpus tpu-vm ssh {tpu} --zone {zone} --worker=all "
        "--command \"pids=$(pgrep -af python | grep 'main.py' | grep -v 'grep' | awk '{print $1}'); "
        "if [ ! -z \\\"$pids\\\" ]; then "
        "for pid in $pids; do echo Killing $pid; sudo kill -9 $pid; done; "
        "else echo 'No main.py processes found.'; fi\""
    )    
    cmd2 = (
        f"gcloud compute tpus tpu-vm ssh {tpu} --zone {zone} --worker=all "
        "--command \"pids=$(sudo lsof -w /dev/accel0 | grep 'python' | grep -v 'grep' | awk '{print $2}'); "
        "if [ ! -z \\\"$pids\\\" ]; then "
        "for pid in $pids; do echo Killing $pid; sudo kill -9 $pid; done; "
        "else echo 'No processes found on /dev/accel0.'; fi\""
    )

    try:
        subprocess.run(cmd1, shell=True, timeout=30, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        subprocess.run(cmd2, shell=True, timeout=30, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.TimeoutExpired:
        print(f"{FAIL} kill_jobs_tpu: Killing jobs timed out")
        return 'timeout'
    except subprocess.CalledProcessError as e:
        print(f"{FAIL} kill_jobs_tpu: Killing jobs failed.")
        print(f"{YELLOW}stdout:{NC} {e.stdout.strip()}")
        print(f"{YELLOW}stderr:{NC} {e.stderr.strip()}")
        return 'kill failed'
    print(f"{GOOD} kill_jobs_tpu: Killing jobs done")
    return 'success'

def set_wandb(tpu):
    zone, pre, spot, tpu = get_zone_pre_spot(tpu)
    if zone is None:
        print(f"{FAIL} set_wandb: TPU {tpu} not found")
        return
    
    print(f"{INFO} Setting up remote wandb in TPU {tpu}...")

    data = read_data()
    wandb_key, conda_env = data["wandb_api_key"], data["conda_env_name"]
    data_root = "kmh-nfs-ssd-eu-mount" if 'eu' in zone else "kmh-nfs-us-mount"
    conda_path = f"/{data_root}/code/qiao/anaconda3/envs/{conda_env}/bin/python"

    remote_cmd = f'{conda_path} -m wandb login {wandb_key}'


    cmd = f"gcloud compute tpus tpu-vm ssh {tpu} --zone {zone} --worker=all --command \"{remote_cmd}\" "

    try:
        setup_process = subprocess.run(cmd, shell=True, timeout=300, check=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.TimeoutExpired:
        print(f"{FAIL} set_wandb: setting wandb timed out")
        return 'timeout'
    except subprocess.CalledProcessError as e:
        print(f"{FAIL} set_wandb: setting wandb failed.")
        print(f"{YELLOW}stdout:{NC} {e.stdout.strip()}")
        print(f"{YELLOW}stderr:{NC} {e.stderr.strip()}")
        return 'wandb failed'

    print(f"{GOOD} set_wandb: Setting wandb done")
    return 'success'

def apply(args):
    if '-norm' in args:
        tpu = args[1] if args[0] == '-norm' else args[0]
        return apply_and_set_env(tpu, preemptible=False, delete=False)
    else:
        return apply_and_set_env(args[0], preemptible=True, delete=False)

def apply_until_success(args):
    if '-norm' in args:
        tpu = args[1] if args[0] == '-norm' else args[0]
        return apply_and_set_env(tpu, preemptible=False, delete=False, repeat_time = 36000)
    else:
        return apply_and_set_env(args[0], preemptible=True, delete=False, repeat_time = 36000)
    
def reapply(args):
    if '-norm' in args:
        tpu = args[1] if args[0] == '-norm' else args[0]
        return apply_and_set_env(tpu, preemptible=False, delete=True)
    else:
        return apply_and_set_env(args[0], preemptible=True, delete=True)

def apply_and_set_env(tpu, preemptible = False, spot = False, delete=True, repeat_time=None, retry_interval=20):
    info_str = 'pre' if preemptible else 'norm'
    zone, pre, spot, tpu = get_zone_pre_spot(tpu)
    print(zone, pre, spot, tpu, preemptible)
    if zone is None:
        return
    if preemptible != (pre or spot):
        print(f"{FAIL} apply_and_set_env: TPU {tpu} in zone {zone} is not {info_str}")
        return
    if not delete:
        print(f"{INFO} Apply TPU {tpu} in zone {zone}...")
    else:
        print(f"{INFO} Re-apply TPU {tpu} in zone {zone}...")

    acc_type = None
    for key in NAME_TO_TYPE:
        if key in tpu:
            acc_type = NAME_TO_TYPE[key]
    if acc_type is None:
        raise ValueError(f"{FAIL} apply_{info_str}: Unknown TPU type {tpu}")

    if delete:
        try:
            delete_tpu(tpu)
        except subprocess.CalledProcessError as e:
            print(f"{WARNING} apply_{info_str}: TPU deletion failed: {e}")

    if 'v6' in acc_type:
        base_cmd = f"gcloud compute tpus tpu-vm create {tpu} --zone={zone} --accelerator-type={acc_type} --version=v2-alpha-tpuv6e"
    else:
        base_cmd = f"gcloud compute tpus tpu-vm create {tpu} --zone={zone} --accelerator-type={acc_type} --version=tpu-ubuntu2204-base"
    if preemptible:
        base_cmd += " --preemptible"
    if spot:
        base_cmd += " --spot"

    start_time = time.time()
    attempt = 0

    while True:
        attempt += 1
        cmd_timeout = 600
        try:
            subprocess.run(base_cmd, shell=True, timeout=cmd_timeout, check=True, stdout=subprocess.DEVNULL)
            break  # success
        except subprocess.CalledProcessError as e:
            print(f"{FAIL} apply_{info_str}: TPU creation failed (attempt {attempt}) with return code {e.returncode}")
        except subprocess.TimeoutExpired:
            print(f"{FAIL} apply_{info_str}: applying TPU timed out")
            return 'timeout'

        # if no repeat_time → only try once
        if repeat_time is None:
            raise NotADirectoryError(f'xibo 活着. create failed')
            return 'create failed'

        # check repeat_time limit
        if (time.time() - start_time) > repeat_time - cmd_timeout:
            print(f"{FAIL} apply_{info_str}: repeat_time {repeat_time}s exceeded, giving up")
            raise NotADirectoryError(f'xibo 活着. timeout')
            return 'timeout'

        print(f"{INFO} Retrying TPU creation in {retry_interval}s...")

        time.sleep(retry_interval + random.randint(0, 10))

    # short pause before querying state
    time.sleep(5)

    cmd = f"gcloud compute tpus describe {tpu} --zone={zone} --format='value(state)'"
    try:
        state = subprocess.check_output(cmd, shell=True).decode().strip()
    except subprocess.CalledProcessError:
        print(f"{FAIL} apply_{info_str}: Failed to query TPU state")
        return 'describe failed'

    if state == 'READY':
        print(f"{GOOD} Now, TPU VM {tpu} is good, ready to use")

        print(f"{INFO} Update Spreadsheet info for {tpu}...")
        tpu_info = get_tpu_info_sheet(tpu)
        tpu_info['other_note'] = f'{get_edt_time_str()}'
        write_sheet_info(tpu_info)

        print(f"{INFO} Mounting disk in TPU {tpu}...")
        res = mount_disk(tpu, quiet=True)
        if res != 'success':
            print(f"{FAIL} apply_{info_str}: mounting disk {res}")
            return f'mount {res}'

        print(f"{GOOD} apply_{info_str}: TPU {tpu} is good to use!")
        return 'success'
    else:
        print(f"{FAIL} apply_{info_str}: TPU {tpu} not ready, state: {state}")
        return 'unknown'

def delete_tpu(tpu):
    print(f"{INFO} delete_tpu: Deleting TPU {tpu}...")
    zone, pre, spot, tpu = get_zone_pre_spot(tpu)
    if zone is None: return
    status = check_tpu_status(tpu, quiet=True)
    if status == 'failed':
        print(f"{WARNING} delete_tpu: TPU {tpu} not found")
        return 'delete failed'
    print(f"{INFO} Deleting TPU {tpu} in zone {zone}...")
    cmd = f"gcloud compute tpus tpu-vm delete {tpu} --zone={zone} --quiet"
    try:
        subprocess.run(cmd.split(), timeout=300, check=True, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"{FAIL} delete_tpu: TPU deletion failed: {e}")
        return 'delete failed'
    return 'success'
    
def check_tpu_status(tpu, quiet = False):
    """
    Check whether a TPU is preempted or not.
    return value: ['no tpu found', 'preempted', 'terminated', 'creating', 'ready', 'failed']
    """
    zone, pre, spot, tpu = get_zone_pre_spot(tpu)
    if zone is None: return 'no tpu found'
    cmd = f"gcloud compute tpus describe {tpu} --zone={zone} --format='value(state)'"
    try:
        state = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip()
    except subprocess.CalledProcessError:
        if not quiet:
            print(f"{FAIL} check_tpu_status: Failed to query TPU {tpu} state")
        return 'failed'
    
    return state.lower()

def check_tpu_running(tpu, quiet = True):
    """
    Check whether a TPU is running or not.
    return value: ['running', 'free', 'failed']
    """
    zone, pre, spot, tpu = get_zone_pre_spot(tpu)
    if zone is None: return
    cmd = f"gcloud compute tpus tpu-vm ssh {tpu} --zone {zone} --worker=all --command \"sudo lsof -w /dev/accel0\" "
    try:
        if quiet:
            result = subprocess.run(cmd, shell=True, timeout=30, check=False,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            result = subprocess.run(cmd, shell=True, timeout=30, check=False,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            if quiet:
                return 'running'
            else:
                print(f"{INFO} check_tpu_running: TPU {tpu} is running")
                return 'running'
        else:
            if quiet:
                return 'free'
            else:
                print(f"{INFO} check_tpu_running: TPU {tpu} is free")
                return 'free'
    except subprocess.TimeoutExpired:
        if quiet:
            return 'failed'
        else:
            print(f"{FAIL} check_tpu_running: Timeout expired")
            return 'failed'
        

def describe_tpu(tpu, quiet = False):
    """
    Describe the TPU.
    Return value: ['no tpu found', 'preempted', 'test env failed', 'file error', 'unknown', 'running', 'free', 'failed']
    """
    zone, pre, spot, tpu = get_zone_pre_spot(tpu)
    if zone is None: 
        print(f"{FAIL} describe_tpu: TPU {tpu} not found")
        return 'no tpu found'
    if not quiet:
        print(f"{INFO} describe_tpu: Starting to describe TPU {tpu} in zone {zone}...")
        print(f"{INFO} describe_tpu: Querying TPU {tpu} state...")
    res = check_tpu_status(tpu)
    if res == 'preempted':
        if not quiet:
            print(f"{INFO} describe_tpu: TPU {tpu} is {RED}PREEMPTED{NC}")
        return 'preempted'
    elif res == 'terminated':
        if not quiet:
            print(f"{INFO} describe_tpu: TPU {tpu} is {RED}TERMINATED{NC}")
        return 'terminated'
    elif res == 'creating':
        if not quiet:
            print(f"{INFO} describe_tpu: TPU {tpu} is {YELLOW}CREATING{NC}")
        return 'creating'
    elif res == 'ready':
        if not quiet:
            print(f"{INFO} describe_tpu: TPU {tpu} is {GREEN}READY{NC}")
            # print(f"{INFO} describe_tpu: Checking environment in TPU {tpu}...")
        state = check_env(tpu, quiet=quiet)
        if state == 'success':
            if not quiet:
                print(f"{GOOD} describe_tpu: Environment in TPU {tpu} is good!")
                print(f"{INFO} describe_tpu: Checking TPU {tpu} running state...")
            running = check_tpu_running(tpu, quiet=quiet)
            if running == 'running':
                if not quiet:
                    print(f"{GOOD} describe_tpu: TPU {tpu} is {GREEN}running{NC}")
                    return 'running'
            elif running == 'free':
                if not quiet:
                    print(f"{GOOD} describe_tpu: TPU {tpu} is {GREEN}free{NC}")
                    return 'free'
            else:
                if not quiet:
                    print(f"{FAIL} describe_tpu: TPU {tpu} is getting unknown error, please contact the admin.")
                    return 'failed'
        elif state == 'failed':
            if not quiet:
                print(f"{FAIL} describe_tpu: Environment in TPU {tpu} is not good")
                print(f"state: {state}")
                print("Unexpected error, please check the TPU manually, or contact the admin")
            return 'failed'
        elif state == 'file error':
            if not quiet:
                print(f"{FAIL} describe_tpu: Environment in TPU {tpu} has file error")
                print(f"{INFO} describe_tpu: You may need to {PURPLE}mount the NFS{NC} by `tpu mount-disk`, or solve the env by `tpu solve`")
            return 'file error'
        elif state == 'occupied':
            if not quiet:
                print(f"{FAIL} describe_tpu: Environment in TPU {tpu} is {YELLOW}occupied{NC}")
                # print(f"{INFO} describe_tpu: You may need to {PURPLE}kill the jobs{NC} by `tpu kill-jobs`")
            return 'occupied'
        elif state == 'unknown':
            if not quiet:
                print(f"{FAIL} Environment in TPU {tpu} is getting unkown error, please contact the admin.")
                print(f"state: {state}")
            return 'unknown'
        else:
            if not quiet:
                print(f"{FAIL} describe_tpu: TPU {tpu} is getting unkown error, please contact the admin.")
                print(f"state: {state}")
            return 'unknown'
    elif res == 'timeout':
        if not quiet:
            print(f"{FAIL} describe_tpu: Timeout expired")
        return 'timeout'

def check_env(tpu, quiet = False):
    """
    Check if the environment in the TPU is good.
    Return value: ['no tpu found', 'success', 'failed', 'file error', 'unknown', 'timeout', 'occupied']
    """
    zone, pre, spot, tpu = get_zone_pre_spot(tpu)
    if zone is None: return 'no tpu found'
    data = read_data()
    conda_env = data["conda_env_name"]
    data_root = "kmh-nfs-ssd-eu-mount" if 'eu' in zone else "kmh-nfs-us-mount"
    conda_path = f"/{data_root}/code/qiao/anaconda3/envs/{conda_env}/bin/python"
    cmd = f"gcloud compute tpus tpu-vm ssh {tpu} --zone {zone} --worker=all --command \"{conda_path} -c 'import jax; print(jax.devices())'\""
    if not quiet:
        print(f"{INFO} check_env: Checking environment in TPU {tpu}... This may take a while...")
    try:
        # get the output of the command
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout = 600)
        stdout, stderr= result.stdout, result.stderr

    except subprocess.CalledProcessError:
        if not quiet:
            print(f"{FAIL} check_env: Failed to query TPU {tpu} state")
        return 'failed'
    except subprocess.TimeoutExpired:
        if not quiet:
            print(f"{FAIL} check_env: Checking {tpu}: Timeout expired")
        return 'timeout'

    if 'No such file or directory' in stderr:
        if not quiet:
            print(f"{FAIL} check_remote_env: Can't find directory")
            print(f"{INFO} You may need to {PURPLE}mount the NFS{NC} first")
        return 'file error'
    
    if 'The TPU is already in use' in stderr:
        if not quiet:
            print(f"{FAIL} check_env: TPU {tpu} is already in use")
        return 'occupied'
    
    if "TpuDevice" in stdout:
        print(f"{GOOD} check_remote_env: TPU {tpu} is good!")
        return 'success'
    
    else:
        print(f"{FAIL} check_remote_env: TPU {tpu} is getting unkown error, please contact the admin.")
        print(f"stdout: {stdout}")
        print(f"stderr: {stderr}")
        return 'unknown'

def mount_disk(tpu, quiet = False):
    """
    Mount the disk and setup remote wandb.
    """
    zone, pre, spot, tpu = get_zone_pre_spot(tpu)
    if zone is None: return
    print(f"{INFO} Mounting disk in TPU {tpu}...")

    cmd1 = f'''
    gcloud compute tpus tpu-vm ssh {tpu} --zone {zone} --worker=all \
      --command "
        for i in {{1..3}}; do
          ps -ef | grep -i unattended | grep -v 'grep' | awk '{{print \\$2}}' | xargs -r sudo kill -9
          sleep 2
        done
        sudo DEBIAN_FRONTEND=noninteractive apt-get -y update
        sudo DEBIAN_FRONTEND=noninteractive apt-get -y install nfs-common
        ps -ef | grep -i unattended | grep -v 'grep' | awk '{{print \\$2}}' | xargs -r sudo kill -9
        sleep 2
      "
    '''

    cmd2 = f"""
    gcloud compute tpus tpu-vm ssh {tpu} --zone {zone} --worker=all --command "
    sudo mkdir -p /kmh-nfs-us-mount
    sudo mount -t nfs -o vers=3 10.26.72.146:/kmh_nfs_us /kmh-nfs-us-mount
    sudo chmod go+rw /kmh-nfs-us-mount
    ls /kmh-nfs-us-mount
    sudo mkdir -p /kmh-nfs-ssd-eu-mount
    sudo mount -t nfs -o vers=3 10.150.179.250:/kmh_nfs_ssd_eu /kmh-nfs-ssd-eu-mount
    sudo chmod go+rw /kmh-nfs-ssd-eu-mount
    ls /kmh-nfs-ssd-eu-mount
    "

    gcloud compute tpus tpu-vm ssh {tpu} --zone {zone} \
    --worker=all --command "
    sudo rm -rf /home/\$(whoami)/.local
    echo 'Current dir: '
    pwd
    conda create -n NNX python==3.10.14 -y
    conda activate NNX # These two lines are very smart. If on a device there is no conda, then these two lines error out, but the remaining can still be run.
    pip install 'setuptools==69.5.1'
    pip install jax[tpu]==0.4.37 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    pip install jaxlib==0.4.37 'flax>=0.8'
    # pip install -r requirements.txt # other tang dependencies
    pip install pillow clu tensorflow==2.15.0 'keras<3' 'torch<=2.4' torchvision tensorflow_datasets matplotlib==3.9.2
    pip install orbax-checkpoint==0.6.4 ml-dtypes==0.5.0 tensorstore==0.1.67
    pip install diffusers dm-tree cached_property ml-collections
    pip install flax==0.10.2
    pip install 'wandb==0.19.9'
    pip install gcsfs
    "
    """
    try:
        download_process = \
            subprocess.run(cmd1, shell=True, timeout=600, check=True,\
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL) if quiet \
            else subprocess.run(cmd1, shell=True, timeout=600, check=True)

        time.sleep(5)

        mount_process = \
            subprocess.run(cmd2, shell=True, timeout=600, check=True,\
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) if quiet \
            else subprocess.run(cmd2, shell=True, timeout=600, check=True)

    except subprocess.TimeoutExpired:
        print(f"{FAIL} mount_disk: mounting disk timed out")
        return 'mounting timeout'
    except subprocess.CalledProcessError as e:
        print(f"{FAIL} mount_disk: {e}")
        print(f"stderr: {e.stderr}")
        print(f"stdout: {e.stdout}")
        return 'mounting failed'
    
    print(f"{INFO} Mounting disk in TPU {tpu} done")
    print(f"{INFO} Checking environment in TPU {tpu}...")
    state = check_env(tpu)

    if state == 'success':
        print(f"{GOOD} Environment in TPU {tpu} is good, done mounting disk")
        print(f"{INFO} Setting wandb again to make sure it works...")
        res = set_wandb(tpu)
        if res != 'success':
            print(f"{FAIL} mount_disk: setting wandb failed")
            return 'wandb failed'
        return 'success'
    else:
        print(f"{FAIL} Environment in TPU {tpu} is not good")
        print(f"state: {state}")
        print("Unexpected error, please check the TPU manually, or contact the admin")
        return 'checking env failed'

def test_remote(tpu):
    zone, pre, spot, tpu = get_zone_pre_spot(tpu)
    if zone is None: return
    print(f"{INFO} Testing remote TPU {tpu} in zone {zone}...")
    print(f"{INFO} Do you want python test? (y/n)")
    ans = input()
    if ans == 'y':
        print(f"{INFO} please enter the python command:")
        cmd = input()    
        data = read_data()
        conda_env = data["conda_env_name"]
        data_root = "kmh-nfs-ssd-eu-mount" if 'eu' in zone else "kmh-nfs-us-mount"
        conda_path = f"/{data_root}/code/qiao/anaconda3/envs/{conda_env}/bin/python"
        cmd = f"gcloud compute tpus tpu-vm ssh {tpu} --zone {zone} --worker=all --command \"{conda_path} -c '{cmd}'\""
        try:
            result = subprocess.run(cmd, shell=True, timeout=300, check=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print(f"{INFO} {result.stdout}")
            print(f"{INFO} {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"{FAIL} test_remote: Timeout expired")
            return 'timeout'
        except subprocess.CalledProcessError as e:
            print(f"{FAIL} test_remote: {e}")
            print(f"{YELLOW}stdout:{NC} {e.stdout.strip()}")
            print(f"{YELLOW}stderr:{NC} {e.stderr.strip()}")
            return 'test failed'
    elif ans == 'n':
        print(f"{INFO} please enter the bash command:")
        cmd = input()
        cmd = f"gcloud compute tpus tpu-vm ssh {tpu} --zone {zone} --worker=all --command \"{cmd}\""
        try:
            result = subprocess.run(cmd, shell=True, timeout=300, check=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print(f"{INFO} \nstdout:\n{result.stdout}")
            print(f"{INFO} \nstderr:\n{result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"{FAIL} test_remote: Timeout expired")
            return 'timeout'
        except subprocess.CalledProcessError as e:
            print(f"{FAIL} test_remote: {e}")
            print(f"{YELLOW}stdout:{NC} {e.stdout.strip()}")
            print(f"{YELLOW}stderr:{NC} {e.stderr.strip()}")
            return 'test failed'
    else:
        print(f"{FAIL} test_remote: Unknown command")
        return 'unknown command'
    print(f"{GOOD} test_remote: TPU {tpu} tested successfully")
    return 'success'
    
def restart(tpu):
    zone, pre, spot, tpu = get_zone_pre_spot(tpu)

    print(f"{INFO} Rebooting {tpu}... This may take a while...")

    reboot_cmd = (
        f'timeout 20s gcloud compute tpus tpu-vm ssh {tpu} --zone {zone} '
        '--worker=all --command "sudo reboot" '
        '--ssh-flag="-o ConnectionAttempts=1" --ssh-flag="-o ConnectTimeout=5"'
    )
    try:
        subprocess.run(reboot_cmd, shell=True, check=True, 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"{INFO} Expected SSH disconnect during reboot: {e}")
    except subprocess.TimeoutExpired:
        print(f"{INFO} Expected SSH timeout during reboot")
    except Exception as e:
        print(f"{FAIL} Unexpected error while rebooting: {e}")
        return 'reboot failed'

    print(f"{INFO} Reboot command sent. Sleeping 3 minutes...")
    time.sleep(180)

    print(f"{INFO} Checking if VM is ready...")
    while True:
        check_cmd = (
            f'gcloud compute tpus tpu-vm ssh {tpu} --zone {zone} '
            '--worker=all --command "ls"'
        )
        try:
            subprocess.run(check_cmd, shell=True, timeout=60, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"{GOOD} VM is ready!")
            break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            print(f"{INFO} VM not ready yet, sleeping 60s...")
            time.sleep(60)
        except Exception as e:
            print(f"{FAIL} Unexpected error while checking VM readiness: {e}")
            time.sleep(60)

    print(f"{GOOD} VM is ready! Doing mounting...")

    try:
        mount_disk(tpu, quiet=True)
        print(f"{GOOD} Mounting done!")
    except Exception as e:
        print(f"{FAIL} Mounting failed: {e}")
        return 'mount failed'

    print(f"{GOOD} Restart done!")
    
    
