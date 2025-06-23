import os, random, time
import subprocess
from .data_io import read_and_lock_data, write_and_unlock_data, release_lock_data, read_data
from .helpers import *

def get_zone_pre(tpu):
    """
    Get the zone of the TPU, and check if it is preemptible.
    If the input is alias, it will be replaced with the real TPU name.
    Return zone, pre, tpu_full_name
    """
    data = read_data()
    tpu_aliases = data['tpu_aliases']
    all_tpus = []
    for z, tpu_list in data['all_tpus'].items():
        if z == 'preemptible':
            continue
        all_tpus.extend(tpu_list)
    if tpu in tpu_aliases:
        tpu = tpu_aliases[tpu]
    if tpu not in all_tpus:
        print(f"{FAIL} get_zone_pre: TPU {tpu} not found")
        return None, None, None
    all_tpus = data['all_tpus']
    zone = None
    for z, tpu_list in all_tpus.items():
        if z == 'preemptible':
            continue
        if tpu in tpu_list:
            zone = z
            break
    if zone is None:
        print(f"{FAIL} get_zone_pre: TPU {tpu} not found in any zone")
        return None, None, None
    return zone, tpu in data['all_tpus']['preemptible'], tpu
 
def kill_jobs_tpu(tpu, username = None):
    zone, pre, tpu = get_zone_pre(tpu)
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
    zone, pre, tpu = get_zone_pre(tpu)
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
    zone, pre, tpu = get_zone_pre(tpu)
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
        return apply_tpu(tpu, preemptible=False, delete=False)
    else:
        return apply_tpu(args[0], preemptible=True, delete=False)
    
def reapply(args):
    if '-norm' in args:
        tpu = args[1] if args[0] == '-norm' else args[0]
        return apply_tpu(tpu, preemptible=False, delete=True)
    else:
        return apply_tpu(args[0], preemptible=True, delete=True)

def apply_tpu(tpu, preemptible, delete=True):
    info_str = 'pre' if preemptible else 'norm'
    zone, pre, tpu = get_zone_pre(tpu)
    if zone is None: return
    if pre != preemptible:
        print(f"{FAIL} apply_tpu: TPU {tpu} in zone {zone} is not {info_str}")
        return
    if not delete:
        print(f"{INFO} Apply TPU {tpu} in zone {zone}...")
    else:
        print(f"{INFO} Re-apply TPU {tpu} in zone {zone}...")
    acc_type = None
    if 'v3-32' in tpu: acc_type = 'v3-32'
    elif 'v2-32' in tpu: acc_type = 'v2-32'
    elif 'v3-64' in tpu: acc_type = 'v3-64'
    elif 'v4-32' in tpu: acc_type = 'v4-32'
    elif 'v4-8' in tpu: acc_type = 'v4-8'
    else: raise ValueError(f"{FAIL} apply_{info_str}: Unknown TPU type {tpu}")
    if delete:
        cmd = f"gcloud compute tpus tpu-vm delete {tpu} --zone={zone} --quiet"
        try:
            subprocess.run(cmd.split(), timeout=300, check=True, stdout=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"{FAIL} apply_{info_str}: TPU deletion failed: {e}")
            return 'delete failed'
    cmd = f"gcloud compute tpus tpu-vm create {tpu} --zone={zone} --accelerator-type={acc_type} --version=tpu-ubuntu2204-base"
    if preemptible:
        cmd += " --preemptible"
    try:
        subprocess.run(cmd, shell=True, timeout=600, check=True, stdout=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        print(f"{FAIL} apply_{info_str}: applying TPU timed out")
        return 'timeout'

    time.sleep(5)
    
    cmd = f"gcloud compute tpus describe {tpu} --zone={zone} --format='value(state)'"
    try:
        state = subprocess.check_output(cmd, shell=True).decode().strip()
    except subprocess.CalledProcessError:
        print(f"{FAIL} apply_{info_str}: Failed to query TPU state")
        return 'describe failed'
    
    if state == 'READY':
        print(f"{GOOD} Now, TPU VM {tpu} is good, ready to use")
        # mount the disk
        print(f"{INFO} Mounting disk in TPU {tpu}...")
        res = mount_disk(tpu, quiet = True)
        if res != 'success':
            print(f"{FAIL} apply_{info_str}: mounting disk {res}")
            return f'mount {res}'
        
        print(f"{GOOD} apply_{info_str}: TPU {tpu} is good to use!")
        
        return 'success'
    
    else:
        print(f"{FAIL} apply_{info_str}: TPU {tpu} not ready, state: {state}")
        return 'unknown'

def delete_tpu(tpu):
    zone, pre, tpu = get_zone_pre(tpu)
    if zone is None: return
    print(f"{INFO} Deleting TPU {tpu} in zone {zone}...")
    cmd = f"gcloud compute tpus tpu-vm delete {tpu} --zone={zone} --quiet"
    try:
        subprocess.run(cmd.split(), timeout=300, check=True, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"{FAIL} delete_tpu: TPU deletion failed: {e}")
        return 'delete failed'
    
def check_tpu_status(tpu):
    """
    Check whether a TPU is preempted or not.
    return value: ['no tpu found', 'preempted', 'terminated', 'creating', 'ready', 'failed']
    """
    zone, pre, tpu = get_zone_pre(tpu)
    if zone is None: return
    cmd = f"gcloud compute tpus describe {tpu} --zone={zone} --format='value(state)'"
    try:
        state = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip()
    except subprocess.CalledProcessError:
        print(f"{FAIL} check_tpu_status: Failed to query TPU state")
        return 'failed'
    
    return state.lower()

def check_tpu_running(tpu, quiet = True):
    """
    Check whether a TPU is running or not.
    return value: ['running', 'free', 'failed']
    """
    zone, pre, tpu = get_zone_pre(tpu)
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
    zone, pre, tpu = get_zone_pre(tpu)
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
    zone, pre, tpu = get_zone_pre(tpu)
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
            print(f"{FAIL} check_env: Failed to query TPU state")
        return 'failed'
    except subprocess.TimeoutExpired:
        if not quiet:
            print(f"{FAIL} check_env: Timeout expired")
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
    zone, pre, tpu = get_zone_pre(tpu)
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
    zone, pre, tpu = get_zone_pre(tpu)
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
    zone, pre, tpu = get_zone_pre(tpu)

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
    
    
