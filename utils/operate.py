import os, random, time
import subprocess
from .data_io import read_and_lock_data, write_and_unlock_data, release_lock_data, read_data
RED, GREEN, YELLOW, PURPLE, NC = "\033[1;31m", "\033[1;32m", "\033[1;33m", "\033[1;34m", "\033[0m"
GOOD, INFO, WARNING, FAIL = f"{GREEN}[GOOD]{NC}", f"{PURPLE}[INFO]{NC}", f"{YELLOW}[WARNING]{NC}", f"{RED}[FAIL]{NC}"
OPERATE_PATH = "/home/jzc/zhichengjiang/working/xibo_tpu_manager"
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

def kill_jobs_tpu(tpu):
    zone, pre, tpu = get_zone_pre(tpu)
    if zone is None:
        print("[FAIL] kill_jobs_tpu: Could not determine zone.")
        return

    print(f"{INFO} kill_jobs_tpu: Killing jobs in TPU {tpu} in zone {zone}...")

    cmd = (
        f"gcloud compute tpus tpu-vm ssh {tpu} --zone {zone} --worker=all "
        "--command \"pids=$(pgrep -af python | grep 'main.py' | grep -v 'grep' | awk '{print $1}'); "
        "if [ ! -z \\\"$pids\\\" ]; then "
        "for pid in $pids; do echo Killing $pid; sudo kill -9 $pid; done; "
        "else echo 'No main.py processes found.'; fi\""
    )

    try:
        subprocess.run(cmd, shell=True, timeout=300, check=True,
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

def apply_pre(tpu, delete=True):
    zone, pre, tpu = get_zone_pre(tpu)
    if zone is None: return
    if not pre:
        print(f"{FAIL} apply_pre: TPU {tpu} in zone {zone} is not preemptible")
        return
    if not delete:
        print(f"{INFO} Apply TPU {tpu} in zone {zone}...")
    else:
        print(f"{INFO} Re-apply TPU {tpu} in zone {zone}...")
    acc_type = None
    if 'v3-32' in tpu: acc_type = 'v3-32'
    elif 'v2-32' in tpu: acc_type = 'v2-32'
    elif 'v4-32' in tpu: acc_type = 'v4-32'
    elif 'v4-8' in tpu: acc_type = 'v4-8'
    else: raise ValueError(f"{FAIL} apply_pre: Unknown TPU type {tpu}")
    if delete:
        cmd = f"gcloud compute tpus tpu-vm delete {tpu} --zone={zone} --quiet"
        try:
            subprocess.run(cmd.split(), timeout=300, check=True, stdout=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"{FAIL} apply_pre: TPU deletion failed: {e}")
            return 'delete failed'

    cmd = f"gcloud compute tpus tpu-vm create {tpu} --zone={zone} --accelerator-type={acc_type} --version=tpu-ubuntu2204-base --preemptible"
    try:
        subprocess.run(cmd, shell=True, timeout=600, check=True, stdout=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        print(f"{FAIL} apply_pre: applying preemptible TPU timed out")
        return 'timeout'

    cmd = f"gcloud compute tpus describe {tpu} --zone={zone} --format='value(state)'"
    try:
        state = subprocess.check_output(cmd, shell=True).decode().strip()
    except subprocess.CalledProcessError:
        print(f"{FAIL} apply_pre: Failed to query TPU state")
        return 'describe failed'

    if state == 'READY':
        print(f"Now, TPU VM {tpu} is good, ready to use")
        # mount the disk
        print(f"{INFO} Mounting disk in TPU {tpu}...")
        res = mount_disk(tpu, quiet = True)
        if res != 'success':
            print(f"{FAIL} apply_pre: mounting disk failed")
            return 'mount failed'
        print(f"{GOOD} apply_pre: TPU {tpu} is good, done mounting disk")

        # setup remote wandb
        print(f"{INFO} Setting up remote wandb in TPU {tpu}...")
        res = set_wandb(tpu)
        if res != 'success':
            print(f"{FAIL} apply_pre: setting wandb failed")
            return 'wandb failed'
        print(f"{GOOD} apply_pre: Setting wandb done")
        print(f"{GOOD} apply_pre: TPU {tpu} is good to use!")
        
        return 'success'
    else:
        print(f"{FAIL} apply_pre: TPU {tpu} not ready, state: {state}")
        return 'unknown'

def apply_norm(tpu, delete=True):
    zone, pre, tpu = get_zone_pre(tpu)
    if zone is None: return
    if pre:
        print(f"{FAIL} apply_norm: TPU {tpu} in zone {zone} is preemptible")
        return
    if not delete:
        print(f"{INFO} Apply TPU {tpu} in zone {zone}...")
    else:
        print(f"{INFO} Re-apply TPU {tpu} in zone {zone}...")
    acc_type = None
    if 'v3-32' in tpu: acc_type = 'v3-32'
    elif 'v2-32' in tpu: acc_type = 'v2-32'
    elif 'v4-32' in tpu: acc_type = 'v4-32'
    elif 'v4-8' in tpu: acc_type = 'v4-8'
    else: raise ValueError(f"{FAIL} apply_pre: Unknown TPU type {tpu}")
    if delete:
        cmd = f"gcloud compute tpus tpu-vm delete {tpu} --zone={zone} --quiet"
        try:
            subprocess.run(cmd.split(), timeout=300, check=True, stdout=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"{FAIL} apply_pre: TPU deletion failed: {e}")
            return 'delete failed'

    cmd = f"gcloud compute tpus tpu-vm create {tpu} --zone={zone} --accelerator-type={acc_type} --version=tpu-ubuntu2204-base"
    try:
        subprocess.run(cmd, shell=True, timeout=600, check=True, stdout=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        print(f"{FAIL} apply_norm: applying TPU timed out")
        return 'timeout'

    cmd = f"gcloud compute tpus describe {tpu} --zone={zone} --format='value(state)'"
    try:
        state = subprocess.check_output(cmd, shell=True).decode().strip()
    except subprocess.CalledProcessError:
        print(f"{FAIL} apply_norm: Failed to query TPU state")
        return 'describe failed'

    if state == 'READY':
        print(f"Now, TPU VM {tpu} is good, ready to use")
        # mount the disk
        print(f"{INFO} Mounting disk in TPU {tpu}...")
        res = mount_disk(tpu, quiet = True)
        if res != 'success':
            print(f"{FAIL} apply_norm: mounting disk failed")
            return 'mount failed'
        print(f"{GOOD} apply_norm: TPU {tpu} is good, done mounting disk")

        # setup remote wandb
        print(f"{INFO} Setting up remote wandb in TPU {tpu}...")
        res = set_wandb(tpu)
        if res != 'success':
            print(f"{FAIL} apply_norm: setting wandb failed")
            return 'wandb failed'
        print(f"{GOOD} apply_norm: Setting wandb done")
        print(f"{GOOD} apply_norm: TPU {tpu} is good to use!")
        
        return 'success'
    else:
        print(f"{FAIL} apply_norm: TPU {tpu} not ready, state: {state}")
        return 'unknown'
    
def check_tpu_status(tpu):
    zone, pre, tpu = get_zone_pre(tpu)
    if zone is None: return
    cmd = f"gcloud compute tpus describe {tpu} --zone={zone} --format='value(state)'"
    try:
        state = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip()
    except subprocess.CalledProcessError:
        print(f"{FAIL} check_tpu_status: Failed to query TPU state")
        return 'failed'
    
    return state

def describe_tpu(tpu, quiet = False):
    """
    Describe the TPU.
    Return value: ['no tpu found', 'preempted', 'terminated', 'creating', 'success', 'test env failed', 'file error', 'unknown']
    """
    zone, pre, tpu = get_zone_pre(tpu)
    if zone is None: 
        print(f"{FAIL} describe_tpu: TPU {tpu} not found")
        return 'no tpu found'
    if not quiet:
        print(f"{INFO} describe_tpu: Starting to describe TPU {tpu} in zone {zone}...")
        print(f"{INFO} describe_tpu: Querying TPU {tpu} state...")
    res = check_tpu_status(tpu)
    if res == 'PREEMPTED':
        if not quiet:
            print(f"{INFO} describe_tpu: TPU {tpu} is {RED}PREEMPTED{NC}")
        return 'preempted'
    elif res == 'TERMINATED':
        if not quiet:
            print(f"{INFO} describe_tpu: TPU {tpu} is {RED}TERMINATED{NC}")
        return 'terminated'
    elif res == 'CREATING':
        if not quiet:
            print(f"{INFO} describe_tpu: TPU {tpu} is {YELLOW}CREATING{NC}")
        return 'creating'
    elif res == 'READY':
        if not quiet:
            print(f"{INFO} TPU {tpu} is {GREEN}READY{NC}")
            print(f"{INFO} Checking environment in TPU {tpu}...")
        state = check_env(tpu, quiet=True)
        if state == 'success':
            if not quiet:
                print(f"{GOOD} Environment in TPU {tpu} is good!")
            return 'success'
        elif state == 'failed':
            if not quiet:
                print(f"{FAIL} Environment in TPU {tpu} is not good")
                print(f"state: {state}")
                print("Unexpected error, please check the TPU manually, or contact the admin")
            return 'failed'
        elif state == 'file error':
            if not quiet:
                print(f"{FAIL} Environment in TPU {tpu} has file error")
                print(f"{INFO} You may need to {PURPLE}mount the NFS{NC} by `tpu mount-disk`, or solve the env by `tpu solve`")
            return 'file error'
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
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout = 120)
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
        return 'timeout'
    except subprocess.CalledProcessError as e:
        print(f"{FAIL} mount_disk: {e}")
        print(f"stderr: {e.stderr}")
        print(f"stdout: {e.stdout}")
        return 'failed'
    
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
        return 'failed'

    
    
    


    


