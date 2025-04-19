import os, random
import subprocess
from .data_io import read_and_lock_data, write_and_unlock_data, release_lock_data, read_data
RED="\033[1;31m"
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
PURPLE="\033[1;34m"
NC="\033[0m"
OPERATE_PATH = "/home/jzc/zhichengjiang/working/xibo_tpu_manager"
def get_zone_pre(tpu):
    """
    Get the zone of the TPU, and check if it is preemptible.
    If the input is alias, it will be replaced with the real TPU name.
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
        print(f"{RED}[ERROR]{NC} get_zone_pre: TPU {tpu} not found")
        return
    all_tpus = data['all_tpus']
    zone = None
    for z, tpu_list in all_tpus.items():
        if z == 'preemptible':
            continue
        if tpu in tpu_list:
            zone = z
            break
    if zone is None:
        print(f"{RED}[ERROR]{NC} get_zone_pre: TPU {tpu} not found in any zone")
        return None, None
    return zone, tpu in data['all_tpus']['preemptible'], tpu

def kill_tpu(tpu):
    zone, pre, tpu = get_zone_pre(tpu)
    if zone is None: return
    print(f"Killing jobs in TPU {tpu} in zone {zone}...")
    cmd = '''
    gcloud compute tpus tpu-vm ssh {tpu} --zone {zone} --worker=all \
        --command "pgrep -af python | grep 'main.py' | grep -v 'grep' | awk '{print \"sudo kill -9 \" $1}' | sh"
    '''
    os.system(cmd.format(tpu=tpu, zone=zone))

def apply_pre(tpu, delete=True):
    zone, pre, tpu = get_zone_pre(tpu)
    if zone is None: return
    if not pre:
        print(f"{RED}[ERROR]{NC} apply_pre: TPU {tpu} in zone {zone} is not preemptible")
        return
    print(f"Re-apply in TPU {tpu} in zone {zone}...")
    acc_type = None
    if 'v3-32' in tpu: acc_type = 'v3-32'
    elif 'v2-32' in tpu: acc_type = 'v2-32'
    elif 'v4-32' in tpu: acc_type = 'v4-32'
    elif 'v4-8' in tpu: acc_type = 'v4-8'
    else: raise ValueError(f"{RED}[ERROR]{NC} apply_pre: Unknown TPU type {tpu}")
    if delete:
        cmd = f"gcloud compute tpus tpu-vm delete {tpu} --zone={zone} --quiet"
        try:
            subprocess.run(cmd.split(), timeout=300, check=True)
        except subprocess.CalledProcessError as e:
            print(f"{RED}[ERROR]{NC} apply_pre: TPU deletion failed: {e}")
            return 'delete failed'

    cmd = f"gcloud compute tpus tpu-vm create {tpu} --zone={zone} --accelerator-type={acc_type} --version=tpu-ubuntu2204-base --preemptible"
    try:
        subprocess.run(cmd, shell=True, timeout=600, check=True)
    except subprocess.TimeoutExpired:
        print("{RED}[ERROR]{NC} apply_pre: applying preemptible TPU timed out")
        return 'timeout'

    cmd = f"gcloud compute tpus describe {tpu} --zone={zone} --format='value(state)'"
    try:
        state = subprocess.check_output(cmd, shell=True).decode().strip()
    except subprocess.CalledProcessError:
        print("{RED}[ERROR]{NC} apply_pre: Failed to query TPU state")
        return 'describe failed'

    if state == 'READY':
        print(f"Now, TPU VM {tpu} is good, ready to use")
        cmd = f"bash xibo_init_pre.sh {tpu} {zone}"
        try:
            subprocess.run(cmd, shell=True, timeout=600, check=True, cwd=OPERATE_PATH)
        except subprocess.TimeoutExpired:
            print("{RED}[ERROR]{NC} apply_pre: initializing preemptible TPU timed out")
            return 'init failed'
        
        cmd = f"bash test_remote_env.sh {tpu} {zone}"
        try:
            subprocess.run(cmd, shell=True, timeout=300, check=True, cwd=OPERATE_PATH)
        except subprocess.TimeoutExpired:
            print("{RED}[ERROR]{NC} apply_pre: testing remote env timed out")
            return 'test failed'
        
        return 'success'
    else:
        print(f"{RED}[ERROR]{NC} apply_pre: TPU {tpu} not ready, state: {state}")
        return 'unknown'


    


    


