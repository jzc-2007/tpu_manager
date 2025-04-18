import os, random
import subprocess
from data_io import read_and_lock_data, write_and_unlock_data, release_lock_data, read_data
SESSION_NAME = "tpu_operate"
OPERATE_PATH = "/home/jzc/zhichengjiang/working/xibo_tpu_manager"
def get_zone_pre(tpu):
    """
    Get the zone of the TPU, and check if it is preemptible.
    """
    data = read_data()
    all_tpus = data['all_tpus']
    zone = None
    for z, tpu_list in all_tpus.items():
        if z == 'preemptible':
            continue
        if tpu in tpu_list:
            zone = z
            break
    if zone is None:
        print(f"TPU {tpu} not found in any zone")
        return
    return zone, tpu in data['preemptible_tpus']

def kill_job(tpu):
    zone, pre = get_zone_pre(tpu)
    if zone is None: return
    print(f"Killing jobs in TPU {tpu} in zone {zone}...")
    cmd = '''
    gcloud compute tpus tpu-vm ssh {tpu} --zone {zone} --worker=all \
        --command "pgrep -af python | grep 'main.py' | grep -v 'grep' | awk '{print \"sudo kill -9 \" $1}' | sh"
    '''
    os.system(cmd.format(tpu=tpu, zone=zone))

def reapply_pre(tpu):
    zone, pre = get_zone_pre(tpu)
    if zone is None: return
    if not pre:
        print(f"TPU {tpu} in zone {zone} is not preemptible")
        return
    print(f"Re-apply in TPU {tpu} in zone {zone}...")
    acc_type = None
    if 'v3-32' in tpu: acc_type = 'v3-32'
    elif 'v2-32' in tpu: acc_type = 'v2-32'
    elif 'v4-32' in tpu: acc_type = 'v4-32'
    elif 'v4-8' in tpu: acc_type = 'v4-8'
    
    cmd = f"yes | gcloud compute tpus tpu-vm delete {tpu} --zone={zone} --quiet"
    os.system(cmd)
    cmd = f"yes | gcloud compute tpus tpu-vm create {tpu} --zone={zone} --accelerator-type={acc_type} --version=tpu-ubuntu2204-base --preemptible"
    try:
        subprocess.run(cmd, shell=True, timeout=300, check=True)
    except subprocess.TimeoutExpired:
        print("applying preemptible TPU timed out")
        return 'timeout'

    cmd = f"gcloud compute tpus describe {tpu} --zone={zone} --format='value(state)'"
    state = os.popen(cmd).read().strip()
    if state == 'READY':
        print(f"Now, TPU VM {tpu} is good, ready to use")
        cmd = f"bash xibo_init_pre.sh {tpu} {zone}"
        try:
            subprocess.run(cmd, shell=True, timeout=300, check=True, cwd=OPERATE_PATH)
        except subprocess.TimeoutExpired:
            print("initializing preemptible TPU timed out")
            return 'init failed'
        
        cmd = f"bash test_remote_env.sh {tpu} {zone}"
        try:
            subprocess.run(cmd, shell=True, timeout=300, check=True, cwd=OPERATE_PATH)
        except subprocess.TimeoutExpired:
            print("testing remote env timed out")
            return 'test failed'
        
        return 'success'
    else:
        print(f"TPU {tpu} not ready, state: {state}")
        return 'unknown'


    


    


