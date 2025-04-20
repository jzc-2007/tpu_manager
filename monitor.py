import os, sys
import json
import time
import multiprocessing
import utils.users as users
from utils.data_io import read_and_lock_data, write_and_unlock_data, release_lock_data, read_data
from utils.operate import apply_pre, kill_tpu
DATA_PATH="/home/jzc/zhichengjiang/working/xibo_tpu_manager/data.json"
running_processes = []
RED="\033[1;31m"
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
PURPLE="\033[1;34m"
NC="\033[0m"
def check_status(job):
    if job["log_dir"] == '' or job["log_dir"] is None:
        return None
    log_dir = job["log_dir"]+"/output.log"
    if not os.path.exists(log_dir):
        print(f"{RED}[ERROR]{NC} check_status: log file {log_dir} not found")
        return None
    with open(log_dir, 'r') as file:
        lines = file.readlines()
    for line in lines:
        if "This TPU has terminal state \"PREEMPTED\"" in line:
            return 'preempted'
        if "GRPC error" in line:
            return 'grpc'
    return None

def rerun_job(job):
    data = read_and_lock_data()
    try:
        user = data['users'][job["user"]]
        user_obj = users.user_from_dict(user)
        new_stage = int(job['stage']) + 1
        if new_stage > 10:
            print(f"{RED}[ERROR]{NC} rerun_job: job {job['windows_id']} for user {user_obj.name} has reached max stage, cannot rerun")
            release_lock_data()
            return
        id = user_obj.windows_offset
        data['users'][user_obj.name]['windows_offset'] = id + 1
        new_job = {
            'user': user_obj.name,
            'windows_id': id,
            'job_dir_id': job["job_dir_id"],
            'job_dir': job["job_dir"],
            'tpu': job["tpu"],
            'job_tags': job["job_tags"],
            'log_dir': None,
            'extra_configs': job["extra_configs"],
            'finished': False,
            'status': None,
            'stage': new_stage,
            'monitor': job["monitor"],
            'rules': job["rules"],
            'error': None,
            'extra_msgs': job["extra_msgs"] | {"father": job["windows_id"]},
        }
        data['users'][user_obj.name]['job_data'].append(new_job)
        user_obj.windows_offset = id + 1
        data['users'][user_obj.name] = user_obj.to_dict()
        # find the current job in the job_data list and set its status to 'rerunned'
        for jb in data["users"][user_obj.name]["job_data"]:
            if jb["windows_id"] == job["windows_id"]:
                jb["status"] = 'rerunned'
                jb["extra_msgs"].update({"child": id})
        
        session_name = user_obj.tmux_name
        tpu = job["tpu"]
        config_args = job["extra_configs"]
        tags = job["job_tags"]
        job_dir = job["job_dir"]
        log_dir = job["log_dir"]
        print(f"job:{job}, new_job:{new_job}")
        print(f"Rerun job {job['windows_id']} for user {user_obj.name} with new windows id {id}")
        if os.system(f"tmux list-windows -t {session_name} | grep {id}") == 0:
            print(f"Killing tmux window {session_name}:{id}")
            os.system(f"tmux kill-window -t {session_name}:{id}")
            time.sleep(1.5)

                # create the tmux window
        os.system(f"tmux new-window -t {session_name}:{id} -n {tags}")
        time.sleep(0.5)
        os.system(f"tmux send-keys -t {session_name}:{id} 'cd {job_dir}' Enter")
        os.system(f"tmux send-keys -t {session_name}:{id} 'source kill_remote.sh {tpu}; source staging.sh ka={tpu} {config_args} --config.load_from={log_dir}' Enter") 
        
        print(f"Successfully created job in tmux window {session_name}:{id}")

        
        write_and_unlock_data(data)


    except Exception as e:
        print(f"{RED}[ERROR]{NC} rerun_job: Failed to rerun job {job['windows_id']} for user {user_obj.name}, error: {e}")
        release_lock_data()


def reapply_worker(ka, result_queue):
    sys.stdout = open(os.devnull, 'w')
    try:
        result = apply_pre(ka, delete=True)
        result_queue.put(result)
    except Exception as e:
        print(f"{RED}[ERROR]{NC} reapply_worker: Failed to reapply TPU {ka}: {e}")
        result_queue.put(e)

def kill_rerun(job):
    ka = job["tpu"]
    print(f"Kill TPU {ka}...")
    kill_tpu(ka)
    print("Rerun job...")
    rerun_job(job)


def reapply_rerun(job, timeout=1800):
    ka = job["tpu"]
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=reapply_worker, args=(ka, result_queue))
    running_processes.append(process)
    process.start()
    process.join(timeout)
    if process.is_alive():
        print(f"Reapply TPU {ka} timeout, killing the process")
        process.terminate()
        process.join()
        running_processes.remove(process)
        print(f"{YELLOW}[WARNING] {NC}reapply_rerun: Reapply TPU {ka} failed, process killed")
    else:
        if not result_queue.empty():
            result = result_queue.get()
            if isinstance(result, Exception):
                print(f"{RED}[ERROR] {NC}reapply_rerun: Reapply TPU {ka} failed: {result}")
            else:
                print(f"{GREEN}[SUCCESS]{NC} Reapply TPU {ka} success: {result}, start rerun job")
                rerun_job(job)
        else:
            print(f"{RED}[ERROR] {NC}reapply_rerun: Reapply TPU {ka} failed, no result returned")

def mainloop():
    error_jobs = {'preempted': [], 'grpc': []}
    data = read_data()
    print(f"{PURPLE}[INFO]{NC} mainloop: checking jobs")
    for user in data["user_list"]:
        for job in data["users"][user]["job_data"]:
            if job['status'] == 'finished' or job['status'] == 'rerunned' or not job['monitor']:
                continue
            status = job['error'] if job['status'] == 'error' else check_status(job)
            if status == 'preempted':
                error_jobs['preempted'].append(job)
            elif status == 'grpc':
                error_jobs['grpc'].append(job)

    print(f"{PURPLE}[INFO]{NC} mainloop: found {len(error_jobs['preempted'])} preempted jobs and {len(error_jobs['grpc'])} grpc jobs")

    for error_type in error_jobs:
        for job in error_jobs[error_type]:
            user = job["user"]
            data = read_and_lock_data()
            try:
                for jb in data["users"][user]["job_data"]:
                    if jb["windows_id"] == job["windows_id"]:
                        jb["status"] = 'error'
                        jb['error'] = error_type
                write_and_unlock_data(data)
            except:
                print(f"{RED}[ERROR]{NC} mainloop: Failed to update job {job['windows_id']} for user {user}")
                release_lock_data()

    for error_type in error_jobs:
        for job in error_jobs[error_type]:
            rule = job["rules"][error_type]
            if rule == 'pass':
                continue
            elif rule == 'reapply':
                reapply_rerun(job, timeout=1800)
            elif rule == 'rerun':
                kill_rerun(job)
    

if __name__ == "__main__":
    num_loops = 0
    try:
        while True:
            num_loops += 1
            last_time = time.time()
            mainloop()
            cur_time = time.time()
            time_used = cur_time - last_time # in seconds
            print(f"Loop {num_loops} finished, time used: {time_used:.2f} seconds")
            time.sleep(max(0, 600 - time_used))
    except KeyboardInterrupt:
        print("KeyboardInterrupt, exiting...")
        # kill all the processes
        for process in running_processes:
            process.terminate()
            process.join()
        print("All processes killed")