import os, sys
import json
import time
import multiprocessing
import utils.users as users
import utils.data_io as data_io
import utils.operate as operate
import utils.unit_tests as unit_tests
import utils.jobs as jobs
DATA_PATH="/home/jzc/zhichengjiang/working/xibo_tpu_manager/data.json"
running_processes = []
RED, GREEN, YELLOW, PURPLE, NC = "\033[1;31m", "\033[1;32m", "\033[1;33m", "\033[1;34m", "\033[0m"
GOOD, INFO, WARNING, FAIL = f"{GREEN}[GOOD]{NC}", f"{PURPLE}[INFO]{NC}", f"{YELLOW}[WARNING]{NC}", f"{RED}[FAIL]{NC}"
def check_job_status(job):
    if job["log_dir"] == '' or job["log_dir"] is None:
        return None
    tpu = job["tpu"]
    if tpu == '':
        print(f"{FAIL} check_job_status: tpu is empty")
        return None
    tpu_status = operate.check_tpu_status(tpu)
    if tpu_status == 'preempted':
        return 'preempted'
    
    log_dir = job["log_dir"]+"/output.log"
    if not os.path.exists(log_dir):
        print(f"{FAIL} check_tpu_status: log file {log_dir} not found")
        return None
    with open(log_dir, 'r') as file:
        lines = file.readlines()
    for line in lines:
        if "GRPC error" in line:
            return 'grpc'
    return None

def reapply_worker(ka, result_queue):
    sys.stdout = open(os.devnull, 'w')
    try:
        result = operate.apply_pre(ka, delete=True)
        result_queue.put(result)
    except Exception as e:
        print(f"{FAIL} reapply_worker: Failed to reapply TPU {ka}: {e}")
        result_queue.put(e)

def kill_resume(job):
    ka = job["tpu"]
    print(f"{INFO} kill_resume:Kill TPU {ka}...")
    operate.kill_jobs_tpu(ka)
    print(f"{INFO} resume job...")
    jobs.resume_rerun_job(job, load_ckpt=True)


def reapply_resume(job, timeout=1800):
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
        print(f"{WARNING} reapply_resume: Reapply TPU {ka} failed, process killed")
    else:
        if not result_queue.empty():
            result = result_queue.get()
            if isinstance(result, Exception):
                print(f"{FAIL} reapply_resume: Reapply TPU {ka} failed: {result}")
            else:
                print(f"{GOOD} Reapply TPU {ka} success: {result}, start resume job")
                jobs.resume_rerun_job(job, load_ckpt=True)
        else:
            print(f"{FAIL} reapply_resume: Reapply TPU {ka} failed, no result returned")

def mainloop():
    error_jobs = {'preempted': [], 'grpc': []}
    data = data_io.read_data()
    print(f"{INFO} mainloop: checking jobs")
    for user in data["user_list"]:
        for job in data["users"][user]["job_data"]:
            if job['status'] in ['finished', 'rerunned', 'resumed', 'killed'] or not job['monitor']:
                continue
            status = job['error'] if job['status'] == 'error' else check_job_status(job)
            if status == 'preempted':
                error_jobs['preempted'].append(job)
            elif status == 'grpc':
                error_jobs['grpc'].append(job)

    print(f"{INFO} mainloop: found {len(error_jobs['preempted'])} preempted jobs and {len(error_jobs['grpc'])} grpc jobs")

    for error_type in error_jobs:
        for job in error_jobs[error_type]:
            user = job["user"]
            data = data_io.read_and_lock_data()
            try:
                for jb in data["users"][user]["job_data"]:
                    if jb["windows_id"] == job["windows_id"]:
                        jb["status"] = 'error'
                        jb['error'] = error_type
                data_io.write_and_unlock_data(data)
            except:
                print(f"{FAIL} mainloop: Failed to update job {job['windows_id']} for user {user}")
                data_io.release_lock_data()

    for error_type in error_jobs:
        for job in error_jobs[error_type]:
            rule = job["rules"][error_type]
            if rule == 'pass':
                continue
            elif rule == 'reapply':
                reapply_resume(job, timeout=1800)
            elif rule == 'resume':
                kill_resume(job)
    

if __name__ == "__main__":
    num_loops = 0
    last_test_time = time.time()

    if data_io.check_code_lock():
        print(f"{FAIL} Code is locked for developing, please unlock it first.")
        sys.exit(1)
    try:
        while True:
            data = data_io.read_data()
            checking_freq, test_freq = data["monitor_config"]["checking_freq"], data["monitor_config"]["test_freq"]

            num_loops += 1
            last_time = time.time()
            mainloop()
            cur_time = time.time()
            time_used = cur_time - last_time # in seconds
            print(f"Loop {num_loops} finished, time used: {time_used:.2f} seconds")
            time.sleep(max(0, checking_freq - time_used))

            if time.time() - last_test_time > test_freq:
                try:
                    print(f"{INFO} Running unit tests...")
                    unit_tests.sanity_check()
                except Exception as e:
                    print(f"{FAIL} Unit tests failed: {e}")
                last_test_time = time.time()

                
    except KeyboardInterrupt:
        print("KeyboardInterrupt, exiting...")
        # kill all the processes
        for process in running_processes:
            process.terminate()
            process.join()
        print("All processes killed")