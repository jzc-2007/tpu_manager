import os, sys
import json
import time
import multiprocessing
import utils.users as users
import utils.data_io as data_io
import utils.operate as operate
import utils.unit_tests as unit_tests
import utils.jobs as jobs
from utils.helpers import *

running_processes = []

def add_MONITOR_log(log):
    data = data_io.read_and_lock_data()
    try:
        data["MONITOR_logs"].append({
            "time": get_abs_time_str(),
            "msg": log
        })
        data_io.write_and_unlock_data(data)
    except Exception as e:
        print(f"{FAIL} add_MONITOR_log: Failed to add log {log}: {e}")
        data_io.release_lock_data()

def show_MONITOR_log(timezone = 'us'):
    data = data_io.read_data()
    for log in data["MONITOR_logs"]:
        cur_time = log["time"]
        msg = log["msg"]
        show_time = None
        if timezone == 'us':
            show_time = convert_utcstr_to_edtstr(cur_time)
        elif timezone == 'cn':
            show_time = convert_utcstr_to_chnstr(cur_time)
        else:
            show_time = cur_time
        print(f"{LOG} {show_time}: {msg}")

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
        if "Could not open any log file" in line:
            return 'locked'
    return None

def reapply_worker(ka, result_queue):
    sys.stdout = open(os.devnull, 'w')
    try:
        result = operate.apply_pre(ka, delete=True)
        if result == 'success':
            print(f"{GOOD} reapply_worker: Reapply TPU {ka} done")
            add_MONITOR_log(f"{GOOD} reapply_worker: Reapply TPU {ka} done")
        else:
            raise Exception(f"Reapply TPU {ka} failed, please contact the admin")
        result_queue.put(result)
    except Exception as e:
        print(f"{FAIL} reapply_worker: Failed to reapply TPU {ka}: {e}")
        add_MONITOR_log(f"{FAIL} reapply_worker: Failed to reapply TPU {ka}: {e}")
        result_queue.put(e)

def kill_resume(job):
    ka = job["tpu"]
    # print(f"{INFO} kill_resume: Killing jobs  TPU {ka}...")
    operate.kill_jobs_tpu(ka)
    print(f"{INFO} resume job...")
    jobs.resume_rerun_job(job, load_ckpt=True)

def kill_rerun(job):
    ka = job["tpu"]
    # print(f"{INFO} kill_rerun: Killing jobs  TPU {ka}...")
    operate.kill_jobs_tpu(ka)
    print(f"{INFO} rerun job...")
    jobs.resume_rerun_job(job, load_ckpt=False)

def restart_rerun(job):
    ka = job["tpu"]
    # print(f"{INFO} restart_rerun: Killing jobs  TPU {ka}...")
    operate.restart(ka)
    print(f"{INFO} rerun job...")
    jobs.resume_rerun_job(job, load_ckpt=False)


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
                add_MONITOR_log(f"{FAIL} reapply_resume: Reapply TPU {ka} failed: {result}")
            else:
                print(f"{GOOD} Reapply TPU {ka} success: {result}, start resume job")
                jobs.resume_rerun_job(job, load_ckpt=True)
        else:
            print(f"{FAIL} reapply_resume: Reapply TPU {ka} failed, no result returned")
            add_MONITOR_log(f"{FAIL} reapply_resume: Reapply TPU {ka} failed, no result returned")

def mainloop():
    error_jobs = {'preempted': [], 'grpc': [], 'locked': []}
    data = data_io.read_data()
    print(f"{INFO} mainloop: checking jobs")
    for user in data["user_list"]:
        for job in data["users"][user]["job_data"]:
            if job['status'] in ['finished', 'rerunned', 'resumed', 'killed'] or not job['monitor']:
                continue
            status = job['error'] if job['status'] == 'error' else check_job_status(job)
            if status in error_jobs:
                error_jobs[status].append(job)

    print(f"{INFO} mainloop: found {len(error_jobs['preempted'])} preempted jobs, {len(error_jobs['grpc'])} grpc jobs, {len(error_jobs['locked'])} locked jobs")
    if len(error_jobs['locked']) != 0:
        add_MONITOR_log(f"{INFO} Found {len(error_jobs['locked'])} locked jobs, restart them")
    if len(error_jobs['preempted']) != 0:
        add_MONITOR_log(f"{INFO} Found {len(error_jobs['preempted'])} preempted jobs, reapply them")
    if len(error_jobs['grpc']) != 0:
        add_MONITOR_log({
            "msg": f"Found {len(error_jobs['grpc'])} grpc jobs, reapply them"
        })

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
                add_MONITOR_log(f"{FAIL} mainloop: Failed to update job {job['windows_id']} for user {user}")
                data_io.release_lock_data()

    for error_type in error_jobs:
        for job in error_jobs[error_type]:
            rule = job["rules"][error_type]
            try:
                if rule == 'pass':
                    continue
                elif rule == 'reapply':
                    reapply_resume(job, timeout=1800)
                elif rule == 'resume':
                    kill_resume(job)
                elif rule == 'rerun':
                    kill_rerun(job)
                elif rule == 'restart':
                    restart_rerun(job)
            except:
                print(f"{FAIL} mainloop: Failed to handle job {job['windows_id']} for user {user}, (error type {error_type}, rule {rule})")
                add_MONITOR_log(f"{FAIL} mainloop: Failed to handle job {job['windows_id']} for user {user}, (error type {error_type}, rule {rule})")
    

if __name__ == "__main__":
    num_loops = 0
    last_test_time = time.time()
    add_MONITOR_log(f"{GOOD} Starting monitor...")

    if data_io.check_code_lock():
        print(f"{FAIL} Code is locked for developing, please unlock it first.")
        sys.exit(1)
    try:
        while True:
            data = data_io.read_data()
            checking_freq, test_freq = data["MONITOR_config"]["checking_freq"], data["MONITOR_config"]["test_freq"]

            num_loops += 1
            last_time = time.time()
            mainloop()
            cur_time = time.time()
            time_used = cur_time - last_time # in seconds
            print(f"{INFO} Time: {convert_utcstr_to_edtstr(get_abs_time_str())}")
            print(f"Loop {num_loops} finished, time used: {time_used:.2f} seconds")
            time.sleep(max(0, checking_freq - time_used))

            if time.time() - last_test_time > test_freq:
                try:
                    print(f"{INFO} Running unit tests...")
                    passed, failed = unit_tests.sanity_check()
                    tot = passed + failed
                    if failed == 0:
                        add_MONITOR_log(f"{GOOD} All unit tests passed")
                    else:
                        add_MONITOR_log(f"{FAIL} {failed}/{tot} unit tests failed")
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