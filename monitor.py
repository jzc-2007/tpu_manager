import os
import json
import time
import multiprocessing
import utils.users as users
from utils.data_io import read_and_lock_data, write_and_unlock_data, release_lock_data, read_data
from utils.operate import reapply_pre
DATA_PATH="/home/jzc/zhichengjiang/working/xibo_tpu_manager/data.json"
running_processes = []
def check_status(job):
    if job["log_dir"] == '':
        return None
    log_dir = job["log_dir"]+"/output.log"
    if not os.path.exists(log_dir):
        return None
    with open(log_dir, 'r') as file:
        lines = file.readlines()
    # Check if there is: This TPU has terminal state "PREEMPTED", so it cannot be used anymore. in log
    for line in lines:
        if "This TPU has terminal state \"PREEMPTED\"" in line:
            return 'preempted'
        if "GRPC error" in line:
            return 'grpc'

def rerun_job(job):
    #{
    #     'windows_id': id,
    #     'job_dir_id': dir,
    #     'job_dir': dir_path,
    #     'tpu': tpu,
    #     'job_tags': tag,
    #     'log_dir': None,
    #     'extra_configs': config_args,
    #     'finished': False,
    #     'status': '0',
    #     'monitor': monitor,
    #     'rules': rule,
    #}
    data = read_and_lock_data()
    try:
        user = data['users'][job["user"]]
        user_obj = users.user_from_dict(user)
        new_status = str(int(job["status"]) + 1)
        job["status"] = new_status
        id = user_obj.windows_offset
        data['users'][user_obj.name]['windows_offset'] = id + 1
        new_job = {
            'windows_id': id,
            'job_dir_id': job["job_dir_id"],
            'job_dir': job["job_dir"],
            'tpu': job["tpu"],
            'job_tags': job["job_tags"],
            'log_dir': None,
            'extra_configs': job["extra_configs"],
            'finished': False,
            'status': new_status,
            'monitor': job["monitor"],
            'rules': job["rules"],
        }
        data['users'][user_obj.name]['job_data'].append(new_job)
        user_obj.windows_offset = id + 1
        data['users'][user_obj.name] = user_obj.to_dict()
        
        session_name = user_obj.tmux_name
        tpu = job["tpu"]
        config_args = job["extra_configs"]
        print(f"Rerun job {job['windows_id']} for user {user_obj.name} with new windows id {id}")
        if os.system(f"tmux list-windows -t {session_name} | grep {id}") == 0:
            print(f"Killing tmux window {session_name}:{id}")
            os.system(f"tmux kill-window -t {session_name}:{id}")
            time.sleep(0.5)

                # create the tmux window
        os.system(f"tmux new-window -t {session_name}:{id} -n {job["job_tags"]}")
        time.sleep(0.5)
        os.system(f"tmux send-keys -t {session_name}:{id} 'cd {job["job_dir"]}' Enter")
        os.system(f"tmux send-keys -t {session_name}:{id} 'source kill_remote.sh {tpu}; source staging.sh ka={tpu} {config_args} --config.load_from={job['log_dir']}' Enter") 
        
        print(f"Successfully created job in tmux window {session_name}:{id}")

        
        write_and_unlock_data(data)


    except:
        release_lock_data()


def reapply_worker(ka, result_queue):
    try:
        result = reapply_pre(ka)
        result_queue.put(result)
    except Exception as e:
        result_queue.put(e)


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
        print(f"Reapply TPU {ka} timeout")
    else:
        if not result_queue.empty():
            result = result_queue.get()
            if isinstance(result, Exception):
                print(f"Reapply TPU {ka} failed: {result}")
            else:
                print(f"Reapply TPU {ka} success: {result}, start rerun job")
                rerun_job(job)
        else:
            print(f"Reapply TPU {ka} failed, no result returned")

def mainloop():
    dead_job_list = []
    data = read_data()
    for user in data["user_list"]:
        for job in data["users"][user]["job_data"]:
            if not job["finished"] and job['monitor']:
                if check_status(job) == 'preempted':
                    dead_job_list.append(job)
                    # log the dead job
                    data = read_and_lock_data()
                    try:
                        for job in data["users"][user]["job_data"]:
                            if job["windows_id"] == job["windows_id"]:
                                job['error'] = 'preempted'
                        write_and_unlock_data(data)
                    except:
                        release_lock_data()
    for job in dead_job_list:
        if job["rules"] == 'auto-reapply':
            reapply_rerun(job, timeout=1800)
    

if __name__ == "__main__":
    try:
        while True:
            last_time = time.time()
            mainloop()
            cur_time = time.time()
            time_used = cur_time - last_time # in seconds
            time.sleep(max(0, 1800 - time_used))
    except KeyboardInterrupt:
        print("KeyboardInterrupt, exiting...")
        # kill all the processes
        for process in running_processes:
            process.terminate()
            process.join()
        print("All processes killed")

# DATA 
# {
#     "users":{
#        "user1": {
#             "id": 1,
#             "name": "user1",
#             "tmux_name": "user1",
#             "working_dir": "/home/user1",
#             "config_aliases": {
#                 "lr": "config.training.learning_rate",
#                 "bs": "config.training.batch_size"
#             },
#             "settings": {
#                 "auto attach": True,
#             }
#             "job_data":[
#                 {
#                     "windows_id": 1,
#                     "job_dir_id": 1,
#                     "job_dir": "/home/user1/job1",
#                     "tpu": "v2-32-1",
#                     "job_tags": "residual",
#                     "log_dir": "/home/user1/job1/logs",
#                     "extra_configs": "--config1=value1 --config2=value2",
#                     "finished": false
#                 }
#             ]
#         }
#     },
#     "tpu_aliases": {
#     },
#     "user_list": [
#         "user1",
#     ],
#     "id_list": [
#         1,
#     ],
#     "id_user_dict": {
#         1: "user1",
#     },
#     "user_id_dict": {
#         "user1": 1,
#     },
# }