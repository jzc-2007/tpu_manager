import os
import json
import time
DATA_PATH="/home/jzc/zhichengjiang/working/xibo_tpu_manager/data.json"
def check_status(ka):
    pass

def apply_tpu(ka):
    pass

def rerun_job(job):
    pass

def mainloop():
    dead_job_list = []
    with open(DATA_PATH, 'r') as file:
        data = json.load(file)
    for user in data["user_list"]:
        for job in data["users"][user]["job_data"]:
            if not job["finished"]:
                ka = job["tpu"]
                if 'pre' not in ka:
                    continue
                if check_status(ka) == 'preempted':
                    dead_job_list.append(job)
    for job in dead_job_list:
        ka = job["tpu"]
        result = apply_tpu(ka)
        print(f"Applying TPU {ka} for job {job['job_dir']}, result: {result}")
        if result == 'success':
            print(f'rerun job...')
            rerun_job(job)
    

if __name__ == "__main__":
    while True:
        mainloop()
        time.sleep(600)

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