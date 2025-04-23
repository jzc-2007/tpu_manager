import json
import os
import time
import re
import sys
import utils.descriptions as desc
import utils.directories as dirs
import utils.users as users
import utils.jobs as jobs
import utils.logger as logger
import utils.operate as operate
from utils.helpers import is_integer, is_boolean, to_boolean, DATA_PATH
import utils.error_handler as handler
import utils.data_io as data_io
import utils.unit_tests as unit_tests
RED, GREEN, YELLOW, PURPLE, NC = "\033[1;31m", "\033[1;32m", "\033[1;33m", "\033[1;34m", "\033[0m"
def find_user(data, args):
    for arg in args:
        if arg in data['user_list'] or arg.startswith('id=') or arg.startswith('user='):
            if arg.startswith('id='):
                id = arg.split('=')[1]
                assert id in data['id_user_dict'], f"User with id {id} not found"
                return data['id_user_dict'][id]
            elif arg.startswith('user='):
                user = arg.split('=')[1]
                assert user in data['user_list'], f"User {user} not found"
                return user
            else:
                return arg
    return None

def input_user(data):
    res = input(f'Please enter user name/id, or empty for default user 0({data["id_user_dict"]["0"]}):')
    if res == '':
        return data['id_user_dict']["0"]
    elif res in data['user_list']:
        return res
    elif res in data['id_user_dict']:
        return data['id_user_dict'][res]
    else:
        raise ValueError(f"User {res} not found")

if __name__ == '__main__':
    args = sys.argv

    if len(args) < 2:
        print(f"{RED}[ERROR]{NC} No command provided")
        sys.exit(1)

    cmd = args[1]

    if cmd == 'lock-code' or cmd == '-lc': 
        data_io.lock_code(args[2]) if len(args) > 2 else data_io.lock_code()
        exit(0)
    elif cmd == 'unlock-code' or cmd == '-ulc':
        data_io.unlock_code(args[2]) if len(args) > 2 else data_io.unlock_code()
        exit(0)

    if data_io.check_code_lock():
        print(f"{RED}[ERROR]{NC} Code is locked for developing, please unlock it first.")
        sys.exit(1)

    ############### JOBS that don't require a user ###############
    if cmd == 'tldr': desc.tldr()
    elif cmd == 'upd-log': jobs.upd_log(args[2], args[3], args[4], args[5]) #windows, log_dir, ka, time
    elif cmd == 'finish-job': jobs.finish_job(args[2])
    elif cmd == 'help' or cmd == '-h': desc.explain(args[2])
    elif cmd == 'add-tpu-alias' or cmd == '-ta': logger.add_tpu_alias(args[2], args[3])
    elif cmd == 'check-status' or cmd == '-cktpu': print(operate.check_tpu_status(args[2]))
    elif cmd == 'describe' or cmd == '-dtpu': operate.describe_tpu(args[2])
    elif cmd == '-lta': logger.explain_tpu_aliases()
    elif cmd == 'add-user': users.create_user()
    elif cmd == 'del-user': users.del_user()
    elif cmd == 'check-env': operate.check_env(args[2])
    elif cmd == 'list-users' or cmd == '-lu': users.list_users()
    elif cmd == 'init': handler.initialization()
    elif cmd == 'reapply': operate.apply_pre(args[2], delete=True)
    elif cmd == 'apply': operate.apply_pre(args[2], delete=False)
    elif cmd == 'solve' or cmd == 'solve-env': handler.solve_env(args[2])
    elif cmd == 'mount-disk': operate.mount_disk(args[2])
    elif cmd == 'set-wandb': operate.set_wandb(args[2])
    elif cmd == 'kill-jobs' or cmd == '-k' or cmd =='-kj': operate.kill_jobs(args[2])
    elif cmd == 'set-monitor-config' or cmd == '-smc': logger.set_monitor_config(args[2:])
    elif cmd == 'get-monitor-config' or cmd == '-gmc': logger.get_monitor_config()
    else: 
    ############### JOBS that require a user ###############
        with open(DATA_PATH, 'r') as file: data = json.load(file)
        user = find_user(data, args[1:])
        if user is None: user = input_user(data)
        user = data['users'][user]
        user_object = users.user_from_dict(user)
        if cmd == 'set-cur': dirs.set_cur(user_object, args[2:])
        elif cmd == 'set-dir': dirs.set_dir(user_object, args[2:])
        elif cmd == 'get-settings': logger.get_settings(user_object)
        elif cmd == 'set-settings': logger.set_settings(user_object, args[2:])
        elif cmd == 'get-dir': print(dirs.get_dir(user_object, args[2]))
        elif cmd == 'check': jobs.check_jobs(user_object, args[2:])
        elif cmd == 'monitor': jobs.monitor_jobs(user_object, args[2:])
        elif cmd == 'run': jobs.run(user_object, args[2:])
        elif cmd == 'ls' or cmd == 'lsdir': dirs.list_dir(user_object, args[2:])
        elif cmd == 'kill-window' or cmd == '-kw': jobs.kill_window(user_object, args[2:])
        elif cmd == 'add-config-alias' or cmd == '-a' or cmd == '-alias': logger.add_config_alias(user_object, args[2:])
        elif cmd == 'show-config-alias' or cmd == '-sa': logger.show_config_alias(user_object)
        elif cmd == 'del-config-alias': logger.del_config_alias(user_object, args[2:])
        elif cmd == 'add-tag': jobs.add_tag(user_object, args[2], args[3])
        elif cmd == 'clear-finished': jobs.clear_finished_jobs(user_object)
        elif cmd == 'clear-error': jobs.clear_error_jobs(user_object)
        elif cmd == 'clear-all' or cmd == 'clear': jobs.clear_all_jobs(user_object)
        elif cmd == '-czw': handler.clear_zombie_windows(user_object)
        elif cmd == '-czj': jobs.clear_zombie_jobs(user_object)
        elif cmd == 'clean': jobs.clear_all_jobs(user_object), handler.clear_zombie_windows(user_object), jobs.clear_zombie_jobs(user_object)
        else: print(f"Unknown command {cmd}")

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


