import json, sys, os
import utils.descriptions as desc
import utils.directories as dirs
import utils.users as users
import utils.jobs as jobs
import utils.logger as logger
import utils.operate as operate
import utils.error_handler as handler
import utils.data_io as data_io
import utils.unit_tests as unit_tests
import utils.develop as develop
from utils.helpers import *

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
        print(f"{FAIL} No command provided")
        sys.exit(1)

    cmd = args[1]

    if cmd == 'lock-code' or cmd == '-lc': 
        data_io.lock_code(args[2]) if len(args) > 2 else data_io.lock_code()
        exit(0)
    elif cmd == 'unlock-code' or cmd == '-ulc':
        data_io.unlock_code(args[2]) if len(args) > 2 else data_io.unlock_code()
        exit(0)

    if data_io.check_code_lock():
        print(f"{FAIL} Code is locked for developing, please unlock it first.")
        sys.exit(1)
    try:
        ############### JOBS that don't require a user ###############
        if cmd == 'tldr': desc.tldr()
        elif cmd == 'change-ip': handler.change_ip()
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
        elif cmd == 'check-rules': jobs.check_rules()
        elif cmd == 'reapply': operate.apply_pre(args[2], delete=True)
        elif cmd == 'apply': operate.apply_pre(args[2], delete=False)
        elif cmd == 'apply-norm': operate.apply_norm(args[2], delete=False)
        elif cmd == 'reapply-norm': operate.apply_norm(args[2], delete=True)
        elif cmd == 'solve' or cmd == 'solve-env': handler.solve_env(args[2])
        elif cmd == 'mount-disk': operate.mount_disk(args[2])
        elif cmd == 'set-wandb': operate.set_wandb(args[2])
        elif cmd == 'kill-tpu' or cmd == '-kt': operate.kill_jobs_tpu(args[2])
        elif cmd == 'set-monitor-config' or cmd == '-smc': logger.set_monitor_config(args[2:])
        elif cmd == 'get-monitor-config' or cmd == '-gmc': logger.get_monitor_config()
        elif cmd == 'maj': jobs.monitor_all_jobs()
        elif cmd == 'caj': jobs.check_all_jobs()
        elif cmd == 'lock-data': data_io.lock_data()
        elif cmd == 'unlock-data': data_io.release_lock_data()

        # ------------ For development only ------------
        elif cmd == 'add_global_config' or cmd == '-agc': develop.add_global_config(args[2], args[3])
        elif cmd == 'merge_global_config' or cmd == '-mgc': develop.merge_global_config(args[2])
        elif cmd == '-Ml': develop.show_MONITOR_log() if len(args) < 3 else develop.show_MONITOR_log(args[2])
        elif cmd == '-Mc': develop.clear_MONITOR_log()
        # ------------ End of development only ------------

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
            elif cmd == 'reset-user-settings': users.reset_settings(user_object)
            elif cmd == 'get-dir': print(dirs.get_dir(user_object, args[2]))
            elif cmd == 'check': jobs.check_jobs(user_object, args[2:])
            elif cmd == 'monitor': jobs.monitor_jobs(user_object, args[2:])
            elif cmd == 'kill-job' or cmd == '-kj' or cmd == '-k': jobs.kill_job(user_object, args[2:])
            elif cmd == 'run': jobs.run(user_object, args[2:])
            elif cmd == 'resume': jobs.resume(user_object, args[2:])
            elif cmd == 'rerun': jobs.rerun(user_object, args[2:])
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
    except Exception as e:
        print(f"{FAIL} {e}")
        sys.exit(1)


