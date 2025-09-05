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
import utils.sheet as sheet
import utils.clean as clean
import utils.autenticate as autenticate
import utils.queue as queue
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

    if cmd in ['add-user', 'del-user']:
        if not autenticate.autenticate(cmd):
            print(f"{FAIL} Authentication failed, exiting.")
            sys.exit(1)

    if cmd == 'lock':
        data_io.lock(args[2:])
        exit(0)
    elif cmd == 'rl' or cmd == 'unlock':
        data_io.release_lock(args[2:])
        exit(0)
    if data_io.check_code_lock():
        print(f"{FAIL} Code is locked for developing, please unlock it first.")
        sys.exit(1)
    try:
        ############### JOBS that don't require a user ###############
        if cmd == 'tldr': desc.tldr()
        # elif cmd == 'docs' or cmd == 'doc': desc.full_doc()
        if cmd == 'vq': queue.visualize_queue() if len(args) < 3 else queue.visualize_queue(user = args[2])
        elif cmd == 'dqr': queue.dequeue_and_run(args[2], args[3])
        elif cmd == 'ssn': sheet.set_spreadsheet_notes(args[2], args[3])
        elif cmd == 'asn': sheet.add_spreadsheet_notes(args[2], args[3])
        elif cmd == 'change-ip': handler.change_ip()
        elif cmd == 'upd-log': jobs.upd_log(args[2], args[3], args[4], args[5], args[6]) #windows, log_dir, stage_dir, ka, time
        elif cmd =='upd-staging-info': queue.upd_staging_info(args[2], args[3], args[4])
        elif cmd == 'finish-job': queue.finish_job(args[2])
        elif cmd == 'fail-job': queue.fail_job(args[2])
        elif cmd == 'help' or cmd == '-h': desc.explain(args[2])
        elif cmd == 'add-tpu-alias' or cmd == '-ta' or cmd == '-ata': logger.add_tpu_alias(args[2], args[3])
        elif cmd == 'add-applied-tpu' or cmd =='add-applied' or cmd == '-atpu' or cmd == 'register': logger.register_tpu()
        elif cmd == 'del-registered' or cmd == 'del-register' or cmd == 'del-info' or cmd == 'del-reg': logger.del_registered_tpu(args[2])
        elif cmd == 'check-status' or cmd == '-cktpu': print(operate.check_tpu_status(args[2]))
        elif cmd == 'describe' or cmd == '-dtpu': operate.describe_tpu(args[2])
        elif cmd == '-lta' or cmd == '-sta': logger.explain_tpu_aliases()
        elif cmd == 'add-user': users.create_user()
        elif cmd == 'del-user': users.del_user()
        elif cmd == 'check-env': operate.check_env(args[2])
        elif cmd == 'list-users' or cmd == '-lu': users.list_users()
        elif cmd == 'init': handler.initialization()
        elif cmd == 'check-rules': jobs.check_rules()
        elif cmd == 'upd-status-spreadsheet' or cmd == 'uss': operate.update_tpu_status_for_spreadsheet()
        elif cmd == 'reapply': operate.reapply(args[2:])
        elif cmd == 'reapplyy': operate.reapply_until_success(args[2:])
        elif cmd == 'apply': operate.apply(args[2:])
        elif cmd == 'applyy': operate.apply_until_success(args[2:])
        elif cmd == 'delete': operate.delete_tpu(args[2])
        elif cmd == 'restart': operate.restart(args[2])
        elif cmd == 'apply-norm': operate.apply_and_set_env(args[2], preemptible=False, delete=False)
        elif cmd == 'reapply-norm': operate.apply_and_set_env(args[2], preemptible=False, delete=True)
        elif cmd == 'solve' or cmd == 'solve-env': handler.solve_env(args[2])
        elif cmd == 'mount-disk': operate.mount_disk(args[2])
        elif cmd == 'set-wandb': operate.set_wandb(args[2])
        elif cmd == 'set-monitor-config' or cmd == '-smc': logger.set_monitor_config(args[2:])
        elif cmd == 'get-monitor-config' or cmd == '-gmc': logger.get_monitor_config()
        elif cmd == 'maj': jobs.monitor_all_jobs()
        elif cmd == 'caj': jobs.check_all_jobs(args[2:])
        elif cmd == 'lock-data': data_io.lock_data()
        elif cmd == 'unlock-data': data_io.release_lock_data()
        elif cmd == 'ack': jobs.ack_MONITOR()
        # elif cmd == 'rs': sheet.read_sheet_info()
        elif cmd == 'find': sheet.find_tpu_from_type(args[2:])
        elif cmd == 'rel' or cmd == 'release': sheet.release_tpu(args[2:])
        elif cmd == 'clean-eu': clean.clean_eu(safe = ('-f' not in args), quiet = ('-q' in args))
        elif cmd == 'clean-us': clean.clean_us(safe = ('-f' not in args), quiet = ('-q' in args))

        # ------------ For development only ------------
        elif cmd == 'test': operate.test_remote(args[2])
        elif cmd == 'add_global_config' or cmd == '-agc': develop.add_global_config(args[2], args[3])
        elif cmd == 'merge_global_config' or cmd == '-mgc': develop.merge_global_config(args[2])
        elif cmd == '-Ml': develop.show_MONITOR_log() if len(args) < 3 else develop.show_MONITOR_log(args[2])
        elif cmd == '-Mc': develop.clear_MONITOR_log()
        elif cmd == 'debug-stats': develop.debug_stats(args[2])
        elif cmd == 'debug-kill': develop.kill_jobs_tpu_new(args[2])
        elif cmd == 'gtis' or cmd == 'get-tpu-info-sheet': print(sheet.get_tpu_info_sheet(args[2]))
        elif cmd == 'twsi': unit_tests.test_write_sheet_info(args[2])
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
            elif cmd == 'del-dir': dirs.del_dir(user_object, args[2:])
            elif cmd == 'swap-dir': dirs.swap_dir(user_object, args[2:])
            elif cmd == 'get-settings': logger.get_settings(user_object)
            elif cmd == 'set-settings': logger.set_settings(user_object, args[2:])
            elif cmd == 'reset-settings': users.reset_settings(user_object)
            elif cmd == 'get-dir': print(dirs.get_dir(user_object, args[2]))
            elif cmd == 'check': jobs.check_jobs(user_object, args[2:])
            elif cmd == 'monitor': jobs.monitor_jobs(user_object, args[2:])
            elif cmd == 'kill-job' or cmd == '-kj' or cmd == '-k' or cmd == 'kill': jobs.kill_job_or_tpu(user_object, args[2:])
            elif cmd == 'run': jobs.run(user_object, args[2:])
            elif cmd == 'queue': queue.Queue(user_object, args[2:])
            elif cmd == 'resume': jobs.resume(user_object, args[2:])
            elif cmd == 'rerun': jobs.rerun(user_object, args[2:])
            # elif cmd == 'copy-cfg': dirs.read_job_config_to_file(user_object, args[2])
            elif cmd == 'get-stage-dir': print(dirs.get_job_stage_dir(user_object, args[2]))
            elif cmd == 'dequeue' or cmd == 'dq': queue.dequeue(user_object, args[2:])
            elif cmd == 'ignore-error': jobs.ignore_error(user_object, args[2:])
            elif cmd == 'restart-run': jobs.restart_run(user_object, args[2:])
            elif cmd == 'ls' or cmd == 'lsdir': dirs.list_dir(user_object, args[2:])
            elif cmd == 'kill-window' or cmd == '-kw': jobs.kill_window(user_object, args[2:])
            elif cmd == 'add-config-alias' or cmd == '-a' or cmd == '-alias': logger.add_config_alias(user_object, args[2:])
            elif cmd == 'show-config-alias' or cmd == '-sa' or cmd == '-la': logger.show_config_alias(user_object)
            elif cmd == 'del-config-alias': logger.del_config_alias(user_object, args[2:])
            elif cmd == 'add-tag': jobs.add_tag(user_object, args[2], args[3])
            elif cmd == 'clear-finished': jobs.clear_finished_jobs(user_object)
            elif cmd == 'clear-error': jobs.clear_error_jobs(user_object)
            elif cmd == 'clear-all' or cmd == 'clear': jobs.clear_all_jobs(user_object)
            elif cmd == 'reset-window-num' or cmd == 'reset-window': users.reset_window_num(user_object, args[2:])
            elif cmd == '-czw': handler.clear_zombie_windows(user_object)
            elif cmd == '-czj': jobs.clear_zombie_jobs(user_object)
            elif cmd == 'clean': jobs.clear_all_jobs(user_object, args[2:]), handler.clear_zombie_windows(user_object), jobs.clear_zombie_jobs(user_object)

            else: print(f"{FAIL} Unknown command {cmd}")
    except Exception as e:
        print(f"{FAIL} {e}")
        sys.exit(1)


