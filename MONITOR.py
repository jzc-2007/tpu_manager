import os, sys, subprocess
import json
import time
import multiprocessing
import re
import ast
import utils.users as users
import utils.data_io as data_io
import utils.operate as operate
import utils.unit_tests as unit_tests
import utils.jobs as jobs
import utils.clean as clean
from utils.helpers import *

running_processes = []
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
TYPE_RE = re.compile(r"^(v[0-9a-z]+-\d+)")
ZONE_RE = re.compile(r"(us|asia|europe|australia|northamerica|southamerica)-[a-z0-9-]+-[a-z]")

_tpu = 'python /home/jzc/zhichengjiang/working/xibo_tpu_manager/tpu.py'
_tou = 'python /kmh-nfs-ssd-us-mount/code/qiao/work/tpu_dls/wrap_master.py'

def read_sqa():
    with open(f'/kmh-nfs-ssd-us-mount/code/qiao/work/tpu_manager/sqa.json', 'r') as file:
        x = file.read()
        x = json.loads(x)
    return x

def write_sqa(window_id):
    x = read_sqa()
    x['running'].append(window_id)
    with open(f'/kmh-nfs-ssd-us-mount/code/qiao/work/tpu_manager/sqa.json', 'w') as file:
        # write as json
        json.dump(x, file)
        file.write('\n')
    return

def remove_sqa(window_id):
    x = read_sqa()
    x['running'].remove(window_id)
    with open(f'/kmh-nfs-ssd-us-mount/code/qiao/work/tpu_manager/sqa.json', 'w') as file:
        # write as json
        json.dump(x, file)
        file.write('\n')
    return

def finish_sqa(window_id):
    x = read_sqa()
    x['running'].remove(window_id)
    x['finished'].append(window_id)
    with open(f'/kmh-nfs-ssd-us-mount/code/qiao/work/tpu_manager/sqa.json', 'w') as file:
        # write as json
        json.dump(x, file)
        file.write('\n')
    return

def add_MONITOR_log(log):
    # data = data_io.read_and_lock_data()
    # try:
    #     data["MONITOR_logs"].append({
    #         "time": get_abs_time_str(),
    #         "msg": log
    #     })
    #     data_io.write_and_unlock_data(data)
    # except Exception as e:
    #     print(f"{FAIL} add_MONITOR_log: Failed to add log {log}: {e}")
    #     data_io.release_lock_data()
    with open(f'/kmh-nfs-ssd-us-mount/code/qiao/work/tpu_manager/output.log', 'a') as file:
        file.write(f"{get_abs_time_str()}: {log}\n")
    print(f"{get_abs_time_str()}: {log}")
    return

def _append_resume_file_log(window_id, command_name, command, result):
    log_dir = os.path.join("logs", str(window_id))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{command_name}.txt")

    with open(log_path, "a") as log_file:
        log_file.write(f"[{get_abs_time_str()}] {command_name}\n")
        log_file.write(f"cmd: {command}\n")
        log_file.write(f"returncode: {result.returncode}\n")
        log_file.write("stdout:\n")
        log_file.write((result.stdout or "") + "\n")
        log_file.write("stderr:\n")
        log_file.write((result.stderr or "") + "\n")
        log_file.write("-" * 80 + "\n")

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
    
def avilable_aliases(tpu_type: str, zone):
    if zone.startswith('us-central1'):
        return [tpu_type + '-tmp' + str(i) for i in range(2, 9)]
    elif zone.startswith('us-east5'):
        if tpu_type.startswith('v5p'):
            return [tpu_type + '-tmp' + str(i) for i in range(201, 209)]
        if tpu_type.startswith('v6e'):
            return [tpu_type + '-tmp' + str(i) for i in range(51, 59)]
    else:
        raise ValueError(f"Invalid zone: {zone}")

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
    if tpu_status == 'deleted':
        return 'deleted'
    
    return None

def _strip_ansi(text):
    """
    去掉颜色代码
    """
    return ANSI_ESCAPE_RE.sub('', text or '')

def _extract_tpu_type(name):
    """
    Extract TPU type from TPU name
    Example:
        kmh-tpuvm-v6e-64-spot-sqa24jgcg -> v6e-64
        kmh-tpuvm-v5p-64-spot-llqnomzp1 -> v5p-64
        kmh-tpuvm-v6e-32-sqa-3210 -> v6e-32
        kmh-tpuvm-v5e-32-spot-gzy24jgcg -> v5e-32
        kmh-tpuvm-v6e-64-spot-sqa24jgcg -> v6e-64
        kmh-tpuvm-v5p-64-spot-llqnomzp1 -> v5p-64
        kmh-tpuvm-v6e-32-sqa-3210 -> v6e-32
    """
    if not name:
        return None
    norm_name = name.strip()
    if norm_name.startswith('kmh-tpuvm-'):
        norm_name = norm_name[len('kmh-tpuvm-'):]
    match = TYPE_RE.match(norm_name)
    return match.group(1) if match else None

def _extract_zone(tpu_name):
    """
    Extract zone from text
    Example:
        us-central1-a -> us-central1-a
        us-central1-b -> us-central1-b
        us-central2-b -> us-central2-b
        us-east1-d -> us-east1-d
        us-east5-b -> us-east5-b
        asia-northeast1-b -> asia-northeast1-b
    """
    # we need to look into data.json, to find the zone of the tpu
    data = data_io.read_data()
    for zone, tpu_list in data['all_tpus'].items():
        if tpu_name in tpu_list:
            return zone
    return None

def _get_job_type_zone(old_tpu):
    target_type = _extract_tpu_type(old_tpu)
    target_zone = _extract_zone(old_tpu)

    return target_type, target_zone

def _parse_idle_tpus_from_tou(stdout):
    idle_tpus = []
    clean_stdout = _strip_ansi(stdout)
    for line in clean_stdout.splitlines():
        match = re.search(r"\[IDLE\]\s+([^\s]+)\s+\(([^)]+)\)", line)
        if not match:
            continue
        tpu_name, zone = match.group(1), match.group(2)
        idle_tpus.append((tpu_name.strip(), zone.strip()))
    return idle_tpus

def _pick_idle_tpu(idle_tpus, target_type, target_zone):
    for tpu_name, zone in idle_tpus:
        tpu_type = _extract_tpu_type(tpu_name)
        # check if this tpu is in the data.json
        data = data_io.read_data()
        if tpu_name in data['all_tpus'][zone]:
            continue
        if tpu_type == target_type and zone == target_zone:
            return tpu_name
    return None

def _pick_new_alias(target_type, zone):
    available_aliases = avilable_aliases(target_type, zone)
    # do tou
    tou_result = subprocess.run(
        _tou,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if tou_result.returncode != 0:
        return None
    # for each alias, find its tpu full name, and look whether it is in output of tou
    clean_stdout = _strip_ansi(tou_result.stdout)
    for a in available_aliases:
        data = data_io.read_data()
        full_name = data['tpu_aliases'][a]
        if full_name not in clean_stdout:
            return a
    return None

def reapply_worker(ka, result_queue):
    sys.stdout = open(os.devnull, 'w')
    try:
        result = operate.apply_and_set_env(ka, preemptible=True, delete=True)
        if result == 'success':
            print(f"{GOOD} reapply_worker: Reapply TPU {ka} done")
            add_MONITOR_log(f"{GOOD} reapply_worker: Reapply TPU {ka} done")
        else:
            raise Exception(f"Reapply TPU {ka} failed, please contact the admin, result: {result}")
        result_queue.put(result)
    except Exception as e:
        print(f"{FAIL} reapply_worker: Failed to reapply TPU {ka}: {e}")
        add_MONITOR_log(f"{FAIL} reapply_worker: Failed to reapply TPU {ka}: {e}")
        result_queue.put(e)

def restart_worker(ka, result_queue):
    sys.stdout = open(os.devnull, 'w')
    try:
        print(f"{INFO} restart_worker: Restarting TPU {ka}...")
        result = operate.restart(ka)
        if result == 'success':
            print(f"{GOOD} restart_worker: Restart TPU {ka} done")
            add_MONITOR_log(f"{GOOD} restart_worker: Restart TPU {ka} done")
        else:
            raise Exception(f"Restart TPU {ka} failed, please contact the admin")
        result_queue.put(result)
    except Exception as e:
        print(f"{FAIL} restart_worker: Failed to restart TPU {ka}: {e}")
        add_MONITOR_log(f"{FAIL} restart_worker: Failed to restart TPU {ka}: {e}")
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

def restart_rerun(job, timeout=900):
    ka = job["tpu"]
    print(f"{INFO} restart_rerun: Restarting TPU {ka}...")
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=restart_worker, args=(ka, result_queue))
    running_processes.append(process)
    process.start()
    process.join(timeout)
    if process.is_alive():
        print(f"Restart TPU {ka} timeout, killing the process")
        process.terminate()
        process.join()
        running_processes.remove(process)
        print(f"{WARNING} restart_rerun: Restart TPU {ka} failed, process killed")
    else:
        if not result_queue.empty():
            result = result_queue.get()
            if isinstance(result, Exception):
                print(f"{FAIL} restart_rerun: Restart TPU {ka} failed: {result}")
                add_MONITOR_log(f"{FAIL} restart_rerun: Restart TPU {ka} failed: {result}")
            else:
                print(f"{GOOD} Restart TPU {ka} success: {result}, start rerun job")
                jobs.resume_rerun_job(job, load_ckpt=False)
        else:
            print(f"{FAIL} restart_rerun: Restart TPU {ka} failed, no result returned")
            add_MONITOR_log(f"{FAIL} restart_rerun: Restart TPU {ka} failed, no result returned")


def reapply_resume(job, timeout=900):
    ka = job["tpu"]
    add_MONITOR_log(f"{INFO} reapply_resume: Reapply TPU {ka}...")
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
    error_jobs = {'preempted': [], 'deleted': []}
    data = data_io.read_data()
    sqa = read_sqa()
    check_result = subprocess.run('python /home/jzc/zhichengjiang/working/xibo_tpu_manager/tpu.py check sqa', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    add_MONITOR_log(f"{INFO} 我来看看job活着没")
    for job in data["users"]['sqa']["job_data"]:
        if job['windows_id'] in sqa['running'] or job['windows_id'] in sqa['finished']: continue # have tried this before
        if job['status'] in ['finished', 'rerunned', 'resumed', 'killed'] or not job['monitor']:
            continue
        if f'Window {job["windows_id"]}' not in check_result.stdout:
            add_MONITOR_log(f"{INFO} job {job['windows_id']}, 跳过这个窗口")
            continue
        if job['status'] != 'error':
            continue
        error_type = check_job_status(job)
        if error_type in error_jobs:
            error_jobs[error_type].append(job)

    if len(error_jobs['deleted']) != 0:
        error_windows_list = [(job['user'], job['windows_id']) for job in error_jobs['deleted']]
        add_MONITOR_log(f"{INFO} 我找到了 {len(error_jobs['deleted'])} 个被删掉的卡, 窗口列表是: {error_windows_list}")
    
    # if len(error_jobs['preempted']) != 0:
    #     error_windows_list = [(job['user'], job['windows_id']) for job in error_jobs['preempted']]
    #     print(f"{INFO} mainloop: Found {len(error_jobs['preempted'])} preempted jobs, windows list: {error_windows_list}")
    #     add_MONITOR_log(f"{INFO} mainloop: Found {len(error_jobs['preempted'])} preempted jobs, windows list: {error_windows_list}")
    
    # if len(error_jobs['grpc']) != 0:
    #     error_windows_list = [(job['user'], job['windows_id']) for job in error_jobs['grpc']]
    #     print(f"{INFO} mainloop: Found {len(error_jobs['grpc'])} grpc jobs, windows list: {error_windows_list}")
    #     add_MONITOR_log(
    #         f"{INFO} mainloop: Found {len(error_jobs['grpc'])} grpc jobs, windows list: {error_windows_list}"
    #     )
    
    all_good = all(len(error_jobs[error_type]) == 0 for error_type in error_jobs)

    if all_good:
        add_MONITOR_log(f"{INFO} 好像都没问题，睡大觉")
        
    if not all_good:
        for job in error_jobs["deleted"]:
            
            _window = job['windows_id']
            _old_tpu = job['tpu']
            write_sqa(_window)
            try:
                add_MONITOR_log(f'我在试着 resume window {_window}. 这卡没了\n')
                target_type, target_zone = _get_job_type_zone(_old_tpu)
                add_MONITOR_log(f'这个老登的卡型号是 {target_type}, 所在区域是 {target_zone}\n')
                if not target_type or not target_zone:
                    add_MONITOR_log(f'{WARNING} 我无法确定这个老登的卡型号和所在区域, 跳过这个老登\n')
                    remove_sqa(_window)
                    continue

                tou_result = subprocess.run(
                    _tou,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                if tou_result.returncode != 0:
                    add_MONITOR_log(f'{FAIL} _tou failed, skip this job\n')
                    continue

                idle_tpus = _parse_idle_tpus_from_tou(tou_result.stdout)
                new_tpu_name = _pick_idle_tpu(idle_tpus, target_type, target_zone)
                if not new_tpu_name:
                    add_MONITOR_log(f'{WARNING} 我找不到可用的 IDLE 卡, 型号是 {target_type}, 所在区域是 {target_zone}\n')
                    remove_sqa(_window)
                    continue

                new_alias = _pick_new_alias(target_type, target_zone)
                if not new_alias:
                    add_MONITOR_log(f'{WARNING} 我找不到可用的 alias, 型号是 {target_type}, 跳过这个老登\n')
                    remove_sqa(_window)
                    continue
                
                add_MONITOR_log(f'我找到了可用的 alias: {new_alias}, 和卡: {new_tpu_name}\n')
                fmd_cmd = f'{_tpu} fmd {new_tpu_name} {new_alias}'
                add_MONITOR_log(f'运行放他妈的命令: {fmd_cmd}\n')
                fmd_result = subprocess.run(
                    fmd_cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                _append_resume_file_log(_window, "ftmd", fmd_cmd, fmd_result)
                if fmd_result.returncode != 0:
                    add_MONITOR_log(f'{FAIL} fmd failed, skip resume for window {_window}\n')
                    remove_sqa(_window)
                    continue
                add_MONITOR_log(f'放完了，哈哈')

                resume_cmd = f'{_tpu} resume sqa window={_window} tpu={new_tpu_name}'
                add_MONITOR_log(f'运行 resume 命令: {resume_cmd}\n')
                resume_result = subprocess.run(
                    resume_cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                _append_resume_file_log(_window, "resume", resume_cmd, resume_result)
                if resume_result.returncode != 0:
                    add_MONITOR_log(f'{FAIL} resume failed, skip finish for window {_window}\n')
                    remove_sqa(_window)
                    continue
                add_MONITOR_log(f'resume 上了，siuuuuuuuuuuuuuuu')
                finish_sqa(_window)
            except Exception as e:
                add_MONITOR_log(f"{FAIL} 我失败了: {e}")
                # remove job from sqa.json
                remove_sqa(_window)
            subprocess.run('sleep 2', shell=True)

            # user = job["user"]
            # data = data_io.read_and_lock_data()
            # try:
            #     for jb in data["users"][user]["job_data"]:
            #         if jb["windows_id"] == job["windows_id"]:
            #             jb["status"] = 'error'
            #             jb['error'] = error_type
            #     data_io.write_and_unlock_data(data)
            # except:
            #     print(f"{FAIL} mainloop: Failed to update job {job['windows_id']} for user {user}")
            #     add_MONITOR_log(f"{FAIL} mainloop: Failed to update job {job['windows_id']} for user {user}")
            #     data_io.release_lock_data()

    # if not all_good:
    #     for error_type in error_jobs:
    #         for job in error_jobs[error_type]:
    #             rule = job["rules"][error_type]
    #             try:
    #                 if rule == 'pass':      continue
    #                 elif rule == 'reapply': reapply_resume(job, timeout=1800)
    #                 elif rule == 'resume':  kill_resume(job)
    #                 elif rule == 'rerun':   kill_rerun(job)
    #                 elif rule == 'restart': restart_rerun(job)
    #             except:
    #                 print(f"{FAIL} mainloop: Failed to handle job {job['windows_id']} for user {user}, (error type {error_type}, rule {rule})")
    #                 add_MONITOR_log(f"{FAIL} mainloop: Failed to handle job {job['windows_id']} for user {user}, (error type {error_type}, rule {rule})")
    

if __name__ == "__main__":
    num_loops = 0
    last_test_time = time.time()
    last_clean_time = time.time()
    add_MONITOR_log(f"{GOOD} \n\n\n哈哈，我上线了。我是何凯明的狗")

    if data_io.check_code_lock():
        print(f"{FAIL} Code is locked for developing, please unlock it first.")
        sys.exit(1)
    try:
        while True:
            # data = data_io.read_data()
            # checking_freq, test_freq, clean_freq = data["MONITOR_config"]["checking_freq"], data["MONITOR_config"]["test_freq"], data["MONITOR_config"]["clean_freq"]

            num_loops += 1
            last_time = time.time()
            mainloop()
            time_used = time.time() - last_time # in seconds

            add_MONITOR_log(f"{INFO} 我看完了。现在是第 {num_loops} 轮，用时 {time_used:.2f} 秒。现在的时间是 {convert_utcstr_to_edtstr(get_abs_time_str())}")
            subprocess.run('sleep 900', shell=True)

            # while time.time() - last_time < checking_freq:
            #     data = data_io.read_data()
            #     time.sleep(10)
            #     if data['ack_MONITOR']:
            #         print(f"{INFO} Acknowledged by user, start checking...")
            #         data = data_io.read_and_lock_data()
            #         data['ack_MONITOR'] = False
            #         data_io.write_and_unlock_data(data)
            #         break
            
            # if num_loops > 100:
            #     print(f"{GOOD} successfully run {num_loops} loops, exiting...")
            #     add_MONITOR_log(f"{GOOD} successfully run {num_loops} loops, exiting...")
            #     sys.exit(0)
                
    except KeyboardInterrupt:
        print("KeyboardInterrupt, exiting...")
        # kill all the processes
        for process in running_processes:
            process.terminate()
            process.join()
        print("All processes killed")
        add_MONITOR_log(f"{FAIL} KeyboardInterrupt, all processes killed")
        sys.exit(1)