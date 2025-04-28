from .data_io import read_and_lock_data, write_and_unlock_data, release_lock_data, read_data, write_data
from .helpers import *
from .operate import get_zone_pre
import json, subprocess, os, time

def clear_MONITOR_log():
    data = read_and_lock_data()
    data["MONITOR_logs"] = []
    write_and_unlock_data(data)

def show_MONITOR_log(timezone = 'us'):
    data = read_data()
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
        print(f"{show_time} {msg}")

def add_global_config(key, value):
    """
    Add a global configuration key-value pair to the data.json file.
    """
    data = read_and_lock_data()
    try:
        print(f'{INFO} add_global_config: {key} = {value}')
        if key in data:
            print(f'{WARNING} add_global_config: {key} already exists, updating value, do you want to continue? (y/n)')
            answer = input().strip().lower()
            if answer != 'y' and answer != 'yes':
                print(f'{INFO} add_global_config: operation cancelled')
                release_lock_data()
                return
            
        if is_integer(value):
            print(f'{INFO} add_global_config: value is an integer, do you want to use it as a string? (y/n)')
            answer = input().strip().lower()
            if answer != 'y' and answer != 'yes':
                value = str(value)
        elif is_boolean(value):
            print(f'{INFO} add_global_config: value is a boolean, do you want to use it as a string? (y/n)')
            answer = input().strip().lower()
            if answer != 'y' and answer != 'yes':
                value = str(value)
        elif not isinstance(value, str):
            print(f'{FAIL} add_global_config: value is not a string, integer or boolean')
            release_lock_data()
            return
            
        data[key] = value
        write_and_unlock_data(data)
        print(f"Added global config: {key} = {value}")
        
    except Exception as e:
        print(f"[FAIL] add_global_config: {e}")
        release_lock_data()

def merge_global_config(dict_path):
    """
    merge a global configuration file with the data.json file.
    """
    data = read_and_lock_data()
    try:
        # if the dict path is a relative path, make it absolute
        if not os.path.isabs(dict_path):
            dict_path = os.path.join(BASE_DIR, dict_path)
        if not os.path.exists(dict_path):
            raise ValueError(f"File {dict_path} does not exist")
        # Check if the file is a valid JSON file
        if not dict_path.endswith('.json'):
            raise ValueError(f"File {dict_path} is not a JSON file")
        # Check if the file is a valid JSON dictionary
        with open(dict_path, 'r') as file:
            new_data = json.load(file)
        assert isinstance(new_data, dict), "The provided file is not a valid JSON dictionary"
        # Merge the new data into the existing data
        data = recursive_update(data, new_data)
        # Save the updated data back to the file
        write_and_unlock_data(data)
        print(f"{GOOD} Merged global config dict from {dict_path}")
    except Exception as e:
        print(f"{FAIL} merge_global_config: {e}")
        release_lock_data()


def debug_stats(tpu):
    """
    check whether some jobs is in stats 'D'
    """
    zone, pre, tpu = get_zone_pre(tpu)
    if zone is None:
        print(f"{FAIL} debug_stats: TPU {tpu} not found")
        return
    print(f"{INFO} debug_stats: Checking TPU {tpu} in zone {zone}...")
    cmd = f"gcloud compute tpus tpu-vm ssh {tpu} --zone {zone} --worker=all --command \"ps -eo pid,stat,cmd | grep 'main.py' | grep -v grep\""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        stdout, stderr = result.stdout, result.stderr
    except subprocess.CalledProcessError:
        print(f"{FAIL} debug_stats: Failed to query TPU state")
        return 'failed'
    except subprocess.TimeoutExpired:
        print(f"{FAIL} debug_stats: Timeout expired")
        return 'timeout'
    print(f"stdout: {stdout}")
    print(f"stderr: {stderr}")

def kill_jobs_tpu_new(tpu):
    zone, pre, tpu = get_zone_pre(tpu)
    if zone is None:
        print(f"{FAIL} kill_jobs_tpu: Could not determine zone.")
        return

    print(f"{INFO} kill_jobs_tpu: Killing jobs on TPU {tpu} zone {zone}...")

    try:
        data = read_data()
        for user in data["users"]:
            user_tmux_name = data["users"][user]["tmux_name"]
            for job in data["users"][user]["job_data"]:
                if job["tpu"] == tpu:
                    window = job["windows_id"]
                    if window is not None:
                        subprocess.run(f"tmux send-keys -t {user_tmux_name}:{window} C-c", shell=True, check=False)

        time.sleep(3)

        list_cmd = (
            f"gcloud compute tpus tpu-vm ssh {tpu} --zone {zone} --worker=all "
            "--command \"ps -eo pid,ppid,stat,cmd | grep 'main.py' | grep -v 'grep' || true\""
        )
        result = subprocess.run(list_cmd, shell=True, timeout=30, check=False,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        lines = result.stdout.strip().splitlines()

        if not lines:
            print(f"{INFO} No main.py processes found.")
            return 'success'

        pids = set()

        for line in lines:
            parts = line.strip().split(None, 3)
            if len(parts) >= 2:
                pid, ppid = parts[0], parts[1]
                pids.add(pid)
                pids.add(ppid)

        if not pids:
            print(f"{INFO} No processes to kill.")
            return 'success'

        pid_list = " ".join(pids)
        print(f"{INFO} Killing PIDs: {pid_list}")
        
        kill_cmd = (
            f"gcloud compute tpus tpu-vm ssh {tpu} --zone {zone} --worker=all "
            f"--command \"sudo kill -9 {pid_list} || true\""
        )
        subprocess.run(kill_cmd, shell=True, timeout=30, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        print(f"{INFO} Cleaning /dev/accel0 occupation...")
        kill_accel_cmd = (
            f"gcloud compute tpus tpu-vm ssh {tpu} --zone {zone} --worker=all "
            "--command \"pids=$(sudo lsof -w /dev/accel0 | grep 'python' | grep -v 'grep' | awk '{print $2}'); "
            "if [ ! -z \\\"$pids\\\" ]; then sudo kill -9 $pids; fi\""
        )
        subprocess.run(kill_accel_cmd, shell=True, timeout=30, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    except subprocess.TimeoutExpired:
        print(f"{FAIL} kill_jobs_tpu: Timeout.")
        return 'kill timeout'
    except Exception as e:
        print(f"{FAIL} kill_jobs_tpu: {e}")
        return 'kill error'

    print(f"{GOOD} kill_jobs_tpu: Jobs killed successfully.")
    return 'success'
