from utils import operate, logger, users, jobs
from utils.data_io import read_and_lock_data, write_and_unlock_data, release_lock_data, read_data
from utils import directories as dirs
import os

RED="\033[1;31m"
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
PURPLE="\033[1;34m"
NC="\033[0m"

def test_get_zone_pre():
    try:
        assert operate.get_zone_pre("kmh-tpuvm-v2-32-preemptible-2") == ("europe-west4-a", True, "kmh-tpuvm-v2-32-preemptible-2"), f"T1, Expected ('europe-west4-a', True, 'kmh-tpuvm-v2-32-preemptible-2'), got {operate.get_zone_pre('kmh-tpuvm-v2-32-preemptible-2')}"
        assert operate.get_zone_pre("kmh-tpuvm-v2-32-1") == ("us-central1-a", False, "kmh-tpuvm-v2-32-1"), f"T2, Expected ('us-central1-a', False, 'kmh-tpuvm-v2-32-1'), got {operate.get_zone_pre('kmh-tpuvm-v2-32-1')}"
        assert operate.get_zone_pre("v2-32-p2") == ("europe-west4-a", True, "kmh-tpuvm-v2-32-preemptible-2"), f"T3, Expected ('europe-west4-a', True, 'kmh-tpuvm-v2-32-preemptible-2'), got {operate.get_zone_pre('kmh-tpuvm-v2-32-preemptible-2')}"
        assert operate.get_zone_pre("v2-32-1") == ("us-central1-a", False, "kmh-tpuvm-v2-32-1"), f"T4, Expected ('us-central1-a', False, 'kmh-tpuvm-v2-32-1'), got {operate.get_zone_pre('kmh-tpuvm-v2-32-1')}"
        print(f"{GREEN}[PASSED]{NC} test_get_zone_pre")
    except Exception as e:
        print(f"{RED}[FAILED]{NC} test_get_zone_pre")
        print(e)

def test_no_same_window():
    try:
        data = read_data()
        for user, user_data in data['users'].items():
            all_windows = set()
            for job in user_data['job_data']:
                if 'windows_id' in job:
                    if job['windows_id'] in all_windows:
                        raise Exception(f"Duplicate windows_id {job['windows_id']} found for user {user}")
                    all_windows.add(job['windows_id'])
        print(f"{GREEN}[PASSED]{NC} test_no_same_window")
    except Exception as e:
        print(f"{RED}[FAILED]{NC} test_no_same_window")
        print(e)

def test_deadlock():
    try:
        data = read_and_lock_data()
        write_and_unlock_data(data)
        print(f"{GREEN}[PASSED]{NC} test_deadlock")
    except Exception as e:
        print(f"{RED}[FAILED]{NC} test_deadlock")
        print(e)

def test_zombie_windows():
    try:
        data = read_data()
        zombie = False
        zombies = []
        for user, user_data in data['users'].items():
            windows = os.popen(f'tmux list-windows -t {user_data["tmux_name"]}').read().splitlines()
                # get all the windows in the job data
            all_windows = []
            for job in user_data["job_data"]:
                all_windows.append(job['windows_id'])
            # remove the windows that not in the job data
            for window in windows:
                window_num = int(window.split(':')[0])
                if window_num == 0:
                    continue
                if window_num not in all_windows:
                    zombies.append(f'user {user}, window {window_num}')
                    zombie = True
        if zombie:
            raise Exception(f"Zombie windows found:\n{'\n'.join(zombies)}\nUse \"tpu -czw <user>\" to clear them")
        print(f"{GREEN}[PASSED]{NC} test_zombie_windows")
    except Exception as e:
        print(f"{RED}[FAILED]{NC} test_zombie_windows")
        print(e)

def test_zombie_jobs():
    try:
        data = read_data()
        zombie = False
        zombies = []
        for user, user_data in data['users'].items():
            windows = os.popen(f'tmux list-windows -t {user_data["tmux_name"]}').read().splitlines()
            all_windows = []
            for window in windows:
                window_num = int(window.split(':')[0])
                all_windows.append(window_num)
            for job in user_data["job_data"]:
                if job['windows_id'] not in all_windows:
                    zombies.append(f'user {user}, job {job["windows_id"]}')
                    zombie = True
        if zombie:
            raise Exception(f"Zombie jobs found: {', '.join(zombies)}, use \"tpu -czj <user>\" to clear them")
        print(f"{GREEN}[PASSED]{NC} test_zombie_jobs")
    except Exception as e:
        print(f"{RED}[FAILED]{NC} test_zombie_jobs")
        print(e)

def test_has_child():
    """
    Test if all the jobs with status 'rerunned' have a child job.
    """
    try:
        data = read_data()
        for user, user_data in data['users'].items():
            for job in user_data['job_data']:
                if job['status'] == 'rerunned':
                    if 'child' not in job['extra_msgs']:
                        raise Exception(f"Job {job['windows_id']} has status 'rerunned' but no child job")
        print(f"{GREEN}[PASSED]{NC} test_has_child")
    except Exception as e:
        print(f"{RED}[FAILED]{NC} test_has_child")
        print(e)

if __name__ == "__main__":
    all_tests = [
        test_get_zone_pre,
        test_no_same_window,
        test_deadlock,
        test_zombie_windows,
        test_zombie_jobs,
        test_has_child,
    ]
    for test in all_tests:
        test()
