import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import operate, data_io
from utils import directories as dirs
from utils.sheet import read_sheet_info, write_sheet_info
from utils.helpers import *
from .constants import *

def test_get_zone_pre(quiet = False):
    try:
        assert get_zone_pre("kmh-tpuvm-v2-32-preemptible-2") == ("europe-west4-a", True, "kmh-tpuvm-v2-32-preemptible-2"), f"T1, Expected ('europe-west4-a', True, 'kmh-tpuvm-v2-32-preemptible-2'), got {get_zone_pre('kmh-tpuvm-v2-32-preemptible-2')}"
        assert get_zone_pre("kmh-tpuvm-v2-32-1") == ("us-central1-a", False, "kmh-tpuvm-v2-32-1"), f"T2, Expected ('us-central1-a', False, 'kmh-tpuvm-v2-32-1'), got {get_zone_pre('kmh-tpuvm-v2-32-1')}"
        assert get_zone_pre("v2-32-p2") == ("europe-west4-a", True, "kmh-tpuvm-v2-32-preemptible-2"), f"T3, Expected ('europe-west4-a', True, 'kmh-tpuvm-v2-32-preemptible-2'), got {get_zone_pre('kmh-tpuvm-v2-32-preemptible-2')}"
        assert get_zone_pre("v2-32-1") == ("us-central1-a", False, "kmh-tpuvm-v2-32-1"), f"T4, Expected ('us-central1-a', False, 'kmh-tpuvm-v2-32-1'), got {get_zone_pre('kmh-tpuvm-v2-32-1')}"
        if not quiet:
            print(f"{GREEN}[PASSED]{NC} test_get_zone_pre")
        return True
    except Exception as e:
        print(f"{RED}[FAILED]{NC} test_get_zone_pre")
        print(e)
        return False

def test_no_same_window(quiet = False):
    try:
        data = data_io.read_data()
        for user, user_data in data['users'].items():
            all_windows = set()
            for job in user_data['job_data']:
                if 'windows_id' in job:
                    if job['windows_id'] in all_windows:
                        raise Exception(f"Duplicate windows_id {job['windows_id']} found for user {user}")
                    all_windows.add(job['windows_id'])
        if not quiet:
            print(f"{GREEN}[PASSED]{NC} test_no_same_window")
        return True
    except Exception as e:
        print(f"{RED}[FAILED]{NC} test_no_same_window")
        print(e)
        return False

def test_deadlock(quiet = False):
    try:
        data = data_io.read_and_lock_data()
        data_io.write_and_unlock_data(data)
        if not quiet:
            print(f"{GREEN}[PASSED]{NC} test_deadlock")
        return True
    except Exception as e:
        print(f"{RED}[FAILED]{NC} test_deadlock")
        print(e)
        return False

def test_zombie_windows(quiet = False):
    try:
        data = data_io.read_data()
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
            raise Exception(f"Zombie windows found:\n{' '.join(zombies)}\nUse \"tpu -czw <user>\" to clear them")
        if not quiet:
            print(f"{GREEN}[PASSED]{NC} test_zombie_windows")
        return True
    except Exception as e:
        print(f"{RED}[FAILED]{NC} test_zombie_windows")
        print(e)
        return False

def test_zombie_jobs(quiet = False):
    try:
        data = data_io.read_data()
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
        if not quiet:
            print(f"{GREEN}[PASSED]{NC} test_zombie_jobs")
        return True
    except Exception as e:
        print(f"{RED}[FAILED]{NC} test_zombie_jobs")
        print(e)
        return False

def test_has_child(quiet = False):
    """
    Test if all the jobs with status 'resumed' and 'rerunned' have a child job.
    """
    try:
        data = data_io.read_data()
        for user, user_data in data['users'].items():
            for job in user_data['job_data']:
                if job['status'] == 'resumed' or job['status'] == 'rerunned':
                    if 'child' not in job['extra_msgs']:
                        raise Exception(f"Job {job['windows_id']} has status {job['status']} but no child job")
        if not quiet:
            print(f"{GREEN}[PASSED]{NC} test_has_child")
        return True
    except Exception as e:
        print(f"{RED}[FAILED]{NC} test_has_child")
        print(e)
        return False

def test_code_locked(quiet = False):
    """
    Test if the code is locked for development.
    """
    if not data_io.check_code_lock():
        if not quiet:
            print(f"{GREEN}[PASSED]{NC} test_code_locked")
        return True
    else:
        print(f"{RED}[FAILED]{NC} test_code_locked")

def test_check_tpu_status():
    """
    Test if the check_tpu_status function works as expected.
    """
    # print(operate.check_tpu_status("kmh-tpuvm-v2-32-preemptible-2"))
    # print(operate.check_tpu_status("kmh-tpuvm-v2-32-1"))
    # print(operate.check_tpu_status("v2-32-p2"))
    print(operate.check_tpu_status("v4-32-py2"))


def test_write_sheet_info(tpu):
    """
    Test the write_sheet_info function. Write additional notes "test" in 'user_note' on the tpu args 
    with other information remaining the same.
    """
    tpu_information = read_sheet_info()
    _, _, full_name = get_zone_pre(tpu)
    if full_name in tpu_information:
        tpu_info = tpu_information[full_name]
        tpu_info['running_status'] = 'reserved'
        tpu_info['user'] = 'jzc'
        tpu_info['user_note'] = 'test'
        write_sheet_info(tpu_info)
    else:
        print(f"{FAIL} TPU {tpu} not found in the sheet")
        return None
    return tpu_information[full_name]

def sanity_check():
    if data_io.check_code_lock():
        print(f"{WARNING} Code is locked for developing, skipping sanity checks.")
        return
    print(f"{INFO} Running sanity checks...")
    all_tests = [
        test_get_zone_pre,
        test_no_same_window,
        test_deadlock,
        test_zombie_windows,
        test_zombie_jobs,
        test_has_child,
        test_code_locked,
        # test_check_tpu_status,
    ]
    passed, failed = 0, 0
    failed_list = []
    for test in all_tests:
        if test(quiet = True):
            passed += 1
        else:
            failed += 1
            failed_list.append(test.__name__)
    if passed == len(all_tests):
        print(f"{GREEN}[UNIT TEST]{NC} All tests {passed}/{len(all_tests)} passed")
    else:
        print(f"{RED}[UNIT TEST]{NC} {failed}/{len(all_tests)} tests failed")
        print(f"Failed tests: {', '.join(failed_list)}")
        print(f"{YELLOW}Please check the failed tests and fix them before proceeding.{NC}")
    return passed, failed

if __name__ == "__main__":
    # run the functions
    sanity_check()
