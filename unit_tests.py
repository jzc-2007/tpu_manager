from utils import operate, logger, users, jobs
from utils.data_io import read_and_lock_data, write_and_unlock_data, release_lock_data, read_data
from utils import directories as dirs

RED="\033[1;31m"
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
PURPLE="\033[1;34m"
NC="\033[0m"

def test_get_zone_pre():
    try:
        assert operate.get_zone_pre("kmh-tpuvm-v2-32-preemptible-2") == ("europe-west4-a", True), f"T1, Expected ('europe-west4-a', True), got {operate.get_zone_pre('kmh-tpuvm-v2-32-preemptible-2')}"
        assert operate.get_zone_pre("kmh-tpuvm-v2-32-1") == ("us-central1-a", False), f"T2, Expected ('us-central1-a', False), got {operate.get_zone_pre('kmh-tpuvm-v2-32-1')}"
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

if __name__ == "__main__":
    all_tests = [
        test_get_zone_pre,
        test_no_same_window,
        test_deadlock,
    ]
    for test in all_tests:
        test()
