import json
import os, time
from .helpers import DATA_PATH, LOCK_FILE
from .data_io import read_and_lock_data, write_and_unlock_data, release_lock_data, read_data
RED, GREEN, YELLOW, PURPLE, NC = "\033[1;31m", "\033[1;32m", "\033[1;33m", "\033[1;34m", "\033[0m"
def clear_zombie_windows(user_obj):
    """
    Clean up all the windows that not appear in user's joblist, except the window 0.
    """
    # get all the windows in the tmux session
    windows = os.popen(f'tmux list-windows -t {user_obj.tmux_name}').read().splitlines()
    # get all the windows in the job data
    all_windows = []
    for job in user_obj.job_data:
        all_windows.append(job['windows_id'])
    # remove the windows that not in the job data
    for window in windows:
        window_num = int(window.split(':')[0])
        if window_num == 0:
            continue
        if window_num not in all_windows:
            print(f"{PURPLE}[INFO]{NC} clear_zombie_windows: killing window {user_obj.tmux_name}:{window_num}")
            os.system(f'tmux kill-window -t {user_obj.tmux_name}:{window_num}')
            time.sleep(0.5)

def initialization():
    """
    remove all the job data for all users
    """
    data = read_data()
    for user in data['users']:
        data['users'][user]['job_data'] = []
    write_and_unlock_data(data)
