import json
import os
from .data_io import read_and_lock_data, write_and_unlock_data, release_lock_data
from .helpers import is_integer, DATA_PATH

def get_dir(user_obj, num):
    if not is_integer(num):
        raise ValueError(f"Directory number {num} is not an integer")
    if num not in user_obj.working_dir:
        raise ValueError(f"Directory number {num} not found")
    dir = user_obj.working_dir[num]
    if not os.path.exists(dir):
        raise ValueError(f"Directory {dir} does not exist")
    return dir


def set_cur(user_obj, args):
    data = read_and_lock_data()
    try:
        current_dir = os.getcwd()
        if len(args) == 0:
            print("Please specify a number")
            release_lock_data()
            return
        if not is_integer(args[0]):
            print("Please specify a number")
            release_lock_data()
            return
        dir_num = int(args[0])
        assert dir_num >= 1 and dir_num <= 100, "Directory number must be between 1 and 100"
        data['users'][user_obj.name]['working_dir'][str(dir_num)] = current_dir

        print(f"Set directory {dir_num} to {current_dir}")
        print("Current directories:")
        for i, dir in data['users'][user_obj.name]['working_dir'].items():
            print(f"{i}: {dir}", end='')
            if i == "1":
                print("(default)")
            else:
                print()

        write_and_unlock_data(data)
    except:
        print("Error setting directory")
        release_lock_data()

def set_dir(user_obj, args):
    data = read_and_lock_data()
    try:
        assert len(args) >= 2, "Please specify a directory and a number"
        dir, num = args[0], args[1]
        if not os.path.exists(dir):
            raise ValueError(f"Directory {dir} does not exist")
        
        data['users'][user_obj.name]['working_dir'][num] = dir
        print(f"Set directory {num} to {dir}")
        print("Current directories:")
        
        for i, dir in data['users'][user_obj.name]['working_dir'].items():
            print(f"{i}: {dir}", end='')
            if i == "1":
                print("(default)")
            else:
                print()

        write_and_unlock_data(data)
        
    except:
        release_lock_data()

def list_dir(user_obj, args):
    all_dirs = user_obj.working_dir
    if len(all_dirs) == 0:
        print("No directories found")
        return
    print("Directories:")
    for i, dir in all_dirs.items():
        print(f"{i}: {dir}", end='')
        if i == "1":
            print("(default)")
        else:
            print()
