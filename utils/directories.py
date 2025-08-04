import json
import os
from .data_io import read_and_lock_data, write_and_unlock_data, release_lock_data
from .helpers import *

def get_dir(user_obj, num):
    if not is_integer(num):
        raise ValueError(f"Directory number {num} is not an integer")
    if num == '0':
        # return the father dir of the default directory
        default_dir = user_obj.working_dir['1']
        if not os.path.exists(default_dir):
            raise ValueError(f"Default directory {default_dir} does not exist")
        return os.path.dirname(default_dir)
    if num not in user_obj.working_dir:
        raise ValueError(f"Directory number {num} not found")
    dir = user_obj.working_dir[num]
    if not os.path.exists(dir):
        raise ValueError(f"Directory {dir} does not exist")
    return dir


def set_cur(user_obj, args):
    data = None
    try:
        data = read_and_lock_data()
        current_dir = os.getcwd()
        if len(args) == 0 or not is_integer(args[0]):
            print("Please specify a valid directory number")
            return

        dir_num = int(args[0])
        assert 1 <= dir_num <= 100, "Directory number must be between 1 and 100"
        data['users'][user_obj.name]['working_dir'][str(dir_num)] = current_dir

        print(f"Set directory {dir_num} to {current_dir}")
        print("Current directories:")
        for i, dir in data['users'][user_obj.name]['working_dir'].items():
            suffix = " (default)" if i == "1" else ""
            print(f"{i}: {dir}{suffix}")
    except Exception as e:
        print(f"[FAIL] set_cur: {e}")
    finally:
        if data is not None:
            write_and_unlock_data(data)
        else:
            release_lock_data()
def del_dir(user_obj, args):
    data = read_and_lock_data()
    try:
        assert len(args) >= 1, "Please specify a directory number"
        num = args[0]
        if not is_integer(num):
            print(f"{FAIL} Directory number {num} is not an integer")
            raise ValueError(f"Directory number {num} is not an integer")
        if num not in data['users'][user_obj.name]['working_dir']:
            print(f"{FAIL} Directory number {num} not found")
            raise ValueError(f"Directory number {num} not found")
        
        del data['users'][user_obj.name]['working_dir'][num]
        print(f"Deleted directory {num}")
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

def swap_dir(user_obj, args):
    data = read_and_lock_data()
    try:
        assert len(args) >= 2, "Please specify two directory numbers"
        num1, num2 = args[0], args[1]
        if not is_integer(num1) or not is_integer(num2):
            raise ValueError(f"Directory numbers {num1} and {num2} are not integers")
        if num1 not in data['users'][user_obj.name]['working_dir'] or num2 not in data['users'][user_obj.name]['working_dir']:
            raise ValueError(f"Directory numbers {num1} and {num2} not found")
        
        data['users'][user_obj.name]['working_dir'][num1], data['users'][user_obj.name]['working_dir'][num2] = \
            data['users'][user_obj.name]['working_dir'][num2], data['users'][user_obj.name]['working_dir'][num1]
        
        print(f"Swapped directories {num1} and {num2}")
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
