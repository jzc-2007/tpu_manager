from .data_io import read_and_lock_data, write_and_unlock_data, release_lock_data, read_data, write_data
from .helpers import *
import json

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