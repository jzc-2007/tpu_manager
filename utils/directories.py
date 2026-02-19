import json, yaml
import os
from .data_io import read_and_lock_data, write_and_unlock_data, release_lock_data, lock_data, read_data
from .helpers import *
from .constants import *

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

def get_job_stage_dir(user_obj, job_id):
    data = read_data()
    all_job_data = data['users'][user_obj.name]['job_data']
    for job in all_job_data:
        if str(job['windows_id']) == str(job_id):
            if 'stage_dir' in job and job['stage_dir'] != '':
                return job['stage_dir']
            else:
                raise ValueError(f"Stage directory for job {job_id} not set")
    raise ValueError(f"Job {job_id} not found")

def get_job_config(user_obj, job_id, config_path = '/configs/remote_run_config.yml'):
    try:
        print(f"Getting config for job {job_id} from {config_path}")
        stage_dir = get_job_stage_dir(user_obj, job_id)
        print(f"Stage directory for job {job_id} is {stage_dir}")
        # get config from stage_dir/config_path
        config_file = os.path.join(stage_dir, config_path.lstrip('/'))
        print(f"Looking for config file at {config_file}")
        if not os.path.exists(config_file):
            # check if it is yaml instead of yml
            config_path_modified = config_path.replace('.yml', '.yaml')
            config_file_modified = os.path.join(stage_dir, config_path_modified.lstrip('/'))
            if not os.path.exists(config_file_modified):
                raise ValueError(f"Config file {config_file_modified} does not exist")
            config_file = config_file_modified
        else:
            config_file = config_file
            raise ValueError(f"Config file {config_file} does not exist")
        with open(config_file, 'r') as f:
            if config_file.endswith('.json'):
                config = json.load(f)
            elif config_file.endswith('.yaml') or config_file.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                raise ValueError(f"Config file {config_file} is not a json or yaml file")
        return config
    except Exception as e:
        raise ValueError(f"Failed to get config for job {job_id}: {e}")

def read_job_config_to_file(user_obj, job_id, config_path = '/configs/remote_run_config.yml', write_path = None):
    print(f"{INFO} read_job_config_to_file: Reading config for job {job_id} in user {user_obj.name}, path {config_path}, writing to {write_path if write_path is not None else os.getcwd()}")
    if write_path is None:
        write_path = os.getcwd()
    # put the config to write_path/config_jobid.yaml
    try:
        config = get_job_config(user_obj, job_id, config_path=config_path)
        if not os.path.exists(write_path):
            raise ValueError(f"Write path {write_path} does not exist")
        filename = f"config_{job_id}.yaml"
        write_file = os.path.join(write_path, filename)
        print(f"Writing config to {write_file}")
        os.system("rm -f " + write_file)
        os.makedirs(os.path.dirname(write_file), exist_ok=True)
        with open(write_file, 'w') as f:
            yaml.dump(config, f)
        print(f"Config for job {job_id} written to {write_file}")
    except Exception as e:
        print(f"{FAIL} read_job_config_to_file: {e}")


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
