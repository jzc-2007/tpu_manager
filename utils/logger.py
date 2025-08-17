from .helpers import *
from .constants import *
from .data_io import read_and_lock_data, write_and_unlock_data, release_lock_data, read_data
from .users import user_from_dict

import os, yaml



def get_monitor_config():
    data = read_data()
    for key, value in data['monitor_config'].items():
        print(f"{key}: {value}")

def set_monitor_config(args):
    data = read_and_lock_data()
    try:
        key, value = args[0], args[1]
        if key not in data['monitor_config']:
            raise ValueError(f"Monitor config {key} not found")
        if is_integer(value):
            value = int(value)
        elif is_boolean(value):
            value = to_boolean(value)
        data['monitor_config'][key] = value
        print(f"Set {key} to {value}")
        write_and_unlock_data(data)
    except:
        print(f"{FAIL} Failed to set monitor config")
        release_lock_data()

def explain_tpu_aliases():
    data = read_data()
    for alias, name in data['tpu_aliases'].items():
        print(f"{alias}: {name}")

def add_tpu_alias(alias, name):
    data = read_and_lock_data()
    try:
        data['tpu_aliases'][alias] = name
        write_and_unlock_data(data)
    except:
        print(f"{FAIL} Failed to add tpu alias")
        release_lock_data()

def register_tpu():
    tpu_alias, spreadsheet_name, full_name, zone, pre = None, None, None, None, False
    try:
        tpu_alias = input("Enter TPU alias(e.g. v4-32-py): ")
        spreadsheet_name = input("Enter spreadsheet name(e.g. v4-32-preemptible-yiyang): ")
        full_name = input("Enter full name(e.g. kmh-tpuvm-v4-32-preemptible-yiyang): ")
        zone = input("Enter zone(e.g. us-central1-a): ")
        pre = input("Is it preemptible? (yes/no): ").strip().lower() == 'yes'
    except Exception as e:
        print(f"{FAIL} Failed to read input: {e}")
        return
    data = read_and_lock_data()
    try:
        if tpu_alias in data['tpu_aliases']:
            raise ValueError(f"TPU alias {tpu_alias} already exists")
        data['tpu_aliases'][tpu_alias] = full_name
        data['tpu_aliases'][spreadsheet_name] = full_name
        if zone not in data['all_tpus']:
            data['all_tpus'][zone] = []
            print(f"{WARNING} Zone {zone} not found, creating new zone entry")
        data['all_tpus'][zone].append(full_name)
        if pre:
            data['all_tpus']['preemptible'].append(full_name)
        write_and_unlock_data(data)
        print(f"{GOOD} Successfully registered TPU {tpu_alias} with full name {full_name}")
    except Exception as e:
        print(f"{FAIL} Failed to register TPU: {e}")
        release_lock_data()

def del_registered_tpu(alias):
    data = read_and_lock_data()
    try:
        if alias not in data['tpu_aliases']:
            raise ValueError(f"TPU alias {alias} not found")
        full_name = data['tpu_aliases'][alias]
        all_aliases = [alias for alias in list(data['tpu_aliases'].keys()) if data['tpu_aliases'][alias] == full_name]
        for alias_ in all_aliases:
            del data['tpu_aliases'][alias_]
        for zone in data['all_tpus']:
            if full_name in data['all_tpus'][zone]:
                data['all_tpus'][zone].remove(full_name)
        write_and_unlock_data(data)
        print(f"{GOOD} Successfully deleted TPU alias {alias}")
    except Exception as e:
        print(f"{FAIL} Failed to delete TPU alias: {e}")
        release_lock_data()

def get_settings(user_object):
    for key, value in user_object.settings.items():
        print(f"{key}: {value}")

def set_settings(user_object, args):
    data = read_and_lock_data()
    try:
        key, value = args[0], args[1]
        if key not in user_object.settings:
            raise ValueError(f"Setting {key} not found")
        if is_integer(value):
            value = int(value)
        elif is_boolean(value):
            value = to_boolean(value)
        else:
            raise ValueError(f"Value {value} is not a valid type")
        user_object.settings[key] = value
        data['users'][user_object.name]['settings'] = user_object.settings
        print(f"Set {key} to {value}")
        write_and_unlock_data(data)
    except:
        print(f"{FAIL} Failed to set settings")
        release_lock_data()

def add_config_alias(user_object, args):
    data = read_and_lock_data()
    try:
        alias, command = args[0], args[1]
        user_object.add_config_alias(alias, command)
        data['users'][user_object.name] = user_object.to_dict()
        write_and_unlock_data(data)
    except:
        print(f"{FAIL} Failed to add config alias")
        release_lock_data()

def show_config_alias(user_object):
    for alias, command in user_object.config_aliases.items():
        print(f"{alias}: {command}")

def del_config_alias(user_object, args):
    data = read_and_lock_data()
    user_object = user_from_dict(data['users'][user_object.name])
    try:
        alias = args[0]
        user_object.del_config_alias(alias)
        data['users'][user_object.name] = user_object.to_dict()
        write_and_unlock_data(data)
    except:
        print(f"{FAIL} Failed to delete config alias")
        release_lock_data()

def read_user_logs(user_object, args):
    for msg in user_object.logs:
        print(msg)

def clear_user_logs(user_object):
    data = read_and_lock_data()
    try:
        user_object.logs = []
        data['users'][user_object.name]['logs'] = user_object.logs
        write_and_unlock_data(data)
    except:
        print(f"{FAIL} Failed to clear user logs")
        release_lock_data()

def get_wandb_notes(dir):
    # read the config file in dir/configs/remote_run_config.yaml
    # and check whether there's a key called wandb_notes
    config_path = os.path.join(dir, 'configs', 'remote_run_config.yml')
    if not os.path.exists(config_path):
        print(f"{FAIL} Config file not found at {config_path}")
        return None
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if 'wandb_notes' in config:
        return config['wandb_notes']
    else:
        print(f"{WARNING} wandb_notes not found in config file")
        return None