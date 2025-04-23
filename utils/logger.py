from .helpers import DATA_PATH, is_integer, is_boolean, to_boolean
from .data_io import read_and_lock_data, write_and_unlock_data, release_lock_data, read_data
from .users import user_from_dict
import json
RED, GREEN, YELLOW, PURPLE, NC = "\033[1;31m", "\033[1;32m", "\033[1;33m", "\033[1;34m", "\033[0m"
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
        print(f"{RED}[ERROR]{NC} Failed to set monitor config")
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
        print(f"{RED}[ERROR]{NC} Failed to add tpu alias")
        release_lock_data()

def get_settings(user_object, args):
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
        print(f"{RED}[ERROR]{NC} Failed to set settings")
        release_lock_data()

def add_config_alias(user_object, args):
    data = read_and_lock_data()
    try:
        alias, command = args[0], args[1]
        user_object.add_config_alias(alias, command)
        data['users'][user_object.name] = user_object.to_dict()
        write_and_unlock_data(data)
    except:
        print(f"{RED}[ERROR]{NC} Failed to add config alias")
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
        print(f"{RED}[ERROR]{NC} Failed to delete config alias")
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
        print(f"{RED}[ERROR]{NC} Failed to clear user logs")
        release_lock_data()