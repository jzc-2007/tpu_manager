import os, datetime
from .data_io import read_data, write_and_unlock_data
from .constants import *

def get_zone_pre(tpu):
    """
    Get the zone of the TPU, and check if it is preemptible.
    If the input is alias, it will be replaced with the real TPU name.
    Return zone, pre, tpu_full_name
    """
    data = read_data()
    tpu_aliases = data['tpu_aliases']
    all_tpus = []
    for z, tpu_list in data['all_tpus'].items():
        all_tpus.extend(tpu_list)
    if tpu in tpu_aliases:
        tpu = tpu_aliases[tpu]
    if tpu not in all_tpus:
        print(f"{FAIL} get_zone_pre: TPU {tpu} not found")
        return None, None, None
    
    zone = None
    for z, tpu_list in data['all_tpus'].items():
        if tpu in tpu_list:
            zone = z
            break
    if zone is None:
        print(f"{FAIL} get_zone_pre: TPU {tpu} not found in any zone")
        return None, None, None
    return zone, tpu in data['pre_info']['preemptible'], tpu

def get_zone_pre_spot(tpu):
    """
    Get the zone of the TPU, and check if it is preemptible/spot.
    If the input is alias, it will be replaced with the real TPU name.
    Return zone, pre, spot, tpu_full_name
    """
    data = read_data()
    tpu_aliases = data['tpu_aliases']
    all_tpus = []
    for z, tpu_list in data['all_tpus'].items():
        all_tpus.extend(tpu_list)
    if tpu in tpu_aliases: tpu = tpu_aliases[tpu]
    if tpu not in all_tpus:
        print(f"{FAIL} get_zone_pre: TPU {tpu} not found")
        return None, None, None, None
    zone = None
    for z, tpu_list in data['all_tpus'].items():
        if tpu in tpu_list:
            zone = z
            break
    if zone is None:
        print(f"{FAIL} get_zone_pre: TPU {tpu} not found in any zone")
        return None, None, None
    return zone, tpu in data['pre_info']['preemptible'], tpu in data['pre_info']['spot'], tpu


def get_abs_time_str():
    """
    Get the current time in UTC in the format YYYY-MM-DD HH:MM:SS.
    """
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_chn_time_str():
    return convert_utcstr_to_chnstr(get_abs_time_str())

def get_edt_time_str():
    return convert_utcstr_to_edtstr(get_abs_time_str())

def convert_utcstr_to_edtstr(utc_str):
    """
    Convert UTC string to EDT string.
    """
    utc_time = datetime.datetime.strptime(utc_str, "%Y-%m-%d %H:%M:%S")
    edt_time = utc_time - datetime.timedelta(hours=4)
    return edt_time.strftime("%Y-%m-%d %H:%M:%S")

def convert_utcstr_to_chnstr(utc_str):
    """
    Convert UTC string to China string.
    """
    utc_time = datetime.datetime.strptime(utc_str, "%Y-%m-%d %H:%M:%S")
    chn_time = utc_time + datetime.timedelta(hours=8)
    return chn_time.strftime("%Y-%m-%d %H:%M:%S")

def is_integer(string):
    try:
        int(string)
        return True
    except ValueError:
        return False

def is_boolean(string):
    if string.lower() in ['true', 'false']:
        return True
    return False

def to_boolean(string):
    s = string.strip().lower()
    if s in ['true', 'yes', '1', 'y']:
        return True
    elif s in ['false', 'no', '0', 'n']:
        return False
    else:
        raise ValueError(f"Cannot convert {string} to boolean")
    
def recursive_update(dict1, dict2):
    """
    Recursively update dict1 with dict2.
    """
    for key, value in dict2.items():
        if isinstance(value, dict):
            dict1[key] = recursive_update(dict1.get(key, {}), value)
        else:
            dict1[key] = value
    return dict1

def filter_tpu_information(tpu_information, **kwargs):
    """
    Filter the TPU information based on the specified criteria.
    Keys: ['zone', 'pre', 'belong', 'running_status', 'user', 'version', 'type']
    Values: list of strings, boolean, integer, or just a string
    """
    filtered_tpu_information = {}

    for key, value in kwargs.items():
        assert key in ['zone', 'pre', 'belong', 'running_status', 'user', 'version', 'type'], f"Key {key} not recognized"
        
    for tpu, info in tpu_information.items():
        match = True
        for key, value in kwargs.items():
            if isinstance(value, list):
                if info[key] not in value:
                    match = False
                    break
            elif isinstance(value, str) or isinstance(value, bool) or isinstance(value, int):
                if str(info[key]).lower() != str(value).lower():
                    match = False
                    break
            else:
                raise ValueError(f"Value {value} for key {key} not recognized")
        if match:
            filtered_tpu_information[tpu] = info

    return filtered_tpu_information

def display_tpu_information(tpu_information, style = None, **kwargs):
    """
    Display the TPU information in a formatted way.
    styles:
    - full: full information dict
    - category: ['free', 'reserved', 'running']
    - category_note (default): ['free', 'reserved', 'running'] with user note
    """
    if style is None:
        style = 'category_note'
    if style == 'full':
        for tpu, info in tpu_information.items():
            print(f"{tpu}: {info}")
    elif style == 'category':
        free_tpus = [tpu for tpu, info in tpu_information.items() if info['running_status'] == 'free']
        reserved_tpus = [tpu for tpu, info in tpu_information.items() if info['running_status'] == 'reserved']
        running_tpus = [tpu for tpu, info in tpu_information.items() if info['running_status'] == 'running']
        free_tpus = sorted(free_tpus, key=lambda x: tpu_information[x]['alias'])
        reserved_tpus = sorted(reserved_tpus, key=lambda x: tpu_information[x]['alias'])
        running_tpus = sorted(running_tpus, key=lambda x: tpu_information[x]['alias'])
        print(f"{GREEN}Free TPUs{NC} (Total: {len(free_tpus)})")
        for tpu in free_tpus:
            print(tpu_information[tpu]['alias'], end='; ')
        print(f"\n\n{YELLOW}Reserved TPUs{NC} (Total: {len(reserved_tpus)})")
        for tpu in reserved_tpus:
            print(f"{tpu_information[tpu]['alias']}({tpu_information[tpu]['user']})", end='; ')
        print(f"\n\n{PURPLE}Running TPUs{NC} (Total: {len(running_tpus)})")
        for tpu in running_tpus:
            print(f"{tpu_information[tpu]['alias']}({tpu_information[tpu]['user']})", end='; ')
        print()
    elif style == 'category_note':
        deleted_tpus = [tpu for tpu, info in tpu_information.items() if info['script_note'].lower() == 'not found' or info['script_note'].lower() == 'preempted']
        creating_tpus = [tpu for tpu, info in tpu_information.items() if info['script_note'].lower() == 'creating']
        free_tpus = [tpu for tpu, info in tpu_information.items() if info['running_status'] == 'free' and tpu not in deleted_tpus and tpu not in creating_tpus]
        reserved_tpus = [tpu for tpu, info in tpu_information.items() if info['running_status'] == 'reserved' and tpu not in deleted_tpus and tpu not in creating_tpus]
        running_tpus = [tpu for tpu, info in tpu_information.items() if info['running_status'] == 'running' and tpu not in deleted_tpus and tpu not in creating_tpus]
        free_tpus = sorted(free_tpus, key=lambda x: tpu_information[x]['alias'])
        reserved_tpus = sorted(reserved_tpus, key=lambda x: tpu_information[x]['alias'])
        running_tpus = sorted(running_tpus, key=lambda x: tpu_information[x]['alias'])
        deleted_tpus = sorted(deleted_tpus, key=lambda x: tpu_information[x]['alias'])
        creating_tpus = sorted(creating_tpus, key=lambda x: tpu_information[x]['alias'])
        if len(deleted_tpus) > 0:
            print(f"{RED}Deleted/Preempted TPUs{NC} (Total: {len(deleted_tpus)})")
            for tpu in deleted_tpus:
                info = tpu_information[tpu]
                print(f"{info['alias']} ({info['user']}: {info['user_note']})")
        if len(creating_tpus) > 0:
            print(f"{YELLOW}Creating TPUs{NC} (Total: {len(creating_tpus)})")
            for tpu in creating_tpus:
                info = tpu_information[tpu]
                print(f"{info['alias']} ({info['user']}: {info['user_note']})")
        if len(free_tpus) > 0:
            print(f"{GREEN}Free TPUs{NC} (Total: {len(free_tpus)})")
            for tpu in free_tpus:
                info = tpu_information[tpu]
                print(info['alias'], end='; ')
        if len(reserved_tpus) > 0:
            print(f"\n\n{YELLOW}Reserved TPUs{NC} (Total: {len(reserved_tpus)})")
            for tpu in reserved_tpus:
                info = tpu_information[tpu]
                print(f"{info['alias']} ({info['user']}: {info['user_note']})")
        if len(running_tpus) > 0:
            print(f"\n{PURPLE}Running TPUs{NC} (Total: {len(running_tpus)})")
            for tpu in running_tpus:
                info = tpu_information[tpu]
                print(f"{info['alias']} ({info['user']}: {info['user_note']})")
        print()
    else:
        raise ValueError(f"Style {style} not recognized")

def get_all_tpus():
    data = read_data()
    all_tpu_list_ = []
    for k, v in data['all_tpus'].items():
        all_tpu_list_.extend(v)
    all_tpu_list_ = list(set(all_tpu_list_))
    return all_tpu_list_


