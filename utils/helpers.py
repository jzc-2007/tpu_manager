import os, datetime
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data.json")
LOCK_PATH = os.path.join(BASE_DIR, "lock.json")
SECRET_PATH = os.path.join(BASE_DIR, "secret.json")

RED, GREEN, YELLOW, PURPLE, NC = "\033[1;31m", "\033[1;32m", "\033[1;33m", "\033[1;34m", "\033[0m"
GOOD, INFO, WARNING, FAIL = f"{GREEN}[GOOD]{NC}", f"{PURPLE}[INFO]{NC}", f"{YELLOW}[WARNING]{NC}", f"{RED}[FAIL]{NC}"
LOG = f"{PURPLE}[LOG]{NC}"

NAME_TO_TYPE = {'v2-8':'v2-8','v2-32':'v2-32','v2-64':'v2-64','v2-128':'v2-128','v3-8':'v3-8','v3-32':'v3-32','v3-64':'v3-64','v3-128':'v3-128','v4-8':'v4-8','v4-32':'v4-32','v4-64':'v4-64','v4-128':'v4-128','v5e-8':'v5litepod-8','v5e-16':'v5litepod-16','v5e-32':'v5litepod-32','v5e-64':'v5litepod-64','v6e-8':'v6litepod-8','v6e-16':'v6litepod-16','v6e-32':'v6litepod-32','v6e-64':'v6litepod-64'}
NAME_TO_VER = {'v2':'v2','v3':'v3','v4':'v4','v5e':'v5e','v6e':'v6e'}
v2_list = ['v2-8', 'v2-16', 'v2-32', 'v2-64', 'v2-128']
v3_list = ['v3-8', 'v3-16', 'v3-32', 'v3-64', 'v3-128']
v4_list = ['v4-8', 'v4-16', 'v4-32', 'v4-64', 'v4-128']
v5_list = ['v5-8', 'v5-16', 'v5-32', 'v5-64', 'v5-128']
v6_list = ['v6-8', 'v6-16', 'v6-32', 'v6-64', 'v6-128']
v5e_list = ['v5litepod-8', 'v5litepod-16', 'v5litepod-32', 'v5litepod-64', 'v5litepod-128']
v6e_list = ['v6litepod-8', 'v6litepod-16', 'v6litepod-32', 'v6litepod-64', 'v6litepod-128']
v5_list = v5_list + v5e_list
v6_list = v6_list + v6e_list
all_type_list = v2_list + v3_list + v4_list + v5_list + v6_list
ARG_TO_LIST = NAME_TO_TYPE | {'v2':v2_list, 'v3':v3_list, 'v4': v4_list, 'v5':v5_list, 'v6': v6_list, 'v5e': v5e_list, 'v6e': v6e_list, 'v2+': v2_list + v3_list + v4_list + v5_list + v6_list, 'v3+': v3_list + v4_list + v5_list + v6_list, 'v4+': v4_list + v5_list + v6_list, 'v5+': v5_list + v6_list, 'v6+': v6_list, 'v*': all_type_list, '-a': all_type_list, '--all': all_type_list}

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
        print(f"\n\n{RED}Running TPUs{NC} (Total: {len(running_tpus)})")
        for tpu in running_tpus:
            print(f"{tpu_information[tpu]['alias']}({tpu_information[tpu]['user']})", end='; ')
        print()
    elif style == 'category_note':
        free_tpus = [tpu for tpu, info in tpu_information.items() if info['running_status'] == 'free']
        reserved_tpus = [tpu for tpu, info in tpu_information.items() if info['running_status'] == 'reserved']
        running_tpus = [tpu for tpu, info in tpu_information.items() if info['running_status'] == 'running']
        free_tpus = sorted(free_tpus, key=lambda x: tpu_information[x]['alias'])
        reserved_tpus = sorted(reserved_tpus, key=lambda x: tpu_information[x]['alias'])
        running_tpus = sorted(running_tpus, key=lambda x: tpu_information[x]['alias'])
        print(f"{GREEN}Free TPUs{NC} (Total: {len(free_tpus)})")
        for tpu in free_tpus:
            info = tpu_information[tpu]
            print(info['alias'], end='; ')
        print(f"\n\n{YELLOW}Reserved TPUs{NC} (Total: {len(reserved_tpus)})")
        for tpu in reserved_tpus:
            info = tpu_information[tpu]
            print(f"{info['alias']} ({info['user']}: {info['user_note']})")
        print(f"\n{RED}Running TPUs{NC} (Total: {len(running_tpus)})")
        for tpu in running_tpus:
            info = tpu_information[tpu]
            print(f"{info['alias']} ({info['user']}: {info['user_note']})")
        print()
    else:
        raise ValueError(f"Style {style} not recognized")

def safe_input(prompt):
    """
    Safely get input from the user, handling EOFError and KeyboardInterrupt.
    """
    try:
        return input(prompt)
    except EOFError:
        print("\nYou pressed Ctrl+D. Bye...")
        exit(0)
    except KeyboardInterrupt:
        print("\nYou pressed Ctrl+C. Bye...")
        exit(0)