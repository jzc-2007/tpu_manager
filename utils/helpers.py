import os, datetime
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data.json")
LOCK_PATH = os.path.join(BASE_DIR, "lock.json")
SECRET_PATH = os.path.join(BASE_DIR, "secret.json")

RED, GREEN, YELLOW, PURPLE, NC = "\033[1;31m", "\033[1;32m", "\033[1;33m", "\033[1;34m", "\033[0m"
GOOD, INFO, WARNING, FAIL = f"{GREEN}[GOOD]{NC}", f"{PURPLE}[INFO]{NC}", f"{YELLOW}[WARNING]{NC}", f"{RED}[FAIL]{NC}"
LOG = f"{PURPLE}[LOG]{NC}"

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
            elif isinstance(value, str):
                if info[key] != value:
                    match = False
                    break
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
        print(f"{GREEN}Free TPUs{NC}")
        for tpu in free_tpus:
            print(tpu_information[tpu]['alias'], end='; ')
        print(f"\n{YELLOW}Reserved TPUs{NC}")
        for tpu in reserved_tpus:
            print(f"{tpu_information[tpu]['alias']}({tpu_information[tpu]['user']})", end='; ')
        print(f"\n{RED}Running TPUs{NC}")
        for tpu in running_tpus:
            print(f"{tpu_information[tpu]['alias']}({tpu_information[tpu]['user']})", end='; ')
        print()
    elif style == 'category_note':
        free_tpus = [tpu for tpu, info in tpu_information.items() if info['running_status'] == 'free']
        reserved_tpus = [tpu for tpu, info in tpu_information.items() if info['running_status'] == 'reserved']
        running_tpus = [tpu for tpu, info in tpu_information.items() if info['running_status'] == 'running']
        print(f"{GREEN}Free TPUs{NC}")
        for tpu in free_tpus:
            info = tpu_information[tpu]
            print(info['alias'], end='; ')
        print(f"\n{YELLOW}Reserved TPUs{NC}")
        for tpu in reserved_tpus:
            info = tpu_information[tpu]
            print(f"{info['alias']} ({info['user']}: {info['user_note']})")
        print(f"{RED}Running TPUs{NC}")
        for tpu in running_tpus:
            info = tpu_information[tpu]
            print(f"{info['alias']} ({info['user']}: {info['user_note']})")
    else:
        raise ValueError(f"Style {style} not recognized")
