import os, datetime
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data.json")
LOCK_FILE = os.path.join(BASE_DIR, "lock.json")

RED, GREEN, YELLOW, PURPLE, NC = "\033[1;31m", "\033[1;32m", "\033[1;33m", "\033[1;34m", "\033[0m"
GOOD, INFO, WARNING, FAIL = f"{GREEN}[GOOD]{NC}", f"{PURPLE}[INFO]{NC}", f"{YELLOW}[WARNING]{NC}", f"{RED}[FAIL]{NC}"
LOG = f"{PURPLE}[LOG]{NC}"

def get_abs_time_str():
    """
    Get the current time in UTC in the format YYYY-MM-DD HH:MM:SS.
    """
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
