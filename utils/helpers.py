DATA_PATH = "/home/jzc/zhichengjiang/working/xibo_tpu_manager/data.json"
LOCK_FILE = "/home/jzc/zhichengjiang/working/xibo_tpu_manager/lock.json"
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
    if string.lower() == 'true':
        return True
    elif string.lower() == 'false':
        return False
    else:
        raise ValueError(f"Cannot convert {string} to boolean")