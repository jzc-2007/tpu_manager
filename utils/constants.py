import os, datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data.json")
LEGACY_PATH = os.path.join(BASE_DIR, "legacy.json")
QUEUE_PATH = os.path.join(BASE_DIR, "queue.json")
LOCK_PATH = os.path.join(BASE_DIR, "lock.json")
SECRET_PATH = os.path.join(BASE_DIR, "secret.json")
APPLY_PATH = os.path.join(BASE_DIR, "apply.json")

RED, GREEN, YELLOW, PURPLE, NC = "\033[1;31m", "\033[1;32m", "\033[1;33m", "\033[1;34m", "\033[0m"
GOOD, INFO, WARNING, FAIL = f"{GREEN}[GOOD]{NC}", f"{PURPLE}[INFO]{NC}", f"{YELLOW}[WARNING]{NC}", f"{RED}[FAIL]{NC}"
LOG = f"{PURPLE}[LOG]{NC}"

NAME_TO_TYPE = {'v2-8':'v2-8','v2-32':'v2-32','v2-64':'v2-64','v2-128':'v2-128','v3-8':'v3-8','v3-32':'v3-32','v3-64':'v3-64','v3-128':'v3-128','v4-8':'v4-8','v4-32':'v4-32','v4-64':'v4-64','v4-128':'v4-128','v5e-8':'v5litepod-8','v5e-16':'v5litepod-16','v5e-32':'v5litepod-32','v5e-64':'v5litepod-64','v6e-8':'v6e-8','v6e-16':'v6e-16','v6e-32':'v6e-32','v6e-64':'v6e-64'}
NAME_TO_VER = {'v2':'v2','v3':'v3','v4':'v4','v5e':'v5e','v6e':'v6e'}
v2_list = ['v2-8', 'v2-16', 'v2-32', 'v2-64', 'v2-128']
v3_list = ['v3-8', 'v3-16', 'v3-32', 'v3-64', 'v3-128']
v4_list = ['v4-8', 'v4-16', 'v4-32', 'v4-64', 'v4-128']
v5_list = ['v5-8', 'v5-16', 'v5-32', 'v5-64', 'v5-128']
v6_list = ['v6-8', 'v6-16', 'v6-32', 'v6-64', 'v6-128']
v5e_list = ['v5litepod-8', 'v5litepod-16', 'v5litepod-32', 'v5litepod-64', 'v5litepod-128']
v6e_list = ['v6e-8', 'v6e-16', 'v6e-32', 'v6e-64', 'v6e-128']
v5_list = v5_list + v5e_list
v6_list = v6_list + v6e_list
all_type_list = v2_list + v3_list + v4_list + v5_list + v6_list
ARG_TO_LIST = NAME_TO_TYPE | {'v2':v2_list, 'v3':v3_list, 'v4': v4_list, 'v5':v5_list, 'v6': v6_list, 'v5e': v5e_list, 'v6e': v6e_list, 'v2+': v2_list + v3_list + v4_list + v5_list + v6_list, 'v3+': v3_list + v4_list + v5_list + v6_list, 'v4+': v4_list + v5_list + v6_list, 'v5+': v5_list + v6_list, 'v6+': v6_list, 'v*': all_type_list, '-a': all_type_list, '--all': all_type_list}

QUEUE_LIST = ['v2', 'v3', 'v4', 'v5', 'v6', 'v3-32', 'v4-32', 'v2-32', 'v4-8', 'v3-64', 'v3-128', 'v2-128', 'v4-32-p', 'v3-32-p', 'v2-32-p']

RULE_DICT ={
    'pre':{
        'preempted': 'reapply',
        'grpc': 'resume',
        'locked': 'pass',
    },
    'pass':{
        'preempted': 'pass',
        'grpc': 'pass',
        'locked': 'pass',
    },
    'reapply':{
        'preempted': 'reapply',
        'grpc': 'reapply',
        'locked': 'pass',
    },
    'rerun':{
        'preempted': 'reapply',
        'grpc': 'rerun',
        'locked': 'pass',
    },
    'resume':{
        'preempted': 'pass',
        'grpc': 'resume',
        'locked': 'pass',
    }
}
