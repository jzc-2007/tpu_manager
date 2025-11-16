import os, datetime, re

# BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_DIR = "/kmh-nfs-ssd-us-mount/code/zhichengjiang/working/xibo_tpu_manager"
DATA_PATH = os.path.join(BASE_DIR, "data.json")
LEGACY_PATH = os.path.join(BASE_DIR, "legacy.json")
QUEUE_PATH = os.path.join(BASE_DIR, "queue.json")
LOCK_PATH = os.path.join(BASE_DIR, "lock.json")
SECRET_PATH = os.path.join(BASE_DIR, "secret.json")
APPLY_PATH = os.path.join(BASE_DIR, "apply.json")

MAX_LEGACY_LENGTH = 500

RED, GREEN, YELLOW, PURPLE, NC = "\033[1;31m", "\033[1;32m", "\033[1;33m", "\033[1;34m", "\033[0m"
GOOD, INFO, WARNING, FAIL = f"{GREEN}[GOOD]{NC}", f"{PURPLE}[INFO]{NC}", f"{YELLOW}[WARNING]{NC}", f"{RED}[FAIL]{NC}"
LOG = f"{PURPLE}[LOG]{NC}"

NAME_TO_TYPE = {'v2-8':'v2-8','v2-32':'v2-32','v2-64':'v2-64','v2-128':'v2-128','v3-8':'v3-8','v3-32':'v3-32','v3-64':'v3-64','v3-128':'v3-128','v4-8':'v4-8','v4-16':'v4-16','v4-32':'v4-32','v4-64':'v4-64','v4-128':'v4-128','v4-256':'v4-256','v4-512':'v4-512','v5e-8':'v5litepod-8','v5e-16':'v5litepod-16','v5e-32':'v5litepod-32','v5e-64':'v5litepod-64','v5p-8':'v5p-8','v5p-16':'v5p-16','v5p-32':'v5p-32','v5p-64':'v5p-64','v5p-128':'v5p-128','v5p-256':'v5p-256','v5p-512':'v5p-512','v6e-8':'v6e-8','v6e-16':'v6e-16','v6e-32':'v6e-32','v6e-64':'v6e-64', 'v6e-128': 'v6e-128', 'v6e-256': 'v6e-256', 'v6e-512': 'v6e-512'} # acc_type for apply
NAME_TO_VER = {'v2':'v2','v3':'v3','v4':'v4','v5e':'v5e','v5p':'v5p','v6e':'v6e'}

TPU_NUM_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
v2_list = ['v2-' + str(i) for i in TPU_NUM_LIST]
v3_list = ['v3-' + str(i) for i in TPU_NUM_LIST]
v4_list = ['v4-' + str(i) for i in TPU_NUM_LIST]
v5_list = ['v5-' + str(i) for i in TPU_NUM_LIST]
v6_list = ['v6-' + str(i) for i in TPU_NUM_LIST]
v5e_list = ['v5litepod-' + str(i) for i in TPU_NUM_LIST]
v5p_list = ['v5p-' + str(i) for i in TPU_NUM_LIST]
v6e_list = ['v6e-' + str(i) for i in TPU_NUM_LIST]
v5_list = v5_list + v5e_list + v5p_list
v6_list = v6_list + v6e_list
all_type_list = v2_list + v3_list + v4_list + v5_list + v6_list
ARG_TO_LIST = NAME_TO_TYPE | {'v2':v2_list, 'v3':v3_list, 'v4': v4_list, 'v5':v5_list, 'v6': v6_list, 'v5e': v5e_list, 'v5p': v5p_list, 'v6e': v6e_list, 'v2+': v2_list + v3_list + v4_list + v5_list + v6_list, 'v3+': v3_list + v4_list + v5_list + v6_list, 'v4+': v4_list + v5_list + v6_list, 'v5+': v5_list + v6_list, 'v6+': v6_list, 'v*': all_type_list, '-a': all_type_list, '--all': all_type_list}

TYPE_DICT = {
    'all': ['v4-32', 'v6e-32', 'v6e-64'],
    'v4-32': ['v4-32'],
    'v5-32': ['v5p-32', 'v5litepod-32'],
    'v5-64': ['v5p-64', 'v5litepod-64'],
    'v6-32': ['v6e-32'],
    'v6-64': ['v6e-64'],
    'v6e-32': ['v6e-32'],
    'v6e-64': ['v6e-64'],
    'v6':['v6e-32', 'v6e-64'],
    'v6e':['v6e-32', 'v6e-64'],
}

ZONE_DICT = {
    'all': ['us-central1-a', 'us-central1-b', 'us-central2-b', 'us-east1-d', 'us-east5-b', 'asia-northeast1-b'],
    # big zones
    'us': ['us-central1-a', 'us-central1-b', 'us-central2-b', 'us-east1-d', 'us-east5-b'],
    'asia': ['asia-northeast1-b'],
    # small zones
    'us-central': ['us-central1-a', 'us-central1-b', 'us-central2-b'],
    'us-east': ['us-east1-d', 'us-east5-b'],
    'asia-northeast': ['asia-northeast1-b'],
    # specific zones
    'us-central1': ['us-central1-a', 'us-central1-b'],
    'us-central2': ['us-central2-b'],
    'us-east1': ['us-east1-d'],
    'us-east5': ['us-east5-b'],
    'asia-northeast1': ['asia-northeast1-b'],
    # single zones
    'us-central1-a': ['us-central1-a'],
    'us-central1-b': ['us-central1-b'],
    'us-central2-b': ['us-central2-b'],
    'us-east1-d': ['us-east1-d'],
    'us-east5-b': ['us-east5-b'],
    'asia-northeast1-b': ['asia-northeast1-b'],
}


# ------------ monitor config ----------
COL_WIDTH = 50       # width of EACH job column
KEY_COL_MIN = 7
KEY_COL_MAX = 25
GAP = " | "             # gap between the two columns
ANSI_RE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

# ------------ rules ----------
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
