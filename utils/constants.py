import os, datetime, re

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data.json")
LEGACY_PATH = os.path.join(BASE_DIR, "legacy.json")
QUEUE_PATH = os.path.join(BASE_DIR, "queue.json")
LOCK_PATH = os.path.join(BASE_DIR, "lock.json")
SECRET_PATH = os.path.join(BASE_DIR, "secret.json")
APPLY_PATH = os.path.join(BASE_DIR, "apply.json")

MAX_LEGACY_LENGTH = 500
PROJECT = 'he-vision-group'

RED, GREEN, YELLOW, PURPLE, NC = "\033[1;31m", "\033[1;32m", "\033[1;33m", "\033[1;34m", "\033[0m"
GOOD, INFO, WARNING, FAIL = f"{GREEN}[GOOD]{NC}", f"{PURPLE}[INFO]{NC}", f"{YELLOW}[WARNING]{NC}", f"{RED}[FAIL]{NC}"
LOG = f"{PURPLE}[LOG]{NC}"

NAME_TO_VER = {'v2':'v2','v3':'v3','v4':'v4','v5e':'v5e','v5p':'v5p','v6':'v6e'}

TPU_NUM_LIST = [2**i for i in range(10)]
NAME_TO_TYPE = {}

for i in TPU_NUM_LIST:
    NAME_TO_TYPE['v2-' + str(i)] = 'v2-' + str(i)
    NAME_TO_TYPE['v3-' + str(i)] = 'v3-' + str(i)
    NAME_TO_TYPE['v4-' + str(i)] = 'v4-' + str(i)
    NAME_TO_TYPE['v5e-' + str(i)] = 'v5litepod-' + str(i)
    NAME_TO_TYPE['v5-' + str(i)] = 'v5p-' + str(i)
    NAME_TO_TYPE['v5p-' + str(i)] = 'v5p-' + str(i)
    NAME_TO_TYPE['v6-' + str(i)] = 'v6e-' + str(i)
    NAME_TO_TYPE['v6e-' + str(i)] = 'v6e-' + str(i)

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

TYPE_DICT = {}
for i in TPU_NUM_LIST:
    TYPE_DICT['v2-' + str(i)] = ['v2-' + str(i)]
    TYPE_DICT['v3-' + str(i)] = ['v3-' + str(i)]
    TYPE_DICT['v4-' + str(i)] = ['v4-' + str(i)]
    TYPE_DICT['v5-' + str(i)] = ['v5p-' + str(i), 'v5litepod-' + str(i)]
    TYPE_DICT['v6-' + str(i)] = ['v6e-' + str(i)]
    TYPE_DICT['v5e-' + str(i)] = ['v5litepod-' + str(i)]
    TYPE_DICT['v5p-' + str(i)] = ['v5p-' + str(i)]
    TYPE_DICT['v6e-' + str(i)] = ['v6e-' + str(i)]

TYPE_DICT['v4'] = ['v4-' + str(i) for i in TPU_NUM_LIST]
TYPE_DICT['v5'] = ['v5p-' + str(i) for i in TPU_NUM_LIST] + ['v5litepod-' + str(i) for i in TPU_NUM_LIST]
TYPE_DICT['v6'] = ['v6e-' + str(i) for i in TPU_NUM_LIST]
TYPE_DICT['all'] = TYPE_DICT['v4'] + TYPE_DICT['v5'] + TYPE_DICT['v6']

ZONE_DICT = {
    'all': ['us-central1-a', 'us-central1-b', 'us-central2-b', 'us-east1-d', 'us-east5-a', 'us-east5-b', 'asia-northeast1-b', 'europe-west4-a'],
    # big zones
    'us': ['us-central1-a', 'us-central1-b', 'us-central2-b', 'us-east1-d', 'us-east5-a', 'us-east5-b'],
    'asia': ['asia-northeast1-b'],
    'europe': ['europe-west4-a'],
    # small zones
    'us-central': ['us-central1-a', 'us-central1-b', 'us-central2-b'],
    'us-east': ['us-east1-d', 'us-east5-a', 'us-east5-b'],
    'asia-northeast': ['asia-northeast1-b'],
    'europe-west': ['europe-west4-a'],
    # specific zones
    'us-central1': ['us-central1-a', 'us-central1-b'],
    'us-central2': ['us-central2-b'],
    'europe-west4': ['europe-west4-a'],
    'us-east1': ['us-east1-d'],
    'us-east5': ['us-east5-a', 'us-east5-b'],
    'asia-northeast1': ['asia-northeast1-b'],
    'europe-west4-a': ['europe-west4-a'],
    # single zones
    'us-central1-a': ['us-central1-a'],
    'us-central1-b': ['us-central1-b'],
    'us-central2-b': ['us-central2-b'],
    'us-east1-d': ['us-east1-d'],
    'us-east5-a': ['us-east5-a'],
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

REGION_SA_MAP = {
    "us-central1": "bucket-us-central1@he-vision-group.iam.gserviceaccount.com",
    "us-central2": "bucket-us-central2@he-vision-group.iam.gserviceaccount.com",
    "us-east1": "tpu-gcs-writer@he-vision-group.iam.gserviceaccount.com", # master SA for now
    "us-east5": "bucket-us-east5@he-vision-group.iam.gserviceaccount.com",
    "asia-northeast1": "bucket-asia@he-vision-group.iam.gserviceaccount.com",
    "europe-west4": "373438850578-compute@developer.gserviceaccount.com",
}
DEFAULT_SA = "373438850578-compute@developer.gserviceaccount.com"