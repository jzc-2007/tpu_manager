import os

# The monitor keeps local queue/state, but the authoritative TPU job database
# still lives in the external xibo checkout.
BASE_DIR = "/kmh-nfs-ssd-us-mount/code/zhichengjiang/working/xibo_tpu_manager"
DATA_PATH = os.path.join(BASE_DIR, "data.json")
LOCK_PATH = os.path.join(BASE_DIR, "lock.json")

RED = "\033[1;31m"
GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
PURPLE = "\033[1;34m"
NC = "\033[0m"

GOOD = f"{GREEN}[GOOD]{NC}"
INFO = f"{PURPLE}[INFO]{NC}"
WARNING = f"{YELLOW}[WARNING]{NC}"
FAIL = f"{RED}[FAIL]{NC}"
GAOCHAO = f"{GREEN}[高潮]{NC}"
MADE = f"{RED}[妈的]{NC}"
LOG = f"{PURPLE}[LOG]{NC}"

RESUME_QUEUE_ALLOWED_REGIONS = {"us-central1", "us-east5"}
RESUME_QUEUE_ALLOWED_ZONES = {"asia-northeast1-b"}
RESUME_QUEUE_ALLOWED_ZONE_LABEL = "us-central1 / us-east5 / asia-northeast1-b"


def zone_to_region(zone):
    if not zone:
        return None
    parts = str(zone).split("-")
    if len(parts) >= 3:
        return "-".join(parts[:-1])
    return str(zone)


def is_resume_queue_allowed_zone(zone):
    if not zone:
        return False
    zone = str(zone)
    return zone in RESUME_QUEUE_ALLOWED_ZONES or zone_to_region(zone) in RESUME_QUEUE_ALLOWED_REGIONS
