from .constants import *
import os

try:
    import gcsfs
except Exception as e:
    raise RuntimeError('please install gcsfs') from e

FS = gcsfs.GCSFileSystem()

zones_list = ['us-central1', 'us-east1', 'us-east5', 'us-central2', 'asia-northeast1-b', 'europe-west4-a', 'code/qiao/work']

def get_zone_from_workdir(workdir: str) -> str:
    matched_zones = [z for z in zones_list if z in workdir]
    if not matched_zones:
        raise ValueError(f'Cannot infer zone from workdir {workdir}. Did you change XIBO script?')
    assert len(matched_zones) == 1, f'Multiple matched zones {matched_zones} from workdir {workdir}, 快去买彩票'
    return matched_zones[0]

def check_gs_dir_exists(gs_dir):
    exit_code = os.system(f"gsutil ls {gs_dir} > /dev/null 2>&1")
    return exit_code == 0

def is_checkpoint(path: str): return True # trivial

def convert_to_gs_by_zone(path: str, zone: str):
    if zone.startswith('us-central1'):
        return path.replace('/kmh-nfs-ssd-us-mount/logs/sqa', 'gs://kmh-gcp-us-central1/qiao_zhicheng_hanhong_files')
    if zone.startswith('us-east1'):
        return path.replace('/kmh-nfs-ssd-us-mount/logs/sqa', 'gs://kmh-gcp-us-east1/qiao_zhicheng_hanhong_files')
    if zone.startswith('us-east5'):
        return path.replace('/kmh-nfs-ssd-us-mount/logs/sqa', 'gs://kmh-gcp-us-east5/qiao_zhicheng_hanhong_files')
    if zone.startswith('us-central2'):
        return path.replace('/kmh-nfs-ssd-us-mount/logs/sqa', 'gs://kmh-gcp-us-central2/qiao_zhicheng_hanhong_files')
    if zone.startswith('asia-northeast1-b'):
        return path.replace('/kmh-nfs-ssd-us-mount/logs/sqa', 'gs://kmh-gcp-asia-northeast1-b/qiao_zhicheng_hanhong_files')
    if zone.startswith('europe-west4-a'):
        return path.replace('/kmh-nfs-ssd-us-mount/logs/sqa', 'gs://kmh-gcp/qiao_zhicheng_hanhong_files')
    return None

def check_gs_logdir_exists(logdir, zone, quiet=True):
    '''
    logdir is the dir for resume checkpoint
    zone the is the zone of current / new tpu to resume
    '''
    if logdir.startswith('/kmh-nfs-ssd-eu-mount/logs/sqa'):
        logdir = logdir.replace('/kmh-nfs-ssd-eu-mount/logs/sqa', '/kmh-nfs-ssd-us-mount/logs/sqa')
    # turn to general case (ssd-us-mount)
    if logdir.startswith('/kmh-nfs-us-mount/logs/sqa'):
        logdir = logdir.replace('/kmh-nfs-us-mount/logs/sqa', '/kmh-nfs-ssd-us-mount/logs/sqa')
    
    for z in [zone] + zones_list:
        converted = convert_to_gs_by_zone(logdir, z)
        if converted is not None and check_gs_dir_exists(converted):
            # find correct ckpt path
            if zone == z: return True  # prefer current zone. nothing needs to do
            # Here, a different zone is found. we first copy the checkpoint to current zone gs bucket, and use the same zone
            print(f'Copy checkpoint from {converted} to current zone {zone} gs bucket.')
            dest_path = convert_to_gs_by_zone(logdir, zone)
            FS.copy(converted, dest_path, recursive=True)
            print(f'here')
            return True

    if not quiet: print(f"{FAIL} {logdir} does not exist, please check the path!!\n" * 10)
    return False
