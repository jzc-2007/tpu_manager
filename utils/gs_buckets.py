from .constants import *
import os

def check_gs_dir_exists(gs_dir):
    exit_code = os.system(f"gsutil ls {gs_dir} > /dev/null 2>&1")
    return exit_code == 0

def convert_to_gs(path):
    if path.startswith('gs://'):
        return path
    assert os.path.isabs(path), f'ckpt path {path} is not absolute.'
    for k, v in {
        '/kmh-nfs-ssd-eu-mount/logs/sqa': 'gs://kmh-gcp/qiao_zhicheng_hanhong_files',
        '/kmh-nfs-us-mount/logs/sqa': 'gs://kmh-gcp-us-central2/qiao_zhicheng_hanhong_files',
        '/kmh-nfs-ssd-us-mount/logs/sqa': 'gs://kmh-gcp-us-central2/qiao_zhicheng_hanhong_files',
    }.items():
        if path.startswith(k):
            return path.replace(k, v)
    raise ValueError(f'Cannot convert {path} to gs path, please check the path.')

def check_gs_logdir_exists(logdir, quiet=True):
    gs_path = convert_to_gs(logdir)
    if not check_gs_dir_exists(gs_path):
        if not quiet:
            print(f"{FAIL} {gs_path} does not exist, please check the path.")
        return False
    return True
