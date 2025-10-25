from .constants import *
import os

def check_gs_dir_exists(gs_dir):
    exit_code = os.system(f"gsutil ls {gs_dir} > /dev/null 2>&1")
    return exit_code == 0

def is_checkpoint(path: str): return True # trivial

def convert_to_gs(path: str):
    assert os.path.isabs(path), f'ckpt path {path} is not absolute.'
    if path.startswith('/kmh-nfs-ssd-eu-mount/logs/sqa'):
        new_path = path.replace('/kmh-nfs-ssd-eu-mount/logs/sqa', 'gs://kmh-gcp/qiao_zhicheng_hanhong_files')
        if is_checkpoint(new_path): yield new_path
    if path.startswith('/kmh-nfs-ssd-us-mount/logs/sqa'):
        if 'kmh-tpuvm-v6e' in path:
            new_path = path.replace('/kmh-nfs-ssd-us-mount/logs/sqa', 'gs://kmh-gcp-us-east5/qiao_zhicheng_hanhong_files')
            if is_checkpoint(new_path): yield new_path
            new_path = path.replace('/kmh-nfs-ssd-us-mount/logs/sqa', 'gs://kmh-gcp-us-east1/qiao_zhicheng_hanhong_files')
            if is_checkpoint(new_path): yield new_path
            new_path = path.replace('/kmh-nfs-ssd-us-mount/logs/sqa', 'gs://kmh-gcp-us-central1/qiao_zhicheng_hanhong_files')
            if is_checkpoint(new_path): yield new_path
        new_path = path.replace('/kmh-nfs-ssd-us-mount/logs/sqa', 'gs://kmh-gcp-us-central2/qiao_zhicheng_hanhong_files')
        if is_checkpoint(new_path): yield new_path
        new_path = path.replace('/kmh-nfs-ssd-us-mount/logs/sqa', 'gs://kmh-gcp/qiao_zhicheng_hanhong_files')
        if is_checkpoint(new_path): yield new_path
    if path.startswith('/kmh-nfs-us-mount/logs/sqa'):
        if 'kmh-tpuvm-v6e' in path:
            new_path = path.replace('/kmh-nfs-us-mount/logs/sqa', 'gs://kmh-gcp-us-east5/qiao_zhicheng_hanhong_files')
            if is_checkpoint(new_path): yield new_path
            new_path = path.replace('/kmh-nfs-us-mount/logs/sqa', 'gs://kmh-gcp-us-east1/qiao_zhicheng_hanhong_files')
            if is_checkpoint(new_path): yield new_path
            new_path = path.replace('/kmh-nfs-us-mount/logs/sqa', 'gs://kmh-gcp-us-central1/qiao_zhicheng_hanhong_files')
            if is_checkpoint(new_path): yield new_path
        new_path = path.replace('/kmh-nfs-us-mount/logs/sqa', 'gs://kmh-gcp-us-central2/qiao_zhicheng_hanhong_files')
        if is_checkpoint(new_path): yield new_path
        new_path = path.replace('/kmh-nfs-us-mount/logs/sqa', 'gs://kmh-gcp/qiao_zhicheng_hanhong_files')
        if is_checkpoint(new_path): yield new_path
    if os.path.exists(path): return []
    raise ValueError(f'Cannot convert path {path} to gs path!')

def check_gs_logdir_exists(logdir, quiet=True):
    gs_path = convert_to_gs(logdir)
    for gs_pat in gs_path:
        if check_gs_dir_exists(gs_pat): return True
    if not quiet: print(f"{FAIL} {gs_pat} does not exist, please check the path!!\n" * 10)
    return False
