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
            
            # Here, a different zone is found.
            print(f'Found existing logdir in {z}: {converted}')
            print(f'Preparing to copy checkpoint from {converted} to current zone {zone} gs bucket...')
            
            dest_path = convert_to_gs_by_zone(logdir, zone)
            
            # --- Start of Modification: Find Latest Checkpoint ---
            try:
                # 1. List files in the source directory
                # gcsfs.ls usually returns paths like 'bucket/path/file' (without gs://)
                files = FS.ls(converted)

                print(f'files: {files}')
                
                ckpt_candidates = []
                for f in files:
                    # Handle path strings carefully to get just the folder name
                    basename = f.rstrip('/').split('/')[-1]
                    
                    # 2. Filter for checkpoint_XXXX folders
                    if basename.startswith('checkpoint_'):
                        try:
                            # 3. Parse step number (e.g., checkpoint_200160 -> 200160)
                            step = int(basename.split('_')[-1])
                            ckpt_candidates.append((step, f))
                        except ValueError:
                            raise ValueError(f'Unexpected checkpoint folder name format: {basename}')
                            continue
                
                print(f'Checkpoint candidates found: {ckpt_candidates}')
                if ckpt_candidates:
                    # 4. Sort by step (descending) and pick the first one
                    ckpt_candidates.sort(key=lambda x: x[0], reverse=True)
                    latest_step, latest_src_path = ckpt_candidates[0]

                    print(f'Latest checkpoint determined: step {latest_step}, path {latest_src_path}')
                    
                    # Ensure source path has gs:// prefix for consistency
                    if not latest_src_path.startswith('gs://'):
                        latest_src_path = 'gs://' + latest_src_path
                    
                    ckpt_foldername = f"checkpoint_{latest_step}"
                    
                    # Construct specific destination path
                    # We need to append the checkpoint folder name to dest_path
                    final_dest_path = f"{dest_path.rstrip('/')}/{ckpt_foldername}"
                    
                    print(f'Detected latest checkpoint: {ckpt_foldername}')
                    print(f'Copying ONLY {latest_src_path} to {final_dest_path} ...')
                    
                    FS.copy(latest_src_path, final_dest_path, recursive=True)
                    
                else:
                    raise ValueError('No checkpoint_ folders found in the source directory.')
                    # Fallback: No checkpoint_ folders found (maybe just logs?), copy everything
                    print(f'No specific "checkpoint_X" folders found. Copying entire directory...')
                    FS.copy(converted, dest_path, recursive=True)
                    
            except Exception as e:
                raise e
                print(f"Error during smart checkpoint copy: {e}")
                print("Falling back to full copy...")
                FS.copy(converted, dest_path, recursive=True)
            # --- End of Modification ---

            print(f'Copy done.')
            return True

    if not quiet: print(f"{FAIL} {logdir} does not exist, please check the path!!\n" * 10)
    return False

if __name__ == '__main__':
    logdir = '/kmh-nfs-ssd-eu-mount/logs/sqa/TS-imgnet/20251026_035445_aq9ubo_kmh-tpuvm-v6e-32-spot-101_us-central1-b__b_lr_ep_eval'
    zone = 'us-east1-d'
    check_gs_logdir_exists(logdir, zone, quiet=False)
    print('done')