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
    if path.startswith('gs://'):
        return path
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
    print(f"{WARNING} convert_to_gs_by_zone: Unknown zone {zone}")
    return None

def convert_name(path:str, zone: str):
    if zone is None:
        return path
    # convert all the substring with ['us-central1', 'us-east1', 'us-east5', 'us-central2', 'asia-northeast1-b', 'europe-west4-a'] to the corresponding zone
    for z in ['us-central1', 'us-east1', 'us-east5', 'us-central2', 'asia-northeast1-b', 'europe-west4-a']:
        if z in path:
            return path.replace(z, zone)
    return path

def copy_ckpt(path:str, src_zone: str = None, dst_zone: str = None):
    if src_zone is None:
        src_zone = get_zone_from_workdir(path)
    if dst_zone is None:
        dst_zone = get_zone_from_workdir(path)
    converted_src = convert_to_gs_by_zone(path, src_zone)
    converted_dst = convert_to_gs_by_zone(path, dst_zone)
    # converted_src = convert_name(converted_src, src_zone)
    # converted_dst = convert_name(converted_dst, dst_zone)
    print(f'converted_src: {converted_src}')
    print(f'converted_dst: {converted_dst}')
    print(f'copying {converted_src} to {converted_dst}...')
    # if converted_dst is not None, check if it exists
    if converted_dst is not None:
        if FS.exists(converted_dst):
            print(f'{converted_dst} already exists, are you sure you want to overwrite it? (y/n): ')
            answer = input()
            if answer.strip().lower() != 'y' and answer.strip().lower() != 'yes':
                print(f'{converted_dst} already exists, operation cancelled')
                return False
            else:
                print(f'{converted_dst} already exists, overwriting...')
    if converted_src is not None and converted_dst is not None:
        FS.copy(converted_src, converted_dst, recursive=True)
    return True

def check_gs_logdir_exists(logdir, zone, quiet=True, all=False):
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

            if all:
                FS.copy(converted, dest_path, recursive=True)
                print(f'Copy done.')
                return True
            
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

def copy_checkpoint(dir, target_zone, src_zone=None, all=False):
    if target_zone not in zones_list:
        print(f"{FAIL} copy_checkpoint: target_zone {target_zone} is not valid. Supported zones: {zones_list}")
        return False
    # return copy_ckpt(dir, src_zone=src_zone, dst_zone=target_zone)
    return check_gs_logdir_exists(dir, target_zone, quiet=False, all=all)

if __name__ == '__main__':
    logdir = '/kmh-nfs-ssd-eu-mount/logs/sqa/TS-imgnet/20251026_035445_aq9ubo_kmh-tpuvm-v6e-32-spot-101_us-central1-b__b_lr_ep_eval'
    zone = 'us-east1-d'
    check_gs_logdir_exists(logdir, zone, quiet=False)
    print('done')