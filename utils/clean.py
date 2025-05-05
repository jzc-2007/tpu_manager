import os
import shutil
import re

def file_to_num(file):
    content = re.match(r'checkpoint_(\d+)', file)
    if content:
        return int(content.group(1))
    else:
        raise ValueError(f'File name {file} does not match the pattern')

def remove(file,reason,safe=True,quiet=False):
    do = 'y'
    if safe:
        do = input(f'\033[91mRemove\033[0m {file}? \n\033[92mReason\033[0m: {reason} (y/n)')
    if do == 'y':
        if not quiet:
            print('\033[93mRemoving\033[0m', file)
        os.system(f'sudo rm -rf {file}')
    else:
        if not quiet:
            print(f'File {file} not removed')
    os.system('sleep 1;clear')
        
def resursive_get_ckpt(root):
    out = os.walk(root)
    for father,folders,files in list(out):
        if len([c for c in folders if 'checkpoint' in c]) > 0:
            yield father

def get_human_read_size(file):
    return os.popen(f'du -sh {file}').read().split('\t')[0]

def clean(base_dir, safe = True, quiet = False):
    out = os.walk(base_dir)
    for father,folders,files in out:
        if not os.path.exists(father):
            continue
        folders = [f for f in folders if os.path.exists(os.path.join(father,f))]
        files = [f for f in files if os.path.exists(os.path.join(father,f))]
        
        if 'output.log' in files:
            for father_ckpt in resursive_get_ckpt(father):
                checkpoints_in_father = os.listdir(father_ckpt)
                if 'FINAL_MODEL' in checkpoints_in_father:
                    for c in father_ckpt:
                        if 'checkpoint' in c:
                            remove(os.path.join(father_ckpt,c), 'The FINAL model {} exists with size {}, this checkpoint also have size {}'.format(os.path.join(father_ckpt,'FINAL_MODEL'), get_human_read_size(os.path.join(father_ckpt,'FINAL_MODEL')), get_human_read_size(os.path.join(father_ckpt,c))), safe = safe, quiet = quiet)
                else:
                    fs = [c for c in checkpoints_in_father if 'checkpoint' in c]
                    if len(fs) > 1:
                        fs.sort(key=file_to_num)
                        for c in fs[:-1]:
                            remove(os.path.join(father_ckpt,c), 'There is a older checkpoint {} with size {}, this checkpoint also have size {}'.format(fs[-1], get_human_read_size(os.path.join(father_ckpt,fs[-1])), get_human_read_size(os.path.join(father_ckpt,c))), safe = safe, quiet = quiet)

def clean_us(safe=True, quiet=False):
    clean('/kmh-nfs-us-mount/logs/sqa/', safe=safe, quiet=quiet)

def clean_eu(safe=True, quiet=False):
    clean('/kmh-nfs-ssd-eu-mount/logs/sqa/', safe=safe, quiet=quiet)