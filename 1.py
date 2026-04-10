import json

x = {
    'running': [],
    'finished': [],
}

with open(f'/kmh-nfs-ssd-us-mount/code/qiao/work/tpu_manager/sqa.json', 'w') as file:
    # write as json
    json.dump(x, file)
    file.write('\n')
