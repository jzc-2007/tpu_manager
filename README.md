# Auto TPU job manager for Google Cloud Platform (experimental)

This is a automatic job manager to run TPU jobs. It supports to auto-resume the preempted/grpc TPUs, and monitor the jobs status.

### Basic Usage
- **IMPORTANT**:
You should **update your scripts** to the newest version that supports command-line arguments. The newest scripts can be pulled from zhh's repo, and please update ka.sh(which **will not be auto-updated**) according to the README. The current finishing check is based on wandb final output, so please make sure your scripts are using wandb to log the final output.
#### Setup
```bash
tpu add-user # Then follow the guide
```
#### Read common commands
```bash
tpu tldr
tpu -h command # details of the command
```
The documentation is not very complete now.
#### Run jobs
```bash
tpu set-cur num username # Set the working directory<num> to the current directory, default directory is 1
tpu ls username # List all the working directories
tpu run tpu_name username [dir=1] [tag=suibian] # Run the job in working directory<dir>, tag is optional and you can see it in the monitor window
```
The ``tpu_name`` is of the format ``v2-32-6``or ``v2-32-p1`` or ``v4-32-py2``. For more detail use ``tpu -lta``(list tpu aliases), or ``tpu -ta alias FULL_TPU_NAME``(to add a new alias).  

#### monitor jobs
If the TPU is a preemptible TPU, ``tpu run`` will **auto-rerun when GRPC**, and will **auto-reapply** and rerun when preempted. 
The ``tpu run`` command will open a monitor window to monitor all the jobs you have, and you can also use 
```bash
tpu monitor username
``` 
to get that. It will update every 5 seconds, and for one-time check, you can use ``tpu check username``.

### More Functions

#### Pass configs(alias) on command line
We support passing configs on command line, and you can also set your own config alias by
```bash
tpu -a/-alias your_alias FULL_NAME username # add a new alias
tpu -sa username # list all the aliases
tpu del-config-alias your_alias username # delete the alias
```
For example, you can do
```bash
tpu -a lr config.training.learning_rate xibo
```
then,
```bash
tpu run v2-32-6 xibo lr=0.01
tpu run v2-32-6 xibo config.training.learning_rate=0.01 # This is also supported
```
#### kill windows/jobs
We recommend to use our kill windows command to kill the windows. You can use this to kill the specific tmux window
```bash
tpu -kw/kill-window window_number username
```

Also, you can clear all the jobs that is finished/error by
```bash
tpu clear-finished username # clear all the finished jobs
tpu clear-error username # clear all the error jobs
tpu clear-all username # clear all the jobs
```

These commands will kill the zombie windows that don't have any jobs running, or zombie jobs that are not running anymore.
```bash
tpu -czw username # clear all the zombie windows
tpu -czj username # clear all the zombie jobs
```
#### apply/re-apply TPU
```bash
tpu apply tpu_alias/tpu_name # apply the TPU
tpu reapply tpu_alias/tpu_name # delete and apply the TPU
```

#### add tags to jobs
```bash
tpu add-tag window_num tag_name username # add a tag to the job
```

#### Sanity check and tests
Some very naive sanity checks is implemented in ``unit_tests.py``.

### New Scripts
```bash
# 卡.sh
source config.sh

if [ -z "$OWN_CONDA_ENV_NAME" ]; then
    echo "Please set your own config.sh. See README for reference"
    sleep 60
    exit 1
fi

if [ -z "$TASKNAME" ]; then
    echo "Please set your own config.sh. See README for reference"
    sleep 60
    exit 1
fi

if [ -z "$1" ]; then

############## TPU VMs ##############

# export VM_NAME=kmh-tpuvm-v2-32-1
# export VM_NAME=kmh-tpuvm-v2-32-2
# export VM_NAME=kmh-tpuvm-v2-32-3
# export VM_NAME=kmh-tpuvm-v2-32-4
# export VM_NAME=kmh-tpuvm-v2-32-5
# export VM_NAME=kmh-tpuvm-v2-32-6
# export VM_NAME=kmh-tpuvm-v2-32-7
# export VM_NAME=kmh-tpuvm-v2-32-8
# export VM_NAME=kmh-tpuvm-v3-32-1
export VM_NAME=kmh-tpuvm-v2-32-preemptible-1
# export VM_NAME=kmh-tpuvm-v2-32-preemptible-2
# export VM_NAME=kmh-tpuvm-v3-32-preemptible-1
# export VM_NAME=kmh-tpuvm-v3-32-11
# export VM_NAME=kmh-tpuvm-v3-32-12
# export VM_NAME=kmh-tpuvm-v3-32-13
# export VM_NAME=kmh-tpuvm-v4-8-6

#####################################
else
    echo ka: use command line arguments
        export VM_NAME=$1
fi
# Zone: your TPU VM zone
if [[ $VM_NAME == *"v4"* ]]; then
    export ZONE=us-central2-b
elif [[ $VM_NAME == *"v3"* ]]; then
    export ZONE=europe-west4-a
else
    if [[ $VM_NAME == *"v2-32-4"* ]]; then
        export ZONE=europe-west4-a
    elif [[ $VM_NAME == *"v2-32-preemptible-2"* ]]; then
        export ZONE=europe-west4-a
    else
        export ZONE=us-central1-a
    fi
fi

# DATA_ROOT: the disk mounted
# FAKE_DATA_ROOT: the fake data (imagenet_fake) link
# USE_CONDA: 1 for europe, 2 for us (common conda env)

if [[ $ZONE == *"europe"* ]]; then
    export DATA_ROOT="kmh-nfs-ssd-eu-mount"
    # export TFDS_DATA_DIR='gs://kmh-gcp/tensorflow_datasets'  # use this for imagenet
    export TFDS_DATA_DIR='/kmh-nfs-ssd-eu-mount/code/hanhong/dot/tensorflow_datasets'
    export USE_CONDA=1
else
    export DATA_ROOT="kmh-nfs-us-mount"
    export USE_CONDA=1
    # export TFDS_DATA_DIR='gs://kmh-gcp-us-central2/tensorflow_datasets'  # use this for imagenet
    export TFDS_DATA_DIR='/kmh-nfs-us-mount/data/tensorflow_datasets'
fi

if [[ $USE_CONDA == 1 ]]; then
    export CONDA_PY_PATH=/$DATA_ROOT/code/qiao/anaconda3/envs/$OWN_CONDA_ENV_NAME/bin/python
    export CONDA_PIP_PATH=/$DATA_ROOT/code/qiao/anaconda3/envs/$OWN_CONDA_ENV_NAME/bin/pip
    echo $CONDA_PY_PATH
    echo $CONDA_PIP_PATH
fi

```

### For Developers

The user interface is implemented in ``tpu.py``, and the specific detail are in ``utils/``. ``monitor.py`` do the check and rerun work, and will be run all day.

For ``utils/``,``desciptions.py`` do all the documentation work,``operate.py`` do the tpu remote operations, and ``job.py`` do the job management. ``directories.py`` deal with the user working dirs, and ``logger.py`` do most of the logging with meta-data.(see more in next paragraph)

The key data is stored in ``data.json``, and the program read and write it using the API in ``data_io.py``, which implements locking.(in ``lock.json``) The structure of ``data.json`` is of the following:
```bash
{
    "users": {
        username: {
            "id": id,
            "name": name,
            "tmux_name": tmux_name,
            "working_dir": {num: path},
            "job_data": [ job1, job2 ],
            "config_aliases": {
                alias: config_name
            },
            "settings": {
                "monitor_after_run": true,
                "monitor_upd_time": 5,
                "monitor_length": 800,
                "monitor_verbose": false,
                "show_length": 300,
                "time_zone": "us"
            },
            "windows_offset": 42,
            "logs": []
        }
    },
    "user_list": [
        username
    ],
    "id_list": [
        id
    ],
    "id_user_dict": {
        id: username
    },
    "user_id_dict": {
        username: id
    },
    "tpu_aliases": {
        "alias": "kmh-tpu-name"
    },
    "all_tpus": {
        "europe-west4-a": ["..."],
        "us-central1-a": ["..."],
        "us-central2-b": ["..."],
        "preemptible": ["..."]
    }
}
```

and each job is described as
```bash
{
    "user": username,
    "windows_id": windows_id,
    "job_dir_id": NUM_CODE_DIRECTORY,
    "job_dir": CODE_DIRECTORY,
    "tpu": "kmh-tpuvm-v2-32-preemptible-1",
    "job_tags": null,
    "log_dir": LOG_DIR,
    "extra_configs": EXTRA_COMMAND_FOR_CONFIG, # e.g." --config.training.grpc=True"
    "status": status, # "running", "finished", "error", "rerunned"
    "error": null,
    "stage": 0, # The number of times the job has been rerunned
    "monitor": true, # whether to monitor the job
    "rules": {
        "preempted": "reapply",
        "grpc": "rerun"
    }, # The rules for the job when error
    "extra_msgs": {}, # record parent/child when rerun
    "start_time": "20250420_011026"
},
```

Future work:
- [ ] More testing/docs
- [ ] Support to read the spreadsheet, then we can auto-choose the TPU to run a job
- [ ] More auto env solvers