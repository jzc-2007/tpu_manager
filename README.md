# Auto TPU job manager for Google Cloud Platform (experimental)

This is an automatic job manager for running TPU jobs. It supports auto-resuming the preempted/grpc TPUs, and monitoring the jobs status.

### Basic Usage
- **IMPORTANT**:
You should **update your scripts** to the newest version supporting command-line arguments. The newest scripts can be pulled from zhh's repo. The current finishing check is based on wandb final output, so please make sure your scripts are using wandb to log the final output.

Also, this script is not very robust to attack, so try not to do OOD things, for example, setting username to be `run`, `false` or Chinese characters.

#### Setup
```bash
tpu add-user # Then follow the guide
```
#### Read common commands
```bash
tpu tldr
tpu -h command # details of the command
```
The documentation is not very complete now, please refer README as the main source of truth. (Or, look at the code lol)
#### Run jobs
```bash
tpu set-cur num username # Set the working directory<num> to the current directory, default directory is 1
tpu ls username # List all the working directories
tpu run tpu_name username [dir=1] [tag=suibian] # Run the job in working directory<dir>, tag is optional and you can see it in the monitor window
```
The ``tpu_name`` is of the format ``v2-32-6``or ``v2-32-p1`` or ``v4-32-py2``. For more detail use ``tpu -lta``(list tpu aliases), or ``tpu -ta alias FULL_TPU_NAME``(to add a new alias). 

Example:
```bash
tpu run v2-32-6 xibo # default run the job in working directory 1
tpu run v2-32-p1 lyy dir=2 tag=bird # run the job in working directory 2 
```

#### monitor jobs
If the TPU is a preemptible TPU, ``tpu run`` will **auto-rerun when GRPC**, and will **auto-reapply and rerun** when preempted. 
The ``tpu run`` command will open a monitor window to monitor all the jobs you have, and you can also use 
```bash
tpu monitor username
``` 
to get that. It will update in every 5 seconds, and for one-time check, you can use ``tpu check username``.

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
We recommend using our kill-windows command to kill the windows instead of killing by yourself. You can use this command to kill the specific tmux window:
```bash
tpu -kw/kill-window window_number username
```

Also, you can clear all the jobs that are finished/error by
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
Some very naive sanity checks are implemented in ``unit_tests.py``.


### For Developers

The user interface is implemented in ``tpu.py``, and the specific detail is in ``utils/``. ``monitor.py`` does the check and rerun work, and will be run all day.

For ``utils/``,``desciptions.py`` does all the documentation work,``operate.py`` does the tpu remote operations, and ``job.py`` does the job management. ``directories.py`` deals with the user working dirs, and ``logger.py`` does most of the logging with meta-data.(see more in next paragraph)

The key data is stored in ``data.json``, and the program read and write it using the API in ``data_io.py``, which implements locking(in ``lock.json``). The structure of ``data.json`` is of the following:
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
- [ ] Logging for every user so that you can check the things happen since last time

### New Scripts
```bash
# ka.sh
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

```bash
# staging.sh
local PASS_KA=0
if [ -n "$2" ]; then
	if [ "$2" == "ka="* ]; then
		ka=${2#*=}
		export VM_NAME=$ka
		export PASS_KA=1
	fi
fi

source ka.sh $VM_NAME
now=`date '+%y%m%d%H%M%S'`
salt=`head /dev/urandom | tr -dc a-z0-9 | head -c6`
git config --global --add safe.directory $(pwd)
HERE=$(pwd)
commitid=`git show -s --format=%h`  # latest commit id; may not be exactly the same as the commit
export STAGEDIR=/$DATA_ROOT/staging/$USER/${now}-${salt}-${commitid}-code

echo 'Staging files...'
rsync -av . $STAGEDIR --exclude=tmp --exclude=.git --exclude=__pycache__ --exclude="*.png" --exclude="history" --exclude=wandb --exclude="zhh_code" --exclude="zhh"
cp -r /kmh-nfs-ssd-eu-mount/code/hanhong/MyFile/research_utils/Jax/zhh $STAGEDIR
echo 'Done staging.'

sudo chmod 777 -R $STAGEDIR

cd $STAGEDIR
echo 'Current dir: '`pwd`
# ------------------------------------------------

if [ $PASS_KA -eq 0 ]; then
	source run_remote.sh ${@:1}
else
	source run_remote.sh ${@:2}
fi

cd $HERE
```

```bash
# run_remote.sh
source config.sh
CONDA_ENV=$OWN_CONDA_ENV_NAME

echo Running at $VM_NAME $ZONE

now=`date '+%Y%m%d_%H%M%S'`
export salt=`head /dev/urandom | tr -dc a-z0-9 | head -c6`
JOBNAME=${TASKNAME}/${now}_${salt}_${VM_NAME}_${CONFIG}_b${batch}_lr${lr}_ep${ep}_eval

LOGDIR=/$DATA_ROOT/logs/$USER/$JOBNAME

sudo mkdir -p ${LOGDIR}
sudo chmod 777 -R ${LOGDIR}
echo 'Log dir: '$LOGDIR
echo 'Staging dir: '$STAGEDIR

current_window=`tmux display-message -p '#S:#I'`
echo "Current tmux window: $current_window"

echo 'tpu: '$VM_NAME
tpu upd-log $current_window $LOGDIR $VM_NAME $now

export cmd="cd $STAGEDIR
echo 'Current dir: '
pwd
$CONDA_PY_PATH main.py --workdir=${LOGDIR} --mode=remote_run --config=configs/load_config.py:remote_run "

# add all the configs pass in to cmd
for arg in "$@"; do
    export cmd="$cmd $arg"
done

echo "Running command: $cmd"

gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
    --worker=all --command "${cmd}" 2>&1 | tee -a $LOGDIR/output.log

if grep -q "wandb: Run history:" $LOGDIR/output.log; then
    echo "Job completed successfully"
    tpu finish-job $current_window
else
    echo "Job failed"
fi
```

```bash
# kill_remote.sh
if [ -n "$1" ]; then
	export VM_NAME=$1
	source ka.sh $1
else
	source ka.sh
fi

echo 'To kill jobs in: '$VM_NAME 'in' $ZONE' after 2s...'
sleep 2s

echo 'Killing jobs...'
gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE --worker=all \
    --command "
pgrep -af python | grep 'main.py' | grep -v 'grep' | awk '{print \"sudo kill -9 \" \$1}' | sh
" # &> /dev/null
echo 'Killed jobs.'
```


gcloud compute tpus describe kmh-tpuvm-v2-32-preemptible-1 --zone=us-central1-a --format='value(state)'