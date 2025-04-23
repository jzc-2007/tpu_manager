# Auto TPU job manager for Google Cloud Platform (experimental)

This is an automatic job manager for running TPU jobs. It supports auto-resuming the preempted/grpc TPUs, and monitoring the jobs status.

## Quickstart in 2 mins

Tldr usage in **two sentences**: Use ``tpu add-user`` to add your username, then go to your working directory and use ``tpu set-cur 1 username`` to set the working directory. Use ``tpu run <tpu> username``(e.g. ``tpu run v2-32-p2 xibo``) to run the job, and use ``tpu monitor/check username`` to see the status of all your jobs. (The ``tpu run`` command will auto-resume the job when preempted/grpc for preempted TPUs, you don't have to set it.)

More usage in **two sentences**: Use ``tpu tldr`` to see useful commands, and ``tpu clear`` to clear the finished/crashed jobs; use ``tpu -a alias_name full_name username``(e.g. ``tpu -a lr config.training.learning_rate``) to add a new alias, then you can pass the configs such as ``tpu run v2-32-6 xibo lr=0.01``. Use ``tpu describe <tpu>`` to check the environment of the TPU, and ``tpu solve <tpu>`` to solve the environment automatically.

**REMEMBER TO UPDATE YOUR SCRIPTS!**

## Full docs

---

### Basic Usage

- **IMPORTANT**:  
  You should **update your scripts** to the newest version supporting command-line arguments. The newest scripts can be pulled from zhh's repo. The current finishing check is based on wandb final output, so please make sure your scripts are using wandb to log the final output.  
  Also, this script is not very robust to attack, so try not to do OOD things, for example, setting username to be `run`, `false` or Chinese characters.

---

#### Setup

```bash
tpu add-user # Then follow the guide
```

---

#### Read common commands

```bash
tpu tldr
tpu -h command # details of the command
```

The documentation is not very complete now, please refer README as the main source of truth. (Or, look at the code lol)

---

#### Run jobs

```bash
tpu set-cur num username # Set the working directory<num> to the current directory, default directory is 1
tpu ls username # List all the working directories
tpu run tpu_name username [dir=1] [tag=suibian] # Run the job in working directory<dir>, tag is optional and you can see it in the monitor window
```

The `tpu_name` is of the format `v2-32-6` or `v2-32-p1` or `v4-32-py2`.  
For more detail use `tpu -lta` (list tpu aliases), or `tpu -ta alias FULL_TPU_NAME` (to add a new alias).  

**Example:**

```bash
tpu run v2-32-6 xibo # default run the job in working directory 1
tpu run v2-32-p1 lyy dir=2 tag=bird # run the job in working directory 2 
```

The `run` command will ask you whether to reapply when the TPU is preempted.  
You can also add flag `-apply` to avoid asks.

---

#### Kill jobs
We don't recommend you to kill jobs manually, but if you want to do that, you can use:

```bash
tpu kill-job/-k/-kj/-kw -w=<windows_id> username # kill all the jobs in the TPU
```

The command will not kill the tmux window, but will mark the job as "killed". If you want, you can use this command to kill the windows:

```bash
tpu clean username # kill all the tmux windows whose jobs are finished/error/killed
```

Those with child jobs rerunned/resumed will be killed according to the status of their children.

---

#### Monitor jobs

If the TPU is a preemptible TPU, `tpu run` will **auto-resume when GRPC**, and will **auto-reapply and resume** when preempted. You can also use `tpu resume <windows_id> username` to resume jobs. 
The `tpu run` command will open a monitor window to monitor all the jobs you have, and you can also use:

```bash
tpu monitor username
```

to get that. It will update in every 10 seconds, and for one-time check, you can use:

```bash
tpu check username
```

#### Resume/rerun jobs manually
```bash
tpu resume windows=<windows_id> username # resume the job
tpu resume windows=<windows_id> tpu=<tpu> username # resume the job in a new TPU
tpu rerun windows=<windows_id> username # rerun the job
tpu rerun windows=<windows_id> tpu=<tpu> username # rerun the job in a new TPU
```
The difference between `resume` and `rerun` is that `resume` will load the job from the last checkpoint, while `rerun` will start a new job from the beginning.

---

### TPU/environment operations

We support common operations for TPUs, such as:

```bash
tpu apply/reapply tpu_name # apply/reapply the TPU, reapply will delete the TPU and create a new one
```

There're also environment operations supported, such as:

```bash
tpu mount-disk tpu_name # mount the disk and setup wandb for the TPU
tpu describe tpu_name # describe the environment of the TPU
tpu check-status tpu_name # check the status of the TPU, e.g. PREEMPTED, READY, CREATING, etc.
```

Also, an automatic environment solver is used to solve the TPU environment.  
Now it is very simple and can only deal with mounting issue.  
But you are **very welcome** to contribute to it when facing **every environment issue**, to make it a **powerful automatic one-line tool** for solving the complicated TPU environment issue, then we will only need to face same issue **once**!

```bash
tpu solve tpu_name # integrated automatic env solver
```

---

### More Functions

#### Pass configs(alias) on command line

We support passing configs on command line, and you can also set your own config alias by:

```bash
tpu -a/-alias your_alias FULL_NAME username # add a new alias
tpu -sa username # list all the aliases
tpu del-config-alias your_alias username # delete the alias
```

For example, you can do:

```bash
tpu -a lr config.training.learning_rate xibo
```

then,

```bash
tpu run v2-32-6 xibo lr=0.01
tpu run v2-32-6 xibo config.training.learning_rate=0.01 # This is also supported
```

---

#### Kill windows

We recommend using our kill-windows command to kill the windows instead of killing by yourself.  
You can use this command to kill the specific tmux window:

```bash
tpu -kw/kill-window window_number username
```

Also, you can clear all the jobs that are finished/error by:

```bash
tpu clear-finished username # clear all the finished jobs
tpu clear-error username # clear all the error jobs
tpu clear-all username # RECOMMENDED, clear all the finished/error jobs
```

These commands will kill the zombie windows that don't have any jobs running, or zombie jobs that are not running anymore:

```bash
tpu -czw username # clear all the zombie windows
tpu -czj username # clear all the zombie jobs
```

---

#### Add tags to jobs

```bash
tpu add-tag window_num tag_name username # add a tag to the job
```

---

#### Sanity check and tests

Some very naive sanity checks are implemented in `unit_tests.py`.

---

### For Developers

The user interface is implemented in `tpu.py`, and the specific detail is in `utils/`.  
`monitor.py` does the check and resume work, and will be run all day.

For `utils/`:  
- `desciptions.py` does all the documentation work  
- `operate.py` does the tpu remote operations  
- `job.py` does the job management  
- `directories.py` deals with the user working dirs  
- `logger.py` does most of the logging with meta-data  
(see more in next paragraph)

---

### Data Format

The key data is stored in `data.json`, and the program reads and writes it using the API in `data_io.py`, which implements locking (in `lock.json`).  
The structure of `data.json` is as follows:

<details>
<summary><strong>Click to expand full data.json structure</strong></summary>

```json
{
    "users": {
        "username": {
            "id": 0,
            "name": "username",
            "tmux_name": "username",
            "working_dir": {"1": "/path"},
            "job_data": [],
            "config_aliases": {"lr": "config.training.lr"},
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
    "user_list": ["username"],
    "id_list": [0],
    "id_user_dict": {"0": "username"},
    "user_id_dict": {"username": 0},
    "tpu_aliases": {"v2-1": "kmh-tpuvm-v2-32-1"},
    "all_tpus": {
        "europe-west4-a": ["..."],
        "us-central1-a": ["..."],
        "us-central2-b": ["..."],
        "preemptible": ["..."]
    },
    "monitor_config": {
        "test_freq": 3600,
        "checking_freq": 600
    },
    "wandb_api_key": "...",
    "conda_env_name": "NNX"
}
```

</details>

Each job is described as:

```json
{
    "user": "username",
    "windows_id": 1,
    "job_dir_id": 1,
    "job_dir": "/your/code/path",
    "tpu": "kmh-tpuvm-v2-32-preemptible-1",
    "job_tags": null,
    "log_dir": "/your/log/path",
    "extra_configs": "--lr=0.01",
    "status": "running",
    "error": null,
    "stage": 0,
    "monitor": true,
    "rules": {
        "preempted": "reapply",
        "grpc": "resume"
    },
    "extra_msgs": {},
    "start_time": "20250420_011026"
}
```

---

### Future Work

- [ ] More testing/docs
- [ ] Customized monitor window
- [ ] Support to read the spreadsheet, then we can auto-choose the TPU to run a job  
- [ ] More auto env solvers  
- [ ] Logging for every user so that you can check the things happen since last time  

---
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
PASS_KA=0

if [ -n "$1" ]; then
	echo "1st arg(ka): $1"
	if [[ "$1" == ka=* ]]; then
		ka=${1#*=}
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
# add all the configs pass in to cmd
for arg in "$@"; 
    do
        if [[ $arg == --config* ]]; then
            export cmd="$cmd $arg"
        fi
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
