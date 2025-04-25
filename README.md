# Auto TPU job manager for Google Cloud Platform (experimental)

This is an automatic job manager for running TPU jobs. It supports auto-resuming the preempted/grpc TPUs, and monitoring the jobs status.

## Quickstart in 2 mins

Here is a quick guide of the common usage, and you can find more details in the full docs below.

Tldr usage in **two sentences**: Use ``tpu add-user`` to add your username, then go to your working directory(where you have your scripts and code) and use ``tpu set-cur 1 username`` to set the working directory. Use ``tpu run <tpu> username``(e.g. ``tpu run v2-32-p2 xibo``) to run the job, and use ``tpu monitor/check username`` to see the status of all your jobs. (The ``tpu run`` command will auto-resume the job when preempted/grpc for preempted TPUs, you don't have to set it.)

More usage in **two sentences**: Use ``tpu tldr`` to see useful commands, and ``tpu clear username`` to clear the finished/crashed jobs; use ``tpu -a alias_name full_name username``(e.g. ``tpu -a lr config.training.learning_rate``) to add a new alias, then you can pass the configs such as ``tpu run v2-32-6 xibo lr=0.01``. Use ``tpu describe <tpu>`` to check the environment of the TPU, and ``tpu solve <tpu>`` to solve the environment automatically.

**REMEMBER TO UPDATE YOUR SCRIPTS!**

## Full docs

<details>
<summary> <strong>Setup(<strong>IMPORTANT</strong>) </strong></summary>

You should **update your scripts** to the newest version supporting command-line arguments. The newest scripts can be pulled from zhh's repo. The current finishing check is based on wandb final output, so please make sure your scripts are using wandb to log the final output.  
Also, this script is not very robust to attack, so try not to do OOD things, for example, setting username to be `run`, `false` or Chinese characters.

Use ``tpu add-user`` and follow the instructions to add your username.

</details>

<details>
<summary> <strong>Setting Working Directory & Running Jobs (<strong>IMPORTANT</strong>) </strong></summary>

The working directory is where you have your scripts and code. You can set multiple working directories and choose one when running code. The default working directory is `1`.  
You can set the working directory and run the job by:

```bash
tpu set-cur num username # Set the working directory <num> to the current directory, default directory is 1
tpu ls username # List all the working directories
tpu run tpu_name username [dir=1] [tag=suibian] # Run the job in working directory <dir>, tag is optional and visible in the monitor window
```

The `tpu_name` is of the format of the **pre-defined tpu aliases** , like `v2-32-6`, `v2-32-p1`, or `v4-32-py2`. You can also pass full-name such as `kmh-tpuvm-v2-32-1`.

For all the aliases, use `tpu -lta` (list TPU aliases) to see. You can also add aliases by `tpu -ta alias FULL_TPU_NAME`. Please don't add aliases that may lead to contradictions to other things, for example `username` or `tag` or `config` or `s`.

**Example:**

```bash
tpu run v2-32-6 xibo # Default: run the job in working directory 1
tpu run v2-32-p1 lyy dir=2 tag=bird # Run the job in working directory 2 
```

The `run` command will ask whether to reapply when the TPU is preempted.  
You can also add the flag `-apply` to skip the prompt.

</details>

<details>
<summary> <strong>Kill Jobs/Windows </strong></summary>

To kill a job, use:

```bash
tpu kill-job/-k/-kj -w=<windows_id>/window=<windows_id> username # Kill all the jobs in the TPU
```

This command will not kill the tmux window but will mark the job as "killed." To clean up, use the integrated `clean` command:

```bash
tpu clean username # Kill all tmux windows whose jobs are finished/error/killed
```

Jobs with child jobs that were rerun/resumed will be killed based on the status of their children.

To kill a specific tmux window:

```bash
tpu -kw/kill-window window_number username
```

After killing windows, some jobs may become "zombies" (i.e., jobs without associated windows). Use these helpers to clean zombies:

```bash
tpu -czw username # Clear all zombie windows
tpu -czj username # Clear all zombie jobs
tpu clear-finished username # Clear all finished jobs
tpu clear-error username # Clear all error jobs
tpu clear-all username # RECOMMENDED: Clear all finished/error jobs
```

The `clean` command integrates these actions, so using `kill-job + clean` is strongly recommended instead of manually killing windows with `tmux kill-window`.

</details>

<details>
<summary> <strong>Monitoring </strong></summary>

If the TPU is preemptible, `tpu run` will **auto-resume on GRPC errors** and **auto-reapply and resume** when preempted. You can also use `tpu resume <windows_id> username` to resume jobs.  
The `tpu run` command opens a monitor window to track all your jobs. Alternatively, you can use:

```bash
tpu monitor username
```

This updates every 10 seconds. For a one-time check, use:

```bash
tpu check username
```

</details>

<details>
<summary> <strong>TPU/Environment Operations </strong></summary>

We support common TPU operations, such as:

```bash
tpu apply/reapply tpu_name # Apply/reapply the TPU; reapply deletes and recreates the TPU
```

Environment operations are also supported:

```bash
tpu mount-disk tpu_name # Mount the disk and set up wandb for the TPU
tpu describe tpu_name # Describe the TPU environment
tpu check-status tpu_name # Check the TPU status (e.g., PREEMPTED, READY, CREATING, etc.)
```

An automatic environment solver is available to address TPU environment issues.  
Currently, it handles mounting issues, but contributions are welcome to enhance it into a **powerful one-line tool** for solving complex TPU environment problems you have encountered. This way, ideally we only need to manully fix every possible issue **once**!

```bash
tpu solve tpu_name # Integrated automatic environment solver
```

</details>


<details>
<summary> <strong>Passing Configs/Adding Tags </strong></summary>

We support passing configs on the command line, and you can also set your own config alias by:


```bash
tpu -a/-alias your_alias FULL_NAME username # add/change an alias
tpu -sa username # list all the aliases
tpu del-config-alias your_alias username # delete the alias
```

For example, you can do:

```bash
tpu -a lr config.training.learning_rate xibo
```

Then:

```bash
tpu run v2-32-6 xibo lr=0.01
tpu run v2-32-6 xibo config.training.learning_rate=0.01 # This is also supported
```

<details>
<summary> <strong>Some default aliases </strong></summary>

```bash
"lr": "config.training.learning_rate"
"bs": "config.training.batch_size"
"ep": "config.training.num_epochs"
"wd": "config.training.weight_decay"
"b1": "config.training.adam_b1"
"b2": "config.training.adam_b2"
"ckpt": "config.training.checkpoint_per_epoch"
```

</details>

You can add tags to the existing jobs (so that they will be shown in the monitor) by:

```bash
tpu add-tag window_num tag_name username # add a tag to the job
```

</details>

<details>
<summary> <strong>Resuming/Rerunning Manually </strong></summary>

```bash
tpu resume windows=<windows_id> username # resume the job
tpu resume windows=<windows_id> tpu=<tpu> username # resume the job in a new TPU
tpu rerun windows=<windows_id> username # rerun the job
tpu rerun windows=<windows_id> tpu=<tpu> username # rerun the job in a new TPU
```

The difference between `resume` and `rerun` is that `resume` will load the job from the last checkpoint, while `rerun` will start a new job from the beginning.

</details>

<details>
<summary> <strong>More Resuming/Rerunning Rules</strong></summary>

Our default rules for resuming/rerunning are as follows:  
For preempted TPUs, we will reapply the TPU and resume the job when the job is preempted, and resume the job when the job encounters a GRPC error. For non-preempted TPUs, we will not perform any operations.  

You can pass the `rule=<rule>` to the `tpu run` command to set the rules. The available rules are:  
- `reapply`: Reapply when GRPC error occurs or when preempted.  
- `pass` (default for non-preempted TPUs): Do nothing.  
- `rerun`: Rerun when GRPC error occurs, reapply when preempted.  
- `pre` (default for preempted TPUs): Reapply when GRPC error occurs, resume
- `resume`(recommend for non-preempted TPUs, may change to default someday): Resume when GRPC error occurs, pass when preempted.

For example, if you want a job running in preempted TPUs to be rerunned instead of resumed when grpc, you can do:
```bash
tpu run v2-32-p2 xibo rule=rerun
```

If you want a job running in non-preempted TPUs to be resumed when grpc, you can do:

```bash
tpu run v2-32-2 xibo rule=resume
```

You can see all the rules using

```bash
tpu check-rules
```

</details>
<details>
<summary> <strong>Documentation </strong></summary>

```bash
tpu tldr
tpu -h command # details of the command
```

</details>

## For Developers

<details>
<summary> <strong>Code Structure </strong></summary>

The user interface is implemented in `tpu.py`, and the specific function implementation is in `utils/`.  
`MONITOR.py` does the check and resume work, and will be run all day, it will check the jobs and do unit tests occansionally according to ``data["MONITOR_config"]``(You can see the full format of ``data.json`` below, which is the key matadata we maintain to manage all the jobs).

We use MONITOR to referr to the global monitor process to separate it from the local monitor window for 
each user. 

For `utils/`:  
- `desciptions.py` does all the documentation work  
- `operate.py` does the tpu remote operations  
- `jobs.py` does the job management  
- `directories.py` deals with the user working dirs  
- `logger.py` does most of the logging with meta-data  
- `helpers.py` does the helper functions
- `error_handler.py` does the error handling works
- `unit_tests.py` does the unit tests (sanity checks)
- `develop.py` does the developer tools, to safely modify the metadata and avoid conflicts with current jobs
(see more in next paragraph)
<details>
<summary> <strong>Data Format </strong></summary>

The key data is stored in `data.json`, and the program reads and writes it using the API in `data_io.py`, which implements locking (in `lock.json`).  
The structure of `data.json` is as follows:

<details>
<summary> <strong>Full data.json structure </strong></summary>

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

Each job is described as:

<details>
<summary> <strong>Full job structure </strong></summary>

```json
{
    "user": "username",
    "windows_id": 1,
    "job_dir_id": 1,
    "job_dir": "/your/code/path",
    "tpu": "kmh-tpuvm-v2-32-preemptible-1",
    "job_tags": null,
    "log_dir": "/your/log/path",
    "staage_dir": "/your/staging/path",
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
    "start_time": "20250420_011026",
    "customized_settings": {}
}
```

</details>

</details>

</details>
</details>

</details>

</details>

## Future Work

- [ ] More testing/docs
- [ ] Customized monitor window
- [ ] Support to read the spreadsheet, then we can auto-choose the TPU to run a job  
- [ ] More auto env solvers  
- [ ] Logging for every user so that you can check the things happen since last time  

## New Scripts
<details>
<summary> <strong>ka.sh </strong></summary>

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

</details>

<details>
<summary> <strong>staging.sh </strong></summary>

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

</details>

<details>
<summary> <strong>run_remote.sh </strong></summary>

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
tpu upd-log $current_window $LOGDIR $STAGEDIR $VM_NAME $now

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

</details>
