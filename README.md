# Auto TPU job manager for Google Cloud Platform (experimental)

This is an automatic job manager for running TPU jobs. It supports auto-resuming the preempted/grpc TPUs, and monitoring the jobs status.

## Quickstart in 2 mins

Here is a quick guide of the common usage, and you can find more details in the full docs below.

Tldr usage in **two sentences**: Use ``tpu add-user`` to add your username, then go to your working directory(where you have your scripts and code) and use ``tpu set-cur 1 username`` to set the working directory. Use ``tpu run <tpu> username``(e.g. ``tpu run v2-32-p2 xibo``) to run the job, and use ``tpu monitor/check username`` to see the status of all your jobs. (The ``tpu run`` command will auto-resume the job when preempted/grpc for preempted TPUs, you don't have to set it.)

More usage in **two sentences**: Use ``tpu tldr`` to see useful commands, and ``tpu clear username`` to clear the finished/crashed jobs; use ``tpu -a alias_name full_name username``(e.g. ``tpu -a lr config.training.learning_rate``) to add a new alias, then you can pass the configs such as ``tpu run v2-32-6 xibo lr=0.01``. Use ``tpu describe <tpu>`` to check the environment of the TPU, and ``tpu solve <tpu>`` to solve the environment automatically.

**REMEMBER TO UPDATE YOUR SCRIPTS!**

## Full docs

<details>
<summary> <strong>1. Setup(<strong>IMPORTANT</strong>) </strong></summary>

You should **update your scripts** to the newest version supporting command-line arguments. The newest scripts can be pulled from zhh's repo. The current finishing check is based on wandb final output, so please make sure your scripts are using wandb to log the final output.  
Also, this script is not very robust to attack, so please try **not to do OOD things**, for example, setting username to be `run`, `false`, `v2-32-2` or Chinese characters.

Use ``tpu add-user`` and follow the instructions to add your username.

</details>

<details>
<summary> <strong>2. Setting Working Directory & Running/Monitoring Jobs (<strong>IMPORTANT</strong>) </strong></summary>

The working directory is where you have your scripts and code. 

You can set multiple working directories and choose any of them when running code. The default working directory is `1`.  
You can set the working directory and run the job by:

```bash
tpu set-cur num username # Set the working directory <num> to the current directory, default directory is 1
tpu ls username # List all the working directories
tpu run tpu_name username [dir=1] # Run the job in working directory <dir>
```

The `tpu_name` is of the format of the **pre-defined tpu aliases** , like `v2-32-6`, `v2-32-p1`, or `v4-32-py2`. You can also pass full-name such as `kmh-tpuvm-v2-32-1`.

For all the aliases, use `tpu -lta` (list TPU aliases) to see. You can also add aliases by `tpu -ta alias FULL_TPU_NAME`. Please don't add aliases that may lead to contradictions to other things, for example `username` or `tag` or `config` or `s`.

**Example:**

```bash
tpu run v2-32-6 xibo # Default: run the job in working directory 1
tpu run v2-32-p1 lyy dir=2 # Run the job in working directory 2 
```

The `tpu run` command opens a monitor window to track all your jobs. Alternatively, you can use:

```bash
tpu monitor username
```

which updates the monitor window every 10 seconds. For one-time checks, use:

```bash
tpu check username
```

<details>
    <summary> <strong>2A. More Directory Operations (OPTIONAL)</strong></summary>

    ```bash
    tpu del-dir <num> username # Delete the working directory <num>
    tpu swap-dir <num1> <num2> username # Swap the working directory <num1> and <num2>
    ```
</details>

<details>
<summary> <strong>2B. Advanced Running Settings (OPTIONAL)</strong></summary>

The `run` command will ask whether to reapply when the TPU is preempted. 

You can add the flag `-apply` to skip the prompt.

You can add the flag `-q` to skip the monitor window.

You can add the tag by `tag=your_tag` to add a tag to the job, which will be shown in the monitor window.
You can add tags to **existing** jobs by:`tpu add-tag window_num tag_name username`

You can change the default rules for resuming/rerunning by passing `rule=<rule>` to the `tpu run` command. (Default: **auto-resume** on GRPC errors and **auto-reapply and resume** when preempted for preemptible TPUs and **do nothing** for other TPUs (you can set `rule=resume` to make it resume). See more in the **More on Resuming/Rerunning** section.)

</details>

<!-- BEGIN OF 2C -->
<details>
<summary> <strong>2C. Advanced Monitor Configs(OPTIONAL)</strong></summary>

The monitor will show four things: the windows number(`w`), the directory(`d`), the tpu(`t`), and the job status(`s`). You can choose which to show by adding commands. There's also an additional flag "verbose"(`v`) available, meaning to show the messages(cut) from tmux windows even for the running jobs with known status.(Should be used with `s`) For example, to only show the working directory and the job status and detailed output of xibo, use:

```bash
tpu monitor xibo -dsv
```

If you don't want `tpu run` to open the monitor window, you can use `tpu set-settings monitor_after_run False username` to disable it. Also, you can set the default monitoring whether to monitor tpu/directory. See the **Customizing User Settings** section for more details.

</details> 
<!-- END OF 2C -->

<!-- BEGIN OF 2D -->
<details>
<summary> <strong>2D. Spreadsheet Support(OPTIONAL)</strong></summary>

The `tpu run` command will automatically set the status in the spreadsheet to be running by you. If you want to set the notes, you can add a `-ssn` flag(short for `--set-spreadsheet-notes`) to set the notes interactively, or you can pass `ssn=your_notes` to set the notes directly.(In this case, space will not be supported, and you need to use `_` instead of space.)

You can use `tpu find <all_tpu_types>` to look at the status of the TPUs in the spreadsheet. The format of tpu_types is like `v2`, `v3`, `v234`(or `v*`) or `v2-32`. You can also pass `-n` for normal TPUs and `-p` for preemptible TPUs. For example, to show the status of all non-preemptible v3 and v4 TPUs, you can do:
`tpu find v34 -n`.

You can release the TPU by `tpu release/rel <tpu_name>`, which set the status and the user to be free('闲的') in the spreadsheet. You can also use `tpu release/rel <tpu_name> <username>` to make sure that the TPU is currently owned by you(recommended).

</details>
<!-- END OF 2D -->
</details>

<details>
<summary> <strong>3. Killing Jobs/Windows & Cleaning up (<strong>USEFUL</strong>)</strong></summary>

As you run more and more jobs, there will be a lot of tmux windows, which is messy.

You can use (**recommend to do occasionally**):

```bash
tpu clean username 
```

to kill all the tmux windows whose jobs are finished/error/killed.

To kill a job, use:

```bash
tpu kill/kill-job/-k/-kj w=/-w=/window=/<windows_id> username
```
You can also just enter windows_id, in this case the command will find the integer in the 
arguments to be the windows id. For example you can just use `tpu kill 101 xibo` to kill the job with windows id 101, but passing `w=` is safer for future use.

Jobs with children jobs that were rerun/resumed will be killed based on the status of their children. Use `tpu clean username -re` to make all the rerun/resumed job be cleaned too.

**IMPORTANT**: If you have a job that has rerun setting, and it has been grpc, please remember to use `clean` to clear it if you **manually kill the window**, otherwise it may be rerunned.
<details>
<summary> <strong>3A. Other killing commands(OPTIONAL)</strong></summary>
To kill a specific tmux window (NOT RECOMMENDED):

```bash
tpu -kw/kill-window window_number username
```

After killing windows, some jobs may become "zombies" (i.e., jobs without associated windows). You can use these helpers to clean zombies (Supported, but NOT RECOMMENDED):

```bash
tpu -czw username # Clear all zombie windows
tpu -czj username # Clear all zombie jobs
tpu clear-finished username # Clear all finished jobs
tpu clear-error username # Clear all error jobs
tpu clear-all username # RECOMMENDED: Clear all finished/error jobs
```

The `clean` command integrates these actions, so using `kill-job + clean` is strongly recommended instead of manually killing windows with `tmux kill-window` or exit the window yourself. (If you like to kill the window yourself, we recommend doing `tpu clean username` occasionally to clear the job data associated with these windows, or others may get the annoying warning of the TPU occupied by your dead jobs.)
</details>

</details>

<details>
<summary> <strong>4. Environment Operations </strong></summary>

We support common operations, such as:

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
<summary> <strong>5. Passing Configs in Command Line</strong></summary>

We support passing configs in the command line by config aliases or full config name. You can also set your own config alias by:


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


</details>

<details>
<summary> <strong>6. More on Resuming/Rerunning </strong></summary>
You can manually resume/rerun a job by:

```bash
tpu resume windows=<windows_id> username # resume the job
tpu resume windows=<windows_id> tpu=<tpu> username # resume the job in a new TPU
tpu rerun windows=<windows_id> username # rerun the job
tpu rerun windows=<windows_id> tpu=<tpu> username # rerun the job in a new TPU
```

The difference between `resume` and `rerun` is that `resume` will load the job from the last checkpoint, while `rerun` will start a new job from the beginning.

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

If you want to know whether the job is a resumed job in the program(for example, use that to set a new wandb name/note), you can add `--log-stage` flag in `tpu run`, then it will pass an additional argument `config.stage` to indicate the number of resumes of this job. (For example, if the job has been resumed twice, that is, there're 3 runs in total including the current one, the current one will recieve an extra `config.stage=2` config).

We have a MONITOR to occasionally keep tract of all the job status and decide whether to resume/rerun. The default checking frequency for the jobs to be rerun is about 30 mins, that is, the jobs will wait at most 30 mins to be resumed. If you run a job that leads to a GRPC immediately, you can acknowledge the MONITOR to rerun that immediately by:

```bash
tpu ack
```

Then after no more than 3 mins you should expect the job to be resumed(if not, contact the admin).

</details>

<details>
<summary> <strong>7. Customizing User Settings </strong></summary>

We support customizing settings for users, and you can set/get them by:

```bash
tpu set-settings key value username # set the settings
tpu get-settings username # get the settings
tpu reset-settings username # reset all the settings
```

The current default settings and their meanings are:

```bash
{
    "monitor_after_run": True, # Whether to monitor the job after running
    "monitor_upd_time": 5, # The update time for the monitor window
    "monitor_length": 800, # The output capturing length for the monitor window to determine the job status
    "monitor_dir": True, # Whether to show the working directory in the monitor window
    "monitor_tpu": True, # Whether to show the TPU name in the monitor window
    "monitor_verbose": False, # Whether to show the output in the monitor window when the status is known
    "show_length": 200, # The output capturing length for the monitor window to show the job output
    "time_zone": "us", # The user timezone, only support 'us'(UTC-4)/'cn'(UTC+8) for now.
    "extra_settings": {} # The extra settings for future development
}
```

Also, to avoid concurrency issues of tmux windows creation, we use a `windows_offset` to offset the windows number for each user, and the number goes up by 1 for each new job. If you think the offset is too large, you can set it to a smaller number by:

```bash
tpu reset-window-num <num> <username>  # reset the offset to <num>
```

Please be careful not to have conflicts with current jobs.


</details>


<details>
<summary> <strong>8. Documentation</strong></summary>

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
- `sheet.py` does the spreadsheet operations
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
                "monitor_dir": true,
                "monitor_tpu": true,
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
    "conda_env_name": "NNX",
    "monitor_all_check_time": 20,
    "MONITOR_logs": [],
    "ack_MONITOR": false
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
- [ ] Support restarting TPU
- [ ] Customized monitor window
- [ ] Auto-choose the TPU to run a job  
- [ ] More auto env solvers  
- [ ] Logging for every user so that you can check the things happen since last time  

<!-- ## New Scripts
<details>

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

pane_id=$TMUX_PANE
current_window=$(tmux display-message -p -t "$pane_id" '#S:#I')
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
    --worker=all --ssh-flag="-n" --command "${cmd}" 2>&1 | tee -a $LOGDIR/output.log

if grep -q "wandb: Run history:" $LOGDIR/output.log; then
    echo "Job completed successfully"
    tpu finish-job $current_window
else
    echo "Job failed"
fi
```

</details> -->
