RED, GREEN, YELLOW, PURPLE, NC = "\033[1;31m", "\033[1;32m", "\033[1;33m", "\033[1;34m", "\033[0m"
def explain(cmd):
    print(f"{PURPLE}[INFO]{NC} Help for command: {cmd}")
    
    match cmd:
        # ========== Core Job Commands ==========
        case 'run':
            print("Run a job in the specified working directory on a TPU.")
            print("Usage: tpu run <tpu> dir=<dir_num> user=<name_or_id> [tag=<tag>] [config1=val1 ...]")

        case 'monitor':
            print("Continuously monitor the jobs of a user.")
            print("Usage: tpu monitor user=<name_or_id>")

        case 'check':
            print("Check the status of all tmux windows and jobs of a user.")
            print("Usage: tpu check user=<name_or_id>")

        # ========== Directory Management ==========
        case 'set-dir':
            print("Manually set a working directory to a specific number.")
            print("Usage: tpu set-dir <abs_path> <number> user=<name_or_id>")

        case 'set-cur':
            print("Set the current directory (current pwd) to <number> in user's directory map.")
            print("Usage: tpu set-cur <number> user=<name_or_id>")

        case 'get-dir':
            print("Get the path of the <number>th working directory of the user.")
            print("Usage: tpu get-dir <number> user=<name_or_id>")

        case 'ls' | 'list-dir':
            print("List all working directories of the user.")
            print("Usage: tpu ls user=<name_or_id>")

        # ========== Config Alias ==========
        case 'add-config-alias' | '-a' | '-alias':
            print("Add a config alias for the user (e.g., lr -> config.train.lr).")
            print("Usage: tpu -a <alias> <full_config_key> user=<name_or_id>")

        case 'show-config-alias' | '-sa':
            print("Show all config aliases of the user.")
            print("Usage: tpu -sa user=<name_or_id>")

        case 'del-config-alias':
            print("Delete a config alias from the user.")
            print("Usage: tpu del-config-alias <alias> user=<name_or_id>")

        # ========== Job Metadata ==========
        case 'add-tag':
            print("Add a tag to a job window for easier tracking.")
            print("Usage: tpu add-tag <window_id> <tag> user=<name_or_id>")

        case 'kill-window' | '-kw':
            print("Kill a specific tmux window of the user.")
            print("Usage: tpu -kw <window_id> user=<name_or_id>")

        case 'upd-log':
            print("Update the log_dir, tpu, and start_time of a running job.")
            print("Usage: tpu upd-log <window> <log_dir> <tpu> <start_time>")

        case 'finish-job':
            print("Mark a job as finished in the job tracker.")
            print("Usage: tpu finish-job <tmux_window>")

        # ========== User Settings ==========
        case 'get-settings':
            print("Get all user-level settings.")
            print("Usage: tpu get-settings user=<name_or_id>")

        case 'set-settings':
            print("Set a user-level setting.")
            print("Usage: tpu set-settings <key> <value> user=<name_or_id>")

        # ========== User Management ==========
        case 'add-user':
            print("Interactively create a new user and associated tmux session.")
            print("Usage: tpu add-user")

        case 'del-user':
            print("Delete a user and associated job data.")
            print("Usage: tpu del-user")

        case 'list-users' | '-lu':
            print("List all users currently registered.")
            print("Usage: tpu list-users")

        # ========== TPU Alias ==========
        case 'add-tpu-alias' | '-ta':
            print("Add a TPU alias (e.g., alias = real_tpu_name).")
            print("Usage: tpu -ta <alias> <tpu_name>")

        case 'list-tpu-alias' | '-lta':
            print("List all TPU aliases.")
            print("Usage: tpu -lta")

        # ========== System Maintenance ==========
        case 'clear-finished':
            print("Clear all finished jobs from the user's job list.")
            print("Usage: tpu clear-finished user=<name_or_id>")

        case 'clear-error':
            print("Clear all error/killed jobs from the user's job list.")
            print("Usage: tpu clear-error user=<name_or_id>")

        case 'clear' | 'clear-all':
            print("Clear all finished and error jobs.")
            print("Usage: tpu clear user=<name_or_id>")

        case '-czw':
            print("Clear zombie tmux windows (no corresponding job data).")
            print("Usage: tpu -czw user=<name_or_id>")

        case '-czj':
            print("Clear zombie jobs (job entries with no tmux window).")
            print("Usage: tpu -czj user=<name_or_id>")

        case 'clean':
            print("Clear all job states and zombie windows.")
            print("Usage: tpu clean user=<name_or_id>")

        # ========== TPU Environment ==========
        case 'check-status' | '-cktpu':
            print("Check the current status (READY, PREEMPTED, etc.) of the TPU.")
            print("Usage: tpu -cktpu <tpu_name>")

        case 'describe' | '-dtpu':
            print("Describe the TPU state and environment sanity.")
            print("Usage: tpu -dtpu <tpu_name>")

        case 'check-env':
            print("Check the python env (e.g. JAX visibility) of the TPU.")
            print("Usage: tpu check-env <tpu_name>")

        case 'apply':
            print("Apply a new TPU VM (non-preemptible).")
            print("Usage: tpu apply <tpu_name>")

        case 'reapply':
            print("Delete and re-apply a preemptible TPU.")
            print("Usage: tpu reapply <tpu_name>")

        case 'mount-disk':
            print("Mount NFS disk in the TPU.")
            print("Usage: tpu mount-disk <tpu_name>")

        case 'solve' | 'solve-env':
            print("Solve TPU environment issues (auto check and mount).")
            print("Usage: tpu solve <tpu_name>")

        case 'set-wandb':
            print("Setup wandb key on remote TPU.")
            print("Usage: tpu set-wandb <tpu_name>")

        case 'kill-jobs' | '-kj':
            print("Kill all jobs on a TPU across all workers.")
            print("Usage: tpu -kj <tpu_name>")

        # ========== Locks ==========
        case 'lock-code' | '-lc':
            print("Lock the entire system (no new jobs).")
            print("Usage: tpu -lc [username]")

        case 'unlock-code' | '-ulc':
            print("Unlock the system.")
            print("Usage: tpu -ulc [username]")

        # ========== Unknown ==========
        case _:
            print(f"{RED}[ERROR]{NC} Unknown command '{cmd}'. Try `tpu tldr` for summary.")

def tldr():
    Usage = f"""
{YELLOW}== Core Job Commands =={NC}
- `tpu run <tpu> dir=<n> [tag=<tag>] [key=value ...]`: Run a job in the specified directory on a TPU.
e.g. `tpu run v2-32-6 dir=1 tag=baseline lr=3e-4`
- `tpu monitor/check user=<name_or_id>`: Continuously(one-time) monitor jobs for a user.
- `tpu resume window=<window_id> [tpu=<tpu>] user=<name_or_id>`: Resume a job in a tmux window.
{YELLOW}== Directory Management =={NC}
- `tpu ls user=<name_or_id>` or `tpu list-dir user=<name_or_id>`: List all working directories.
- `tpu set-cur <n> user=<name_or_id>`: Set current working directory (pwd) to slot `<n>`.
{YELLOW}== Config Aliases =={NC}
- `tpu -a <alias> <full_config_key> user=<name_or_id>`: Add a config alias. 
e.g. `tpu -a lr config.training.learning_rate`
- `tpu -sa user=<name_or_id>`: Show all config aliases.
{YELLOW}== Job Controls =={NC}
- `tpu add-tag <window_id> <tag> user=<name_or_id>`: Add a tag to a job.
{YELLOW}== TPU Management =={NC}
- `tpu -dtpu <tpu>`: Describe TPU with environment info.
- `tpu solve <tpu>`: Fix common environment errors automatically.
{YELLOW}== Cleanup Commands =={NC}
- `tpu clear-all/clear user=<name_or_id>`: Clear finished + error jobs.
{YELLOW}== Info & Help =={NC}
- `tpu help/-h <command>`: Show usage for a specific command.
- `tpu tldr`: Show this summary.
    """
    print(Usage)
