from .helpers import *

def explain(cmd):
    print(f"{INFO} Help for command: {cmd}")
    
    match cmd:
        # ========== Core Job Commands ==========
        case 'run':
            print("Run a job in the specified working directory on a TPU.")
            print("Usage: tpu run <tpu> dir=<dir_num> <username> [tag=<tag>] [config1=val1 ...]")

        case 'monitor':
            print("Continuously monitor the jobs of a user.")
            print("Usage: tpu monitor <username> [-flag]")

        case 'check':
            print("Check the status of all tmux windows and jobs of a user.")
            print("Usage: tpu check <username> [-flag]")

        # ========== Directory Management ==========
        case 'set-dir':
            print("Manually set a working directory to a specific number.")
            print("Usage: tpu set-dir <abs_path> <number> <username>")

        case 'set-cur':
            print("Set the current directory (current pwd) to <number> in user's directory map.")
            print("Usage: tpu set-cur <number> <username>")

        case 'get-dir':
            print("Get the path of the <number>th working directory of the user.")
            print("Usage: tpu get-dir <number> <username>")

        case 'ls' | 'list-dir':
            print("List all working directories of the user.")
            print("Usage: tpu ls <username>")

        # ========== Config Alias ==========
        case 'add-config-alias' | '-a' | '-alias':
            print("Add a config alias for the user (e.g., lr -> config.train.lr).")
            print("Usage: tpu -a <alias> <full_config_key> <username>")

        case 'show-config-alias' | '-sa':
            print("Show all config aliases of the user.")
            print("Usage: tpu -sa <username>")

        case 'del-config-alias':
            print("Delete a config alias from the user.")
            print("Usage: tpu del-config-alias <alias> <username>")

        # ========== Job Metadata ==========
        case 'add-tag':
            print("Add a tag to a job window for easier tracking.")
            print("Usage: tpu add-tag <window_id> <tag> <username>")

        case 'kill-window' | '-kw':
            print("Kill a specific tmux window of the user.")
            print("Usage: tpu -kw <window_id> <username>")

        case 'upd-log':
            print("Update the log_dir, tpu, and start_time of a running job.")
            print("Usage: tpu upd-log <window> <log_dir> <tpu> <start_time>")

        case 'finish-job':
            print("Mark a job as finished in the job tracker.")
            print("Usage: tpu finish-job <tmux_window>")

        # ========== User Settings ==========
        case 'get-settings':
            print("Get all user-level settings.")
            print("Usage: tpu get-settings <username>")

        case 'set-settings':
            print("Set a user-level setting.")
            print("Usage: tpu set-settings <key> <value> <username>")

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
            print("Usage: tpu clear-finished <username>")

        case 'clear-error':
            print("Clear all error/killed jobs from the user's job list.")
            print("Usage: tpu clear-error <username>")

        case 'clear' | 'clear-all':
            print("Clear all finished and error jobs.")
            print("Usage: tpu clear <username>")

        case '-czw':
            print("Clear zombie tmux windows (no corresponding job data).")
            print("Usage: tpu -czw <username>")

        case '-czj':
            print("Clear zombie jobs (job entries with no tmux window).")
            print("Usage: tpu -czj <username>")

        case 'clean':
            print("Clear all job states and zombie windows.")
            print("Usage: tpu clean <username>")

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
            print(f"{FAIL} Unknown command '{cmd}'. Try `tpu tldr` for summary.")

def tldr():
    Usage = f"""
{YELLOW}== Core Commands =={NC}
- `tpu run <tpu> <username> [dir=1] [tag=<tag>] [key=value ...] [rule=<rule>]`: Run a job in the specified directory on a TPU.
e.g. `tpu run v2-32-6 dir=2 tag=baseline lr=3e-4 rule=resume xibo`
- `tpu monitor/check <username>`: Continuously(one-time) monitor jobs for a user.
{YELLOW}== Directory Management =={NC}
- `tpu ls <username>`: List all working directories.
- `tpu set-cur <n> <username>`: Set current working directory (pwd) to slot `<n>`.
{YELLOW}== Config Aliases =={NC}
- `tpu -a <alias> <full_config_key> <username>`: Add a config alias. 
e.g. `tpu -a lr config.training.learning_rate`
- `tpu -sa <username>`: Show all config aliases.
{YELLOW}== TPU Management =={NC}
- `tpu describe <tpu>`: Describe TPU with environment info.
- `tpu solve <tpu>`: Fix common environment errors automatically.
- `tpu -lta`: List all TPU aliases.
{YELLOW}== Others =={NC}
- `tpu clean <username>`: Clear finished + error jobs.
- `tpu help/-h <command>`: Show usage for a specific command.
- `tpu tldr`: Show this summary.
    """
    print(Usage)

def full_doc():
    raise NotImplementedError
    Usage = f"""
{YELLOW}== Start=={NC}
- Add yourself: `tpu add-user` → follow prompt.
- From your project directory: `tpu set-cur 1 <username>` then `tpu run <tpu‑alias> <username>`.
- Track jobs: `tpu monitor <username>` or one‑shot with `tpu check <username>`.
- See command cheatsheet: `tpu tldr`; clean up: `tpu clear <username>`.

{YELLOW}== Setup=={NC}
- **Update your training scripts** to accept CLI args & log *wandb* final output.
- Clone latest helper scripts from *zhh* repo.
- Avoid weird usernames like `run`, `false`, Chinese chars, or TPU‑like strings.

{YELLOW}== Working Directories & Running Jobs=={NC}
- A numbered *working directory* points to a source‑code folder.
- `tpu set-cur <n> <username>` — mark *pwd* as dir *n* (default 1).
- `tpu ls <username>` — list all saved directories.
- `tpu run <tpu> <username> [dir=1] [tag=<tag>] [key=value …] [rule=<rule>]`
  · `tpu` can be an alias (e.g. `v2-32-6`) or the full TPU name.
  · Pre‑emptible TPUs auto‑resume/on‑GRPC; non‑preemptible default *pass* — override with `rule=`.
  · `-apply` flag skips confirmation when (re)applying.

Examples:
```bash
tpu run v2-32-6 xibo                 # run in dir 1
tpu run v2-32-p1 lyy dir=2 tag=bird  # run in dir 2 with tag
```

{YELLOW}== Killing / Cleaning=={NC}
- Bulk clean finished/errored:
  `tpu clean <username>` (alias for kill‑job → clear)
- Kill specific job:
  `tpu kill-job -w=<window_id> <username>`
  `tpu kill <tpu> <username>`
- Zombie helpers (rarely needed):
  `tpu -czw/-czj`, `tpu clear‑finished`, `tpu clear‑error`, `tpu clear-all`.

{YELLOW}== Monitoring=={NC}
- Auto monitor window opens after `tpu run` (toggle via `set-settings monitor_after_run False`).
- Manual: `tpu monitor <username>` (updates every *monitor_upd_time* sec) or `tpu check <username>`.
- Display filters: `-d` dir, `-t` tpu, `-s` status, `-v` verbose output.
  Example: `tpu monitor xibo -dsv`.

{YELLOW}== TPU / Environment Ops=={NC}
- Provision: `tpu apply/reapply <tpu>`.
- Inspect: `tpu describe <tpu>`  |  Status: `tpu check-status <tpu>`.
- Fix common mount/wandb issues: `tpu solve <tpu>` (community‑extendable).

{YELLOW}== Config Aliases & Tags=={NC}
- Add alias: `tpu -a <alias> <full_config_key> <username>`.
- List aliases: `tpu -sa <username>`  |  Delete: `tpu del-config-alias <alias> <username>`.
- Built‑ins: lr, bs, ep, wd, b1, b2, ckpt.
- Pass configs: `tpu run v2-32-6 xibo lr=0.01` (or full key).
- Tag running job in monitor: `tpu add-tag <window_id> <tag> <username>`.

{YELLOW}== Manual Resume / Rerun=={NC}
```bash
tpu resume windows=<id> <username>
tpu resume windows=<id> tpu=<new_tpu> <username>
tpu rerun  windows=<id> <username>
tpu rerun  windows=<id> tpu=<new_tpu> <username>
```
`resume` loads last ckpt, `rerun` starts fresh.

{YELLOW}== Auto‑Resume / Rerun Rules=={NC}
Default rules:
- **Preemptible TPUs** → `pre`: reapply on GRPC, *resume* on preemption.
- **Non‑preemptible** → `pass`: no action.
Available `rule=` values:
- `pre`, `resume`, `reapply`, `rerun`, `pass`.
See all: `tpu check-rules`.
- Immediate manual trigger for stuck job: `tpu ack` (MONITOR reacts ≤3 min).
- `--log-stage` injects `config.stage` integer into script.

{YELLOW}== User Settings=={NC}
Manage via `tpu set-settings/get-settings/reset-settings <username>`.
Window‑offset reset: `tpu reset-window-num <n> <username>` (avoid clashes).

{YELLOW}== Built‑in Help=={NC}
- `tpu tldr` — ultra‑short cheatsheet.
- `tpu -h <command>` — detailed help for any command.
"""
    print(Usage)
