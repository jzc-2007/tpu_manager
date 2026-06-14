# AGENTS.md — xibo_tpu_manager (Quick on-boarding for agents)

This repo is a multi-user **Google Cloud TPU job manager**: it shells out
`gcloud compute tpus tpu-vm ...` to apply / reapply / SSH into TPU VMs,
runs each user's training job inside a per-user `tmux` session, and keeps
a global JSON state (`data.json`) describing every job. A background
**MONITOR** daemon polls the state and auto-resumes / reapplies jobs.

This file is meant for *other AI / human agents* who need to navigate
the codebase quickly. The full user-facing docs live in `README.md`.

---

## 1. Top-level layout

```
xibo_tpu_manager/
├── tpu.py                  # CLI entrypoint -> dispatches to utils/*
├── MONITOR.py              # Long-running daemon: detects + reacts to errors
├── web_cli.py              # Web UI wrapping the same `utils/*` API
├── utils/                  # All real logic lives here (see §3)
├── data.json               # PRIMARY STATE: users, jobs, tpu aliases, MONITOR config
├── legacy.json             # Archived jobs / overflow from data.json
├── queue.json              # Pending jobs waiting for a free TPU
├── lock.json               # fcntl-style locks: code / data / queue / legacy / apply
├── mounted.json            # Cache of which TPUs have their disk mounted
├── apply.json              # Cache used during TPU apply flow
├── secret.json             # Wandb / GCP secrets (do not commit)
├── passwords.json          # web_cli auth
├── README.md               # User-facing docs (the long version)
├── NOTES.md                # Hard-won gotchas, read this once
├── ab.ab / 新.sh / monitor.sh / batch_run.sh / apply_tpu.sh / create_tmux_queue.py
│                           # Misc shell helpers
└── data.json.bak.*         # Manual backups; safe to ignore
```

Other agent breadcrumbs:

- `.git/` — yes this is a git repo.
- `__pycache__/`, `utils/__pycache__/` — generated.
- `utils/dependency.md` — author's view of the import DAG between
  `utils/*.py`. **Respect it when editing**, breaking it causes circular
  imports.
- Several `*.bak_*.py` files in `utils/` are point-in-time snapshots;
  do not edit them, do not import them. Treat as read-only history.

---

## 2. CLI entrypoint (`tpu.py`)

`tpu.py` is one giant `if/elif cmd == "..."` table. There is no Click /
argparse. Conventions:

- `args = sys.argv`, so `args[1]` is the command and `args[2:]` are
  positional flags. Many subcommands accept `username` as one of the
  positional args and find it via `find_user(data, args[1:])`.
- Many user-targeted subcommands first resolve the `user` object, then
  call into `utils.jobs` / `utils.directories` / etc. with `user_obj`.
- The lock check at the top (`data_io.check_code_lock()`) refuses every
  command except `lock` / `unlock` if `lock.json -> code.status` is true.
  This is the "dev freeze" switch used by `develop.py`.

Roughly grouped command families (already grouped in `tpu.py` with
section comments — search for `# ------------`):

- **Users**: `add-user`, `del-user`, `list-users` / `-lu`, `init`,
  `register-web`.
- **Directories**: `set-cur`, `set-dir`, `del-dir`, `swap-dir`, `ls`,
  `get-dir`.
- **Settings**: `get-settings`, `set-settings`, `reset-settings`.
- **Config aliases**: `-a/-alias`, `-sa/-la`, `del-config-alias`.
- **Run / resume / rerun**: `run`, `resume`, `rerun`, `restart-run`,
  `queue`, `dequeue`, `dqr`.
- **Monitor / status** (no side effects on TPU but may write `data.json`):
  `check`, `check-simp`, `monitor`, `caj`, `maj`.
- **Kill / clean**: `kill / kill-job / -k / -kj`, `kill-window / -kw`,
  `clean`, `clear-finished`, `clear-error`, `clear-all / clear`,
  `-czw`, `-czj`, `ignore-error`.
- **TPU lifecycle**: `apply`, `applyy`, `reapply`, `reapplyy`,
  `apply-norm`, `reapply-norm`, `delete`, `restart`, `register`,
  `del-registered/del-reg`, `check-status`, `describe`, `test`,
  `mount-disk`, `mount-disk-new`, `set-wandb`, `kill-remote`,
  `change-ip`, `solve`, `lian`, `find`, `release/rel`, `keng`, `zhan`,
  `fang`, `fmd`.
- **Spreadsheet**: `ssn`, `asn`, `gtis`, `uss/upd-status-spreadsheet`,
  `twsi`.
- **Locks / dev**: `lock`, `unlock/rl`, `lock-data`, `unlock-data`,
  `rm-lock`, `-Ml/-Mc`, `debug-stats`, `debug-kill`,
  `add_global_config/-agc`, `merge_global_config/-mgc`.
- **Job ack to scripts**: `upd-log`, `upd-staging-info`, `finish-job`,
  `fail-job`, `ack`.
- **Misc**: `cp` / `cc` / `cca` (GCS bucket copies), `tldr`, `help/-h`,
  `vq`.

Aliases for the same command (e.g. `kill / kill-job / -k / -kj`,
`-a / -alias / add-config-alias`) live in the same `elif` row.

---

## 3. `utils/` package — what's in each file

Listed in dependency order (see `utils/dependency.md`). When editing,
keep the layering: lower-level files must not import higher-level ones.

| File | Purpose |
|---|---|
| `constants.py` | Paths (`DATA_PATH`, `LOCK_PATH`, ...), TPU type tables, `ZONE_DICT`, `RULE_DICT`, color codes (`RED`, `GOOD`, etc.), `PROJECT`, `REGION_SA_MAP`. |
| `clean.py` | `clean_us` / `clean_eu` — sweep stale TPUs in a region. |
| `autenticate.py` | Password-based auth for `add-user` / `del-user` / `register-web`. |
| `helpers.py` | Tiny utilities re-exported via `from utils.helpers import *`: time helpers (`get_abs_time_str`, `convert_utcstr_to_edtstr`, ...), the color constants, `get_zone_pre_spot`, `is_integer`, `read_data` re-export, ... |
| `data_io.py` | The **only** file that touches `data.json` / `lock.json` directly. `read_data`, `read_and_lock_data`, `write_and_unlock_data`, `release_lock_data`, plus the non-blocking `read_data_if_unlocked` used by `lck`/`ack` to avoid blocking. Locking is fcntl + a status field in `lock.json`. |
| `descriptions.py` | `tldr()` and `explain(cmd)` — help text. |
| `gs_buckets.py` | `cp` / `cc` / `cca` — `gsutil`-based checkpoint copy across regions. |
| `develop.py` | Dev-only tools: lock the codebase, edit metadata safely, dump MONITOR logs. |
| `directories.py` | Per-user working-dir bookkeeping (`set-cur`, `set-dir`, `del-dir`, `swap-dir`, `ls`, `get_dir`, `get_job_stage_dir`). |
| `users.py` | `User` dataclass (`user_from_dict`), `create_user`, `del_user`, `reset_settings`, `reset_window_num`, `list_users`. |
| `sheet.py` | Google Sheets I/O for the TPU spreadsheet (`get_tpu_info_sheet`, `write_sheet_info`, `find_tpu_from_type`, `release_tpu`, `keng_tpu`, spreadsheet notes). |
| `logger.py` | Aliases (TPU and config), `register_tpu*`, `add_tpu_alias`, monitor config, file locks, per-user `logs[]` append. |
| `operate.py` | All `gcloud` shellouts: `apply`, `reapply`, `apply_and_set_env`, `delete_tpu`, `restart`, `check_tpu_status`, `mount_disk`, `set_wandb`, `kill_jobs_tpu`, `describe_tpu`, `lian_tpu`, `check_env`, `test_remote`. **This is where the TPU is actually touched.** |
| `jobs.py` | The thickest file (~2k lines). Defines `Job`, runs the job lifecycle, parses tmux pane output to derive status, writes `error`/`status` back to `data.json`. See §4. |
| `error_handler.py` | `solve_env`, `change_ip`, `clear_zombie_windows`, `initialization`. Higher-level recovery flows. |
| `unit_tests.py` | Sanity checks the daemon runs occasionally (`MONITOR_config.test_freq`). |
| `queue.py` | The job queue (`queue.json`): `Queue.add`, `dequeue`, `dequeue_and_run`, `visualize_queue`, `finish_job`, `fail_job`, `upd_staging_info`. |

---

## 4. `utils/jobs.py` — the core lifecycle

This is what you'll edit most. Important entry points:

- `Job` class + `to_dict()` — the schema written into `data.json`.
- `run(user_obj, args)` — primary `tpu run` flow. Resolves a TPU, builds
  a `Job`, calls `run_job_on_tpu`, optionally opens the monitor window.
- `run_job_on_tpu(job, tpu)` — performs the scoped remote-user kill/mount,
  opens a new tmux window, `cd`s into the stage_dir, runs `staging.sh`.
- `resume(user_obj, args)` / `rerun(user_obj, args)` —
  manual `tpu resume / rerun`. Both call `resume_rerun_job`.
- `resume_rerun_job(job, new_tpu=None, load_ckpt=True)` — the *only*
  function that creates a child job (with `stage = parent_stage + 1`
  and `extra_msgs.father`). It performs scoped remote-user cleanup before
  relaunching.
- `kill_job_or_tpu(user_obj, args)` — `tpu kill <window|tpu>`.
- `kill_window(user_obj, args)` — `tpu kill-window`.
- `check_jobs` / `check_jobs_simp` / `monitor_jobs` — read-side. They
  parse `tmux capture-pane`, classify status, and, **as a side effect**,
  call `write_error_to_job(...)` to persist an error type, and sometimes
  `ack_MONITOR()` to wake the daemon. See §5.
- `_render_rows_for_job(...)` — the same status-detection logic as
  `check_jobs`, but in a row-builder form used by `check_jobs_simp`.
- `clear_finished_jobs`, `clear_error_jobs`, `clear_all_jobs`,
  `clear_zombie_jobs` — list mutators on `user.job_data`. Used by
  `clean`.
- `write_error_to_job(user, job, error)` — idempotent
  `status = 'error', error = '<type>'` writer, opportunistic
  (skips if data lock is held).
- `ack_MONITOR()` — flips `data["ack_MONITOR"] = True` so the MONITOR
  daemon wakes up within 10s. Opportunistic.

Durable pretrained checkpoint convention:
- `ltgcp <absolute_local_logdir>` maps the normal regional run path under
  `qiao_zhicheng_hanhong_files/` to the same bucket's
  `pretrained-ckpts/qiao_zhicheng_hanhong_files/` prefix.
- `utils.gs_buckets.ensure_pretrained_checkpoint_for_launch(...)` preflights
  finetune launches with `load_from_pretrained`: normal target-zone checkpoint
  first, same-zone pretrained fallback second, then copy latest checkpoint from
  another zone's pretrained prefix into the target pretrained prefix if needed.
- `utils.queue.finish_job(...)` mirrors the finished job's latest checkpoint
  into same-zone `pretrained-ckpts/` as a best-effort durable backup.

Status values used on jobs:

```
None | 'starting' | 'running' | 'finished'
'error'   (with non-empty `error` field describing the type)
'killed'  (manual)
'resumed' | 'rerunned'  (this job has a child; child id in extra_msgs.child)
```

`error` field values seen: `'preempted'`, `'grpc'`, `'locked'`, `'OOM'`,
`'file error'`, `'invalid pointer'`, `'deadline exceeded'`, `'unknown'`.

### Remote Linux User Mode

This sandbox copy is being adapted so xibo logical users still live in
`data.json` exactly as before, but TPU VM jobs run under separate Linux
accounts instead of always running as remote `sqa`.

- New jobs default `remote_linux_user` to the xibo user name. For example,
  xibo user `cyx` launches remote training as `cyx@<tpu>`.
- Existing/legacy jobs without `extra_msgs.remote_linux_user` must be treated
  as remote `sqa`. This preserves compatibility for currently running jobs and
  old resume chains.
- `run`, `run_job_on_tpu`, `resume`, and `rerun` store the resolved Linux user
  in `job.extra_msgs.remote_linux_user`.
- `resume`/`rerun` inherit the parent job's remote Linux user unless explicitly
  overridden with `remote_user=<linux_user>` (aliases:
  `linux_user=`, `remote_linux_user=`, `linux-user=`, `remote-linux-user=`).
- Launch commands prepend `export TPU_REMOTE_USER=<linux_user>;` before
  `source staging.sh ...`. The repo `staging.sh` / `run_remote.sh` side must
  honor `TPU_REMOTE_USER` by SSHing to `<linux_user>@<tpu>`.
- `mount_disk(..., remote_user=...)` is remote-user aware. NFS mounting is
  global to the VM, but Python packages, W&B login, and `.disk_mounted` markers
  are per Linux user under `/home/<linux_user>`.
- `mount_disk` must not remove other users' home directories. Separate Linux
  users are the isolation boundary.
- `ensure_remote_linux_user(...)` creates the Linux account on all TPU workers,
  adds passwordless sudo, copies available SSH authorized keys, and verifies
  `whoami` plus `sudo -n true`.
- `kill_jobs_tpu` now supports scoped kills. Default xibo-user kills only target
  the caller's inferred remote Linux user(s). `--all-users` / `--all-remote-users`
  is the explicit all-user kill mode and is the only mode that cleans global
  `/tmp` TPU state.
- `tpu kill-remote <tpu>` remains a legacy all-user kill. Use
  `tpu kill-remote <tpu> remote_user=<linux_user>` for a scoped kill.
- Do not kill PPIDs gathered from `ps`: many training worker processes have
  PPID 1. Only kill matching process PIDs and matching TPU device-holder PIDs.

---

## 5. The MONITOR daemon (`MONITOR.py`)

Long-running process, intended to run all day under the
`monitor.sh` wrapper or a tmux window.

Loop: every `data["MONITOR_config"]["checking_freq"]` seconds
(default ~600s), or immediately whenever `data["ack_MONITOR"] == True`,
it runs `mainloop()`:

1. Walk every user's `job_data`. Skip jobs whose status is
   `finished/rerunned/resumed/killed` or whose `monitor == False`.
2. For each remaining job, decide an `error_type`:
   - If `status == 'error'` and `error != 'unknown'`, trust the
     existing `error` field.
   - Otherwise call `check_job_status(job)` which:
     - asks `operate.check_tpu_status(tpu)` → returns `'preempted'`
       if the TPU is preempted;
     - else greps `<log_dir>/output.log` for `GRPC error`
       (→ `'grpc'`) or `Could not open any log file` (→ `'locked'`).
3. Bucket jobs into `error_jobs = {'preempted': [], 'grpc': [], 'locked': []}`.
   **Errors not in this set (`OOM`, `file error`, `invalid pointer`,
   `deadline exceeded`, `unknown`) are silently dropped by the daemon
   and never auto-handled.**
4. Mark every bucketed job as `status='error', error=<type>` in
   `data.json`.
5. Per job, look up `job['rules'][error_type]` and dispatch:
   - `pass`    — do nothing.
   - `reapply` — `reapply_resume(job)` → spawns `reapply_worker` in a
     subprocess (timeout 1800s) that runs `operate.apply_and_set_env(...,
     delete=True)`, then on success calls
     `jobs.resume_rerun_job(job, load_ckpt=True)`.
   - `resume`  — `kill_resume(job)` does a scoped remote-user kill, then
     `jobs.resume_rerun_job(job, load_ckpt=True)`.
   - `rerun`   — `kill_rerun(job)` does a scoped remote-user kill, then
     `jobs.resume_rerun_job(job, load_ckpt=False)`.
   - `restart` — `restart_rerun(job)` → `operate.restart(tpu)` then rerun.

`RULE_DICT` in `utils/constants.py` shows the rule-name → (per-error
behavior) mapping. The default rule attached to a job is:
- preemptible TPU → `pre` (`preempted: reapply`, `grpc: resume`)
- non-preemptible → `pass` (do nothing)

The daemon also logs to `data["MONITOR_logs"]` (visible via `tpu -Ml`).

---

## 6. `data.json` schema (cheat sheet)

Top-level keys:

```
users: { <name>: <user_dict> }
user_list: [<name>, ...]
id_list / id_user_dict / user_id_dict
tpu_aliases: { "<short>": "<full vm name>" }
all_tpus: { <zone or 'preemptible'>: [<tpu names>] }
monitor_config / MONITOR_config: { checking_freq, test_freq, clean_freq, ... }
ack_MONITOR: bool         # set true by ack_MONITOR() to wake the daemon
MONITOR_logs: [{time, msg}]
wandb_api_key, conda_env_name, monitor_all_check_time
```

`user_dict`:

```
id, name, tmux_name, working_dir: { "1": "/path", ... },
job_data: [Job, ...],
config_aliases: { "lr": "config.training.learning_rate", ... },
settings: { monitor_after_run, monitor_upd_time, monitor_length,
            show_length, monitor_dir, monitor_tpu, monitor_verbose,
            time_zone, extra_settings, ... },
windows_offset: int,      # next free tmux window number
logs: [...]
```

`Job` (see also `utils/jobs.py::Job.to_dict`):

```
user, windows_id, job_dir_id, job_dir, tpu,
job_tags, log_dir, stage_dir, extra_configs,
status, error, stage, monitor,
rules: { preempted, grpc, locked },
extra_msgs: { father?, child?, fail_time*?, spreadsheet_notes?, ... },
customized_settings: { log_stage?, ... },
start_time: { chn, edt, utc }
```

---

## 7. Locking — read first to avoid deadlocks

All locks live in `lock.json` keyed by lock type: `code`, `data`,
`queue`, `legacy`, `apply`. Each is `{ status: bool, user: str|null }`.
Implementation: `utils/data_io.py::_mutate_lock_file` uses `fcntl.LOCK_EX`
on `lock.json` itself for atomic compare-and-set.

Rules to follow (also documented in `NOTES.md`):

- **Once you take a lock you must release it before returning**, even on
  exception. Use try/except + `release_lock_data()` like the rest of the
  codebase.
- `KeyboardInterrupt` is **not** an `Exception` — catch it separately if
  you hold a lock.
- For read-side / status commands (`check`, `ack`), prefer
  `read_data_if_unlocked()` and silently no-op if it returns
  `(None, False)`. Never block `tpu check` behind a writer.
- Lock-acquire helpers spin every 10s and give up after 30 minutes
  (`num_ack > 180`).

To release a stuck lock by hand: `tpu rl/unlock <code|data|queue|legacy|apply>`.

---

## 8. Conventions worth knowing

- All printed output uses the color tags from `constants.py`:
  `GOOD`, `INFO`, `WARNING`, `FAIL`, `LOG`. Stick to them; users grep on
  the brackets.
- TPU names: aliases like `v2-32-p1` resolve via
  `data["tpu_aliases"]`. `helpers.get_zone_pre_spot(tpu)` returns
  `(zone, preemptible, spot, full_name)` and is the canonical resolver.
- All status detection regexes live in `_render_rows_for_job` and in
  `check_jobs`. Keep the two in sync if you add a new pattern.
- Job lookups are still by `str(windows_id)` in some places and by
  `int(windows_id)` in others. Be defensive with the comparison.
- Backups: many files have `*.bak_<reason>_<timestamp>` siblings, e.g.
  `utils/jobs.py.bak_zone_default_20260521_090359`. Treat them as
  read-only history; do not import or `Read` them by accident when
  searching.

---

## 9. Where to start for common asks

- "Add a CLI subcommand" → add a `elif cmd == "..."` row in `tpu.py`,
  implement it in `utils/<area>.py`.
- "Change how errors are detected" → grep for `write_error_to_job` in
  `utils/jobs.py` and the `check_job_status` in `MONITOR.py`.
- "Change auto-resume rules" → `RULE_DICT` in `utils/constants.py` and
  the dispatch table in `MONITOR.py::mainloop`.
- "Touch the TPU itself" → `utils/operate.py`. **Never** call gcloud
  from anywhere else.
- "Change the data layout" → update `users.User.to_dict` /
  `jobs.Job.to_dict`, write a migration in `utils/develop.py`, and
  back up `data.json` first.
- "Debug a stuck lock" → `tpu rl <type>` from the CLI, or edit
  `lock.json` and set the offending `status` to `false`.

---

## 10. TPU mount-disk environment wheel

Active environment setup is `utils/operate.py::mount_disk`. It resolves the
TPU zone to a same-zone bucket, downloads a tar containing `wheels/*.whl`,
and installs those wheels offline with:

```
pip install --no-index --find-links=wheels wheels/*.whl --no-deps --force-reinstall
```

As of 2026-06-14 the active tar is:

```
hanhong/env_wheels/20260614_wandb0272_standard/v5_wheels_xin_20260614_wandb0272.tar.gz
```

It is present in the three active launch buckets:

- `gs://kmh-gcp-us-central1/...`
- `gs://kmh-gcp-us-east5/...`
- `gs://kmh-gcp-asia-northeast1-b/...`

Artifact details:

- size: `5117040627` bytes
- sha256:
  `280217eb780ef990f4dc122ed079f7f31970048bf775fe1c8eb91f3127a0e8ad`
- GCS md5 on all three copies: `WLj6hJwGeLWylWsf8rtpTw==`
- built from the old `hanhong/v5_wheels_xin.tar.gz` plus the packages that
  used to be installed after extraction:
  `datasets==4.4.2`, `transformers==4.41.2`,
  `huggingface-hub==0.36.2`, `numpy==1.26.4`, `ftfy==6.3.1`,
  `promise==2.3`, `torchdata==0.8.0`, and upgraded
  `wandb==0.27.2`.
- `protobuf` is pinned to `4.25.9` so the newer `wandb` stays compatible
  with `tensorflow==2.15.0`'s `<5` protobuf requirement.

Important upload caveat: do **not** use parallel composite upload for this
tar. TPU-side `gsutil` may lack compiled `crcmod` and can refuse to download
composite objects. Use a standard-object upload, e.g.:

```
gsutil -o 'GSUtil:parallel_composite_upload_enabled=False' cp <tar> <gs://...>
```

Validation performed on `kmh-tpuvm-v5p-64-spot-llq5xnxmz` in `us-east5-a`:

- manually installed the standard tar on all eight workers;
- import/version smoke passed for `wandb`, `datasets`, `transformers`,
  `numpy`, `huggingface_hub`, `protobuf`, `torchdata`, `promise`, `jax`,
  and `tensorflow`;
- `tpu run kmh-tpuvm-v5p-64-spot-llq5xnxmz sqa dir=1` reached
  `Status: Compiling`, then the test window was killed intentionally.

There is currently no xibo alias named `ka` in `data.json`; if someone asks
for `tpu run ka ...`, first check aliases with `tpu -lta` or use the full TPU
name.
