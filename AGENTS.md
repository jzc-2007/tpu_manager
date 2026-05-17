# Repository Notes For Agents

This repository is a lightweight working copy for the local `MONITOR.py` flow.
Do not assume every file in this directory is active production code.

## Active Path

- Treat `MONITOR.py` as the main script for automatic resume and local queue handling.
- `MONITOR.py` intentionally calls `_tpu = 'python /home/jzc/zhichengjiang/working/xibo_tpu_manager/tpu.py'`.
  That external xibo checkout is the desired TPU command backend.
- The local queue used by this monitor is `./queue.json`, via `MONITOR.py`'s `QUEUE_FILE`.
  This is separate from the xibo checkout's queue files.
- `find_saving_window.py` is still active: `MONITOR.py` shells out to it when deciding
  which ancestor window has a useful checkpoint.
- `see_log.py` is kept for `/home/sqa/.bash_aliases` helper `cl <window_id>`.
- The only active local `utils` modules are:
  - `utils/constants.py`: external xibo paths, log labels, allowed TPU regions.
  - `utils/data_io.py`: minimal reads from the external xibo `data.json` and `lock.json`.

## Queue Semantics

- Queue jobs are consumed only by `USER == "sqa"` in `MONITOR.py`.
- The monitor handles resume work before queue work. Queue priority is only used among
  queued jobs after resume handling has finished for the loop.
- Queue entries may have:
  - `priority`: any integer; larger values are tried first.
  - `wandb_notes`: copied from the job config for `queue.json` readability.
- The usual user-facing helper is `qsqa` from `/home/sqa/.bash_aliases`.
  It stages the current directory, runs `tpu set-cur`, then calls local `MONITOR.py queue`.
  Supported forms include `qsqa <dir>`, `qsqa <dir> 10`, `qsqa <dir> priority=10`,
  and `qsqa <dir> p=10`.
- Missing `priority` and `wandb_notes` are backfilled lazily for old pending/running
  entries when queue list/run logic reads them.
- `preferred_zone` exists in the queue schema but is not a current workflow dependency.
- A queue dispatch is considered consumed once the external `tpu run` output shows that
  a tmux job was created. Later `[FAIL]` lines from sheet/alias bookkeeping should not
  put the queue entry back into pending, or the same queued job can create many windows.
- Conversely, `tpu run` returning code 0 is not enough. The xibo backend can print
  `TPU ... is reserved by <user>` and `Quiting...` with returncode 0; without a tmux
  creation signal, the queue entry must stay pending.
- Before queue creates a job, `MONITOR.py` first runs `zhan/fang/mount-disk`, treats
  `[FAIL]`/error text in that output as failure even if returncode is 0, then SSHes to
  all TPU workers and requires `python -c "import jax"` to print `JAX_OK`.
- Within each queue selection pass, normal jobs try larger allowed TPU types first.
  GPT-B/DeiT jobs invert this and try smaller allowed cards first. Size is primary and
  generation breaks ties, so `v6-64 > v5-64 > v6-32`.
- TPU selection must not trust `tou`/`wrap_master.py` IDLE alone. Newly-created jobs can
  look IDLE before worker-side processes appear. Before queue uses an idle TPU, also
  check external `data.json` plus local tmux windows; TPUs with existing `status=running`
  windows or `status=error` windows that are judged false-positive errors are occupied.
- `tou` is backed by the `yizhitou` tmux session, which continuously refreshes
  `wrap_master.py`'s cache. `MONITOR.py` parses the reported cache age; if the delay is
  too large, it aborts the current loop, recreates `tmux new-session -d -s yizhitou`,
  waits briefly, and sends `yizhitou` into that session before trying again later.

## Resume State

- `sqa.json` is the monitor's local memory. `running` is only an in-progress lock,
  while `finished` means "this window has already been handled by the monitor".
- A resume/rerun attempt is considered consumed once the external command output shows
  that a tmux job was created. The triggering failed window must then be moved to
  `finished`, even if later sheet/alias bookkeeping prints `[FAIL]`.
- If a child fails before writing a new saving line, `find_saving_window.py` can resolve
  back to the same parent checkpoint. That is expected; the important invariant is that
  each successful child creation marks the triggering failed window finished.
- If an error window already has a recorded `child`, or its checkpoint source has a newer
  child than this window, treat it as stale and mark it handled locally instead of
  resuming it again.
- When checking whether a checkpoint source's child supersedes an error window, first
  walk the error window's `extra_msgs.father` chain. If that child is on the father
  chain, it is an ancestor path to the current leaf, not a competing newer child.
- Do not add retry cooldowns to solve duplicate resumes. The desired behavior is to try
  a still-failed job again on the next loop. The real invariant is: once a new tmux job is
  created, the source window must be marked finished locally.
- Resume selection follows the same extra occupancy check as queue selection: an idle TPU
  is unavailable if `data.json` and tmux show an existing running window on that card, or
  an error window whose recent log tail does not contain `Retrying: SSH command error`.
- Before resume/rerun creates a child tmux job, `MONITOR.py` runs the same
  `zhan/fang/mount-disk` output check and all-worker JAX import sanity check as queue.
  If that check fails, clear the local `running` marker and retry the failed source in a
  later loop instead of creating a doomed child window.
- Within each resume selection pass, normal jobs try larger allowed TPU types first.
  GPT-B/DeiT jobs try smaller allowed types first. Type ranking treats size as primary
  and generation as the tie breaker, so `v6-64 > v5-64 > v6-32`.
- If `tou` cache is stale during any resume path, do not continue selecting TPUs from
  that output. Restart `yizhitou`, clear any local in-progress marker for the current
  window, and let the next monitor loop retry with fresh cache.

## Duplicate Window Incident

- On 2026-05-14, repeated queue/resume windows were caused by success misclassification:
  the external `tpu run`/`resume` command created a tmux window, then later printed
  `[FAIL] get_zone_pre...` or sheet/alias bookkeeping errors. Older monitor logic treated
  any `[FAIL]` as command failure, moved queue entries back to pending, or failed to
  mark the source error window as finished.
- Correct handling is based on creation signals such as `Successfully created job in
  tmux window` or `run/resume/rerun job ... with new windows id N`. Once those appear,
  queue entries are consumed and source resume windows are finished, regardless of later
  bookkeeping errors.
- Manual cleanup for duplicate error windows should be non-destructive. Group current
  `status == "error"` windows by logical job, usually `(job_dir_id, spreadsheet_notes or
  job_tags)`. Ignore `resumed` and `rerunned` windows when counting how many error
  windows remain.
- When choosing which duplicate error leaf to keep, use an effective step, not only the
  error leaf's own log. Read the leaf's `log_dir/output.log` and also walk
  `extra_msgs.father` through its resume/rerun ancestry, reading each ancestor
  `output.log`. Keep the error leaf whose leaf-plus-ancestor chain has the largest parsed
  step; this preserves leaves like `6528` whose own resume failed early but whose father
  `6517` had trained much farther than another duplicate branch. Ties should prefer the
  latest leaf window.
- Step parsing must handle more than `step N` text. Training logs also use bracketed
  metric lines such as `logging_util.py:105] [75300] train_...`, and restored checkpoint
  paths like `checkpoint_400320` are meaningful progress signals for failed resume
  attempts.
- If `spreadsheet_notes`/`job_tags` are empty, do not blindly merge all errors with the
  same `job_dir_id`. Inspect `extra_configs`, `stage_dir`, `log_dir`, and the loaded
  checkpoint before deciding whether they are the same logical job.
- Mark the other duplicate error windows as `killed` with `extra_msgs.dedupe_*` metadata.
- The 2026-05-14 cleanup wrote a backup of the external xibo `data.json` and a report
  under local `logs/manual_dedupe_error_windows_*.json`. The cleanup intentionally did
  not delete historical job records or logs.
- On 2026-05-17, window `6623` failed with `ModuleNotFoundError: No module named 'jax'`
  because the preceding `ftmd` step printed `[FAIL] Failed to fang new TPU...` while
  returning 0. The monitor treated that as success, then `resume` quick-registered the
  TPU and created a tmux job on a card that had not actually received the mounted JAX
  environment. This is why `ftmd` output markers and explicit JAX checks matter.

## TPU Region Policy

- Resume and local queue dispatch should only use TPUs in:
  - `us-central1`
  - `us-east5`
  - `asia-northeast1-b`
- There is no longer a special LLaVA-OV1.5 restriction to `us-central1`; datasets are
  available in all allowed regions.
- Parse resume target zones from the real zone suffix in `log_dir`, not from the first
  zone-like string in the TPU name. TPU names can contain text such as
  `us-central1-spot-...`; that is not a zone. Prefer the last allowed zone match from
  the log path, such as the `_<zone>__...` suffix.

## Non-Goals In Current Flow

- `preempted` jobs are intentionally not handled by this monitor. Another script deletes
  those TPUs, and the next monitor loop should see them as `deleted`.
- `job["rules"]` is not used by the active monitor path.
- The old local TPU manager implementation (`tpu.py`, `web_cli.py`, old shell helpers,
  and legacy `utils/*` modules for users/jobs/sheet/queue/etc.) was removed from this
  lightweight checkout. Use the external xibo `_tpu` backend for those operations.

## Editing Guidance

- Keep changes focused in `MONITOR.py`, `find_saving_window.py`, or the two retained
  `utils` modules unless the active path proves otherwise.
- Preserve the external xibo `_tpu` path unless explicitly asked to change it.
- Keep `queue.json` backwards compatible; old entries without new fields should still run.
