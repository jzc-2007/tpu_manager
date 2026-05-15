# Repository Notes For Agents

This repository is a lightweight working copy for the local `MONITOR.py` flow.
Do not assume every file in this directory is active production code.

## Active Path

- Treat `MONITOR.py` as the main script for automatic resume and local queue handling.
- `MONITOR.py` intentionally calls `_tpu = 'python /home/jzc/zhichengjiang/working/xibo_tpu_manager/tpu.py'`.
  That external xibo checkout is the desired TPU command backend.
- The local queue used by this monitor is `./queue.json`, via `MONITOR.py`'s `QUEUE_FILE`.
  This is separate from the queue implementation under `utils/queue.py` and from the
  xibo checkout's queue files.

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
- Do not add retry cooldowns to solve duplicate resumes. The desired behavior is to try
  a still-failed job again on the next loop. The real invariant is: once a new tmux job is
  created, the source window must be marked finished locally.

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
- Most files under `utils/` are legacy or support code for the external `tpu.py` style
  workflow. Read them for context, but verify they are actually on the active path before
  changing behavior.

## Editing Guidance

- Keep changes focused in `MONITOR.py` unless the active path proves otherwise.
- Preserve the external xibo `_tpu` path unless explicitly asked to change it.
- Keep `queue.json` backwards compatible; old entries without new fields should still run.
