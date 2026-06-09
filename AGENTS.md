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
  Pass the current user to it; window ids are only unique within a tmux/user namespace,
  and different users can have the same numeric window id.
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
- Queue/staging dir numbers must stay in xibo's accepted range `1..100`. Do not use
  `101`, `102`, etc.; `qsqa` can still add a local pending queue entry after `tpu set-cur`
  rejects those numbers, but that entry has no usable xibo working-dir mapping.
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
- `tpu mount-disk` must not trust `/home/sqa/.disk_mounted` alone. That marker can
  survive while `/kmh-nfs-ssd-us-mount` is not actually mounted, causing staged code
  copies to fail with `cp: cannot stat ... staging/sqa/...` and then
  `/home/sqa/main.py` missing. The xibo mount guard should require both the marker and
  `mountpoint -q /kmh-nfs-ssd-us-mount`; monitor's `tmd` wrapper also verifies the
  mount on all workers and reruns `tpu mount-disk --force` once if the check fails.
  The `resume_next_round` direct-old-TPU path must do the same check explicitly because
  external `tpu resume ... tpu=<old_tpu>` creates the child tmux window without running
  `mount-disk` first.
  For the xibo CLI, put `--force` after the TPU/alias (`tpu mount-disk <name> --force`);
  `tpu.py` reads `args[2]` as the TPU name.
- Some xibo alias rows are dirty: one old TPU can have more than the normal pair of
  tmp aliases, for example `v5-64-tmp201`, `v5p-64-tmp201`, and the real TPU name.
  The external `fang` command refuses these with "found more than 2 aliases for the old
  TPU"; `MONITOR.py` should skip those aliases when selecting a new alias.
- Before queue runs any destructive `mount-disk`/`tmd` step, it SSHes to the candidate
  TPU and runs a remote occupancy probe. The probe first checks whether TPU device files
  are held (`/dev/vfio/*` or `/dev/accel0`), then falls back to a narrow xibo
  training-command signature (`python main.py --workdir=/kmh-nfs-ssd-us-mount/logs/...`).
  If that probe is `busy` or `unknown`, skip the candidate. This guards against
  xianbang-style cases where two lease names point at the same physical VM without
  treating every unrelated Python process as a TPU job.
- The remote occupancy probe may time out while still returning partial stdout from
  workers that did answer. If stdout contains matching processes, classify it as `busy`
  rather than `unknown`; the important signal is that at least one worker is occupied.
- The central xibo reserve locks live under `/kmh-nfs-ssd-us-mount/code/qiao/tpu_lock`
  and are keyed by exact TPU lease name with a 30-minute lifetime. They are useful as an
  extra same-name race guard, but they do not prove a physical worker is free: a lock for
  `...-b6da37` will not block `...-c63dcd` even if both lease names reach the same VM.
- Within each queue selection pass, non-Asia candidates are tried before
  `asia-northeast1-b`. Queue jobs are from-scratch launches and must use exact
  cost classes, not "any larger card": low-cost jobs use only `v6e-32` or
  `v5/v5p-64`, high-cost jobs use only `v6e-64` or `v5/v5p-128`, and GPT-B/DeiT
  jobs keep the small-card rule (`v6e<=16` or `v5/v5p<=32`). A queue
  `preferred_type` is only a preference inside these allowed classes; it must not
  force a larger low/high-cost card.
- TPU selection must not trust `tou`/`wrap_master.py` IDLE alone. Newly-created jobs can
  look IDLE before worker-side processes appear. Before queue uses an idle TPU, also
  check external `data.json` plus local tmux windows; TPUs with existing `status=running`
  windows or `status=error` windows that are judged false-positive errors are occupied.
- `tou` is backed by the `yizhitou` tmux session, which continuously refreshes
  `wrap_master.py`'s cache. `MONITOR.py` parses the reported cache age; if the delay is
  greater than 60 seconds, it aborts the current loop, recreates
  `tmux new-session -d -s yizhitou`, waits briefly, and starts an explicit
  `wrap_master.py --cache false` loop in that session before trying again later. If the
  cache is stale but that refresh process is already running, do not kill it; just abort
  the current monitor loop and let the refresh finish. Stale-cache aborts sleep briefly
  before retrying; normal monitor loops still sleep five minutes.

## Resume State

- `sqa.json` is the monitor's local memory. `running` is only an in-progress lock,
  while `finished` means "this window has already been handled by the monitor".
- When reporting the user's current active jobs, do not summarize every historical
  `data.json` entry. The user's active jobs are the leaf jobs whose tmux windows still
  exist in the `sqa` tmux session and have not been superseded by `extra_msgs.child`.
  Old `data.json` records can remain `error` or even stale `running` after their tmux
  windows are gone; those are history, not current work.
- When reporting active job status, immediately diagnose any active `Error` leaf enough
  to classify it. No-card / TPU preempt / deleted-card failures are expected operational
  churn; code, config, checkpoint, environment, or data-loader errors are abnormal and
  should be called out instead of only reporting `Error`.
- `tpu check` can show a false `Staging` status for a `data.json status=running` leaf
  whose tmux pane is actually idle at a shell prompt after a bare `^C`. In that case,
  inspect the pane/log for the last real training step and checkpoint. `mer <window>`
  only appends the resume-needed `Retrying: SSH command error` marker; if xibo still
  reports `Staging`, also mark that leaf `status: error` in external `data.json`, keep
  pretty JSON formatting, and ensure the window is not in local `sqa.json` `running` or
  `finished` so the monitor can resume it.
- To manually re-enable a historical error leaf that the user still wants resumed,
  recreate an empty tmux window with the same id in the `sqa` session and remove that
  window id from local `sqa.json.finished` / `sqa.json.running` if present. The monitor
  will then see it in `tpu check sqa` and apply the normal checkpoint-source logic.
  The recreated tmux pane does not need to run the old command.
- A resume/rerun attempt is considered consumed once the external command output shows
  that a tmux job was created. The triggering failed window must then be moved to
  `finished`, even if later sheet/alias bookkeeping prints `[FAIL]`.
- If a child fails before writing a complete checkpoint, `find_saving_window.py` can
  resolve back to the same parent checkpoint. That is expected; the important invariant
  is that each successful child creation marks the triggering failed window finished.
- `find_saving_window.py` must not treat a bare `Saving checkpoint...` line as usable.
  A checkpoint is usable only after a later `saved to` line. If the latest `Saving`
  step is newer than the latest `saved to` step, remove the corresponding
  `checkpoint_<step>` candidate from the logdir/GCS path before returning that window;
  otherwise the external xibo resume path may load an empty half-written checkpoint.
- After walking a window's father chain, `find_saving_window.py` also does a conservative
  same-logical-job scan using `(job_dir_id, spreadsheet_notes/job_tags)` and chooses the
  highest complete checkpoint if it is newer than the father-chain checkpoint. This catches
  stale duplicate branches such as window `6886`, whose father was old `6833` even though
  sibling branch `6852 -> 6856 -> 6871` had already produced a newer complete checkpoint.
- The external xibo backend used by `_tpu` should strip inherited resume-only config
  overrides (`--config.load_from`, `--config.wandb_resume_id`, `--config.stage`) before
  composing a resume/rerun command. Otherwise a window whose own logdir has no usable
  checkpoint can still launch with an old parent `--config.load_from`, causing stale
  checkpoint restores and WandB monotonically-increasing-step warnings.
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
  an error window whose recent log tail does not contain a resume-needed marker.
- Resume-needed error markers include `Retrying: SSH command error`,
  `Unable to initialize backend 'tpu': ABORTED`, and random JAX distributed aborts such
  as `Fatal Python error: Aborted` / `JAX distributed service detected fatal errors` /
  `CoordinationService/Heartbeat` with `DEADLINE_EXCEEDED`. These random aborts often
  leave the TPU visible, so the monitor should try the old TPU first when it is idle,
  while still respecting `tou`, data/tmux occupancy, and remote busy probes.
- Failed worker startup where the staged code is missing, for example
  `python: can't open file '/home/sqa/main.py'` or `cp: cannot stat ... staging/sqa`,
  is also a resume-needed marker. This usually means not all TPU workers received the
  staged code before distributed JAX startup; treating it as a false-positive running
  error strands the job.
- Failed worker startup with `ModuleNotFoundError: No module named 'jax'` is also a
  resume-needed marker. It usually means a launch raced an incomplete or wrong
  `mount-disk`/alias environment; first force-mount the correct alias/TPU and verify
  all workers can `import jax`, then rerun from the same root window if no checkpoint
  was ever written.
- For random JAX aborts, `tou` may still report the old TPU as busy under `sqa` because
  dead worker processes have not fully cleared. If the busy users are only the current
  user and `data.json` plus tmux do not show any other active window on that TPU, the
  monitor may resume directly on the old TPU instead of treating it as unavailable.
- Before resume/rerun creates a child tmux job, `MONITOR.py` runs the same
  `zhan/fang/mount-disk` output check and all-worker JAX import sanity check as queue.
  If that check fails, clear the local `running` marker and retry the failed source in a
  later loop instead of creating a doomed child window.
- Resume/rerun also runs the same pre-`mount-disk` remote occupancy probe as queue. A
  candidate whose TPU devices or xibo training command are active, or whose occupancy
  cannot be checked, must be skipped before `tmd`, because `mount-disk` can remove
  `/home/sqa/.local` and kill jobs on the same physical worker.
- Checkpoint resume is exact-type: if `find_saving_window.py` returns action
  `resume`, `MONITOR.py` must only choose the same TPU type as the original job,
  although it may use another allowed zone/region with that same type. Do not fall
  back from a checkpointed `v6e-64` run to `v5p-64`, or from any checkpointed
  topology to a different TPU type.
- Only a checkpoint-free action `rerun` may choose a new type by cost class, because
  it is starting from scratch. The cost classes are exact: low-cost `v6e-32` or
  `v5/v5p-64`; high-cost `v6e-64` or `v5/v5p-128`; GPT-B/DeiT `v6e<=16` or
  `v5/v5p<=32`. `--same-type-only` now means even checkpoint-free reruns must keep
  the original type; checkpoint resumes are always same-type regardless of the flag.
- Jobs whose notes contain `llava-1.5 reproduction` are low-cost but v5-only:
  from-scratch launches should use exactly `v5/v5p-64` and must not use v6/v6e,
  because the reproduction job OOMs on v6. This restriction applies to queue,
  resume fallback, and direct old-TPU resume buffers; queue `preferred_type`
  must not override it.
- If `tou` cache is stale during any resume path, do not continue selecting TPUs from
  that output. Restart `yizhitou`, clear any local in-progress marker for the current
  window, and let the next monitor loop retry with fresh cache.
- `resume_next_round` buffer resumes must still run the remote occupancy probe before
  directly reusing the old TPU. `tou` can report the old TPU as idle while worker-side
  `python main.py --workdir=...` processes are still alive. If the old TPU is remotely
  busy, remove the buffer entry and treat the job like the old TPU is unavailable so it
  falls back to a different card.
- `tou` may omit a TPU whose GCP state is `DELETING`; this is not a true ambiguous
  state. `check_job_status` should classify `DELETING`/`deleted` as `deleted`, and
  `_get_tpu_usage_from_tou` should distinguish a missing lease from a malformed visible
  line. Missing old TPUs should go through the normal deleted-card fallback path instead
  of being skipped as "状态不确定". This happened on 2026-06-02 for window `7154` on
  `kmh-tpuvm-v5p-64-kangyang-01-5bdd6a4b`.

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
- When more than one idle TPU is available, prefer non-`asia-northeast1-b` TPUs
  before Asia TPUs. LAION-400M is only available in Asia and Kaiming wants to
  preserve those cards for that workload. If Asia is the only viable idle
  capacity, monitor/queue may still use it.
- There is no longer a special LLaVA-OV1.5 restriction to `us-central1`; datasets are
  available in all allowed regions.
- Parse resume target zones from the real zone suffix in `log_dir`, not from the first
  zone-like string in the TPU name. TPU names can contain text such as
  `us-central1-spot-...`; that is not a zone. Prefer the last allowed zone match from
  the log path, such as the `_<zone>__...` suffix.
- On 2026-06-02, quota was being held by stale GCP TPU nodes stuck in `DELETING`.
  To audit this, run `gcloud compute tpus locations list` and query each location with
  `gcloud compute tpus tpu-vm list --zone=<zone> --filter=state=DELETING`. A second
  `gcloud compute tpus tpu-vm delete <name> --zone=<zone> --quiet --async` can unblock
  stale deletes. This cleanup removed old `DELETING` nodes including several v5p-64s in
  `us-east5-a` from 2026-05-28 and `kmh-tpuvm-v5p-64-spot-llqakha3n` from 2026-04-12;
  final full-location scan showed no remaining `DELETING` TPU VMs.

## ImageNet kNN / Remote Environment Notes

- On 2026-05-28, `PaliGemma-baseline` and `beifen-Paligemma` moved their
  PyTorch-based ImageNet kNN eval loaders to zone-local TFDS. The intended behavior is
  to read ImageNet from `gs://kmh-gcp-<region>/tensorflow_datasets` and avoid copying
  ImageNet into each new logdir or stagedir.
- For remote eval sanity, prefer a finished stagedir from `tpu check sqa`, copy only that
  small code stagedir, set `eval_only: True`, `final_eval_tasks: [knn_full]`, clear
  `wandb_resume_id`, and point `load_from` at the finished run's logdir/checkpoint.
- Before debugging remote eval on a reused TPU, run `tpu mount-disk <tpu> --force` and
  verify all workers can import the required Python modules. A missing package can look
  like a model/eval bug while the real issue is an incomplete mount environment.
- The xibo `mount-disk` v5/v6 Python env must include `promise`, because
  `tensorflow_datasets` imports it. Requirement strings containing shell metacharacters,
  such as `huggingface-hub<1.0,>=0.23.0`, must be quoted in the generated remote shell
  command or bash interprets `<` as redirection.
- The first TFDS kNN sanity used window `6952` as the reference finished job. The final
  correct run restored `checkpoint_150000` from the same us-east5 run and reported
  `knn_full_acc_final=1.488` versus the old loader's `1.628`. This is close in absolute
  percentage points but not bit-identical, likely due TFDS/TF resize/decode versus the
  old PyTorch/PIL preprocessing path.
- On 2026-05-31, `jax_llava` and `beifen-Paligemma` added PCA whitening for ImageNet
  kNN after feature extraction. On 2026-06-01 this was changed to report both normal
  raw-feature cosine KNN and PCA-whitened cosine KNN from one feature extraction.
  The legacy metric name, for example `knn_full_acc_final`, is the raw/normal KNN;
  the explicit duplicate `knn_full_acc_raw_final` is also logged, and
  `knn_full_acc_pca_whitened_final` is the whitened result. Fit whitening on the KNN
  train/reference features only, then apply the same transform to train and validation
  features: `(x - train_mean) @ eigvecs / sqrt(eigvals + eps)`. Defaults:
  `eval.knn_eval_raw_and_pca_whitened=True`, `eval.knn_pca_whitening=True`,
  `eval.knn_pca_whitening_eps=1e-5`, `eval.knn_pca_whitening_dim=0` (keep all dims),
  `eval.knn_pca_whitening_batch_size=65536`.
- On 2026-06-01, run `z8ceupo9` was re-evaluated with the copied stagedir plus current
  TFDS/PCA-whitened KNN code. Because the original checkpoint lived in `us-central1` but
  no clean us-central1 card was available, only `checkpoint_150000` was copied to the
  same relative path under `gs://kmh-gcp-us-east5/...`, then eval ran on
  `kmh-tpuvm-v6e-16-kaiminghe-269fe1` in `us-east5-b` using
  `gs://kmh-gcp-us-east5/tensorflow_datasets`. Result:
  `knn_full_acc_final=56.73` at step `150000`, W&B eval run `8pzhuygq`, logdir
  `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260601_0044_z8ceupo9_whitened_knn_kmh-tpuvm-v6e-16-kaiminghe-269fe1_us-east5-b__eval_only`.

## PaliGemma SFT From Pretrain

- To run an SFT stage from a finished pretrain job, start from the finished pretrain
  job's stagedir rather than the current worktree. Copy only the small code stagedir
  into `/kmh-nfs-ssd-us-mount/code/qiao/work/...`, then patch configs. This keeps model
  code and checkpoint structure matched to the pretrain run.
- In that copied stagedir, set `configs/remote_run_config.yml` `finetune: True`.
  `configs/load_config.py:remote_run` will then load `configs/finetune_config.yml`.
- Keep the finetune model architecture aligned with the pretrain checkpoint before
  setting `load_from_pretrained`: `patch_size`, `image_size`, `txt_feature_layer`,
  encoder depth, and `enc_cross_attn_split` must match. Training-only choices such as
  optimizer, LR, weight decay, and whether to freeze LM can be changed for SFT.
- SFT is an explicit exception to pretrain soft-cap matching: disable soft caps by
  setting `model.attn_logits_soft_cap = 0.0` and `model.final_logit_softcap = 0.0`
  in `configs/finetune_config.yml` unless the user explicitly asks otherwise.
- Use `load_from_pretrained` for the pretrain logdir when starting a fresh SFT run.
  Reserve `load_from` and `wandb_resume_id` for resuming an interrupted SFT run.
- `PaliGemma-baseline` and `beifen-Paligemma` also support an in-process
  pretrain->SFT curriculum through `training.curriculum: pretrain_sft_two_stage`
  with `finetune: False`. This keeps one workdir and global checkpoint stream:
  `load_from` step `< stage1_steps` resumes pretrain, `== stage1_steps` restores
  params only into the SFT/no-decoder model with a fresh optimizer, and
  `> stage1_steps` resumes SFT fully. Do not use `load_from_pretrained` in this
  two-stage path; keep it for the legacy `finetune: True` finetune-only config.
  The stage-1 boundary checkpoint is copied to durable `pretrained-ckpts/...`
  before switching stages.
- As of 2026-06-08, `beifen-Paligemma/configs/remote_run_config.yml` is the active
  runnable two-stage config: 150K-step 512px MAE-B pretrain followed by 90K-step SFT
  on shuffled LLaVA-OV1.5 plus the current VQA/grounding mix, with stage-2 Adam
  `1e-5` and weight decay `0.002`.
- As of 2026-06-08, `PaliGemma-baseline/configs/remote_run_config.yml` is also an
  active runnable two-stage config under `sharding: hsdp`: 150K-step 512px sandwich
  MAE-L + Gemma3-1B pretrain followed by 90K-step SFT on the same shuffled
  LLaVA/current VQA/grounding mix, with stage-2 Adam `1e-5` and weight decay `0.002`.
- Before queueing full SFT runs across the normal monitor, copy the chosen pretrain
  checkpoint to every allowed zone with `tpu cc` so cross-zone restore can work:
  `us-central1`, `us-east5`, and `asia-northeast1-b`.
- For the 2026-05-30 SFT follow-up to
  `[150K steps, muon 2e-5 wd2e-3]`, keep the newer expanded eval set from
  `beifen-Paligemma`, including `gqa`, `vizwiz`, `scienceqa_img`, and `seed_bench`.
  The SFT training mix is `llava-ov-1.5-instruct`, `vqav2`, and `gqa-train`; use the
  GQA question count in millions as its mix weight, currently `0.94`.
- Before queueing the real sweep, run a lightweight remote finetune debug config that
  restores the pretrain checkpoint and exercises all three paths: a train step, online
  eval, and final eval. Keep KNN and VLM eval sample counts capped in that smoke run.
- On 2026-05-30, the SFT runs showed an apparent train-accuracy spike/drop around
  steps 15k-18k. This was not a plain dataloader seed reset: the staged
  `beifen-Paligemma` code sets `data_seed_offset = dataset.data_seed_offset +
  checkpoint_step(load_from)` for `load_from` resumes, and the logs showed offsets
  `15000` and `18000` after resume. The artifact came from logging steps past the last
  complete checkpoint, then resuming from that older checkpoint with the same WandB run
  id. Example: window `7056` logged through `17900` but only had a complete checkpoint
  at `15000`; window `7061` resumed from `15000` and re-ran `15100..18000`, producing
  lower train acc and W&B warnings like `Tried to log to step 17400 that is less than
  the current step 17701`. Treat these duplicate-step ranges as W&B/history artifacts
  unless eval metrics or later accepted logical checkpoints show a real regression.
- Current dataloader resume seeding is a non-repetition heuristic, not exact replay of
  the infinite mixed data stream. Exact continuation would require saving DataLoader /
  WebDataset / RandomMix worker RNG state or skipping the stream by global sample count.
  For SFT, shorter checkpoint intervals reduce the amount of logged-but-uncheckpointed
  work that can be replayed after preemption.
- Be careful with `rerun` on interrupted SFT windows. A plain `rerun` may keep the old
  `wandb_resume_id` but drop `load_from`, causing a fresh train-from-pretrain run whose
  early-step logs are ignored by WandB because the run already reached later steps.
  For interrupted SFT continuation, prefer `resume` from the latest complete finetune
  checkpoint and keep `wandb_resume_id`; use a fresh WandB id only for an intentional
  restart from pretrain.
- Also on 2026-05-30, SFT `valid_tokens_per_sample` decreased steadily over training.
  `valid_tokens` is computed directly from `labels != -100`, so this is a data-stream
  effect, not model behavior. Audits were same-region only (`us-east5` TPU reading
  `gs://kmh-gcp-us-east5/...`). Shard-level audits did not show a simple per-shard
  monotonic sort: all-shard prefix audit `u8hxsbpk`, deep first-position audit
  `yytlsvoj`, and light first-12-position audit `j1g964fg` all had near-flat
  first-vs-last shard curves. The decisive reproduction was the stream-level simulator
  `1oqq2dm5`: no model, no image decode, just the runtime-like `RandomMix` /
  `VQAv2IterableDataset` label stream, and rank0 still dropped from
  `valid_tokens_per_sample ~= 85` at step 100 to `~=65` at step 5000 while
  `llava_ov15_fraction` stayed around `0.94`. Therefore the drift is from the actual
  iterable stream order/buffering, not W&B, optimizer, eval, or dataset-source mix.
- Be careful when auditing SFT stagedirs after the fact. The running windows' `output.log`
  `FLAGS.config` is the source of truth. Some SFT windows ran an older staged copy whose
  log showed `input_pipeline.py:1358`, `dataset.num_workers: 32`, `prefetch_factor: 4`,
  only `item_shuffle_size.default: 512`, and no `stream_start_skip`, even though the NFS
  stagedir later visible to agents had been patched to `num_workers: 16`, per-type
  `item_shuffle_size: 1024`, and `stream_start_skip`. Do not assume the current stagedir
  contents are exactly what an already-running remote process imported.
- The mitigation added to `beifen-Paligemma` is a per-worker `dataset.stream_start_skip`
  plus larger per-type `item_shuffle_size`; SFT configs use `num_workers: 16`,
  `prefetch_factor: 2`, `item_shuffle_size` 1024 for `llava_ov15`/`vqav2`/`gqa`, and
  startup skip 2048/1024/1024 respectively. Existing already-running jobs do not pick
  this up until restarted/resumed from patched code. A same-region stream audit of this
  patched config, W&B `yjugt0wy`, did not flatten the metric: rank0 went
  `86 -> 64 -> 82` over the first 5000 simulated steps while the source mix stayed
  stable. Treat `stream_start_skip` as a phase/jitter mitigation only. It cannot remove
  long-range length drift caused by sequential per-worker tar streams plus finite local
  shuffle buffers. A real fix should shuffle after `RandomMix` with a larger mixed
  buffer, or otherwise make batching draw from a stationary cross-stream pool.
- A test-only "resample one random shard per worker epoch" simulation, W&B `o40migae`,
  was not a fix either: it started high but dropped to about `54` tokens/sample by
  step 5000 with the same stable source mix. Do not assume random shard order alone
  solves the SFT length curriculum; the fix needs example-level mixing or an offline
  globally shuffled dataset.
- A QA-expanded mixed shuffled dataset experiment was tried and abandoned because it
  duplicated image bytes and was too slow/heavy for the workflow. The related
  `sft_mix` reader, aliases, and builder were removed from `beifen-Paligemma`; the
  active path is the lighter raw LLaVA image-level shuffle below.
- The replacement one-time shuffle should only shuffle `llava-ov-1.5-instruct`, because
  the length drift appears to come from that source. Keep `vqav2`/`gqa` on their
  original roots. The shuffle unit is one raw LLaVA image-level WebDataset sample
  containing the image plus all conversations/questions; do not expand to QA turns
  offline, because that duplicates image bytes and creates TB-scale waste.
- Builder: `beifen-Paligemma/tools/build_llava_image_shuffle_wds.py`. It reads source
  LLaVA shards once in random shard order, keeps a bounded raw-sample buffer, randomly
  emits image-level samples into new tar shards, and preserves original LLaVA sample
  fields while rewriting only tar sample keys. Temporary tar files must stay under
  `/dev/shm`.
- Region launcher: `beifen-Paligemma/tools/run_llava_image_shuffle_region.sh`.
  Run it on a TPU VM in the same region as the source/target bucket:
  `bash tools/run_llava_image_shuffle_region.sh <zone> gs://kmh-gcp-<zone>/data/llava-ov-1.5-instruct-image-shuffled-v1 16`.
  It launches 16 parallel parts, uses seed `20260531`, `shuffle_buffer_samples=25000`,
  `samples_per_shard=10000`, and writes root `_SUCCESS` after all parts finish.
  If a spot card dies after writing partial shards, restart with `NO_CLOBBER=1`; the
  build deterministically replays the same seed and skips already-present tar uploads.
  The launcher also supports `MAX_PARALLEL`; use it when increasing the per-part
  buffer, because the buffer stores raw image bytes in Python memory while `/dev/shm`
  is only for temporary tar files.
- The pilot in `us-east5` wrote
  `gs://kmh-gcp-us-east5/data/llava-ov-1.5-instruct-image-shuffled-v1-pilot`:
  `200000` raw samples, `20` shards, `34.7GB`, duration `1039s`, `_SUCCESS` present.
  It validated schema compatibility with the existing `llava_ov15` dataloader. Do not
  overinterpret its token-length curve: with only `20` shards it is too small for the
  production `world*num_workers` stream partitioning and has visible buffer-fill/drain
  boundary effects.
- Full `us-east5` generation was started on 2026-05-31 on
  `kmh-tpuvm-v6e-16-kaiminghe-40c143` in tmux `llava_image_shuffle_full`, with a local
  lock refresher tmux `llava_image_shuffle_full_lock`. It runs 16 parallel parts using
  one shared shuffled source-shard order and `--shard-offset i --shard-stride 16`,
  writing to `gs://kmh-gcp-us-east5/data/llava-ov-1.5-instruct-image-shuffled-v1/part-XX`.
  Each part uses `shuffle_buffer_samples=25000`, `samples_per_shard=10000`, seed
  `20260531`. The root alias `llava-ov-1.5-instruct-image-shuffled-v1` resolves
  `part-*/shard-*.tar`; the full build does not have root-level `shard-*.tar` files.
- The `us-east5` full image-level shuffle completed on 2026-05-31 with root `_SUCCESS`,
  `16/16` part `_SUCCESS` files, and `1483` tar shards. The remote tmux exited cleanly,
  `/dev/shm` was empty afterward, and the local manual lock was removed.
- On 2026-06-01 the same full image-level shuffle was completed for the remaining
  regions without cross-region data reads:
  `gs://kmh-gcp-us-central1/data/llava-ov-1.5-instruct-image-shuffled-v1` and
  `gs://kmh-gcp-asia-northeast1-b/data/llava-ov-1.5-instruct-image-shuffled-v1`
  both have root `_SUCCESS`, `summary.json`, `16/16` part `_SUCCESS` files, and
  `1483` tar shards. `us-central1` used
  `kmh-tpuvm-v6e-16-kaiminghe-4b4bd1`; all part outputs finished but the final root
  marker was missing after the remote session ended, so root `_SUCCESS`/`summary.json`
  were manually written after verifying `16/16` parts and `1483` tar shards. Asia first
  wrote `368` partial tar shards on `kmh-tpuvm-v6e-16-kaiminghe-b95139`, which was then
  deleted/preempted; it was resumed on `kmh-tpuvm-v6e-16-kaiminghe-69d29b` with
  `NO_CLOBBER=1` and completed normally. The manual TPU locks for both cards were
  removed after verification.
- Important result: full image-level shuffle did **not** fully fix the SFT
  `valid_tokens_per_sample` drift. Same-zone stream audit
  `llava-full-image-shuffle-us-east5-rank0-stream-sim-5k-20260531`, W&B `rn36i3xu`,
  used `world=8`, `rank0`, `num_workers=16`, current online item buffer `1024`, full
  shuffled LLaVA plus original us-east5 VQAv2/GQA. It still declined from about
  `83-85` tokens/sample in the first few hundred steps to about `61` by step `5000`,
  while the LLaVA mix fraction stayed stable around `0.94`. Do not spend time generating
  extra variants from this image-level shuffle as a presumed fix. The likely next fix
  needs shuffling after LLaVA conversation expansion, or another QA-level/mixed-token
  buffer that avoids duplicating image bytes on disk.
- Follow-up: the `1024` online item buffer above was too small, roughly one global batch.
  Re-running the same full shuffled LLaVA root with `item_shuffle_size.llava_ov15=50000`
  and `num_workers=4` showed the image-level shuffle is useful: first 500-step mean
  `valid_tokens_per_sample ~=80.9`, last 500-step mean `~=76.4`, fitted slope
  `~-1.08 tokens/sample/1k steps`, with LLaVA fraction still around `0.94`. This is much
  better than the `1024` buffer curve (`~83-85 -> ~61` by 5k), though not perfectly flat.
  The 50k audit had a substantial startup cost because each worker must fill 50k raw
  image samples before emitting. With production `num_workers=16`, startup/memory cost
  scales up, but each worker's buffer drains more slowly than in the 4-worker audit.
- `beifen-Paligemma` and `jax_llava` were updated to use this policy for LLaVA SFT:
  `llava-ov-1.5-instruct-image-shuffled-v1` as the LLaVA dataset alias and
  `item_shuffle_size.llava_ov15=50000`. `jax_llava` also ports the `beifen` raw
  image-level buffer logic so the 50k buffer holds image samples plus pending QA groups
  rather than many expanded copies of the same image. As of 2026-06-01, the full
  shuffled dataset alias exists and is verified in all three allowed regions:
  `us-east5`, `us-central1`, and `asia-northeast1-b`.
- On 2026-06-02 the larger `us-east5` one-time shuffle trial completed:
  `gs://kmh-gcp-us-east5/data/llava-ov-1.5-instruct-image-shuffled-v2-buf125k`.
  It uses the same image-level shuffle unit and seed `20260531`, but with
  `SHUFFLE_BUFFER_SAMPLES=125000` (5x the v1 per-part buffer). The final layout is a
  mixed resume layout: original `part-00..11` plus split `part-*-of-64` subparts for
  the remaining offsets. The root has `_SUCCESS`, `summary.json`, and `1493` tar shards
  under `part-*/shard-*.tar`; a schema spot-check showed paired `.jpg`/`.json` image
  samples. Do **not** switch training configs to this alias: same-zone 50k online-buffer
  sanity did not beat the v1 baseline.
- The v2 build took much longer than expected for operational reasons, not because the
  final dataset was huge: two spot TPUs disappeared mid-build, dirty idle entries made
  xibo aliases unreliable, `gcloud storage ls .../_SUCCESS` could hang without a timeout,
  and xibo SSH returned false 255s even while direct SSH to worker0 was healthy. The
  launcher now supports `SUCCESS_CHECK_TIMEOUT`, `MAX_PARALLEL`, `SHARD_STRIDE`, and
  `SHARD_OFFSETS`; use these instead of hand-editing loops when resuming partial builds.
- The same-zone v2 sanity run
  `llava-v2-buf125k-us-east5-rank0-stream-sim-2k-1worker-20260602`, W&B `8sa4ql3z`,
  used `world=8`, `rank0`, `num_workers=1`, `item_shuffle_size.llava_ov15=50000`,
  v2 LLaVA plus original us-east5 VQAv2/GQA. Result: first 500-step mean
  `valid_tokens_per_sample=80.974`, last 500-step mean `77.758`, fitted slope
  `-2.11 tokens/sample/1k steps`, LLaVA fraction mean `0.943`. This is still a
  downward drift, so a 5x larger offline image-level buffer is not the fix.
- Be careful when running the stream simulator as a sanity check. The current
  token-only preprocessing path supports only `llava_ov15`, `vqav2`, and `gqa`; running
  it on the expanded current SFT mix with OKVQA/AOKVQA/OCRVQA/TextCaps/etc. can spin
  while filtering unsupported samples. Use explicit `--override-roots`,
  `--override-types`, and `--override-mix-weights` for comparable audits, or extend the
  simulator before auditing the full mix.
- A 50k online buffer sanity has high startup and memory cost because each worker must
  fill a raw image-level buffer containing JPEG bytes before yielding samples. A
  `num_workers=4` sanity reached about `31GB` RSS and still had no first bin after
  12 minutes because it needed four 50k buffers before step 100. For quick diagnostics,
  start with `num_workers=1`; for production representativeness, expect long warmup or
  write a metadata-only audit that does not buffer image bytes.

## JAX LLaVA HSDP Notes

- On 2026-05-29, the LLaVA two-stage curriculum failed when resuming stage 2 from a
  stage-1 checkpoint saved on a different topology. The immediate error was Orbax
  restoring with checkpoint-saved sharding after `restore_checkpoint(None, ...)`:
  `sharding passed to deserialization ... Got None`, with old metadata from 64 devices
  and current availability of 32 devices.
- For jit/HSDP code, params-only restore should pass the current params tree as the
  restore target and also pass `ocp.checkpoint_utils.construct_restore_args(target)` to
  `PyTreeRestore`. Passing only `item=target` is insufficient with Orbax 0.11.32: a
  v6e-64 checkpoint resumed on v5p-64 can still fail with
  `sharding passed to deserialization ... Got None`. Do not use target-less restore for
  topology-portable params-only loads.
- HSDP checkpoint saves should follow the `text-jit` pattern: all-gather the sharded
  train state to host with `multihost_utils.process_allgather(..., tiled=True)`, then
  save the host tree. This avoids baking one TPU topology's sharding into checkpoints
  that may later be resumed on v5p/v6e cards with different device layouts.
- The LLaVA HSDP kNN eval failure after stage-1 training was not fundamentally a
  checkpoint-save issue. The kNN path used host-local TFDS loops around HSDP/pjit feature
  extraction, and could trigger `unexpected peer shows up in the launch group with a
  different launch id` when hosts did not execute the same compiled steps. Keep kNN
  feature extraction in the jit/HSDP style, but force a synchronized fixed number of
  pjit steps on every process, pad exhausted local batches, all-gather train/val features,
  then run the JAX KNN computation on every process instead of rank 0 only.
- `jax_llava/debug_remote_knn_eval.sh` is the small remote smoke launcher for this path.
  It accepts `none`/`fresh` as a no-checkpoint eval-only run, creates one launcher-side
  workdir for every worker, and caps TFDS ImageNet reads with `KNN_IMAGES_PER_CLASS`,
  `KNN_VAL_EXAMPLES`, and `KNN_BATCH_SIZE`.
- If remote debug uses a TPU lease name that is not registered in xibo aliases, pass the
  zone explicitly to mount setup, e.g. `tpu mount-disk <tpu> --force --zone=us-central1-a`;
  otherwise xibo `get_zone_pre` can fail even though the TPU exists. Do not trust a broad
  `MOUNTED` label if one worker cannot see `/kmh-nfs-ssd-us-mount`; force mount before
  blaming JAX.
- On 2026-05-29, a tiny LLaVA HSDP kNN smoke passed on
  `kmh-tpuvm-v5p-64-spot-llql5v63k` in `us-central1-a` after force-mounting:
  `KNN_IMAGES_PER_CLASS=8`, `KNN_VAL_EXAMPLES=32`, no `load_from`, TFDS from
  `gs://kmh-gcp-us-central1/tensorflow_datasets`, gathered train features `(8000, 1024)`,
  gathered val features `(32, 1024)`, replicated `KNN-1`, and ended with
  `knn_partial_acc_final=53.125` / `Eval Over.`. This validates plumbing, not model
  quality, because the no-checkpoint path initializes pretrained CLIP/Gemma plus a random
  projector.
- For the original queued LLaVA run, window `7034` did not save a new checkpoint; it
  failed while restoring. The latest complete checkpoint before that failure is window
  `6962`, step `2180`, from
  `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260528_170253_qr1pu0_kmh-tpuvm-v6e-64-dmy-0e3f_us-central1-b__b_lr_ep_eval`,
  and the WandB resume id is `q043krc3`. Requeueing this logical run should set
  `load_from` to that logdir and `wandb_resume_id` to `q043krc3`.
- Do not schedule ImageNet kNN as an online eval for the LLaVA curriculum when the
  image encoder is frozen. In `jax_llava/configs/remote_run_config.yml`, keep stage 1
  kNN disabled and remove `knn_partial` / `knn_full` from stage 2 eval lists as well;
  current already-running staged jobs can be left alone.
- For `jax_llava` SFT detection/refcocog, use the exact shared suffix
  `Output exactly four location tokens, indicating up, left, down, right.` Do not use
  older "Return exactly..." / "Do not output any other text" wording. The order is
  verified against the code: targets are emitted as `ymin, xmin, ymax, xmax`, i.e.
  up, left, down, right. Training `genome_det` and RefCOCOg eval share
  `input_pipeline.format_detection_prompt`, while bbox targets are transformed through
  the same deterministic resize/letterbox transform before conversion to 0..1023 loc
  bins. MMBench eval uses a separate prompt-fitting path and compiled sampler shape:
  default `eval.mmbench_max_txt_len=512`, short "option letter only" suffix, drop hint
  first, then truncate question/options structurally instead of blindly chopping off the
  tail.

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
- When manually editing the external xibo `data.json`, prefer xibo's Python lock
  interface (`utils.data_io.read_and_lock_data`, `write_and_unlock_data`, and
  `release_lock_data` in a `finally` path) instead of bare file reads/writes. This
  avoids races with monitor/xibo scripts that also touch `data.json`.
- If a fallback raw write is truly necessary, preserve `data.json`'s pretty JSON format
  (`indent=4`, `ensure_ascii=False`). A bare `json.dump` rewrites it as one very long
  line, which is valid JSON but bad for inspection and review.
