#!/usr/bin/env python3
"""
find_saving_window.py <window_id> [user]

Starting from the given window, find the safest window to resume from. The
normal path walks up extra_msgs.father, then a conservative same-logical-job
scan catches stale duplicate branches whose father chain points at an older
checkpoint than a sibling branch.
"""
import json
import os
import re
import shutil
import subprocess
import sys

DATA_PATH = "/kmh-nfs-ssd-us-mount/code/zhichengjiang/working/xibo_tpu_manager/data.json"
CHECKPOINT_DIR_RE = re.compile(r"^checkpoint_(\d+)$")
CHECKPOINT_STEP_RE = re.compile(r"checkpoint_(\d+)")
STEP_RE = re.compile(r"\bstep\s*:?\s*(\d+)\b", re.IGNORECASE)


class CheckpointScanResult:
    def __init__(
        self,
        has_complete_checkpoint,
        latest_saving_step=None,
        latest_saved_step=None,
        latest_saved_path=None,
        cleanup_ok=True,
        messages=None,
    ):
        self.has_complete_checkpoint = has_complete_checkpoint
        self.latest_saving_step = latest_saving_step
        self.latest_saved_step = latest_saved_step
        self.latest_saved_path = latest_saved_path
        self.cleanup_ok = cleanup_ok
        self.messages = messages or []


def _load_all_jobs():
    jobs = {}
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    for user, udata in data["users"].items():
        for job in udata.get("job_data", []):
            wid = job.get("windows_id")
            if wid is not None:
                jobs[(user, int(wid))] = job
    return jobs


def _job_logical_key(job):
    """Return a conservative key for windows that are the same logical job."""
    extra = job.get("extra_msgs") or {}
    note = extra.get("spreadsheet_notes") or job.get("job_tags")
    if not note:
        # job_dir_id alone is too broad in this repo; without a human-readable
        # note/tag, avoid cross-branch guesses.
        return None
    return (str(job.get("job_dir_id")), str(note))


def _same_user_jobs(all_jobs, user):
    return [
        job
        for (job_user, _window_id), job in all_jobs.items()
        if job_user == user
    ]


def _extract_step(line):
    # Prefer an explicit "step N" from the log message over checkpoint_N in a
    # path, because restored/saved paths can contain older checkpoint names.
    step_match = STEP_RE.search(line)
    if step_match:
        return int(step_match.group(1))

    ckpt_match = CHECKPOINT_STEP_RE.search(line)
    if ckpt_match:
        return int(ckpt_match.group(1))

    return None


def _extract_saved_path(line):
    match = re.search(r"\bsaved to\s+(.+?)\s*$", line, re.IGNORECASE)
    if not match:
        return None

    # Current logs end saved-to paths with a sentence period. Strip only outer
    # punctuation/quotes; do not split inside a path.
    return match.group(1).strip().strip("'\"").rstrip(".")


def _is_checkpoint_completion_line(line):
    """Return true only for model checkpoint completion, not sidecar writes."""
    lower = line.lower()
    if "saved to" not in lower:
        return False

    # Dataloader sidecars are written before Orbax completes the checkpoint.
    # They contain checkpoint_N in their path, so treating them as checkpoint
    # completion would make the monitor resume from half-written checkpoints.
    if (
        "dataloader state" in lower
        or "_pending_dataloader_state" in lower
        or "/dataloader_state/" in lower
    ):
        return False

    return "checkpoint at step" in lower or "checkpoint step" in lower


def _normalize_checkpoint_parent(path):
    if not path:
        return None

    path = path.rstrip("/")
    basename = path.rsplit("/", 1)[-1]
    if CHECKPOINT_DIR_RE.match(basename):
        return path.rsplit("/", 1)[0]
    return path


def _candidate_incomplete_checkpoint_paths(log_dir, saved_path, step):
    checkpoint_dir = f"checkpoint_{step}"
    candidates = []
    if log_dir:
        candidates.append(os.path.join(log_dir, checkpoint_dir))

    saved_parent = _normalize_checkpoint_parent(saved_path)
    if saved_parent:
        candidates.append(f"{saved_parent.rstrip('/')}/{checkpoint_dir}")

    deduped = []
    seen = set()
    for path in candidates:
        if path and path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


def _safe_checkpoint_basename(path, expected_step):
    basename = path.rstrip("/").rsplit("/", 1)[-1]
    match = CHECKPOINT_DIR_RE.match(basename)
    return match is not None and int(match.group(1)) == int(expected_step)


def _remove_gs_checkpoint(path):
    commands = [
        ["gsutil", "-m", "rm", "-r", path],
        ["gcloud", "storage", "rm", "-r", path],
    ]
    last_error = ""
    for cmd in commands:
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=300,
            )
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired as exc:
            last_error = f"{cmd[0]} timeout after {exc.timeout}s"
            continue

        merged = f"{result.stdout or ''}\n{result.stderr or ''}"
        lowered = merged.lower()
        if result.returncode == 0:
            return True, f"removed {path}"
        if "no urls matched" in lowered or "not found" in lowered or "matched no objects" in lowered:
            return True, f"{path} not found"
        last_error = merged.strip()

    return False, f"failed to remove {path}: {last_error}"


def _remove_incomplete_checkpoint_paths(paths, step):
    ok = True
    messages = []
    for path in paths:
        if not _safe_checkpoint_basename(path, step):
            ok = False
            messages.append(f"skip unsafe checkpoint cleanup path: {path}")
            continue

        if path.startswith("gs://"):
            removed, message = _remove_gs_checkpoint(path)
            ok = ok and removed
            messages.append(message)
            continue

        if os.path.isdir(path):
            shutil.rmtree(path)
            messages.append(f"removed {path}")
        elif os.path.exists(path):
            ok = False
            messages.append(f"skip non-directory checkpoint cleanup path: {path}")
        else:
            messages.append(f"{path} not found")
    return ok, messages


def _scan_checkpoint_state(log_dir):
    if not log_dir:
        return CheckpointScanResult(False, messages=["missing log_dir"])

    log_path = os.path.join(log_dir, "output.log")
    if not os.path.isfile(log_path):
        return CheckpointScanResult(False, messages=[f"missing output.log at {log_path}"])

    latest_saving_step = None
    latest_saved_step = None
    latest_saved_path = None

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                lower = line.lower()
                if "checkpoint" not in lower:
                    continue

                if "saving" in lower:
                    step = _extract_step(line)
                    if step is not None:
                        latest_saving_step = step

                if _is_checkpoint_completion_line(line):
                    step = _extract_step(line)
                    saved_path = _extract_saved_path(line)
                    if step is not None:
                        latest_saved_step = step
                        latest_saved_path = saved_path or latest_saved_path
    except Exception:
        return CheckpointScanResult(False, messages=[f"failed to read {log_path}"])

    if latest_saved_step is None:
        return CheckpointScanResult(
            False,
            latest_saving_step=latest_saving_step,
            messages=[f"no completed `saved to` checkpoint in {log_path}"],
        )

    messages = [
        f"latest saved checkpoint step={latest_saved_step}"
        + (f" path={latest_saved_path}" if latest_saved_path else "")
    ]

    if latest_saving_step is not None and latest_saving_step > latest_saved_step:
        paths = _candidate_incomplete_checkpoint_paths(log_dir, latest_saved_path, latest_saving_step)
        cleanup_ok, cleanup_messages = _remove_incomplete_checkpoint_paths(paths, latest_saving_step)
        messages.append(
            f"latest saving step={latest_saving_step} is newer than latest saved step={latest_saved_step}; "
            "cleaned incomplete checkpoint candidates"
        )
        messages.extend(cleanup_messages)
        return CheckpointScanResult(
            cleanup_ok,
            latest_saving_step=latest_saving_step,
            latest_saved_step=latest_saved_step,
            latest_saved_path=latest_saved_path,
            cleanup_ok=cleanup_ok,
            messages=messages,
        )

    if latest_saving_step is not None and latest_saving_step != latest_saved_step:
        messages.append(
            f"latest saving step={latest_saving_step} differs from latest saved step={latest_saved_step}; "
            "saved checkpoint is newer, so no cleanup is needed"
        )

    return CheckpointScanResult(
        True,
        latest_saving_step=latest_saving_step,
        latest_saved_step=latest_saved_step,
        latest_saved_path=latest_saved_path,
        messages=messages,
    )


def _find_best_logical_family_checkpoint(all_jobs, user, start_job):
    """Find the highest complete checkpoint in the same conservative job group."""
    key = _job_logical_key(start_job)
    if key is None:
        return None, None

    best_window = None
    best_scan = None
    for job in _same_user_jobs(all_jobs, user):
        if _job_logical_key(job) != key:
            continue
        if job.get("status") == "killed":
            continue

        scan = _scan_checkpoint_state(job.get("log_dir"))
        if not scan.has_complete_checkpoint:
            continue

        if (
            best_scan is None
            or scan.latest_saved_step > best_scan.latest_saved_step
            or (
                scan.latest_saved_step == best_scan.latest_saved_step
                and int(job["windows_id"]) > int(best_window)
            )
        ):
            best_window = int(job["windows_id"])
            best_scan = scan

    return best_window, best_scan


def main():
    if len(sys.argv) < 2:
        print("Usage: find_saving_window.py <window_id> [user]", file=sys.stderr)
        sys.exit(1)

    try:
        start_id = int(sys.argv[1])
    except ValueError:
        print(f"[find_saving] invalid window id: {sys.argv[1]}", file=sys.stderr)
        sys.exit(1)

    user = sys.argv[2] if len(sys.argv) >= 3 else "sqa"
    all_jobs = _load_all_jobs()
    start_job = all_jobs.get((user, start_id))
    if start_job is None:
        print(f"[find_saving] window {start_id} for user {user} not found in data", file=sys.stderr)
        sys.exit(1)

    visited = set()
    cur_id = start_id
    chain_window = None
    chain_scan = None
    root_id = None

    while True:
        if cur_id in visited:
            print(f"[find_saving] cycle detected at window {cur_id}", file=sys.stderr)
            sys.exit(1)
        visited.add(cur_id)

        job = all_jobs.get((user, cur_id))
        if job is None:
            print(f"[find_saving] window {cur_id} for user {user} not found in data", file=sys.stderr)
            sys.exit(1)

        scan = _scan_checkpoint_state(job.get("log_dir"))
        if scan.messages:
            print(f"[find_saving] window {cur_id}: " + " | ".join(scan.messages), file=sys.stderr)

        if scan.has_complete_checkpoint:
            chain_window = cur_id
            chain_scan = scan
            break

        father = job.get("extra_msgs", {}).get("father")
        if father is None:
            root_id = cur_id
            break

        cur_id = int(father)

    family_window, family_scan = _find_best_logical_family_checkpoint(all_jobs, user, start_job)
    if family_scan is not None:
        if chain_scan is None or family_scan.latest_saved_step > chain_scan.latest_saved_step:
            if chain_scan is None:
                print(
                    f"[find_saving] logical family window {family_window} has complete "
                    f"checkpoint step={family_scan.latest_saved_step}; father chain had none",
                    file=sys.stderr,
                )
            else:
                print(
                    f"[find_saving] logical family window {family_window} has newer complete "
                    f"checkpoint step={family_scan.latest_saved_step} than chain window "
                    f"{chain_window} step={chain_scan.latest_saved_step}",
                    file=sys.stderr,
                )
            print(family_window)
            return

    if chain_scan is not None:
        print(chain_window)
        return

    print(
        f"[find_saving] no complete checkpoint found anywhere in chain; "
        f"reached root window {root_id} (no father), log_dir={job.get('log_dir')}",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
