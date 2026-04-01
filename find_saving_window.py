#!/usr/bin/env python3
"""
find_saving_window.py <window_id>

Starting from the given window, walk up the father chain (via extra_msgs.father)
to find the most recent window whose output.log contains a "saving" line.
Prints the window number on success; exits with non-zero status on failure.
"""
import sys
import json
import os

DATA_PATH = "/kmh-nfs-ssd-us-mount/code/zhichengjiang/working/xibo_tpu_manager/data.json"


def _load_all_jobs():
    jobs = {}
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    for udata in data["users"].values():
        for job in udata.get("job_data", []):
            wid = job.get("windows_id")
            if wid is not None:
                jobs[int(wid)] = job
    return jobs


def _has_saving(log_dir):
    if not log_dir:
        return False
    log_path = os.path.join(log_dir, "output.log")
    if not os.path.isfile(log_path):
        return False
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "saving" in line.lower():
                    return True
    except Exception:
        pass
    return False


def main():
    if len(sys.argv) < 2:
        print("Usage: find_saving_window.py <window_id>", file=sys.stderr)
        sys.exit(1)

    try:
        start_id = int(sys.argv[1])
    except ValueError:
        print(f"[find_saving] invalid window id: {sys.argv[1]}", file=sys.stderr)
        sys.exit(1)

    all_jobs = _load_all_jobs()
    visited = set()
    cur_id = start_id

    while True:
        if cur_id in visited:
            print(f"[find_saving] cycle detected at window {cur_id}", file=sys.stderr)
            sys.exit(1)
        visited.add(cur_id)

        job = all_jobs.get(cur_id)
        if job is None:
            print(f"[find_saving] window {cur_id} not found in data or legacy", file=sys.stderr)
            sys.exit(1)

        log_dir = job.get("log_dir")
        if _has_saving(log_dir):
            print(cur_id)
            return

        father = job.get("extra_msgs", {}).get("father")
        if father is None:
            print(
                f"[find_saving] no 'saving' found anywhere in chain; "
                f"reached root window {cur_id} (no father), log_dir={log_dir}",
                file=sys.stderr,
            )
            sys.exit(1)

        cur_id = int(father)


if __name__ == "__main__":
    main()
