#!/usr/bin/env python3
"""Detect real owners and manager-zombie TPU holders for remote user sqa.

The script uses the same record cache as `tou` to find BUSY TPUs whose remote
device holder includes Linux user `sqa`, then SSHes to those TPUs and correlates
the actual accelerator-holding Python processes with xibo `data.json` and local
tmux windows.
"""

import argparse
import base64
import concurrent.futures
import datetime as dt
import json
import os
import re
import signal
import shlex
import subprocess
import sys
import time
from collections import Counter, defaultdict


TPU_DLS_DIR = "/kmh-nfs-ssd-us-mount/code/qiao/work/tpu_dls"
TOU_CACHE = os.path.join(TPU_DLS_DIR, ".tpu_audit_records.json")
WRAP_MASTER = os.path.join(TPU_DLS_DIR, "wrap_master.py")
DATA_JSON = "/kmh-nfs-ssd-us-mount/code/zhichengjiang/working/xibo_tpu_manager/data.json"
PROJECT = "he-vision-group"

REMOTE_USER = "sqa"
DEFAULT_MAX_CACHE_AGE_SECONDS = 600
DEFAULT_REFRESH_TIMEOUT_SECONDS = 300
DEFAULT_MAX_WORKERS = 8
DEFAULT_STALE_LOG_SECONDS = 6 * 3600

TERMINAL_STATUSES = {
    "finished",
    "killed",
    "rerunned",
    "resumed",
    "deleted",
}

WORKDIR_RE = re.compile(r"(?:^|\s)--workdir(?:=|\s+)(\"[^\"]+\"|'[^']+'|\S+)")
ZONE_SUFFIX_RE = re.compile(r"_(us-central1-[abcdf]|us-east5-[ab]|asia-northeast1-b|us-central2-b|europe-west4-a)(?:__|_|$)")

COLORS = {
    "LIVE": "\033[32m",
    "ZOMBIE": "\033[31m",
    "ZOMBIE?": "\033[33m",
    "UNKNOWN": "\033[33m",
    "EMPTY": "\033[36m",
    "ERROR": "\033[31m",
    "MIXED": "\033[35m",
    "NOT_SQA": "\033[36m",
}
RESET = "\033[0m"
USER_COLOR_CODES = ("31", "32", "33", "34", "35", "36", "91", "92", "93", "94", "95", "96")
VERDICT_DISPLAY_ORDER = ("ZOMBIE", "ZOMBIE?", "MIXED", "UNKNOWN", "ERROR", "LIVE", "NOT_SQA", "EMPTY")
VERDICT_SORT_RANK = {name: idx for idx, name in enumerate(VERDICT_DISPLAY_ORDER)}


REMOTE_PROBE = r'''
set +e
host=$(hostname)
pids=$( (sudo lsof -w -t /dev/vfio/* 2>/dev/null; sudo lsof -w -t /dev/accel* 2>/dev/null) | sort -u )
if [ -z "$pids" ]; then
  echo "DZ_NO_PIDS|$host"
  exit 0
fi
for p in $pids; do
  user=$(ps -p "$p" -o user= 2>/dev/null | awk '{$1=$1};1')
  ppid=$(ps -p "$p" -o ppid= 2>/dev/null | awk '{$1=$1};1')
  etimes=$(ps -p "$p" -o etimes= 2>/dev/null | awk '{$1=$1};1')
  etime=$(ps -p "$p" -o etime= 2>/dev/null | awk '{$1=$1};1')
  args=$(ps -p "$p" -o args= -ww 2>/dev/null)
  cwd=$(readlink -f "/proc/$p/cwd" 2>/dev/null || true)
  cwd_b64=$(printf '%s' "$cwd" | base64 | tr -d '\n')
  args_b64=$(printf '%s' "$args" | base64 | tr -d '\n')
  printf 'DZ|%s|%s|%s|%s|%s|%s|%s|%s\n' "$host" "$user" "$p" "$ppid" "$etimes" "$etime" "$cwd_b64" "$args_b64"
done
'''


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print a tou-like owner/zombie report for TPUs held by remote user sqa."
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="refresh tou records before selecting candidate TPUs",
    )
    parser.add_argument(
        "--cache",
        choices=("true", "false"),
        default="true",
        help="true uses tou's cache unless it is stale; false always refreshes",
    )
    parser.add_argument(
        "--max-cache-age",
        type=int,
        default=DEFAULT_MAX_CACHE_AGE_SECONDS,
        help="refresh tou cache when older than this many seconds; use -1 to disable",
    )
    parser.add_argument(
        "--refresh-timeout",
        type=int,
        default=DEFAULT_REFRESH_TIMEOUT_SECONDS,
        help="seconds to wait for tou refresh before falling back to existing cache",
    )
    parser.add_argument(
        "--stale-log-seconds",
        type=int,
        default=DEFAULT_STALE_LOG_SECONDS,
        help="no-data matches older than this are shown as ZOMBIE?",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="parallel TPU SSH probes",
    )
    parser.add_argument(
        "--tpu",
        action="append",
        default=[],
        help="restrict to one TPU name; can be passed multiple times",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI color")
    parser.add_argument("--no-bell", action="store_true", help="do not emit a terminal bell when done")
    parser.add_argument("--verbose", action="store_true", help="print per-workdir group lines")
    return parser.parse_args()


def log_stderr(message):
    print(f"[detect_zombie] {message}", file=sys.stderr)


def ring_bell(args):
    if args.no_bell:
        return
    sys.stderr.write("\a")
    sys.stderr.flush()


def norm_path(path):
    if not path:
        return None
    return os.path.normpath(path.strip())


def shell_result(args, timeout, kill_process_group=False):
    process = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=kill_process_group,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        if kill_process_group:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = process.communicate()
        else:
            process.kill()
            stdout, stderr = process.communicate()
        exc.stdout = stdout
        exc.stderr = stderr
        raise exc
    return subprocess.CompletedProcess(args, process.returncode, stdout, stderr)


def maybe_refresh_tou_cache(args):
    need_refresh = args.refresh or args.cache == "false" or not os.path.exists(TOU_CACHE)
    if not need_refresh and args.max_cache_age >= 0:
        try:
            with open(TOU_CACHE, "r", encoding="utf-8") as f:
                cache = json.load(f)
            age = time.time() - float(cache["ts"])
            need_refresh = age > args.max_cache_age
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            need_refresh = True

    if not need_refresh:
        return

    log_stderr("refreshing tou cache with wrap_master.py --cache false")
    try:
        result = shell_result(
            [sys.executable, WRAP_MASTER, "--cache", "false"],
            timeout=args.refresh_timeout,
            kill_process_group=True,
        )
    except subprocess.TimeoutExpired as exc:
        if os.path.exists(TOU_CACHE):
            log_stderr(
                f"WARNING: tou refresh timed out after {args.refresh_timeout}s; "
                "falling back to existing cache"
            )
            if exc.stdout:
                sys.stderr.write(str(exc.stdout)[-2000:])
            if exc.stderr:
                sys.stderr.write(str(exc.stderr)[-2000:])
            return
        raise RuntimeError("tou refresh timed out and no cache is available") from exc
    if result.returncode != 0:
        sys.stderr.write(result.stdout[-4000:])
        sys.stderr.write(result.stderr[-4000:])
        if os.path.exists(TOU_CACHE):
            log_stderr("WARNING: tou refresh failed; falling back to existing cache")
            return
        raise RuntimeError("failed to refresh tou cache and no cache is available")


def read_tou_records(args):
    maybe_refresh_tou_cache(args)
    with open(TOU_CACHE, "r", encoding="utf-8") as f:
        cache = json.load(f)
    records = list(cache.get("records") or [])
    cache_age = time.time() - float(cache.get("ts", 0))

    selected = []
    tpu_filter = set(args.tpu or [])
    for rec in records:
        if tpu_filter and rec.get("name") not in tpu_filter:
            continue
        if rec.get("status") != "BUSY":
            continue
        users = rec.get("users") or []
        if REMOTE_USER not in users:
            continue
        selected.append(rec)

    selected.sort(key=lambda r: (str(r.get("prefix") or ""), str(r.get("name") or "")))
    return cache_age, selected


def tmux_windows(session_name):
    if not session_name:
        return set()
    try:
        result = shell_result(
            ["tmux", "list-windows", "-t", str(session_name), "-F", "#{window_index}"],
            timeout=5,
        )
    except Exception:
        return set()
    if result.returncode != 0:
        return set()

    windows = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            windows.add(int(line))
        except ValueError:
            continue
    return windows


def safe_window_id(job):
    try:
        return int(job.get("windows_id"))
    except (TypeError, ValueError):
        return None


def load_manager_state():
    with open(DATA_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    users = data.get("users") or {}
    windows_by_user = {}
    for user, user_data in users.items():
        windows_by_user[user] = tmux_windows(user_data.get("tmux_name") or user)

    jobs_by_logdir = defaultdict(list)
    jobs_by_tpu = defaultdict(list)
    for user, user_data in users.items():
        for job in user_data.get("job_data", []) or []:
            window_id = safe_window_id(job)
            status = str(job.get("status") or "").lower()
            extra_msgs = job.get("extra_msgs") or {}
            record = {
                "user": user,
                "window": window_id,
                "status": status or "none",
                "tpu": job.get("tpu"),
                "log_dir": norm_path(job.get("log_dir")),
                "stage_dir": norm_path(job.get("stage_dir")),
                "tmux_active": window_id in windows_by_user.get(user, set()) if window_id is not None else False,
                "child": extra_msgs.get("child"),
                "father": extra_msgs.get("father"),
                "job_tags": job.get("job_tags") or "",
            }
            if record["log_dir"]:
                jobs_by_logdir[record["log_dir"]].append(record)
            if record["tpu"]:
                jobs_by_tpu[record["tpu"]].append(record)

    return {
        "jobs_by_logdir": dict(jobs_by_logdir),
        "jobs_by_tpu": dict(jobs_by_tpu),
    }


def is_job_active(job):
    status = job.get("status") or ""
    if status in TERMINAL_STATUSES:
        return False
    if job.get("child") is not None:
        return False
    return bool(job.get("tmux_active"))


def extract_workdir(args):
    if not args:
        return None
    match = WORKDIR_RE.search(args)
    if not match:
        return None
    value = match.group(1).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return norm_path(value)


def decode_b64(value):
    if not value:
        return ""
    try:
        return base64.b64decode(value.encode("ascii"), validate=False).decode("utf-8", errors="replace")
    except Exception:
        return ""


def probe_tpu(record):
    name = record["name"]
    zone = record["zone"]
    cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        name,
        f"--zone={zone}",
        f"--project={PROJECT}",
        "--worker=all",
        "--ssh-flag=-n",
        "--command",
        "bash -lc " + shlex.quote(REMOTE_PROBE),
    ]

    started = time.time()
    try:
        result = shell_result(cmd, timeout=90)
    except subprocess.TimeoutExpired as exc:
        return {
            "tpu": name,
            "zone": zone,
            "prefix": record.get("prefix") or "__OTHER__",
            "tou_users": record.get("users") or [],
            "probe_error": f"timeout after {exc.timeout}s",
            "holders": [],
            "probe_seconds": time.time() - started,
        }
    except Exception as exc:
        return {
            "tpu": name,
            "zone": zone,
            "prefix": record.get("prefix") or "__OTHER__",
            "tou_users": record.get("users") or [],
            "probe_error": str(exc),
            "holders": [],
            "probe_seconds": time.time() - started,
        }

    holders = []
    no_pids_workers = 0
    merged = "\n".join([result.stdout or "", result.stderr or ""])
    for line in merged.splitlines():
        if line.startswith("DZ_NO_PIDS|"):
            no_pids_workers += 1
            continue
        if not line.startswith("DZ|"):
            continue
        parts = line.split("|", 8)
        if len(parts) != 9:
            continue
        _, host, user, pid, ppid, etimes, etime, cwd_b64, args_b64 = parts
        args_text = decode_b64(args_b64)
        cwd = norm_path(decode_b64(cwd_b64))
        workdir = extract_workdir(args_text)
        holders.append(
            {
                "host": host,
                "user": user,
                "pid": pid,
                "ppid": ppid,
                "etimes": int(etimes) if etimes.isdigit() else None,
                "etime": etime,
                "cwd": cwd,
                "args": args_text,
                "workdir": workdir,
            }
        )

    probe_error = None
    if result.returncode != 0 and not holders:
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-1:]
        probe_error = tail[0] if tail else f"ssh returncode={result.returncode}"

    return {
        "tpu": name,
        "zone": zone,
        "prefix": record.get("prefix") or "__OTHER__",
        "tou_users": record.get("users") or [],
        "probe_error": probe_error,
        "holders": holders,
        "no_pids_workers": no_pids_workers,
        "probe_seconds": time.time() - started,
    }


def log_age_seconds(workdir):
    if not workdir:
        return None
    log_path = os.path.join(workdir, "output.log")
    try:
        return max(0, time.time() - os.path.getmtime(log_path))
    except OSError:
        return None


def format_age(seconds):
    if seconds is None:
        return "?"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 48:
        return f"{hours}h"
    return f"{hours // 24}d"


def owner_hint_from_path(workdir):
    if not workdir:
        return "unknown"
    logs_prefix = "/kmh-nfs-ssd-us-mount/logs/"
    if workdir.startswith(logs_prefix):
        tail = workdir[len(logs_prefix):]
        parts = tail.split("/")
        if len(parts) >= 2:
            if parts[0] == "jzc" and parts[1] == "xibo_manager_for_agents":
                return "jzc"
            if parts[0] != REMOTE_USER:
                return parts[0]
            return f"log:{parts[0]}/{parts[1]}"
        if parts:
            return parts[0] if parts[0] != REMOTE_USER else f"log:{parts[0]}"
    marker = "/staging/"
    if marker in workdir:
        tail = workdir.split(marker, 1)[1]
        parts = tail.split("/")
        if parts:
            return f"path:{parts[0]}"
    return "unknown"


def short_workdir(workdir, max_len=110):
    if not workdir:
        return "no-workdir"
    logs_prefix = "/kmh-nfs-ssd-us-mount/logs/"
    if workdir.startswith(logs_prefix):
        text = workdir[len(logs_prefix):]
    elif "/staging/" in workdir:
        text = workdir.split("/staging/", 1)[1]
    else:
        text = workdir
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def job_ref(job):
    status = job.get("status") or "none"
    active = "tmux" if job.get("tmux_active") else "no-tmux"
    child = f" child={job['child']}" if job.get("child") is not None else ""
    return f"{job['user']}:{job.get('window')}/{status}/{active}{child}"


def classify_group(tpu_result, workdir, holders, manager_state, stale_log_seconds):
    holder_users = sorted({h["user"] for h in holders if h.get("user")})
    if REMOTE_USER not in holder_users:
        owners = holder_users or [owner_hint_from_path(workdir)]
        return {
            "verdict": "NOT_SQA",
            "owners": owners,
            "workdir": workdir,
            "workdir_short": short_workdir(workdir),
            "holders": holders,
            "holder_users": holder_users,
            "pids": sorted({h["pid"] for h in holders if h.get("pid")}),
            "log_age": log_age_seconds(workdir),
            "reason": f"tou cache listed {REMOTE_USER}, but fresh holder_users={holder_users}",
            "jobs": [],
        }

    jobs = manager_state["jobs_by_logdir"].get(workdir, []) if workdir else []
    active_jobs = [job for job in jobs if is_job_active(job)]
    age = log_age_seconds(workdir)

    if active_jobs:
        verdict = "LIVE"
        owners = sorted({job["user"] for job in active_jobs})
        reason = "active " + ",".join(job_ref(job) for job in active_jobs[:3])
    elif jobs:
        verdict = "ZOMBIE"
        owners = sorted({job["user"] for job in jobs})
        reason = "inactive " + ",".join(job_ref(job) for job in jobs[:3])
    else:
        owners = [owner_hint_from_path(workdir)]
        if age is not None and age >= stale_log_seconds:
            verdict = "ZOMBIE?"
            reason = f"no data.json match, stale log_age={format_age(age)}"
        else:
            verdict = "UNKNOWN"
            if age is None:
                reason = "no data.json match, no local output.log"
            else:
                reason = f"no data.json match, recent log_age={format_age(age)}"

    return {
        "verdict": verdict,
        "owners": owners,
        "workdir": workdir,
        "workdir_short": short_workdir(workdir),
        "holders": holders,
        "holder_users": holder_users,
        "pids": sorted({h["pid"] for h in holders if h.get("pid")}),
        "log_age": age,
        "reason": reason,
        "jobs": jobs,
    }


def classify_tpu_result(tpu_result, manager_state, stale_log_seconds):
    if tpu_result.get("probe_error"):
        tpu_result["verdict"] = "ERROR"
        tpu_result["groups"] = []
        return tpu_result

    holders = tpu_result.get("holders") or []
    if not holders:
        tpu_result["verdict"] = "EMPTY"
        tpu_result["groups"] = []
        return tpu_result

    groups_by_workdir = defaultdict(list)
    for holder in holders:
        workdir = holder.get("workdir") or holder.get("cwd") or "unknown"
        groups_by_workdir[norm_path(workdir) if workdir != "unknown" else None].append(holder)

    groups = [
        classify_group(tpu_result, workdir, group_holders, manager_state, stale_log_seconds)
        for workdir, group_holders in sorted(groups_by_workdir.items(), key=lambda item: str(item[0]))
    ]
    tpu_result["groups"] = groups

    verdicts = {g["verdict"] for g in groups}
    if len(verdicts) == 1:
        verdict = next(iter(verdicts))
    elif verdicts & {"ZOMBIE", "ZOMBIE?"}:
        verdict = "MIXED"
    elif verdicts & {"UNKNOWN", "NOT_SQA"}:
        verdict = "MIXED"
    else:
        verdict = "LIVE"

    tpu_result["verdict"] = verdict
    return tpu_result


def summarize_results(results):
    by_prefix = defaultdict(list)
    for result in results:
        by_prefix[result.get("prefix") or "__OTHER__"].append(result)
    return by_prefix


def timestamp_prefix():
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    return f"{now} [INFO] "


def maybe_color(text, verdict, no_color):
    if no_color:
        return text
    color = COLORS.get(verdict)
    return f"{color}{text}{RESET}" if color else text


def user_color(user):
    idx = sum(ord(ch) for ch in str(user)) % len(USER_COLOR_CODES)
    return f"\033[1;{USER_COLOR_CODES[idx]}m"


def color_user(user, no_color):
    user = str(user)
    if no_color:
        return user
    return f"{user_color(user)}{user}{RESET}"


def format_user_list(users, no_color):
    users = [str(u) for u in (users or []) if str(u)]
    if not users:
        return "unknown"
    return ",".join(color_user(u, no_color) for u in users)


def format_remote_user_segment(holder_users, no_color):
    holder_users = sorted({str(u) for u in (holder_users or []) if str(u)})
    if not holder_users or holder_users == [REMOTE_USER]:
        return ""
    return f" remote_user={format_user_list(holder_users, no_color)}"


def format_pid_list(pids, max_items=6):
    pids = [str(p) for p in (pids or []) if str(p)]
    if len(pids) <= max_items:
        return ",".join(pids)
    return ",".join(pids[:max_items]) + f",+{len(pids) - max_items}"


def format_count_text(counts, no_color):
    parts = []
    for verdict in VERDICT_DISPLAY_ORDER:
        count = counts.get(verdict, 0)
        if count:
            parts.append(maybe_color(f"{verdict.lower()} {count}", verdict, no_color))
    return ", ".join(parts)


def unique_owners(groups):
    owners = []
    for group in groups:
        owners.extend(group.get("owners") or [])
    return sorted(set(owners))


def format_tpu_line(result, no_color):
    verdict = result["verdict"]
    label = result.get("prefix") or "__OTHER__"
    label = "OTHER" if label == "__OTHER__" else label
    tag = maybe_color(f"[{verdict}]", verdict, no_color)

    if verdict == "ERROR":
        return (
            f"[{label}] {tag} {result['tpu']} ({result['zone']}) "
            f"probe_error={result.get('probe_error')}"
        )
    if verdict == "EMPTY":
        return (
            f"[{label}] {tag} {result['tpu']} ({result['zone']}) "
            f"remote=NO_ACCEL_PIDS"
        )

    groups = result.get("groups") or []
    owners = unique_owners(groups)
    holder_users = sorted({u for g in groups for u in g.get("holder_users", [])})
    n_holders = sum(len(g.get("holders", [])) for g in groups)
    zombie_groups = sum(1 for g in groups if g["verdict"] in {"ZOMBIE", "ZOMBIE?"})

    first = groups[0] if groups else {}
    workdir = first.get("workdir_short", "no-workdir")
    reason = first.get("reason", "")
    if len(groups) > 1:
        reason = f"{len(groups)} groups, zombie_groups={zombie_groups}"

    return (
        f"[{label}] {tag} {result['tpu']} ({result['zone']}) "
        f"holder={format_user_list(owners, no_color)}"
        f"{format_remote_user_segment(holder_users, no_color)} "
        f"procs={n_holders} workdir={workdir} reason={reason}"
    )


def format_group_line(result, group, no_color):
    verdict = group["verdict"]
    label = result.get("prefix") or "__OTHER__"
    label = "OTHER" if label == "__OTHER__" else label
    tag = maybe_color(f"[{verdict}]", verdict, no_color)
    holder_users = group.get("holder_users", [])
    pids = format_pid_list(group.get("pids", []))
    return (
        f"[{label}]    {tag} holder={format_user_list(group.get('owners', []), no_color)}"
        f"{format_remote_user_segment(holder_users, no_color)} "
        f"procs={len(group.get('holders', []))} pids={pids} "
        f"log_age={format_age(group.get('log_age'))} workdir={group.get('workdir_short')} "
        f"reason={group.get('reason')}"
    )


def print_text_report(results, cache_age, args):
    prefix = timestamp_prefix()
    by_prefix = summarize_results(results)
    ordered_prefixes = ["llq", "keya", "dmy", "gzy", "kangyang"]
    if any((r.get("prefix") or "__OTHER__") == "__OTHER__" for r in results):
        ordered_prefixes.append("__OTHER__")
    for p in sorted(by_prefix):
        if p not in ordered_prefixes:
            ordered_prefixes.append(p)

    print(prefix + "========== SQA HOLDER / ZOMBIE SUMMARY ==========")
    print(prefix + f"[CACHE] tou records age={format_age(cache_age)}; remote probes are fresh")

    total = 0
    totals = Counter()
    first_group = True
    for p in ordered_prefixes:
        items = by_prefix.get(p, [])
        if not items:
            continue
        if not first_group:
            print(prefix + "-------")
        first_group = False
        total += len(items)
        counts = Counter(item["verdict"] for item in items)
        totals.update(counts)
        label = "OTHER" if p == "__OTHER__" else p
        count_text = format_count_text(counts, args.no_color)
        print(prefix + f"[{label}] total {len(items)}, {count_text or 'none'}")
        for result in sorted(items, key=lambda r: (VERDICT_SORT_RANK.get(r["verdict"], 99), r["tpu"])):
            print(prefix + format_tpu_line(result, args.no_color))
            if args.verbose or len(result.get("groups") or []) > 1:
                for group in result.get("groups") or []:
                    print(prefix + format_group_line(result, group, args.no_color))

    print(prefix + "-------")
    count_text = format_count_text(totals, args.no_color)
    print(prefix + f"[ALL] total {total}, {count_text or 'none'}")


def main():
    args = parse_args()
    cache_age, candidates = read_tou_records(args)
    if not candidates:
        if args.json:
            print(json.dumps({"cache_age": cache_age, "results": []}, indent=2))
        else:
            print(timestamp_prefix() + "No BUSY TPU candidates found in tou records.")
        ring_bell(args)
        return 0

    log_stderr(f"probing {len(candidates)} BUSY TPUs (max_workers={args.max_workers})")
    manager_state = load_manager_state()

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        future_to_record = {executor.submit(probe_tpu, record): record for record in candidates}
        for future in concurrent.futures.as_completed(future_to_record):
            raw = future.result()
            results.append(classify_tpu_result(raw, manager_state, args.stale_log_seconds))

    results.sort(key=lambda r: (str(r.get("prefix") or ""), str(r.get("tpu") or "")))

    if args.json:
        print(json.dumps({"cache_age": cache_age, "results": results}, indent=2, sort_keys=True))
    else:
        print_text_report(results, cache_age, args)
    ring_bell(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
