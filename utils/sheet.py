"""Local TPU registry adapter.

The old implementation used the ka Google Spreadsheet as a live state store.
That state is not reliable enough for automatic scheduling, so this module no
longer performs any Google Sheet I/O.  The public function names are kept for
old xibo call sites, but reads are synthesized from local ``data.json`` and
writes are explicit no-ops.
"""

import csv
import re
import subprocess
from io import StringIO

from .constants import *
from .data_io import read_data
from .helpers import display_tpu_information, filter_tpu_information, get_zone_pre_spot


def _type_and_version_from_name(full_name):
    tpu_type = None
    tpu_version = None
    for key, value in NAME_TO_TYPE.items():
        if key in full_name:
            tpu_type = value
    for key, value in NAME_TO_VER.items():
        if key in full_name:
            tpu_version = value
    return tpu_type, tpu_version


def _best_alias_for_tpu(data, full_name):
    aliases = [
        alias
        for alias, mapped in data.get("tpu_aliases", {}).items()
        if mapped == full_name
    ]
    if not aliases:
        return full_name

    def score(alias):
        # Prefer the short temporary aliases used by the scheduling scripts.
        if re.match(r"^v\d+(e|p)?-\d+-tmp\d+$", alias):
            return (0, len(alias), alias)
        if alias.startswith("kmh-tpuvm-"):
            return (2, len(alias), alias)
        return (1, len(alias), alias)

    return sorted(aliases, key=score)[0]


def _job_status_by_tpu(data):
    """Return best-effort local job ownership by TPU from data.json.

    This is intentionally display-only.  It should not be used as the source of
    truth for scheduling; use tou and reservation locks for that.
    """
    status = {}
    for user, user_data in data.get("users", {}).items():
        for job in user_data.get("job_data", []):
            tpu = job.get("tpu")
            if not tpu:
                continue
            job_status = job.get("status")
            if job_status == "running":
                status[tpu] = (
                    "running",
                    user,
                    (job.get("extra_msgs") or {}).get("spreadsheet_notes")
                    or job.get("job_tags")
                    or "",
                )
            elif job_status == "error" and tpu not in status:
                status[tpu] = (
                    "reserved",
                    user,
                    (job.get("extra_msgs") or {}).get("spreadsheet_notes")
                    or job.get("job_tags")
                    or "",
                )
    return status


def read_sheet_info() -> dict:
    """Return spreadsheet-shaped TPU info synthesized from local data.json."""
    data = read_data()
    job_status = _job_status_by_tpu(data)
    preemptible = set((data.get("pre_info") or {}).get("preemptible", []))
    spot = set((data.get("pre_info") or {}).get("spot", []))

    tpu_information = {}
    line = 1
    for zone, tpu_list in (data.get("all_tpus") or {}).items():
        for full_name in tpu_list:
            tpu_type, tpu_version = _type_and_version_from_name(full_name)
            if tpu_type is None or tpu_version is None:
                continue
            running_status, user, user_note = job_status.get(
                full_name, ("free", "free", "")
            )
            tpu_information[full_name] = {
                "zone": zone,
                "pre": full_name in preemptible,
                "spot": full_name in spot,
                "belong": "local",
                "version": tpu_version,
                "type": tpu_type,
                "running_status": running_status,
                "user": user,
                "env": zone,
                "user_note": user_note,
                "script_note": "READY",
                "alias": _best_alias_for_tpu(data, full_name),
                "other_note": "local data.json; ka sheet disabled",
                "line": line,
            }
            line += 1
    return tpu_information


def write_sheet_info(info_to_write):
    """Compatibility no-op for old call sites that used to update ka sheet."""
    alias = (info_to_write or {}).get("alias") or (info_to_write or {}).get("tpu") or "<unknown>"
    print(f"{INFO} write_sheet_info: ka sheet disabled; skipped update for {alias}")
    return True


def read_tpu_info_from_type(args):
    """Read TPU information from local data.json and filter by type/pre flag."""
    type_list = []
    pre_filter = None

    for arg in args:
        if arg in ARG_TO_LIST:
            if isinstance(ARG_TO_LIST[arg], list):
                type_list += ARG_TO_LIST[arg]
            else:
                type_list.append(ARG_TO_LIST[arg])
        if arg in ["-p", "-pre"]:
            pre_filter = True
        if arg in ["-n", "-norm"]:
            pre_filter = False

    if len(type_list) == 0:
        type_list = all_type_list

    tpu_information = read_sheet_info()
    if pre_filter is not None:
        return filter_tpu_information(tpu_information, type=type_list, pre=pre_filter)
    return filter_tpu_information(tpu_information, type=type_list)


def find_tpu_from_type(args):
    """Display local TPU registry rows. Live state comes from tou, not here."""
    style = None
    for arg in args:
        if arg.startswith("style="):
            style = arg.split("=", 1)[1]
            break
    print(f"{WARNING} ka sheet disabled; showing local data.json registry only.")
    information = read_tpu_info_from_type(args)
    return display_tpu_information(information, style=style)


def get_tpu_info_sheet(tpu):
    """Return a spreadsheet-shaped row for a TPU from local data.json."""
    _, _, _, full_name = get_zone_pre_spot(tpu, quiet=True)
    if full_name is None:
        print(f"{FAIL} TPU {tpu} not found in local data.json")
        return None
    tpu_information = read_sheet_info()
    if full_name in tpu_information:
        return tpu_information[full_name]
    print(f"{FAIL} TPU {tpu} not found in local data.json registry")
    return None


def release_tpu(args):
    """Compatibility no-op.

    Old behavior only updated ka sheet. Reservation locks are managed by zhan /
    fang and expire independently, so there is no local sheet state to release.
    """
    assert len(args) in [1, 2], f"release_tpu: {args} is not valid"
    print(f"{INFO} release_tpu: ka sheet disabled; no local action for {args[0]}")
    return True


def set_spreadsheet_notes(tpu, notes):
    print(f"{INFO} set_spreadsheet_notes: ka sheet disabled; skipped {tpu}")
    return True


def add_spreadsheet_notes(tpu, notes):
    print(f"{INFO} add_spreadsheet_notes: ka sheet disabled; skipped {tpu}")
    return True


def keng_tpu(args):
    """Show temporary aliases that are registered locally.

    This no longer knows ka sheet's deleted/preempted marker. Use tou/zombie
    tools for live deleted/preempted state.
    """
    zone_filter = None
    type_filter = None
    for arg in args:
        if "-" in arg and (
            arg.startswith("us-")
            or arg.startswith("asia-")
            or arg.startswith("europe-")
        ):
            zone_filter = arg
        elif arg.startswith("v") and re.match(r"v\d+(e|p)?(-\d+)?$", arg):
            type_filter = re.sub(r"v(\d+)(e|p)(-\d+)?", r"v\1\3", arg)

    tmp_pattern = re.compile(r"^v\d+(e|p)?-\d+-tmp", re.IGNORECASE)
    rows = []
    for full_name, info in read_sheet_info().items():
        alias = info.get("alias", "")
        if not tmp_pattern.match(alias):
            continue
        if zone_filter and info.get("zone") != zone_filter:
            continue
        if type_filter:
            alias_match = re.match(r"(v\d+(e|p)?-\d+)", alias, re.IGNORECASE)
            if not alias_match:
                continue
            alias_type = re.sub(
                r"v(\d+)(e|p)(-\d+)", r"v\1\3", alias_match.group(1)
            )
            if "-" in type_filter:
                if alias_type != type_filter:
                    continue
            elif not alias_type.startswith(type_filter):
                continue
        rows.append((alias, info.get("zone", ""), full_name))

    if not rows:
        print(f"{INFO} No local temporary TPU aliases found matching the criteria")
        return
    print(f"{WARNING} ka sheet disabled; showing local tmp aliases, not deletion state.")
    print(f"{'Alias':<20} {'Zone':<20} {'TPU':<60}")
    print("-" * 105)
    for alias, zone, full_name in sorted(rows):
        print(f"{alias:<20} {zone:<20} {full_name:<60}")


def get_tpu_usage_by_zone_and_type():
    """Use gcloud list command to get TPU chip usage by zone and type."""
    usage_stats = {}
    for zone in ZONE_DICT["all"]:
        cmd = (
            "gcloud compute tpus tpu-vm list "
            f"--zone={zone} --project={PROJECT} "
            "--format='csv(name,acceleratorType,state)'"
        )
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                timeout=60,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode != 0:
                print(f"{WARNING} Failed to list TPUs in zone {zone}: {result.stderr}")
                continue
            lines = result.stdout.strip().splitlines()
            if len(lines) < 2:
                continue
            csv_reader = csv.reader(StringIO("\n".join(lines[1:])))
            for row in csv_reader:
                if len(row) < 3:
                    continue
                acc_type = row[1].strip()
                state = row[2].strip()
                if state.upper() not in ["READY", "CREATING"]:
                    continue
                acc_lower = acc_type.lower()
                if "v6e" in acc_lower or acc_lower.startswith("v6"):
                    tpu_version = "v6"
                elif "v5p" in acc_lower:
                    tpu_version = "v5p"
                elif "v5litepod" in acc_lower or "v5e" in acc_lower:
                    tpu_version = "v5e"
                elif acc_lower.startswith("v5"):
                    tpu_version = "v5"
                elif acc_lower.startswith("v4"):
                    tpu_version = "v4"
                elif acc_lower.startswith("v3"):
                    tpu_version = "v3"
                elif acc_lower.startswith("v2"):
                    tpu_version = "v2"
                else:
                    continue
                numbers = re.findall(r"\d+", acc_type)
                if not numbers:
                    continue
                chip_count = int(numbers[-1])
                key = f"{tpu_version}({zone})"
                usage_stats[key] = usage_stats.get(key, 0) + chip_count
        except subprocess.TimeoutExpired:
            print(f"{WARNING} Timeout listing TPUs in zone {zone}")
        except Exception as e:
            print(f"{WARNING} Error listing TPUs in zone {zone}: {e}")
    return usage_stats


def write_tpu_usage_to_sheet(usage_stats):
    print(f"{INFO} write_tpu_usage_to_sheet: ka sheet disabled; skipped")
    return True


def read_tpu_total_counts_from_sheet():
    """Return local registered chip counts in the legacy structure."""
    counts = {}
    for info in read_sheet_info().values():
        tpu_type = info.get("type", "")
        zone = info.get("zone")
        if not zone:
            continue
        m = re.search(r"(\d+)$", tpu_type)
        if not m:
            continue
        chips = int(m.group(1))
        if tpu_type.startswith("v6"):
            version = "v6"
        elif tpu_type.startswith("v5"):
            version = "v5"
        elif tpu_type.startswith("v4"):
            version = "v4"
        elif tpu_type.startswith("v3"):
            version = "v3"
        elif tpu_type.startswith("v2"):
            version = "v2"
        else:
            continue
        counts.setdefault(version, {})
        counts[version][zone] = counts[version].get(zone, 0) + chips
    return counts
