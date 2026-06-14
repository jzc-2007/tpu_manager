#!/usr/bin/env python3
"""Clean duplicate TPU aliases in xibo data.json.

Default mode is dry-run. Use --apply to write changes. The write path uses
xibo's data lock and writes a backup before modifying data.json. Use
--no-wait-lock with --apply for startup hygiene paths that should skip instead
of blocking when another xibo writer is active.
"""

import argparse
import copy
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime

XIBO_BASE = "/kmh-nfs-ssd-us-mount/code/zhichengjiang/working/xibo_tpu_manager"
LOCAL_REPO = os.path.dirname(os.path.abspath(__file__))
BACKUP_DIR = os.path.join(LOCAL_REPO, "logs", "alias_cleanup")

# Put xibo ahead of this repo so `from utils import data_io` imports xibo's
# lock-aware data_io rather than this monitor repo's read-only shim.
sys.path.insert(0, XIBO_BASE)
from utils import constants as xibo_constants  # noqa: E402
from utils import data_io as xibo_data_io  # noqa: E402

SCRIPT_TMP_ALIAS_RE = re.compile(r"^v\d+(?:e|p)?-\d+-tmp\d*$")
TMP_ALIAS_RE = re.compile(r"^(v\d+(?:e|p)?-\d+)-tmp(\d*)$")


def _alias_sort_key(alias):
    text = str(alias)
    head, sep, tail = text.rpartition("tmp")
    if sep and tail.isdigit():
        return head, int(tail), text
    return text, -1, text


def _canonical_alias_rank(alias):
    """Rank aliases that monitor-style scripts are likely to request.

    Prefer current explicit generations (`v5p`/`v6e`) over legacy shorthand
    (`v5`/`v6`), then natural-sort by tmp suffix.
    """
    alias = str(alias)
    if SCRIPT_TMP_ALIAS_RE.match(alias):
        if alias.startswith(("v5p-", "v6e-")):
            family_rank = 0
        elif alias.startswith(("v5-", "v6-")):
            family_rank = 1
        else:
            family_rank = 2
        return (0, family_rank, _alias_sort_key(alias))
    return (1, 9, _alias_sort_key(alias))


def _dummy_target_for_alias(alias):
    safe = re.sub(r"[^A-Za-z0-9-]+", "-", str(alias)).strip("-")
    return f"kmh-tpuvm-{safe}-do-not-exist"


def _infer_zone_for_tmp_alias(alias):
    match = TMP_ALIAS_RE.match(str(alias))
    if not match:
        return "us-central1-a"
    tpu_type, suffix = match.groups()
    number = int(suffix) if suffix.isdigit() else None
    if number is not None and 201 <= number <= 208:
        if tpu_type.startswith("v6e-"):
            return "asia-northeast1-b"
        return "us-east5-a"
    if number is not None and 51 <= number <= 58:
        return "us-east5-b"
    if tpu_type.startswith("v6e-"):
        return "us-central1-b"
    return "us-central1-a"


def _build_tpu_zone_map(data):
    zones = defaultdict(list)
    for zone, tpus in (data.get("all_tpus") or {}).items():
        for tpu in tpus or []:
            zones[str(tpu)].append(str(zone))
    return zones


def _add_tpu_to_zone(data, tpu_name, zone):
    all_tpus = data.setdefault("all_tpus", {})
    zone_list = all_tpus.setdefault(zone, [])
    if tpu_name not in zone_list:
        zone_list.append(tpu_name)
    # Some older checkouts used zone_tpus; keep it clean if it exists.
    if isinstance(data.get("zone_tpus"), dict):
        zone_tpus = data.setdefault("zone_tpus", {})
        zone_list = zone_tpus.setdefault(zone, [])
        if tpu_name not in zone_list:
            zone_list.append(tpu_name)


def _remove_tpu_from_extra_zones(data, tpu_name, keep_zone):
    for key in ("all_tpus", "zone_tpus"):
        zone_map = data.get(key)
        if not isinstance(zone_map, dict):
            continue
        for zone, tpus in zone_map.items():
            if zone == keep_zone or not isinstance(tpus, list):
                continue
            zone_map[zone] = [t for t in tpus if t != tpu_name]


def _dedupe_zone_maps(data):
    changes = []
    first_zone_for_tpu = {}
    for key in ("all_tpus", "zone_tpus"):
        zone_map = data.get(key)
        if not isinstance(zone_map, dict):
            continue
        first_zone_for_tpu.clear()
        for zone, tpus in list(zone_map.items()):
            if not isinstance(tpus, list):
                continue
            new_list = []
            seen_in_zone = set()
            for tpu in tpus:
                if tpu in seen_in_zone:
                    changes.append({
                        "action": "remove_duplicate_zone_entry",
                        "field": key,
                        "zone": zone,
                        "tpu": tpu,
                        "reason": "duplicate in same zone list",
                    })
                    continue
                seen_in_zone.add(tpu)
                if tpu in first_zone_for_tpu and first_zone_for_tpu[tpu] != zone:
                    changes.append({
                        "action": "remove_duplicate_zone_entry",
                        "field": key,
                        "zone": zone,
                        "tpu": tpu,
                        "reason": f"already present in {first_zone_for_tpu[tpu]}",
                    })
                    continue
                first_zone_for_tpu[tpu] = zone
                new_list.append(tpu)
            zone_map[zone] = new_list
    return changes


def _dedupe_list(values):
    if not isinstance(values, list):
        return values, []
    new_values = []
    seen = set()
    removed = []
    for value in values:
        if value in seen:
            removed.append(value)
            continue
        seen.add(value)
        new_values.append(value)
    return new_values, removed


def clean_data(data):
    """Return (new_data, changes) after applying alias cleanup rules."""
    new_data = copy.deepcopy(data)
    alias_map = new_data.setdefault("tpu_aliases", {})
    tpu_zones = _build_tpu_zone_map(new_data)

    by_tpu = defaultdict(list)
    for alias, target in alias_map.items():
        if target:
            by_tpu[str(target)].append(str(alias))

    changes = []
    for target, aliases in sorted(by_tpu.items()):
        if len(aliases) <= 1:
            continue
        aliases = sorted(aliases, key=_alias_sort_key)
        script_aliases = sorted(
            [alias for alias in aliases if SCRIPT_TMP_ALIAS_RE.match(alias)],
            key=_canonical_alias_rank,
        )
        if script_aliases:
            keep_alias = script_aliases[0]
            for alias in aliases:
                if alias == keep_alias:
                    continue
                if alias in script_aliases:
                    dummy = _dummy_target_for_alias(alias)
                    zone = _infer_zone_for_tmp_alias(alias)
                    alias_map[alias] = dummy
                    _add_tpu_to_zone(new_data, dummy, zone)
                    pre_info = new_data.setdefault("pre_info", {})
                    spot_list = pre_info.setdefault("spot", [])
                    if dummy not in spot_list:
                        spot_list.append(dummy)
                    changes.append({
                        "action": "retarget_alias_to_dummy",
                        "alias": alias,
                        "old_target": target,
                        "new_target": dummy,
                        "zone": zone,
                        "reason": f"extra script tmp alias; kept {keep_alias}",
                    })
                else:
                    alias_map.pop(alias, None)
                    changes.append({
                        "action": "delete_alias",
                        "alias": alias,
                        "old_target": target,
                        "reason": f"non-script alias duplicate; kept {keep_alias}",
                    })
            # Keep the original target in only one zone if it was duplicated.
            zones = tpu_zones.get(target) or []
            if zones:
                _remove_tpu_from_extra_zones(new_data, target, zones[0])
        else:
            keep_alias = aliases[0]
            for alias in aliases[1:]:
                alias_map.pop(alias, None)
                changes.append({
                    "action": "delete_alias",
                    "alias": alias,
                    "old_target": target,
                    "reason": f"no script tmp alias in group; arbitrarily kept {keep_alias}",
                })

    changes.extend(_dedupe_zone_maps(new_data))
    for pre_key in ("spot", "preemptible"):
        pre_info = new_data.get("pre_info") or {}
        values, removed = _dedupe_list(pre_info.get(pre_key))
        if removed:
            pre_info[pre_key] = values
            for value in removed:
                changes.append({
                    "action": "remove_duplicate_pre_info_entry",
                    "field": f"pre_info.{pre_key}",
                    "tpu": value,
                })

    return new_data, changes


def _summarize(changes):
    counts = defaultdict(int)
    for change in changes:
        counts[change["action"]] += 1
    return dict(sorted(counts.items()))


def _print_report(changes, limit=None):
    summary = _summarize(changes)
    print("summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"  total_changes: {len(changes)}")
    print()
    items = changes if limit is None else changes[:limit]
    for change in items:
        print(json.dumps(change, ensure_ascii=False, sort_keys=True))
    if limit is not None and len(changes) > limit:
        print(f"... ({len(changes) - limit} more changes omitted)")


def _write_report(path, changes):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": _summarize(changes),
            "changes": changes,
        }, f, indent=2, ensure_ascii=False)


def _backup_data_json():
    os.makedirs(BACKUP_DIR, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUP_DIR, f"data_before_alias_cleanup_{stamp}.json")
    shutil.copy2(xibo_constants.DATA_PATH, backup_path)
    return backup_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="write cleaned data.json")
    parser.add_argument(
        "--no-wait-lock",
        action="store_true",
        help="with --apply, skip cleanup instead of waiting if data lock is busy",
    )
    parser.add_argument("--limit", type=int, default=120, help="max changes to print; use -1 for all")
    parser.add_argument("--report", help="write full JSON report to this path")
    args = parser.parse_args()

    if args.apply:
        if args.no_wait_lock and hasattr(xibo_data_io, "read_data_if_unlocked"):
            data, locked = xibo_data_io.read_data_if_unlocked()
            if not locked:
                print(f"data.json: {xibo_constants.DATA_PATH}")
                print("mode: APPLY")
                print("data lock is busy; alias cleanup skipped")
                sys.exit(2)
        else:
            data = xibo_data_io.read_and_lock_data()
        locked = True
    else:
        data = xibo_data_io.read_data()
        locked = False

    try:
        new_data, changes = clean_data(data)
        print(f"data.json: {xibo_constants.DATA_PATH}")
        print(f"mode: {'APPLY' if args.apply else 'DRY-RUN'}")
        print()
        print_limit = None if args.limit < 0 else args.limit
        _print_report(changes, limit=print_limit)
        if args.report:
            _write_report(args.report, changes)
            print(f"\nreport written: {args.report}")

        if args.apply:
            if not changes:
                print("\nno changes; data.json left untouched")
                xibo_data_io.release_lock_data()
                locked = False
                return
            backup_path = _backup_data_json()
            xibo_data_io.write_and_unlock_data(new_data)
            locked = False
            print(f"\napplied {len(changes)} changes")
            print(f"backup: {backup_path}")
    finally:
        if locked:
            xibo_data_io.release_lock_data()


if __name__ == "__main__":
    main()
