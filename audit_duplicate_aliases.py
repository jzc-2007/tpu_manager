#!/usr/bin/env python3
"""Print TPU alias collisions in the external xibo data.json.

`data["tpu_aliases"]` maps alias -> real TPU lease name.  `tpu fang` expects
the old TPU behind an alias to have only the normal alias set.  If stale aliases
also point to the same old TPU, fang refuses to swap the alias and mount-disk can
keep using the dead old TPU.
"""

from collections import defaultdict

from utils import data_io
from utils.constants import DATA_PATH


def _alias_sort_key(alias):
    """Keep tmp aliases together and make numeric tmp suffixes sort naturally."""
    text = str(alias)
    head, sep, tail = text.rpartition("tmp")
    if sep and tail.isdigit():
        return head, int(tail), text
    return text, -1, text


def main():
    data = data_io.read_data()
    alias_map = data.get("tpu_aliases") or {}

    by_tpu = defaultdict(list)
    for alias, tpu_name in alias_map.items():
        if not tpu_name:
            continue
        by_tpu[str(tpu_name)].append(str(alias))

    duplicate_groups = [
        (tpu_name, sorted(aliases, key=_alias_sort_key))
        for tpu_name, aliases in by_tpu.items()
        if len(aliases) > 1
    ]
    duplicate_groups.sort(key=lambda item: (-len(item[1]), item[0]))

    involved_alias_count = sum(len(aliases) for _, aliases in duplicate_groups)
    print(f"data.json: {DATA_PATH}")
    print(f"total aliases: {len(alias_map)}")
    print(f"duplicate TPU targets: {len(duplicate_groups)}")
    print(f"aliases in duplicate groups: {involved_alias_count}")
    print()

    for tpu_name, aliases in duplicate_groups:
        # More than two aliases, or a self-alias plus tmp aliases, is the pattern
        # that makes xibo fang abort with "found more than 2 aliases".
        is_self_alias = tpu_name in aliases
        status = "DIRTY" if len(aliases) > 2 or is_self_alias else "DUP"
        markers = []
        if is_self_alias:
            markers.append("self-alias")
        marker_text = f" ({', '.join(markers)})" if markers else ""
        print(f"[{status}] {tpu_name}{marker_text}")
        for alias in aliases:
            print(f"  - {alias}")
        print()


if __name__ == "__main__":
    main()
