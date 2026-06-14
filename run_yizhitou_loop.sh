#!/usr/bin/env bash
set -u

TOU_CMD=(python /kmh-nfs-ssd-us-mount/code/qiao/work/tpu_dls/wrap_master.py)
EVENT_FILE="${YIZHITOU_EVENT_FILE:-/tmp/yizhitou_refresh_sqa.event}"
INTERVAL_SECONDS="${YIZHITOU_INTERVAL_SECONDS:-60}"

while true; do
  if "${TOU_CMD[@]}" --cache false; then
    mkdir -p "$(dirname "$EVENT_FILE")" 2>/dev/null || true
    date +%s > "$EVENT_FILE"
  fi
  sleep "$INTERVAL_SECONDS"
done
