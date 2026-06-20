#!/bin/bash
# Parallel per-repo verification with hard per-repo timeout (true kill).
# Resumable: pending list excludes already-cached results. Re-run safe.
cd "$(dirname "$0")" || exit 1
PENDING="${1:-cache/pending_ids.txt}"
WORKERS="${2:-12}"
PERREPO_TIMEOUT="${3:-150}"
echo "start: $(wc -l < "$PENDING") repos | workers=$WORKERS | timeout=${PERREPO_TIMEOUT}s"
xargs -P "$WORKERS" -I REPO -a "$PENDING" \
  bash -c 'timeout '"$PERREPO_TIMEOUT"' python3 verify_one.py "$0" >/dev/null 2>&1 || echo "$0" >> cache/failed.log' REPO
echo "done: $(ls cache/verify | wc -l) result files"
