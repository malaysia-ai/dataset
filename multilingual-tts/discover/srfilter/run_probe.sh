#!/usr/bin/env bash
# Run the sample-rate probe ON THE POD. Reads HF_TOKEN from .env (login => speed +
# no anonymous rate-limit + gated access). Multiprocessing across all candidates.
set -euo pipefail
REPO=/srfilter
VENV=$REPO/.venv
cd "$REPO"
set -a; [ -f "$REPO/.env" ] && source "$REPO/.env"; set +a
export HF_HOME=/hf_cache
export HF_XET_HIGH_PERFORMANCE=1

WORKERS=${WORKERS:-48}
SAMPLES=${SAMPLES:-16}
TARGET=${TARGET:-44000}
BUCKETS=${BUCKETS:-datasets,likely,gated}
mkdir -p /hf_cache "$REPO/out"

PASSES=${PASSES:-4}
echo "===== [run] probing (workers=$WORKERS samples=$SAMPLES target=$TARGET buckets=$BUCKETS passes=$PASSES) ====="
# Multiple resume passes: a segfaulting worker breaks its chunk's pool; each pass
# resumes (skips done) and mops up chunks lost to the previous pass's crashes.
for pass in $(seq 1 "$PASSES"); do
    echo "===== [run] pass $pass/$PASSES ====="
    "$VENV/bin/python" probe_samplerate.py \
        --json new_tts_stt_datasets.json \
        --out-dir out --target "$TARGET" --samples "$SAMPLES" \
        --workers "$WORKERS" --buckets "$BUCKETS" || echo "[run] pass $pass exited non-zero (continuing)"
done
echo "===== [run] all passes done ====="
