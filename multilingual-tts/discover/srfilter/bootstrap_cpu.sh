#!/usr/bin/env bash
# Bootstrap a RunPod CPU pod for the sample-rate probing job. Runs as root.
# Everything under / (container disk). Idempotent enough to re-run.
set -euo pipefail
REPO=/srfilter
VENV=$REPO/.venv
export DEBIAN_FRONTEND=noninteractive

echo "===== [bootstrap] apt deps ====="
apt-get update -y
apt-get install -y --no-install-recommends libsndfile1 ffmpeg git rsync curl ca-certificates || true

echo "===== [bootstrap] uv ====="
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "===== [bootstrap] venv + deps ====="
uv venv "$VENV" --python 3.11
uv pip install --python "$VENV/bin/python" pyarrow soundfile "huggingface_hub>=0.34" hf_xet tqdm

"$VENV/bin/python" - <<'PY'
import pyarrow, soundfile, huggingface_hub
print("deps ok: pyarrow", pyarrow.__version__, "| hub", huggingface_hub.__version__)
PY
echo "===== [bootstrap] done ====="
