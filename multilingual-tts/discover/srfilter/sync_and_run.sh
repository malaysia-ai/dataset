#!/usr/bin/env bash
# Local driver: rsync the probe job + dataset JSON to the CPU pod, bootstrap, run.
#   ./sync_and_run.sh sync | bootstrap | run | tail | pull | all
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
POD_JSON="$HERE/pod.json"
[ -f "$POD_JSON" ] || { echo "no $POD_JSON — run launch_cpu_pod.py launch first"; exit 1; }
read IP PORT KEY < <(python3 -c "import json;m=json.load(open('$POD_JSON'));print(m['ip'],m['ssh_port'],m.get('ssh_key','~/.ssh/id_rsa'))")
KEY="${KEY/#\~/$HOME}"
SSH="ssh -p $PORT -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o ServerAliveInterval=8"

do_sync() {
    echo "[sync] -> root@$IP:/srfilter"
    $SSH root@$IP "mkdir -p /srfilter"
    rsync -az -e "$SSH" \
        "$HERE/probe_samplerate.py" "$HERE/bootstrap_cpu.sh" "$HERE/run_probe.sh" "$HERE/.env" \
        "$HERE/../new_tts_stt_datasets.json" \
        "root@$IP:/srfilter/"
}
do_bootstrap() { $SSH root@$IP "bash /srfilter/bootstrap_cpu.sh 2>&1 | tee /srfilter/bootstrap.log"; }
do_run() {
    echo "[run] starting probe detached on pod"
    $SSH root@$IP "cd /srfilter && setsid bash run_probe.sh </dev/null >/srfilter/probe.log 2>&1 & disown; echo started"
}
do_tail() { $SSH root@$IP "tail -n 60 -f /srfilter/probe.log"; }
do_pull() {
    echo "[pull] out/ -> $HERE/out"
    rsync -az -e "$SSH" "root@$IP:/srfilter/out/" "$HERE/out/"
}
case "${1:-all}" in
    sync) do_sync ;;
    bootstrap) do_bootstrap ;;
    run) do_run ;;
    tail) do_tail ;;
    pull) do_pull ;;
    all) do_sync; do_bootstrap; do_run ;;
    *) echo "usage: $0 [sync|bootstrap|run|tail|pull|all]"; exit 1 ;;
esac
