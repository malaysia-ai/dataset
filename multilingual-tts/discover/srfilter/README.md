# Sample-rate filter: HF audio datasets ≥ 44 kHz

Probe the **native sample rate** of every parquet-backed candidate in
`../new_tts_stt_datasets.json` and keep those whose audio is **≥ 44 kHz** — clean
fullband sources usable as TTS/restoration targets. Runs on a **CPU RunPod pod in
the US** (close to the HF CDN → fast), with HF login + multiprocessing.

## Why a re-probe (vs the JSON's `samplerate`)
The discovery JSON records `samplerate` from a **single** sample of the smallest
shard (unreliable for mixed-rate datasets), and the 70 *likely* / 341 *gated*
candidates have **no** samplerate at all. This re-probes robustly:
- decodes up to **16 audio samples** per repo and takes the **mode** sample rate;
- covers the **gated** bucket too (HF token unlocks accepted licenses);
- reads only the **first row group of the smallest parquet shard** over
  `HfFileSystem` byte-range (no full download), so it's cheap and fast.

## Pod (US, CPU, `/` only — never `/workspace`)
RunPod **caps CPU-pod container disk at 160 GB (gen-3) / 240 GB (gen-5)** — 500 GB
is not possible on a CPU pod's `/` (only a `/workspace` network volume gives more,
which we avoid). We use **160 GB at `/`** (`volumeInGb=0`); the probe only needs
~1 GB (byte-range reads) so this is ample. `computeType=CPU`, `dataCenterIds` =
US-only subset, 16 vCPU, gen-5/gen-3 flavors by availability.

## Run
```bash
# 0. .env (gitignored): RUNPOD_API_KEY + HF_TOKEN (login => speed + gated access)
python3 launch_cpu_pod.py launch --vcpu 16 --disk-gb 160   # US CPU pod, 160GB at /
./sync_and_run.sh sync                                     # rsync probe + dataset JSON
./sync_and_run.sh bootstrap                                # venv: pyarrow, soundfile, hub, hf_xet
# run (detached on pod): WORKERS workers, PASSES resume passes
ssh … "cd /srfilter && WORKERS=16 PASSES=4 setsid bash run_probe.sh >/srfilter/probe.log 2>&1 &"
./sync_and_run.sh tail                                     # follow progress
./sync_and_run.sh pull                                     # pull out/ back here
python3 launch_cpu_pod.py terminate                        # STOP THE BILL (~$0.48/hr)
```

## Robustness
A worker can **segfault** in libsndfile on a malformed audio blob (or hit a huge
row group), which breaks its process pool. `probe_samplerate.py` therefore:
- runs in **small per-chunk pools** (a crash loses only that chunk's in-flight tasks);
- is **resumable** — re-reads `out/sr_probe_results.jsonl` and skips done ids;
- `run_probe.sh` does **N resume passes** so chunks lost to a crash are mopped up;
- **skips** any first row-group whose audio column chunk is > 1 GB (would OOM).

## Outputs (`out/`)
| file | what |
|---|---|
| `sr_probe_results.jsonl` | one line per probed repo (id, mode_sr, sr_counts, dur, error, …) |
| `datasets_ge_44k.md` | **the deliverable**: table of datasets with mode SR ≥ 44 kHz |
| `sr_summary.json` | counts (total, ≥target, decoded, errors, gated-blocked, SR histogram) |

## Files
| file | role |
|---|---|
| `launch_cpu_pod.py` | provision / status / ssh / terminate the US CPU pod (REST, stdlib) |
| `bootstrap_cpu.sh` | apt libsndfile + uv venv (pyarrow, soundfile, huggingface_hub, hf_xet) |
| `probe_samplerate.py` | multiprocessing mode-SR prober + markdown writer |
| `run_probe.sh` | on-pod runner (HF login, N resume passes) |
| `sync_and_run.sh` | local: rsync + bootstrap + run + tail + pull |
