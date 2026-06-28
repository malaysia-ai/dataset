#!/usr/bin/env python3
"""Probe the NATIVE sample rate of HF audio datasets and filter those >= target.

Why re-probe: new_tts_stt_datasets.json records `samplerate` from a SINGLE sample
of the smallest shard (unreliable for mixed-rate datasets), and the 70 likely / 341
gated candidates have no samplerate at all. Here we decode SEVERAL audio samples per
repo (mode sample rate) and also probe the gated bucket (with the HF token).

Method (cheap): read ONLY the first row group of the smallest parquet shard over
HfFileSystem byte-range (no full download), decode up to --samples audio cells with
soundfile, take the MODE sample rate. HF login (HF_TOKEN) avoids anonymous rate
limits (the real bottleneck at ~1.6k repos) + unlocks accepted gated repos.

Multiprocessing across repos (I/O-bound). Outputs JSONL of every probe + a markdown
report of the >= target datasets.

Usage:
  python probe_samplerate.py --json new_tts_stt_datasets.json --out-dir out \
      --target 44000 --samples 16 --workers 48 [--buckets datasets,likely,gated] [--limit N]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import signal
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

# Per-item wall-clock cap inside each worker: a hung HfFileSystem byte-range read
# (no fsspec timeout) would otherwise stall its whole chunk forever (seen as a
# stall at NPROC=few). SIGALRM interrupts the blocked socket read -> TimeoutError.
ITEM_TIMEOUT = int(os.environ.get("PROBE_ITEM_TIMEOUT", "45"))


def _alarm(signum, frame):
    raise TimeoutError("probe item timeout")

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download  # noqa: F401

# Column-name conventions mined by the original discovery (verify_one.py).
AUDIO_NAMES = {"audio", "audio_filename", "path", "wav_path", "target_wav",
               "reference_audio", "file_name", "audio_file", "wav_filename",
               "wav", "normalized_wav", "filename", "mp3", "flac", "speech",
               "audio_path", "file", "filepath", "audio_bytes"}

BUCKET_KEYS = {
    "datasets": "datasets",
    "likely": "likely_suitable_audio_not_byte_verified",
    "gated": "gated_candidates_need_license_acceptance",
}


def is_audio_struct(field) -> bool:
    t = field.type
    if pa.types.is_struct(t):
        names = {t.field(i).name for i in range(t.num_fields)}
        return "bytes" in names or "path" in names
    return False


def decode_sr(val):
    """Return (samplerate, duration_s) for one audio cell, or (None, None)."""
    import soundfile as sf
    try:
        data = None
        if isinstance(val, dict):
            if val.get("bytes"):
                data = val["bytes"]
            elif val.get("path") and isinstance(val["path"], str) and os.path.exists(val["path"]):
                data = open(val["path"], "rb").read()
        elif isinstance(val, (bytes, bytearray)):
            data = bytes(val)
        if not data:
            return None, None
        with sf.SoundFile(io.BytesIO(data)) as f:
            return int(f.samplerate), len(f) / float(f.samplerate)
    except Exception:
        return None, None


def rank_shard(p: str):
    pl = p.lower()
    return (0 if "test" in pl else 1 if ("valid" in pl or "dev" in pl) else 2,
            0 if "00000-of-" in pl else 1, len(p))


def pick_shard(api: HfApi, repo: str, token: str | None) -> str | None:
    files = api.list_repo_files(repo, repo_type="dataset", token=token)
    parquets = [f for f in files if f.endswith(".parquet")]
    if not parquets:
        return None
    sizes = {}
    if len(parquets) <= 100:
        try:
            for inf in api.get_paths_info(repo, parquets, repo_type="dataset", token=token):
                sizes[inf.path] = getattr(inf, "size", None) or (1 << 62)
        except Exception:
            pass
    if sizes:
        return sorted(parquets, key=lambda p: (sizes.get(p, 1 << 62), rank_shard(p)))[0]
    return sorted(parquets, key=rank_shard)[0]


def probe_one(args_tuple):
    entry, n_samples, token = args_tuple
    repo = entry["id"]
    out = {
        "id": repo, "mode_sr": None, "sr_counts": {}, "n_decoded": 0,
        "duration_s": None, "audio_col": entry.get("audio_col"),
        "shard": entry.get("verified_shard"), "bucket": entry.get("_bucket"),
        "language": entry.get("language"), "downloads": entry.get("downloads"),
        "likes": entry.get("likes"), "json_samplerate": entry.get("samplerate"),
        "error": None,
    }
    try:
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(ITEM_TIMEOUT)
    except Exception:  # noqa: BLE001 — not main thread / unsupported; proceed
        pass
    try:
        api = HfApi(token=token)
        shard = out["shard"] or pick_shard(api, repo, token)
        if not shard:
            out["error"] = "no-parquet"
            return out
        out["shard"] = shard
        fs = HfFileSystem(token=token)
        with fs.open(f"datasets/{repo}/{shard}", "rb") as fh:
            pf = pq.ParquetFile(fh)
            schema = pf.schema_arrow
            audio_col = out["audio_col"]
            if not audio_col or audio_col not in schema.names:
                audio_col = (next((f.name for f in schema if is_audio_struct(f)), None)
                             or next((f.name for f in schema if f.name.lower() in AUDIO_NAMES), None))
            out["audio_col"] = audio_col
            if not audio_col:
                out["error"] = "no-audio-col"
                return out
            # Guard: skip a pathologically large first row-group audio chunk (reading
            # it would OOM the worker). Sum the physical columns under `audio_col`.
            try:
                rg0 = pf.metadata.row_group(0)
                unc = sum((rg0.column(j).total_uncompressed_size or 0)
                          for j in range(rg0.num_columns)
                          if rg0.column(j).path_in_schema.split(".")[0] == audio_col)
                if unc > 1_000_000_000:
                    out["error"] = f"huge-rowgroup-audio:{unc // 10**6}MB"
                    return out
            except Exception:
                pass
            batch = next(pf.iter_batches(batch_size=max(8, n_samples), columns=[audio_col]), None)
            if batch is None:
                out["error"] = "empty-shard"
                return out
            srs, durs = [], []
            for v in pa.Table.from_batches([batch]).column(audio_col).to_pylist()[:n_samples]:
                sr, dur = decode_sr(v)
                if sr:
                    srs.append(sr)
                    if dur:
                        durs.append(dur)
            if not srs:
                out["error"] = "audio-not-decoded"
                return out
            cnt = Counter(srs)
            out["sr_counts"] = {str(k): v for k, v in cnt.items()}
            out["mode_sr"] = cnt.most_common(1)[0][0]
            out["n_decoded"] = len(srs)
            out["duration_s"] = round(sum(durs) / len(durs), 2) if durs else None
    except TimeoutError:
        out["error"] = "timeout"
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}:{str(e)[:140]}"
        low = msg.lower()
        if any(k in low for k in ("403", "gated", "awaiting", "access", "restricted")):
            out["error"] = f"gated/forbidden:{msg[:80]}"
        else:
            out["error"] = msg
    finally:
        try:
            signal.alarm(0)
        except Exception:  # noqa: BLE001
            pass
    return out


def load_candidates(json_path: str, buckets: list[str]) -> list[dict]:
    d = json.load(open(json_path))
    items = []
    seen = set()
    for b in buckets:
        for e in d.get(BUCKET_KEYS[b], []):
            if e["id"] in seen:
                continue
            seen.add(e["id"])
            e = dict(e)
            e["_bucket"] = b
            items.append(e)
    return items


def write_markdown(results: list[dict], target: int, path: str, total: int) -> None:
    keep = [r for r in results if isinstance(r.get("mode_sr"), int) and r["mode_sr"] >= target]
    keep.sort(key=lambda r: (-r["mode_sr"], -(r.get("downloads") or 0)))
    errs = [r for r in results if r.get("error")]
    gated_blocked = [r for r in errs if str(r.get("error", "")).startswith("gated")]
    sr_hist = Counter(r["mode_sr"] for r in results if isinstance(r.get("mode_sr"), int))
    lines = []
    lines.append(f"# HF audio datasets with native sample rate ≥ {target} Hz\n")
    lines.append(f"Probed **{total}** parquet-backed candidates (mode of up to N decoded "
                 f"samples from the smallest shard, via HfFileSystem byte-range). "
                 f"**{len(keep)}** have mode sample rate ≥ {target} Hz.\n")
    lines.append(f"- decoded OK: {sum(1 for r in results if r.get('mode_sr'))}  |  "
                 f"errors: {len(errs)} (gated/forbidden: {len(gated_blocked)})\n")
    lines.append("- sample-rate histogram (mode): " +
                 ", ".join(f"{k}={v}" for k, v in sorted(sr_hist.items(), reverse=True)) + "\n")
    lines.append("\n| # | dataset | mode SR | #samples (sr_counts) | ~dur(s) | bucket | langs | ⬇ downloads | ♥ |\n")
    lines.append("|---|---|---:|---|---:|---|---|---:|---:|\n")
    for i, r in enumerate(keep, 1):
        langs = r.get("language") or []
        langs = ",".join(langs[:6]) + ("…" if len(langs) > 6 else "")
        scnt = ";".join(f"{k}×{v}" for k, v in sorted(r["sr_counts"].items(), key=lambda x: -x[1]))
        lines.append(f"| {i} | [{r['id']}](https://huggingface.co/datasets/{r['id']}) | "
                     f"{r['mode_sr']} | {r['n_decoded']} ({scnt}) | {r.get('duration_s') or ''} | "
                     f"{r.get('bucket')} | {langs} | {r.get('downloads') or 0} | {r.get('likes') or 0} |\n")
    open(path, "w").write("".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="new_tts_stt_datasets.json")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--target", type=int, default=44000)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--buckets", default="datasets,likely,gated")
    ap.add_argument("--limit", type=int, default=0, help="probe only first N (debug)")
    a = ap.parse_args()

    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")  # fast Xet transfer
    token = os.environ.get("HF_TOKEN")
    if token:
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
            print("[hf] logged in with HF_TOKEN", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[hf] login warning: {e}", flush=True)
    else:
        print("[hf] WARNING no HF_TOKEN — anonymous (slow / rate-limited)", flush=True)

    buckets = [b.strip() for b in a.buckets.split(",") if b.strip()]
    cands = load_candidates(a.json, buckets)
    if a.limit:
        cands = cands[: a.limit]
    os.makedirs(a.out_dir, exist_ok=True)
    jsonl_path = os.path.join(a.out_dir, "sr_probe_results.jsonl")

    # RESUME: keep results already written (a worker OOM aborts a pool; we recover).
    results = []
    done_ids = set()
    if os.path.exists(jsonl_path):
        for line in open(jsonl_path):
            try:
                r = json.loads(line)
                results.append(r)
                done_ids.add(r["id"])
            except Exception:
                pass
    todo = [e for e in cands if e["id"] not in done_ids]
    print(f"[probe] {len(cands)} candidates (buckets={buckets}); {len(done_ids)} done, "
          f"{len(todo)} to probe; workers={a.workers} samples={a.samples} target={a.target}", flush=True)

    # Process in fresh per-chunk pools: a BrokenProcessPool (worker OOM-killed on a
    # pathological huge row group) only loses that chunk's in-flight tasks, which a
    # re-run picks up via resume. Keeps memory bounded with modest workers.
    CHUNK = 24  # small: a worker segfault breaks its pool, losing only this chunk's
    t0 = time.time()  # in-flight tasks; resume passes mop up the rest.
    done = len(done_ids)
    with open(jsonl_path, "a") as jf:
        for ci in range(0, len(todo), CHUNK):
            chunk = todo[ci:ci + CHUNK]
            try:
                with ProcessPoolExecutor(max_workers=a.workers, max_tasks_per_child=20) as ex:
                    futs = [ex.submit(probe_one, (e, a.samples, token)) for e in chunk]
                    for fut in as_completed(futs):
                        r = fut.result()
                        results.append(r)
                        jf.write(json.dumps(r, ensure_ascii=False) + "\n")
                        jf.flush()
                        done += 1
                        if done % 50 == 0 or done == len(cands):
                            ge = sum(1 for x in results if isinstance(x.get("mode_sr"), int) and x["mode_sr"] >= a.target)
                            print(f"[probe] {done}/{len(cands)}  ≥{a.target}={ge}  "
                                  f"({(time.time()-t0)/max(1,done-len(done_ids)):.2f}s/ea)", flush=True)
            except Exception as e:  # noqa: BLE001 — pool died; resume covers the gap
                print(f"[probe] WARNING chunk {ci//CHUNK} pool error: {type(e).__name__}: "
                      f"{str(e)[:120]} — continuing (re-run resumes)", flush=True)
                time.sleep(2)

    md_path = os.path.join(a.out_dir, f"datasets_ge_{a.target//1000}k.md")
    write_markdown(results, a.target, md_path, len(cands))
    summary = {
        "total_probed": len(cands), "buckets": buckets, "target": a.target,
        "ge_target": sum(1 for r in results if isinstance(r.get("mode_sr"), int) and r["mode_sr"] >= a.target),
        "decoded_ok": sum(1 for r in results if r.get("mode_sr")),
        "errors": sum(1 for r in results if r.get("error")),
        "gated_blocked": sum(1 for r in results if str(r.get("error", "")).startswith("gated")),
        "sr_hist": {str(k): v for k, v in Counter(
            r["mode_sr"] for r in results if isinstance(r.get("mode_sr"), int)).most_common()},
    }
    json.dump(summary, open(os.path.join(a.out_dir, "sr_summary.json"), "w"), indent=2)
    print(f"[done] {summary}", flush=True)
    print(f"[done] markdown -> {md_path}", flush=True)


if __name__ == "__main__":
    main()
