#!/usr/bin/env python3
"""Generic end-to-end prepare pipeline for adding a source dataset to
malaysia-ai/Multilingual-TTS.

Stages (each checkpointed so re-runs skip finished work):
  1. extract  : stream the source HF dataset, decode audio -> mono mp3 (saves
                storage), build rows {audio_filename, text, speaker?}.
  2. speaker  : use the source speaker/gender column if present; otherwise extract
                192-d TitaNet vectors (embedding.py) and greedy faiss-cluster them.
  3. neucodec : audio mp3 -> neucodec token json (convert_neucodec.py).
  4. push     : Dataset.from_list(rows).push_to_hub(<name>) + upload mp3 zip +
                neucodec zip, then write a dataset-level checkpoint.

Usage:
  python3 prepare.py --repo CAiRE/ASCEND --name ASCEND [--config X] \
      [--audio-col A] [--text-col T] [--speaker-col S] [--max-samples N]

Re-run is safe: if checkpoints/<name>.done exists the whole dataset is skipped.
"""
import os, io, sys, json, time, glob, subprocess, traceback
import numpy as np
import click

HF_REPO = "malaysia-ai/Multilingual-TTS"
HERE = os.path.dirname(os.path.abspath(__file__))

AUDIO_NAMES = {"audio", "wav", "speech", "audio_filename", "file", "path",
               "audio_path", "audio_file", "file_name", "filepath"}
TEXT_NAMES = {"text", "transcription", "sentence", "transcript", "normalized_text",
              "raw_transcription", "asr_transcript", "human_transcript", "words",
              "voice_text", "transcript_text", "text_clean", "normalized"}
TEXT_SUBSTR = ("transcript", "text", "sentence", "caption", "normaliz", "utter", "words")
SPEAKER_SUBSTR = ("speaker", "gender", "spk")


def ckpt_dir(workdir):
    d = os.path.join(workdir, "checkpoints")
    os.makedirs(d, exist_ok=True)
    return d


def is_id_name(n):
    n = n.lower()
    return n == "id" or n.endswith("_id") or n.endswith("_idx") or "idx" in n


def detect_cols(features, audio_col, text_col, speaker_col):
    from datasets import Audio
    names = list(features.keys())
    low = {n: n.lower() for n in names}
    if not audio_col:
        audio_col = next((n for n, f in features.items() if isinstance(f, Audio)), None) \
            or next((n for n in names if low[n] in AUDIO_NAMES), None)
    if not text_col:
        def is_str(f):
            return getattr(f, "dtype", None) in ("string", "large_string")
        strings = [n for n in names if is_str(features[n])]
        text_col = next((n for n in strings if low[n] in TEXT_NAMES), None) \
            or next((n for n in strings if not is_id_name(n)
                     and any(k in low[n] for k in TEXT_SUBSTR)), None)
    if not speaker_col:
        speaker_col = next((n for n in names if any(k in low[n] for k in SPEAKER_SUBSTR)), None)
    return audio_col, text_col, speaker_col


def decode_audio(a):
    import soundfile as sf
    data = None
    if isinstance(a, dict):
        if a.get("array") is not None:
            return np.asarray(a["array"], dtype="float32"), int(a["sampling_rate"])
        if a.get("bytes"):
            data = a["bytes"]
        elif a.get("path") and isinstance(a["path"], str) and os.path.exists(a["path"]):
            data = open(a["path"], "rb").read()
    elif isinstance(a, (bytes, bytearray)):
        data = bytes(a)
    if not data:
        return None, None
    try:
        arr, sr = sf.read(io.BytesIO(data))
        return arr.astype("float32"), sr
    except Exception:
        pass
    try:  # fallback for codecs libsndfile can't read (m4a/aac/webm) via audioread
        import librosa
        arr, sr = librosa.load(io.BytesIO(data), sr=None, mono=False)
        if getattr(arr, "ndim", 1) > 1:
            arr = arr.T  # (ch, n) -> (n, ch) for downstream mean(axis=1)
        return np.asarray(arr, dtype="float32"), int(sr)
    except Exception:
        return None, None


def free_gb(path="."):
    import shutil
    return shutil.disk_usage(path).free / 1e9


def stage_extract(repo, config, name, audio_col, text_col, speaker_col,
                  max_samples, min_free_gb=50):
    """Stream source -> mp3 + rows json. Resumable via <name>.rows.json."""
    import soundfile as sf
    from datasets import (load_dataset, get_dataset_split_names,
                          load_dataset_builder, Audio)

    rows_json = f"{name}.rows.json"
    audio_list_json = f"{name}-audio.json"
    if os.path.exists(rows_json):
        rows = json.load(open(rows_json))
        print(f"[extract] resume: {len(rows)} rows already extracted")
        return rows, speaker_col

    builder = load_dataset_builder(repo, config)
    features = builder.info.features
    audio_col, text_col, speaker_col = detect_cols(features, audio_col, text_col, speaker_col)
    print(f"[extract] cols -> audio={audio_col} text={text_col} speaker={speaker_col}")
    if not audio_col or not text_col:
        raise SystemExit(f"could not detect audio/text columns from {list(features)}")

    audio_dir = f"{name}_audio"
    os.makedirs(audio_dir, exist_ok=True)
    try:
        splits = get_dataset_split_names(repo, config)
    except Exception:
        splits = ["train"]

    rows, idx = [], 0
    for split in splits:
        ds = load_dataset(repo, config, split=split, streaming=True)
        # don't let datasets auto-decode audio during iteration: one malformed
        # clip would crash the whole stream. Get raw bytes, decode per-row below.
        try:
            ds = ds.cast_column(audio_col, Audio(decode=False))
        except Exception:
            pass
        it = iter(ds)
        while True:
            try:
                ex = next(it)
            except StopIteration:
                break
            except Exception as e:
                print("[extract] skip (iter):", type(e).__name__, str(e)[:60])
                continue
            try:
                text = (ex.get(text_col) or "")
                text = text.strip() if isinstance(text, str) else ""
                if len(text) < 2:
                    continue
                arr, sr = decode_audio(ex.get(audio_col))
                if arr is None:
                    continue
                if arr.ndim > 1:
                    arr = arr.mean(axis=1)
                if arr.shape[0] < 10000:
                    continue
                fn = os.path.join(audio_dir, f"{name}-{split}-{idx}.mp3")
                sf.write(fn, arr, sr)
                row = {"audio_filename": fn, "text": text}
                if speaker_col and ex.get(speaker_col) is not None:
                    row["speaker"] = f"{name}_{ex[speaker_col]}"
                rows.append(row)
                idx += 1
                if idx % 500 == 0:
                    print(f"[extract] {idx} clips...")
                    if free_gb() < min_free_gb:
                        raise SystemExit(f"low disk (<{min_free_gb}GB free) at {idx} "
                                         f"clips; aborting {name} to protect the batch")
                if max_samples and idx >= max_samples:
                    break
            except Exception as e:
                print("[extract] skip row:", type(e).__name__, str(e)[:80])
        if max_samples and idx >= max_samples:
            break

    json.dump(rows, open(rows_json, "w"), ensure_ascii=False)
    json.dump([r["audio_filename"] for r in rows], open(audio_list_json, "w"))
    print(f"[extract] done: {len(rows)} clips -> {audio_dir}/")
    return rows, speaker_col


def stage_speaker(name, rows, speaker_col, threshold=0.1, dim=192):
    """If no speaker column, cluster TitaNet embeddings into pseudo-speakers."""
    if speaker_col and all("speaker" in r for r in rows):
        print("[speaker] using source speaker column")
        return rows

    rows_json = f"{name}.rows.json"
    # embedding.py derives its output folder as <file-without-.json>_embedding
    emb_dir = rows_json[:-5] + "_embedding"
    n_npy = len(glob.glob(os.path.join(emb_dir, "*.npy")))
    if n_npy < len(rows):
        print(f"[speaker] extracting embeddings ({n_npy}/{len(rows)} done) via embedding.py")
        # tolerate nonzero exit from interpreter-teardown races; judge by output count
        subprocess.run([sys.executable, os.path.join(HERE, "embedding.py"),
                        "--file", rows_json], cwd=os.getcwd())
        n_npy = len(glob.glob(os.path.join(emb_dir, "*.npy")))
        print(f"[speaker] embeddings now {n_npy}/{len(rows)}")
        # only fail on near-total failure (systemic GPU/model issue worth retrying);
        # a few missing vectors just become '<name>_unknown' in the clustering loop.
        if n_npy < 0.1 * len(rows):
            raise SystemExit(f"embedding.py produced too few vectors ({n_npy}/{len(rows)})")

    import faiss
    index = faiss.IndexFlatL2(dim)
    centroids = 0

    def assign(x):
        nonlocal centroids
        x = np.ascontiguousarray(x[None].astype("float32"))
        if centroids == 0:
            index.add(x); centroids += 1; return 0
        D, I = index.search(x, 1)
        if D[0][0] > threshold:
            index.add(x); centroids += 1; return centroids - 1
        return int(I[0][0])

    # match cluster-*.ipynb: a row with no (loadable) embedding is SKIPPED, not
    # kept with a fake speaker.
    kept = []
    for i, row in enumerate(rows):
        vf = os.path.join(emb_dir, f"{i}.npy")
        if not os.path.exists(vf):
            continue
        try:
            v = np.load(vf)
        except Exception:
            continue
        row["speaker"] = f"{name}_{assign(v)}"
        kept.append(row)
    dropped = len(rows) - len(kept)
    n_spk = len({r["speaker"] for r in kept})
    print(f"[speaker] clustered {len(kept)} rows into {n_spk} speakers "
          f"({dropped} dropped: no embedding)")
    return kept


def stage_neucodec(name, n_rows):
    audio_list_json = f"{name}-audio.json"
    print("[neucodec] encoding mp3 -> tokens via convert_neucodec.py")
    # tolerate nonzero exit from interpreter-teardown races; judge by output count
    subprocess.run([sys.executable, os.path.join(HERE, "convert_neucodec.py"),
                    "--file", audio_list_json], cwd=os.getcwd())
    n = len(glob.glob(f"{name}_audio_neucodec/**/*.json", recursive=True))
    # NOT a hard failure: convert_neucodec.py skips clips >20s by design, so
    # long-form datasets legitimately yield few/zero tokens. Parquet + mp3 still
    # publish; the neucodec zip is skipped if empty (see stage_push).
    print(f"[neucodec] {n} token files in {name}_audio_neucodec/ "
          f"(of {n_rows} clips; clips >20s are skipped)")
    return n


PARTITION_SIZE = 5e9  # 5 GB per zip part


def _upload_retry(api, path, repo, tries=12):
    for k in range(tries):
        try:
            api.upload_file(path_or_fileobj=path, path_in_repo=path,
                            repo_id=repo, repo_type="dataset")
            return
        except Exception as e:
            print(f"[push] upload {path} failed ({type(e).__name__}: {str(e)[:80]}), "
                  f"retry {k+1}/{tries} in 60s")
            time.sleep(60)
    raise RuntimeError(f"upload failed after {tries} tries: {path}")


def upload_folder_chunked(api, folder, repo, partition_size=PARTITION_SIZE):
    """Zip + upload a folder to HF. Small (<=5GB): one <folder>.zip. Large: split
    into ~5GB parts <folder>-0-<part>.zip, zipping -> uploading -> deleting each
    part incrementally so peak disk stays ~one part (needed for >50GB folders)."""
    import zipfile
    files = sorted(os.path.join(dp, fn)
                   for dp, _, fns in os.walk(folder) for fn in fns)
    if not files:
        print(f"[push] {folder} empty, skip")
        return
    total = sum(os.path.getsize(f) for f in files)

    def write_upload(part_files, zname):
        with zipfile.ZipFile(zname, "w", zipfile.ZIP_STORED) as zf:
            for f in part_files:
                zf.write(f, arcname=f)  # keep relative path so audio_filename resolves
        sz = os.path.getsize(zname)
        _upload_retry(api, zname, repo)
        os.remove(zname)
        print(f"[push] uploaded {zname} ({sz/1e9:.2f} GB, {len(part_files)} files)")

    if total <= partition_size:
        write_upload(files, f"{folder}.zip")
        return
    part, cur, cur_sz = 0, [], 0
    for f in files:
        s = os.path.getsize(f)
        if cur and cur_sz + s >= partition_size:
            write_upload(cur, f"{folder}-0-{part}.zip")
            part += 1
            cur, cur_sz = [], 0
        cur.append(f)
        cur_sz += s
    if cur:
        write_upload(cur, f"{folder}-0-{part}.zip")


def stage_push(name, rows, repo_src, config):
    from datasets import Dataset
    from huggingface_hub import HfApi

    print(f"[push] push_to_hub parquet: {HF_REPO} :: {name} ({len(rows)} rows)")
    Dataset.from_list(rows).push_to_hub(HF_REPO, name)

    api = HfApi()
    for folder in (f"{name}_audio", f"{name}_audio_neucodec"):
        if not os.path.isdir(folder):
            print(f"[push] WARN missing {folder}, skip")
            continue
        upload_folder_chunked(api, folder, HF_REPO)


def stage_cleanup(name):
    """Delete local artifacts after a successful push (keep the checkpoint).
    Essential for batch runs so /share does not fill up."""
    import shutil
    targets = [f"{name}_audio", f"{name}_audio_neucodec", f"{name}.rows_embedding",
               f"{name}_audio.zip", f"{name}_audio_neucodec.zip",
               f"{name}.rows.json", f"{name}-audio.json"]
    freed = 0
    for t in targets:
        if os.path.isdir(t):
            for dp, _, fns in os.walk(t):
                for fn in fns:
                    try: freed += os.path.getsize(os.path.join(dp, fn))
                    except OSError: pass
            shutil.rmtree(t, ignore_errors=True)
        elif os.path.exists(t):
            try: freed += os.path.getsize(t)
            except OSError: pass
            try: os.remove(t)
            except OSError: pass
    print(f"[cleanup] removed local artifacts (~{freed/1e6:.0f} MB freed)")


@click.command()
@click.option("--repo", required=True, help="source HF dataset id")
@click.option("--name", required=True, help="output config name in Multilingual-TTS")
@click.option("--config", default=None, help="source config (default: first)")
@click.option("--audio-col", default=None)
@click.option("--text-col", default=None)
@click.option("--speaker-col", default=None)
@click.option("--max-samples", default=None, type=int, help="cap clips (testing)")
@click.option("--cluster-threshold", default=0.1, type=float,
              help="L2 threshold for speaker clustering when no speaker column "
                   "(0.1 = per-utterance unique, notebook default; titanet vectors "
                   "are L2-normalized, NN distances ~0.65-1.02 so raise to group)")
@click.option("--workdir", default=".", help="where audio/token folders are written")
@click.option("--keep-local", is_flag=True, default=False,
              help="keep local mp3/token/embedding artifacts after push (default: delete)")
def main(repo, name, config, audio_col, text_col, speaker_col, max_samples,
         cluster_threshold, workdir, keep_local):
    os.chdir(workdir)
    done = os.path.join(ckpt_dir(workdir), f"{name}.done")
    if os.path.exists(done):
        print(f"[skip] checkpoint exists: {done}")
        return

    t0 = time.time()
    try:
        if config is None:
            from datasets import get_dataset_config_names
            cfgs = get_dataset_config_names(repo)
            config = cfgs[0] if cfgs else "default"
            print(f"[config] using '{config}' of {cfgs}")

        rows, speaker_col = stage_extract(repo, config, name, audio_col, text_col,
                                          speaker_col, max_samples)
        if not rows:
            raise SystemExit("no rows extracted")
        rows = stage_speaker(name, rows, speaker_col, threshold=cluster_threshold)
        stage_neucodec(name, len(rows))
        stage_push(name, rows, repo, config)

        json.dump({"repo": repo, "config": config, "name": name, "rows": len(rows),
                   "speakers": len({r["speaker"] for r in rows}),
                   "seconds": round(time.time() - t0, 1)},
                  open(done, "w"), indent=2)
        print(f"[done] {name}: {len(rows)} rows in {time.time()-t0:.0f}s -> checkpoint {done}")
        if not keep_local:
            stage_cleanup(name)
    except BaseException as e:
        print(f"[fail] {name}: {type(e).__name__}: {str(e)[:200]}")
        if not keep_local:
            stage_cleanup(name)  # don't leave partial artifacts on disk
        raise


if __name__ == "__main__":
    main()
    # avoid noisy interpreter-teardown GIL crash (faiss/numba/torch) masking success
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
