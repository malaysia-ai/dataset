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
    if isinstance(a, dict):
        if a.get("array") is not None:
            return np.asarray(a["array"], dtype="float32"), int(a["sampling_rate"])
        if a.get("bytes"):
            arr, sr = sf.read(io.BytesIO(a["bytes"]))
            return arr.astype("float32"), sr
        if a.get("path") and isinstance(a["path"], str) and os.path.exists(a["path"]):
            arr, sr = sf.read(a["path"])
            return arr.astype("float32"), sr
    return None, None


def stage_extract(repo, config, name, audio_col, text_col, speaker_col, max_samples):
    """Stream source -> mp3 + rows json. Resumable via <name>.rows.json."""
    import soundfile as sf
    from datasets import load_dataset, get_dataset_split_names, load_dataset_builder

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
        for ex in ds:
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
        if n_npy < 0.5 * len(rows):
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

    for i, row in enumerate(rows):
        vf = os.path.join(emb_dir, f"{i}.npy")
        if not os.path.exists(vf):
            row["speaker"] = f"{name}_unknown"
            continue
        try:
            row["speaker"] = f"{name}_{assign(np.load(vf))}"
        except Exception:
            row["speaker"] = f"{name}_unknown"
    n_spk = len({r["speaker"] for r in rows})
    print(f"[speaker] clustered into {n_spk} pseudo-speakers")
    json.dump(rows, open(rows_json, "w"), ensure_ascii=False)
    return rows


def stage_neucodec(name, n_rows):
    audio_list_json = f"{name}-audio.json"
    print("[neucodec] encoding mp3 -> tokens via convert_neucodec.py")
    # tolerate nonzero exit from interpreter-teardown races; judge by output count
    subprocess.run([sys.executable, os.path.join(HERE, "convert_neucodec.py"),
                    "--file", audio_list_json], cwd=os.getcwd())
    n = len(glob.glob(f"{name}_audio_neucodec/**/*.json", recursive=True))
    print(f"[neucodec] {n} token files in {name}_audio_neucodec/")
    if n < 0.5 * n_rows:
        raise SystemExit(f"neucodec produced too few token files ({n}/{n_rows})")


def zip_dir(folder):
    import shutil
    z = f"{folder}.zip"
    if os.path.exists(z):
        os.remove(z)
    # pure-python zip (no external `zip` binary dependency)
    shutil.make_archive(folder, "zip", root_dir=".", base_dir=folder)
    return z


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
        z = zip_dir(folder)
        print(f"[push] upload {z} ({os.path.getsize(z)/1e6:.1f} MB)")
        api.upload_file(path_or_fileobj=z, path_in_repo=z,
                        repo_id=HF_REPO, repo_type="dataset")


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
def main(repo, name, config, audio_col, text_col, speaker_col, max_samples,
         cluster_threshold, workdir):
    os.chdir(workdir)
    done = os.path.join(ckpt_dir(workdir), f"{name}.done")
    if os.path.exists(done):
        print(f"[skip] checkpoint exists: {done}")
        return

    t0 = time.time()
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


if __name__ == "__main__":
    main()
    # avoid noisy interpreter-teardown GIL crash (faiss/numba/torch) masking success
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
