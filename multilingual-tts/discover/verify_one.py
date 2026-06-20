#!/usr/bin/env python3
"""Verify ONE HF dataset repo is TTS/STT-suitable.

Reads the repo's README + ONE parquet shard, but only byte-range reads the
first row group (audio + transcription columns) over HfFileSystem -- no full
shard download. Checks for an audio column + transcription column, decodes one
audio sample, and scans the README for non-speech (environment/animal/music)
signals. Column conventions mined from prepare/*.ipynb.

Usage: python verify_one.py <owner/name>
Result cached to discover/cache/verify/<owner__name>.json
"""
import sys, os, json, io, time, traceback

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download

ROOT = os.path.dirname(os.path.abspath(__file__))
VDIR = os.path.join(ROOT, "cache", "verify")
os.makedirs(VDIR, exist_ok=True)

AUDIO_NAMES = {"audio", "audio_filename", "path", "wav_path", "target_wav",
               "reference_audio", "file_name", "audio_file", "wav_filename",
               "wav", "normalized_wav", "filename", "mp3", "flac", "speech",
               "audio_path", "file", "filepath", "audio_bytes"}
TEXT_NAMES = {"text", "transcription", "sentence", "transcript", "voice_text",
              "utt", "transcription_normalised", "transcription_normalized",
              "transcript_a", "text_original", "target_text", "reference_text",
              "original_text", "cleaned_text", "normalized_text", "words",
              "caption", "raw_transcription", "normalized", "transcripts",
              "sentences", "utterance", "label_text", "ortho", "phonemes",
              "transcript_text", "text_clean"}
TEXT_SUBSTR = ("transcript", "text", "sentence", "caption", "utter",
               "normaliz", "ortho", "phonem", "words", "translation")
NEG_KW = ["environmental sound", "sound event", "acoustic scene", "urbansound",
          "birdsong", "bird song", "animal sound", "animal vocal", "music genre",
          "instrument classification", "impulse response", "room acoustic",
          "soundscape", "esc-50", "audioset", "machine sound", "heart sound",
          "respiratory sound", "snoring", "gunshot", "drum loop"]


def is_audio_struct(field):
    t = field.type
    if pa.types.is_struct(t):
        names = {t.field(i).name for i in range(t.num_fields)}
        return "bytes" in names or "path" in names
    return False


def decode_audio_value(val):
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
            return False, None, None
        with sf.SoundFile(io.BytesIO(data)) as f:
            return True, f.samplerate, len(f) / float(f.samplerate)
    except Exception as e:
        return False, None, f"{type(e).__name__}"


def looks_like_sentence(s):
    return isinstance(s, str) and len(s.strip()) >= 2 and any(c.isalpha() for c in s)


def is_id_name(n):
    n = n.lower()
    return n == "id" or n.endswith("_id") or n.endswith("_idx") or "idx" in n or n.endswith("_ids")


def text_candidates(schema):
    """String columns likely to be transcription: exact-name first, then substring."""
    strings = [f.name for f in schema
               if pa.types.is_string(f.type) or pa.types.is_large_string(f.type)]
    exact = [n for n in strings if n.lower() in TEXT_NAMES]
    substr = [n for n in strings if n not in exact and not is_id_name(n)
              and any(k in n.lower() for k in TEXT_SUBSTR)]
    return exact + substr


def pick_shard(api, repo):
    files = api.list_repo_files(repo, repo_type="dataset")
    parquets = [f for f in files if f.endswith(".parquet")]
    readme = "README.md" in files
    if not parquets:
        other = sorted({os.path.splitext(f)[1] for f in files if "." in f})
        return None, parquets, readme, other
    # pick the globally SMALLEST shard -> smallest row groups -> fast range read.
    # get_paths_info 413s on big repos, so batch it in chunks.
    def rank(p):
        pl = p.lower()
        return (0 if "test" in pl else 1 if ("valid" in pl or "dev" in pl) else 2,
                0 if "00000-of-" in pl else 1, len(p))
    # Size only when cheap (<=100 parquet = 1 paths-info call). For mega-repos,
    # fall back to the name heuristic to avoid blowing the API rate limit.
    sizes = {}
    if len(parquets) <= 100:
        try:
            for inf in api.get_paths_info(repo, parquets, repo_type="dataset"):
                sizes[inf.path] = getattr(inf, "size", None) or 1 << 62
        except Exception:
            pass
    if sizes:
        cands = sorted(parquets, key=lambda p: (sizes.get(p, 1 << 62), rank(p)))
    else:
        cands = sorted(parquets, key=rank)
    return cands[0], parquets, readme, None


def verify(repo):
    api = HfApi()
    res = {"id": repo, "ok": False, "error": None, "has_parquet": False,
           "audio_col": None, "text_col": None, "audio_decoded": False,
           "samplerate": None, "duration_s": None, "sample_text": None,
           "neg_domain_hit": None, "shard": None, "n_parquet": 0,
           "schema": None, "readme_bytes": 0, "verdict": None, "reason": None}
    try:
        shard, parquets, has_readme, other = pick_shard(api, repo)
    except Exception as e:
        res["error"] = f"list:{type(e).__name__}:{e}"; save(res); return res

    res["has_parquet"] = bool(parquets); res["n_parquet"] = len(parquets)
    # README
    if has_readme:
        try:
            p = hf_hub_download(repo, "README.md", repo_type="dataset")
            txt = open(p, encoding="utf-8", errors="ignore").read()
            res["readme_bytes"] = len(txt)
            res["neg_domain_hit"] = next((k for k in NEG_KW if k in txt.lower()), None)
        except Exception:
            pass
    if not parquets:
        res["error"] = "no-parquet"; res["other_formats"] = other
        res["verdict"] = "needs-manual"; res["reason"] = f"no parquet ({other})"
        save(res); return res
    res["shard"] = shard

    try:
        fs = HfFileSystem()
        with fs.open(f"datasets/{repo}/{shard}", "rb") as fh:
            pf = pq.ParquetFile(fh)
            schema = pf.schema_arrow
            res["schema"] = [(f.name, str(f.type)[:50]) for f in schema]
            audio_col = next((f.name for f in schema if is_audio_struct(f)), None) \
                or next((f.name for f in schema if f.name.lower() in AUDIO_NAMES), None)
            txt_cands = text_candidates(schema)
            res["audio_col"] = audio_col
            cols = [c for c in [audio_col, *txt_cands] if c]
            if cols:
                batch = next(pf.iter_batches(batch_size=8, columns=cols), None)
                if batch is not None:
                    t = pa.Table.from_batches([batch])
                    if audio_col:
                        for v in t.column(audio_col).to_pylist():
                            ok, sr, dur = decode_audio_value(v)
                            if ok:
                                res["audio_decoded"] = True
                                res["samplerate"] = sr
                                res["duration_s"] = round(dur, 2) if isinstance(dur, float) else None
                                break
                    # pick first text candidate whose sampled value is sentence-like
                    for cand in txt_cands:
                        for tv in t.column(cand).to_pylist():
                            if looks_like_sentence(tv):
                                res["text_col"] = cand
                                res["sample_text"] = tv[:200]
                                break
                        if res["text_col"]:
                            break
    except Exception as e:
        res["error"] = f"parquet:{type(e).__name__}:{str(e)[:120]}"
        res["verdict"] = "error"; res["reason"] = res["error"]; save(res); return res

    suitable = (res["audio_col"] and res["text_col"] and res["audio_decoded"]
                and res["sample_text"] and not res["neg_domain_hit"])
    res["ok"] = True
    res["verdict"] = "suitable" if suitable else "not-suitable"
    res["reason"] = ("audio+transcription present, audio decodes, no neg-domain"
                     if suitable else "; ".join(filter(None, [
                         None if res["audio_col"] else "no-audio-col",
                         None if res["text_col"] else "no-text-col",
                         None if res["audio_decoded"] else "audio-not-decoded",
                         None if res["sample_text"] else "no-sample-text",
                         f"neg:{res['neg_domain_hit']}" if res["neg_domain_hit"] else None,
                     ])) or "ok")
    save(res); return res


def save(res):
    with open(os.path.join(VDIR, res["id"].replace("/", "__") + ".json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)


def is_ratelimit(res):
    e = (res.get("error") or "").lower()
    return any(k in e for k in ("429", "too many requests", "rate limit", "readtimeout",
                                "remoteprotocol", "peer closed", "connection",
                                "we had to rate limit"))


if __name__ == "__main__":
    repo = sys.argv[1]
    r = verify(repo)
    tries = 0
    while is_ratelimit(r) and tries < 4:
        tries += 1
        time.sleep(15 + 15 * tries)
        r = verify(repo)
    print(json.dumps(r, indent=2, ensure_ascii=False))
