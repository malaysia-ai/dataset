# Multilingual-TTS — repo guide for Claude

Gather multilingual **TTS/STT** datasets and push them to
https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS.

- `prepare/prepare-<name>.ipynb` — one notebook per source dataset, normalizing it
  into the published dataset. These are the **source of truth for column-name
  conventions** (audio vs transcription) — mine them before guessing.
- `convert_neucodec*.py`, `embedding.py`, `trim_silence.py` — audio tokenization /
  cleanup utilities.
- `discover/` — pipeline that finds **new** HF audio datasets to add (see below).

## Column conventions (from `prepare/`)

- **Audio columns:** `audio` (usually `struct<bytes,path>`), `audio_filename`,
  `path`, `wav_path`, `wav`, `file_name`, `audio_file`, `target_wav`,
  `reference_audio`, `normalized_wav`.
- **Transcription columns:** `text`, `transcription`, `sentence`, `transcript`,
  `normalized_text`, `voice_text`, `raw_transcription`, `asr_transcript`,
  `human_transcript`, … (match by substring `transcript/text/sentence/…`,
  excluding `*_id`/`*_idx`).

## `discover/` — finding new datasets (ALWAYS CACHE; steps are resumable)

Goal: list HF audio datasets suitable for TTS/STT that are **not already** in
Multilingual-TTS, verified by actually reading each repo. **Every step writes a
cache so re-runs skip completed work** — never re-fetch what's already cached.

Pipeline & caches (all under `discover/cache/`):

1. **Full HF audio dataset list** → `all_audio_datasets.json`
   Paginate `https://huggingface.co/api/datasets?filter=modality:audio&sort=trendingScore&direction=-1&limit=1000`
   following the `Link: rel="next"` cursor to the end (~32k datasets, 33 pages).
   Reuse this file; only re-paginate to refresh.
2. **Already-processed list** → `mtts_processed_configs.txt`
   `config_name`s parsed from Multilingual-TTS `README.md`
   (`/raw/main/README.md`). These are the datasets we already have.
3. **Overlap removal + candidate filter** → `candidates_strong.json`
   (declared ASR/TTS/text-to-audio task category), `candidates_keyword.json`
   (speech keywords, no task tag), `overlap_hits.json` (dropped as duplicates).
   Overlap = normalized (lowercased, alnum-only) match of the HF repo's
   name-segment against a processed config name. `malaysia-ai/*` is excluded.
4. **Per-repo verification** → `cache/verify/<owner>__<name>.json` (one file per
   repo; presence = done → skipped on re-run).
   `verify_one.py <owner/name>` reads the repo README + the **smallest** parquet
   shard, byte-range reading only **row group 0** over `HfFileSystem` (no full
   download — full shards are 0.3–4 GB). It detects an audio column + a
   transcription column, **decodes one audio sample**, and scans the README for
   non-speech signals (environment/animal/music). `verdict ∈ {suitable,
   not-suitable, needs-manual, error}`.

Run / resume the batch:
```bash
cd discover
# rebuild pending list (ids in candidates_strong.json without a verify/ file)
jq -r '.[].id' cache/candidates_strong.json > cache/strong_ids.txt
python3 - <<'PY'
import os
ids=[l.strip() for l in open("cache/strong_ids.txt") if l.strip()]
done={f[:-5] for f in os.listdir("cache/verify") if f.endswith(".json")}
open("cache/pending_ids.txt","w").write("\n".join(i for i in ids if i.replace("/","__") not in done)+"\n")
PY
bash run_batch.sh cache/pending_ids.txt 12 150   # workers=12, per-repo hard timeout=150s
```

5. **Final deliverable** → `discover/new_tts_stt_datasets.json` — non-overlapping,
   verified-suitable datasets (built by `build_final.py` from the verify cache).

### Conventions
- Use the HF JSON API + cursor pagination, not HTML scraping.
- Verify per-repo by reading the real parquet (row group 0), **not** the
  datasets-server aggregate API.
- Always pick the **smallest** shard; `get_paths_info` 413s on huge repos, so
  batch it in chunks of 100.
