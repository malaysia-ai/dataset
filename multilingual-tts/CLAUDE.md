# Multilingual-TTS — repo guide for Claude

Gather multilingual **TTS/STT** datasets and push them to
https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS.

- `prepare/prepare-<name>.ipynb` — one notebook per source dataset, normalizing it
  into the published dataset. These are the **source of truth for column-name
  conventions** (audio vs transcription) — mine them before guessing.
- `convert_neucodec*.py`, `embedding.py`, `trim_silence.py` — audio tokenization /
  cleanup utilities.
- `discover/` — pipeline that finds **new** HF audio datasets to add (see below).

## Prepare pipeline (adding a source dataset)

Each published row is `{audio_filename, text, speaker}`. Steps (see
`prepare/prepare-indicTTS.ipynb` as the canonical example):

1. **Extract audio → mp3** to save storage. Read the source parquet, decode
   `audio['bytes']`, downmix to mono, **skip** empty text (`len < 2`) and very
   short clips (`< 10000` samples), write `<dataset>_audio/<shard>_<i>.mp3`.
2. **Speaker label.**
   - If the source has a speaker/gender column, use it (e.g. `f"{base}_{gender}"`).
   - Otherwise **cluster**: `embedding.py` extracts a 192-d TitaNet speaker vector
     per row → greedy online clustering with faiss `IndexFlatL2(192)`,
     `assign(x, threshold=0.1)` (see `prepare/cluster-*.ipynb`), then append the
     cluster id to the speaker label.
   - `embedding.py` uses **titanet-vectors-fp16**, NOT malaya-speech:
     `pip3 install git+https://github.com/Scicom-AI-Enterprise-Organization/titanet-vectors-fp16`
     — `model = load('huseinzol05/nemo-titanet_large').cuda().eval().to(torch.float16)`,
     then `logits, embs = model(wav[1,N].half(), lengths)`; `embs[0]` is the 192-d
     vector (L2-normalized before saving so the threshold stays valid). Audio is
     loaded with `librosa.load(path, sr=16000)`. Output is one `<file>_embedding/<i>.npy`
     per row — unchanged from before, so `cluster-*.ipynb` works as-is.
3. **Audio tokens (neucodec).** `convert_neucodec.py --file <audio-list>.json`:
   `librosa.load(f, sr=16000)`, **skip** clips `> 20s`, `NeuCodec.encode_code` →
   token list saved as `<folder>_neucodec/<...>.json`.
4. **Publish.** `Dataset.from_list(rows).push_to_hub('malaysia-ai/Multilingual-TTS', <config>)`,
   then zip the `_audio` / `_neucodec` folders and `api.upload_file` them.

### `prepare.py` — runs all 4 stages, checkpointed

```bash
python3 prepare.py --repo <owner/name> --name <config> --workdir /share \
    [--config X] [--audio-col A] [--text-col T] [--speaker-col S] \
    [--cluster-threshold 0.1] [--max-samples N]
```
- Columns auto-detect from the source schema (override with flags). Audio is
  streamed and re-encoded to mono mp3 (skips empty text / clips `< 10000` samples).
- **Checkpointing**: `<workdir>/checkpoints/<name>.done` is written only after the
  parquet + mp3 zip + neucodec zip all upload; on re-run that dataset is skipped.
  Within a dataset it also resumes: `<name>.rows.json` skips re-extraction, and
  `embedding.py` / `convert_neucodec.py` skip per-file outputs already present.
- **Auto-cleanup**: after a successful push, local mp3/token/embedding/zip/json
  artifacts are deleted (only the `.done` checkpoint is kept) so `/share` doesn't
  fill up across a batch. Pass `--keep-local` to retain them.
- **Speaker clustering threshold**: `--cluster-threshold 0.1` (default) = the
  notebook behavior = every clip its own speaker. titanet vectors are
  L2-normalized so same-speaker NN distances are ~0.65-1.02; raise the threshold
  (~0.9-1.1) to actually group utterances into speakers.
- Reuses `embedding.py` and `convert_neucodec.py` as multi-GPU subprocesses;
  exit codes are ignored (interpreter-teardown GIL races on this box) — success is
  judged by output file counts.

### Remote run (GPU box)

`ssh -i scicom -p 1024 root@8.222.165.68`, work in `/share` (8× H20). Deps that
needed fixing there: `faiss-cpu`, `neucodec`, titanet-vectors-fp16, and aligning
`torchaudio`/`torchvision` to `torch 2.9.1+cu128` (`--no-deps`). No `zip` binary —
`prepare.py` uses `shutil.make_archive`. First validated dataset: `Lingua_Libre_br`
(Breton, 3086 clips).

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
