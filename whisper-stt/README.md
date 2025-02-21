# Whisper STT

## Force Alignment

```bash
CUDA_VISIBLE_DEVICES=2 \
python3.10 force_alignment.py \
--filename 'prepare-force-alignment.json' \
--language 'ms' \
--replication 3
```