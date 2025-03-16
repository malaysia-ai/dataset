# Emotional Malaysian Emilia

Synthetic Emotional label on Malaysian Emilia.

## how to

### Predict Audioset sliding window

```bash
CUDA_VISIBLE_DEVICES=0 \
python3 audioset_sliding_v2.py --path 'malaysian-podcast_processed/**/*.mp3' --global-index 1 --local-index 0
```

### Predict Emotion