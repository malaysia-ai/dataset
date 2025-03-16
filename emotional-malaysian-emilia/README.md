# Emotional Malaysian Emilia

Synthetic Emotional label on Malaysian Emilia.

## how to

### Predict Audioset sliding window

```bash
CUDA_VISIBLE_DEVICES=0 \
python3.10 audioset_sliding.py --path 'malaysian-podcast_processed/**/*.mp3' --global-index 1 --local-index 0
```

### Predict Pitch Estimation

```bash
CUDA_VISIBLE_DEVICES=0 \
python3.10 pitch_estimation.py --path 'malaysian-podcast_processed/**/*.mp3' --global-index 1 --local-index 0
```

### Predict Emotion