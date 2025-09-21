## Convert to audio tokens

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python3 convert_neucodec_batch.py --file 'emilia-audio.json' --replication 2
```

But we prefer to use [convert_neucodec_emilia.py](convert_neucodec_emilia.py) in GH200,

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 convert_neucodec_emilia.py --file 'emilia-audio.json' --replication 13
```

Way faster!