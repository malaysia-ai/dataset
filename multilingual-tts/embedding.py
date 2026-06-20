import click
import json
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from multiprocess import Pool
import os

# Speaker embeddings now come from titanet-vectors-fp16 (NOT malaya-speech):
#   pip3 install git+https://github.com/Scicom-AI-Enterprise-Organization/titanet-vectors-fp16
# Same TitaNet-L weights as before (huseinzol05/nemo-titanet_large), run in fp16.
# Output is the RAW 192-d embedding (NOT normalized) -- exactly like the original
# malaya-based embedding.py -- so the greedy faiss IndexFlatL2 clustering in
# cluster-*.ipynb at threshold 0.1 groups same-voice clips as intended.
# (Normalizing to unit length inflates same-voice L2 above 0.1 and over-splits.)

def chunks(l, devices, folder):
    chunk_size = len(l) // len(devices)
    remainder = len(l) % len(devices)
    start = 0
    for i in range(len(devices)):
        extra = 1 if i < remainder else 0
        end = start + chunk_size + extra
        yield (l[start:end], devices[i], folder)
        start = end

def loop(rows):
    rows, index, folder = rows
    os.environ['CUDA_VISIBLE_DEVICES'] = str(index)

    import torch
    import librosa
    from titanet_vectors import load

    torch.autograd.set_grad_enabled(False)
    model = load('huseinzol05/nemo-titanet_large').cuda().eval()
    model = model.to(torch.float16)

    for row in tqdm(rows, desc = f'loop {index}'):
        no, row = row
        new_f = os.path.join(folder, f'{no}.npy')
        if os.path.exists(new_f):
            continue
        try:
            y, sr = librosa.load(row['audio_filename'], sr = 16000)
            # cap to 30s: a speaker vector needs only a few seconds, and an
            # uncapped long clip spikes titanet activations to tens of GB -> OOM
            if len(y) > 30 * 16000:
                y = y[:30 * 16000]
            x = torch.from_numpy(y).float().unsqueeze(0).cuda().half()
            lengths = torch.tensor([x.shape[-1]], device = x.device)
            _, embs = model(x, lengths)
            e = embs[0].float().cpu().numpy()
            np.save(new_f, e, allow_pickle=True)
        except Exception as ex:
            print(ex)

@click.command()
@click.option('--file')
@click.option('--replication', default = 1)
def main(file, replication):

    folder = file.replace('.json', '') + '_embedding'
    os.makedirs(folder, exist_ok = True)
    devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if devices is None:
        devices = list(range(torch.cuda.device_count()))
    else:
        devices = [d.strip() for d in devices.split(',')]

    devices = replication * devices
    print(devices)

    with open(file) as fopen:
        rows = json.load(fopen)
    rows = [(i, rows[i]) for i in range(len(rows))]

    df_split = chunks(rows, devices, folder)
    pool = Pool(len(devices))
    pooled = pool.map(loop, df_split)
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()
