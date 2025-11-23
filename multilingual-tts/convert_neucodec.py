import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import soundfile as sf
import json
import click
import re
import librosa
from glob import glob
from functools import partial
from multiprocess import Pool
from tqdm import tqdm
import numpy as np
import itertools

def old_chunks(l, n):
    for i in range(0, len(l), n):
        yield (l[i: i + n], i // n)
        
def chunks(l, devices):
    chunk_size = len(l) // len(devices)
    remainder = len(l) % len(devices)
    start = 0
    for i in range(len(devices)):
        extra = 1 if i < remainder else 0
        end = start + chunk_size + extra
        yield (l[start:end], devices[i])
        start = end
        
def new_path(f):
    splitted = f.split('/')
    folder = f.split('/')[0]
    folder = folder + '_neucodec'
    new_f = os.path.join(folder, '/'.join(splitted[1:]))
    new_f = new_f.replace('.mp3', '.json').replace('.wav', '.json')
    return new_f
    
def multiprocessing(strings, function, cores=6, returned=True):
    df_split = old_chunks(strings, len(strings) // cores)
    pool = Pool(cores)
    pooled = pool.map(function, df_split)
    pool.close()
    pool.join()

    if returned:
        return list(itertools.chain(*pooled))
        
def check(files):
    files, _ = files
    filtered = []
    for file in tqdm(files):
        filename_done = new_path(file)

        if os.path.exists(filename_done):
            try:
                with open(filename_done) as fopen:
                    json.load(fopen)
                    continue
            except:
                pass
            
        filtered.append(file)
    return filtered
    
def loop(
    indices_device_pair,
):
    files, device = indices_device_pair
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    
    from neucodec import NeuCodec
    import torchaudio
    import torch
    torch.autograd.set_grad_enabled(False) 

    model = NeuCodec.from_pretrained("neuphonic/neucodec")
    model.eval().cuda()   

    for f in tqdm(files):
        filename = new_path(f)
        if os.path.exists(filename):
            try:
                with open(filename) as fopen:
                    json.load(fopen)
                continue
            except:
                pass

        try:
            y, sr = librosa.load(f, sr = 16000)
            if len(y) / sr > 20:
                continue
            wav_tensor = torch.from_numpy(y).float().unsqueeze(0)
            fsq_codes = model.encode_code(wav_tensor.unsqueeze(1))
            tokens = fsq_codes[0, 0].tolist()

            os.makedirs(os.path.split(filename)[0], exist_ok = True)
            with open(filename, 'w') as fopen:
                json.dump(tokens, fopen)
        except Exception as e:
            print(e)

@click.command()
@click.option('--file')
@click.option('--replication', default = 1)
def main(
    file, 
    replication,
):
    devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if devices is None:
        
        import torch
        devices = list(range(torch.cuda.device_count()))
    else:
        devices = [d.strip() for d in devices.split(',')]

    devices = replication * devices
    print(devices)

    with open(file) as fopen:
        files = json.load(fopen)
    filtered = multiprocessing(files, check, 30)
    
    print(len(files), len(filtered))

    df_split = list(chunks(filtered, devices))

    loop_partial = partial(loop)

    with Pool(len(devices)) as pool:
        pooled = pool.map(loop_partial, df_split)

if __name__ == '__main__':
    main()

    
