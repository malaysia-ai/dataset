import os
import json
import click
import math
import librosa
import itertools
from glob import glob
from functools import partial
from multiprocess import Pool
from tqdm import tqdm

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
    
def loop(rows):
    rows, index = rows
    os.environ['CUDA_VISIBLE_DEVICES'] = str(index)
        
    import torch
    torch.set_grad_enabled(False)

    from neucodec import NeuCodec
    import torchaudio
    from torch.utils.data import Dataset, DataLoader
    
    torch.autograd.set_grad_enabled(False) 

    model = NeuCodec.from_pretrained("neuphonic/neucodec")
    model.eval().cuda()   

    class CustomDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows
    
        def __len__(self):
            return len(self.rows)
    
        def __getitem__(self, idx):
            try:
                return self.rows[idx], librosa.load(self.rows[idx], sr = 16000)[0]
            except Exception as e:
                print('error in dataset', e, self.rows[idx])
                return None
    
    def collator(batch):
        batch = [b for b in batch if b is not None]
        filenames = [b[0] for b in batch]
        audio = [b[1] for b in batch]
        
        padded_audio = []
        padded_audio_len = []
        for y in audio:
            y = torch.from_numpy(y).float()
            padded_audio_len.append(y.shape[-1])
            pad_for_wav = 320 - (y.shape[-1] % 320)
            y = torch.nn.functional.pad(y, (0, pad_for_wav))
            padded_audio_len.append(y.shape[-1])
            padded_audio.append(y)
        
        semantic_features = model.feature_extractor(
            padded_audio, sampling_rate=16_000, return_tensors="pt", padding=True,
        ).input_features
        padded_audio = torch.nn.utils.rnn.pad_sequence(
            padded_audio, 
            batch_first=True, 
            padding_value=0.0, 
            padding_side='right',
        )
        padded_audio_len = torch.tensor(padded_audio_len)
            
        return {
            'filenames': filenames, 
            'semantic_features': semantic_features,
            'padded_audio': padded_audio,
            'padded_audio_len': padded_audio_len,
        }

    data = CustomDataset(rows)
    dataloader = DataLoader(
        data, 
        batch_size=16, 
        collate_fn=collator, 
        num_workers=10, 
        prefetch_factor=5, 
        pin_memory=True,
    )
    
    for batch in tqdm(iter(dataloader)):
        try:
            mask_len = batch['padded_audio_len'] // 320
            y = batch['padded_audio'].to(model.device).unsqueeze(1)
            semantic_features = batch['semantic_features'].to(model.device)
            
            acoustic_emb = model.CodecEnc(y)
            acoustic_emb = acoustic_emb.transpose(1, 2)
            
            semantic_output = (
                model.semantic_model(semantic_features).hidden_states[16].transpose(1, 2)
            )
            semantic_encoded = model.SemanticEncoder_module(semantic_output)
            
            # concatenate embeddings
            if acoustic_emb.shape[-1] != semantic_encoded.shape[-1]:
                min_len = min(acoustic_emb.shape[-1], semantic_encoded.shape[-1])
                acoustic_emb = acoustic_emb[:, :, :min_len]
                semantic_encoded = semantic_encoded[:, :, :min_len]        
            concat_emb = torch.cat([semantic_encoded, acoustic_emb], dim=1)
            concat_emb = model.fc_prior(concat_emb.transpose(1, 2)).transpose(1, 2)
            
            # quantize
            _, fsq_codes, _ = model.generator(concat_emb, vq=True)
            fsq_codes = fsq_codes[:,0].tolist()
            
            for no, f in enumerate(batch['filenames']):
                splitted = new_path(f)
                os.makedirs(os.path.split(splitted)[0], exist_ok = True)
                token = fsq_codes[no][:mask_len[no]]
                with open(splitted, 'w') as fopen:
                    json.dump(token, fopen)
        except Exception as e:
            print('error in iter', e)

@click.command()
@click.option('--file')
@click.option('--replication', default = 1)
def main(file, replication):
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