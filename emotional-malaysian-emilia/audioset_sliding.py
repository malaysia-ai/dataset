from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm
from glob import glob
from datasets import Audio
import torch
import torchaudio
import numpy as np
import click
import os
import json

def new_path(f):
    f = f.replace('.mp3', '.audioset')
    splitted = f.split('/')
    base_folder = splitted[0] + '_audioset'
    splitted = '/'.join([base_folder] + splitted[1:])
    return splitted


@click.command()
@click.option("--path", help="files path in glob pattern")
@click.option("--global-index", default=1, help="global index")
@click.option("--local-index", default=0, help="local index")
@click.option("--sliding", default=0.25)
@click.option("--model", default='MIT/ast-finetuned-audioset-10-10-0.4593')
def function(path, global_index, local_index, sliding, model):

    feature_extractor = AutoFeatureExtractor.from_pretrained(model, return_attention_mask = True)
    model = AutoModelForAudioClassification.from_pretrained(model, torch_dtype = torch.float16).eval().cuda()
    id2label = model.config.id2label
    sr = feature_extractor.sampling_rate
    sliding = int(sliding * sr)
    audio = Audio(sampling_rate = sr)

    files = glob(path)
    filtered_files = []
    for f in files:
        new_f = new_path(f)
        if os.path.exists(new_f) and os.path.getsize(new_f) > 2:
            continue
        filtered_files.append(f)

    global_size = len(filtered_files) // global_index
    filtered_files = filtered_files[global_size * local_index: global_size * (local_index + 1)]
    files = filtered_files

    with torch.no_grad():
        for f in tqdm(files):
            y = audio.decode_example(audio.encode_example(f))['array']
            timestamps = []
            slided = []
            for i in range(0, len(y), sliding):
                y_ = y[i: i + sliding]
                if len(y_) < 1000:
                    continue
                slided.append(y[i: i + sliding])
                start = i / sr
                end = min(len(y) / sr, (i + sliding) / sr)
                timestamps.append((start, end))
            
            inputs = feature_extractor(slided, sampling_rate=sr, 
                           return_tensors="pt", return_attention_mask = True)
            inputs['input_values'] = inputs['input_values'].to(torch.float16).cuda()
            logits = model(**inputs).logits.softmax(-1)
            topk = torch.topk(logits, 5, dim = -1)
            probs = topk.values.cpu().numpy().tolist()

            for i in range(len(probs)):
                for k in range(len(probs[i])):
                    probs[i][k] = round(probs[i][k], 4)
                    
            labels = []
            for row in topk.indices.cpu().numpy():
                label = [id2label[r] for r in row]
                labels.append(label)

            splitted = new_path(f)
            os.makedirs(os.path.split(splitted)[0], exist_ok = True)
            with open(splitted, 'w') as fopen:
                json.dump({'timestamps': timestamps, 'labels': labels, 'probs': probs}, fopen)
            
if __name__ == '__main__':
    function()