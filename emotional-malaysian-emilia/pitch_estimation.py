import click
import torch
import torchaudio
from glob import glob
from tqdm import tqdm
import os
import penn
import torch
import huggingface_hub

def new_path(f):
    f = f.replace('.mp3', '.pitch')
    splitted = f.split('/')
    base_folder = splitted[0] + '_pitch'
    splitted = '/'.join([base_folder] + splitted[1:])
    return splitted

@click.command()
@click.option("--path", help="files path in glob pattern")
@click.option("--global-index", default=1, help="global index")
@click.option("--local-index", default=0, help="local index")
def function(path, global_index, local_index):
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

    model = penn.Model()
    checkpoint = huggingface_hub.hf_hub_download(
    'maxrmorrison/fcnf0-plus-plus',
    'fcnf0++.pt')
    checkpoint = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    model = model.to('cuda').to(torch.float16)

    with torch.no_grad():
        for f in tqdm(files):
            y, sr = torchaudio.load(f)
            y = torchaudio.functional.resample(y, sr, penn.SAMPLE_RATE)
            pitch, periodicity = [], []
            with torch.no_grad():
                for frames in penn.preprocess(
                    y,
                ):  
                    logits = model(frames.to(torch.float16).to('cuda'))
                    result = penn.postprocess(logits)
                    pitch.append(result[1])
                    periodicity.append(result[2])
            pitch, periodicity = torch.cat(pitch, 1), torch.cat(periodicity, 1)
            pitch = penn.voicing.interpolate(
                pitch,
                periodicity,
                interp_unvoiced_at)
            pitch = pitch[0].cpu().numpy().tolist()
            pitch = [round(p, 4) for p in pitch]
            periodicity = periodicity[0].cpu().numpy().tolist()
            periodicity = [round(p, 4) for p in periodicity]
            splitted = new_path(f)
            os.makedirs(os.path.split(splitted)[0], exist_ok = True)

            with open(splitted, 'w') as fopen:
                json.dump({'pitch': pitch, 'periodicity': periodicity}, fopen)

if __name__ == '__main__':
    function()