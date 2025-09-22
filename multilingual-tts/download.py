from huggingface_hub import snapshot_download
import time
import os

dataset = """
https://huggingface.co/datasets/SPRINGLab/IndicTTS_Tamil
https://huggingface.co/datasets/SPRINGLab/IndicTTS_Bengali
https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu
https://huggingface.co/datasets/SPRINGLab/IndicTTS_Malayalam
https://huggingface.co/datasets/SPRINGLab/IndicTTS_Punjabi
https://huggingface.co/datasets/shb777/gemini-flash-2.0-speech
https://huggingface.co/datasets/ylacombe/cml-tts
https://huggingface.co/datasets/facebook/multilingual_librispeech
https://huggingface.co/datasets/parler-tts/libritts_r_filtered
https://huggingface.co/datasets/BAAI/ChildMandarin
https://huggingface.co/datasets/AISHELL/AISHELL-3
https://huggingface.co/datasets/Wenetspeech4TTS/WenetSpeech4TTS
https://huggingface.co/datasets/MBZUAI/ClArTTS
https://huggingface.co/datasets/MBZUAI/ArVoice
https://huggingface.co/datasets/MohamedRashad/multilingual-tts
"""

dataset = [d.strip().split('datasets/')[1] for d in dataset.split('\n') if len(d.strip()) > 2]
print(dataset)

for d in dataset:
    print(d, os.path.split(d)[1])
    while True:
        try:
            snapshot_download(repo_id=d, repo_type="dataset", local_dir=os.path.split(d)[1])
            break
        except Exception as e:
            print(e)
            time.sleep(60)