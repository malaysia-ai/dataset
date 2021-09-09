from gtts import gTTS
from smart_open import open
from pathlib import Path
from time import sleep

def process_text(a):
    text = a.replace(".","").replace(",",'').replace("?","").replace("!","")\
            .replace('-'," ").replace('"',"").replace("'",'').replace('–'," ")\
            .replace("\\"," ").replace("/"," ").replace("*","").replace("&","dan").lower().strip().split()
    return " ".join(text)

corpus = [ # https://github.com/huseinzol05/malay-dataset/tree/master/dumping/clean 7d45be3
    "https://f000.backblazeb2.com/file/malay-dataset/dumping/clean/dumping-academia.txt",
    "https://f000.backblazeb2.com/file/malay-dataset/dumping/clean/dumping-cleaned-common-crawl.txt",
    "https://f000.backblazeb2.com/file/malay-dataset/dumping/clean/dumping-cleaned-news.txt",
    "https://f000.backblazeb2.com/file/malay-dataset/dumping/clean/dumping-common-crawl.txt",
    "https://f000.backblazeb2.com/file/malay-dataset/dumping/clean/dumping-iium.txt",
    "https://f000.backblazeb2.com/file/malay-dataset/dumping/clean/dumping-instagram.txt",
    "https://f000.backblazeb2.com/file/malay-dataset/dumping/clean/dumping-news.txt",
    "https://f000.backblazeb2.com/file/malay-dataset/dumping/clean/dumping-parliament.txt",
    "https://f000.backblazeb2.com/file/malay-dataset/dumping/clean/dumping-pdf.txt",
    "https://f000.backblazeb2.com/file/malay-dataset/dumping/clean/dumping-twitter.txt",
    "https://f000.backblazeb2.com/file/malay-dataset/dumping/clean/dumping-watpadd.txt",
    "https://f000.backblazeb2.com/file/malay-dataset/dumping/clean/dumping-wiki.txt",
    "https://f000.backblazeb2.com/file/malay-dataset/dumping/clean/filtered-dumping-academia.txt",
    "https://f000.backblazeb2.com/file/malay-dataset/dumping/clean/filtered-dumping-cleaned-common-crawl.txt",
    "https://f000.backblazeb2.com/file/malay-dataset/dumping/clean/filtered-dumping-wiki.txt"
]

demo = True

if demo:
    corpus = corpus[:1]

for c in corpus:
    folder = Path(c.split("/")[-1].split(".")[0].replace("-","_"))
    folder.mkdir(exist_ok=True)
    print(f"start {str(folder)}")
    for idx,line in enumerate(open(c)):
        if demo and idx > 10:
            break
        line = process_text(line)
        print(idx,line)
        if not line:
            continue
        fn = str(idx).zfill(7)
        out_fn = f"{str(folder)}/{fn}.txt"
        test = gTTS(line,lang="ms")
        test.save(f"{str(folder)}/{fn}.mp3")
        with open(out_fn,"w+") as f:
            f.write(line)
        sleep(0.5)
    print(f"done {str(folder)}")

## idx is line number starting with 0
## store line number as filename so that can retrieve capital or punctuation for other task such as expressive TTS