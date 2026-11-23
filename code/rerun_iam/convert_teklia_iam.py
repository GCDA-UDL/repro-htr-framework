#!/usr/bin/env python3
"""Convert Teklia/IAM-line (Hugging Face, standard Aachen split 6,482 / 976 / 2,915) to Kraken 'path' format.

Same procedure as code/scripts/convert_catk12hw.py of the work repository (CatK12HW-actions) and as the
RIMES-2011-line conversion: images as released by Teklia (128 px high, mode L), saved as 8-bit grayscale PNG, no
resizing, no binarization; transcription exactly as released (IAM tokenized spacing kept), no normalization, newline
replaced by a space.

Input : <hf>/data/{train,validation,test}.parquet   (columns: text, image{bytes,path})
Output: <out>/kraken/<split>/NNNNNNNN.png and NNNNNNNN.gt.txt, <out>/manifest_<split>.tsv, <out>/conversion_stats.json
Usage : python -I convert_teklia_iam.py <hf dir> <out dir>
"""
import io, json, os, sys, unicodedata as ud
from collections import Counter
import pyarrow.parquet as pq
from PIL import Image

hf, out = sys.argv[1], sys.argv[2]
stats = {}
for split in ("train", "validation", "test"):
    t = pq.read_table(os.path.join(hf, "data", f"{split}.parquet"))
    imgs, texts = t.column("image").to_pylist(), t.column("text").to_pylist()
    odir = os.path.join(out, "kraken", split)
    os.makedirs(odir, exist_ok=True)
    W, H, modes, charset = [], Counter(), Counter(), set()
    n_nl = n_ws = n_empty = 0
    man = ["index\tfile\ttext\twidth\theight\torig_mode"]
    for i, (im, txt) in enumerate(zip(imgs, texts)):
        img = Image.open(io.BytesIO(im["bytes"]))
        modes[img.mode] += 1
        g = img.convert("L")
        W.append(g.width); H[g.height] += 1
        name = f"{i:08d}"
        g.save(os.path.join(odir, name + ".png"))
        txt = txt or ""
        if "\n" in txt or "\r" in txt:
            n_nl += 1; txt = txt.replace("\r", " ").replace("\n", " ")
        n_ws += txt != txt.strip(); n_empty += not txt.strip()
        with open(os.path.join(odir, name + ".gt.txt"), "w", encoding="utf-8") as fp:
            fp.write(txt)
        charset.update(txt)
        man.append(f"{i}\t{name}.png\t{txt.replace(chr(9), ' ')}\t{g.width}\t{g.height}\t{img.mode}")
    open(os.path.join(out, f"manifest_{split}.tsv"), "w", encoding="utf-8").write("\n".join(man) + "\n")
    n = len(texts)
    stats[split] = dict(lines=n, chars_total=sum(map(len, texts)), chars_nfd=sum(len(ud.normalize("NFD", x)) for x in texts),
                        charset_size=len(charset), texts_with_newline=n_nl, texts_with_edge_whitespace=n_ws, texts_empty=n_empty,
                        orig_modes=dict(modes), heights=dict(H), width_min=min(W), width_max=max(W), width_mean=round(sum(W) / n, 1))
    print(split, json.dumps(stats[split]))
stats["source"] = dict(dataset="Teklia/IAM-line", revision=os.environ.get("HF_REVISION", ""), split="Aachen (train / validation / test)")
json.dump(stats, open(os.path.join(out, "conversion_stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
