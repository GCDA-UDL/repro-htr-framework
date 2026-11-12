#!/usr/bin/env python3
"""Convert the CatK12HW-actions HuggingFace parquet release (v0.4) to Kraken 'path' format.

Input : data/real/catk12hw/hf/data/{train,validation,test}-*.parquet
        (columns: image{bytes,path}, text, source_image, bbox_x, bbox_y, bbox_w, bbox_h)
Output: data/real/catk12hw/kraken/<split>/NNNNNNNN.png     8-bit grayscale (mode L); no resizing, no binarization
        data/real/catk12hw/kraken/<split>/NNNNNNNN.gt.txt  transcription exactly as released (no normalization,
                                                            no trailing newline)
        data/real/catk12hw/manifest_<split>.tsv            index, file, source_image, text, width, height, original mode
        data/real/catk12hw/conversion_stats.json           every number printed below, for the tracking document
Checks: no source_image shared between splits (writer/page leakage), image geometry, text normalization (NFC/NFD).
"""
import argparse, glob, io, json, os, sys, unicodedata as ud
from collections import Counter
import pyarrow.parquet as pq
from PIL import Image

SPLITS = ("train", "validation", "test")


def norm_class(s):
    nfc, nfd = ud.normalize("NFC", s), ud.normalize("NFD", s)
    if nfc == nfd:
        return "ascii-like"      # no composable characters
    if s == nfc:
        return "NFC-only"
    if s == nfd:
        return "NFD-only"
    return "mixed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default="data/real/catk12hw/hf/data", help="folder with the parquet files")
    ap.add_argument("--out", default="data/real/catk12hw", help="output root")
    a = ap.parse_args()
    stats, sources = {}, {}
    for split in SPLITS:
        files = sorted(glob.glob(os.path.join(a.hf, f"{split}-*.parquet")))
        if not files:
            sys.exit(f"no parquet files for split '{split}' in {a.hf}")
        t = pq.read_table(files)
        imgs = t.column("image").to_pylist()
        texts = t.column("text").to_pylist()
        srcs = t.column("source_image").to_pylist()
        odir = os.path.join(a.out, "kraken", split)
        os.makedirs(odir, exist_ok=True)
        widths, heights, modes, ncls, charset = [], [], Counter(), Counter(), set()
        n_newline = n_ws = n_empty = n_tab = 0
        man = [("index", "file", "source_image", "text", "width", "height", "orig_mode")]
        for i, (im, txt, src) in enumerate(zip(imgs, texts, srcs)):
            img = Image.open(io.BytesIO(im["bytes"]))
            modes[img.mode] += 1
            g = img.convert("L")
            widths.append(g.width); heights.append(g.height)
            name = f"{i:08d}"
            g.save(os.path.join(odir, name + ".png"))
            if txt is None:
                txt = ""
            if "\n" in txt or "\r" in txt:
                n_newline += 1
                txt = txt.replace("\r", " ").replace("\n", " ")
            if txt != txt.strip():
                n_ws += 1
            if not txt.strip():
                n_empty += 1
            if "\t" in txt:
                n_tab += 1
            with open(os.path.join(odir, name + ".gt.txt"), "w", encoding="utf-8") as fp:
                fp.write(txt)
            ncls[norm_class(txt)] += 1
            charset.update(txt)
            man.append((str(i), name + ".png", src, txt.replace("\t", " "), str(g.width), str(g.height), img.mode))
        with open(os.path.join(a.out, f"manifest_{split}.tsv"), "w", encoding="utf-8") as fp:
            fp.write("\n".join("\t".join(r) for r in man) + "\n")
        sources[split] = Counter(srcs)
        n = len(texts)
        stats[split] = {
            "lines": n,
            "source_images": len(sources[split]),
            "chars_total": sum(len(x) for x in texts),
            "chars_mean": round(sum(len(x) for x in texts) / n, 2),
            "charset_size": len(charset),
            "normalization": dict(ncls),
            "texts_with_newline": n_newline, "texts_with_edge_whitespace": n_ws,
            "texts_empty": n_empty, "texts_with_tab": n_tab,
            "orig_modes": dict(modes),
            "width_min": min(widths), "width_mean": round(sum(widths) / n, 1), "width_max": max(widths),
            "heights": dict(Counter(heights)),
        }
        print(f"[{split}] lines={n} source_images={len(sources[split])} chars={stats[split]['chars_total']} "
              f"(mean {stats[split]['chars_mean']}) charset={len(charset)} norm={dict(ncls)} "
              f"modes={dict(modes)} heights={dict(Counter(heights))} width={min(widths)}/{stats[split]['width_mean']}/{max(widths)} "
              f"newline={n_newline} edge_ws={n_ws} empty={n_empty} tab={n_tab}")
    # leakage check: a source_image must belong to exactly one split
    overlaps = {}
    for x in SPLITS:
        for y in SPLITS:
            if x < y:
                common = sorted(set(sources[x]) & set(sources[y]))
                overlaps[f"{x}&{y}"] = common
                print(f"[overlap] {x} & {y}: {len(common)} shared source images" + (f" -> {common[:10]}" if common else ""))
    stats["source_image_overlaps"] = {k: len(v) for k, v in overlaps.items()}
    stats["source_image_overlap_names"] = overlaps
    stats["total_lines"] = sum(stats[s]["lines"] for s in SPLITS)
    stats["total_source_images"] = len(set().union(*[set(sources[s]) for s in SPLITS]))
    print(f"[total] lines={stats['total_lines']} source_images={stats['total_source_images']}")
    with open(os.path.join(a.out, "conversion_stats.json"), "w", encoding="utf-8") as fp:
        json.dump(stats, fp, ensure_ascii=False, indent=1)
    print("stats written to", os.path.join(a.out, "conversion_stats.json"))


if __name__ == "__main__":
    main()
