#!/usr/bin/env python
"""TrOCR line recognition on the Kraken arrow datasets of the thesis (same files, same splits as the Kraken runs).
Usage:
  python trocr_run.py test  --model <hf id | dir> --test <arrow> --out <dir> [--device cuda:0]
  python trocr_run.py train --model <hf id | dir> --train <arrow> [<arrow> ...] --val <arrow> --out <dir> --seed <S>
                            [--lr 5e-5] [--batch 8] [--warmup 100] [--lag 5] [--min-epochs 3] [--max-epochs 40]
Text is NFC-normalised for training. CER is computed like `ketos test -u NFD`: reference and prediction in NFD,
Characters = total NFD reference characters, Errors = total Levenshtein distance; case-sensitive and case-insensitive
(.lower()). Greedy decoding. Early stopping on validation CER-cs (lag epochs without improvement after min-epochs);
only the best model is kept (out/best/, with val.json). Each epoch is appended to out/train_log.jsonl.
"""
import argparse, io, json, math, os, random, re, sys, time, unicodedata
import pyarrow as pa
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

MAX_LEN = 64
GEN = {}          # extra generation kwargs (e.g. no_repeat_ngram_size), set from the command line


def norm_text(t):
    """Like ketos: any whitespace -> space, strip; then NFC."""
    return unicodedata.normalize("NFC", re.sub(r"\s", " ", t).strip())


def seed_all(s):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    try:
        import numpy as np; np.random.seed(s)
    except ImportError:
        pass


class LineDS(Dataset):
    """Lines of one or more Kraken binary arrows (column lines: struct<text, im>); all rows of each file are used."""
    def __init__(self, paths, proc):
        self.items, self.proc = [], proc
        for p in paths:
            for ch in pa.ipc.open_file(p).read_all().column("lines").chunks:   # per chunk: combine_chunks() overflows the 2 GB binary offsets on big pools
                self.items.append((ch.field("im"), ch.field("text")))
        self.index = [(k, i) for k, (ims, _) in enumerate(self.items) for i in range(len(ims))]

    def __len__(self):
        return len(self.index)

    def text(self, j):
        k, i = self.index[j]
        return norm_text(self.items[k][1][i].as_py())

    def __getitem__(self, j):
        k, i = self.index[j]
        im = Image.open(io.BytesIO(self.items[k][0][i].as_py())).convert("RGB")
        return self.proc(images=im, return_tensors="pt").pixel_values[0], self.text(j)


def collate_eval(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


class CollateTrain:
    def __init__(self, proc):
        self.tok = proc.tokenizer

    def __call__(self, batch):
        px = torch.stack([b[0] for b in batch])
        enc = self.tok([b[1] for b in batch], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        return px, enc.input_ids.masked_fill(enc.attention_mask == 0, -100)


def edit_distance(a, b):
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer_stats(refs, hyps):
    n = e = e_ci = nw = ew = 0
    for r, h in zip(refs, hyps):
        r, h = unicodedata.normalize("NFD", r), unicodedata.normalize("NFD", re.sub(r"\s", " ", h).strip())
        n += len(r); e += edit_distance(r, h); e_ci += edit_distance(r.lower(), h.lower())
        nw += len(r.split()); ew += edit_distance(r.split(), h.split())
    return dict(chars=n, errors=e, errors_ci=e_ci, words=nw, word_errors=ew,
                cer_cs=100 * e / n, cer_ci=100 * e_ci / n, wer_cs=100 * ew / max(nw, 1))


@torch.no_grad()
def predict(model, proc, ds, dev, batch=16):
    model.eval(); hyps = []
    for px, _ in DataLoader(ds, batch_size=batch, shuffle=False, num_workers=4, collate_fn=collate_eval):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(px.to(dev), max_new_tokens=MAX_LEN, num_beams=1, do_sample=False, **GEN)
        hyps += proc.batch_decode(out, skip_special_tokens=True)
    return [h.strip() for h in hyps]


def load(model_path, dev):
    proc = TrOCRProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    meta = [n for n, p in model.named_parameters() if p.device.type == "meta"]   # weights absent from the checkpoint (transformers 5)
    if meta:
        if all(n.startswith("encoder.pooler.") for n in meta):
            model.encoder.pooler = None   # ViT pooler missing in trocr-base-stage1; unused by the seq2seq head
        else:
            raise RuntimeError(f"uninitialised parameters: {meta}")
    model = model.to(dev)
    for mod in model.modules():   # sinusoidal position table of trocr-base-stage1: a plain tensor attribute that transformers 5
        w = getattr(mod, "weights", None)   # leaves on the meta device at load and never moves with the model
        if isinstance(w, torch.Tensor) and hasattr(mod, "get_embedding"):
            if w.device.type == "meta":
                w = mod.get_embedding(w.size(0), mod.embedding_dim, mod.padding_idx)
            mod.weights = w.to(dev)
    # transformers 5 drops generation ids from the top-level config, but forward() still reads them there to shift labels
    dec = model.config.decoder
    dsid, pid = getattr(dec, "decoder_start_token_id", None), getattr(dec, "pad_token_id", None)
    model.config.decoder_start_token_id = dsid if dsid is not None else proc.tokenizer.eos_token_id
    model.config.pad_token_id = pid if pid is not None else proc.tokenizer.pad_token_id
    return proc, model


def cmd_test(a):
    dev = torch.device(a.device); proc, model = load(a.model, dev)
    ds = LineDS([a.test], proc); refs = [ds.text(j) for j in range(len(ds))]
    t0 = time.time(); hyps = predict(model, proc, ds, dev)
    st = cer_stats(refs, hyps)
    st.update(model=a.model, test=a.test, lines=len(refs), chars_nfc=sum(len(r) for r in refs),
              decoding="greedy" + (f" no_repeat_ngram_size={GEN['no_repeat_ngram_size']}" if GEN else ""), normalization="NFD", seconds=round(time.time() - t0), date=time.strftime("%Y-%m-%dT%H:%M:%S"))
    os.makedirs(a.out, exist_ok=True)
    json.dump(st, open(os.path.join(a.out, "test_results.json"), "w"), indent=1, ensure_ascii=False)
    with open(os.path.join(a.out, "test_predictions.tsv"), "w", encoding="utf-8") as f:
        f.write("reference\tprediction\n")
        for r, h in zip(refs, hyps):
            f.write(f"{r}\t{h}\n")
    print(f"{st['chars']}\tCharacters\n{st['errors']}\tErrors\n{100 - st['cer_cs']:.2f}%\tCharacter Accuracy\n"
          f"{100 - st['cer_ci']:.2f}%\tCharacter Accuracy (Case-insensitive)\n{100 - st['wer_cs']:.2f}%\tWord Accuracy")
    print(f"CER-cs {st['cer_cs']:.2f}  CER-ci {st['cer_ci']:.2f}  WER-cs {st['wer_cs']:.2f}  "
          f"({len(refs)} lines, {st['chars_nfc']} NFC chars, {st['seconds']} s) -> {a.out}/test_results.json")


def cmd_train(a):
    seed_all(a.seed); dev = torch.device(a.device); proc, model = load(a.model, dev)
    if a.grad_ckpt:
        model.gradient_checkpointing_enable()
    tr, va = LineDS(a.train, proc), LineDS([a.val], proc)
    g = torch.Generator().manual_seed(a.seed)
    dl = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=4, collate_fn=CollateTrain(proc), generator=g)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps_total = max(1, (len(dl) // a.accum) * a.max_epochs)
    warm = max(1, min(a.warmup, steps_total // 10))
    def lr_lambda(s):   # linear warmup, then constant or linear decay to 0 at max_epochs
        if s < warm:
            return (s + 1) / warm
        return 1.0 if a.schedule == "constant" else max(0.0, (steps_total - s) / max(1, steps_total - warm))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    os.makedirs(a.out, exist_ok=True)
    info = dict({k: v for k, v in vars(a).items() if k != "f"}, train_lines=len(tr), val_lines=len(va), decoder_start_token_id=model.config.decoder_start_token_id, pad_token_id=model.config.pad_token_id, steps_per_epoch=len(dl), date=time.strftime("%Y-%m-%dT%H:%M:%S"))
    json.dump(info, open(os.path.join(a.out, "run_info.json"), "w"), indent=1)
    print(json.dumps(info), flush=True)
    val_refs = [va.text(j) for j in range(len(va))]
    best, best_ep, bad, log = math.inf, -1, 0, open(os.path.join(a.out, "train_log.jsonl"), "a")
    for ep in range(a.max_epochs):
        model.train(); t0 = time.time(); tot = 0.0
        for i, (px, labels) in enumerate(dl, 1):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(pixel_values=px.to(dev), labels=labels.to(dev)).loss
            (loss / a.accum).backward(); tot += loss.item()
            if i % a.accum == 0 or i == len(dl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        st = cer_stats(val_refs, predict(model, proc, va, dev))
        rec = dict(epoch=ep, train_loss=round(tot / len(dl), 4), val_cer_cs=round(st["cer_cs"], 4),
                   val_cer_ci=round(st["cer_ci"], 4), seconds=round(time.time() - t0), lr=sched.get_last_lr()[0],
                   gpu_gb=round(torch.cuda.max_memory_allocated(dev) / 2**30, 1))
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        if st["cer_cs"] < best:
            best, best_ep, bad = st["cer_cs"], ep, 0
            model.save_pretrained(os.path.join(a.out, "best")); proc.save_pretrained(os.path.join(a.out, "best"))
            json.dump(dict(best_epoch=ep, val_cer_cs=st["cer_cs"], val_cer_ci=st["cer_ci"], val_lines=len(va)),
                      open(os.path.join(a.out, "best", "val.json"), "w"), indent=1)
        else:
            bad += 1
            if ep + 1 >= a.min_epochs and bad >= a.lag:
                break
    print(f"DONE best epoch {best_ep} val CER-cs {best:.2f} -> {a.out}/best", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("test"); t.add_argument("--model", required=True); t.add_argument("--test", required=True)
    t.add_argument("--out", required=True); t.add_argument("--device", default="cuda:0"); t.add_argument("--no-repeat-ngram", type=int, default=0); t.set_defaults(f=cmd_test)
    r = sp.add_parser("train"); r.add_argument("--model", required=True); r.add_argument("--train", nargs="+", required=True)
    r.add_argument("--val", required=True); r.add_argument("--out", required=True); r.add_argument("--seed", type=int, required=True)
    r.add_argument("--lr", type=float, default=5e-5); r.add_argument("--batch", type=int, default=8)
    r.add_argument("--warmup", type=int, default=100); r.add_argument("--lag", type=int, default=5)
    r.add_argument("--min-epochs", type=int, default=3); r.add_argument("--max-epochs", type=int, default=40)
    r.add_argument("--accum", type=int, default=1); r.add_argument("--schedule", choices=["constant", "linear"], default="constant")
    r.add_argument("--no-repeat-ngram", type=int, default=0); r.add_argument("--grad-ckpt", action="store_true"); r.add_argument("--device", default="cuda:0"); r.set_defaults(f=cmd_train)
    a = ap.parse_args()
    if getattr(a, "no_repeat_ngram", 0):
        GEN["no_repeat_ngram_size"] = a.no_repeat_ngram
    a.f(a)
