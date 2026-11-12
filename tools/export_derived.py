#!/usr/bin/env python
"""Export the numeric, text-free derivatives of the private CatK12HW-actions evaluation that the analysis needs.

Runs only where the private data lives (the experiment server); its outputs are what this package ships instead of
the corpus. Nothing written here contains an image, a transcription or a prediction:
  results/perline/lines.csv        one row per test line (642): pseudonymous page index (0-23, order of the sorted page
                                   identifiers) and the stratum of the line under every grouping used in the article
  results/perline/runs/<cell>.csv  one row per (run, test line): characters, CER-cs errors, CER-ci errors, words and
                                   word errors, scored exactly as `ketos test -u NFD` (Kraken: cost and nchars of the
                                   saved predictions) or as code/trocr/trocr_run.py (TrOCR: Levenshtein in NFD)
  results/perline/g3_words.json    word coverage of the test lines whose text is absent from the training set
  results/corpus/catk12hw.json     corpus statistics of Table I and the text coverage of Tables VI and IX
Usage (server only): python tools/export_derived.py --root /home/joc13/workrepo-htr-synthetic-framework
"""
import argparse, csv, glob, json, os, re, unicodedata
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ROOT = ap.parse_args().root
OUT = os.path.join(HERE, "results")
PRED = [f"{ROOT}/audit_round2/results/predictions", f"{ROOT}/audit_round3/results/predictions"]
CAT = f"{ROOT}/data/real/catk12hw"


def norm(s):
    return " ".join(unicodedata.normalize("NFC", s).split())


def ed(a, b):
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def arrow_texts(p):
    out = []
    with pa.memory_map(p) as src:
        r = pa.ipc.open_file(src)
        for i in range(r.num_record_batches):
            b = r.get_batch(i)
            out += b.column(b.schema.get_field_index("lines")).field("text").to_pylist()
    return [norm(s) for s in out]


man = list(csv.DictReader(open(f"{CAT}/manifest_test.tsv", encoding="utf-8"), delimiter="\t"))
L = len(man)
pages = sorted(set(r["source_image"] for r in man))
page_of = [pages.index(r["source_image"]) for r in man]
test_text = [norm(r["text"]) for r in man]


def per_line(tg, pr, nch=None, cost=None):
    rows = []
    for i, (t, p) in enumerate(zip(tg, pr)):
        rows.append(dict(line=i, n=nch[i] if nch is not None else len(t), e=cost[i] if cost is not None else ed(t, p),
                         eci=ed(t.lower(), p.lower()), w=len(t.split()), ew=ed(t.split(), p.split())))
    return rows


def kraken(prefix):
    runs = []
    for d in PRED:
        for f in sorted(glob.glob(f"{d}/{prefix}_seed*.tsv")):
            if not re.fullmatch(re.escape(prefix) + r"_seed\d+\.tsv", os.path.basename(f)):
                continue
            rows = list(csv.reader(open(f, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\"))[1:]
            assert len(rows) == L, f
            runs.append((os.path.basename(f)[:-4], per_line([r[0] for r in rows], [r[1] for r in rows],
                                                            [int(r[3]) for r in rows], [int(r[2]) for r in rows])))
    return runs


def trocr(cell):
    root, block, prefix = cell.split("/")
    runs = []
    mt = [unicodedata.normalize("NFC", re.sub(r"\s", " ", r["text"]).strip()) for r in man]
    for d in sorted(glob.glob(f"{ROOT}/models/{root}/{block}/{prefix}_seed*")):
        if not re.fullmatch(re.escape(prefix) + r"_seed\d+", os.path.basename(d)):
            continue
        tp, tj = f"{d}/test_predictions.tsv", f"{d}/test_results.json"
        if not (os.path.exists(tp) and os.path.exists(tj)):
            continue
        refs, hyps = [], []
        with open(tp, encoding="utf-8") as fh:
            next(fh)
            for line in fh:
                line = line.rstrip("\n")
                r, h = line.split("\t", 1) if "\t" in line else (line, "")
                refs.append(r); hyps.append(h)
        assert len(refs) == L and all(r == m for r, m in zip(refs, mt)), d
        tg = [unicodedata.normalize("NFD", r) for r in refs]
        pr = [unicodedata.normalize("NFD", re.sub(r"\s", " ", h).strip()) for h in hyps]
        rows = per_line(tg, pr)
        js = json.load(open(tj))
        tot = tuple(sum(x[k] for x in rows) for k in ("n", "e", "eci", "w", "ew"))
        assert tot == (js["chars"], js["errors"], js["errors_ci"], js["words"], js["word_errors"]), d
        runs.append((f"{root}/{block}/{os.path.basename(d)}", rows))
    return runs


K = {"RO 500": "c2_ca_real-only_500", "RO+aug 500": "c2_ca_real-only_500_aug", "HWxgen 500": "c2_ca_curriculum_500",
     "Kidsxgen 500": "c2_ca_curriculum_500_kids", "HWxtxt 500": "c2_ca_curriculum_500_text", "kidstext 500": "c2_ca_curriculum_500_kidstext",
     "kidstext x5 500": "c2_ca_curriculum_500_kidstext30k", "kidstext-pert 500": "c2_ca_curriculum_500_kidstextpert",
     "pert+aug F2 500": "c2_ca_curriculum_500_kidstextpert-aug", "pert+aug F1 500": "c2_ca_curriculum_500_kidstextpert-augp1",
     "S1 500": "r2_ca_curriculum_500_kidstextpert500",
     "S1f 500": "ca_curriculum_500_s1_fixed20", "S2f 500": "ca_curriculum_500_s2_fixed20",
     "S2-415ext 500": "ca_curriculum_500_s2x415_fixed20", "S2-extall 500": "ca_curriculum_500_s2xall_fixed20",
     "RO 1k": "c2_ca_real-only_1k", "kidstext 1k": "c2_ca_curriculum_1k_kidstext", "kidstext-pert 1k": "c2_ca_curriculum_1k_kidstextpert",
     "RO 2k": "c2_ca_real-only_2k", "kidstext 2k": "c2_ca_curriculum_2k_kidstext", "kidstext-pert 2k": "c2_ca_curriculum_2k_kidstextpert",
     "RO full": "c2_ca_real-only", "RO+aug full": "c2_ca_real-only_aug", "HWxgen full": "c2_ca_curriculum",
     "Kidsxgen full": "c2_ca_curriculum_kids", "HWxtxt full": "c2_ca_curriculum_text", "kidstext full": "c2_ca_curriculum_kidstext",
     "kidstext x5 full": "c2_ca_curriculum_kidstext30k", "kidstext-pert full": "c2_ca_curriculum_kidstextpert",
     "pert+aug F2 full": "c2_ca_curriculum_kidstextpert-aug", "pert+aug F1 full": "c2_ca_curriculum_kidstextpert-augp1",
     "mix full": "c2_ca_mix", "curriculum-LR full": "c2_ca_curriculum-LR",
     "scratch RO 500": "c2_ca_real-only_500_scratchr2", "scratch cur 500": "c2_ca_curriculum_500_scratchr2-white30k",
     "scratch RO full": "c2_ca_real-only_scratchr2", "scratch cur full": "c2_ca_curriculum_scratchr2-white30k",
     "scratch curpert full": "c2_ca_curriculum_scratchr2-whitepert30k"}
T = {"TrOCR-hw RO 500": "trocr_r2/blockC/ca_real-only_500", "TrOCR-hw kidstext 500": "trocr_r2/blockC/ca_curriculum_500_kidstext",
     "TrOCR-hw kidstext-pert 500": "trocr_r2/blockC/ca_curriculum_500_kidstextpert", "TrOCR-hw x5 500": "trocr_r2/blockC/ca_curriculum_500_kidstext30k",
     "TrOCR-hw RO full": "trocr_r2/blockA/ca_real-only", "TrOCR-hw kidstext full": "trocr_r2/blockA/ca_curriculum_kidstext",
     "TrOCR-hw kidstext-pert full": "trocr_r2/blockA/ca_curriculum_kidstextpert", "TrOCR-hw x5 full": "trocr_r2/blockA/ca_curriculum_kidstext30k",
     "TrOCR-s1 RO 500": "trocr_s1/blockC/ca_real-only_500", "TrOCR-s1 kidstext 500": "trocr_s1/blockC/ca_curriculum_500_kidstext",
     "TrOCR-s1 RO full": "trocr_s1/blockA/ca_real-only", "TrOCR-s1 kidstext full": "trocr_s1/blockA/ca_curriculum_kidstext"}

os.makedirs(f"{OUT}/perline/runs", exist_ok=True)
os.makedirs(f"{OUT}/corpus", exist_ok=True)
index = []
for name, src in list(K.items()) + list(T.items()):
    runs = kraken(src) if name in K else trocr(src)
    if not runs:
        print("no runs:", name); continue
    fn = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") + ".csv"
    with open(f"{OUT}/perline/runs/{fn}", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n"); w.writerow(["run", "line", "n", "e", "eci", "w", "ew"])
        for k, (rn, rows) in enumerate(runs):
            for x in rows:
                w.writerow([k, x["line"], x["n"], x["e"], x["eci"], x["w"], x["ew"]])
    index.append(dict(cell=name, file=fn, model="kraken" if name in K else "trocr", runs=[rn for rn, _ in runs]))
json.dump(index, open(f"{OUT}/perline/cells.json", "w"), indent=1)

# ---------------------------------------------------------------- strata labels (text-free)
TRAIN = set(arrow_texts(f"{CAT}/catk12hw_train.arrow"))
sub = {n: arrow_texts(f"{CAT}/catk12hw_{n}_train.arrow") for n in ("500", "1k", "2k")}
pool_files = {
    "kidstext": "data/synthetic/ca_kidstext_white_6k/ca_kidstext_white_4152_train_bl.arrow",
    "kidstextpert": "data/synthetic/ca_kidstextpert_white_6k/ca_kidstextpert_white_4152_train_bl.arrow",
    "kidstext30k": "data/synthetic/ca_kidstext_white_30k/ca_kidstext_white_30k_train_bl.arrow",
    "s1": "audit_round2/data/synthetic/ca_kidstextpert500_white_6k/ca_kidstextpert500_white_4152_train_bl.arrow",
    "s2x415": "audit_round3/data/synthetic/ca_kidstextpert415ext_white_6k/ca_kidstextpert415ext_white_4152_train_bl.arrow",
    "s2xall": "audit_round3/data/synthetic/ca_kidstextpertextall_white_6k/ca_kidstextpertextall_white_4152_train_bl.arrow"}
pools = {k: arrow_texts(f"{ROOT}/{v}") for k, v in pool_files.items()}
pools["s2"] = pools["kidstextpert"]


def groups(subt, pool, ci_):
    f = (lambda s: s.lower()) if ci_ else (lambda s: s)
    S, Pl, Tr = {f(t) for t in subt}, {f(t) for t in pool}, {f(t) for t in TRAIN}
    return ["G1" if f(t) in S else "G2" if f(t) in Pl else "G2b" if f(t) in Tr else "G3" for t in test_text]


cols = {}
for pool in ("kidstext", "kidstextpert", "kidstext30k", "s1", "s2", "s2x415", "s2xall"):
    for ci_ in (False, True):
        cols[f"single_{pool}_{'ci' if ci_ else 'cs'}"] = groups(sub["500"], pools[pool], ci_)
voc500 = {w for t in sub["500"] for w in t.split()}
for ci_ in (False, True):
    f = (lambda s: s.lower()) if ci_ else (lambda s: s)
    S500, TR = {f(t) for t in sub["500"]}, {f(t) for t in TRAIN}
    cols[f"multi_{'ci' if ci_ else 'cs'}"] = ["G1" if f(t) in S500 else "G2'" if f(t) in TR else "G3" for t in test_text]
vs2 = voc500 | {w for t in pools["s2"] for w in t.split()}
cols["g3_new_word"] = [int(g == "G3" and any(w not in vs2 for w in t.split())) for g, t in zip(cols["multi_cs"], test_text)]
with open(f"{OUT}/perline/lines.csv", "w", newline="") as fh:
    w = csv.writer(fh, lineterminator="\n"); w.writerow(["line", "page"] + list(cols))
    for i in range(L):
        w.writerow([i, page_of[i]] + [cols[c][i] for c in cols])

S500 = set(sub["500"]); g = ["G1" if t in S500 else "G2'" if t in TRAIN else "G3" for t in test_text]
tok = [w for t, gg in zip(test_text, g) if gg == "G3" for w in t.split()]
vtr = {w for t in TRAIN for w in t.split()}
g3 = {}
for lab, v in (("500", voc500), ("500 (sin caja)", {w.lower() for w in voc500}), ("train", vtr), ("500 + S2", vs2)):
    g3[lab] = 100 * sum((w.lower() if "sin caja" in lab else w) in v for w in tok) / len(tok)
g3["lines"] = sum(gg == "G3" for gg in g)
g3["lines_ci_in_train"] = sum(gg == "G3" and t.lower() in {x.lower() for x in TRAIN} for gg, t in zip(g, test_text))
g3["lines_ci_in_500"] = sum(gg == "G3" and t.lower() in {x.lower() for x in S500} for gg, t in zip(g, test_text))
json.dump(g3, open(f"{OUT}/perline/g3_words.json", "w"), indent=1, ensure_ascii=False)

# ---------------------------------------------------------------- corpus statistics and text coverage (Tables I, VI, IX)
C = {"splits": {}}
for sp in ("train", "validation", "test"):
    t = pq.read_table(f"{CAT}/hf/data/{sp}-00000-of-00001.parquet", columns=["text", "source_image"])
    tx = t.column("text").to_pylist(); pg = t.column("source_image").to_pylist()
    nt = [norm(x) for x in tx]
    up = sum(1 for x in nt if any(c.isalpha() for c in x) and x == x.upper())
    C["splits"][sp] = dict(lines=len(tx), pages=len(set(pg)), distinct=len(set(nt)), distinct_ci=len(set(x.lower() for x in nt)),
                           chars_nfc=sum(len(unicodedata.normalize("NFC", x)) for x in tx),
                           mean_len=sum(len(x) for x in tx) / len(tx), upper_pct=100 * up / len(tx))
    C["splits"][sp]["_pages"] = set(pg)
P = {k: v.pop("_pages") for k, v in C["splits"].items()}
C["page_overlaps"] = {f"{a}&{b}": len(P[a] & P[b]) for a in P for b in P if a < b}
C["writer_ids_available"] = False
C["test_chars_nfd_ketos"] = sum(int(r["n"]) for r in csv.DictReader(open(f"{OUT}/perline/runs/RO_full.csv")) if r["run"] == "0")
tl = test_text
TRl = {t.lower() for t in TRAIN}
C["test_text_in_train_pct"] = 100 * sum(t in TRAIN for t in tl) / L
C["test_text_in_train_ci_pct"] = 100 * sum(t.lower() in TRl for t in tl) / L
C["test_words_in_train_vocab_pct"] = 100 * sum(w in vtr for t in tl for w in t.split()) / sum(len(t.split()) for t in tl)
cov = {}
for n in ("500", "1k", "2k"):
    cov[f"real {n}"] = 100 * sum(t in set(sub[n]) for t in tl) / L
    for p in ("kidstext", "kidstextpert", "s1", "s2x415", "s2xall"):
        U = set(sub[n]) | set(pools[p])
        cov[f"real {n} + {p}"] = 100 * sum(t in U for t in tl) / L
cov["real full"] = C["test_text_in_train_pct"]
C["test_text_seen_pct"] = cov
C["synthetic_subsample_distinct_texts"] = {p: len(set(pools[p])) for p in ("kidstext", "kidstextpert", "s1", "s2x415", "s2xall")}
json.dump(C, open(f"{OUT}/corpus/catk12hw.json", "w"), indent=1)
print("cells", len(index), "lines", L, "pages", len(pages))

# ---------------------------------------------------------------- ketos test summaries (counts only) and phase-1 checkpoints
SUM = re.compile(r"^\s*([\d.]+)%?\s+(Characters|Errors|Insertions|Deletions|Substitutions|Character Accuracy|Character Accuracy \(Case-insensitive\)|Word Accuracy)\s*$")
rows = []
for f in sorted(glob.glob(f"{ROOT}/models/**/eval_full.log", recursive=True)) + sorted(glob.glob(f"{ROOT}/audit_round*/results/eval_logs/*.eval_full.log")):
    run = os.path.relpath(f, ROOT)
    if re.search(r"(^|/)es_|_aborted", run) or "/vin" in run:
        continue   # runs of the fourth (unreported) setting and aborted runs are not part of the package
    d = {}
    for line in open(f, errors="replace"):
        m = SUM.match(line)
        if m and m.group(2) not in d:
            d[m.group(2)] = m.group(1)
    if "Characters" in d:
        rows.append(dict(run=run, characters=d["Characters"], errors=d.get("Errors", ""), ketos_insertions=d.get("Insertions", ""),
                         ketos_deletions=d.get("Deletions", ""), substitutions=d.get("Substitutions", ""),
                         char_accuracy=d.get("Character Accuracy", ""), char_accuracy_ci=d.get("Character Accuracy (Case-insensitive)", ""),
                         word_accuracy=d.get("Word Accuracy", "")))
os.makedirs(f"{OUT}/kraken", exist_ok=True)
with open(f"{OUT}/kraken/ketos_test_summaries.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)
ck = []
for f in sorted(glob.glob(f"{ROOT}/models/blockC/ca_curriculum_*_seed*/run_info.txt")) + sorted(glob.glob(f"{ROOT}/audit_round2/models/*/run_info.txt")):
    for line in open(f):
        m = re.search(r"size=(\S+) phase1=(\S+)", line)
        if m:
            e = re.search(r"checkpoint_(\d+)-([\d.]+)\.ckpt", m.group(2))
            ck.append(dict(run=os.path.relpath(os.path.dirname(f), ROOT), size=m.group(1), phase1_checkpoint=os.path.basename(m.group(2)),
                           phase1_epoch=e.group(1) if e else "", phase1_val_acc=e.group(2) if e else ""))
with open(f"{OUT}/kraken/phase1_checkpoints.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(ck[0]), lineterminator="\n"); w.writeheader(); w.writerows(ck)
print("ketos summaries", len(rows), "phase-1 checkpoints", len(ck))

# ---------------------------------------------------------------- TrOCR test results (aggregate metrics only)
tr = []
for f in sorted(glob.glob(f"{ROOT}/models/trocr_r2/block*/*/test_results.json")) + sorted(glob.glob(f"{ROOT}/models/trocr_s1/block*/*/test_results.json")):
    d = json.load(open(f))
    tr.append(dict(run=os.path.relpath(os.path.dirname(f), f"{ROOT}/models"), **{k: d[k] for k in ("chars", "errors", "errors_ci", "words", "word_errors", "cer_cs", "cer_ci", "wer_cs", "lines", "decoding", "normalization")}))
os.makedirs(f"{OUT}/trocr", exist_ok=True)
with open(f"{OUT}/trocr/test_results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(tr[0]), lineterminator="\n"); w.writeheader(); w.writerows(tr)
print("trocr runs", len(tr))
