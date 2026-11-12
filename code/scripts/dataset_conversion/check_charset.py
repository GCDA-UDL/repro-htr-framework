#!/usr/bin/env python3
"""
check_charset.py — Extract and display all unique characters in a dataset.

Supports two input formats:
- .arrow files (Kraken compiled datasets)
- metadata.jsonl files (HuggingFace-style datasets from the framework)

Optionally compares two datasets to detect character mismatches, which is
useful before training when you want to verify that the synthetic and real
datasets share a compatible character set.

Usage:
    # Show character set of a single dataset
    python check_charset.py --input data/real/rimes/rimes_test.arrow

    # Compare two datasets
    python check_charset.py \\
      --input data/synthetic/fr_page/fr_synth_10k_train.arrow \\
      --compare data/real/rimes/rimes_only_train.arrow
"""

import argparse
import json
import sys
from pathlib import Path


def read_texts_from_arrow(path):
    """Read all text values from a Kraken-style .arrow file."""
    import pyarrow.ipc as ipc
    texts = []
    with open(path, "rb") as f:
        reader = ipc.open_file(f)
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            for j in range(batch.num_rows):
                row = batch.column("lines")[j].as_py()
                text = row.get("text", "") or ""
                texts.append(text)
    return texts


def read_texts_from_jsonl(path):
    """Read all text values from a HuggingFace metadata.jsonl file."""
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            texts.append(data.get("text", ""))
    return texts


def read_texts(path):
    """Auto-detect format and read texts."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix == ".arrow":
        return read_texts_from_arrow(path)
    elif path.suffix == ".jsonl":
        return read_texts_from_jsonl(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix} (expected .arrow or .jsonl)")


def analyze_charset(texts, label=""):
    """Extract unique characters and print a summary."""
    chars = set()
    for text in texts:
        chars.update(text)

    print(f"\n=== Character analysis{' — ' + label if label else ''} ===")
    print(f"  Total samples: {len(texts)}")
    print(f"  Unique characters: {len(chars)}")
    print(f"  Character set: {sorted(chars)}")

    # Categorize
    latin_basic = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    digits = set("0123456789")
    accented = {c for c in chars if ord(c) > 127}
    punctuation = chars - latin_basic - digits - accented - {" "}

    print(f"  Categories:")
    print(f"    Basic latin: {len(chars & latin_basic)}")
    print(f"    Digits: {len(chars & digits)}")
    print(f"    Accented / Unicode: {len(accented)} → {sorted(accented)}")
    print(f"    Punctuation / other: {len(punctuation)} → {sorted(punctuation)}")

    return chars


def compare_charsets(chars1, chars2, label1="A", label2="B"):
    """Show differences between two character sets."""
    only_1 = chars1 - chars2
    only_2 = chars2 - chars1
    common = chars1 & chars2

    print(f"\n=== Comparison {label1} vs {label2} ===")
    print(f"  Common characters: {len(common)}")
    print(f"  Only in {label1}: {len(only_1)} → {sorted(only_1)}")
    print(f"  Only in {label2}: {len(only_2)} → {sorted(only_2)}")

    if only_1 or only_2:
        print(f"\n  Warning: character sets differ. If training a model with {label1} data")
        print(f"  and evaluating on {label2}, mismatched characters will cause errors.")
    else:
        print(f"\n  Character sets are IDENTICAL. Safe to mix or transfer between them.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Input .arrow or .jsonl file.")
    parser.add_argument("--compare", help="Optional second dataset to compare against.")
    args = parser.parse_args()

    print(f"Reading: {args.input}")
    texts_1 = read_texts(args.input)
    label_1 = Path(args.input).stem
    chars_1 = analyze_charset(texts_1, label_1)

    if args.compare:
        print(f"\nReading: {args.compare}")
        texts_2 = read_texts(args.compare)
        label_2 = Path(args.compare).stem
        chars_2 = analyze_charset(texts_2, label_2)
        compare_charsets(chars_1, chars_2, label_1, label_2)


if __name__ == "__main__":
    main()
