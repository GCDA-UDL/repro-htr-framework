#!/usr/bin/env python3
"""
convert_hf_to_kraken.py — Convert a HuggingFace-style dataset (with
metadata.jsonl) to Kraken format (image + .gt.txt pairs).

The framework's run_pipeline.py produces datasets in HuggingFace layout:
    output_dir/
        train/
            00000000.png
            00000001.png
            ...
            metadata.jsonl
        validation/
            ...
        test/
            ...
        dataset_info.json

Kraken expects each image alongside a .gt.txt file with its ground truth:
    output_dir/
        train/
            00000000.png
            00000000.gt.txt
            ...

Usage:
    python convert_hf_to_kraken.py --input INPUT_DIR --output OUTPUT_DIR
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def convert_split(input_dir, output_dir, split_name):
    """Convert one split (train/validation/test)."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    metadata_path = input_dir / "metadata.jsonl"

    if not metadata_path.exists():
        print(f"  [WARNING] {split_name}: no metadata.jsonl found at {metadata_path}", file=sys.stderr)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    # Count entries first for progress bar
    with open(metadata_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    count = 0
    iterator = tqdm(entries, desc=f"  {split_name}", unit="img") if HAS_TQDM else entries

    for entry in iterator:
        img_name = entry["file_name"]
        text = entry["text"]
        src_img = input_dir / img_name
        dst_img = output_dir / img_name

        if not src_img.exists():
            continue

        shutil.copy2(src_img, dst_img)
        gt_name = img_name.replace(".png", ".gt.txt")
        gt_path = output_dir / gt_name
        with open(gt_path, "w", encoding="utf-8") as gt_file:
            gt_file.write(text)
        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Input directory (HuggingFace-style with train/, validation/, test/).")
    parser.add_argument("--output", required=True, help="Output directory (Kraken format with .gt.txt files).")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"],
                        help="Splits to convert (default: train validation test).")
    args = parser.parse_args()

    input_base = Path(args.input).resolve()
    output_base = Path(args.output).resolve()

    if not input_base.exists():
        print(f"Error: input directory does not exist: {input_base}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("HuggingFace -> Kraken format converter")
    print("=" * 60)
    print(f"Input:  {input_base}")
    print(f"Output: {output_base}")
    print(f"Splits: {', '.join(args.splits)}")
    print()

    total = 0
    for split in args.splits:
        input_dir = input_base / split
        output_dir = output_base / split
        if not input_dir.exists():
            print(f"  [SKIP] {split}: input directory not found ({input_dir})")
            continue
        count = convert_split(input_dir, output_dir, split)
        print(f"  {split}: {count} images converted")
        total += count

    print()
    print("=" * 60)
    print(f"Conversion complete. Total: {total} images")
    print(f"Output directory: {output_base}")
    print("=" * 60)


if __name__ == "__main__":
    main()
