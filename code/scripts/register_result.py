#!/usr/bin/env python3
"""
register_result.py — Register or update experiment results in all_results.csv

Usage:
    # Register a new (planned) experiment
    python register_result.py --exp-id E01 --exp-name fr_baseline \\
        --protocol baseline --language fr --status planned \\
        --model-base FoNDUE-GD_v2_fr --dataset-real rimes

    # Update an existing experiment with results
    python register_result.py --update --exp-id E01 --status completed \\
        --cer-cs 7.83 --cer-ci 6.90 --wer-cs 23.05 --wer-ci 22.15 \\
        --best-stage 45 --wall-time-hours 8.5

    # Show info about an experiment
    python register_result.py --show E01
"""

import argparse
import csv
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parents[2] / "results" / "all_results.csv"

COLUMNS = [
    "exp_id", "exp_name", "protocol", "language", "status",
    "model_base", "dataset_real", "real_train_size", "dataset_synth",
    "synth_samples_count", "fonts_count", "font_category", "quality_level",
    "phase1_lr", "phase2_lr", "decoder", "filter_qmark",
    "cer_cs", "cer_ci", "wer_cs", "wer_ci",
    "best_stage", "best_val_acc", "wall_time_hours", "gpu_used",
    "commit_sha", "timestamp", "notes", "eval_pipeline",
]

VALID_PROTOCOLS = ["baseline", "synth-only", "mix", "real-only", "curriculum", "curriculum-LR"]
VALID_LANGUAGES = ["fr", "en", "ca", "es"]
VALID_STATUSES = ["planned", "running", "completed", "failed", "abandoned"]
VALID_FONT_CATEGORIES = ["Handwritten", "School", "KidsCurated"]
VALID_QUALITY_LEVELS = ["Clean", "Degraded", "Severe"]


def get_git_commit_sha():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=CSV_PATH.parent.parent
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return ""


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def validate_exp_id(exp_id):
    if not re.match(r"^E\d{2}(v\d+)?(s\d+)?(_(fr|en|ca|es))?$", exp_id):
        raise ValueError(f"Invalid exp_id '{exp_id}'. Format must be E##, E##v#, or with lang suffix (e.g., E01, E05v2, E01_fr, E06_en).")


def validate_choice(value, valid_values, field_name):
    if value and value not in valid_values:
        raise ValueError(f"Invalid {field_name} '{value}'. Must be one of: {', '.join(valid_values)}")


def validate_percentage(value, field_name):
    # error rates (CER/WER) may exceed 100 when insertions outnumber the reference units
    # (e.g. ca synth-only: WER 101.01), so only a lower bound is enforced
    if value is not None:
        try:
            v = float(value)
        except (ValueError, TypeError):
            raise ValueError(f"{field_name} must be a number, got '{value}'")
        if v < 0:
            raise ValueError(f"{field_name} must be >= 0, got {v}")


def validate_val_acc(value):
    if value is not None:
        try:
            v = float(value)
            if not (0 <= v <= 1):
                raise ValueError(f"best_val_acc must be between 0 and 1, got {v}")
        except (ValueError, TypeError):
            raise ValueError(f"best_val_acc must be a number, got '{value}'")


def validate_positive(value, field_name):
    if value is not None:
        try:
            v = float(value)
            if v <= 0:
                raise ValueError(f"{field_name} must be positive, got {v}")
        except (ValueError, TypeError):
            raise ValueError(f"{field_name} must be a number, got '{value}'")


def load_csv():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}")

    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if reader.fieldnames != COLUMNS:
        raise ValueError(
            f"CSV columns don't match expected schema.\n"
            f"Expected: {COLUMNS}\n"
            f"Got: {reader.fieldnames}"
        )
    return rows


def save_csv(rows):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def find_row(rows, exp_id):
    for i, row in enumerate(rows):
        if row["exp_id"] == exp_id:
            return i, row
    return None, None


def args_to_dict(args):
    d = {}
    mapping = {
        "exp_id": args.exp_id, "exp_name": args.exp_name,
        "protocol": args.protocol, "language": args.language, "status": args.status,
        "model_base": args.model_base, "dataset_real": args.dataset_real,
        "real_train_size": args.real_train_size, "dataset_synth": args.dataset_synth,
        "synth_samples_count": args.synth_samples_count, "fonts_count": args.fonts_count,
        "font_category": args.font_category, "quality_level": args.quality_level,
        "phase1_lr": args.phase1_lr, "phase2_lr": args.phase2_lr,
        "decoder": args.decoder, "filter_qmark": args.filter_qmark,
        "cer_cs": args.cer_cs, "cer_ci": args.cer_ci,
        "wer_cs": args.wer_cs, "wer_ci": args.wer_ci,
        "best_stage": args.best_stage, "best_val_acc": args.best_val_acc,
        "wall_time_hours": args.wall_time_hours, "gpu_used": args.gpu_used,
        "notes": args.notes,
    }
    for k, v in mapping.items():
        if v is not None:
            d[k] = str(v)
    return d


def show_experiment(exp_id):
    rows = load_csv()
    idx, row = find_row(rows, exp_id)
    if row is None:
        print(f"No experiment with exp_id '{exp_id}' found.", file=sys.stderr)
        sys.exit(1)
    print(f"=== Experiment {exp_id} ===")
    for col in COLUMNS:
        val = row.get(col, "")
        if val:
            print(f"  {col:25s}: {val}")


def register(args):
    validate_exp_id(args.exp_id)
    validate_choice(args.protocol, VALID_PROTOCOLS, "protocol")
    validate_choice(args.language, VALID_LANGUAGES, "language")
    validate_choice(args.status, VALID_STATUSES, "status")
    validate_choice(args.font_category, VALID_FONT_CATEGORIES, "font_category")
    validate_choice(args.quality_level, VALID_QUALITY_LEVELS, "quality_level")
    validate_percentage(args.cer_cs, "cer_cs")
    validate_percentage(args.cer_ci, "cer_ci")
    validate_percentage(args.wer_cs, "wer_cs")
    validate_percentage(args.wer_ci, "wer_ci")
    validate_val_acc(args.best_val_acc)
    validate_positive(args.wall_time_hours, "wall_time_hours")

    rows = load_csv()
    idx, existing = find_row(rows, args.exp_id)

    new_data = args_to_dict(args)
    new_data["commit_sha"] = get_git_commit_sha()
    new_data["timestamp"] = get_timestamp()

    if args.update:
        if existing is None:
            print(f"Cannot update: exp_id '{args.exp_id}' not found. Register it first without --update.", file=sys.stderr)
            sys.exit(1)
        for k, v in new_data.items():
            if v:
                existing[k] = v
        rows[idx] = existing
        save_csv(rows)
        print(f"Updated experiment {args.exp_id}")
    else:
        if existing is not None:
            print(f"exp_id '{args.exp_id}' already exists. Use --update to modify it.", file=sys.stderr)
            sys.exit(1)
        new_row = {col: new_data.get(col, "") for col in COLUMNS}
        rows.append(new_row)
        save_csv(rows)
        print(f"Registered new experiment {args.exp_id}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--update", action="store_true", help="Update an existing experiment.")
    parser.add_argument("--show", metavar="EXP_ID", help="Show info about an experiment and exit.")
    parser.add_argument("--exp-id", help="Experiment ID (format E##).")
    parser.add_argument("--exp-name", help="Short descriptive name.")
    parser.add_argument("--protocol", help=f"One of: {', '.join(VALID_PROTOCOLS)}")
    parser.add_argument("--language", help=f"One of: {', '.join(VALID_LANGUAGES)}")
    parser.add_argument("--status", help=f"One of: {', '.join(VALID_STATUSES)}")
    parser.add_argument("--model-base")
    parser.add_argument("--dataset-real")
    parser.add_argument("--real-train-size", type=int)
    parser.add_argument("--dataset-synth")
    parser.add_argument("--synth-samples-count", type=int)
    parser.add_argument("--fonts-count", type=int)
    parser.add_argument("--font-category", help=f"One of: {', '.join(VALID_FONT_CATEGORIES)}")
    parser.add_argument("--quality-level", help=f"One of: {', '.join(VALID_QUALITY_LEVELS)}")
    parser.add_argument("--phase1-lr", type=float)
    parser.add_argument("--phase2-lr", type=float)
    parser.add_argument("--decoder", default="greedy_ctc")
    parser.add_argument("--filter-qmark", default="no")
    parser.add_argument("--cer-cs", type=float)
    parser.add_argument("--cer-ci", type=float)
    parser.add_argument("--wer-cs", type=float)
    parser.add_argument("--wer-ci", type=float)
    parser.add_argument("--best-stage", type=int)
    parser.add_argument("--best-val-acc", type=float)
    parser.add_argument("--wall-time-hours", type=float)
    parser.add_argument("--gpu-used")
    parser.add_argument("--notes", help="Free text (max 150 characters recommended).")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.show:
        show_experiment(args.show)
        return

    if not args.exp_id:
        parser.error("--exp-id is required (unless using --show).")

    if args.notes and len(args.notes) > 150:
        print(f"Warning: notes is {len(args.notes)} chars (recommended max: 150).", file=sys.stderr)

    register(args)


if __name__ == "__main__":
    main()
