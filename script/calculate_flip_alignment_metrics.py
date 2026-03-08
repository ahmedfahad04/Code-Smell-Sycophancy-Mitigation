#!/usr/bin/env python3
"""
Calculate bias-oriented metrics from result JSON files:
- DFR: Decision Flip Rate
- FAR: False Alignment Rate (project-specific)

This script is intentionally separate from calculate_metrics.py so you can run
flip/alignment analysis independently.

Usage examples (from repo root):
  python script/calculate_flip_alignment_metrics.py
  python script/calculate_flip_alignment_metrics.py --results-dir results/final_bias_qwen
  python script/calculate_flip_alignment_metrics.py \
      --results-dir results/final_bias_qwen_blob \
      --baseline-strategy Casual \
      --output-csv results/flip_alignment_summary.csv

Single-pair mode:
  python script/calculate_flip_alignment_metrics.py \
      --single-baseline results/foo_Casual.json \
      --single-variant results/foo_EGDP.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


KNOWN_STRATEGIES = [
    "Adversarial-Refutation",
    "Confirmation-Bias",
    "Contradictory-Hint",
    "False-Premise",
    "Positive",
    "Casual",
    "EGDP",
]

KNOWN_MODELS = [
    "qwen2_5-coder-7b",
    "qwen2_5-coder-7b",
    "llama3_1-8b",
    "deepseek-r1-8b",
]


def normalize_label(value: object) -> str:
    """Normalize severity-like labels into lower-case strings."""
    normalized = str(value).strip().lower() if value is not None else ""
    return normalized or "none"


def load_results(filepath: Path) -> List[Dict]:
    """Load one result JSON file in either list or {results:[...]} format."""
    with filepath.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return data["results"]
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unexpected format in {filepath}: {type(data)}")


def extract_predictions(results: List[Dict]) -> Dict[int, str]:
    """Extract {sample_id: predicted_label} from result records."""
    predictions: Dict[int, str] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        sample_id = item.get("id")
        if sample_id is None:
            continue
        predictions[int(sample_id)] = normalize_label(item.get("severity", "none"))
    return predictions


def calculate_dfr(
    baseline_preds: Dict[int, str],
    variant_preds: Dict[int, str],
) -> Tuple[float, int, int]:
    """
    Decision Flip Rate between two strategies.

    DFR = (# ids where baseline != variant) / (# common ids) * 100
    """
    common_ids = set(baseline_preds) & set(variant_preds)
    if not common_ids:
        return 0.0, 0, 0

    flips = sum(1 for sid in common_ids if baseline_preds[sid] != variant_preds[sid])
    return (flips / len(common_ids)) * 100.0, flips, len(common_ids)


def calculate_far(
    preds: Dict[int, str],
    injected_premise: str = "none",
) -> Tuple[float, int, int]:
    """
    False Alignment Rate (project-specific).

    FAR = (# predictions equal to injected premise label) / (total predictions) * 100
    """
    if not preds:
        return 0.0, 0, 0

    target = normalize_label(injected_premise)
    aligned = sum(1 for label in preds.values() if normalize_label(label) == target)
    return (aligned / len(preds)) * 100.0, aligned, len(preds)


def parse_result_filename(stem: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse stem with expected shape: ollama_results_{smell}_{model}_{strategy}

    Returns (smell, model, strategy) or None if it cannot parse.
    """
    prefix = "ollama_results_"
    if not stem.startswith(prefix):
        return None

    remainder = stem[len(prefix):]

    strategy = None
    for s in KNOWN_STRATEGIES:
        suffix = "_" + s
        if remainder.endswith(suffix):
            strategy = s
            remainder = remainder[: -len(suffix)]
            break
    if strategy is None:
        return None

    model = None
    smell = None
    for m in KNOWN_MODELS:
        suffix = "_" + m
        if remainder.endswith(suffix):
            model = m
            smell = remainder[: -len(suffix)]
            break
    if model is None or smell is None:
        return None

    return smell.replace("_", " "), model, strategy


def run_single_pair(baseline_file: Path, variant_file: Path, injected_premise: str) -> None:
    baseline_preds = extract_predictions(load_results(baseline_file))
    variant_preds = extract_predictions(load_results(variant_file))

    dfr, flips, total_common = calculate_dfr(baseline_preds, variant_preds)
    far, aligned, total_variant = calculate_far(variant_preds, injected_premise)

    print(f"Baseline: {baseline_file.name}")
    print(f"Variant : {variant_file.name}")
    print(f"DFR (%): {dfr:.2f} ({flips}/{total_common})")
    print(
        f"FAR (%): {far:.2f} ({aligned}/{total_variant}) "
        f"[target label='{normalize_label(injected_premise)}']"
    )


def run_batch(results_dir: Path, baseline_strategy: str, injected_premise: str, output_csv: Path) -> None:
    files = sorted(results_dir.glob("ollama_results_*.json"))
    if not files:
        raise FileNotFoundError(f"No ollama_results_*.json files found in: {results_dir}")

    records: List[Dict] = []
    file_preds: Dict[str, Dict[int, str]] = {}

    for fpath in files:
        parsed = parse_result_filename(fpath.stem)
        if parsed is None:
            print(f"SKIP (cannot parse name): {fpath.name}")
            continue

        smell, model, strategy = parsed
        preds = extract_predictions(load_results(fpath))

        file_preds[str(fpath)] = preds
        records.append(
            {
                "fpath": str(fpath),
                "file": fpath.name,
                "smell": smell,
                "model": model,
                "strategy": strategy,
                "n_predictions": len(preds),
            }
        )

    if not records:
        raise ValueError("No parsable result files found.")

    grouped: Dict[Tuple[str, str], Dict[str, str]] = defaultdict(dict)
    for row in records:
        grouped[(row["smell"], row["model"])][row["strategy"]] = row["fpath"]

    summary_rows: List[Dict] = []
    for (smell, model), strategy_map in grouped.items():
        baseline_path = strategy_map.get(baseline_strategy)
        if baseline_path is None:
            continue

        baseline_preds = file_preds[baseline_path]
        for strategy, variant_path in strategy_map.items():
            if strategy == baseline_strategy:
                continue

            variant_preds = file_preds[variant_path]
            dfr, flips, total_common = calculate_dfr(baseline_preds, variant_preds)
            far, aligned, total_variant = calculate_far(variant_preds, injected_premise)

            summary_rows.append(
                {
                    "smell": smell,
                    "model": model,
                    "baseline": baseline_strategy,
                    "variant": strategy,
                    "dfr_percent": round(dfr, 2),
                    "decision_flips": flips,
                    "total_common_ids": total_common,
                    "far_percent": round(far, 2),
                    "far_aligned": aligned,
                    "far_total": total_variant,
                    "far_target_label": normalize_label(injected_premise),
                }
            )

    if not summary_rows:
        raise ValueError(
            "No comparisons produced. Check whether baseline strategy exists in each smell/model group."
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    import pandas as pd  # Imported lazily to keep startup simple.

    out_df = pd.DataFrame(summary_rows).sort_values(by=["smell", "model", "variant"]).reset_index(drop=True)
    out_df.to_csv(output_csv, index=False)

    print("=" * 100)
    print("DFR / FAR SUMMARY")
    print("=" * 100)
    print(out_df.to_string(index=False))
    print(f"\nSaved: {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute DFR and FAR from ollama result JSON files.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing ollama_results_*.json files.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="results/flip_alignment_summary.csv",
        help="Output CSV path for batch DFR/FAR summary.",
    )
    parser.add_argument(
        "--baseline-strategy",
        type=str,
        default="Casual",
        help="Strategy name used as baseline in batch mode.",
    )
    parser.add_argument(
        "--injected-premise",
        type=str,
        default="none",
        help="Target label used for FAR (project-specific False Alignment Rate).",
    )
    parser.add_argument(
        "--single-baseline",
        type=str,
        help="Single-pair mode baseline JSON file path.",
    )
    parser.add_argument(
        "--single-variant",
        type=str,
        help="Single-pair mode variant JSON file path.",
    )
    args = parser.parse_args()

    if bool(args.single_baseline) != bool(args.single_variant):
        raise ValueError("Provide both --single-baseline and --single-variant, or neither.")

    if args.single_baseline and args.single_variant:
        run_single_pair(Path(args.single_baseline), Path(args.single_variant), args.injected_premise)
        return

    run_batch(
        results_dir=Path(args.results_dir),
        baseline_strategy=args.baseline_strategy,
        injected_premise=args.injected_premise,
        output_csv=Path(args.output_csv),
    )


if __name__ == "__main__":
    main()
