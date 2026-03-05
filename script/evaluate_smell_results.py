import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set


def normalize_smell(smell: str) -> str:
    if not smell:
        return "none"
    s = str(smell).strip().lower().replace("_", " ")
    s = " ".join(s.split())
    return s


def load_dataset_smells(dataset_path: Path) -> Dict[int, str]:
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    smells = {}
    for row in data:
        try:
            sample_id = int(row.get("id"))
        except Exception:
            continue
        smells[sample_id] = normalize_smell(row.get("smell"))
    return smells


def load_results(results_path: Path) -> List[dict]:
    with results_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(y_true: List[str], y_pred: List[str]) -> Tuple[float, float, float, float]:
    if not y_true:
        return 0.0, 0.0, 0.0, 0.0

    labels: Set[str] = set(y_true) | set(y_pred)
    tp = {label: 0 for label in labels}
    fp = {label: 0 for label in labels}
    fn = {label: 0 for label in labels}

    correct = 0
    for t, p in zip(y_true, y_pred):
        if t == p:
            correct += 1
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    accuracy = correct / len(y_true)

    precisions = []
    recalls = []
    f1s = []
    for label in labels:
        p_den = tp[label] + fp[label]
        r_den = tp[label] + fn[label]
        precision = tp[label] / p_den if p_den > 0 else 0.0
        recall = tp[label] / r_den if r_den > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    # print the confusion matrix smell wise correctly predicted counts
    print("Confusion Matrix (smell wise correct predictions):")
    print("Label\t\t\tTP\tFP\tFN")
    for label in labels:
        print(f"{label}\t\t{tp[label]}\t{fp[label]}\t{fn[label]}")

    precision_macro = sum(precisions) / len(precisions)
    recall_macro = sum(recalls) / len(recalls)
    f1_macro = sum(f1s) / len(f1s)

    return accuracy, precision_macro, recall_macro, f1_macro


def compute_metrics_per_smell(y_true: List[str], y_pred: List[str]) -> Dict[str, Tuple[int, float, float, float, float]]:
    """
    Compute metrics for each smell separately.
    Returns: {smell: (count, accuracy, precision, recall, f1)}
    Where count is the number of true instances of that smell.
    accuracy is the proportion of correct predictions among the true instances of that smell (i.e., recall).
    """
    if not y_true:
        return {}
    
    smells = set(y_true)
    results = {}
    
    for smell in smells:
        # True positives: predicted correctly as this smell
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == smell and p == smell)
        # False positives: predicted as this smell but was something else
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != smell and p == smell)
        # False negatives: was this smell but predicted as something else
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == smell and p != smell)
        
        count = tp + fn  # number of true instances
        if count == 0:
            continue
        
        accuracy = tp / count  # this is actually recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        results[smell] = (count, accuracy, precision, recall, f1)
    
    return results


def extract_strategy_name(results_path: Path, results: List[dict]) -> str:
    # Prefer explicit prompt_strategy if all entries share it
    strategies = {r.get("prompt_strategy") for r in results if r.get("prompt_strategy")}
    if len(strategies) == 1:
        return next(iter(strategies))
    # Fall back to filename suffix
    stem = results_path.stem
    parts = stem.split("_")
    return parts[-1] if parts else stem


def evaluate_results(dataset_smells: Dict[int, str], results_path: Path) -> Tuple[str, str, int, float, float, float, float]:
    """
    Evaluate results from a detection file against ground truth.
    Returns: (model_name, strategy_name, num_evaluated_samples, accuracy, precision, recall, f1)
    
    Note: num_evaluated_samples = number of samples with valid ground truth that were evaluated.
    This may be less than total results if some samples are not in the dataset.
    """
    results = load_results(results_path)
    y_true: List[str] = []
    y_pred: List[str] = []
    skipped_count = 0
    invalid_id_count = 0
    model_name = "unknown"

    for row in results:
        # Extract model name from first row
        if model_name == "unknown" and row.get("model"):
            model_name = row.get("model")
        
        sample_id = row.get("id")
        if sample_id is None:
            invalid_id_count += 1
            continue
        try:
            sample_id = int(sample_id)
        except Exception:
            invalid_id_count += 1
            continue
        if sample_id not in dataset_smells:
            skipped_count += 1
            continue

        true_smell = dataset_smells[sample_id]
        pred_smell = normalize_smell(row.get("smell"))

        y_true.append(true_smell)
        y_pred.append(pred_smell)

    accuracy, precision, recall, f1 = compute_metrics(y_true, y_pred)
    strategy = extract_strategy_name(results_path, results)
    
    # Debug logging
    if skipped_count > 0 or invalid_id_count > 0:
        print(f"  Note: Skipped {skipped_count} samples not in ground truth, {invalid_id_count} with invalid IDs")
    
    return model_name, strategy, len(y_true), accuracy, precision, recall, f1, y_true, y_pred


def save_to_csv(rows: List[Tuple[str, str, int, float, float, float, float]], csv_path: Path) -> None:
    """Save evaluation results to CSV file."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["Model", "Strategy", "N", "Accuracy", "Precision(Macro)", "Recall(Macro)", "F1(Macro)"]
    with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for r in rows:
            model, strategy, n, acc, prec, rec, f1 = r
            writer.writerow([model, strategy, n, f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}"])

def sanitize_strategy_name(strategy: str) -> str:
    """Convert strategy name for safe filesystem usage (underscore to hyphen)."""
    return strategy.replace("_", "-")

def save_per_smell_csv(model: str, strategy: str, smell_metrics: Dict[str, Tuple[int, float, float, float, float]], base_dir: Path) -> None:
    """Save per-smell evaluation results organized by smell, with strategy-specific files."""
    sanitized_strategy = sanitize_strategy_name(strategy)
    for smell, (n, acc, prec, rec, f1) in smell_metrics.items():
        # Organize by smell first
        smell_dir = base_dir / smell
        smell_dir.mkdir(parents=True, exist_ok=True)
        csv_path = smell_dir / f"eval_{model}_{sanitized_strategy}.csv"
        
        headers = ["Model", "Strategy", "N", "Accuracy", "Precision", "Recall", "F1"]
        with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerow([model, strategy, n, f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}"])

def print_table(rows: List[Tuple[str, str, int, float, float, float, float]]) -> None:
    headers = ["Model", "Strategy", "N", "Accuracy", "Precision(Macro)", "Recall(Macro)", "F1(Macro)"]
    col_widths = [len(h) for h in headers]

    formatted_rows = []
    for r in rows:
        model, strategy, n, acc, prec, rec, f1 = r
        row = [
            str(model),
            str(strategy),
            str(n),
            f"{acc:.4f}",
            f"{prec:.4f}",
            f"{rec:.4f}",
            f"{f1:.4f}",
        ]
        formatted_rows.append(row)
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(items):
        return " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(items))

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for row in formatted_rows:
        print(fmt_row(row))
    
    print("\nNote: N = total evaluated samples (may be less than total results if some samples are missing in ground truth).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate smell prediction results")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "dataset" / "MLCQCodeSmellSamples_min5lines.json"),
        help="Path to dataset JSON with ground truth smell labels",
    )
    parser.add_argument(
        "--results-glob",
        type=str,
        default="ollama_results_*.json",
        help="Glob pattern for results files (relative to script dir)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        help="Directory to search for results files (defaults to script directory)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        help="Path to save evaluation results as CSV file",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    dataset_smells = load_dataset_smells(dataset_path)

    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = Path(__file__).resolve().parent
    
    results_files = sorted(results_dir.glob(args.results_glob))

    if not results_files:
        print(f"No results files found with pattern: {args.results_glob} in {results_dir}")
        return

    # Group results by strategy
    strategy_results = {}  # strategy -> list of (results_path, evaluation_result)
    
    for results_path in results_files:
        result = evaluate_results(dataset_smells, results_path)
        model, strategy, n, acc, prec, rec, f1, y_true, y_pred = result
        
        if strategy not in strategy_results:
            strategy_results[strategy] = []
        strategy_results[strategy].append((results_path, result))

    # Process each strategy separately
    for strategy, results_list in strategy_results.items():
        print(f"\n=== Evaluating Strategy: {strategy} ===")
        
        rows = []
        per_smell_data = {}  # model -> smell_metrics
        
        for results_path, result in results_list:
            model, _, n, acc, prec, rec, f1, y_true, y_pred = result
            rows.append((model, strategy, n, acc, prec, rec, f1))
            
            # Compute per-smell metrics
            smell_metrics = compute_metrics_per_smell(y_true, y_pred)
            if model not in per_smell_data:
                per_smell_data[model] = {}
            per_smell_data[model].update(smell_metrics)

        print_table(rows)
        
        # Save to CSV if requested
        if args.output_csv:
            csv_path = Path(args.output_csv)
            # Create strategy-specific filename with sanitized strategy name
            sanitized_strategy = sanitize_strategy_name(strategy)
            # Construct filename: evaluation_<smell>_<strategy>.csv
            base_name = csv_path.stem  # e.g., "evaluation_blob"
            strategy_csv_path = csv_path.parent / f"{base_name}_{sanitized_strategy}{csv_path.suffix}"
            save_to_csv(rows, strategy_csv_path)
            print(f"\nResults for {strategy} saved to CSV: {strategy_csv_path}")
            
            # Save per-smell results organized by smell
            base_per_smell_dir = csv_path.parent / "per_smell_results"
            for model, smell_metrics in per_smell_data.items():
                save_per_smell_csv(model, strategy, smell_metrics, base_per_smell_dir)
            print(f"Per-smell results for {strategy} saved to: {base_per_smell_dir}")


if __name__ == "__main__":
    main()
