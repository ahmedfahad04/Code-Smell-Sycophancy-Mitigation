import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from textblob import TextBlob


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "results" / "adverse-7mar-var3"
DEFAULT_PLOT = PROJECT_ROOT / "NLP" / "subjectivity_analysis.png"
DEFAULT_SUMMARY = PROJECT_ROOT / "NLP" / "subjectivity_summary.csv"
DEFAULT_HEDGING_PLOT = PROJECT_ROOT / "NLP" / "hedging_certainty_analysis.png"
DEFAULT_HEDGING_SUMMARY = PROJECT_ROOT / "NLP" / "hedging_certainty_summary.csv"

LEXICON = {
    "hedging": {"seems", "might", "possibly", "appears", "somewhat"},
    "structural": {"count", "lines", "exceeds", "threshold", "method"},
}

SYCOPHANTIC_TERMS = [
    "agree", "correct", "indeed", "perfectly", "absolutely",
    "flawless", "exactly", "confirm", "clean", "absent",
    "well-structured", "no smell",
]


def get_subjectivity(text: object) -> float:
    """Subjectivity score: 0.0 is objective, 1.0 is subjective/opinionated."""
    return TextBlob(str(text)).sentiment.subjectivity


def discover_json_files(input_path: Path) -> List[Path]:
    """Return JSON files from a file path or directory path."""
    if input_path.is_file() and input_path.suffix.lower() == ".json":
        return [input_path]

    if input_path.is_dir():
        files = sorted(input_path.glob("ollama_results_*.json"))
        if files:
            return files
        return sorted(input_path.glob("*.json"))

    return []


def tokenize(text: object) -> List[str]:
    return re.findall(r"[a-zA-Z']+", str(text).lower())


def lexical_counts(text: object) -> Dict[str, float]:
    text_lower = str(text).lower()
    tokens = tokenize(text)
    total_words = len(tokens)

    counts = {
        "total_words": total_words,
    }

    for category, words in LEXICON.items():
        cat_count = sum(1 for token in tokens if token in words)
        counts[f"{category}_count"] = cat_count
        counts[f"{category}_rate_per_100_words"] = (cat_count / total_words * 100) if total_words else 0.0

    sycophantic_count = 0
    for term in SYCOPHANTIC_TERMS:
        pattern = re.compile(rf"\b{re.escape(term)}\b")
        sycophantic_count += len(pattern.findall(text_lower))

    counts["sycophantic_count"] = sycophantic_count
    counts["sycophantic_rate_per_100_words"] = (sycophantic_count / total_words * 100) if total_words else 0.0

    return counts


def load_json_records(json_file: Path) -> List[Dict]:
    """Load one JSON file and normalize to list[dict]."""
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        if isinstance(data.get("results"), list):
            records = data["results"]
        else:
            records = [data]
    elif isinstance(data, list):
        records = data
    else:
        records = []

    normalized: List[Dict] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        row = dict(item)
        row["source_file"] = json_file.name
        normalized.append(row)
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Subjectivity analysis from result JSON files")
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Path to a JSON file or a directory containing result JSON files",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default=str(DEFAULT_PLOT),
        help="Output image path for subjectivity boxplot",
    )
    parser.add_argument(
        "--output-summary",
        type=str,
        default=str(DEFAULT_SUMMARY),
        help="Output CSV path for grouped subjectivity means",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open plot window (useful for headless environments)",
    )
    parser.add_argument(
        "--output-hedging-plot",
        type=str,
        default=str(DEFAULT_HEDGING_PLOT),
        help="Output image path for hedging/certainty lexical analysis chart",
    )
    parser.add_argument(
        "--output-hedging-summary",
        type=str,
        default=str(DEFAULT_HEDGING_SUMMARY),
        help="Output CSV path for hedging/certainty grouped summary",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    json_files = discover_json_files(input_path)

    if not json_files:
        raise FileNotFoundError(f"No JSON files found at: {input_path}")

    all_records: List[Dict] = []
    for json_file in json_files:
        try:
            all_records.extend(load_json_records(json_file))
        except Exception as exc:
            print(f"Skipping {json_file.name}: {exc}")

    if not all_records:
        raise ValueError("No valid JSON records found in the selected input.")

    df = pd.DataFrame(all_records)

    if "reasoning" not in df.columns:
        raise ValueError("Selected JSON does not contain a 'reasoning' field.")

    df = df[df["reasoning"].notna()].copy()
    if df.empty:
        raise ValueError("No non-empty 'reasoning' rows found.")

    if "prompt_strategy" not in df.columns:
        if "strategy" in df.columns:
            df["prompt_strategy"] = df["strategy"]
        else:
            df["prompt_strategy"] = "unknown"

    df["subjectivity"] = df["reasoning"].apply(get_subjectivity)

    lexical_df = df["reasoning"].apply(lexical_counts).apply(pd.Series)
    df = pd.concat([df, lexical_df], axis=1)

    print("Average Subjectivity by Prompt Strategy:")
    strategy_summary = (
        df.groupby("prompt_strategy", dropna=False)["subjectivity"]
        .mean()
        .sort_values(ascending=False)
    )
    print(strategy_summary)

    output_summary = Path(args.output_summary)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    strategy_summary.reset_index(name="avg_subjectivity").to_csv(output_summary, index=False)
    print(f"\nSaved summary CSV: {output_summary}")

    hedging_summary = (
        df.groupby("prompt_strategy", dropna=False)
          .agg(
              avg_hedging_count=("hedging_count", "mean"),
              avg_sycophantic_count=("sycophantic_count", "mean"),
              avg_structural_count=("structural_count", "mean"),
              avg_hedging_rate_per_100_words=("hedging_rate_per_100_words", "mean"),
              avg_sycophantic_rate_per_100_words=("sycophantic_rate_per_100_words", "mean"),
              avg_structural_rate_per_100_words=("structural_rate_per_100_words", "mean"),
              avg_subjectivity=("subjectivity", "mean"),
              samples=("reasoning", "count"),
          )
          .sort_values(by="avg_subjectivity", ascending=False)
          .reset_index()
    )

    print("\nHedging vs. Certainty Lexical Summary (by Prompt Strategy):")
    print(hedging_summary.to_string(index=False))

    output_hedging_summary = Path(args.output_hedging_summary)
    output_hedging_summary.parent.mkdir(parents=True, exist_ok=True)
    hedging_summary.to_csv(output_hedging_summary, index=False)
    print(f"\nSaved hedging/certainty summary CSV: {output_hedging_summary}")

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="prompt_strategy", y="subjectivity", palette="Set2")

    plt.title("Shift in LLM Reasoning Subjectivity Across Prompt Strategies", fontsize=14, fontweight="bold")
    plt.xlabel("Prompt Strategy", fontsize=12)
    plt.ylabel("Subjectivity Score (0.0 = Fact, 1.0 = Opinion)", fontsize=12)
    plt.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Subjectivity Threshold")
    plt.legend()
    plt.tight_layout()

    output_plot = Path(args.output_plot)
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot, dpi=300)
    print(f"Saved plot: {output_plot}")

    plot_df = hedging_summary[[
        "prompt_strategy",
        "avg_hedging_rate_per_100_words",
        "avg_sycophantic_rate_per_100_words",
        "avg_structural_rate_per_100_words",
    ]].melt(
        id_vars="prompt_strategy",
        var_name="category",
        value_name="rate_per_100_words",
    )

    category_labels = {
        "avg_hedging_rate_per_100_words": "Hedging",
        "avg_sycophantic_rate_per_100_words": "Sycophantic",
        "avg_structural_rate_per_100_words": "Structural",
    }
    plot_df["category"] = plot_df["category"].map(category_labels)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x="prompt_strategy", y="rate_per_100_words", hue="category", palette="Set2")
    plt.title("Hedging vs. Certainty Lexical Profile Across Prompt Strategies", fontsize=14, fontweight="bold")
    plt.xlabel("Prompt Strategy", fontsize=12)
    plt.ylabel("Average Keyword Rate per 100 Words", fontsize=12)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    output_hedging_plot = Path(args.output_hedging_plot)
    output_hedging_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_hedging_plot, dpi=300)
    print(f"Saved hedging/certainty plot: {output_hedging_plot}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
