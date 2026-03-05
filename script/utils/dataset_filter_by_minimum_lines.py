import json
from pathlib import Path


def count_non_empty_lines(code_snippet: str) -> int:
    if not code_snippet:
        return 0
    lines = code_snippet.splitlines()
    return sum(1 for line in lines if line.strip())


def filter_by_min_lines(input_json: Path, output_json: Path, min_lines: int = 5) -> None:
    with input_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    filtered = [entry for entry in data if count_non_empty_lines(entry.get("code_snippet", "")) >= min_lines]

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(filtered)} entries to {output_json}")


if __name__ == "__main__":
    input_json = Path("MLCQCodeSmellSamples_Updated.json")
    output_json = Path("MLCQCodeSmellSamples_min5lines.json")
    filter_by_min_lines(input_json, output_json, min_lines=5)
