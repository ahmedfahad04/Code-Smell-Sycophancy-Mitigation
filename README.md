# Sycophancy Detection, Analysis and Mitigation Research

**Repository:** KamruzzamanAsif/MLCQ

A research codebase and toolkit for analyzing code-smell detection outputs from large language models. This repository hosts datasets, evaluation scripts, result artifacts, and a Streamlit app to compare and inspect model reasoning and severity judgments side-by-side.

---

**Contents**

- **dataset/** — Reference datasets and filtered samples (primary: `mlcq_filtered.json`).
- **results/** — JSON outputs from different models and prompt strategies (many subfolders per-experiment).
- **script/** — Utilities and evaluation scripts (metrics, experiment runners).
- **docs/** — Project documentation and experiment notes.
- **app.py** — Streamlit web app for interactive, side-by-side analysis of results.
- **requirements.txt** — Python dependencies used by the project.
- **APP_README.md** — Quick start guide for the Streamlit app (shorter companion doc).

---

**Project Overview**

This repo is focused on evaluating how different LLMs (and prompt strategies) identify code smells in code snippets and the reasoning they provide. Typical tasks supported by this repo:

- Compare model outputs (severity, reasoning) across models and prompt strategies.
- Aggregate evaluation metrics (precision, recall, F1, confusion matrices) per smell.
- Inspect per-ID code snippets and detailed reasoning side-by-side for analysis and error diagnosis.

---

**Quick Start**

Prerequisites:

- Python 3.9+ (tested with 3.10/3.12)
- Conda or virtualenv recommended

Install dependencies:

```bash
cd /home/fahad/Documents/RESEARCH_CODEBASE/MLCQ
pip install -r requirements.txt
```

Run the Streamlit app locally:

```bash
streamlit run app.py
```

Open the URL printed by Streamlit (usually `http://localhost:8501`).

---

**Streamlit App (`app.py`)**

Purpose: provide an interactive side-by-side comparison of two result files from `results/` and the ground-truth dataset in `dataset/mlcq_filtered.json`.

Main features:

- Choose two result JSON files from `results/` (supports nested subfolders).
- Compare severity labels (`none`, `minor`, `major`, `critical`) and show where files disagree.
- Display model reasoning side-by-side for manual inspection.
- Show dataset `smell` and `severity` per ID and include the original `code_snippet` in a collapsed expander.
- Filters: show only differences, filter by ID, and export differences as CSV.
- Performance: lazy loading — nothing heavy loads until you click `Compare Files`.

How to use the app:

1. Select **File 1** and **File 2** from the sidebar (full path displayed).
2. Click **Compare Files** to load and compute differences.
3. Use filters to narrow results; expand code snippets to inspect source.
4. Download CSV of differing IDs using the download button.

Notes:

- The app caches `mlcq_filtered.json` and keeps comparison state in Streamlit session state.
- File listings are auto-discovered from `results/` and include nested directories.

---

**Dataset**

The canonical dataset for this repo is `dataset/mlcq_filtered.json` and contains ~1,495 samples. Each record includes:

- `id` — numeric identifier
- `repo_url`, `commit_hash` — provenance metadata
- `file_path` — source file path
- `start_line`, `end_line` — snippet location
- `code_snippet` — the code block used for analysis
- `smell` — code smell label (e.g., `feature_envy`, `long_method`, etc.)
- `severity` — ground-truth severity (e.g., `none`, `minor`, `major`, `critical`)

Keep this file in version control; large changes to the dataset should be recorded with notes in `docs/`.

---

**Scripts & Utilities (`script/`)**

Key scripts:

- `calculate_metrics.py` — aggregate evaluation metrics across results
- `evaluate_smell_results.py` — helper to compare outputs and compute per-smell metrics
- `dataset_selector.py` — utilities for selecting dataset subsets for experiments
- `hyperparameter_tuner.py` — experimental tools for model/prompt tuning

Run scripts from the repo root. Many scripts assume `results/` and `dataset/` exist in the workspace root.

---

**Results Organization**

- `results/` contains JSON outputs from experiments and model runs, often organized in subfolders by model or experiment type (e.g., `llama/`, `adverse-conf-bias/`). Files follow a naming convention like `ollama_results_<task>_<model>_<strategy>.json`.
- Each result JSON is an array of objects with keys such as `id`, `model`, `prompt_strategy`, `smell`, `severity`, and `reasoning`.

When adding new experiments, keep results under `results/<experiment-name>/` and include a short README or notes in `docs/` describing the experiment configuration.

---

**Developer Notes**

- The Streamlit app is optimized for interactivity and caches the dataset to avoid reloading on every interaction.
- To add another result file format, update the JSON loader in `app.py` with a compatible parser.
- For reproducibility, capture model versions and prompt templates; store these in a results metadata file or `docs/`.

Testing & linting

```bash
# Basic syntax check
python -m py_compile app.py

# Install dev tools (optional)
pip install black flake8
black .
flake8
```

---

**Contributing**

- Open issues for bugs or feature requests.
- For larger changes, submit a PR and include tests or a demo of updated behavior.

---

**Contact & References**

Maintainer: Repository owner (KamruzzamanAsif)

Project-specific docs: see `docs/` for experimental formats, prompt templates, and workflow guides.

---

**License**

Include your preferred license file if you plan to open-source this repository.

---

This README is designed for researchers and developers using the MLCQ codebase for evaluating LLM-based code-smell detection and analysis.
