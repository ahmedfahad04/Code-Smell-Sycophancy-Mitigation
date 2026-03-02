**Repository Purpose**
- **Summary:** This repository (MLCQ) evaluates how large language models detect code smells in short code samples. It runs model inference (via Ollama or similar runtimes) using multiple prompt strategies, collects JSON results, and produces CSV evaluations.
- **Core components:**
  - `script/src/test_sycophancy.sh` — orchestrates detection & evaluation runs.
  - `script/src/ollama_code_smell_detection.py` — performs model calls and writes JSON results to `results/`.
  - `script/src/evaluate_smell_results.py` — computes CSV evaluation metrics from JSON results.
  - `script/dataset/` — input datasets (JSON files of code samples).
  - `results/` — output JSON and evaluation CSVs.

**How to run (dev)**
- Create and activate a Python environment and install dependencies:
  - `python3 -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
- Run tests/evaluation end-to-end (defaults use `dataset/MLCQCodeSmellSamples_min5lines.json`):
  - `./script/src/test_sycophancy.sh`
- Common flags (see script help): `--smell`, `--models`, `--strategies`, `--limit`, `--temperature`, `--top-p`, `--frequency-penalty`, `--presence-penalty`, `--mixed-dataset`, `--skip-eval`, `--skip-test`.

**File & output conventions**
- Results are written under `results/`. JSON filenames follow the pattern `ollama_results_<sanitized-smell>_<strategy>_<model>_*.json` and evaluation CSVs are `evaluation_<sanitized-smell>.csv`.
- Datasets live in `script/dataset/` and should be JSON arrays of samples. Add new datasets there and reference them with `--dataset`.

**Coding & scripting guidelines (recommended rules)**
- Keep CLI flag handling strict and validate inputs (scripts should fail fast on invalid values).
- Environment variables or optional generation parameters (temperature, top-p, penalties) must only be passed to the model call when explicitly set. Do not hardcode unsafe defaults.
- Use relative paths inside scripts (root-relative) so CI and local runs behave the same.
- Shell scripts: use `set -e` and explicit checks for required files; print informative messages on error.
- Python code: follow existing style in the repo; keep external dependencies in `requirements.txt`.

**Prompt & experiment guidance**
- Prompt strategies are enumerated in `test_sycophancy.sh` (Casual, Positive, Negative, Authority, Social Proof, Contradictory Hint, False-Premise, Confirmation-Bias). When adding strategies, ensure prompt templates are stored clearly and tested.
- For reproducibility, record the model spec, strategy, and hyperparameters in the JSON output.

**Developer notes / common pitfalls**
- Validate variable names and defaults in shell scripts (typos in variable names lead to unexpected behavior).
- Ensure `--mixed-dataset` semantics are documented: it runs detection across all smells but tests for the specified smell.
- Keep `results/` out of version control; only commit small examples or expected-output fixtures.

**Suggested next customizations**
- Add a short `copilot-instructions.md` (this file) for workspace assistants to summarize rules.
- Add templates for prompt strategies in a `prompts/` folder to centralize prompt text and make them easier to audit.