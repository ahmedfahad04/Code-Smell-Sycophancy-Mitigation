# Mitigating LLM Sycophancy in Code Smell Detection using Evidence-Guided Reasoning Prompts

*Abstract*
```
Large Language Models (LLMs) are increasingly used for code smell detection tasks due to their ability to interpret program semantics. However, their reliability in this context remains poorly explored, particularly under varying prompt conditions where model predictions may be influenced by external cues rather than code characteristics. One such limitation is sycophancy bias, where models tend to align their outputs with user-provided assumptions instead of performing objective analysis. In this paper, we present the first systematic empirical study of sycophancy bias in LLM-based code smell detection. Using the MLCQ dataset, we evaluate how different prompt framings like confirmation bias, contradictory hints, and false premises, affect model predictions. Our results show that LLMs are highly sensitive to prompt variations, with Decision Flip Rates reaching up to 72% and False Alignment Rates exceeding 90%, indicating substantial instability and agreement with misleading prompts. To address this issue, we propose Evidence-Guided Debiasing Prompting (EGDP), a structured prompting strategy that enforces evidence-first reasoning. EGDP reduces decision instability and improves robustness, lowering Decision Flip Rates to as low as 12% and False Alignment Rates to as low as 21%, while increasing reliance on structurally grounded evidence. Our findings demonstrate that sycophancy bias poses a critical threat to the reliability of LLM-based code smell detection, and that evidence-guided reasoning provides an effective and generalizable mitigation approach.
```

## Setup

### Requirements
- Python 3.10+
- Ollama with model inference
- 20GB disk space for models

### Installation

```bash
pip install -r requirements.txt
```

### Model Setup

```bash
# Download models
ollama pull llama3.1:8b
ollama pull qwen2.5-coder:7b

# Start server
ollama serve
```

For remote servers, set in scripts:
```python
OLLAMA_API_URL = "http://<your-server>:11434"
```

---

## Quick Start

Run pre-configured experiment phases:

```bash
cd script/experiments
bash run_phases.sh validate        # 5-10 min  
bash run_phases.sh baseline        # 45 min    
bash run_phases.sh sycophancy      # 1-2 hrs   
bash run_phases.sh crossmodel      # 1-2 hrs   
bash run_phases.sh comprehensive   # 2-3 hrs   
bash run_phases.sh all             # 5-8 hrs   
```

---

## Usage

### Configuration

Edit `script/config-phases.sh`:

```bash
DATASET_PATH="$PROJECT_ROOT/dataset/mlcq_filtered.json"
RESULTS_DIR="$PROJECT_ROOT/results"
TEMPERATURE_GLOBAL="0.3"
TOP_P_GLOBAL="0.9"
```

### Phases

- **Phase 1:** Validation (1 model, 1 smell, 1 strategy, 10 samples)
- **Phase 2:** Baseline (all smells, Casual strategy)
- **Phase 3:** Sycophancy (8 bias strategies, Feature Envy)
- **Phase 4:** Cross-model (3 models, all strategies)
- **Phase 5:** Comprehensive (all smells, models, strategies)

### Running Experiments

#### Via Phase Runner

```bash
cd script/experiments
bash run_phases.sh baseline
```

#### Direct Script

```bash
python script/ollama_code_smell_detection.py \
  --smell "feature envy" \
  --models "llama3.1:8b,qwen2.5-coder:7b" \
  --strategies "Casual,Positive,Negative,Contradictory-Hint,False-Premise" \
  --limit 175 \
  --temperature 0.3 \
  --top-p 0.9 \
  --output-dir results/
```

**Options:**
- `--smell`: blob, data-class, feature-envy, long-method, or all
- `--models`: comma-separated model names
- `--strategies`: Casual, Positive, Negative, Authoritative, Social-Proof, Contradictory-Hint, False-Premise, Confirmation-Bias, EGDP
- `--limit`: number of samples

### Metrics

#### Classification Metrics

```bash
python script/calculate_metrics.py \
  --results-file results/ollama_results_blob_llama3_1_8b_Casual.json \
  --ground-truth dataset/mlcq_filtered.json \
  --output results/metrics/blob_metrics.csv
```

#### Behavioral Metrics (DFR, FAR)

```bash
python script/calculate_flip_alignment_metrics.py \
  --baseline results/ollama_results_blob_llama3_1_8b_Casual.json \
  --variant results/ollama_results_blob_llama3_1_8b_False-Premise.json \
  --output results/metrics/dfr_far.csv
```

#### Lexical Analysis

```bash
python script/analyze_lexical_composition.py \
  --input-dir results/ \
  --output results/metrics/lexical_analysis.csv
```

### Interactive Analysis

```bash
streamlit run app.py
```

Web interface at `http://localhost:8501` for comparing result files.

---

## Dataset

**Primary dataset:** `dataset/mlcq_filtered.json` (1,495 samples)

**Schema:**
```json
{
  "id": 1,
  "code_snippet": "...",
  "smell": "blob|data_class|feature_envy|long_method",
  "severity": "none|minor|major|critical",
  "repo_url": "...",
  "commit_hash": "...",
  "file_path": "...",
  "start_line": 42,
  "end_line": 156
}
```

**Distribution:**
| Smell | Samples | Smelly |
|-------|---------|--------|
| Blob | 474 | 88 |
| Data Class | 477 | 95 |
| Feature Envy | 271 | 42 |
| Long Method | 273 | 58 |
| **Total** | **1,495** | **283** |

---

## Results

Results saved to `results/` with naming pattern:
```
ollama_results_<smell>_<model>_<strategy>.json
```

Each result contains:
```json
{
  "id": 1,
  "smell": "blob",
  "severity": "major",
  "reasoning": "...",
  "model": "llama3.1:8b",
  "strategy": "Casual"
}
```

---

## Prompt Strategies

### Neutral
- **Casual:** Standard instruction

### Biased
- **Positive:** "Code is well-written, no smells"
- **Negative:** "Code is problematic, full of smells"  
- **Authoritative:** "As an expert, this code is..."
- **Social-Proof:** "Most developers agree..."
- **Contradictory-Hint:** "Follows SOLID, must be clean"
- **False-Premise:** "Passed code review, no issues"
- **Confirmation-Bias:** "Confirm there are no smells"
- **EGDP:** User comment variation

See `docs/PROMPT_TEMPLATES.md` for full details.

---

## File Structure

```
.
├── README.md
├── requirements.txt
├── dataset/
│   └── mlcq_filtered.json             # Main dataset
├── results/                            # Experiment outputs
├── script/
│   ├── config-phases.sh                # Phase configuration
│   ├── ollama_code_smell_detection.py  # Main runner
│   ├── calculate_metrics.py            # Classification metrics
│   ├── calculate_flip_alignment_metrics.py  # DFR/FAR
│   ├── analyze_lexical_composition.py  # Language analysis
│   ├── evaluate_smell_results.py       # Evaluation utility
│   ├── dataset_selector.py             # Dataset tools
│   ├── experiments/
│   │   └── run_phases.sh               # Phase orchestrator
│   └── utils/
├── app.py                              # Streamlit visualization
├── docs/
│   ├── QUICK_START.md
│   └── PROMPT_TEMPLATES.md
└── test.ipynb                          # Analysis notebook
```

---


