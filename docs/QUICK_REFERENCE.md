# Quick Reference Guide - Prompt Sycophancy Testing

## Most Common Commands

### 1. Quick Test (Default Settings)
```bash
cd /home/sakib/Documents/MLCQ/script/src
./test_sycophancy.sh
```
Tests: feature envy | Model: llama3.1:8b | Strategy: Casual | Samples: 5

### 2. Test Different Code Smell
```bash
./test_sycophancy.sh -c "data class" -l 10
```

### 3. Compare Sycophancy Strategies
```bash
./test_sycophancy.sh -c "feature envy" -s "Casual,Positive,Negative" -l 20
```

### 4. Test Multiple Models
```bash
./test_sycophancy.sh -c "data class" -m "llama3.1:8b,qwen2.5-coder:3b" -l 15
```

### 5. Comprehensive Test
```bash
./test_sycophancy.sh \
  -c "god class" \
  -m "llama3.1:8b,qwen2.5-coder:3b" \
  -s "Casual,Positive,Negative,Authority" \
  -l 50 \
  -o "comprehensive_test.json"
```

### 6. List Available Models
```bash
./test_sycophancy.sh --list-models
```

### 7. Re-evaluate Existing Results
```bash
./test_sycophancy.sh -c "feature envy" --skip-test
```

## Available Prompt Strategies

| Strategy | Purpose |
|----------|---------|
| Casual | Neutral baseline (use for comparison) |
| Positive | Tests positive sycophancy bias |
| Negative | Tests negative bias |
| Authority | Tests authority bias |
| Social Proof | Tests social proof bias |
| Contradictory Hint | Tests contradictory statements |

## Understanding Results

### JSON Output
```
Location: ~/MLCQ/results/ollama_results_*.json
Contains: Individual predictions for each sample
```

### CSV Output  
```
Location: ~/MLCQ/results/evaluation_*.csv
Contains: Summary metrics by strategy
Key Metrics: Accuracy, Precision, Recall, F1
```

### Interpreting CSV Results

- **Higher Accuracy with Casual** = Model is reliable with neutral prompts
- **Large drop with Positive** = Model susceptible to positive sycophancy
- **Large drop with Negative** = Model overly pessimistic
- **Strategy Consistency** = Model robust to prompt manipulation

## File Locations

```
~/MLCQ/script/src/
├── test_sycophancy.sh              (Main script)
├── ollama_code_smell_detection.py  (Detection engine)
├── evaluate_smell_results.py       (Evaluation engine)
└── README_SYCOPHANCY_TESTING.md    (Full documentation)

~/MLCQ/results/
├── ollama_results_*.json           (Detection results)
└── evaluation_*.csv                (Evaluation metrics)
```

## Advanced Options

| Option | Usage |
|--------|-------|
| `-l <N>` | Set number of samples per model |
| `-t <sec>` | Set request timeout |
| `-r <N>` | Set number of retries |
| `-o <file>` | Set output filename |
| `--skip-test` | Only evaluate (no detection) |
| `--skip-eval` | Only detect (no evaluation) |
| `--dataset <path>` | Use custom dataset |

## Example Analysis Workflow

```bash
# 1. Run comprehensive test
./test_sycophancy.sh \
  -c "feature envy" \
  -m "llama3.1:8b" \
  -s "Casual,Positive,Negative" \
  -l 100

# 2. Check results
ls -lah ../../results/

# 3. View CSV summary
cat ../../results/evaluation_feature_envy_*.csv

# 4. Compare models (new test)
./test_sycophancy.sh \
  -c "feature envy" \
  -m "qwen2.5-coder:3b" \
  -s "Casual,Positive,Negative" \
  -l 100

# 5. Compare CSVs
diff ../../results/evaluation_feature_envy_*.csv
```

## Troubleshooting

**"No models found"**
```bash
# Start Ollama if not running
ollama serve
# Verify connection
curl http://localhost:11434/api/models
```

**"No results files found"**
```bash
# Check results directory
ls -la ../../results/

# Verify detection ran without --skip-test
./test_sycophancy.sh --skip-eval  # (not --skip-test)
```

**"Dataset not found"**
```bash
# Verify dataset exists
ls -la ../dataset/MLCQCodeSmellSamples_min5lines.json
```

## Tips

- Start with small limits (`-l 5`) for testing
- Use `--skip-test` to re-evaluate with different parameters
- Save with meaningful names: `-o "llama_vs_qwen_sycophancy.json"`
- Check help for all options: `./test_sycophancy.sh --help`
