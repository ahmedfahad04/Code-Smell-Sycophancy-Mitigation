# Quick Start: Running Sycophancy Bias Experiments

Commands to run your sycophancy bias experiments. Uses 206 balanced samples (13.8% of 1,495 raw).

## Quick One-Liners

### Start experiments:
```bash
cd script/src/experiments
bash run_phases.sh validation       # 5-10 min  - Quick smoke test
bash run_phases.sh baseline         # 45 min    - All 4 code smells baseline
bash run_phases.sh sycophancy       # 1-2 hrs   - 8 bias strategies, 1 smell
bash run_phases.sh crossmodel       # 1-2 hrs   - 3 models, bias consistency
bash run_phases.sh comprehensive    # 2-3 hrs   - Feature Envy + Long Method
```

---

## Configuration

Edit to customize experiment parameters:
```
script/src/config-phases.sh
```

**Key settings:**
- `DATASET_PATH`: Points to 206-sample balanced subset
- `TEMPERATURE_GLOBAL`: 0.3 (optimized from Phase 1)
- `TOP_P_GLOBAL`: 0.9 (optimized from Phase 1)

**Phases defined:**
- **Phase 1**: Validation only (1 model, 1 smell, 10 samples)
- **Phase 2**: Baseline accuracy (all 4 smells, Casual strategy)
- **Phase 3**: Sycophancy bias testing (8 strategies, feature envy)
- **Phase 4**: Cross-model consistency (3 models, 8 strategies)
- **Phase 5**: Comprehensive suite (all smells/models/strategies)

## Available Code Smells

- `feature-envy` - Method overly interested in other class's data
- `long-method` - Methods with too many lines/complexity
- `data-class` - Classes with only data, no logic
- `blob` - God/Large classes with many responsibilities

## Available Models

- `llama3.1:8b` - Meta's Llama 3.1 (8B parameters)
- `qwen2.5-coder:7b` - Alibaba's Qwen Coder (7B parameters)
- `deepseek-r1:8b` - DeepSeek's R1 (8B parameters)

## Available Strategies (Bias Types)

- `Casual` - Neutral baseline
- `Positive` - "Code is well-written" (expects no smells)
- `Negative` - "Code is problematic" (expects smells)
- `Authoritative` - Authority bias ("As an expert...")
- `Social-Proof` - Social proof ("Most developers agree...")
- `Contradictory-Hint` - False premise ("It follows SOLID...")
- `False-Premise` - False claim ("Passed code review...")
- `Confirmation-Bias` - Asks for confirmation ("...has no smells, right?")

## Monitoring Progress

Check results as experiments run:
```bash
ls -lah ../results/ollama_results_*.json
tail -f ../results/experiment_log.txt
```

## Next Steps

1. Run Phase 3 sycophancy testing
2. Analyze Decision Flip Rate (DFR) from baseline vs bias
3. Run Phase 4 cross-model validation
4. Run Phase 5 comprehensive suite for publication data
