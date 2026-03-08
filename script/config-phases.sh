#!/bin/bash
# MLCQ Sycophancy Bias Study - Configuration
# Edit values below to control experiment phases

# Resolve project root relative to this config file
CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$CONFIG_DIR")"

# DATASET: 206 samples (13.8% of 1,495 raw) - balanced, representative subset
DATASET_PATH="$PROJECT_ROOT/dataset/mlcq_filtered.json"
RESULTS_DIR="$PROJECT_ROOT/results"
TEMPERATURE_GLOBAL="0.3"
TOP_P_GLOBAL="0.9"

# EXPERIMENT VARIANTS:
# Phase 1 (5-10 min):  Quick validation with 1 model, 1 smell, 1 strategy
# Phase 2 (45 min):    Baseline accuracy on all 4 smells, 1 model, Casual strategy
# Phase 3 (1-2 hrs):   Single model tests bias with 6 strategies on selected smells
#   - For Adversarial-Refutation: specify user comment(s) via [ar_comments]
# Phase 4 (1-2 hrs):   3 models test same bias/smell combo for cross-model validation
# Phase 5 (2-3 hrs):   Comprehensive: Feature Envy + Long Method, 3 models, all strategies
#
# ADVERSARIAL-REFUTATION USER COMMENTS:
# Available: Casual | Positive | Negative | Authoritative | Social-Proof |
#            Contradictory-Hint | False-Premise | Confirmation-Bias
# Usage in config: Pipe-separated list (will execute once per comment)
# Example: [ar_comments]="Casual|Positive|Confirmation-Bias"
# Note: Each run will use exactly ONE comment; multiple entries = multiple runs

declare -A PHASE1=(
    [name]="VALIDATION"
    [duration]="5-10 min"
    [purpose]="Quick pipeline validation"
    [models]="llama3.1:8b"
    [smells]="feature envy"
    [strategies]="Casual"
    [limit]="10"
)

declare -A PHASE2=(
    [name]="BASELINE DETECTION"
    [duration]="45 min"
    [purpose]="Ground-truth accuracy on labeled samples"
    [models]="llama3.1:8b"
    [smells]="feature envy|long method|data class|blob"
    [smell_limits]="feature-envy:175|long-method:175|data-class:100|blob:100"
    [strategies]="Casual"
)

#! Sycophancy bias testing for all 4 code smells
declare -A PHASE3=(
    [name]="SYCOPHANCY BIAS TESTING"
    [duration]="3-4 hrs"
    [purpose]="Measure Decision Flip Rate (DFR) across bias strategies for all smells"
    [models]="qwen2.5-coder:7b"
    [smells]="blob"
    # [strategies]="Casual|Positive|Contradictory-Hint|False-Premise|Confirmation-Bias|Adversarial-Refutation"
    [strategies]="Casual|Confirmation-Bias"
    # [strategies]="Adversarial-Refutation"
    [ar_comments]="Confirmation-Bias"  # User comments to test (pipe-separated for multiple)
    [limit]="175"
)

declare -A PHASE4=(
    [name]="CROSS-MODEL CONSISTENCY"
    [duration]="1-2 hrs"
    [purpose]="Verify bias patterns across model families"
    [models]="llama3.1:8b|qwen2.5-coder:7b|deepseek-r1:8b"
    [smells]="feature envy"
    [strategies]="Casual|Positive|Negative|Authoritative|Social-Proof|Contradictory-Hint|False-Premise|Confirmation-Bias|Adversarial-Refutation"
    [ar_comments]="Casual|Positive|Confirmation-Bias"  # Comments to cycle through for AR strategy
    [limit]="100"
)

declare -A PHASE5=(
    [name]="COMPREHENSIVE SUITE"
    [duration]="2-3 hrs"
    [purpose]="Full experiment for publication-ready results"
    [experiment_1]="smell:feature-envy|models:llama3.1:8b,qwen2.5-coder:7b,deepseek-r1:8b|strategies:all|limit:150"
    [experiment_2]="smell:long-method|models:llama3.1:8b,qwen2.5-coder:7b,deepseek-r1:8b|strategies:Casual,Positive,Negative,Authoritative,Social-Proof|limit:150"
)

# Quick reference: Strategies, Code Smells, Models, AR User Comments
list_strategies() {
    echo "  • Casual | Positive | Negative | Authoritative"
    echo "  • Social-Proof | Contradictory-Hint | False-Premise | Confirmation-Bias | Adversarial-Refutation"
}

list_smells() {
    echo "  • feature-envy | long-method | data-class | blob"
}

list_models() {
    echo "  • llama3.1:8b | qwen2.5-coder:7b | deepseek-r1:8b"
}

list_ar_comments() {
    echo "Available user comments for Adversarial-Refutation testing:"
    echo "  • Casual - Neutral baseline request"
    echo "  • Positive - Claims flawless quality, no smells"
    echo "  • Negative - Claims terrible quality, many smells"
    echo "  • Authoritative - Authority-based claim (Principal Engineer)"
    echo "  • Social-Proof - Majority opinion claim"
    echo "  • Contradictory-Hint - SOLID principles claim"
    echo "  • False-Premise - Static analysis false confidence"
    echo "  • Confirmation-Bias - Explicit conviction claim"
}

print_all_phases() {
    echo "Phase 1: VALIDATION (5-10 min)      - 1 model, 1 smell, 1 strategy"
    echo "Phase 2: BASELINE (45 min)          - All 4 smells baseline accuracy"
    echo "Phase 3: SYCOPHANCY (3-4 hrs)       - Selected smells × 6 strategies"
    echo "Phase 4: CROSS-MODEL (1-2 hrs)      - 3 models × 9 strategies, 1 smell"
    echo "Phase 5: COMPREHENSIVE (2-3 hrs)    - Feature Envy + Long Method, 3 models"
}

export DATASET_PATH RESULTS_DIR TEMPERATURE_GLOBAL TOP_P_GLOBAL
export -A PHASE1 PHASE2 PHASE3 PHASE4 PHASE5
