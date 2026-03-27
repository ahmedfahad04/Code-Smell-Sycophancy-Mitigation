#!/bin/bash
################################################################################
# Sycophancy Bias Experiment Runner (Configuration-Driven)
#
# Uses config-phases.sh to control all experiment parameters
# Supports running individual phases or the complete suite
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
source "$PARENT_DIR/config-phases.sh"

LOG_FILE="$RESULTS_DIR/experiment_log.txt"
mkdir -p "$RESULTS_DIR"

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Logging functions
log() {
    echo -e "$@" | tee -a "$LOG_FILE"
}

log_info() {
    log "${BLUE}[INFO]${NC} $1"
}

log_success() {
    log "${GREEN}[✓]${NC} $1"
}

log_error() {
    log "${RED}[✗]${NC} $1"
}

log_section() {
    log ""
    log "════════════════════════════════════════════════════════════"
    log "  $1"
    log "════════════════════════════════════════════════════════════"
}

# Initialize log
echo "Experiment Run: $(date '+%Y-%m-%d %H:%M:%S')" > "$LOG_FILE"
echo "Hostname: $(hostname)" >> "$LOG_FILE"
echo "User: $(whoami)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

show_help() {
    cat << 'EOF'
Sycophancy Bias Experiment Runner (Configuration-Driven)

Usage: ./run_experiments.sh [COMMAND] [OPTIONS]

COMMANDS:
    validate       Phase 1: Quick validation (5-10 min)
    baseline       Phase 2: Baseline on all smells (45 min)
    sycophancy     Phase 3: Bias testing - Feature Envy (1-2 hrs)
    crossmodel     Phase 4: 3 models - all strategies (1-2 hrs)
    comprehensive  Phase 5: Full suite (2-3 hrs)
    
    all            Run all phases 1-5 sequentially (5-8 hrs)
    quick          Run phases 1-2 only (baseline + validation)
    bias-only      Run phases 3-4 only (sycophancy + cross-model)

INFORMATION:
    config         Show current phase configuration
    info           Show strategies, smells, models, AR comments available
    status         Check experiment prerequisites

CONFIGURATION:
    Edit config-phases.sh to adjust phase parameters:
    - For EGDP strategy, set [ar_comments] field
    - Format: pipe-separated list (e.g., "Casual|Positive|Confirmation-Bias")
    - Each run will use exactly ONE comment; multiple entries = multiple runs
    - Available: Casual, Positive, Negative, Authoritative, Social-Proof,
                 Contradictory-Hint, False-Premise, Confirmation-Bias

EXAMPLES:
    ./run_experiments.sh validate           # Test pipeline
    ./run_experiments.sh baseline           # Run baseline only
    ./run_experiments.sh quick              # Validation + Baseline
    ./run_experiments.sh all                # Complete study
    ./run_experiments.sh config             # View configuration
    ./run_experiments.sh info               # Show all available options

EOF
}

# Show current configuration
show_config() {
    log_section "EXPERIMENT PHASES CONFIGURATION"
    print_all_phases
}

# Show available options
show_info() {
    log_section "AVAILABLE STRATEGIES"
    list_strategies
    echo ""
    log_section "AVAILABLE CODE SMELLS"
    list_smells
    echo ""
    log_section "AVAILABLE MODELS"
    list_models
    echo ""
    log_section "AVAILABLE USER COMMENTS (for EGDP)"
    list_ar_comments
}

# Check prerequisites
check_status() {
    log_section "CHECKING PREREQUISITES"
    
    local all_ok=true
    
    # Check Python
    if command -v python3 &> /dev/null; then
        log_success "Python3 found: $(python3 --version)"
    else
        log_error "Python3 not found"
        all_ok=false
    fi
    
    # Check dataset
    if [ -f "$DATASET_PATH" ]; then
        log_success "Dataset found: $DATASET_PATH"
        local count=$(jq 'length' "$DATASET_PATH" 2>/dev/null || echo "unknown")
        log "  Samples: $count"
    else
        log_error "Dataset not found: $DATASET_PATH"
        all_ok=false
    fi
    
    # Check ollama models
    log "Checking Ollama models..."
    for model in "${PHASE4[models]//|/ }"; do
        if ollama list 2>/dev/null | grep -q "$model"; then
            log_success "Model available: $model"
        else
            log_error "Model NOT available: $model"
            all_ok=false
        fi
    done
    
    # Check results directory
    if [ -d "$RESULTS_DIR" ]; then
        log_success "Results directory exists: $RESULTS_DIR"
    else
        log "Creating results directory: $RESULTS_DIR"
        mkdir -p "$RESULTS_DIR"
    fi
    
    echo ""
    if [ "$all_ok" = true ]; then
        log_success "All prerequisites met - ready to run experiments!"
    else
        log_error "Some prerequisites missing - see above"
        return 1
    fi
}

# Run a detection test (with support for EGDP comment cycling)
run_detection() {
    local name=$1
    local smell=$2
    local models=$3
    local strategies=$4
    local limit=$5
    local temperature=${6:-$TEMPERATURE_GLOBAL}
    local top_p=${7:-$TOP_P_GLOBAL}
    local ar_comments=${8:-""}  # Optional: AR comments (pipe-separated)
    
    log_info "Test: $name"
    log_info "  Smell: $smell | Models: $models | Strategies: $strategies"
    log_info "  Samples: $limit | Temp: $temperature | Top-p: $top_p"
    
    # Check if EGDP is in strategies and we have AR comments
    if [[ "$strategies" == *"EGDP"* ]] && [[ -n "$ar_comments" ]]; then
        # Split ar_comments by pipe and run once per comment
        local ar_comments_array=()
        IFS='|' read -ra ar_comments_array <<< "$ar_comments"
        
        log_info "  Running EGDP with ${#ar_comments_array[@]} user comments:"
        for comment in "${ar_comments_array[@]}"; do
            log_info "    - $comment"
            
            local cmd="python3 '$PARENT_DIR/ollama_code_smell_detection.py' \
                --dataset '$DATASET_PATH' \
                --smell '$smell' \
                --models '$models' \
                --strategies 'EGDP' \
                --ar-comment '$comment' \
                --limit $limit \
                --output-dir '$RESULTS_DIR' \
                --temperature $temperature \
                --top-p $top_p"
            
            echo "  Command: $cmd" >> "$LOG_FILE"
            
            if eval "$cmd"; then
                log_success "  ✓ AR with comment '$comment' completed"
            else
                log_error "  ✗ AR with comment '$comment' failed (exit code: $?)"
                log_error "  Continuing to next comment..."
            fi
        done
    else
        # Standard (non-AR) execution
        local cmd="python3 '$PARENT_DIR/ollama_code_smell_detection.py' \
            --dataset '$DATASET_PATH' \
            --smell '$smell' \
            --models '$models' \
            --strategies '$strategies' \
            --limit $limit \
            --output-dir '$RESULTS_DIR' \
            --temperature $temperature \
            --top-p $top_p"
        
        echo "  Command: $cmd" >> "$LOG_FILE"
        
        if eval "$cmd"; then
            log_success "$name completed"
        else
            log_error "$name failed (exit code: $?)"
            log_error "Continuing to next test..."
        fi
    fi
    
    log ""
}

# ============================================================================
# PHASE IMPLEMENTATIONS
# ============================================================================

phase_validate() {
    log_section "PHASE 1: ${PHASE1[name]} (${PHASE1[duration]})"
    log "Purpose: ${PHASE1[purpose]}"
    log ""
    
    run_detection \
        "Feature Envy Sanity Check" \
        "feature envy" \
        "${PHASE1[models]}" \
        "${PHASE1[strategies]}" \
        "${PHASE1[limit]}" \
        "${PHASE1[temperature]}" \
        "${PHASE1[top_p]}"
    
    log_success "Phase 1 complete!"
}

phase_baseline() {
    log_section "PHASE 2: ${PHASE2[name]} (${PHASE2[duration]})"
    log "Purpose: ${PHASE2[purpose]}"
    log ""
    
    # Extract metrics from comma-separated lists
    local smells_array=()
    IFS='|' read -ra smells_array <<< "${PHASE2[smells]}"
    
    for smell in "${smells_array[@]}"; do
        # Map smell name to limit
        local limit="${PHASE2[limit]}"
        case "$smell" in
            "feature envy") limit="175" ;;
            "long method") limit="175" ;;
            "data class") limit="100" ;;
            "blob") limit="100" ;;
        esac
        
        run_detection \
            "Baseline: $smell" \
            "$smell" \
            "${PHASE2[models]}" \
            "${PHASE2[strategies]}" \
            "$limit" \
            "${PHASE2[temperature]}" \
            "${PHASE2[top_p]}"
    done
    
    log_success "Phase 2 complete!"
}

phase_sycophancy() {
    log_section "PHASE 3: ${PHASE3[name]} (${PHASE3[duration]})"
    log "Purpose: ${PHASE3[purpose]}"
    log "Metric: Decision Flip Rate (DFR) vs baseline"
    log ""
    
    # Extract smells and strategies
    local smells_array=()
    IFS='|' read -ra smells_array <<< "${PHASE3[smells]}"
    
    local strategies_array=()
    IFS='|' read -ra strategies_array <<< "${PHASE3[strategies]}"
    
    # Get AR comments if defined
    local ar_comments="${PHASE3[ar_comments]:-}"
    
    # For each smell, test all strategies
    for smell in "${smells_array[@]}"; do
        log_info "Testing $smell with ${#strategies_array[@]} strategies..."
        
        for strategy in "${strategies_array[@]}"; do
            run_detection \
                "Sycophancy: $smell - $strategy" \
                "$smell" \
                "${PHASE3[models]}" \
                "$strategy" \
                "${PHASE3[limit]}" \
                "${PHASE3[temperature]}" \
                "${PHASE3[top_p]}" \
                "$ar_comments"
        done
    done
    
    log_success "Phase 3 complete!"
}

phase_crossmodel() {
    log_section "PHASE 4: ${PHASE4[name]} (${PHASE4[duration]})"
    log "Purpose: ${PHASE4[purpose]}"
    log ""
    
    local models_array=()
    IFS='|' read -ra models_array <<< "${PHASE4[models]}"
    
    # Get AR comments if defined
    local ar_comments="${PHASE4[ar_comments]:-}"
    
    for model in "${models_array[@]}"; do
        run_detection \
            "Cross-Model: $model - All Strategies" \
            "${PHASE4[smells]}" \
            "$model" \
            "${PHASE4[strategies]}" \
            "${PHASE4[limit]}" \
            "${PHASE4[temperature]}" \
            "${PHASE4[top_p]}" \
            "$ar_comments"
    done
    
    log_success "Phase 4 complete!"
}

phase_comprehensive() {
    log_section "PHASE 5: ${PHASE5[name]} (${PHASE5[duration]})"
    log "Purpose: ${PHASE5[purpose]}"
    log ""
    
    # Feature Envy - All models - All strategies
    run_detection \
        "Comprehensive: Feature Envy - 3 Models - All Strategies" \
        "feature envy" \
        "llama3.1:8b,qwen2.5-coder:7b,deepseek-r1:8b" \
        "Casual,Positive,Negative,Authoritative,Social-Proof,Contradictory-Hint,False-Premise,Confirmation-Bias" \
        "150" \
        "${TEMPERATURE_GLOBAL}" \
        "${TOP_P_GLOBAL}"
    
    # Long Method - All models - Key strategies
    run_detection \
        "Comprehensive: Long Method - 3 Models - Key Strategies" \
        "long method" \
        "llama3.1:8b,qwen2.5-coder:7b,deepseek-r1:8b" \
        "Casual,Positive,Negative,Authoritative,Social-Proof" \
        "150" \
        "${TEMPERATURE_GLOBAL}" \
        "${TOP_P_GLOBAL}"
    
    log_success "Phase 5 complete!"
}

# ============================================================================
# MAIN
# ============================================================================

command=${1:-help}

case "$command" in
    validate)
        phase_validate
        ;;
    baseline)
        phase_baseline
        ;;
    sycophancy)
        phase_sycophancy
        ;;
    crossmodel)
        phase_crossmodel
        ;;
    comprehensive)
        phase_comprehensive
        ;;
    all)
        phase_validate
        phase_baseline
        phase_sycophancy
        phase_crossmodel
        phase_comprehensive
        ;;
    quick)
        phase_validate
        phase_baseline
        ;;
    bias-only)
        phase_sycophancy
        phase_crossmodel
        ;;
    config)
        show_config
        ;;
    info)
        show_info
        ;;
    status)
        check_status
        ;;
    help)
        show_help
        ;;
    *)
        log_error "Unknown command: $command"
        echo ""
        show_help
        exit 1
        ;;
esac

log ""
log_section "EXPERIMENT RUN COMPLETE"
log "Results saved to: $RESULTS_DIR"
log "Log file: $LOG_FILE"
log "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
log ""
