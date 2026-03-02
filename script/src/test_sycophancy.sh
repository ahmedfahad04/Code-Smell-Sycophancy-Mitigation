#!/bin/bash

# Central Script for Testing Prompt Sycophancy
# This script runs code smell detection with various prompt strategies
# and evaluates their effectiveness at detecting code smells.

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PARENT_DIR/../results"

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Available Models: qwen2.5-coder:7b, llama3.1:8b, deepseek-r1:8b
# Available Smells: feature envy, data class, blob, long method or all
# Detection Modes:
#   - --smell "all"                        → Blind detection (tests any smell)
#   - --smell "blob"                       → Ground-truth validation (only blob data)
#   - --smell "blob" --mixed-dataset       → Cross-smell validation (all data, test for blob)
# Available Strategies: "Positive", "Negative", "Authoritative", "Social-Proof", "Contradictory-Hint", "False-Premise", "Casual", "Confirmation-Bias"


# Default values
SMELL="feature envy"
MODELS="llama3.1:8b"
STRATEGIES="Casual"
LIMIT=1400
OUTPUT=""
TIMEOUT=60
RETRIES=3
DATASET="${PARENT_DIR}/dataset/MLCQCodeSmellSamples_min5lines.json"
SKIP_EVAL=false
SKIP_TEST=false
LIST_MODELS=false
MIXED_DATASET=true
TEMPERATURE="1"
TOP_P=""
FREQUENCY_PENALTY=""
PRESENCE_PENALTY=""

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Helper function to print colored output
print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Helper function to sanitize strategy name (underscore to hyphen)
sanitize_strategy() {
    echo "$1" | tr '_' '-'
}

# Help function
show_help() {
    cat << EOF
Prompt Sycophancy Testing & Evaluation Pipeline

This script tests how various prompt strategies (which may exhibit sycophancy)
affect code smell detection performance across different LLM models.

Usage: ./test_sycophancy.sh [OPTIONS]

OPTIONS:
    -c, --smell <smell>                 Code smell to test
                                        (default: "feature envy")
                                        Examples: "data class", "god class", "long method"

    -m, --models <models>               Comma-separated list of models to use
                                        (default: "llama3.1:8b")
                                        Example: "llama3.1:8b,qwen2.5-coder:3b"

    -s, --strategies <strategies>       Comma-separated list of prompt strategies
                                        (default: "Casual")
                                        Example: "Casual,Positive,Negative,Authority"

    -l, --limit <number>                Number of samples to test per model
                                        (default: 5)

    -o, --output <filename>             Output JSON filename (without path)
                                        (default: auto-generated based on smell/models/strategies)
                                        Example: "my_results.json"

    -t, --timeout <seconds>             Request timeout in seconds
                                        (default: 60)

    -r, --retries <number>              Number of retries on failure
                                        (default: 2)

    --dataset <path>                    Path to dataset JSON file
                                        (default: dataset/MLCQCodeSmellSamples_min5lines.json)

    --skip-eval                         Skip evaluation step, only run detection

    --skip-test                         Skip detection step, only run evaluation

    --mixed-dataset                     Use cross-smell validation mode
                                        Uses all data samples while testing for specific smell
                                        (only with specific --smell, not with "all")

    --temperature <value>               Temperature for model sampling (0.0-2.0, optional)
                                        (default: not set, uses Ollama default)

    --top-p <value>                     Top-p (nucleus) sampling parameter (0.0-1.0, optional)
                                        (default: not set, uses Ollama default)

    --frequency-penalty <value>         Frequency penalty parameter (0.0-2.0, optional)
                                        (default: not set, uses Ollama default)

    --presence-penalty <value>          Presence penalty parameter (0.0-2.0, optional)
                                        (default: not set, uses Ollama default)

    --list-models                       List available Ollama models and exit

    -h, --help                          Show this help message

EXAMPLES:
    # Test feature envy with default settings
    ./test_sycophancy.sh

    # Test data class with multiple strategies and models
    ./test_sycophancy.sh -c "data class" -m "llama3.1:8b,qwen2.5-coder:3b" -s "Casual,Positive,Negative" -l 10

    # Test blob with ground-truth validation (only blob samples)
    ./test_sycophancy.sh -c "blob" -l 20

    # Test blob with cross-smell validation (all samples, detect blob capability)
    ./test_sycophancy.sh -c "blob" --mixed-dataset -l 20

    # Blind detection (detect any smell in mixed dataset)
    ./test_sycophancy.sh -c "all" -l 50

    # Test with custom hyperparameters
    ./test_sycophancy.sh -c "feature envy" --temperature 0.5 --top-p 0.95 --frequency-penalty 0.3

    # Run with specific output filename
    ./test_sycophancy.sh -c "god class" -o my_god_class_test.json -l 20

    # Only evaluate existing results
    ./test_sycophancy.sh --skip-test

    # List available models
    ./test_sycophancy.sh --list-models

PROMPT STRATEGIES:
    Casual              - Normal, straightforward request to analyze for code smells
    Positive            - Biased to suggest no smells (sycophancy bias)
    Negative            - Biased to suggest smells (reverse bias)
    Authority           - Authority bias - appeals to expertise
    Social Proof        - Social proof bias - suggests others agree the code is clean
    Contradictory Hint  - Contradictory hint - claims code follows SOLID principles

DETECTION MODES:
    BLIND DETECTION (--smell "all"):
        - Uses all data samples from all 4 smell types
        - Tests with generic prompt that doesn't mention specific smell
        - Best for: Understanding what the model detects naturally

    GROUND-TRUTH VALIDATION (--smell "blob"):
        - Uses only samples matching the specified smell type
        - Tests with specific prompt for that smell
        - Best for: Baseline accuracy on known smell types

    CROSS-SMELL VALIDATION (--smell "blob" --mixed-dataset):
        - Uses all data samples but tests for specific smell
        - Tests with specific prompt applied to mixed data
        - Best for: Robustness testing across different code patterns

RESULTS:
    JSON results and CSV evaluation scores are saved to: $RESULTS_DIR/

EOF
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--smell)
            SMELL="$2"
            shift 2
            ;;
        -m|--models)
            MODELS="$2"
            shift 2
            ;;
        -s|--strategies)
            STRATEGIES="$2"
            shift 2
            ;;
        -l|--limit)
            LIMIT="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT="$2"
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -r|--retries)
            RETRIES="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        --skip-test)
            SKIP_TEST=true
            shift
            ;;
        --list-models)
            LIST_MODELS=true
            shift
            ;;
        --mixed-dataset)
            MIXED_DATASET=true
            shift
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --top-p)
            TOP_P="$2"
            shift 2
            ;;
        --frequency-penalty)
            FREQUENCY_PENALTY="$2"
            shift 2
            ;;
        --presence-penalty)
            PRESENCE_PENALTY="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Title banner
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   Prompt Sycophancy Testing & Evaluation Pipeline          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# List models if requested
if [ "$LIST_MODELS" = true ]; then
    print_info "Fetching available models from Ollama..."
    python3 "$SCRIPT_DIR/ollama_code_smell_detection.py" --list-models
    exit 0
fi

# Validate LIMIT is a positive integer
if ! [[ "$LIMIT" =~ ^[0-9]+$ ]] || [ "$LIMIT" -eq 0 ]; then
    print_error "Limit must be a positive integer"
    exit 1
fi

# Validate TIMEOUT is a positive integer
if ! [[ "$TIMEOUT" =~ ^[0-9]+$ ]] || [ "$TIMEOUT" -eq 0 ]; then
    print_error "Timeout must be a positive integer"
    exit 1
fi

# Validate RETRIES is a non-negative integer
if ! [[ "$RETRIES" =~ ^[0-9]+$ ]]; then
    print_error "Retries must be a non-negative integer"
    exit 1
fi

# Check if dataset exists
if [ ! -f "$DATASET" ]; then
    print_error "Dataset not found: $DATASET"
    exit 1
fi

# Display configuration
print_info "Configuration:"
echo "  Code Smell:    $SMELL"
echo "  Models:        $MODELS"
echo "  Strategies:    $STRATEGIES"
echo "  Limit:         $LIMIT samples per model"
echo "  Timeout:       $TIMEOUT seconds"
echo "  Retries:       $RETRIES"
echo "  Results Dir:   $RESULTS_DIR"
if [ "$MIXED_DATASET" = true ]; then
    echo "  Detection Mode: CROSS-SMELL VALIDATION"
elif [ "$SMELL" = "all" ]; then
    echo "  Detection Mode: BLIND DETECTION"
else
    echo "  Detection Mode: GROUND-TRUTH VALIDATION"
fi
echo "  Hyperparameters:"
if [ -n "$TEMPERATURE" ]; then
    echo "    - Temperature:        $TEMPERATURE"
else
    echo "    - Temperature:        (not set)"
fi
if [ -n "$TOP_P" ]; then
    echo "    - Top-p:              $TOP_P"
else
    echo "    - Top-p:              (not set)"
fi
if [ -n "$FREQUENCY_PENALTY" ]; then
    echo "    - Frequency Penalty:  $FREQUENCY_PENALTY"
else
    echo "    - Frequency Penalty:  (not set)"
fi
if [ -n "$PRESENCE_PENALTY" ]; then
    echo "    - Presence Penalty:   $PRESENCE_PENALTY"
else
    echo "    - Presence Penalty:   (not set)"
fi
if [ -n "$OUTPUT" ]; then
    echo "  Output file:   $OUTPUT"
fi
echo ""

# Run detection if not skipped
if [ "$SKIP_TEST" = false ]; then
    print_info "━━━ Running Smell Detection ━━━"
    echo ""
    
    TEST_CMD="python3 \"$SCRIPT_DIR/ollama_code_smell_detection.py\" --smell \"$SMELL\" --models \"$MODELS\" --strategies \"$STRATEGIES\" --limit $LIMIT --output-dir \"$RESULTS_DIR\""
    
    if [ -n "$TEMPERATURE" ]; then
        TEST_CMD="$TEST_CMD --temperature $TEMPERATURE"
    fi
    if [ -n "$TOP_P" ]; then
        TEST_CMD="$TEST_CMD --top-p $TOP_P"
    fi
    if [ -n "$FREQUENCY_PENALTY" ]; then
        TEST_CMD="$TEST_CMD --frequency-penalty $FREQUENCY_PENALTY"
    fi
    if [ -n "$PRESENCE_PENALTY" ]; then
        TEST_CMD="$TEST_CMD --presence-penalty $PRESENCE_PENALTY"
    fi
    
    if [ "$MIXED_DATASET" = true ]; then
        TEST_CMD="$TEST_CMD --mixed-dataset"
    fi
    
    if [ -n "$OUTPUT" ]; then
        TEST_CMD="$TEST_CMD --output \"$OUTPUT\""
    fi
    
    print_info "Command: $TEST_CMD"
    echo ""
    
    if eval "$TEST_CMD"; then
        print_success "Smell detection completed successfully"
    else
        print_error "Smell detection failed"
        exit 1
    fi
    echo ""
else
    print_warning "Skipping smell detection (--skip-test)"
    echo ""
fi

# Run evaluation if not skipped
if [ "$SKIP_EVAL" = false ]; then
    print_info "━━━ Running Evaluation ━━━"
    echo ""
    
    # Generate base CSV filename based on smell only (strategies will be appended per strategy)
    SANITIZED_SMELL="${SMELL// /_}"
    CSV_FILENAME="evaluation_${SANITIZED_SMELL}.csv"
    CSV_PATH="$RESULTS_DIR/$CSV_FILENAME"
    
    EVAL_CMD="python3 \"$SCRIPT_DIR/evaluate_smell_results.py\" --dataset \"$DATASET\" --results-dir \"$RESULTS_DIR\" --results-glob \"ollama_results_${SANITIZED_SMELL}_*.json\" --output-csv \"$CSV_PATH\""
    
    print_info "Command: $EVAL_CMD"
    echo ""
    
    if eval "$EVAL_CMD"; then
        print_success "Evaluation completed successfully"
        print_info "CSV results saved to: $CSV_PATH"
    else
        print_error "Evaluation failed"
        exit 1
    fi
    echo ""
else
    print_warning "Skipping evaluation (--skip-eval)"
    echo ""
fi

print_success "Pipeline completed successfully!"
print_info "Results saved to: $RESULTS_DIR"
echo ""
