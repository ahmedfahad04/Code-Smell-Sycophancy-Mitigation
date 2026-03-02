
# Sycophancy Analysis in Code Smell Detection

This document focuses on experiments designed to understand how Large Language Models exhibit sycophantic behaviors when detecting code smells, particularly how different prompt strategies can bias or manipulate their responses.

## Core Sycophancy Experiments

### 🎯 Primary: Prompt Strategy Comparison
**Purpose**: Directly test sycophantic behaviors by comparing biased prompts designed to induce sycophancy against neutral prompts.

**What we learn about sycophancy**:
- How easily LLMs can be manipulated by biased prompts
- Effectiveness of sycophancy-inducing strategies (Positive, Confirmation-Bias, etc.)
- Whether models show preference for user-pleasing responses over accuracy
- Impact of authoritative, social proof, and contradictory prompts on detection

**Key Commands for Sycophancy Analysis**:
```bash
# Compare sycophancy-inducing prompts vs neutral
./test_sycophancy.sh --smell "feature envy" --models "llama3.1:8b" --strategies "Casual,Positive,Confirmation-Bias" --limit 20

# Test all biased strategies together
./test_sycophancy.sh --smell "blob" --models "llama3.1:8b" --strategies "Positive,Negative,Authoritative,Social-Proof,Contradictory-Hint,False-Premise,Confirmation-Bias" --limit 15

# Focus on extreme sycophancy cases
./test_sycophancy.sh --smell "data class" --models "llama3.1:8b" --strategies "Positive,Confirmation-Bias,Social-Proof" --limit 25
```

---

### 🔍 Secondary: Cross-Model Sycophancy Comparison
**Purpose**: Investigate whether different LLM architectures show varying susceptibility to sycophantic prompts.

**What we learn about sycophancy**:
- Which models are more prone to sycophantic behavior
- Whether larger models show more or less sycophancy
- Model-specific vulnerabilities to biased prompting
- Differences between general-purpose vs code-specialized models

**Key Commands for Model Comparison**:
```bash
# Compare sycophancy across different models
./test_sycophancy.sh --smell "long method" --models "llama3.1:8b,qwen2.5-coder:3b,deepseek-r1:8b" --strategies "Casual,Positive,Confirmation-Bias" --limit 15

# Test coding-focused models for sycophancy
./test_sycophancy.sh --smell "feature envy" --models "qwen2.5-coder:7b,qwen2.5-coder:3b" --strategies "Casual,Social-Proof,Contradictory-Hint" --limit 20
```

---

### 📊 Supporting: Cross-Smell Validation with Biased Prompts
**Purpose**: Test whether sycophantic prompts affect the model's ability to distinguish specific smells in mixed data.

**What we learn about sycophancy**:
- How sycophancy impacts robustness across different code patterns
- Whether biased prompts cause more false positives/negatives
- If sycophantic behavior varies by smell type complexity
- Trade-off between user-pleasing responses and detection accuracy

**Key Commands for Robustness Testing**:
```bash
# Test sycophancy in mixed-smell scenarios
./test_sycophancy.sh --smell "blob" --mixed-dataset --models "llama3.1:8b" --strategies "Casual,Positive,Confirmation-Bias" --limit 30

# Compare sycophancy effects across smell types
./test_sycophancy.sh --smell "data class" --mixed-dataset --models "llama3.1:8b" --strategies "Casual,Authoritative,Social-Proof" --limit 30
```

## Key Research Questions for Sycophancy

1. **Prompt Manipulation**: How effectively can biased prompts manipulate LLM responses?
2. **Model Vulnerability**: Which LLM architectures are most susceptible to sycophancy?
3. **Accuracy vs Compliance**: Do models prioritize user-pleasing responses over detection accuracy?
4. **Smell-Specific Effects**: Does sycophancy impact vary across different code smell types?
5. **Robustness Under Bias**: How well do models maintain detection capability when prompted with bias?

## Recommended Sycophancy Experiment Protocol

### Phase 1: Baseline Establishment
```bash
# Establish neutral baseline
./test_sycophancy.sh --smell "feature envy" --models "llama3.1:8b" --strategies "Casual" --limit 20
```

### Phase 2: Sycophancy Testing
```bash
# Test sycophancy-inducing prompts
./test_sycophancy.sh --smell "feature envy" --models "llama3.1:8b" --strategies "Positive,Confirmation-Bias,Social-Proof" --limit 20
```

### Phase 3: Cross-Model Analysis
```bash
# Compare sycophancy across models
./test_sycophancy.sh --smell "feature envy" --models "llama3.1:8b,qwen2.5-coder:3b" --strategies "Casual,Positive,Confirmation-Bias" --limit 15
```

### Phase 4: Robustness Validation
```bash
# Test sycophancy in challenging scenarios
./test_sycophancy.sh --smell "feature envy" --mixed-dataset --models "llama3.1:8b" --strategies "Casual,Positive,Confirmation-Bias" --limit 30
```

## Sycophancy Metrics to Track

- **Accuracy Drop**: Performance difference between neutral and biased prompts
- **False Negative Rate**: How often models miss smells when prompted positively
- **Response Consistency**: Variation in responses across different biased prompts
- **Model Differences**: Comparative sycophancy susceptibility across architectures

This focused approach enables systematic study of LLM sycophantic behaviors in code quality assessment, revealing how prompt engineering can influence automated code smell detection reliability.

