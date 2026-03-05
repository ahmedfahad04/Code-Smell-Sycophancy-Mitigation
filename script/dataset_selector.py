#!/usr/bin/env python3
"""
Dataset Selector & Balancer
Extracts and validates a balanced representative subset from MLCQ dataset.
Ensures 50/50 smelly/clean split per code smell, with proper stratification.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import random


def load_dataset(filepath: str) -> List[Dict]:
    """Load MLCQ dataset from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


def analyze_dataset(dataset: List[Dict]) -> Dict:
    """Analyze dataset composition and balance."""
    stats = defaultdict(lambda: {'smelly': 0, 'clean': 0, 'critical': 0, 'major': 0, 'minor': 0, 'none': 0})
    
    for sample in dataset:
        smell = sample.get('smell', 'unknown').lower()
        severity = sample.get('severity', 'none').lower()
        
        # Determine if smelly based on severity
        is_smelly = severity in ('critical', 'major')
        
        stats[smell]['critical' if severity == 'critical' else 'none' if severity == 'none' else severity] += 1
        if is_smelly:
            stats[smell]['smelly'] += 1
        else:
            stats[smell]['clean'] += 1
    
    return dict(stats)


def print_dataset_analysis(stats: Dict):
    """Print formatted dataset analysis."""
    print("\n" + "="*80)
    print("DATASET ANALYSIS")
    print("="*80)
    
    total_samples = 0
    for smell, counts in sorted(stats.items()):
        smelly_count = counts['smelly']
        clean_count = counts['clean']
        total = smelly_count + clean_count
        
        smelly_pct = (smelly_count / total * 100) if total > 0 else 0
        
        print(f"\n{smell.upper():20} | Total: {total:5} | Smelly: {smelly_count:5} ({smelly_pct:5.1f}%) | Clean: {clean_count:5}")
        
        total_samples += total
    
    print(f"\nTOTAL SAMPLES: {total_samples}")
    print("="*80)


def validate_balance(samples: List[Dict], tolerance: float = 0.1) -> Tuple[bool, str]:
    """
    Validate that selected samples are balanced.
    tolerance: allowed deviation from 50/50 split (e.g., 0.1 = allow 40-60% range)
    """
    smelly_count = sum(1 for s in samples if s.get('severity', '').lower() in ('critical', 'major'))
    clean_count = len(samples) - smelly_count
    
    if len(samples) == 0:
        return False, "No samples selected"
    
    smelly_ratio = smelly_count / len(samples)
    
    lower_bound = 0.5 - tolerance
    upper_bound = 0.5 + tolerance
    
    is_balanced = lower_bound <= smelly_ratio <= upper_bound
    message = f"Smelly: {smelly_count} ({smelly_ratio*100:.1f}%), Clean: {clean_count} ({(1-smelly_ratio)*100:.1f}%)"
    
    return is_balanced, message


def select_balanced_subset(dataset: List[Dict], 
                          target_samples_per_smell: int,
                          seed: int = 42) -> Tuple[List[Dict], Dict]:
    """
    Select balanced subset from dataset.
    Returns both the selected samples and metadata about selection.
    """
    random.seed(seed)
    
    # Group by smell
    by_smell = defaultdict(list)
    for sample in dataset:
        smell = sample.get('smell', 'unknown').lower()
        by_smell[smell].append(sample)
    
    selected = []
    selection_report = {}
    
    for smell, samples in sorted(by_smell.items()):
        # Separate smelly and clean
        smelly = [s for s in samples if s.get('severity', '').lower() in ('critical', 'major')]
        clean = [s for s in samples if s.get('severity', '').lower() not in ('critical', 'major')]
        
        # Target 50/50 split
        target_per_category = target_samples_per_smell // 2
        
        # Select samples with stratification, ensure we have enough
        selected_smelly = random.sample(smelly, min(len(smelly), target_per_category))
        selected_clean = random.sample(clean, min(len(clean), target_per_category))
        
        selected_for_smell = selected_smelly + selected_clean
        selected.extend(selected_for_smell)
        
        is_balanced, balance_msg = validate_balance(selected_for_smell)
        
        selection_report[smell] = {
            'available_total': len(samples),
            'available_smelly': len(smelly),
            'available_clean': len(clean),
            'selected_total': len(selected_for_smell),
            'selected_smelly': len(selected_smelly),
            'selected_clean': len(selected_clean),
            'is_balanced': is_balanced,
            'balance_message': balance_msg,
        }
    
    return selected, selection_report


def print_selection_report(report: Dict):
    """Print formatted selection report."""
    print("\n" + "="*80)
    print("SELECTION REPORT")
    print("="*80)
    
    total_selected = 0
    for smell, stats in sorted(report.items()):
        print(f"\n{smell.upper():20}")
        print(f"  Available:  {stats['available_total']:5} (Smelly: {stats['available_smelly']:5}, Clean: {stats['available_clean']:5})")
        print(f"  Selected:   {stats['selected_total']:5} (Smelly: {stats['selected_smelly']:5}, Clean: {stats['selected_clean']:5})")
        print(f"  Balanced:   {'✓' if stats['is_balanced'] else '✗'} - {stats['balance_message']}")
        
        total_selected += stats['selected_total']
    
    print(f"\nTOTAL SELECTED: {total_selected}")
    print("="*80)


def save_dataset(samples: List[Dict], output_path: str):
    """Save selected dataset to JSON."""
    with open(output_path, 'w') as f:
        json.dump(samples, f, indent=2)
    print(f"\nDataset saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Select and balance dataset from MLCQ samples.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze full dataset
  python dataset_selector.py --dataset MLCQCodeSmellSamples_min5lines.json --analyze
  
  # Select 175 samples per smell (350 total for 2 smells)
  python dataset_selector.py \\
    --dataset MLCQCodeSmellSamples_min5lines.json \\
    --samples-per-smell 175 \\
    --output dataset_3day_subset.json
  
  # Select 350 samples per smell (1400 total for 4 smells)
  python dataset_selector.py \\
    --dataset MLCQCodeSmellSamples_min5lines.json \\
    --samples-per-smell 350 \\
    --smells "long method,feature envy,data class,blob" \\
    --output dataset_full.json
        """
    )
    
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to MLCQ dataset JSON file')
    parser.add_argument('--analyze', action='store_true',
                       help='Only analyze dataset, do not select')
    parser.add_argument('--samples-per-smell', type=int, default=175,
                       help='Number of samples to select per smell type')
    parser.add_argument('--smells', type=str,
                       help='Comma-separated list of smells to include (default: all)')
    parser.add_argument('--output', type=str,
                       help='Output JSON file path')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Load and analyze dataset
    print(f"Loading dataset from: {args.dataset}")
    dataset = load_dataset(args.dataset)
    print(f"Total samples loaded: {len(dataset)}")
    
    stats = analyze_dataset(dataset)
    print_dataset_analysis(stats)
    
    if args.analyze:
        return
    
    # Filter by smell if specified
    if args.smells:
        target_smells = {s.strip().lower() for s in args.smells.split(',')}
        dataset = [s for s in dataset if s.get('smell', '').lower() in target_smells]
        print(f"\nFiltered to target smells: {len(dataset)} samples")
    
    # Select balanced subset
    print(f"\nSelecting {args.samples_per_smell} samples per smell type...")
    selected, report = select_balanced_subset(dataset, args.samples_per_smell, seed=args.seed)
    
    print_selection_report(report)
    
    # Save if output specified
    if args.output:
        save_dataset(selected, args.output)
        print(f"\n✓ Successfully created balanced dataset with {len(selected)} samples")


if __name__ == '__main__':
    main()
