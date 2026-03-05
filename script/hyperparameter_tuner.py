#!/usr/bin/env python3
"""
Hyperparameter Impact Study Script
Tests the effect of temperature, top-p, and penalty parameters on model stability.
Helps select optimal hyperparameters before full experiment run.
"""

import json
import argparse
import subprocess
import json
from pathlib import Path
from typing import Dict, List
import statistics
import pandas as pd


def run_detection(
    dataset_file: str,
    model: str,
    smell: str,
    limit: int,
    temperature: float = None,
    top_p: float = None,
    frequency_penalty: float = None,
    presence_penalty: float = None,
    results_dir: str = None,
) -> Dict:
    """Run code smell detection with specified hyperparameters."""
    
    cmd = [
        'python3',
        'ollama_code_smell_detection.py',
        '--smell', smell,
        '--models', model,
        '--strategies', 'Casual',
        '--limit', str(limit),
        '--output-dir', results_dir or '.',
        '--dataset', dataset_file,
    ]
    
    if temperature is not None:
        cmd.extend(['--temperature', str(temperature)])
    if top_p is not None:
        cmd.extend(['--top-p', str(top_p)])
    if frequency_penalty is not None:
        cmd.extend(['--frequency-penalty', str(frequency_penalty)])
    if presence_penalty is not None:
        cmd.extend(['--presence-penalty', str(presence_penalty)])
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return None
        
        # Parse output to find JSON result file
        output_dir = Path(results_dir or '.')
        json_files = list(output_dir.glob('ollama_results_*.json'))
        
        if json_files:
            latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
            with open(latest_file, 'r') as f:
                return json.load(f)
    except subprocess.TimeoutExpired:
        print("Error: Detection timed out")
    except Exception as e:
        print(f"Error: {e}")
    
    return None


def extract_predictions(results: Dict) -> List[str]:
    """Extract prediction list from results."""
    predictions = []
    # Support two formats: a dict with key 'results' or a plain list of result dicts
    items = []
    if isinstance(results, dict) and 'results' in results:
        items = results.get('results', [])
    elif isinstance(results, list):
        items = results
    else:
        return predictions

    for item in items:
        # Preferred fields from ollama_code_smell_detection.py outputs: 'smell' and 'severity'
        model_smell = item.get('smell') if isinstance(item, dict) else None
        severity = item.get('severity') if isinstance(item, dict) else None

        # Normalize to binary label: 'smelly' if smell is not 'none' or severity indicates issue
        label = 'clean'
        if model_smell and str(model_smell).strip().lower() != 'none':
            label = 'smelly'
        elif severity and str(severity).strip().lower() in ('minor', 'major'):
            label = 'smelly'

        predictions.append(label)

    return predictions


def calculate_consistency(prediction_lists: List[List[str]]) -> Dict:
    """
    Calculate consistency metrics across multiple runs with same hyperparams.
    """
    if len(prediction_lists) < 2:
        total_samples = len(prediction_lists[0]) if prediction_lists else 0
        return {
            'consistency_score': 100.0,
            'variance': 0.0,
            'stable': True,
            'perfect_agreement_samples': total_samples,
            'total_samples': total_samples,
        }
    
    # Agreement rate: % of predictions where all runs agree
    total_samples = len(prediction_lists[0])
    perfect_agreement = 0
    
    for sample_idx in range(total_samples):
        predictions_for_sample = [run[sample_idx] for run in prediction_lists]
        if len(set(predictions_for_sample)) == 1:  # All predictions are same
            perfect_agreement += 1
    
    consistency = (perfect_agreement / total_samples * 100) if total_samples > 0 else 0.0
    
    # Variance in interpretation
    variance = 100 - consistency
    stable = consistency >= 90.0
    
    return {
        'consistency_score': consistency,
        'variance': variance,
        'stable': stable,
        'perfect_agreement_samples': perfect_agreement,
        'total_samples': total_samples,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Study hyperparameter impact on model consistency.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick study: test 5 samples with 2 runs per hyperparameter combo
  python hyperparameter_tuner.py \\
    --dataset /home/sakib/Documents/MLCQ/script/dataset/dataset_3day_balanced.json \\
    --model llama3.1:8b \\
    --smell "long method" \\
    --limit 5 \\
    --runs 2
  
  # Comprehensive study: test multiple hyperparameter values
  python hyperparameter_tuner.py \\
    --dataset /home/sakib/Documents/MLCQ/script/dataset/dataset_3day_balanced.json \\
    --model "llama3.1:8b,qwen2.5-coder:7b" \\
    --smell "feature envy" \\
    --limit 10 \\
    --temperature "0.1,0.3,0.5,0.7" \\
    --runs 3 \\
    --output-csv hyperparameter_study.csv
        """
    )
    
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to dataset subset JSON file')
    parser.add_argument('--model', type=str, default='llama3.1:8b',
                       help='Model to test (single model, no comma separation)')
    parser.add_argument('--smell', type=str, default='long method',
                       help='Code smell to test')
    parser.add_argument('--limit', type=int, default=10,
                       help='Number of samples per run')
    parser.add_argument('--runs', type=int, default=3,
                       help='Number of runs per hyperparameter combination')
    parser.add_argument('--temperature', type=str, default='0.1,0.3,0.5,0.7',
                       help='Comma-separated temperature values to test')
    parser.add_argument('--top-p', type=str, default='0.85,0.9,0.95',
                       help='Comma-separated top-p values to test')
    parser.add_argument('--frequency-penalty', type=float,
                       help='Fixed frequency penalty (optional)')
    parser.add_argument('--presence-penalty', type=float,
                       help='Fixed presence penalty (optional)')
    parser.add_argument('--output-csv', type=str,
                       help='Output CSV file for results')
    parser.add_argument('--results-dir', type=str, default='./results',
                       help='Directory to store temporary results')
    
    args = parser.parse_args()
    
    # Create results directory
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    
    # Parse hyperparameter values
    temperatures = [float(t) for t in args.temperature.split(',')]
    top_ps = [float(p) for p in args.top_p.split(',')]
    
    print("\n" + "="*80)
    print("HYPERPARAMETER IMPACT STUDY")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Smell: {args.smell}")
    print(f"Samples per run: {args.limit}")
    print(f"Runs per configuration: {args.runs}")
    print(f"Temperatures: {temperatures}")
    print(f"Top-p values: {top_ps}")
    print("="*80 + "\n")
    
    results = []
    
    # Test combinations
    combo_count = 0
    total_combos = len(temperatures) * len(top_ps)
    
    for temperature in temperatures:
        for top_p in top_ps:
            combo_count += 1
            print(f"\n[{combo_count}/{total_combos}] Testing T={temperature}, top_p={top_p}")
            print("-" * 60)
            
            run_predictions = []
            
            for run_num in range(args.runs):
                print(f"  Run {run_num + 1}/{args.runs}...", end=" ", flush=True)
                
                result = run_detection(
                    dataset_file=args.dataset,
                    model=args.model,
                    smell=args.smell,
                    limit=args.limit,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=args.frequency_penalty,
                    presence_penalty=args.presence_penalty,
                    results_dir=args.results_dir,
                )
                
                if result:
                    predictions = extract_predictions(result)
                    run_predictions.append(predictions)
                    print(f"✓ ({len(predictions)} samples)")
                else:
                    print("✗ Failed")
            
            # Calculate consistency
            if len(run_predictions) == args.runs:
                consistency_metrics = calculate_consistency(run_predictions)
                
                result_row = {
                    'temperature': temperature,
                    'top_p': top_p,
                    'frequency_penalty': args.frequency_penalty or 'default',
                    'presence_penalty': args.presence_penalty or 'default',
                    'runs_completed': len(run_predictions),
                    'consistency_percent': consistency_metrics['consistency_score'],
                    'variance_percent': consistency_metrics['variance'],
                    'stable': consistency_metrics['stable'],
                    'perfect_agreement': consistency_metrics['perfect_agreement_samples'],
                }
                
                results.append(result_row)
                
                print(f"\n  Consistency: {consistency_metrics['consistency_score']:.1f}%")
                print(f"  Variance: {consistency_metrics['variance']:.1f}%")
                print(f"  Stable: {'YES ✓' if consistency_metrics['stable'] else 'NO ✗'}")
            else:
                print(f"  Skipped: Insufficient successful runs ({len(run_predictions)}/{args.runs})")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if results:
        df = pd.DataFrame(results)
        
        # Sort by consistency
        df_sorted = df.sort_values('consistency_percent', ascending=False)
        
        print("\nResults (sorted by consistency):")
        print(df_sorted.to_string(index=False))
        
        # Recommendations
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        
        stable_results = df[df['stable']]
        if not stable_results.empty:
            best = stable_results.iloc[0]
            print(f"\n✓ MOST STABLE CONFIGURATION:")
            print(f"  Temperature: {best['temperature']}")
            print(f"  Top-p: {best['top_p']}")
            print(f"  Consistency: {best['consistency_percent']:.1f}%")
            
            # Find others with same or similar stability
            similar = stable_results[
                (stable_results['consistency_percent'] >= best['consistency_percent'] - 5)
            ]
            if len(similar) > 1:
                print(f"\n  Also consider:")
                for _, row in similar.iloc[1:].head(3).iterrows():
                    print(f"    - T={row['temperature']}, top_p={row['top_p']} ({row['consistency_percent']:.1f}%)")
        else:
            print("\n⚠ WARNING: No fully stable configurations found!")
            best = df.loc[df['consistency_percent'].idxmax()]
            print(f"  Best option: T={best['temperature']}, top_p={best['top_p']} ({best['consistency_percent']:.1f}%)")
        
        print("\n" + "="*80)
        
        # Save results
        if args.output_csv:
            df_sorted.to_csv(args.output_csv, index=False)
            print(f"\nResults saved to: {args.output_csv}")
    else:
        print("No results collected. Check configuration and try again.")


if __name__ == '__main__':
    main()
