#!/usr/bin/env python3
"""
Test script to compare gentle SOI calibration options.

This script loads existing tax units and tests all gentle calibration approaches
to show their impact on filing status distributions.
"""

import sys
import os
import pandas as pd
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tax.units.status.gentle_calibration import (
    option_1_partial_calibration,
    option_2_probability_nudging, 
    option_3_income_weighted_adjustment,
    option_4_gradual_convergence,
    option_5_hybrid_approach,
    compare_calibration_options
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_latest_tax_units():
    """Load the most recent tax units file."""
    data_dir = Path(__file__).parent.parent / 'data' / 'processed'
    
    # Look for tax units files
    tax_unit_files = list(data_dir.glob('*tax_units*.parquet')) + list(data_dir.glob('hawaii_ctc_*.parquet'))
    
    if not tax_unit_files:
        raise FileNotFoundError(f"No tax units files found in {data_dir}")
    
    # Get the most recent file
    latest_file = max(tax_unit_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Loading tax units from: {latest_file}")
    
    df = pd.read_parquet(latest_file)
    logger.info(f"Loaded {len(df)} tax units")
    
    # Convert to list of dictionaries
    tax_units = df.to_dict('records')
    
    return tax_units


def analyze_filing_status_distribution(tax_units, title="Distribution"):
    """Analyze and display filing status distribution."""
    if not tax_units:
        return {}
    
    total = len(tax_units)
    filing_status_counts = {}
    
    for unit in tax_units:
        status = unit.get('filing_status', 'unknown')
        filing_status_counts[status] = filing_status_counts.get(status, 0) + 1
    
    print(f"\n{title}:")
    print("-" * 50)
    
    # SOI benchmarks for Hawaii
    soi_benchmarks = {
        'single': 0.51,
        'joint': 0.36, 
        'head_of_household': 0.096,
        'married_filing_separate': 0.034
    }
    
    for status in ['single', 'joint', 'head_of_household', 'married_filing_separate']:
        count = filing_status_counts.get(status, 0)
        pct = count / total if total > 0 else 0
        benchmark = soi_benchmarks.get(status, 0)
        gap = pct - benchmark
        
        print(f"  {status:20}: {count:5d} ({pct:5.1%}) | SOI: {benchmark:5.1%} | Gap: {gap:+5.1%}")
    
    calibrated = sum(1 for u in tax_units if u.get('calibrated', False))
    if calibrated > 0:
        print(f"  {'Calibrated units':20}: {calibrated:5d} ({calibrated/total:5.1%})")
    
    return filing_status_counts


def test_individual_options(tax_units):
    """Test each calibration option individually."""
    print("\n" + "="*80)
    print("TESTING INDIVIDUAL CALIBRATION OPTIONS")
    print("="*80)
    
    # Original distribution
    analyze_filing_status_distribution(tax_units, "ORIGINAL DISTRIBUTION")
    
    target_joint_pct = 0.36
    
    # Test each option
    options = [
        ("Option 1: Partial Calibration (max 2% adjustment)", 
         lambda units: option_1_partial_calibration(units, target_joint_pct, max_adjustment_pct=0.02)),
        
        ("Option 2: Probability Nudging (10% nudge strength)", 
         lambda units: option_2_probability_nudging(units, target_joint_pct, nudge_strength=0.1)),
        
        ("Option 3: Income-Weighted (70% confidence threshold)", 
         lambda units: option_3_income_weighted_adjustment(units, target_joint_pct, confidence_threshold=0.7)),
        
        ("Option 4: Gradual Convergence (30% convergence rate)", 
         lambda units: option_4_gradual_convergence(units, target_joint_pct, convergence_rate=0.3)),
        
        ("Option 5: Hybrid Approach (1.5% max total adjustment)", 
         lambda units: option_5_hybrid_approach(units, target_joint_pct, max_total_adjustment=0.015))
    ]
    
    results = {}
    
    for name, func in options:
        print(f"\n{'-'*60}")
        print(f"TESTING: {name}")
        print(f"{'-'*60}")
        
        try:
            adjusted_units = func(tax_units)
            distribution = analyze_filing_status_distribution(adjusted_units, f"RESULT: {name}")
            
            # Calculate key metrics
            original_joint = sum(1 for u in tax_units if u.get('filing_status') == 'joint')
            adjusted_joint = sum(1 for u in adjusted_units if u.get('filing_status') == 'joint')
            calibrated_count = sum(1 for u in adjusted_units if u.get('calibrated', False))
            
            original_pct = original_joint / len(tax_units)
            adjusted_pct = adjusted_joint / len(adjusted_units)
            
            results[name] = {
                'original_joint_pct': original_pct,
                'adjusted_joint_pct': adjusted_pct,
                'units_calibrated': calibrated_count,
                'calibration_rate': calibrated_count / len(adjusted_units),
                'distance_from_target': abs(adjusted_pct - target_joint_pct),
                'improvement': abs(original_pct - target_joint_pct) - abs(adjusted_pct - target_joint_pct)
            }
            
        except Exception as e:
            print(f"ERROR: {e}")
            results[name] = {'error': str(e)}
    
    return results


def compare_all_options(tax_units):
    """Use the built-in comparison function."""
    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON OF ALL OPTIONS")
    print("="*80)
    
    comparison = compare_calibration_options(tax_units, target_joint_pct=0.36)
    
    print(f"\n{'Option':<25} {'Joint %':<8} {'Adjusted':<9} {'Adj %':<6} {'Target Gap':<10} {'Improvement':<11}")
    print("-" * 80)
    
    for option_name, metrics in comparison.items():
        if 'error' in metrics:
            print(f"{option_name:<25} ERROR: {metrics['error']}")
            continue
            
        joint_pct = metrics['joint_pct']
        units_adj = metrics['units_adjusted']
        adj_pct = metrics['adjustment_pct']
        gap = metrics['distance_from_target']
        improvement = metrics.get('improvement', 0)
        
        print(f"{option_name:<25} {joint_pct:<7.1%} {units_adj:<9d} {adj_pct:<5.1%} {gap:<9.1%} {improvement:<+10.1%}")


def recommend_best_option(results):
    """Analyze results and recommend the best option."""
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    # Score each option based on multiple criteria
    scores = {}
    
    for name, metrics in results.items():
        if 'error' in metrics or name == 'original':
            continue
            
        # Criteria (lower is better for gaps, higher is better for improvement)
        target_gap = metrics.get('distance_from_target', 1.0)
        improvement = metrics.get('improvement', 0)
        calibration_rate = metrics.get('calibration_rate', 1.0)  # Lower is better
        
        # Scoring (0-100 scale)
        gap_score = max(0, 100 - (target_gap * 1000))  # Penalize large gaps heavily
        improvement_score = min(100, improvement * 1000)  # Reward improvement
        gentleness_score = max(0, 100 - (calibration_rate * 200))  # Reward fewer adjustments
        
        # Weighted total score
        total_score = (gap_score * 0.4) + (improvement_score * 0.4) + (gentleness_score * 0.2)
        
        scores[name] = {
            'total_score': total_score,
            'gap_score': gap_score,
            'improvement_score': improvement_score,
            'gentleness_score': gentleness_score,
            'metrics': metrics
        }
    
    # Find best option
    if scores:
        best_option = max(scores.keys(), key=lambda x: scores[x]['total_score'])
        best_metrics = scores[best_option]
        
        print(f"RECOMMENDED OPTION: {best_option}")
        print(f"Total Score: {best_metrics['total_score']:.1f}/100")
        print(f"  - Target Accuracy: {best_metrics['gap_score']:.1f}/100")
        print(f"  - Improvement: {best_metrics['improvement_score']:.1f}/100") 
        print(f"  - Gentleness: {best_metrics['gentleness_score']:.1f}/100")
        
        metrics = best_metrics['metrics']
        print(f"\nKey Metrics:")
        print(f"  - Joint filing rate: {metrics['adjusted_joint_pct']:.1%} (target: 36.0%)")
        print(f"  - Units adjusted: {metrics['units_calibrated']} ({metrics['calibration_rate']:.1%})")
        print(f"  - Distance from target: {metrics['distance_from_target']:.1%}")
        print(f"  - Improvement: {metrics['improvement']:+.1%}")
        
        return best_option
    
    return None


def main():
    """Main execution function."""
    try:
        # Load tax units
        tax_units = load_latest_tax_units()
        
        if not tax_units:
            print("No tax units loaded. Exiting.")
            return
        
        print(f"Testing gentle calibration options with {len(tax_units)} tax units")
        
        # Test individual options
        individual_results = test_individual_options(tax_units)
        
        # Compare all options
        compare_all_options(tax_units)
        
        # Make recommendation
        best_option = recommend_best_option(individual_results)
        
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Tested {len(individual_results)} calibration options")
        print(f"Best option: {best_option}")
        print(f"All options preserve data integrity while making targeted adjustments")
        print(f"Choose based on your preference for accuracy vs. minimal intervention")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
