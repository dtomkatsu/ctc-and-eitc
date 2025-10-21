#!/usr/bin/env python3
"""
Test Synthetic Augmentation for Hybrid Calibration

Compares calibration accuracy with and without synthetic augmentation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

from src.tax.validation.synthetic_augmentation import augment_undersampled_brackets
from src.tax.validation.hybrid_tax_calibration import apply_hybrid_tax_calibration, load_hybrid_benchmarks

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compare_calibration_accuracy(
    tax_units_path: Path,
    use_synthetic: bool = False
):
    """
    Compare calibration accuracy with/without synthetic augmentation.
    
    Args:
        tax_units_path: Path to tax units file
        use_synthetic: Whether to use synthetic augmentation
    """
    # Load tax units
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing Calibration {'WITH' if use_synthetic else 'WITHOUT'} Synthetic Augmentation")
    logger.info(f"{'='*80}")
    
    tax_units = pd.read_parquet(tax_units_path)
    logger.info(f"Loaded {len(tax_units):,} tax units")
    
    # Apply synthetic augmentation if requested
    if use_synthetic:
        tax_units = augment_undersampled_brackets(
            tax_units,
            min_sample_size=100,
            target_sample_size=300,
            noise_factor=0.03,  # ±3% noise
            synthetic_weight_factor=0.1  # 10% of original weight
        )
    
    # Apply calibration
    logger.info("\nApplying hybrid calibration...")
    calibrated = apply_hybrid_tax_calibration(
        tax_units,
        weight_col='weight',
        agi_col='agi',
        output_weight_col='weight_calibrated',
        max_iterations=500,
        tolerance=0.0005
    )
    
    # Calculate errors for high-income brackets
    benchmarks = load_hybrid_benchmarks()
    
    test_brackets = [
        ('joint', 150000, 200000),
        ('joint', 200000, 300000),
        ('single', 150000, 200000),
        ('single', 200000, 300000),
        ('hoh', 150000, 200000),
    ]
    
    results = []
    
    for status, agi_min, agi_max in test_brackets:
        # Get target
        bmask = (benchmarks['filing_status'] == status) & \
                (benchmarks['agi_min'] == agi_min) & \
                (benchmarks['agi_max'] == agi_max)
        
        if bmask.sum() == 0:
            continue
        
        target = benchmarks[bmask].iloc[0]
        
        # Get actual (only non-synthetic records for fair comparison)
        dmask = (calibrated['filing_status'].str.replace('married_filing_jointly', 'joint')
                                            .str.replace('head_of_household', 'hoh') == status) & \
                (calibrated['agi'] >= agi_min) & \
                (calibrated['agi'] < agi_max)
        
        if use_synthetic:
            # Count both original and synthetic
            actual_returns = calibrated[dmask]['weight_calibrated'].sum()
            actual_agi = (calibrated[dmask]['weight_calibrated'] * calibrated[dmask]['agi']).sum() / 1e6
            sample_size = dmask.sum()
        else:
            actual_returns = calibrated[dmask]['weight_calibrated'].sum()
            actual_agi = (calibrated[dmask]['weight_calibrated'] * calibrated[dmask]['agi']).sum() / 1e6
            sample_size = dmask.sum()
        
        returns_error = abs(actual_returns - target['returns']) / target['returns'] * 100 if target['returns'] > 0 else 0
        agi_error = abs(actual_agi - target['total_agi_millions']) / target['total_agi_millions'] * 100 if target['total_agi_millions'] > 0 else 0
        
        results.append({
            'bracket': f"{status} ${agi_min//1000}k-${agi_max//1000}k",
            'sample_size': sample_size,
            'returns_error': returns_error,
            'agi_error': agi_error,
            'avg_error': (returns_error + agi_error) / 2
        })
    
    # Print results
    logger.info(f"\n{'='*80}")
    logger.info(f"CALIBRATION ACCURACY RESULTS")
    logger.info(f"{'='*80}")
    
    results_df = pd.DataFrame(results)
    for _, row in results_df.iterrows():
        logger.info(f"\n{row['bracket']}:")
        logger.info(f"  Sample size: {row['sample_size']:>4.0f}")
        logger.info(f"  Returns error: {row['returns_error']:>6.2f}%")
        logger.info(f"  AGI error: {row['agi_error']:>6.2f}%")
        logger.info(f"  Avg error: {row['avg_error']:>6.2f}%")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Overall average error: {results_df['avg_error'].mean():.2f}%")
    logger.info(f"{'='*80}")
    
    return results_df


def main():
    """Compare calibration with and without synthetic augmentation."""
    tax_units_path = Path('data/processed/tax_units_agi.parquet')
    
    # Test without synthetic augmentation
    results_baseline = compare_calibration_accuracy(
        tax_units_path,
        use_synthetic=False
    )
    
    # Test with synthetic augmentation
    results_synthetic = compare_calibration_accuracy(
        tax_units_path,
        use_synthetic=True
    )
    
    # Compare results
    logger.info(f"\n{'='*80}")
    logger.info("IMPROVEMENT SUMMARY")
    logger.info(f"{'='*80}")
    
    comparison = pd.DataFrame({
        'bracket': results_baseline['bracket'],
        'baseline_error': results_baseline['avg_error'],
        'synthetic_error': results_synthetic['avg_error'],
        'improvement': results_baseline['avg_error'] - results_synthetic['avg_error']
    })
    
    for _, row in comparison.iterrows():
        improvement_pct = (row['improvement'] / row['baseline_error'] * 100) if row['baseline_error'] > 0 else 0
        logger.info(f"\n{row['bracket']}:")
        logger.info(f"  Baseline: {row['baseline_error']:>6.2f}%")
        logger.info(f"  Synthetic: {row['synthetic_error']:>6.2f}%")
        logger.info(f"  Improvement: {row['improvement']:>6.2f}pp ({improvement_pct:>+.1f}%)")
    
    avg_baseline = results_baseline['avg_error'].mean()
    avg_synthetic = results_synthetic['avg_error'].mean()
    overall_improvement = (avg_baseline - avg_synthetic) / avg_baseline * 100
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Overall Results:")
    logger.info(f"  Baseline avg error: {avg_baseline:.2f}%")
    logger.info(f"  Synthetic avg error: {avg_synthetic:.2f}%")
    logger.info(f"  Overall improvement: {overall_improvement:+.1f}%")
    logger.info(f"{'='*80}")


if __name__ == '__main__':
    main()
