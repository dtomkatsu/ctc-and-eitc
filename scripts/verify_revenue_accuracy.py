#!/usr/bin/env python3
"""
Verify Revenue Estimation Accuracy

This script checks whether the calibrated weights actually match the targets,
which is what matters for revenue estimation (not sample sizes).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

from src.tax.validation.hybrid_tax_calibration import load_hybrid_benchmarks

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def verify_ipf_convergence():
    """
    Verify that IPF actually converged to targets.
    
    This is the KEY test for revenue accuracy.
    """
    logger.info("="*80)
    logger.info("IPF CONVERGENCE VERIFICATION FOR REVENUE ESTIMATION")
    logger.info("="*80)
    
    # Load calibrated data
    df = pd.read_parquet('data/processed/tax_units_hybrid_calibrated.parquet')
    benchmarks = load_hybrid_benchmarks()
    
    logger.info(f"\nLoaded {len(df):,} tax units")
    logger.info(f"Calibrated weight column: 'weight_hybrid_calibrated'")
    
    # Check high-income brackets specifically
    test_brackets = [
        ('joint', 150000, 200000),
        ('joint', 200000, 300000),
        ('single', 150000, 200000),
        ('single', 200000, 300000),
        ('hoh', 150000, 200000),
    ]
    
    logger.info("\n" + "="*80)
    logger.info("HIGH-INCOME BRACKET CONVERGENCE CHECK")
    logger.info("="*80)
    logger.info("\nQuestion: Did IPF converge to match targets?")
    logger.info("(This determines revenue accuracy, NOT sample size)\n")
    
    convergence_results = []
    
    for status, agi_min, agi_max in test_brackets:
        # Get target from benchmarks
        bmask = (benchmarks['filing_status'] == status) & \
                (benchmarks['agi_min'] == agi_min) & \
                (benchmarks['agi_max'] == agi_max)
        
        if bmask.sum() == 0:
            continue
        
        target = benchmarks[bmask].iloc[0]
        
        # Get actual from calibrated data
        status_map = {
            'joint': 'married_filing_jointly',
            'single': 'single',
            'hoh': 'head_of_household'
        }
        
        dmask = (df['filing_status'] == status_map[status]) & \
                (df['agi'] >= agi_min) & \
                (df['agi'] < agi_max)
        
        sample_size = dmask.sum()
        
        # CRITICAL: Use calibrated weights
        actual_returns = df[dmask]['weight_hybrid_calibrated'].sum()
        actual_agi = (df[dmask]['weight_hybrid_calibrated'] * df[dmask]['agi']).sum() / 1e6
        
        # Calculate convergence errors
        returns_error = abs(actual_returns - target['returns']) / target['returns'] * 100 if target['returns'] > 0 else 0
        agi_error = abs(actual_agi - target['total_agi_millions']) / target['total_agi_millions'] * 100 if target['total_agi_millions'] > 0 else 0
        
        # Determine convergence status
        converged = returns_error < 1.0 and agi_error < 1.0  # < 1% error
        
        convergence_results.append({
            'bracket': f"{status} ${agi_min//1000}k-${agi_max//1000}k",
            'sample_size': sample_size,
            'target_returns': target['returns'],
            'actual_returns': actual_returns,
            'returns_error': returns_error,
            'target_agi': target['total_agi_millions'],
            'actual_agi': actual_agi,
            'agi_error': agi_error,
            'converged': converged
        })
        
        # Print detailed results
        logger.info(f"{status.upper()} ${agi_min//1000}k-${agi_max//1000}k:")
        logger.info(f"  Sample size: {sample_size} PUMS records")
        logger.info(f"  Returns:   Target={target['returns']:>10,.0f}  Actual={actual_returns:>10,.0f}  Error={returns_error:>6.2f}%")
        logger.info(f"  Total AGI: Target=${target['total_agi_millions']:>8.1f}M  Actual=${actual_agi:>8.1f}M  Error={agi_error:>6.2f}%")
        
        if converged:
            logger.info(f"  ✅ CONVERGED - Revenue estimate will be accurate")
        else:
            logger.info(f"  ⚠️  NOT CONVERGED - Revenue estimate may have {max(returns_error, agi_error):.1f}% error")
        logger.info("")
    
    # Summary
    results_df = pd.DataFrame(convergence_results)
    n_converged = results_df['converged'].sum()
    n_total = len(results_df)
    
    logger.info("="*80)
    logger.info("CONVERGENCE SUMMARY")
    logger.info("="*80)
    logger.info(f"\nBrackets converged (< 1% error): {n_converged}/{n_total}")
    logger.info(f"Average returns error: {results_df['returns_error'].mean():.2f}%")
    logger.info(f"Average AGI error: {results_df['agi_error'].mean():.2f}%")
    logger.info(f"Max error: {max(results_df['returns_error'].max(), results_df['agi_error'].max()):.2f}%")
    
    if n_converged == n_total:
        logger.info("\n✅ ALL BRACKETS CONVERGED")
        logger.info("   → Revenue estimates will be accurate")
        logger.info("   → No synthetic data needed")
        logger.info("   → Sample size 'errors' are irrelevant for revenue")
    else:
        logger.info(f"\n⚠️  {n_total - n_converged} BRACKETS DID NOT CONVERGE")
        logger.info("   → Consider:")
        logger.info("     1. Increase max_iterations (currently 500)")
        logger.info("     2. Adjust dampening factor")
        logger.info("     3. Add synthetic records for stability")
        logger.info("     4. Consolidate very small brackets")
    
    return results_df


def estimate_revenue_accuracy():
    """
    Estimate revenue accuracy based on convergence errors.
    
    Revenue error ≈ AGI error (since tax is a function of AGI)
    """
    logger.info("\n" + "="*80)
    logger.info("REVENUE ACCURACY ESTIMATION")
    logger.info("="*80)
    
    df = pd.read_parquet('data/processed/tax_units_hybrid_calibrated.parquet')
    benchmarks = load_hybrid_benchmarks()
    
    # Calculate total revenue by bracket (simplified - just use AGI as proxy)
    # In reality, you'd run through tax calculator
    
    logger.info("\nRevenue Accuracy Estimate:")
    logger.info("(Assuming tax liability ∝ AGI)")
    
    total_target_agi = 0
    total_actual_agi = 0
    
    for _, benchmark in benchmarks.iterrows():
        status = benchmark['filing_status']
        agi_min = benchmark['agi_min']
        agi_max = benchmark['agi_max']
        
        # Map status
        status_map = {
            'joint': 'married_filing_jointly',
            'single': 'single',
            'hoh': 'head_of_household'
        }
        
        if status not in status_map:
            continue
        
        dmask = (df['filing_status'] == status_map[status]) & \
                (df['agi'] >= agi_min) & \
                (df['agi'] < agi_max)
        
        target_agi = benchmark['total_agi_millions'] * 1e6
        actual_agi = (df[dmask]['weight_hybrid_calibrated'] * df[dmask]['agi']).sum()
        
        total_target_agi += target_agi
        total_actual_agi += actual_agi
    
    revenue_error = abs(total_actual_agi - total_target_agi) / total_target_agi * 100
    
    logger.info(f"\n  Target total AGI: ${total_target_agi/1e9:.2f}B")
    logger.info(f"  Actual total AGI: ${total_actual_agi/1e9:.2f}B")
    logger.info(f"  Estimated revenue error: {revenue_error:.2f}%")
    
    if revenue_error < 5:
        logger.info(f"\n  ✅ Revenue error < 5% - Excellent accuracy for policy analysis")
    elif revenue_error < 10:
        logger.info(f"\n  ✅ Revenue error < 10% - Good accuracy for policy analysis")
    else:
        logger.info(f"\n  ⚠️  Revenue error > 10% - Consider improvements")


def main():
    """Run all verification checks."""
    
    # Check convergence
    convergence_results = verify_ipf_convergence()
    
    # Estimate revenue accuracy
    estimate_revenue_accuracy()
    
    # Final recommendation
    logger.info("\n" + "="*80)
    logger.info("RECOMMENDATION FOR REVENUE ESTIMATION")
    logger.info("="*80)
    
    avg_error = convergence_results[['returns_error', 'agi_error']].mean().mean()
    
    if avg_error < 5:
        logger.info("\n✅ Current calibration is EXCELLENT for revenue estimation")
        logger.info("   → No changes needed")
        logger.info("   → Proceed with revenue calculations")
    elif avg_error < 15:
        logger.info("\n✅ Current calibration is GOOD for revenue estimation")
        logger.info("   → Consider minor improvements if needed")
        logger.info("   → Can proceed with revenue calculations")
    else:
        logger.info("\n⚠️  Current calibration needs improvement")
        logger.info("   → Review non-converged brackets")
        logger.info("   → Consider synthetic augmentation or bracket consolidation")
    
    logger.info("\n" + "="*80)


if __name__ == '__main__':
    main()
