#!/usr/bin/env python
"""
Pareto Sensitivity Analysis

Test different Pareto alpha values (1.3-1.6) to find optimal tail representation
for $10M+, $50M+, $100M+ brackets.

Usage:
    python scripts/analyze_pareto_sensitivity.py
"""

import pandas as pd
import numpy as np
from src.tax.adjustments.ultra_high_income_synthesizer_v2 import UltraHighIncomeSynthesizerV2
from src.tax.brackets.hawaii_tax import HawaiiTaxCalculator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_pareto_sensitivity():
    """
    Test Pareto alpha values and tail multipliers to optimize $1M+ bracket matching.
    """
    print("="*80)
    print("PARETO SENSITIVITY ANALYSIS")
    print("="*80)
    
    # Load base tax units (before ultra-high synthesis)
    # This should be the output after income distribution calibration
    print("\nLoading tax units...")
    # For now, we'll use the latest calibrated file as a proxy
    df = pd.read_parquet('data/processed/tax_units_calibrated_20251028_143758.parquet')
    
    # Get current $1M+ baseline
    mask_1m = df['agi'] >= 1_000_000
    baseline_weight_1m = df.loc[mask_1m, 'weight'].sum()
    baseline_tax_1m = (df.loc[mask_1m, 'hi_state_tax'] * df.loc[mask_1m, 'weight']).sum() / 1_000_000
    
    print(f"\nBaseline $1M+ bracket:")
    print(f"  Weight: {baseline_weight_1m:,.0f}")
    print(f"  Tax: ${baseline_tax_1m:,.1f}M")
    print(f"  Target: $663.0M")
    print(f"  Gap: ${baseline_tax_1m - 663.0:+.1f}M")
    
    # Test parameters
    alpha_values = [1.3, 1.4, 1.454, 1.5, 1.6]
    tail_multipliers = [0.15, 0.25, 0.35, 0.45]
    
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS RESULTS")
    print("="*80)
    
    results = []
    
    for alpha in alpha_values:
        print(f"\n{'='*80}")
        print(f"Pareto Alpha: {alpha}")
        print(f"{'='*80}")
        print(f"{'Tail Mult':<12} {'$5M Weight':<12} {'$10M Weight':<12} {'$50M Weight':<12} {'Total Tax':<12} {'Gap to Target':<12}")
        print("-"*80)
        
        for tail_mult in tail_multipliers:
            synthesizer = UltraHighIncomeSynthesizerV2(
                pareto_alpha=alpha,
                tail_multiplier=tail_mult
            )
            
            # Calculate synthetic filer weights using Pareto
            current_filers = baseline_weight_1m
            
            weights = {}
            for agi_level in [5_000_000, 10_000_000, 50_000_000]:
                prob = synthesizer.calculate_pareto_probability(agi_level, threshold=1_000_000)
                if agi_level == 50_000_000:
                    weight = current_filers * prob * tail_mult
                else:
                    weight = current_filers * prob * (0.10 if agi_level == 5_000_000 else 0.08)
                weights[agi_level] = weight
            
            # Estimate tax contribution (rough approximation)
            # Using effective rates: $5M ~10.5%, $10M ~10.7%, $50M ~10.8%
            est_tax_5m = weights[5_000_000] * 5_000_000 * 0.105 / 1_000_000
            est_tax_10m = weights[10_000_000] * 10_000_000 * 0.107 / 1_000_000
            est_tax_50m = weights[50_000_000] * 50_000_000 * 0.108 / 1_000_000
            
            total_synthetic_tax = est_tax_5m + est_tax_10m + est_tax_50m
            total_tax_1m = baseline_tax_1m + total_synthetic_tax
            gap_to_target = 663.0 - total_tax_1m
            
            results.append({
                'alpha': alpha,
                'tail_mult': tail_mult,
                'weight_5m': weights[5_000_000],
                'weight_10m': weights[10_000_000],
                'weight_50m': weights[50_000_000],
                'total_tax_1m': total_tax_1m,
                'gap_to_target': gap_to_target,
            })
            
            print(f"{tail_mult:<12.2f} {weights[5_000_000]:<12.1f} {weights[10_000_000]:<12.1f} "
                  f"{weights[50_000_000]:<12.1f} ${total_tax_1m:<11.1f}M ${gap_to_target:<11.1f}M")
    
    # Find best combination (closest to target)
    results_df = pd.DataFrame(results)
    results_df['abs_gap'] = results_df['gap_to_target'].abs()
    best_idx = results_df['abs_gap'].idxmin()
    best = results_df.iloc[best_idx]
    
    print("\n" + "="*80)
    print("OPTIMAL CONFIGURATION")
    print("="*80)
    print(f"Pareto Alpha: {best['alpha']}")
    print(f"Tail Multiplier: {best['tail_mult']}")
    print(f"Estimated $1M+ Tax: ${best['total_tax_1m']:.1f}M")
    print(f"Gap to Target: ${best['gap_to_target']:+.1f}M")
    print(f"Improvement vs baseline: ${best['total_tax_1m'] - baseline_tax_1m:+.1f}M")
    
    # Save results
    results_df.to_csv('data/analysis/pareto_sensitivity_results.csv', index=False)
    print(f"\nResults saved to: data/analysis/pareto_sensitivity_results.csv")
    
    return results_df


if __name__ == '__main__':
    analyze_pareto_sensitivity()
