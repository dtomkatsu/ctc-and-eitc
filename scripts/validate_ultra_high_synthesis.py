#!/usr/bin/env python
"""
Validate Ultra-High-Income Synthesis

Check current state and identify gaps in $1M+ bracket representation.
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_ultra_high_synthesis():
    """
    Validate ultra-high-income synthesis and identify optimization opportunities.
    """
    print("="*80)
    print("ULTRA-HIGH-INCOME SYNTHESIS VALIDATION")
    print("="*80)
    
    # Load latest calibrated output
    df = pd.read_parquet('data/processed/tax_units_calibrated_20251028_143758.parquet')
    
    print("\n1. SYNTHETIC UNIT PERSISTENCE CHECK:")
    if 'is_synthetic_ultra_high' in df.columns:
        synthetic_count = (df['is_synthetic_ultra_high'] == True).sum()
        print(f"   ✅ Column 'is_synthetic_ultra_high' found")
        print(f"   Synthetic units: {synthetic_count}")
    else:
        print(f"   ❌ Column 'is_synthetic_ultra_high' NOT found (persistence issue)")
    
    print("\n2. $1M+ BRACKET ANALYSIS:")
    mask_1m = df['agi'] >= 1_000_000
    weight_1m = df.loc[mask_1m, 'weight'].sum()
    tax_1m = (df.loc[mask_1m, 'hi_state_tax'] * df.loc[mask_1m, 'weight']).sum() / 1_000_000
    
    print(f"   Filers: {mask_1m.sum()}")
    print(f"   Weight: {weight_1m:,.0f}")
    print(f"   Tax: ${tax_1m:,.1f}M")
    print(f"   Target: $663.0M")
    print(f"   Gap: ${tax_1m - 663.0:+.1f}M ({(tax_1m/663.0 - 1)*100:+.1f}%)")
    
    print("\n3. ULTRA-HIGH-INCOME DISTRIBUTION:")
    print("   AGI Bracket          Units    Weight    Tax ($M)   Avg Tax/Unit")
    print("   " + "-"*70)
    
    brackets = [
        (1_000_000, 5_000_000, "$1M-$5M"),
        (5_000_000, 10_000_000, "$5M-$10M"),
        (10_000_000, 25_000_000, "$10M-$25M"),
        (25_000_000, 50_000_000, "$25M-$50M"),
        (50_000_000, float('inf'), "$50M+"),
    ]
    
    for min_agi, max_agi, label in brackets:
        mask = (df['agi'] >= min_agi) & (df['agi'] < max_agi)
        if mask.sum() > 0:
            units = mask.sum()
            weight = df.loc[mask, 'weight'].sum()
            tax = (df.loc[mask, 'hi_state_tax'] * df.loc[mask, 'weight']).sum() / 1_000_000
            avg_tax = (tax * 1_000_000) / weight if weight > 0 else 0
            print(f"   {label:<15} {units:>6} {weight:>8,.0f} ${tax:>8.1f}M ${avg_tax:>10,.0f}")
    
    print("\n4. PARETO TAIL ANALYSIS:")
    print(f"   $5M+ filers: {(df['agi'] >= 5_000_000).sum()}")
    print(f"   $10M+ filers: {(df['agi'] >= 10_000_000).sum()}")
    print(f"   $50M+ filers: {(df['agi'] >= 50_000_000).sum()}")
    print(f"   $100M+ filers: {(df['agi'] >= 100_000_000).sum()}")
    
    print("\n5. OPTIMIZATION OPPORTUNITIES:")
    
    # Calculate what's needed
    current_tax_1m = tax_1m
    target_tax_1m = 663.0
    gap_1m = target_tax_1m - current_tax_1m
    
    # Estimate improvement from different strategies
    print(f"\n   To reach $663M target ($1M+ bracket):")
    print(f"   Current: ${current_tax_1m:.1f}M")
    print(f"   Gap: ${gap_1m:.1f}M")
    
    # Strategy 1: Increase tail multiplier
    print(f"\n   Strategy 1: Increase tail multiplier (0.15 → 0.35)")
    est_improvement_1 = gap_1m * 0.4  # Rough estimate
    print(f"   Estimated improvement: +${est_improvement_1:.1f}M")
    print(f"   New total: ${current_tax_1m + est_improvement_1:.1f}M")
    
    # Strategy 2: Optimize Pareto alpha
    print(f"\n   Strategy 2: Optimize Pareto alpha (1.454 → 1.5-1.6)")
    est_improvement_2 = gap_1m * 0.3  # Rough estimate
    print(f"   Estimated improvement: +${est_improvement_2:.1f}M")
    print(f"   New total: ${current_tax_1m + est_improvement_2:.1f}M")
    
    # Strategy 3: Combined
    print(f"\n   Strategy 3: Combined (tail multiplier + alpha optimization)")
    est_improvement_3 = gap_1m * 0.6  # Rough estimate
    print(f"   Estimated improvement: +${est_improvement_3:.1f}M")
    print(f"   New total: ${current_tax_1m + est_improvement_3:.1f}M")
    print(f"   Remaining gap: ${target_tax_1m - (current_tax_1m + est_improvement_3):.1f}M")
    
    print("\n6. RECOMMENDATIONS:")
    print("   ✅ IMMEDIATE:")
    print("      1. Fix synthetic unit persistence (ensure column survives)")
    print("      2. Run Pareto sensitivity analysis (test alpha 1.3-1.6)")
    print("      3. Increase tail multiplier from 0.15 to 0.35")
    print("\n   ⚠️  MEDIUM PRIORITY:")
    print("      4. Integrate IRS SOI superbracket targets ($10M+, $50M+)")
    print("      5. Validate synthetic unit counts against admin data")
    print("\n   📊 ANALYSIS:")
    print("      6. Run: python scripts/analyze_pareto_sensitivity.py")
    print("      7. Review: data/analysis/pareto_sensitivity_results.csv")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Edit scripts/regenerate_tax_units.py:")
    print("   - Ensure is_synthetic_ultra_high column is preserved")
    print("   - Use UltraHighIncomeSynthesizerV2 with tail_multiplier=0.35")
    print("\n2. Run sensitivity analysis:")
    print("   python scripts/analyze_pareto_sensitivity.py")
    print("\n3. Update ultra-high synthesis with optimal parameters")
    print("\n4. Re-run full pipeline:")
    print("   python scripts/regenerate_tax_units.py")


if __name__ == '__main__':
    validate_ultra_high_synthesis()
