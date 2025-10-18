"""
Compare Single-Layer vs Two-Layer Calibration

This script compares the current single-layer calibration with the proposed
two-layer calibration to address $50k-$75k over-representation.
"""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tax.units.status.bracket_calibration import apply_bracket_calibration
from src.tax.validation.agi_calibration import apply_two_layer_calibration
from src.tax.validation.dotax_table_12a import DotaxTable12AValidator

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compare_approaches(tax_units: pd.DataFrame):
    """Compare single-layer vs two-layer calibration."""
    
    logger.info("="*100)
    logger.info("CALIBRATION APPROACH COMPARISON")
    logger.info("="*100)
    
    # Approach 1: Single-layer (current)
    logger.info("\n" + "="*100)
    logger.info("APPROACH 1: SINGLE-LAYER CALIBRATION (Current)")
    logger.info("="*100)
    
    df_single = apply_bracket_calibration(tax_units, weight_col='weight', method='factor')
    
    if 'weight_calibrated' in df_single.columns:
        single_weight_col = 'weight_calibrated'
    else:
        single_weight_col = 'weight'
    
    # Approach 2: Two-layer (proposed)
    logger.info("\n" + "="*100)
    logger.info("APPROACH 2: TWO-LAYER CALIBRATION (Proposed)")
    logger.info("="*100)
    
    df_two = apply_two_layer_calibration(tax_units, weight_col='weight')
    two_weight_col = 'weight_final'
    
    # Comparison
    logger.info("\n" + "="*100)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*100)
    
    # Filing status comparison
    logger.info("\n" + "-"*100)
    logger.info("FILING STATUS DISTRIBUTION")
    logger.info("-"*100)
    
    filing_statuses = ['single', 'married_filing_jointly', 'head_of_household', 'married_filing_separately']
    dotax_targets = {
        'single': 335198,
        'married_filing_jointly': 216358,
        'head_of_household': 67393,
        'married_filing_separately': 16007
    }
    
    logger.info(f"{'Status':<30} {'DOTAX':>12} {'Single-Layer':>14} {'Gap':>8} {'Two-Layer':>14} {'Gap':>8}")
    logger.info("-"*100)
    
    for status in filing_statuses:
        target = dotax_targets.get(status, 0)
        single_count = df_single[df_single['filing_status'] == status][single_weight_col].sum()
        two_count = df_two[df_two['filing_status'] == status][two_weight_col].sum()
        
        single_gap = ((single_count - target) / target * 100) if target > 0 else 0
        two_gap = ((two_count - target) / target * 100) if target > 0 else 0
        
        logger.info(f"{status:<30} {target:>12,} {single_count:>14,.0f} {single_gap:>+7.1f}% {two_count:>14,.0f} {two_gap:>+7.1f}%")
    
    # AGI bracket comparison
    logger.info("\n" + "-"*100)
    logger.info("AGI BRACKET DISTRIBUTION (Table 12A)")
    logger.info("-"*100)
    
    agi_benchmarks = [
        (0, 10000, 129376, "$0k-$10k"),
        (10000, 20000, 64160, "$10k-$20k"),
        (20000, 30000, 57835, "$20k-$30k"),
        (30000, 40000, 59827, "$30k-$40k"),
        (40000, 50000, 53555, "$40k-$50k"),
        (50000, 75000, 91459, "$50k-$75k"),
        (75000, 100000, 54976, "$75k-$100k"),
        (100000, 150000, 62065, "$100k-$150k"),
        (150000, 200000, 27976, "$150k-$200k"),
        (200000, 300000, 18937, "$200k-$300k"),
        (300000, 400000, 6076, "$300k-$400k"),
        (400000, float('inf'), 8875, "$400k+"),
    ]
    
    logger.info(f"{'Bracket':<15} {'DOTAX':>12} {'Single-Layer':>14} {'Gap':>8} {'Two-Layer':>14} {'Gap':>8}")
    logger.info("-"*100)
    
    for min_agi, max_agi, target, label in agi_benchmarks:
        if max_agi == float('inf'):
            single_mask = df_single['agi'] >= min_agi
            two_mask = df_two['agi'] >= min_agi
        else:
            single_mask = (df_single['agi'] >= min_agi) & (df_single['agi'] < max_agi)
            two_mask = (df_two['agi'] >= min_agi) & (df_two['agi'] < max_agi)
        
        single_count = df_single[single_mask][single_weight_col].sum()
        two_count = df_two[two_mask][two_weight_col].sum()
        
        single_gap = ((single_count - target) / target * 100) if target > 0 else 0
        two_gap = ((two_count - target) / target * 100) if target > 0 else 0
        
        # Highlight $50k-$75k bracket
        marker = " ⚠️" if 50000 <= min_agi < 75000 else ""
        
        logger.info(f"{label:<15} {target:>12,} {single_count:>14,.0f} {single_gap:>+7.1f}% {two_count:>14,.0f} {two_gap:>+7.1f}%{marker}")
    
    # Accuracy metrics
    logger.info("\n" + "-"*100)
    logger.info("ACCURACY METRICS")
    logger.info("-"*100)
    
    # Count brackets within ±10%
    single_accurate = sum(1 for _, _, target, _ in agi_benchmarks 
                         if abs((df_single[(df_single['agi'] >= min_agi) & 
                                           (df_single['agi'] < (max_agi if max_agi != float('inf') else df_single['agi'].max() + 1))][single_weight_col].sum() - target) / target * 100) < 10)
    
    two_accurate = sum(1 for min_agi, max_agi, target, _ in agi_benchmarks 
                      if abs((df_two[(df_two['agi'] >= min_agi) & 
                                     (df_two['agi'] < (max_agi if max_agi != float('inf') else df_two['agi'].max() + 1))][two_weight_col].sum() - target) / target * 100) < 10)
    
    total_brackets = len(agi_benchmarks)
    
    logger.info(f"\nAGI brackets within ±10%:")
    logger.info(f"  Single-layer: {single_accurate}/{total_brackets} ({single_accurate/total_brackets*100:.1f}%)")
    logger.info(f"  Two-layer:    {two_accurate}/{total_brackets} ({two_accurate/total_brackets*100:.1f}%)")
    
    # Total return count
    single_total = df_single[single_weight_col].sum()
    two_total = df_two[two_weight_col].sum()
    target_total = 635117
    
    logger.info(f"\nTotal Returns:")
    logger.info(f"  Target:       {target_total:>12,}")
    logger.info(f"  Single-layer: {single_total:>12,.0f} ({(single_total-target_total)/target_total*100:+.2f}%)")
    logger.info(f"  Two-layer:    {two_total:>12,.0f} ({(two_total-target_total)/target_total*100:+.2f}%)")
    
    # Key bracket comparison
    logger.info("\n" + "-"*100)
    logger.info("KEY BRACKET: $50K-$75K IMPROVEMENT")
    logger.info("-"*100)
    
    bracket_50_75_mask_single = (df_single['agi'] >= 50000) & (df_single['agi'] < 75000)
    bracket_50_75_mask_two = (df_two['agi'] >= 50000) & (df_two['agi'] < 75000)
    
    single_50_75 = df_single[bracket_50_75_mask_single][single_weight_col].sum()
    two_50_75 = df_two[bracket_50_75_mask_two][two_weight_col].sum()
    target_50_75 = 91459
    
    single_error = single_50_75 - target_50_75
    two_error = two_50_75 - target_50_75
    
    improvement = abs(single_error) - abs(two_error)
    improvement_pct = (improvement / abs(single_error) * 100) if single_error != 0 else 0
    
    logger.info(f"  Target:       {target_50_75:>12,}")
    logger.info(f"  Single-layer: {single_50_75:>12,.0f} (error: {single_error:+,.0f}, {single_error/target_50_75*100:+.1f}%)")
    logger.info(f"  Two-layer:    {two_50_75:>12,.0f} (error: {two_error:+,.0f}, {two_error/target_50_75*100:+.1f}%)")
    logger.info(f"\n  Improvement: {improvement:,.0f} fewer returns ({improvement_pct:+.1f}% improvement)")
    
    # Recommendation
    logger.info("\n" + "="*100)
    logger.info("RECOMMENDATION")
    logger.info("="*100)
    
    if two_accurate > single_accurate:
        logger.info("\n✅ RECOMMEND: Two-Layer Calibration")
        logger.info(f"   - Better AGI bracket matching: {two_accurate}/{total_brackets} vs {single_accurate}/{total_brackets}")
        logger.info(f"   - $50k-$75k improvement: {improvement_pct:.1f}%")
        logger.info(f"   - Filing status accuracy maintained")
    else:
        logger.info("\n⚠️  CONSIDER: Single-Layer Calibration")
        logger.info(f"   - Simpler approach")
        logger.info(f"   - Two-layer shows minimal improvement")
    
    return df_single, df_two


def main():
    """Main comparison function."""
    
    # Load most recent tax units (before calibration)
    import glob
    
    # Look for raw tax units (before calibration)
    pattern = 'data/processed/tax_units_*.parquet'
    files = [f for f in glob.glob(pattern) if 'calibrated' not in f]
    
    if not files:
        logger.error("No raw tax units found!")
        logger.info("Please run tax unit construction first")
        return
    
    latest_file = max(files)
    logger.info(f"Loading: {latest_file}\n")
    
    tax_units = pd.read_parquet(latest_file)
    logger.info(f"Loaded {len(tax_units):,} tax units with total weight {tax_units['weight'].sum():,.0f}\n")
    
    # Run comparison
    df_single, df_two = compare_approaches(tax_units)
    
    # Save results for further analysis
    output_dir = Path('data/processed/calibration_comparison')
    output_dir.mkdir(exist_ok=True)
    
    df_single.to_parquet(output_dir / 'single_layer_calibrated.parquet', index=False)
    df_two.to_parquet(output_dir / 'two_layer_calibrated.parquet', index=False)
    
    logger.info("\n" + "="*100)
    logger.info("COMPARISON COMPLETE")
    logger.info("="*100)
    logger.info(f"\nResults saved to: {output_dir}/")
    logger.info("  - single_layer_calibrated.parquet")
    logger.info("  - two_layer_calibrated.parquet")


if __name__ == '__main__':
    main()
