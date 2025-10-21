#!/usr/bin/env python3
"""
Test Revenue-Focused Calibration

Compares revenue calibration (AGI-only) vs hybrid calibration (returns + AGI).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import logging

from src.tax.validation.revenue_calibration import apply_revenue_calibration, validate_revenue_accuracy

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def main():
    """Test revenue calibration approach."""
    
    # Load data
    logger.info("Loading tax units...")
    tax_units = pd.read_parquet('data/processed/tax_units_agi.parquet')
    logger.info(f"Loaded {len(tax_units):,} tax units\n")
    
    # Load enhanced benchmarks (A-2 + A8 hybrid)
    benchmarks = pd.read_csv('data/processed/comprehensive_tax_benchmarks_enhanced.csv')
    logger.info(f"Loaded enhanced benchmarks: {len(benchmarks)} brackets")
    logger.info(f"  Total returns: {benchmarks['returns'].sum():,.0f}")
    logger.info(f"  Total AGI: ${benchmarks['total_agi_millions'].sum():,.1f}M")
    logger.info(f"  High-income detail: {len(benchmarks[benchmarks['agi_min'] >= 400000])} brackets ≥$400k\n")
    
    # Apply revenue calibration (AGI-only)
    calibrated = apply_revenue_calibration(
        tax_units,
        benchmarks,
        weight_col='weight',
        agi_col='agi',
        output_weight_col='weight_revenue_calibrated',
        max_iterations=100,
        tolerance=0.001
    )
    
    # Validate
    logger.info("\n" + "="*80)
    logger.info("DETAILED VALIDATION REPORT")
    logger.info("="*80)
    
    validation_results = validate_revenue_accuracy(
        calibrated,
        benchmarks,
        weight_col='weight_revenue_calibrated'
    )
    
    # Show high-income brackets
    high_income = validation_results[validation_results['agi_min'] >= 150000].sort_values(
        ['filing_status', 'agi_min']
    )
    
    logger.info("\nHigh-Income Bracket Accuracy:")
    logger.info(f"{'Status':<8} {'Bracket':<15} {'Sample':>7} {'Error':>8} {'Status'}")
    logger.info("-"*50)
    
    for _, row in high_income.iterrows():
        status_short = row['filing_status']
        bracket = f"${row['agi_min']//1000}k-${row['agi_max']//1000}k"
        sample = int(row['sample_size'])
        error = row['error_pct']
        
        if error < 1:
            status_icon = "✅"
        elif error < 5:
            status_icon = "✅"
        else:
            status_icon = "⚠️"
        
        logger.info(f"{status_short:<8} {bracket:<15} {sample:>7} {error:>7.2f}% {status_icon}")
    
    # Summary statistics
    logger.info("\n" + "="*80)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*80)
    
    # Separate taxable from nontaxable brackets
    # Nontaxable brackets have agi_min=10000 and agi_max=999999999 with small AGI totals
    taxable_brackets = validation_results[
        ~((validation_results['agi_min'] == 10000) & 
          (validation_results['agi_max'] == 999999999) & 
          (validation_results['target_agi_millions'] < 500))
    ]
    
    nontaxable_brackets = validation_results[
        (validation_results['agi_min'] == 10000) & 
        (validation_results['agi_max'] == 999999999) & 
        (validation_results['target_agi_millions'] < 500)
    ]
    
    logger.info(f"\nTaxable Brackets ({len(taxable_brackets)} brackets):")
    logger.info(f"  Mean error: {taxable_brackets['error_pct'].mean():.2f}%")
    logger.info(f"  Median error: {taxable_brackets['error_pct'].median():.2f}%")
    logger.info(f"  Max error: {taxable_brackets['error_pct'].max():.2f}%")
    logger.info(f"  Brackets < 1% error: {(taxable_brackets['error_pct'] < 1).sum()}/{len(taxable_brackets)}")
    logger.info(f"  Brackets < 5% error: {(taxable_brackets['error_pct'] < 5).sum()}/{len(taxable_brackets)}")
    
    if len(nontaxable_brackets) > 0:
        logger.info(f"\nNontaxable Brackets ({len(nontaxable_brackets)} brackets - excluded from main stats):")
        logger.info(f"  Note: These have very small AGI totals and cause statistical outliers")
        logger.info(f"  Mean error: {nontaxable_brackets['error_pct'].mean():.2f}%")
    
    high_income_errors = high_income['error_pct']
    logger.info(f"\nHigh-Income Brackets (≥$150k):")
    logger.info(f"  Mean error: {high_income_errors.mean():.2f}%")
    logger.info(f"  Max error: {high_income_errors.max():.2f}%")
    logger.info(f"  Brackets < 5% error: {(high_income_errors < 5).sum()}/{len(high_income)}")
    
    # Save results
    output_path = Path('data/processed/tax_units_revenue_calibrated.parquet')
    calibrated.to_parquet(output_path)
    logger.info(f"\n✅ Saved calibrated data to: {output_path}")
    
    # Final recommendation
    logger.info("\n" + "="*80)
    logger.info("RECOMMENDATION")
    logger.info("="*80)
    
    # Use taxable brackets for overall assessment
    overall_error = taxable_brackets['error_pct'].mean()
    
    if overall_error < 2:
        logger.info("\n✅ EXCELLENT - Revenue calibration is highly accurate")
        logger.info("   → Use 'weight_revenue_calibrated' for all revenue calculations")
        logger.info("   → Expected revenue error: < 2%")
        logger.info(f"   → Taxable bracket mean error: {overall_error:.2f}%")
    elif overall_error < 5:
        logger.info("\n✅ GOOD - Revenue calibration is accurate")
        logger.info("   → Use 'weight_revenue_calibrated' for revenue calculations")
        logger.info("   → Expected revenue error: < 5%")
        logger.info(f"   → Taxable bracket mean error: {overall_error:.2f}%")
    else:
        logger.info("\n⚠️  NEEDS IMPROVEMENT - Review calibration")
        logger.info(f"   → Taxable bracket mean error: {overall_error:.2f}%")
    
    logger.info("\n" + "="*80)


if __name__ == '__main__':
    main()
