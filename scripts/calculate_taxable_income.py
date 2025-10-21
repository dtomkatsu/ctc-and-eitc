#!/usr/bin/env python3
"""
Calculate Taxable Income for All Tax Units

Applies deductions and exemptions to AGI to calculate taxable income.
Uses benchmark assignment approach for accuracy.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import logging

from src.tax.deductions import TaxableIncomeCalculator, DeductionPolicy
from src.tax.deductions.calculator import estimate_num_exemptions

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def estimate_exemptions_for_tax_units(tax_units_df, exemption_benchmarks):
    """
    Estimate number of exemptions for each tax unit.
    
    Uses household composition when available, otherwise AGI bracket averages.
    """
    logger.info("Estimating exemptions for tax units...")
    
    exemptions = []
    
    for _, tax_unit in tax_units_df.iterrows():
        num_exemptions = estimate_num_exemptions(tax_unit, exemption_benchmarks)
        exemptions.append(num_exemptions)
    
    return exemptions


def main():
    """Calculate taxable income for all tax units."""
    logger.info("="*80)
    logger.info("CALCULATING TAXABLE INCOME FOR ALL TAX UNITS")
    logger.info("="*80)
    
    # Load tax units with revenue-calibrated weights
    logger.info("\nLoading tax units...")
    tax_units = pd.read_parquet('data/processed/tax_units_revenue_calibrated.parquet')
    logger.info(f"  Loaded {len(tax_units):,} tax units")
    
    # Use revenue-calibrated weights for accuracy
    if 'weight_revenue_calibrated' in tax_units.columns:
        tax_units['weight'] = tax_units['weight_revenue_calibrated']
        logger.info(f"  Using revenue-calibrated weights")
        logger.info(f"  Total weighted returns: {tax_units['weight'].sum():,.0f}")
        logger.info(f"  Total AGI (weighted): ${(tax_units['agi'] * tax_units['weight']).sum()/1e9:.2f}B")
    
    # Load exemption benchmarks for estimation
    exemption_benchmarks = pd.read_csv('data/processed/exemption_benchmarks.csv')
    
    # Estimate exemptions if not present
    if 'num_exemptions' not in tax_units.columns:
        logger.info("\nEstimating exemptions from household composition...")
        tax_units['num_exemptions'] = estimate_exemptions_for_tax_units(
            tax_units, exemption_benchmarks
        )
        logger.info(f"  Average exemptions per return: {tax_units['num_exemptions'].mean():.2f}")
    
    # Add age_65_plus_count if not present (simplified - would need age data from PUMS)
    if 'age_65_plus_count' not in tax_units.columns:
        tax_units['age_65_plus_count'] = 0  # Conservative estimate
    
    # Initialize calculator
    logger.info("\nInitializing taxable income calculator...")
    calculator = TaxableIncomeCalculator.from_files()
    
    # Calculate taxable income
    logger.info("\nCalculating taxable income...")
    results = calculator.calculate_batch(tax_units)
    
    # Summary statistics
    logger.info("\n" + "="*80)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*80)
    
    logger.info(f"\nIncome Components (Weighted):")
    logger.info(f"  Total AGI: ${(results['agi'] * results['weight']).sum()/1e9:.2f}B")
    logger.info(f"  Total Deductions: ${(results['deduction'] * results['weight']).sum()/1e9:.2f}B")
    logger.info(f"  Total Exemptions: ${(results['exemption_amount'] * results['weight']).sum()/1e9:.2f}B")
    logger.info(f"  Total Taxable Income: ${(results['taxable_income'] * results['weight']).sum()/1e9:.2f}B")
    
    logger.info(f"\nAverage per Return:")
    logger.info(f"  Avg AGI: ${results['agi'].mean():,.0f}")
    logger.info(f"  Avg Deduction: ${results['deduction'].mean():,.0f}")
    logger.info(f"  Avg Exemptions: ${results['exemption_amount'].mean():,.0f}")
    logger.info(f"  Avg Taxable Income: ${results['taxable_income'].mean():,.0f}")
    
    logger.info(f"\nDeduction Type Distribution:")
    itemized_count = (results['deduction_type'] == 'itemized').sum()
    standard_count = (results['deduction_type'] == 'standard').sum()
    logger.info(f"  Itemized: {itemized_count:,} ({itemized_count/len(results)*100:.1f}%)")
    logger.info(f"  Standard: {standard_count:,} ({standard_count/len(results)*100:.1f}%)")
    
    # Weighted deduction type distribution
    weighted_itemized = (results[results['deduction_type'] == 'itemized']['weight']).sum()
    weighted_standard = (results[results['deduction_type'] == 'standard']['weight']).sum()
    weighted_total = results['weight'].sum()
    
    logger.info(f"\nDeduction Type Distribution (Weighted):")
    logger.info(f"  Itemized: {weighted_itemized:,.0f} ({weighted_itemized/weighted_total*100:.1f}%)")
    logger.info(f"  Standard: {weighted_standard:,.0f} ({weighted_standard/weighted_total*100:.1f}%)")
    
    # Distribution by filing status
    logger.info(f"\nBy Filing Status:")
    for status in results['filing_status'].unique():
        status_df = results[results['filing_status'] == status]
        logger.info(f"\n  {status.upper()}:")
        logger.info(f"    Count: {len(status_df):,}")
        logger.info(f"    Avg AGI: ${status_df['agi'].mean():,.0f}")
        logger.info(f"    Avg Deduction: ${status_df['deduction'].mean():,.0f}")
        logger.info(f"    Avg Taxable Income: ${status_df['taxable_income'].mean():,.0f}")
        logger.info(f"    Itemize rate: {(status_df['deduction_type'] == 'itemized').sum()/len(status_df)*100:.1f}%")
    
    # Save results
    output_path = Path('data/processed/tax_units_with_taxable_income.parquet')
    results.to_parquet(output_path)
    
    logger.info("\n" + "="*80)
    logger.info(f"✅ Saved results to: {output_path}")
    logger.info("="*80)
    
    logger.info("\nNext Steps:")
    logger.info("  1. Integrate with Hawaii tax calculator to compute tax liability")
    logger.info("  2. Compare total revenue to SOI benchmarks")
    logger.info("  3. Run policy scenarios to model revenue impacts")
    
    return results


if __name__ == '__main__':
    main()
