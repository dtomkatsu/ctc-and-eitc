#!/usr/bin/env python3
"""
Test Full Pipeline with IPF Calibration

This script tests the complete pipeline with the new IPF-based IRS SOI calibration.
"""

import sys
from pathlib import Path
import pandas as pd
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tax.validation.irs_soi_calibration import (
    apply_irs_soi_calibration,
    validate_irs_soi_calibration,
    DOTAX_FILING_STATUS,
    DOTAX_AGI_BRACKETS
)
from src.tax.income import apply_capital_gains_separation, get_capital_gains_summary

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Test the full pipeline with IPF calibration."""
    
    logger.info("="*80)
    logger.info("TESTING FULL PIPELINE WITH IPF CALIBRATION")
    logger.info("="*80)
    
    # Step 1: Load most recent tax units
    logger.info("\nStep 1: Loading tax units...")
    
    import glob
    pattern = 'data/processed/tax_units_*.parquet'
    files = [f for f in glob.glob(pattern) if 'calibrated' not in f and 'ipf' not in f]
    
    if not files:
        logger.error("❌ No tax units found!")
        logger.info("Please run: python scripts/pipeline/01_construct_tax_units.py")
        return
    
    latest_file = max(files)
    logger.info(f"Loading: {latest_file}")
    
    tax_units = pd.read_parquet(latest_file)
    logger.info(f"✅ Loaded {len(tax_units):,} tax units")
    logger.info(f"   Total weight: {tax_units['weight'].sum():,.0f}")
    
    # Step 2: Apply IPF calibration
    logger.info("\nStep 2: Applying IPF calibration...")
    
    tax_units_calibrated = apply_irs_soi_calibration(
        tax_units,
        weight_col='weight'
    )
    
    logger.info("✅ Calibration complete")
    
    # Step 3: Apply capital gains separation
    logger.info("\nStep 3: Applying capital gains separation...")
    
    tax_units_calibrated = apply_capital_gains_separation(
        tax_units_calibrated,
        agi_col='agi',
        income_col='income'
    )
    
    logger.info("✅ Capital gains separation complete")
    
    # Show capital gains summary
    logger.info("\nCapital Gains Summary by AGI Bracket:")
    cap_gains_summary = get_capital_gains_summary(
        tax_units_calibrated,
        weight_col='weight_irs_calibrated'
    )
    
    # Format for display
    cap_gains_summary['returns'] = cap_gains_summary['returns'].astype(int)
    cap_gains_summary['total_income'] = cap_gains_summary['total_income'] / 1e6  # Convert to millions
    cap_gains_summary['capital_gains'] = cap_gains_summary['capital_gains'] / 1e6
    cap_gains_summary = cap_gains_summary.round(1)
    
    logger.info(f"\n{cap_gains_summary.to_string()}")
    
    # Step 4: Validate results
    logger.info("\nStep 4: Validating results...")
    
    validation_results = validate_irs_soi_calibration(
        tax_units_calibrated,
        weight_col='weight_irs_calibrated'
    )
    
    # Display validation results
    logger.info("\n" + "="*80)
    logger.info("VALIDATION RESULTS")
    logger.info("="*80)
    
    logger.info("\nFiling Status Distribution:")
    logger.info(f"{'Status':<30} {'Actual':>12} {'Target':>12} {'Error':>10}")
    logger.info("-"*70)
    
    for status in ['single', 'married_filing_jointly', 'head_of_household', 'married_filing_separately']:
        actual = tax_units_calibrated[tax_units_calibrated['filing_status'] == status]['weight_irs_calibrated'].sum()
        target = DOTAX_FILING_STATUS.get(status, 0)
        error = abs((actual - target) / target * 100) if target > 0 else 0
        
        symbol = '✅' if error < 0.1 else '⚠️' if error < 1 else '❌'
        logger.info(f"{status:<30} {actual:>12,.0f} {target:>12,.0f} {error:>9.2f}% {symbol}")
    
    logger.info("\nAGI Bracket Distribution (selected):")
    logger.info(f"{'Bracket':<20} {'Actual':>12} {'Target':>12} {'Error':>10}")
    logger.info("-"*60)
    
    # Assign AGI brackets
    def assign_agi_bracket(agi):
        if agi < 10000: return (0, 10000)
        elif agi < 20000: return (10000, 20000)
        elif agi < 30000: return (20000, 30000)
        elif agi < 40000: return (30000, 40000)
        elif agi < 50000: return (40000, 50000)
        elif agi < 75000: return (50000, 75000)
        elif agi < 100000: return (75000, 100000)
        elif agi < 150000: return (100000, 150000)
        elif agi < 200000: return (150000, 200000)
        elif agi < 300000: return (200000, 300000)
        elif agi < 400000: return (300000, 400000)
        else: return (400000, float('inf'))
    
    if 'agi' not in tax_units_calibrated.columns:
        # Estimate AGI from total_income
        tax_units_calibrated['agi'] = tax_units_calibrated['total_income'] * 0.99
    
    tax_units_calibrated['agi_bracket'] = tax_units_calibrated['agi'].apply(assign_agi_bracket)
    
    key_brackets = [(50000, 75000), (75000, 100000), (100000, 150000)]
    for bracket_key in key_brackets:
        actual = tax_units_calibrated[tax_units_calibrated['agi_bracket'] == bracket_key]['weight_irs_calibrated'].sum()
        target = DOTAX_AGI_BRACKETS.get(bracket_key, 0)
        error = abs((actual - target) / target * 100) if target > 0 else 0
        
        min_agi, max_agi = bracket_key
        label = f"${min_agi//1000}k-${max_agi//1000}k"
        symbol = '✅' if error < 0.1 else '⚠️' if error < 1 else '❌'
        logger.info(f"{label:<20} {actual:>12,.0f} {target:>12,.0f} {error:>9.2f}% {symbol}")
    
    # Step 5: Save calibrated results
    logger.info("\nStep 5: Saving results with capital gains separation...")
    
    output_file = 'data/processed/tax_units_ipf_calibrated.parquet'
    
    # Clean up before saving
    if 'agi_bracket' in tax_units_calibrated.columns:
        tax_units_calibrated['agi_bracket'] = tax_units_calibrated['agi_bracket'].astype(str)
    
    tax_units_calibrated.to_parquet(output_file, index=False)
    logger.info(f"✅ Saved to: {output_file}")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("PIPELINE TEST COMPLETE")
    logger.info("="*80)
    logger.info("\n✅ All steps completed successfully!")
    logger.info("\nKey columns added:")
    logger.info("  • weight_irs_calibrated - IPF calibrated weights")
    logger.info("  • regular_income - Income from wages, business, etc.")
    logger.info("  • capital_gains_income - Capital gains component")
    logger.info("\nNext steps:")
    logger.info("  1. Use 'weight_irs_calibrated' for all weighted calculations")
    logger.info("  2. Use 'regular_income' for tax calculations (excludes cap gains)")
    logger.info("  3. Apply preferential cap gains tax rates to 'capital_gains_income'")
    logger.info("  4. Run downstream analysis with improved accuracy")
    logger.info(f"\n📁 Output: {output_file}")


if __name__ == '__main__':
    main()
