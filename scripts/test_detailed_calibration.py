#!/usr/bin/env python3
"""
Test Detailed Tax Liability Calibration

Demonstrates the new calibration approach using DOTAX SOI 2022 Table A-9
with detailed AGI brackets and actual tax liability data.
"""

import sys
from pathlib import Path
import pandas as pd
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tax.validation.detailed_tax_calibration import (
    apply_detailed_tax_calibration,
    validate_detailed_calibration,
    get_benchmark_summary,
    load_detailed_benchmarks
)

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Test the detailed tax liability calibration."""
    logger.info("="*80)
    logger.info("DETAILED TAX LIABILITY CALIBRATION TEST")
    logger.info("="*80)
    
    # Step 1: Show benchmark summary
    logger.info("\nStep 1: Loading detailed benchmarks...")
    logger.info("\nDOTAX SOI 2022 Table A-9 Summary:")
    
    summary = get_benchmark_summary()
    print(summary)
    
    benchmarks = load_detailed_benchmarks()
    logger.info(f"\nTotal brackets: {len(benchmarks)}")
    logger.info(f"Total returns: {benchmarks['returns'].sum():,.0f}")
    logger.info(f"Total AGI: ${benchmarks['total_agi_millions'].sum():,.1f}M")
    logger.info(f"Total tax (after credits): ${benchmarks['tax_after_credits_millions'].sum():,.1f}M")
    logger.info(f"Coverage: AGI under $150,000 (90.3% of all returns)")
    
    # Show sample brackets
    logger.info("\nSample detailed brackets (Joint filers):")
    joint_brackets = benchmarks[benchmarks['filing_status'] == 'joint'][['agi_min', 'agi_max', 'returns', 'avg_tax_after']].head(10)
    print(joint_brackets.to_string(index=False))
    
    # Step 2: Load tax units
    logger.info("\nStep 2: Loading tax units...")
    
    tax_units_path = Path('data/processed/tax_units.parquet')
    if not tax_units_path.exists():
        logger.error(f"Tax units not found at {tax_units_path}")
        logger.info("Please run tax unit construction first.")
        return
    
    tax_units = pd.read_parquet(tax_units_path)
    logger.info(f"Loaded {len(tax_units):,} tax units")
    
    # Check required columns
    if 'agi' not in tax_units.columns and 'income' not in tax_units.columns:
        logger.error("Tax units must have 'agi' or 'income' column")
        return
    
    # Show AGI distribution
    if 'agi' in tax_units.columns:
        under_150k = (tax_units['agi'] < 150000).sum()
        logger.info(f"  Tax units with AGI < $150k: {under_150k:,} ({under_150k/len(tax_units)*100:.1f}%)")
    
    # Step 3: Apply detailed calibration
    logger.info("\nStep 3: Applying detailed tax liability calibration...")
    
    tax_units_calibrated = apply_detailed_tax_calibration(
        tax_units,
        weight_col='weight',
        agi_col='agi' if 'agi' in tax_units.columns else 'income'
    )
    
    # Step 4: Validate calibration
    logger.info("\nStep 4: Validating calibration...")
    
    validation = validate_detailed_calibration(
        tax_units_calibrated,
        weight_col='weight_detailed_calibrated',
        agi_col='agi' if 'agi' in tax_units_calibrated.columns else 'income'
    )
    
    # Step 5: Compare to original weights
    logger.info("\nStep 5: Comparing calibrated vs original weights...")
    
    original_total = tax_units['weight'].sum()
    calibrated_total = tax_units_calibrated['weight_detailed_calibrated'].sum()
    
    logger.info(f"Original total returns: {original_total:,.0f}")
    logger.info(f"Calibrated total returns: {calibrated_total:,.0f}")
    logger.info(f"Difference: {calibrated_total - original_total:+,.0f} ({(calibrated_total/original_total - 1)*100:+.1f}%)")
    
    # Filing status distribution
    logger.info("\nFiling status distribution:")
    logger.info("Status                     Original    Calibrated    Target")
    logger.info("-" * 65)
    
    # Map to DOTAX categories for comparison
    from src.tax.validation.detailed_tax_calibration import map_filing_status_to_dotax
    
    tax_units_calibrated['_dotax_status'] = tax_units_calibrated['filing_status'].apply(map_filing_status_to_dotax)
    
    for status in ['joint', 'single', 'hoh']:
        orig = tax_units[tax_units_calibrated['_dotax_status'] == status]['weight'].sum()
        calib = tax_units_calibrated[tax_units_calibrated['_dotax_status'] == status]['weight_detailed_calibrated'].sum()
        target = benchmarks[benchmarks['filing_status'] == status]['returns'].sum()
        
        logger.info(
            f"{status:<20} {orig:>10,.0f} {calib:>12,.0f} {target:>12,.0f}"
        )
    
    # Step 6: Save calibrated tax units
    logger.info("\nStep 6: Saving calibrated tax units...")
    
    output_path = Path('data/processed/tax_units_detailed_calibrated.parquet')
    tax_units_calibrated.to_parquet(output_path, index=False)
    logger.info(f"✅ Saved to: {output_path}")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("CALIBRATION COMPLETE")
    logger.info("="*80)
    logger.info(f"\nCalibration accuracy:")
    logger.info(f"  Total error: {validation['total_error_pct']:.3f}%")
    logger.info(f"  Max bracket error: {validation['max_error_pct']:.3f}%")
    logger.info(f"  Mean bracket error: {validation['mean_error_pct']:.3f}%")
    logger.info(f"\nKey benefits of detailed calibration:")
    logger.info(f"  • 15 detailed AGI brackets per filing status (vs 12 in Table 12A)")
    logger.info(f"  • Actual tax liability data by bracket")
    logger.info(f"  • More granular income distribution")
    logger.info(f"  • Better alignment with DOTAX SOI data")
    logger.info(f"\nNote: This calibration covers 90.3% of returns (AGI < $150k)")
    logger.info(f"      Higher-income returns retain original PUMS weights")


if __name__ == '__main__':
    main()
