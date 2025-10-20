#!/usr/bin/env python3
"""
Test Hybrid Tax Calibration

Demonstrates hybrid calibration using A-9 (detailed < $150k) + A-2 (≥ $150k)
for 100% coverage with maximum granularity.
"""

import sys
from pathlib import Path
import pandas as pd
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tax.validation.hybrid_tax_calibration import (
    apply_hybrid_tax_calibration,
    validate_hybrid_calibration,
    get_hybrid_benchmark_summary,
    load_hybrid_benchmarks
)

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Test hybrid tax calibration."""
    logger.info("="*80)
    logger.info("HYBRID TAX CALIBRATION TEST")
    logger.info("="*80)
    
    # Step 1: Show hybrid benchmark summary
    logger.info("\nStep 1: Loading hybrid benchmarks (A-9 + A-2)...")
    
    benchmarks = load_hybrid_benchmarks()
    
    logger.info("\nHybrid Benchmark Summary:")
    summary = get_hybrid_benchmark_summary()
    print(summary.to_string(index=False))
    
    logger.info(f"\nTotal brackets: {len(benchmarks)}")
    logger.info(f"Total returns: {benchmarks['returns'].sum():,.0f}")
    logger.info(f"Total AGI: ${benchmarks['total_agi_millions'].sum():,.1f}M")
    logger.info(f"Total tax (after credits): ${benchmarks['tax_after_credits_millions'].sum():,.1f}M")
    logger.info(f"Coverage: 100% of all Hawaii returns")
    
    # Show bracket counts by source
    a9_brackets = benchmarks[benchmarks['agi_max'] <= 150000]
    a2_brackets = benchmarks[benchmarks['agi_min'] >= 150000]
    
    logger.info(f"\nBracket breakdown:")
    logger.info(f"  A-9 (< $150k): {len(a9_brackets)} brackets, {a9_brackets['returns'].sum():,.0f} returns (90.3%)")
    logger.info(f"  A-2 (≥ $150k): {len(a2_brackets)} brackets, {a2_brackets['returns'].sum():,.0f} returns (9.7%)")
    
    # Step 2: Load tax units
    logger.info("\nStep 2: Loading tax units...")
    
    # Try multiple possible tax unit file locations
    possible_paths = [
        Path('data/processed/tax_units.parquet'),
        Path('data/processed/tax_units_agi.parquet'),
        Path('data/processed/tax_units_ipf_calibrated.parquet'),
        Path('data/processed/tax_units_original.parquet')
    ]
    
    tax_units_path = None
    for path in possible_paths:
        if path.exists():
            tax_units_path = path
            break
    
    if tax_units_path is None:
        logger.error("No tax units file found. Tried:")
        for path in possible_paths:
            logger.error(f"  - {path}")
        logger.info("Please run tax unit construction first.")
        return
    
    logger.info(f"Using: {tax_units_path}")
    tax_units = pd.read_parquet(tax_units_path)
    logger.info(f"Loaded {len(tax_units):,} tax units")
    
    # Check required columns
    if 'agi' not in tax_units.columns and 'income' not in tax_units.columns:
        logger.error("Tax units must have 'agi' or 'income' column")
        return
    
    # Show AGI distribution
    agi_col = 'agi' if 'agi' in tax_units.columns else 'income'
    under_150k = (tax_units[agi_col] < 150000).sum()
    over_150k = (tax_units[agi_col] >= 150000).sum()
    
    logger.info(f"  Tax units with AGI < $150k: {under_150k:,} ({under_150k/len(tax_units)*100:.1f}%)")
    logger.info(f"  Tax units with AGI ≥ $150k: {over_150k:,} ({over_150k/len(tax_units)*100:.1f}%)")
    
    # Step 3: Apply hybrid calibration
    logger.info("\nStep 3: Applying hybrid tax calibration...")
    
    tax_units_calibrated = apply_hybrid_tax_calibration(
        tax_units,
        weight_col='weight',
        agi_col=agi_col
    )
    
    # Step 4: Validate calibration
    logger.info("\nStep 4: Validating calibration...")
    
    validation = validate_hybrid_calibration(
        tax_units_calibrated,
        weight_col='weight_hybrid_calibrated',
        agi_col=agi_col
    )
    
    # Step 5: Compare to original weights
    logger.info("\nStep 5: Comparing calibrated vs original weights...")
    
    original_total = tax_units['weight'].sum()
    calibrated_total = tax_units_calibrated['weight_hybrid_calibrated'].sum()
    
    logger.info(f"Original total returns: {original_total:,.0f}")
    logger.info(f"Calibrated total returns: {calibrated_total:,.0f}")
    logger.info(f"Difference: {calibrated_total - original_total:+,.0f} ({(calibrated_total/original_total - 1)*100:+.1f}%)")
    
    # Filing status distribution
    logger.info("\nFiling status distribution:")
    logger.info("Status                     Original    Calibrated    Target")
    logger.info("-" * 65)
    
    from src.tax.validation.hybrid_tax_calibration import map_filing_status_to_dotax
    
    tax_units_calibrated['_dotax_status'] = tax_units_calibrated['filing_status'].apply(map_filing_status_to_dotax)
    
    for status in ['joint', 'single', 'hoh']:
        orig = tax_units[tax_units_calibrated['_dotax_status'] == status]['weight'].sum()
        calib = tax_units_calibrated[tax_units_calibrated['_dotax_status'] == status]['weight_hybrid_calibrated'].sum()
        target = benchmarks[benchmarks['filing_status'] == status]['returns'].sum()
        
        logger.info(
            f"{status:<20} {orig:>10,.0f} {calib:>12,.0f} {target:>12,.0f}"
        )
    
    # AGI distribution by source
    logger.info("\nAGI distribution by calibration source:")
    logger.info("Source                     Returns      Total AGI      Avg AGI")
    logger.info("-" * 65)
    
    under_150k_mask = tax_units_calibrated[agi_col] < 150000
    over_150k_mask = tax_units_calibrated[agi_col] >= 150000
    
    under_returns = tax_units_calibrated[under_150k_mask]['weight_hybrid_calibrated'].sum()
    under_agi = (tax_units_calibrated[under_150k_mask][agi_col] * 
                 tax_units_calibrated[under_150k_mask]['weight_hybrid_calibrated']).sum()
    
    over_returns = tax_units_calibrated[over_150k_mask]['weight_hybrid_calibrated'].sum()
    over_agi = (tax_units_calibrated[over_150k_mask][agi_col] * 
                tax_units_calibrated[over_150k_mask]['weight_hybrid_calibrated']).sum()
    
    logger.info(f"A-9 (< $150k)      {under_returns:>10,.0f} ${under_agi/1e9:>10,.2f}B ${under_agi/under_returns:>10,.0f}")
    logger.info(f"A-2 (≥ $150k)      {over_returns:>10,.0f} ${over_agi/1e9:>10,.2f}B ${over_agi/over_returns:>10,.0f}")
    logger.info(f"Total              {calibrated_total:>10,.0f} ${(under_agi+over_agi)/1e9:>10,.2f}B ${(under_agi+over_agi)/calibrated_total:>10,.0f}")
    
    # Step 6: Save calibrated tax units
    logger.info("\nStep 6: Saving calibrated tax units...")
    
    output_path = Path('data/processed/tax_units_hybrid_calibrated.parquet')
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
    logger.info(f"  A-9 mean error: {validation['a9_mean_error']:.3f}%")
    logger.info(f"  A-2 mean error: {validation['a2_mean_error']:.3f}%")
    logger.info(f"\nHybrid calibration features:")
    logger.info(f"  • 19 brackets per filing status (15 from A-9, 4 from A-2)")
    logger.info(f"  • 100% coverage of all returns")
    logger.info(f"  • Maximum granularity for low-income analysis")
    logger.info(f"  • Complete high-income coverage")
    logger.info(f"\nUse 'weight_hybrid_calibrated' for all weighted calculations")


if __name__ == '__main__':
    main()
