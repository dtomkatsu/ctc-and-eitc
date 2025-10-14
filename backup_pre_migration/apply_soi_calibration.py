"""
Apply SOI 2022 calibration to Hawaii PUMS tax units.

This script demonstrates how to use the SOI calibrator to align
PUMS-based estimates with Hawaii DOTAX SOI 2022 benchmarks.
"""

import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.pums_loader import PUMSDataLoader
from src.tax.units.constructor import TaxUnitConstructor
from src.tax.calibration.soi_calibration import SOICalibrator


def main():
    """Apply SOI calibration and show results."""
    
    print("=" * 80)
    print("Hawaii PUMS Tax Units - SOI 2022 Calibration")
    print("=" * 80)
    print()
    
    # Load PUMS data
    print("Loading PUMS data...")
    loader = PUMSDataLoader('data/raw/pums')
    person_df, hh_df = loader.load_data()
    
    # Construct tax units
    print("Constructing tax units...")
    constructor = TaxUnitConstructor(person_df, hh_df, progress_bar=False)
    tax_units = constructor.create_rule_based_units(parallel=False)
    
    print(f"Created {len(tax_units):,} tax units")
    print()
    
    # Show before calibration
    print("=" * 80)
    print("BEFORE CALIBRATION")
    print("=" * 80)
    print()
    
    total_before = tax_units['weight'].sum()
    print(f"Total Returns: {total_before:,.0f}")
    print()
    
    print("Filing Status Distribution:")
    status_dist = tax_units.groupby('filing_status')['weight'].sum()
    status_pct = (status_dist / total_before) * 100
    for status in ['single', 'married_filing_jointly', 'married_filing_separately', 'head_of_household']:
        if status in status_pct.index:
            print(f"  {status:<30} {status_pct[status]:>6.1f}%")
    print()
    
    # Apply calibration
    print("=" * 80)
    print("APPLYING SOI CALIBRATION")
    print("=" * 80)
    print()
    
    calibrator = SOICalibrator(
        apply_weight_scaling=True,
        apply_filing_status_calibration=True,
        apply_credit_calibration=True
    )
    
    tax_units_calibrated = calibrator.calibrate(tax_units)
    
    # Show calibration summary
    print()
    print("=" * 80)
    print("CALIBRATION SUMMARY")
    print("=" * 80)
    print()
    
    summary = calibrator.get_calibration_summary(tax_units, tax_units_calibrated)
    print(summary.to_string(index=False))
    print()
    
    # Save calibrated results
    output_path = 'data/processed/tax_units_soi_calibrated.parquet'
    tax_units_calibrated.to_parquet(output_path)
    print(f"Saved calibrated tax units to: {output_path}")
    print()
    
    # Show key improvements
    print("=" * 80)
    print("KEY IMPROVEMENTS")
    print("=" * 80)
    print()
    
    # Filing status accuracy
    soi_targets = SOICalibrator.SOI_FILING_STATUS
    after_dist = tax_units_calibrated.groupby('filing_status')['calibrated_weight'].sum()
    after_total = after_dist.sum()
    after_pct = (after_dist / after_total) * 100
    
    print("Filing Status Accuracy:")
    for status, target in soi_targets.items():
        if status in after_pct.index:
            error = abs(after_pct[status] - target)
            symbol = '✅' if error < 1 else '⚠️' if error < 2 else '❌'
            print(f"  {symbol} {status:<30} Error: {error:.2f}pp")
    
    print()
    print("Total Returns Accuracy:")
    returns_error = abs(after_total - SOICalibrator.SOI_TOTAL_RETURNS)
    returns_error_pct = (returns_error / SOICalibrator.SOI_TOTAL_RETURNS) * 100
    symbol = '✅' if returns_error_pct < 1 else '⚠️' if returns_error_pct < 5 else '❌'
    print(f"  {symbol} Error: {returns_error:,.0f} ({returns_error_pct:.2f}%)")
    
    print()
    print("=" * 80)
    print("CALIBRATION COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Use 'calibrated_weight' column for all weighted calculations")
    print("2. Validate income distributions against SOI avg AGI benchmarks")
    print("3. Validate tax calculations against SOI effective rates")
    print()


if __name__ == '__main__':
    main()
