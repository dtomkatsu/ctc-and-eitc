"""
Validate model's income distributions against DOTAX SOI 2022 data.

This script compares the income distribution of our PUMS-based tax units
to the actual income distributions from DOTAX SOI Tables 13A/B/C.

Key validation checks:
1. Income distribution by filing status
2. High-income filer representation
3. Effective tax rate comparison
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.pums_loader import PUMSDataLoader
from src.tax.units.constructor import TaxUnitConstructor
from src.tax.calibration.dotax_soi_parser import DOTAXSOIParser


def validate_mfs_income_distribution(tax_units: pd.DataFrame, parser: DOTAXSOIParser):
    """
    Validate MFS income distribution against Table 13A.
    
    Key question: Why is our MFS average income 40% below SOI?
    """
    print("\n" + "="*80)
    print("MFS INCOME DISTRIBUTION VALIDATION")
    print("="*80)
    
    # Get SOI data
    soi_mfs = parser.parse_tax_rates_by_income('single_mfs')
    
    # Filter to MFS filers only
    mfs_units = tax_units[tax_units['filing_status'] == 'mfs'].copy()
    
    if len(mfs_units) == 0:
        print("⚠️  No MFS filers found in model!")
        return
    
    # Calculate income brackets
    mfs_units['income_bracket'] = pd.cut(
        mfs_units['agi'],
        bins=[0, 2400, 4800, 9600, 14400, 19200, 24000, 36000, 48000, 150000, 175000, 200000, np.inf],
        labels=['<$2.4k', '$2.4-4.8k', '$4.8-9.6k', '$9.6-14.4k', '$14.4-19.2k', 
                '$19.2-24k', '$24-36k', '$36-48k', '$48-150k', '$150-175k', '$175-200k', '>$200k']
    )
    
    # Compare distributions
    model_dist = mfs_units.groupby('income_bracket', observed=True).agg({
        'weight': 'sum',
        'agi': 'mean'
    }).reset_index()
    model_dist['pct'] = model_dist['weight'] / model_dist['weight'].sum()
    
    print("\n### Model vs SOI Income Distribution ###")
    print(f"{'Bracket':<15} {'Model %':>10} {'SOI %':>10} {'Difference':>12}")
    print("-" * 50)
    
    # SOI percentages (from Table 13A)
    soi_pcts = {
        '<$2.4k': 82072 / 351205,
        '$2.4-4.8k': 13773 / 351205,
        '$4.8-9.6k': 24317 / 351205,
        '$9.6-14.4k': 20638 / 351205,
        '$14.4-19.2k': 18647 / 351205,
        '$19.2-24k': 18863 / 351205,
        '$24-36k': 47514 / 351205,
        '$36-48k': 40454 / 351205,
        '$48-150k': 76914 / 351205,
        '$150-175k': 2060 / 351205,
        '$175-200k': 1220 / 351205,
        '>$200k': 4733 / 351205
    }
    
    for bracket in model_dist['income_bracket']:
        model_pct = model_dist[model_dist['income_bracket'] == bracket]['pct'].values[0]
        soi_pct = soi_pcts.get(bracket, 0)
        diff = model_pct - soi_pct
        print(f"{bracket:<15} {model_pct:>9.1%} {soi_pct:>9.1%} {diff:>+11.1%}")
    
    # Key metrics
    print("\n### Key Metrics ###")
    
    # High-income representation
    model_high_income = mfs_units[mfs_units['agi'] > 150000]['weight'].sum()
    model_total = mfs_units['weight'].sum()
    model_pct_high = model_high_income / model_total
    soi_pct_high = (2060 + 1220 + 4733) / 351205
    
    print(f"High-income (>$150k) representation:")
    print(f"  Model: {model_pct_high:.2%}")
    print(f"  SOI:   {soi_pct_high:.2%}")
    print(f"  Gap:   {model_pct_high - soi_pct_high:+.2%}")
    
    # Average income
    model_avg_agi = (mfs_units['agi'] * mfs_units['weight']).sum() / mfs_units['weight'].sum()
    # From previous analysis: SOI MFS avg = $195,040
    soi_avg_agi = 195040
    
    print(f"\nAverage AGI:")
    print(f"  Model: ${model_avg_agi:,.0f}")
    print(f"  SOI:   ${soi_avg_agi:,.0f}")
    print(f"  Gap:   {(model_avg_agi / soi_avg_agi - 1) * 100:+.1f}%")
    
    # Diagnosis
    print("\n### Diagnosis ###")
    if model_pct_high < soi_pct_high * 0.8:
        print("❌ ISSUE: Model is missing high-income MFS filers")
        print("   Recommendation: Relax MFS identification logic for high earners")
    elif model_avg_agi < soi_avg_agi * 0.9:
        print("⚠️  ISSUE: Model has MFS filers but they're lower income")
        print("   Recommendation: Check if high-income married couples are filing joint instead")
    else:
        print("✅ Model's MFS income distribution looks good!")


def validate_hoh_income_distribution(tax_units: pd.DataFrame, parser: DOTAXSOIParser):
    """
    Validate HoH income distribution against Table 13C.
    
    Key question: Why is our HoH average income 34% below SOI?
    """
    print("\n" + "="*80)
    print("HOH INCOME DISTRIBUTION VALIDATION")
    print("="*80)
    
    # Get SOI data
    soi_hoh = parser.parse_tax_rates_by_income('hoh')
    
    # Filter to HoH filers only
    hoh_units = tax_units[tax_units['filing_status'] == 'hoh'].copy()
    
    if len(hoh_units) == 0:
        print("⚠️  No HoH filers found in model!")
        return
    
    # Calculate income brackets
    hoh_units['income_bracket'] = pd.cut(
        hoh_units['agi'],
        bins=[0, 3600, 7200, 14400, 21600, 28800, 36000, 54000, 72000, 225000, 262500, 300000, np.inf],
        labels=['<$3.6k', '$3.6-7.2k', '$7.2-14.4k', '$14.4-21.6k', '$21.6-28.8k',
                '$28.8-36k', '$36-54k', '$54-72k', '$72-225k', '$225-262.5k', '$262.5-300k', '>$300k']
    )
    
    # Compare distributions
    model_dist = hoh_units.groupby('income_bracket', observed=True).agg({
        'weight': 'sum',
        'agi': 'mean'
    }).reset_index()
    model_dist['pct'] = model_dist['weight'] / model_dist['weight'].sum()
    
    print("\n### Model vs SOI Income Distribution ###")
    print(f"{'Bracket':<15} {'Model %':>10} {'SOI %':>10} {'Difference':>12}")
    print("-" * 50)
    
    # SOI percentages (from Table 13C)
    soi_pcts = {
        '<$3.6k': 6671 / 67393,
        '$3.6-7.2k': 2619 / 67393,
        '$7.2-14.4k': 5543 / 67393,
        '$14.4-21.6k': 5904 / 67393,
        '$21.6-28.8k': 7376 / 67393,
        '$28.8-36k': 7732 / 67393,
        '$36-54k': 13878 / 67393,
        '$54-72k': 7835 / 67393,
        '$72-225k': 9117 / 67393,
        '$225-262.5k': 176 / 67393,
        '$262.5-300k': 118 / 67393,
        '>$300k': 424 / 67393
    }
    
    for bracket in model_dist['income_bracket']:
        model_pct = model_dist[model_dist['income_bracket'] == bracket]['pct'].values[0]
        soi_pct = soi_pcts.get(bracket, 0)
        diff = model_pct - soi_pct
        print(f"{bracket:<15} {model_pct:>9.1%} {soi_pct:>9.1%} {diff:>+11.1%}")
    
    # Key metrics
    print("\n### Key Metrics ###")
    
    # High-income representation
    model_high_income = hoh_units[hoh_units['agi'] > 72000]['weight'].sum()
    model_total = hoh_units['weight'].sum()
    model_pct_high = model_high_income / model_total
    soi_pct_high = (9117 + 176 + 118 + 424) / 67393
    
    print(f"High-income (>$72k) representation:")
    print(f"  Model: {model_pct_high:.2%}")
    print(f"  SOI:   {soi_pct_high:.2%}")
    print(f"  Gap:   {model_pct_high - soi_pct_high:+.2%}")
    
    # Average income
    model_avg_agi = (hoh_units['agi'] * hoh_units['weight']).sum() / hoh_units['weight'].sum()
    # From previous analysis: SOI HoH avg = $55,273
    soi_avg_agi = 55273
    
    print(f"\nAverage AGI:")
    print(f"  Model: ${model_avg_agi:,.0f}")
    print(f"  SOI:   ${soi_avg_agi:,.0f}")
    print(f"  Gap:   {(model_avg_agi / soi_avg_agi - 1) * 100:+.1f}%")
    
    # Diagnosis
    print("\n### Diagnosis ###")
    if model_pct_high < soi_pct_high * 0.8:
        print("❌ ISSUE: Model is missing high-income HoH filers")
        print("   Recommendation: Relax '_paid_half_home_cost' test for high earners")
    elif model_avg_agi < soi_avg_agi * 0.9:
        print("⚠️  ISSUE: Model has HoH filers but they're lower income")
        print("   Recommendation: Check if high-income HoH are filing as single instead")
    else:
        print("✅ Model's HoH income distribution looks good!")


def main():
    """Run income distribution validation."""
    print("Loading PUMS data and constructing tax units...")
    
    # Load data
    loader = PUMSDataLoader(year=2022, state='HI')
    person_data = loader.load_data()
    
    # Construct tax units
    constructor = TaxUnitConstructor()
    tax_units = constructor.construct_tax_units(person_data)
    
    print(f"Loaded {len(tax_units):,} tax units")
    print(f"Weighted total: {tax_units['weight'].sum():,.0f}")
    
    # Initialize parser
    parser = DOTAXSOIParser(data_dir='data/raw')
    
    # Run validations
    validate_mfs_income_distribution(tax_units, parser)
    validate_hoh_income_distribution(tax_units, parser)
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Review income distribution gaps above")
    print("2. Adjust MFS/HoH identification logic if needed")
    print("3. Re-run this script to validate improvements")
    print("4. Document any remaining gaps as PUMS data limitations")


if __name__ == '__main__':
    main()
