#!/usr/bin/env python3
"""
Direct comparison to SOI Tables 13A, 13B, 13C using exact bracket definitions.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np

def load_tax_units():
    """Load final tax units with taxable income."""
    # Load most recent calibrated file
    import glob
    files = glob.glob('data/processed/tax_units_calibrated_*.parquet')
    if files:
        latest_file = max(files)
        print(f"Loading: {latest_file}")
        tax_units = pd.read_parquet(latest_file)
    else:
        tax_units = pd.read_parquet('data/processed/tax_units_final_20251015_102701.parquet')
    
    # Calculate taxable income
    standard_deductions = {
        'single': 12950,
        'married_filing_jointly': 25900,
        'married_filing_separately': 12950,
        'head_of_household': 19400
    }
    
    tax_units['standard_deduction'] = tax_units['filing_status'].map(standard_deductions)
    tax_units['taxable_income'] = (tax_units['income'] - tax_units['standard_deduction']).clip(lower=0)
    
    return tax_units


def compare_single_and_mfs(tax_units):
    """Compare to Table 13A: Single and MFS."""
    print("\n" + "="*80)
    print("TABLE 13A: SINGLE AND MARRIED FILING SEPARATELY")
    print("="*80)
    
    # SOI brackets from Table 13A
    soi_brackets = [
        (0, 2400, 82072),
        (2400, 4800, 13773),
        (4800, 9600, 24317),
        (9600, 14400, 20638),
        (14400, 19200, 18647),
        (19200, 24000, 18863),
        (24000, 36000, 47514),
        (36000, 48000, 40454),
        (48000, 150000, 76914),
        (150000, 175000, 2060),
        (175000, 200000, 1220),
        (200000, float('inf'), 4733),
    ]
    
    # Get single and MFS filers
    data = tax_units[tax_units['filing_status'].isin(['single', 'married_filing_separately'])].copy()
    
    print(f"\nTotal Returns:")
    print(f"  SOI:   351,205")
    print(f"  Model: {data['weight'].sum():,.0f}")
    print(f"  Diff:  {data['weight'].sum() - 351205:+,.0f} ({(data['weight'].sum() - 351205)/351205*100:+.1f}%)")
    
    print(f"\n{'Taxable Income Range':<30} {'SOI':>12} {'Model':>12} {'Diff':>12} {'% Diff':>10}")
    print("-"*80)
    
    total_soi = 0
    total_model = 0
    within_10pct = 0
    
    for min_inc, max_inc, soi_count in soi_brackets:
        if max_inc == float('inf'):
            bracket_data = data[data['taxable_income'] >= min_inc]
            label = f"Over ${min_inc:,}"
        else:
            bracket_data = data[(data['taxable_income'] >= min_inc) & (data['taxable_income'] < max_inc)]
            label = f"${min_inc:,} - ${max_inc:,}"
        
        model_count = bracket_data['weight'].sum()
        diff = model_count - soi_count
        pct_diff = (diff / soi_count * 100) if soi_count > 0 else 0
        
        status = "✅" if abs(pct_diff) <= 10 else "⚠️" if abs(pct_diff) <= 20 else "❌"
        
        print(f"{label:<30} {soi_count:>12,} {model_count:>12,.0f} {diff:>+12,.0f} {pct_diff:>+9.1f}% {status}")
        
        total_soi += soi_count
        total_model += model_count
        if abs(pct_diff) <= 10:
            within_10pct += 1
    
    print("-"*80)
    print(f"{'TOTAL':<30} {total_soi:>12,} {total_model:>12,.0f} {total_model - total_soi:>+12,.0f} {(total_model - total_soi)/total_soi*100:>+9.1f}%")
    print(f"\nBrackets within ±10%: {within_10pct}/{len(soi_brackets)} ({within_10pct/len(soi_brackets)*100:.1f}%)")


def compare_mfj(tax_units):
    """Compare to Table 13B: Married Filing Jointly."""
    print("\n" + "="*80)
    print("TABLE 13B: MARRIED FILING JOINTLY")
    print("="*80)
    
    # SOI brackets from Table 13B
    soi_brackets = [
        (0, 4800, 40236),
        (4800, 9600, 5632),
        (9600, 19200, 10798),
        (19200, 28800, 11317),
        (28800, 38400, 11443),
        (38400, 48000, 10914),
        (48000, 72000, 27330),
        (72000, 96000, 26226),
        (96000, 300000, 62358),
        (300000, 350000, 2401),
        (350000, 400000, 1588),
        (400000, float('inf'), 6115),
    ]
    
    # Get MFJ filers
    data = tax_units[tax_units['filing_status'] == 'married_filing_jointly'].copy()
    
    print(f"\nTotal Returns:")
    print(f"  SOI:   216,358")
    print(f"  Model: {data['weight'].sum():,.0f}")
    print(f"  Diff:  {data['weight'].sum() - 216358:+,.0f} ({(data['weight'].sum() - 216358)/216358*100:+.1f}%)")
    
    print(f"\n{'Taxable Income Range':<30} {'SOI':>12} {'Model':>12} {'Diff':>12} {'% Diff':>10}")
    print("-"*80)
    
    total_soi = 0
    total_model = 0
    within_10pct = 0
    
    for min_inc, max_inc, soi_count in soi_brackets:
        if max_inc == float('inf'):
            bracket_data = data[data['taxable_income'] >= min_inc]
            label = f"Over ${min_inc:,}"
        else:
            bracket_data = data[(data['taxable_income'] >= min_inc) & (data['taxable_income'] < max_inc)]
            label = f"${min_inc:,} - ${max_inc:,}"
        
        model_count = bracket_data['weight'].sum()
        diff = model_count - soi_count
        pct_diff = (diff / soi_count * 100) if soi_count > 0 else 0
        
        status = "✅" if abs(pct_diff) <= 10 else "⚠️" if abs(pct_diff) <= 20 else "❌"
        
        print(f"{label:<30} {soi_count:>12,} {model_count:>12,.0f} {diff:>+12,.0f} {pct_diff:>+9.1f}% {status}")
        
        total_soi += soi_count
        total_model += model_count
        if abs(pct_diff) <= 10:
            within_10pct += 1
    
    print("-"*80)
    print(f"{'TOTAL':<30} {total_soi:>12,} {total_model:>12,.0f} {total_model - total_soi:>+12,.0f} {(total_model - total_soi)/total_soi*100:>+9.1f}%")
    print(f"\nBrackets within ±10%: {within_10pct}/{len(soi_brackets)} ({within_10pct/len(soi_brackets)*100:.1f}%)")


def compare_hoh(tax_units):
    """Compare to Table 13C: Head of Household."""
    print("\n" + "="*80)
    print("TABLE 13C: HEAD OF HOUSEHOLD")
    print("="*80)
    
    # SOI brackets from Table 13C
    soi_brackets = [
        (0, 3600, 6671),
        (3600, 7200, 2619),
        (7200, 14400, 5543),
        (14400, 21600, 5904),
        (21600, 28800, 7376),
        (28800, 36000, 7732),
        (36000, 54000, 13878),
        (54000, 72000, 7835),
        (72000, 225000, 9117),
        (225000, 262500, 176),
        (262500, 300000, 118),
        (300000, float('inf'), 424),
    ]
    
    # Get HoH filers
    data = tax_units[tax_units['filing_status'] == 'head_of_household'].copy()
    
    print(f"\nTotal Returns:")
    print(f"  SOI:   67,393")
    print(f"  Model: {data['weight'].sum():,.0f}")
    print(f"  Diff:  {data['weight'].sum() - 67393:+,.0f} ({(data['weight'].sum() - 67393)/67393*100:+.1f}%)")
    
    print(f"\n{'Taxable Income Range':<30} {'SOI':>12} {'Model':>12} {'Diff':>12} {'% Diff':>10}")
    print("-"*80)
    
    total_soi = 0
    total_model = 0
    within_10pct = 0
    
    for min_inc, max_inc, soi_count in soi_brackets:
        if max_inc == float('inf'):
            bracket_data = data[data['taxable_income'] >= min_inc]
            label = f"Over ${min_inc:,}"
        else:
            bracket_data = data[(data['taxable_income'] >= min_inc) & (data['taxable_income'] < max_inc)]
            label = f"${min_inc:,} - ${max_inc:,}"
        
        model_count = bracket_data['weight'].sum()
        diff = model_count - soi_count
        pct_diff = (diff / soi_count * 100) if soi_count > 0 else 0
        
        status = "✅" if abs(pct_diff) <= 10 else "⚠️" if abs(pct_diff) <= 20 else "❌"
        
        print(f"{label:<30} {soi_count:>12,} {model_count:>12,.0f} {diff:>+12,.0f} {pct_diff:>+9.1f}% {status}")
        
        total_soi += soi_count
        total_model += model_count
        if abs(pct_diff) <= 10:
            within_10pct += 1
    
    print("-"*80)
    print(f"{'TOTAL':<30} {total_soi:>12,} {total_model:>12,.0f} {total_model - total_soi:>+12,.0f} {(total_model - total_soi)/total_soi*100:>+9.1f}%")
    print(f"\nBrackets within ±10%: {within_10pct}/{len(soi_brackets)} ({within_10pct/len(soi_brackets)*100:.1f}%)")


def main():
    print("="*80)
    print("SOI TABLES 13A, 13B, 13C COMPARISON")
    print("="*80)
    print("\nLoading tax units...")
    
    tax_units = load_tax_units()
    
    print(f"  Total tax units: {len(tax_units):,}")
    print(f"  Weighted total: {tax_units['weight'].sum():,.0f}")
    print(f"  Average taxable income: ${tax_units['taxable_income'].mean():,.0f}")
    
    # Run comparisons
    compare_single_and_mfs(tax_units)
    compare_mfj(tax_units)
    compare_hoh(tax_units)
    
    # Overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    
    total_soi = 351205 + 216358 + 67393
    total_model = tax_units['weight'].sum()
    
    print(f"\nTotal All Filing Statuses:")
    print(f"  SOI Total:   {total_soi:,}")
    print(f"  Model Total: {total_model:,.0f}")
    print(f"  Difference:  {total_model - total_soi:+,.0f} ({(total_model - total_soi)/total_soi*100:+.1f}%)")
    
    print(f"\nNote: MFS is included in Table 13A (Single and MFS combined)")
    print(f"      Our model has {tax_units[tax_units['filing_status']=='married_filing_separately']['weight'].sum():,.0f} MFS filers")


if __name__ == '__main__':
    main()
