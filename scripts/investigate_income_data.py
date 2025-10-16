#!/usr/bin/env python3
"""
Investigate actual income data to understand bracket misalignment.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np

def main():
    print("="*80)
    print("INCOME DATA INVESTIGATION")
    print("="*80)
    
    # Load tax units
    tax_units = pd.read_parquet('data/processed/tax_units_final_20251015_105109.parquet')
    
    # Calculate taxable income
    standard_deductions = {
        'single': 12950,
        'married_filing_jointly': 25900,
        'married_filing_separately': 12950,
        'head_of_household': 19400
    }
    
    tax_units['standard_deduction'] = tax_units['filing_status'].map(standard_deductions)
    tax_units['taxable_income'] = (tax_units['income'] - tax_units['standard_deduction']).clip(lower=0)
    
    print(f"\n\nIncome Statistics:")
    print(f"  Total tax units: {len(tax_units):,}")
    print(f"  Weighted total: {tax_units['weight'].sum():,.0f}")
    
    print(f"\n\nTotal Income Distribution:")
    print(f"  Min:      ${tax_units['income'].min():,.0f}")
    print(f"  25th:     ${tax_units['income'].quantile(0.25):,.0f}")
    print(f"  Median:   ${tax_units['income'].median():,.0f}")
    print(f"  75th:     ${tax_units['income'].quantile(0.75):,.0f}")
    print(f"  95th:     ${tax_units['income'].quantile(0.95):,.0f}")
    print(f"  Max:      ${tax_units['income'].max():,.0f}")
    print(f"  Mean:     ${tax_units['income'].mean():,.0f}")
    
    print(f"\n\nTaxable Income Distribution:")
    print(f"  Min:      ${tax_units['taxable_income'].min():,.0f}")
    print(f"  25th:     ${tax_units['taxable_income'].quantile(0.25):,.0f}")
    print(f"  Median:   ${tax_units['taxable_income'].median():,.0f}")
    print(f"  75th:     ${tax_units['taxable_income'].quantile(0.75):,.0f}")
    print(f"  95th:     ${tax_units['taxable_income'].quantile(0.95):,.0f}")
    print(f"  Max:      ${tax_units['taxable_income'].max():,.0f}")
    print(f"  Mean:     ${tax_units['taxable_income'].mean():,.0f}")
    
    # Check for zero/negative income
    zero_income = len(tax_units[tax_units['income'] <= 0])
    print(f"\n\nZero/Negative Income:")
    print(f"  Units with income ≤ 0: {zero_income:,} ({zero_income/len(tax_units)*100:.1f}%)")
    
    zero_taxable = len(tax_units[tax_units['taxable_income'] == 0])
    print(f"  Units with taxable income = 0: {zero_taxable:,} ({zero_taxable/len(tax_units)*100:.1f}%)")
    
    # Check by filing status
    print(f"\n\n" + "="*80)
    print("INCOME BY FILING STATUS")
    print("="*80)
    
    for status in ['single', 'married_filing_jointly', 'head_of_household', 'married_filing_separately']:
        subset = tax_units[tax_units['filing_status'] == status]
        if len(subset) == 0:
            continue
        
        print(f"\n{status.replace('_', ' ').title()}:")
        print(f"  Count: {len(subset):,} (weighted: {subset['weight'].sum():,.0f})")
        print(f"  Total Income - Mean: ${subset['income'].mean():,.0f}, Median: ${subset['income'].median():,.0f}")
        print(f"  Taxable Income - Mean: ${subset['taxable_income'].mean():,.0f}, Median: ${subset['taxable_income'].median():,.0f}")
        print(f"  Zero taxable income: {len(subset[subset['taxable_income'] == 0]):,} ({len(subset[subset['taxable_income'] == 0])/len(subset)*100:.1f}%)")
        
        # Show income distribution
        print(f"  Taxable Income Distribution:")
        for pct in [10, 25, 50, 75, 90]:
            val = subset['taxable_income'].quantile(pct/100)
            print(f"    {pct}th percentile: ${val:,.0f}")
    
    # Check single filers specifically
    print(f"\n\n" + "="*80)
    print("SINGLE FILER DEEP DIVE")
    print("="*80)
    
    single = tax_units[tax_units['filing_status'] == 'single']
    
    print(f"\nTotal single filers: {len(single):,} (weighted: {single['weight'].sum():,.0f})")
    
    # Bracket analysis
    brackets = [
        (0, 2400, "Under $2,400"),
        (2400, 4800, "$2,400 - $4,800"),
        (4800, 9600, "$4,800 - $9,600"),
        (9600, 14400, "$9,600 - $14,400"),
        (14400, 19200, "$14,400 - $19,200"),
        (19200, 24000, "$19,200 - $24,000"),
        (24000, 36000, "$24,000 - $36,000"),
        (36000, 48000, "$36,000 - $48,000"),
        (48000, float('inf'), "$48,000+"),
    ]
    
    print(f"\nTaxable Income Brackets:")
    print(f"{'Bracket':<25} {'Count':>10} {'Weighted':>12} {'% of Single':>12}")
    print("-"*65)
    
    for min_inc, max_inc, label in brackets:
        if max_inc == float('inf'):
            bracket = single[single['taxable_income'] >= min_inc]
        else:
            bracket = single[(single['taxable_income'] >= min_inc) & (single['taxable_income'] < max_inc)]
        
        count = len(bracket)
        weighted = bracket['weight'].sum()
        pct = weighted / single['weight'].sum() * 100
        
        print(f"{label:<25} {count:>10,} {weighted:>12,.0f} {pct:>11.1f}%")
    
    # Check what's in the under $2,400 bracket
    print(f"\n\nUnder $2,400 Bracket Analysis:")
    under_2400 = single[single['taxable_income'] < 2400]
    
    print(f"  Total: {len(under_2400):,} (weighted: {under_2400['weight'].sum():,.0f})")
    print(f"  Mean total income: ${under_2400['income'].mean():,.0f}")
    print(f"  Median total income: ${under_2400['income'].median():,.0f}")
    print(f"  Mean taxable income: ${under_2400['taxable_income'].mean():,.0f}")
    
    # How many have zero taxable income?
    zero_tax = under_2400[under_2400['taxable_income'] == 0]
    print(f"  Zero taxable income: {len(zero_tax):,} (weighted: {zero_tax['weight'].sum():,.0f}, {len(zero_tax)/len(under_2400)*100:.1f}%)")
    
    # What's their total income like?
    print(f"\n  Total Income Distribution (under $2,400 taxable):")
    for pct in [10, 25, 50, 75, 90]:
        val = under_2400['income'].quantile(pct/100)
        print(f"    {pct}th percentile: ${val:,.0f}")
    
    # Check if anyone is above $48K taxable
    print(f"\n\n$48K+ Taxable Income Analysis:")
    over_48k = single[single['taxable_income'] >= 48000]
    print(f"  Count: {len(over_48k):,} (weighted: {over_48k['weight'].sum():,.0f})")
    print(f"  Mean total income: ${over_48k['income'].mean():,.0f}")
    print(f"  Mean taxable income: ${over_48k['taxable_income'].mean():,.0f}")
    
    if len(over_48k) > 0:
        print(f"  Max taxable income: ${over_48k['taxable_income'].max():,.0f}")
    else:
        print(f"  ❌ NO SINGLE FILERS WITH TAXABLE INCOME >= $48,000")
        print(f"\n  This explains -100% in brackets $48K+")
        print(f"  Need to investigate why no high-income single filers!")


if __name__ == '__main__':
    main()
