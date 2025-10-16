#!/usr/bin/env python3
"""
Analyze Weighting Methodology

We're creating 47,319 tax units but only getting 562,376 weighted.
SOI target is 635,117.

This script investigates:
1. How are weights calculated?
2. Are we using household weights or person weights?
3. Should we be using different weights?
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np

def main():
    print("="*80)
    print("WEIGHTING ANALYSIS")
    print("="*80)
    
    # Load data
    persons = pd.read_csv('data/raw/pums/psam_p15.csv')
    households = pd.read_csv('data/raw/pums/psam_h15.csv')
    tax_units_file = sorted(Path('data/processed').glob('tax_units_calibrated_*.parquet'))[-1]
    tax_units = pd.read_parquet(tax_units_file)
    
    print(f"\nUsing: {tax_units_file.name}")
    print(f"\nTax Units:")
    print(f"  Unweighted: {len(tax_units):,}")
    print(f"  Weighted:   {tax_units['weight'].sum():,.0f}")
    print(f"  SOI target: 635,117")
    print(f"  Gap:        {635117 - tax_units['weight'].sum():,.0f} ({(635117 - tax_units['weight'].sum())/635117*100:.1f}%)")
    
    # Check weight columns
    print(f"\n\nWeight Columns in Tax Units:")
    weight_cols = [col for col in tax_units.columns if 'weight' in col.lower()]
    for col in weight_cols:
        print(f"  {col}: mean={tax_units[col].mean():.2f}, sum={tax_units[col].sum():,.0f}")
    
    # Compare to PUMS weights
    print(f"\n\nPUMS Weights:")
    print(f"  Household weights (WGTP): sum={households['WGTP'].sum():,.0f}, mean={households['WGTP'].mean():.2f}")
    print(f"  Person weights (PWGTP): sum={persons['PWGTP'].sum():,.0f}, mean={persons['PWGTP'].mean():.2f}")
    
    adults = persons[persons['AGEP'] >= 18]
    print(f"  Adult person weights: sum={adults['PWGTP'].sum():,.0f}, mean={adults['PWGTP'].mean():.2f}")
    
    # Calculate expected weights
    print(f"\n\nExpected Weights for Tax Units:")
    
    # Method 1: If each tax unit used household weight
    avg_hh_weight = households['WGTP'].mean()
    expected_with_hh = len(tax_units) * avg_hh_weight
    print(f"  Method 1 (HH weight): {len(tax_units):,} units × {avg_hh_weight:.2f} = {expected_with_hh:,.0f}")
    
    # Method 2: If each tax unit used person weight
    avg_person_weight = adults['PWGTP'].mean()
    expected_with_person = len(tax_units) * avg_person_weight
    print(f"  Method 2 (Person weight): {len(tax_units):,} units × {avg_person_weight:.2f} = {expected_with_person:,.0f}")
    
    # Actual
    actual_avg = tax_units['weight'].mean()
    print(f"  Actual: {len(tax_units):,} units × {actual_avg:.2f} = {tax_units['weight'].sum():,.0f}")
    
    # What weight would we need?
    needed_avg = 635117 / len(tax_units)
    print(f"  Needed for SOI: {len(tax_units):,} units × {needed_avg:.2f} = 635,117")
    
    # Scaling factor
    scaling_factor = 635117 / tax_units['weight'].sum()
    print(f"\n\nScaling Factor Needed: {scaling_factor:.4f}")
    print(f"  Current: {tax_units['weight'].sum():,.0f}")
    print(f"  Scaled:  {tax_units['weight'].sum() * scaling_factor:,.0f}")
    
    # Analyze weight distribution
    print(f"\n\nWeight Distribution:")
    print(f"  Min:    {tax_units['weight'].min():.2f}")
    print(f"  25th:   {tax_units['weight'].quantile(0.25):.2f}")
    print(f"  Median: {tax_units['weight'].median():.2f}")
    print(f"  75th:   {tax_units['weight'].quantile(0.75):.2f}")
    print(f"  Max:    {tax_units['weight'].max():.2f}")
    print(f"  Mean:   {tax_units['weight'].mean():.2f}")
    
    # Check if weights vary by filing status
    print(f"\n\nAverage Weight by Filing Status:")
    for status in ['single', 'married_filing_jointly', 'head_of_household', 'married_filing_separately']:
        subset = tax_units[tax_units['filing_status'] == status]
        if len(subset) > 0:
            avg_weight = subset['weight'].mean()
            total_weight = subset['weight'].sum()
            print(f"  {status:30s}: {avg_weight:>8.2f} (total: {total_weight:>12,.0f})")
    
    # Check hh_weight vs weight
    if 'hh_weight' in tax_units.columns:
        print(f"\n\nHousehold Weight vs Tax Unit Weight:")
        print(f"  Avg hh_weight:  {tax_units['hh_weight'].mean():.2f}")
        print(f"  Avg weight:     {tax_units['weight'].mean():.2f}")
        print(f"  Ratio:          {tax_units['weight'].mean() / tax_units['hh_weight'].mean():.4f}")
        
        # Check if they're the same
        same = (tax_units['hh_weight'] == tax_units['weight']).sum()
        print(f"  Same values:    {same:,} / {len(tax_units):,} ({same/len(tax_units)*100:.1f}%)")
    
    # Recommendation
    print(f"\n\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    print(f"\nThe issue is NOT the number of tax units ({len(tax_units):,}).")
    print(f"The issue is the WEIGHTING methodology.")
    print(f"\nCurrent weighted total: {tax_units['weight'].sum():,.0f}")
    print(f"SOI target:             635,117")
    print(f"Gap:                    {635117 - tax_units['weight'].sum():,.0f} ({(635117 - tax_units['weight'].sum())/635117*100:.1f}%)")
    
    print(f"\n\nOPTIONS:")
    print(f"\n1. Apply scaling factor: {scaling_factor:.4f}")
    print(f"   Multiply all weights by {scaling_factor:.4f} to match SOI total")
    print(f"   Pros: Simple, exact match")
    print(f"   Cons: Doesn't address root cause")
    
    print(f"\n2. Review weighting methodology")
    print(f"   Current: Using hybrid weights (combination of HH and person weights)")
    print(f"   Alternative: Use person weights directly")
    print(f"   Pros: May be more accurate")
    print(f"   Cons: Requires code changes")
    
    print(f"\n3. Accept the gap")
    print(f"   88.5% coverage is good for most analyses")
    print(f"   Use scaling for absolute estimates")
    print(f"   Pros: No changes needed")
    print(f"   Cons: Not 100% coverage")


if __name__ == '__main__':
    main()
