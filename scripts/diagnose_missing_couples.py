#!/usr/bin/env python3
"""
Diagnose why married couples aren't being captured in tax units.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np

def main():
    print("="*80)
    print("DIAGNOSING MISSING MARRIED COUPLES")
    print("="*80)
    
    # Load PUMS
    persons = pd.read_csv('data/raw/pums/psam_p15.csv')
    households = pd.read_csv('data/raw/pums/psam_h15.csv')
    
    # Load tax units
    tax_units = pd.read_parquet('data/processed/tax_units_calibrated_20251015_111801.parquet')
    
    # Find married couples in PUMS
    married = persons[(persons['AGEP'] >= 18) & (persons['MAR'] == 1)]
    
    # Group by household
    married_by_hh = married.groupby('SERIALNO').size()
    
    print(f"\nMarried Adults in PUMS:")
    print(f"  Total married adults: {len(married):,}")
    print(f"  Weighted: {married['PWGTP'].sum():,.0f}")
    print()
    
    print(f"Households with married adults:")
    print(f"  HH with 1 married adult: {(married_by_hh == 1).sum():,}")
    print(f"  HH with 2 married adults: {(married_by_hh == 2).sum():,}")
    print(f"  HH with 3+ married adults: {(married_by_hh >= 3).sum():,}")
    print()
    
    # Check householder+spouse pairs
    householders = married[married['RELSHIPP'] == 20]
    spouses = married[married['RELSHIPP'] == 21]
    
    print(f"Householder+Spouse Pairs:")
    print(f"  Householders (RELSHIPP=20): {len(householders):,} ({householders['PWGTP'].sum():,.0f} weighted)")
    print(f"  Spouses (RELSHIPP=21): {len(spouses):,} ({spouses['PWGTP'].sum():,.0f} weighted)")
    print()
    
    # Check tax units
    mfj = tax_units[tax_units['filing_status'] == 'married_filing_jointly']
    
    print(f"Tax Units Created:")
    print(f"  MFJ tax units: {len(mfj):,} ({mfj['weight'].sum():,.0f} weighted)")
    print(f"  Expected from PUMS: ~{spouses['PWGTP'].sum():,.0f}")
    print(f"  Missing: {spouses['PWGTP'].sum() - mfj['weight'].sum():,.0f} ({(spouses['PWGTP'].sum() - mfj['weight'].sum())/spouses['PWGTP'].sum()*100:.1f}%)")
    print()
    
    # Sample some households with married couples to see what happened
    print("="*80)
    print("SAMPLE ANALYSIS: Households with Married Couples")
    print("="*80)
    
    # Get households with exactly 2 married adults (householder + spouse)
    hh_with_couples = married_by_hh[married_by_hh == 2].index[:10]
    
    for i, serialno in enumerate(hh_with_couples, 1):
        hh_persons = persons[persons['SERIALNO'] == serialno]
        hh_married = hh_persons[(hh_persons['AGEP'] >= 18) & (hh_persons['MAR'] == 1)]
        
        print(f"\n{i}. Household {serialno}:")
        print(f"   Total persons: {len(hh_persons)}")
        print(f"   Married adults: {len(hh_married)}")
        
        for _, person in hh_married.iterrows():
            print(f"     - SPORDER={person['SPORDER']}, AGEP={person['AGEP']}, "
                  f"RELSHIPP={person['RELSHIPP']}, PINCP=${person.get('PINCP', 0):,.0f}")
        
        # Check if in tax units
        hh_units = tax_units[tax_units['SERIALNO'] == serialno]
        if len(hh_units) == 0:
            print(f"   ❌ NO TAX UNITS FOUND!")
        else:
            print(f"   Tax units created: {len(hh_units)}")
            for _, unit in hh_units.iterrows():
                print(f"     - Status: {unit['filing_status']}, Income: ${unit['income']:,.0f}, "
                      f"Weight: {unit['weight']:,.0f}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nPotential Reasons for Missing Couples:")
    print(f"  1. Tax units not being created for some households")
    print(f"  2. Married couples being split into separate single/HoH filers")
    print(f"  3. Weighting issues")
    print(f"  4. Data filtering or exclusions")


if __name__ == '__main__':
    main()
