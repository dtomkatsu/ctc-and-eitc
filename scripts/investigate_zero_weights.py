#!/usr/bin/env python3
"""
Investigate zero-weight households in PUMS data to understand
how they should be handled according to Census documentation.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.pums_loader import PUMSDataLoader

def investigate_zero_weights():
    """Investigate zero-weight households and their characteristics."""
    
    print("Investigating zero-weight households in PUMS data...")
    
    # Load data
    pums_loader = PUMSDataLoader()
    person_df, hh_df = pums_loader.load_data()
    
    # Analyze household weights
    print(f"\nHousehold Weight Analysis:")
    print(f"Total households: {len(hh_df):,}")
    
    zero_weight_hh = hh_df[hh_df['WGTP'] == 0]
    print(f"Zero-weight households: {len(zero_weight_hh):,} ({len(zero_weight_hh)/len(hh_df)*100:.2f}%)")
    
    if len(zero_weight_hh) > 0:
        print(f"\nCharacteristics of zero-weight households:")
        
        # Check if these are group quarters or special cases
        if 'HHT' in zero_weight_hh.columns:
            hht_counts = zero_weight_hh['HHT'].value_counts()
            print(f"Household types (HHT) for zero-weight households:")
            for hht, count in hht_counts.items():
                print(f"  HHT={hht}: {count:,} households")
        
        # Check number of people in zero-weight households
        if 'NP' in zero_weight_hh.columns:
            np_stats = zero_weight_hh['NP'].describe()
            print(f"\nNumber of people (NP) in zero-weight households:")
            print(f"  Mean: {np_stats['mean']:.2f}")
            print(f"  Min: {int(np_stats['min'])}")
            print(f"  Max: {int(np_stats['max'])}")
        
        # Sample some zero-weight households
        print(f"\nSample zero-weight households:")
        sample_cols = ['SERIALNO', 'WGTP', 'NP', 'HHT', 'HINCP']
        available_cols = [col for col in sample_cols if col in zero_weight_hh.columns]
        print(zero_weight_hh[available_cols].head(10).to_string(index=False))
    
    # Check person weights for people in zero-weight households
    if len(zero_weight_hh) > 0:
        zero_weight_serialnos = zero_weight_hh['SERIALNO'].tolist()
        zero_weight_persons = person_df[person_df['SERIALNO'].isin(zero_weight_serialnos)]
        
        print(f"\nPeople in zero-weight households:")
        print(f"Total people: {len(zero_weight_persons):,}")
        
        if len(zero_weight_persons) > 0:
            pwgtp_stats = zero_weight_persons['PWGTP'].describe()
            print(f"Person weights (PWGTP) for people in zero-weight households:")
            print(f"  Mean: {pwgtp_stats['mean']:.2f}")
            print(f"  Min: {int(pwgtp_stats['min'])}")
            print(f"  Max: {int(pwgtp_stats['max'])}")
            print(f"  Zero person weights: {(zero_weight_persons['PWGTP'] == 0).sum():,}")
    
    # Check tax units from zero-weight households
    tax_units = pd.read_parquet("src/data/processed/tax_units_rule_based.parquet")
    
    if len(zero_weight_hh) > 0:
        zero_weight_tax_units = tax_units[tax_units['hh_id'].isin(zero_weight_serialnos)]
        print(f"\nTax units from zero-weight households:")
        print(f"Total tax units: {len(zero_weight_tax_units):,}")
        
        if len(zero_weight_tax_units) > 0:
            filing_status_counts = zero_weight_tax_units['filing_status'].value_counts()
            print(f"Filing status distribution:")
            for status, count in filing_status_counts.items():
                print(f"  {status}: {count:,}")
    
    # Recommendations based on Census documentation
    print(f"\n{'='*80}")
    print(f"RECOMMENDATIONS FOR HANDLING ZERO-WEIGHT HOUSEHOLDS")
    print(f"{'='*80}")
    
    if len(zero_weight_hh) == 0:
        print("✅ No zero-weight households found - no special handling needed")
    else:
        print("According to Census PUMS documentation:")
        print("1. Zero-weight households typically represent:")
        print("   - Group quarters (dormitories, nursing homes, etc.)")
        print("   - Households that don't meet certain criteria for weighting")
        print("   - Special cases that shouldn't be included in population estimates")
        print()
        print("2. Recommended handling:")
        print("   - EXCLUDE zero-weight households from weighted estimates")
        print("   - Do NOT fill with mean weights (this inflates population)")
        print("   - Keep them in unweighted analysis for completeness")
        print()
        print(f"3. Impact on our analysis:")
        print(f"   - Removing {len(zero_weight_hh):,} zero-weight households")
        print(f"   - This will reduce our total weighted estimate")
        print(f"   - Should improve alignment with actual tax return counts")

if __name__ == "__main__":
    investigate_zero_weights()
