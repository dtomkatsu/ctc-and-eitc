#!/usr/bin/env python3
"""
Step 5: Final refinement of joint filer identification to close the remaining 5.6% gap.
Focus on identifying additional married couples that should file jointly.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.pums_loader import PUMSDataLoader
from tax.units.status.mfj import _are_married

def analyze_joint_filer_gap():
    """Analyze the remaining joint filer gap and identify potential improvements."""
    
    print("Step 5: Analyzing remaining joint filer gap...")
    
    # Load data
    pums_loader = PUMSDataLoader()
    person_df, hh_df = pums_loader.load_data()
    tax_units = pd.read_parquet("src/data/processed/tax_units_rule_based.parquet")
    
    # Filter out zero-weight households
    hh_weights = hh_df[hh_df['WGTP'] > 0]
    person_df = person_df[person_df['SERIALNO'].isin(hh_weights['SERIALNO'])]
    
    # Current status
    current_joint = len(tax_units[tax_units['filing_status'] == 'joint'])
    current_mfs = len(tax_units[tax_units['filing_status'] == 'married_filing_separate'])
    current_single = len(tax_units[tax_units['filing_status'] == 'single'])
    current_hoh = len(tax_units[tax_units['filing_status'] == 'head_of_household'])
    total_units = len(tax_units)
    
    # Target calculations
    target_joint_rate = 0.36  # 36%
    current_joint_rate = current_joint / total_units
    target_joint_count = int(total_units * target_joint_rate)
    additional_joint_needed = target_joint_count - current_joint
    
    print(f"\nCurrent Status:")
    print(f"  Current joint units: {current_joint:,} ({current_joint_rate*100:.1f}%)")
    print(f"  Target joint units: {target_joint_count:,} ({target_joint_rate*100:.1f}%)")
    print(f"  Additional joint needed: {additional_joint_needed:,}")
    print(f"  Current MFS units: {current_mfs:,}")
    print(f"  Current single units: {current_single:,}")
    print(f"  Current HoH units: {current_hoh:,}")
    
    # Analyze married adults in the population
    married_adults = person_df[person_df['MAR'] == 1]  # Married
    print(f"\nMarried Adults in Population:")
    print(f"  Total married adults: {len(married_adults):,}")
    
    # Group by household to find married couples
    married_households = married_adults.groupby('SERIALNO').size()
    households_with_2_married = (married_households == 2).sum()
    households_with_3plus_married = (married_households >= 3).sum()
    
    print(f"  Households with 2 married adults: {households_with_2_married:,}")
    print(f"  Households with 3+ married adults: {households_with_3plus_married:,}")
    
    # Analyze current filing patterns of married couples
    print(f"\nAnalyzing current filing patterns of married couples...")
    
    # Sample analysis on households with exactly 2 married adults
    sample_households = married_adults[married_adults['SERIALNO'].isin(
        married_households[married_households == 2].index
    )]['SERIALNO'].unique()[:1000]  # Sample 1000 households
    
    joint_couples = 0
    mfs_couples = 0
    mixed_couples = 0  # One joint/MFS, one single/HoH
    single_married = 0  # Married people filing as single
    hoh_married = 0     # Married people filing as HoH
    
    for hh_id in sample_households:
        hh_tax_units = tax_units[tax_units['hh_id'] == hh_id]
        hh_persons = person_df[person_df['SERIALNO'] == hh_id]
        married_in_hh = hh_persons[hh_persons['MAR'] == 1]
        
        if len(married_in_hh) == 2:
            # Check filing statuses
            joint_units = hh_tax_units[hh_tax_units['filing_status'] == 'joint']
            mfs_units = hh_tax_units[hh_tax_units['filing_status'] == 'married_filing_separate']
            single_units = hh_tax_units[hh_tax_units['filing_status'] == 'single']
            hoh_units = hh_tax_units[hh_tax_units['filing_status'] == 'head_of_household']
            
            if len(joint_units) > 0:
                joint_couples += 1
            elif len(mfs_units) == 2:
                mfs_couples += 1
            elif len(mfs_units) == 1 and (len(single_units) > 0 or len(hoh_units) > 0):
                mixed_couples += 1
            else:
                # Check if married people are filing as single or HoH
                married_ids = set(married_in_hh['SERIALNO'] + '_' + married_in_hh['SPORDER'].astype(str))
                
                for _, unit in hh_tax_units.iterrows():
                    if unit['primary_filer_id'] in married_ids:
                        if unit['filing_status'] == 'single':
                            single_married += 1
                        elif unit['filing_status'] == 'head_of_household':
                            hoh_married += 1
    
    print(f"  Sample of {len(sample_households):,} households with 2 married adults:")
    print(f"    Joint filing couples: {joint_couples:,}")
    print(f"    MFS couples: {mfs_couples:,}")
    print(f"    Mixed filing couples: {mixed_couples:,}")
    print(f"    Married filing as single: {single_married:,}")
    print(f"    Married filing as HoH: {hoh_married:,}")
    
    # Estimate potential for converting MFS to joint
    print(f"\n{'='*80}")
    print(f"POTENTIAL IMPROVEMENTS")
    print(f"{'='*80}")
    
    # Extrapolate to full dataset
    sample_rate = len(sample_households) / households_with_2_married
    estimated_mfs_couples = mfs_couples / sample_rate if sample_rate > 0 else 0
    estimated_single_married = single_married / sample_rate if sample_rate > 0 else 0
    estimated_hoh_married = hoh_married / sample_rate if sample_rate > 0 else 0
    
    print(f"1. MARRIED FILING SEPARATELY CONVERSION:")
    print(f"   - Estimated MFS couples: {estimated_mfs_couples:,.0f}")
    print(f"   - Current MFS threshold: 50:1 income ratio")
    print(f"   - Potential: Relax MFS threshold to convert some to joint")
    
    print(f"\n2. MARRIED FILING AS SINGLE:")
    print(f"   - Estimated married filing single: {estimated_single_married:,.0f}")
    print(f"   - Potential: Improve spouse identification logic")
    
    print(f"\n3. MARRIED FILING AS HOH:")
    print(f"   - Estimated married filing HoH: {estimated_hoh_married:,.0f}")
    print(f"   - Potential: Review marital status logic in HoH determination")
    
    # Calculate total potential
    total_potential = estimated_mfs_couples * 0.3 + estimated_single_married + estimated_hoh_married
    
    print(f"\n4. TOTAL POTENTIAL NEW JOINT UNITS:")
    print(f"   - From MFS conversion (30%): {estimated_mfs_couples * 0.3:,.0f}")
    print(f"   - From single conversion: {estimated_single_married:,.0f}")
    print(f"   - From HoH conversion: {estimated_hoh_married:,.0f}")
    print(f"   - Total potential: {total_potential:,.0f}")
    print(f"   - Additional needed: {additional_joint_needed:,}")
    
    if total_potential >= additional_joint_needed:
        print(f"   ✅ SUFFICIENT POTENTIAL to close the gap")
    else:
        print(f"   ⚠️  PARTIAL POTENTIAL ({total_potential/additional_joint_needed*100:.1f}% of needed)")
    
    # Specific recommendations
    print(f"\n{'='*80}")
    print(f"SPECIFIC RECOMMENDATIONS")
    print(f"{'='*80}")
    
    print(f"1. RELAX MFS INCOME THRESHOLD:")
    print(f"   - Current: 50:1 income ratio triggers MFS")
    print(f"   - Suggested: Increase to 75:1 or 100:1")
    print(f"   - Impact: Convert some MFS couples to joint filing")
    
    print(f"2. IMPROVE SPOUSE IDENTIFICATION:")
    print(f"   - Review households where married adults file as single")
    print(f"   - Enhance _are_married() logic for edge cases")
    print(f"   - Consider relationship codes beyond 20+21")
    
    print(f"3. REVIEW HOH MARITAL STATUS LOGIC:")
    print(f"   - Ensure married people don't qualify for HoH inappropriately")
    print(f"   - Check _is_unmarried() function for edge cases")
    
    return {
        'additional_joint_needed': additional_joint_needed,
        'estimated_mfs_couples': estimated_mfs_couples,
        'estimated_single_married': estimated_single_married,
        'estimated_hoh_married': estimated_hoh_married,
        'total_potential': total_potential
    }

if __name__ == "__main__":
    analyze_joint_filer_gap()
