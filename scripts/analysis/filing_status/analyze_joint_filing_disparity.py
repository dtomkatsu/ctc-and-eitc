#!/usr/bin/env python3
"""
Analyze why joint filing rates are so low (8.4% vs 36.0% expected).
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import logging
from src.data.pums_loader import PUMSDataLoader
from src.tax.units.status.mfj import _are_married

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def analyze_joint_filing_disparity():
    print("Loading PUMS data...")
    loader = PUMSDataLoader()
    person_df, hh_df = loader.load_data()
    
    print(f"Loaded {len(person_df)} persons and {len(hh_df)} households")
    
    # Analyze marital status distribution in PUMS data
    print("\n" + "="*60)
    print("MARITAL STATUS ANALYSIS")
    print("="*60)
    
    adults = person_df[person_df['AGEP'] >= 18]
    print(f"Total adults: {len(adults)}")
    
    mar_counts = adults['MAR'].value_counts().sort_index()
    print(f"\nMarital status distribution (MAR codes):")
    mar_labels = {1: 'Married', 2: 'Widowed', 3: 'Divorced', 4: 'Separated', 5: 'Never married'}
    for mar_code, count in mar_counts.items():
        label = mar_labels.get(mar_code, f'Unknown ({mar_code})')
        pct = count / len(adults) * 100
        print(f"  {mar_code} ({label}): {count:,} ({pct:.1f}%)")
    
    # Focus on married adults (MAR=1)
    married_adults = adults[adults['MAR'] == 1]
    print(f"\nMarried adults (MAR=1): {len(married_adults):,}")
    
    # Analyze relationship codes for married adults
    print(f"\nRelationship codes for married adults:")
    rel_counts = married_adults['RELSHIPP'].value_counts().sort_index()
    rel_labels = {20: 'Householder', 21: 'Spouse', 22: 'Child', 23: 'Adopted child', 
                  24: 'Stepchild', 25: 'Grandchild', 26: 'Parent', 27: 'Parent-in-law',
                  28: 'Son/daughter-in-law', 29: 'Other relative', 30: 'Roomer/boarder',
                  31: 'Housemate/roommate', 32: 'Unmarried partner', 33: 'Foster child',
                  34: 'Other nonrelative'}
    
    for rel_code, count in rel_counts.items():
        label = rel_labels.get(rel_code, f'Unknown ({rel_code})')
        pct = count / len(married_adults) * 100
        print(f"  {rel_code} ({label}): {count:,} ({pct:.1f}%)")
    
    # Expected joint filers: married householders and spouses
    married_householders = married_adults[married_adults['RELSHIPP'] == 20]
    married_spouses = married_adults[married_adults['RELSHIPP'] == 21]
    
    print(f"\nMarried householders (RELSHIPP=20): {len(married_householders):,}")
    print(f"Married spouses (RELSHIPP=21): {len(married_spouses):,}")
    
    # Theoretical maximum joint filers (assuming all married couples file jointly)
    max_joint_couples = min(len(married_householders), len(married_spouses))
    print(f"Theoretical max joint filing couples: {max_joint_couples:,}")
    
    # Analyze households with married couples
    print(f"\n" + "="*60)
    print("HOUSEHOLD-LEVEL ANALYSIS")
    print("="*60)
    
    # Find households with both householder and spouse
    households_with_couples = set()
    for hh_id in person_df['SERIALNO'].unique():
        hh_members = person_df[person_df['SERIALNO'] == hh_id]
        has_householder = any((hh_members['RELSHIPP'] == 20) & (hh_members['MAR'] == 1))
        has_spouse = any((hh_members['RELSHIPP'] == 21) & (hh_members['MAR'] == 1))
        
        if has_householder and has_spouse:
            households_with_couples.add(hh_id)
    
    print(f"Households with married couples: {len(households_with_couples):,}")
    
    # Sample some households to check joint filer identification
    sample_households = list(households_with_couples)[:20]
    print(f"\nTesting joint filer identification on {len(sample_households)} sample households:")
    
    joint_identified = 0
    for hh_id in sample_households:
        hh_members = person_df[person_df['SERIALNO'] == hh_id]
        adults = hh_members[hh_members['AGEP'] >= 18]
        
        # Check all adult pairs for marriage
        found_couple = False
        for i, (_, adult1) in enumerate(adults.iterrows()):
            for _, adult2 in adults.iloc[i+1:].iterrows():
                if _are_married(adult1, adult2):
                    found_couple = True
                    print(f"  HH {hh_id}: Found married couple {adult1.name} & {adult2.name}")
                    break
            if found_couple:
                break
        
        if found_couple:
            joint_identified += 1
        else:
            print(f"  HH {hh_id}: No married couple identified")
    
    print(f"\nJoint couples identified: {joint_identified}/{len(sample_households)} ({joint_identified/len(sample_households)*100:.1f}%)")
    
    # Load actual tax units to compare
    print(f"\n" + "="*60)
    print("ACTUAL TAX UNIT ANALYSIS")
    print("="*60)
    
    tax_units = pd.read_parquet('src/data/processed/tax_units_rule_based.parquet')
    
    filing_status_counts = tax_units['filing_status'].value_counts()
    total_units = len(tax_units)
    
    print(f"Actual tax unit filing status distribution:")
    for status, count in filing_status_counts.items():
        pct = count / total_units * 100
        print(f"  {status}: {count:,} ({pct:.1f}%)")
    
    joint_units = filing_status_counts.get('joint', 0)
    
    print(f"\nComparison:")
    print(f"  Theoretical max joint couples: {max_joint_couples:,}")
    print(f"  Households with married couples: {len(households_with_couples):,}")
    print(f"  Actual joint tax units: {joint_units:,}")
    print(f"  Conversion rate: {joint_units/len(households_with_couples)*100:.1f}% of households with couples")
    
    # Expected vs actual
    expected_joint_pct = 36.0  # From SOI benchmark
    actual_joint_pct = joint_units / total_units * 100
    
    print(f"\nBenchmark comparison:")
    print(f"  Expected joint filing rate: {expected_joint_pct:.1f}%")
    print(f"  Actual joint filing rate: {actual_joint_pct:.1f}%")
    print(f"  Gap: {expected_joint_pct - actual_joint_pct:.1f} percentage points")

if __name__ == "__main__":
    analyze_joint_filing_disparity()
