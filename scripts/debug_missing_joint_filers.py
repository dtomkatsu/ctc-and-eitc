#!/usr/bin/env python3
"""
Debug why 77.8% of married couples are not getting joint filing status.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import logging
from src.data.pums_loader import PUMSDataLoader
from src.tax.units.constructor import TaxUnitConstructor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def debug_missing_joint_filers():
    print("Loading PUMS data...")
    loader = PUMSDataLoader()
    person_df, hh_df = loader.load_data()
    
    # Find households with married couples
    households_with_couples = []
    for hh_id in person_df['SERIALNO'].unique():
        hh_members = person_df[person_df['SERIALNO'] == hh_id]
        adults = hh_members[hh_members['AGEP'] >= 18]
        
        has_married_householder = any((adults['RELSHIPP'] == 20) & (adults['MAR'] == 1))
        has_married_spouse = any((adults['RELSHIPP'] == 21) & (adults['MAR'] == 1))
        
        if has_married_householder and has_married_spouse:
            households_with_couples.append(hh_id)
    
    print(f"Found {len(households_with_couples)} households with married couples")
    
    # Load actual tax units
    tax_units = pd.read_parquet('src/data/processed/tax_units_rule_based.parquet')
    
    # Find which households produced joint tax units
    joint_units = tax_units[tax_units['filing_status'] == 'joint']
    joint_households = set(joint_units['hh_id'].unique())
    
    print(f"Found {len(joint_units)} joint tax units from {len(joint_households)} households")
    
    # Find households with couples that didn't produce joint units
    missing_joint_households = [hh for hh in households_with_couples if hh not in joint_households]
    print(f"Missing joint filers: {len(missing_joint_households)} households ({len(missing_joint_households)/len(households_with_couples)*100:.1f}%)")
    
    # Analyze what happened to these households
    print(f"\nAnalyzing sample of missing joint filer households...")
    
    sample_missing = missing_joint_households[:10]
    constructor = TaxUnitConstructor(person_df, hh_df)
    
    for hh_id in sample_missing:
        print(f"\n--- Household {hh_id} ---")
        hh_members = person_df[person_df['SERIALNO'] == hh_id]
        adults = hh_members[hh_members['AGEP'] >= 18]
        
        print(f"Adults in household:")
        for _, adult in adults.iterrows():
            print(f"  ID: {adult.name}, Age: {adult.get('AGEP')}, MAR: {adult.get('MAR')}, REL: {adult.get('RELSHIPP')}")
        
        # Check what tax units were actually created
        household_tax_units = tax_units[tax_units['hh_id'] == hh_id]
        print(f"Tax units created: {len(household_tax_units)}")
        for _, unit in household_tax_units.iterrows():
            print(f"  Status: {unit['filing_status']}, Primary: {unit['primary_filer_id']}, Deps: {unit['num_dependents']}")
        
        # Test joint filer identification directly
        print(f"Testing joint filer identification:")
        joint_filers, mfs_filers = constructor._identify_joint_filers(adults, hh_members)
        print(f"  Joint filers identified: {len(joint_filers)} pairs")
        print(f"  MFS filers identified: {len(mfs_filers)} pairs")
        
        if joint_filers:
            for adult1_id, adult2_id in joint_filers:
                print(f"    Joint pair: {adult1_id} & {adult2_id}")
        
        if mfs_filers:
            for adult1_id, adult2_id in mfs_filers:
                print(f"    MFS pair: {adult1_id} & {adult2_id}")
    
    # Analyze filing status distribution for households with couples
    print(f"\n" + "="*60)
    print("FILING STATUS ANALYSIS FOR HOUSEHOLDS WITH COUPLES")
    print("="*60)
    
    couple_household_units = tax_units[tax_units['hh_id'].isin(households_with_couples)]
    status_counts = couple_household_units['filing_status'].value_counts()
    total_units_from_couples = len(couple_household_units)
    
    print(f"Tax units from households with married couples: {total_units_from_couples}")
    print(f"Filing status distribution:")
    for status, count in status_counts.items():
        pct = count / total_units_from_couples * 100
        print(f"  {status}: {count:,} ({pct:.1f}%)")
    
    # Expected vs actual
    expected_joint = len(households_with_couples)  # Assuming 1 joint unit per couple household
    actual_joint = status_counts.get('joint', 0)
    
    print(f"\nExpected joint units (1 per couple household): {expected_joint:,}")
    print(f"Actual joint units: {actual_joint:,}")
    print(f"Missing joint units: {expected_joint - actual_joint:,}")
    print(f"Conversion rate: {actual_joint/expected_joint*100:.1f}%")

if __name__ == "__main__":
    debug_missing_joint_filers()
