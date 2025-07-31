#!/usr/bin/env python3
"""
Debug the duplication bug by processing a single problematic household.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import logging
from src.data.pums_loader import PUMSDataLoader
from src.tax.units.constructor import TaxUnitConstructor

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

def debug_duplication_bug():
    print("Loading PUMS data...")
    loader = PUMSDataLoader()
    person_df, hh_df = loader.load_data()
    
    # Focus on the problematic household that created 66 duplicate units
    problem_hh = '2019HU0030966'
    
    print(f"Analyzing problematic household {problem_hh}...")
    
    hh_members = person_df[person_df['SERIALNO'] == problem_hh]
    hh_data = hh_df[hh_df['SERIALNO'] == problem_hh].iloc[0]
    
    print(f"Household members:")
    for _, member in hh_members.iterrows():
        print(f"  ID: {member.name}, Age: {member.get('AGEP')}, MAR: {member.get('MAR')}, REL: {member.get('RELSHIPP')}")
    
    # Create constructor and process this single household
    constructor = TaxUnitConstructor(person_df, hh_df)
    
    print(f"\nProcessing household directly...")
    
    # Call _process_household directly to see what happens
    tax_units = constructor._process_household(hh_members)
    
    print(f"\nResults:")
    print(f"Number of tax units created: {len(tax_units)}")
    
    # Show first few tax units to see if they're duplicates
    for i, unit in enumerate(tax_units[:10]):  # Show first 10
        print(f"  Unit {i+1}: Status={unit['filing_status']}, Primary={unit['primary_filer_id']}, Deps={unit['num_dependents']}")
    
    if len(tax_units) > 10:
        print(f"  ... and {len(tax_units) - 10} more units")
    
    # Check if all units are identical
    if len(tax_units) > 1:
        first_unit = tax_units[0]
        all_identical = True
        for unit in tax_units[1:]:
            if (unit['filing_status'] != first_unit['filing_status'] or 
                unit['primary_filer_id'] != first_unit['primary_filer_id'] or
                unit['num_dependents'] != first_unit['num_dependents']):
                all_identical = False
                break
        
        print(f"\nAll units identical: {all_identical}")
    
    # Test the MFS identification logic directly
    print(f"\nTesting MFS identification logic:")
    adults = hh_members[hh_members['AGEP'] >= 18]
    joint_filers, mfs_filers = constructor._identify_joint_filers(adults, hh_members)
    
    print(f"Joint filers identified: {len(joint_filers)} pairs")
    print(f"MFS filers identified: {len(mfs_filers)} pairs")
    
    for adult1_id, adult2_id in mfs_filers:
        print(f"  MFS pair: {adult1_id} & {adult2_id}")
        
        # Test the _should_file_separately method directly
        adult1 = adults.loc[adult1_id]
        adult2 = adults.loc[adult2_id]
        
        should_file_sep = constructor._should_file_separately(adult1, adult2, hh_members)
        print(f"    Should file separately: {should_file_sep}")

if __name__ == "__main__":
    debug_duplication_bug()
