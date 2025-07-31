#!/usr/bin/env python3
"""
Debug dependent assignment issues in HoH tax unit construction.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import logging
from src.data.pums_loader import PUMSDataLoader
from src.tax.units.status.hoh import is_head_of_household
from src.tax.units.dependencies import identify_dependents

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

def debug_dependent_assignment():
    print("Loading PUMS data...")
    loader = PUMSDataLoader()
    person_df, hh_df = loader.load_data()
    
    # Focus on one specific household where we know someone qualifies for HoH
    # From the previous debug output, let's use household 2020HU1063920
    target_hh = '2020HU1063920'
    
    print(f"Analyzing household {target_hh}...")
    
    hh_members = person_df[person_df['SERIALNO'] == target_hh]
    print(f"Household members:")
    for _, member in hh_members.iterrows():
        print(f"  ID: {member.name}, Age: {member.get('AGEP')}, Rel: {member.get('RELSHIPP')}, Mar: {member.get('MAR')}")
    
    # Check which adults qualify for HoH
    adults = hh_members[hh_members['AGEP'] >= 18]
    print(f"\nAdults in household:")
    for _, adult in adults.iterrows():
        qualifies = is_head_of_household(adult, person_df)
        print(f"  ID: {adult.name}, Age: {adult.get('AGEP')}, HoH qualified: {qualifies}")
    
    # Check dependent identification
    print(f"\nDependents identified by identify_dependents:")
    dependents = identify_dependents(hh_members)
    for filer_id, deps in dependents.items():
        print(f"  Filer {filer_id}: {len(deps)} dependents -> {deps}")
    
    # Check what happens when we process this household
    print(f"\nDetailed HoH qualification check for each adult:")
    for _, adult in adults.iterrows():
        print(f"\nAdult {adult.name}:")
        print(f"  Age: {adult.get('AGEP')}")
        print(f"  Marital status: {adult.get('MAR')}")
        print(f"  Relationship: {adult.get('RELSHIPP')}")
        
        # Check HoH qualification step by step
        from src.tax.units.status.hoh import _is_unmarried, _has_qualifying_person, _paid_half_home_cost
        
        is_unmarried = _is_unmarried(adult, person_df)
        has_qualifying = _has_qualifying_person(adult, person_df)
        paid_half = _paid_half_home_cost(adult, person_df)
        
        print(f"  Unmarried: {is_unmarried}")
        print(f"  Has qualifying person: {has_qualifying}")
        print(f"  Paid half home cost: {paid_half}")
        print(f"  Overall HoH qualified: {is_head_of_household(adult, person_df)}")
        
        # Check what dependents this adult could claim
        adult_deps = dependents.get(adult.name, [])
        print(f"  Dependents assigned: {adult_deps}")

if __name__ == "__main__":
    debug_dependent_assignment()
