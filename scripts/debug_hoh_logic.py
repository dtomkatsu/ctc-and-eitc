#!/usr/bin/env python3
"""
Debug Head of Household logic to understand why HoH rates are low.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import logging
from src.data.pums_loader import PUMSDataLoader
from src.tax.units.status.hoh import is_head_of_household

# Set up debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s - %(name)s - %(message)s'
)

# Enable debug logging for the hoh module specifically
logging.getLogger('src.tax.units.status.hoh').setLevel(logging.DEBUG)

def main():
    print("Loading PUMS data...")
    loader = PUMSDataLoader()
    person_df, hh_df = loader.load_data()
    
    print(f"Loaded {len(person_df)} persons and {len(hh_df)} households")
    
    # Focus on households with both adults and children
    households_with_children = set(person_df[person_df['AGEP'] < 18]['SERIALNO'].unique())
    households_with_adults = set(person_df[person_df['AGEP'] >= 18]['SERIALNO'].unique())
    households_with_both = households_with_children.intersection(households_with_adults)
    
    print(f"Found {len(households_with_children)} households with children")
    print(f"Found {len(households_with_adults)} households with adults")
    print(f"Found {len(households_with_both)} households with both adults and children")
    
    # Sample a few households for detailed analysis
    sample_households = list(households_with_both)[:5]
    
    for hh_id in sample_households:
        print(f"\n{'='*60}")
        print(f"ANALYZING HOUSEHOLD: {hh_id}")
        print(f"{'='*60}")
        
        hh_members = person_df[person_df['SERIALNO'] == hh_id].copy()
        adults = hh_members[hh_members['AGEP'] >= 18]
        children = hh_members[hh_members['AGEP'] < 18]
        
        print(f"Household composition:")
        print(f"  Adults: {len(adults)}")
        print(f"  Children: {len(children)}")
        
        print(f"\nHousehold members:")
        for _, person in hh_members.iterrows():
            print(f"  {person.name}: age={person['AGEP']}, rel={person.get('RELSHIPP', 'N/A')}, mar={person.get('MAR', 'N/A')}")
        
        # Test each adult for HoH qualification
        print(f"\nTesting adults for HoH qualification:")
        for _, adult in adults.iterrows():
            print(f"\n--- Testing adult {adult.name} ---")
            print(f"Age: {adult['AGEP']}, Relationship: {adult.get('RELSHIPP', 'N/A')}, Marital: {adult.get('MAR', 'N/A')}")
            
            # Test HoH qualification with debug logging
            result = is_head_of_household(adult, person_df)
            print(f"HoH qualification result: {result}")
            
            if result:
                print(f"*** ADULT {adult.name} QUALIFIES AS HEAD OF HOUSEHOLD ***")
        
        print(f"\n")

if __name__ == "__main__":
    main()
