#!/usr/bin/env python3
"""
Debug script to analyze specific households where married people are filing as HoH
instead of joint, to understand why joint filer identification is missing them.
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.pums_loader import PUMSDataLoader
from tax.units.status.mfj import _are_married

def debug_married_hoh_households():
    """Debug households where married people are filing as HoH."""
    
    print("Debugging married Head of Household households...")
    
    # Load tax units and PUMS data
    tax_units = pd.read_parquet("src/data/processed/tax_units_rule_based.parquet")
    pums_loader = PUMSDataLoader()
    person_df, hh_df = pums_loader.load_data()
    
    # Find married HoH filers
    hoh_units = tax_units[tax_units['filing_status'] == 'head_of_household']
    married_hoh_households = []
    
    for idx, tax_unit in hoh_units.iterrows():
        primary_filer_id = tax_unit['primary_filer_id']
        if '_' not in primary_filer_id:
            continue
        hh_id, sporder = primary_filer_id.split('_')
        
        # Get the primary person's data
        primary_person = person_df[(person_df['SERIALNO'] == hh_id) & (person_df['SPORDER'] == int(sporder))]
        if len(primary_person) == 0:
            continue
        primary_person = primary_person.iloc[0]
        
        # Check if married
        if primary_person['MAR'] == 1:
            married_hoh_households.append(hh_id)
    
    print(f"Found {len(married_hoh_households)} households with married HoH filers")
    
    # Analyze the first few households in detail
    for i, hh_id in enumerate(married_hoh_households[:5]):
        print(f"\n{'='*60}")
        print(f"HOUSEHOLD {i+1}: {hh_id}")
        print('='*60)
        
        # Get all people in this household
        hh_persons = person_df[person_df['SERIALNO'] == hh_id].copy()
        print(f"Household has {len(hh_persons)} people:")
        
        for _, person in hh_persons.iterrows():
            print(f"  Person {person['SPORDER']}: MAR={person['MAR']}, RELSHIPP={person['RELSHIPP']}, "
                  f"AGE={person['AGEP']}, SEX={person['SEX']}")
        
        # Check what tax units were created for this household
        hh_tax_units = tax_units[tax_units['hh_id'] == hh_id]
        print(f"\nTax units created: {len(hh_tax_units)}")
        for _, unit in hh_tax_units.iterrows():
            print(f"  {unit['filing_status']}: {unit['primary_filer_id']}, deps={unit['num_dependents']}")
            if unit['secondary_filer_id']:
                print(f"    Secondary: {unit['secondary_filer_id']}")
        
        # Check if there are potential joint filers in this household
        adults = hh_persons[hh_persons['AGEP'] >= 18]
        married_adults = adults[adults['MAR'] == 1]
        
        print(f"\nAdults in household: {len(adults)}")
        print(f"Married adults: {len(married_adults)}")
        
        if len(married_adults) >= 2:
            print("POTENTIAL JOINT FILERS FOUND!")
            
            # Check all possible pairs
            for i, (_, adult1) in enumerate(married_adults.iterrows()):
                for j, (_, adult2) in enumerate(married_adults.iterrows()):
                    if i >= j:
                        continue
                    
                    print(f"  Checking pair: Person {adult1['SPORDER']} & Person {adult2['SPORDER']}")
                    
                    # Check if they would be identified as married couple
                    are_married = _are_married(adult1, adult2, hh_persons)
                    print(f"    _are_married() result: {are_married}")
                    
                    if are_married:
                        print(f"    *** SHOULD BE JOINT FILERS ***")
                        print(f"    Person {adult1['SPORDER']}: MAR={adult1['MAR']}, RELSHIPP={adult1['RELSHIPP']}")
                        print(f"    Person {adult2['SPORDER']}: MAR={adult2['MAR']}, RELSHIPP={adult2['RELSHIPP']}")
        
        # Check spouse relationships specifically
        householders = hh_persons[hh_persons['RELSHIPP'] == 20]
        spouses = hh_persons[hh_persons['RELSHIPP'] == 21]
        
        print(f"\nRelationship analysis:")
        print(f"  Householders (RELSHIPP=20): {len(householders)}")
        print(f"  Spouses (RELSHIPP=21): {len(spouses)}")
        
        if len(householders) > 0 and len(spouses) == 0:
            print("  *** ISSUE: Married householder but no spouse relationship code ***")
            print("  This explains why HoH logic considers them 'unmarried'")
        
        print(f"\nConclusion for {hh_id}:")
        if len(married_adults) >= 2:
            print("  - Household has multiple married adults")
            print("  - Should potentially be creating joint tax units")
            print("  - Joint filer identification may be failing")
        else:
            print("  - Only one married adult in household")
            print("  - May be legitimately single (separated, spouse absent, etc.)")

if __name__ == "__main__":
    debug_married_hoh_households()
