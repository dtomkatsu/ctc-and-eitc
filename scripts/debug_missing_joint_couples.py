#!/usr/bin/env python3
"""
Debug script to find households with married couples that should be filing jointly
but are not being identified by the joint filer identification logic.
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.pums_loader import PUMSDataLoader
from tax.units.status.mfj import _are_married

def debug_missing_joint_couples():
    """Find married couples that should be joint filers but aren't."""
    
    print("Debugging missing joint filer couples...")
    
    # Load data
    tax_units = pd.read_parquet("src/data/processed/tax_units_rule_based.parquet")
    pums_loader = PUMSDataLoader()
    person_df, hh_df = pums_loader.load_data()
    
    print(f"Total tax units: {len(tax_units)}")
    print(f"Joint tax units: {len(tax_units[tax_units['filing_status'] == 'joint'])}")
    
    # Find households with married couples that should be joint filers
    potential_joint_households = []
    
    print("\nAnalyzing households for potential joint filers...")
    
    for hh_id in hh_df['SERIALNO'].unique()[:2000]:  # Sample first 2000 households
        hh_persons = person_df[person_df['SERIALNO'] == hh_id]
        
        # Find adults (18+)
        adults = hh_persons[hh_persons['AGEP'] >= 18]
        
        # Look for married couples (householder + spouse)
        householder = adults[adults['RELSHIPP'] == 20]  # Householder
        spouse = adults[adults['RELSHIPP'] == 21]       # Spouse
        
        if len(householder) == 1 and len(spouse) == 1:
            householder_person = householder.iloc[0]
            spouse_person = spouse.iloc[0]
            
            # Check if both are married
            if householder_person['MAR'] == 1 and spouse_person['MAR'] == 1:
                # This should be a joint filing couple
                potential_joint_households.append({
                    'hh_id': hh_id,
                    'householder_id': f"{hh_id}_{householder_person['SPORDER']}",
                    'spouse_id': f"{hh_id}_{spouse_person['SPORDER']}",
                    'householder_age': householder_person['AGEP'],
                    'spouse_age': spouse_person['AGEP'],
                    'householder_income': householder_person.get('PINCP', 0) or 0,
                    'spouse_income': spouse_person.get('PINCP', 0) or 0
                })
    
    print(f"Found {len(potential_joint_households)} households with married couples (householder + spouse)")
    
    # Check how many of these are actually filing jointly
    joint_count = 0
    missing_joint_examples = []
    
    for couple in potential_joint_households:
        hh_id = couple['hh_id']
        hh_tax_units = tax_units[tax_units['hh_id'] == hh_id]
        
        # Check if there's a joint tax unit for this household
        joint_units = hh_tax_units[hh_tax_units['filing_status'] == 'joint']
        
        if len(joint_units) > 0:
            joint_count += 1
        else:
            # This couple should be joint but isn't
            if len(missing_joint_examples) < 10:
                couple['tax_units'] = []
                for _, unit in hh_tax_units.iterrows():
                    couple['tax_units'].append({
                        'filing_status': unit['filing_status'],
                        'primary_filer_id': unit['primary_filer_id'],
                        'secondary_filer_id': unit['secondary_filer_id'],
                        'num_dependents': unit['num_dependents']
                    })
                missing_joint_examples.append(couple)
    
    print(f"\nResults:")
    print(f"Married couples found: {len(potential_joint_households)}")
    print(f"Filing jointly: {joint_count} ({joint_count/len(potential_joint_households)*100:.1f}%)")
    print(f"Missing joint filers: {len(potential_joint_households) - joint_count}")
    
    # Show examples of missing joint filers
    print(f"\nExamples of married couples NOT filing jointly:")
    for i, couple in enumerate(missing_joint_examples):
        print(f"\n--- Example {i+1}: Household {couple['hh_id']} ---")
        print(f"Householder: Age {couple['householder_age']}, Income ${couple['householder_income']:,.0f}")
        print(f"Spouse: Age {couple['spouse_age']}, Income ${couple['spouse_income']:,.0f}")
        print(f"Tax units created:")
        for unit in couple['tax_units']:
            print(f"  {unit['filing_status']}: {unit['primary_filer_id']}, deps={unit['num_dependents']}")
            if unit['secondary_filer_id']:
                print(f"    Secondary: {unit['secondary_filer_id']}")
    
    # Test the _are_married function on these couples
    print(f"\nTesting _are_married() function on missing joint couples:")
    
    for i, couple in enumerate(missing_joint_examples[:5]):
        hh_id = couple['hh_id']
        hh_persons = person_df[person_df['SERIALNO'] == hh_id]
        
        householder = hh_persons[hh_persons['RELSHIPP'] == 20].iloc[0]
        spouse = hh_persons[hh_persons['RELSHIPP'] == 21].iloc[0]
        
        are_married_result = _are_married(householder, spouse, hh_persons)
        
        print(f"\nHousehold {hh_id}:")
        print(f"  Householder: MAR={householder['MAR']}, RELSHIPP={householder['RELSHIPP']}, SPORDER={householder['SPORDER']}")
        print(f"  Spouse: MAR={spouse['MAR']}, RELSHIPP={spouse['RELSHIPP']}, SPORDER={spouse['SPORDER']}")
        print(f"  _are_married() result: {are_married_result}")
        
        if not are_married_result:
            print(f"  *** ISSUE: _are_married() returns False for obvious married couple! ***")
    
    # Calculate the impact
    missing_joint_rate = (len(potential_joint_households) - joint_count) / len(potential_joint_households) * 100
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Married couples (householder + spouse): {len(potential_joint_households)}")
    print(f"Actually filing jointly: {joint_count} ({joint_count/len(potential_joint_households)*100:.1f}%)")
    print(f"Missing joint filers: {len(potential_joint_households) - joint_count} ({missing_joint_rate:.1f}%)")
    print(f"\nThis could explain a significant portion of the joint filing undercount!")

if __name__ == "__main__":
    debug_missing_joint_couples()
