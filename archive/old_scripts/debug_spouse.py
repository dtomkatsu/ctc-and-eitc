#!/usr/bin/env python3
"""
Debug script to understand spouse identification issues
"""
import pandas as pd
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / 'src'))

from tax.units import TaxUnitConstructor

def create_test_person(**overrides):
    """Create a test person with sensible defaults."""
    person = {
        'SERIALNO': '1',
        'SPORDER': '1',
        'RELSHIPP': 20,  # Reference person
        'SEX': 1,
        'AGEP': 35,
        'MAR': 1,  # Married
        'PINCP': 50000.0,
        'PWGTP': 100,
        'RAC1P': 1,
        'SCHL': 21,
        'ESR': 1,
        'WAGP': 0.0,
        'SEMP': 0.0,
        'INTP': 0.0,
        'RETP': 0.0,
        'OIP': 0.0,
        'PAP': 0.0,
        'SSP': 0.0,
        'SSIP': 0.0
    }
    person.update(overrides)
    return person

def create_test_household(**overrides):
    """Create a test household with sensible defaults."""
    hh = {
        'SERIALNO': '1',
        'HINCP': 100000.0,
        'NP': 2,
        'TYPE': 1,
        'TEN': 1,
        'YBL': 2000
    }
    hh.update(overrides)
    return hh

def main():
    print("=== Debugging Spouse Identification ===")
    
    # Create test data - married couple
    hh_df = pd.DataFrame([create_test_household(NP=2, HINCP=80000)])
    
    person_df = pd.DataFrame([
        create_test_person(
            SPORDER='1',
            RELSHIPP=20,  # Reference person
            MAR=1,  # Married
            PINCP=60000,
            WAGP=60000,
            SERIALNO='1',
            ADJINC=1000000
        ),
        create_test_person(
            SPORDER='2',
            RELSHIPP=21,  # Spouse
            MAR=1,  # Married
            PINCP=-10000,
            SEMP=-10000,
            SERIALNO='1',
            SEX=2,  # Different sex for spouse
            ADJINC=1000000
        )
    ])
    
    print("Input data:")
    print("Person DataFrame:")
    print(person_df[['SERIALNO', 'SPORDER', 'RELSHIPP', 'MAR', 'AGEP', 'SEX', 'PINCP']])
    print("\nHousehold DataFrame:")
    print(hh_df[['SERIALNO', 'HINCP', 'NP']])
    
    # Create constructor and check preprocessing
    constructor = TaxUnitConstructor(person_df, hh_df)
    
    print("\nAfter preprocessing:")
    print("Person DataFrame (relevant columns):")
    cols_to_show = ['person_id', 'SERIALNO', 'SPORDER', 'RELSHIPP', 'MAR', 'AGEP', 'SEX', 'num_adults', 'PINCP']
    available_cols = [col for col in cols_to_show if col in constructor.person_df.columns]
    print(constructor.person_df[available_cols])
    
    # Test spouse identification directly
    person1 = constructor.person_df.iloc[0]
    person2 = constructor.person_df.iloc[1]
    
    print(f"\nTesting spouse identification between:")
    print(f"Person 1: {person1['person_id']} (RELSHIPP={person1['RELSHIPP']}, MAR={person1['MAR']})")
    print(f"Person 2: {person2['person_id']} (RELSHIPP={person2['RELSHIPP']}, MAR={person2['MAR']})")
    
    is_spouse = constructor._is_potential_spouse(person1, person2)
    print(f"Are they spouses? {is_spouse}")
    
    # Create tax units
    print("\n=== Creating Tax Units ===")
    tax_units = constructor.create_rule_based_units()
    
    print(f"Number of tax units created: {len(tax_units)}")
    if len(tax_units) > 0:
        print("\nTax units:")
        for i, unit in tax_units.iterrows():
            print(f"Unit {i}: {unit['filing_status']} - Primary: {unit['primary_filer_id']}, Secondary: {unit.get('secondary_filer_id', 'None')}")

if __name__ == "__main__":
    main()
