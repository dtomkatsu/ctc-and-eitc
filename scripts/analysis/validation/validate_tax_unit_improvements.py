#!/usr/bin/env python3
"""
Validation script to demonstrate tax unit counting improvements.

This script shows that the updated tax unit constructor properly handles:
1. All adults are assigned to tax units
2. Dependents are correctly allocated
3. Different household structures are handled properly
"""

import sys
import pandas as pd
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tax.units.constructor import TaxUnitConstructor

def create_validation_data():
    """Create test data with known tax unit structures."""
    
    # Test households with different structures
    person_data = [
        # HH1: Married couple with 2 children (should be 1 joint tax unit)
        {'SERIALNO': '1', 'SPORDER': '1', 'AGEP': 35, 'SEX': 1, 'MAR': 1, 'RELSHIPP': 20, 'WAGP': 60000, 'HINCP': 100000, 'CIT': 1, 'SEMP': 0, 'ADJINC': 1.0, 'SCHL': 16},
        {'SERIALNO': '1', 'SPORDER': '2', 'AGEP': 33, 'SEX': 2, 'MAR': 1, 'RELSHIPP': 21, 'WAGP': 40000, 'HINCP': 100000, 'CIT': 1, 'SEMP': 0, 'ADJINC': 1.0, 'SCHL': 16},
        {'SERIALNO': '1', 'SPORDER': '3', 'AGEP': 5, 'SEX': 1, 'MAR': 0, 'RELSHIPP': 22, 'WAGP': 0, 'HINCP': 100000, 'CIT': 1, 'SEMP': 0, 'ADJINC': 1.0, 'SCHL': 1},
        {'SERIALNO': '1', 'SPORDER': '4', 'AGEP': 8, 'SEX': 2, 'MAR': 0, 'RELSHIPP': 22, 'WAGP': 0, 'HINCP': 100000, 'CIT': 1, 'SEMP': 0, 'ADJINC': 1.0, 'SCHL': 3},
        
        # HH2: Single parent with 1 child (should be 1 HoH tax unit)
        {'SERIALNO': '2', 'SPORDER': '1', 'AGEP': 30, 'SEX': 2, 'MAR': 5, 'RELSHIPP': 20, 'WAGP': 45000, 'HINCP': 50000, 'CIT': 1, 'SEMP': 0, 'ADJINC': 1.0, 'SCHL': 16},
        {'SERIALNO': '2', 'SPORDER': '2', 'AGEP': 8, 'SEX': 1, 'MAR': 0, 'RELSHIPP': 22, 'WAGP': 0, 'HINCP': 50000, 'CIT': 1, 'SEMP': 0, 'ADJINC': 1.0, 'SCHL': 5},
        
        # HH3: Two unrelated adults (should be 2 single tax units)
        {'SERIALNO': '3', 'SPORDER': '1', 'AGEP': 28, 'SEX': 1, 'MAR': 5, 'RELSHIPP': 20, 'WAGP': 35000, 'HINCP': 75000, 'CIT': 1, 'SEMP': 0, 'ADJINC': 1.0, 'SCHL': 16},
        {'SERIALNO': '3', 'SPORDER': '2', 'AGEP': 26, 'SEX': 2, 'MAR': 5, 'RELSHIPP': 16, 'WAGP': 40000, 'HINCP': 75000, 'CIT': 1, 'SEMP': 0, 'ADJINC': 1.0, 'SCHL': 16},
        
        # HH4: Single adult (should be 1 single tax unit)
        {'SERIALNO': '4', 'SPORDER': '1', 'AGEP': 50, 'SEX': 1, 'MAR': 5, 'RELSHIPP': 20, 'WAGP': 55000, 'HINCP': 55000, 'CIT': 1, 'SEMP': 0, 'ADJINC': 1.0, 'SCHL': 16},
        
        # HH5: Adult child living with parent (should be 2 single tax units)
        {'SERIALNO': '5', 'SPORDER': '1', 'AGEP': 65, 'SEX': 2, 'MAR': 5, 'RELSHIPP': 20, 'WAGP': 25000, 'HINCP': 70000, 'CIT': 1, 'SEMP': 0, 'ADJINC': 1.0, 'SCHL': 16},
        {'SERIALNO': '5', 'SPORDER': '2', 'AGEP': 25, 'SEX': 1, 'MAR': 5, 'RELSHIPP': 22, 'WAGP': 45000, 'HINCP': 70000, 'CIT': 1, 'SEMP': 0, 'ADJINC': 1.0, 'SCHL': 16},
    ]
    
    # Household data
    hh_data = [
        {'SERIALNO': '1', 'HINCP': 100000, 'ADJINC': 1.0},
        {'SERIALNO': '2', 'HINCP': 50000, 'ADJINC': 1.0},
        {'SERIALNO': '3', 'HINCP': 75000, 'ADJINC': 1.0},
        {'SERIALNO': '4', 'HINCP': 55000, 'ADJINC': 1.0},
        {'SERIALNO': '5', 'HINCP': 70000, 'ADJINC': 1.0},
    ]
    
    # Create DataFrames
    person_df = pd.DataFrame(person_data)
    hh_df = pd.DataFrame(hh_data)
    
    # Set up person DataFrame with proper index
    person_df['person_id'] = person_df['SERIALNO'] + '_' + person_df['SPORDER']
    person_df.set_index('person_id', inplace=True)
    
    return person_df, hh_df

def validate_tax_units():
    """Validate that tax unit construction works correctly."""
    print("=== Tax Unit Construction Validation ===\n")
    
    # Create test data
    person_df, hh_df = create_validation_data()
    
    print(f"Test data: {len(person_df)} persons in {len(hh_df)} households")
    
    # Count adults and children
    adults = person_df[person_df['AGEP'] >= 18]
    children = person_df[person_df['AGEP'] < 18]
    print(f"  - {len(adults)} adults")
    print(f"  - {len(children)} children\n")
    
    # Create tax units
    constructor = TaxUnitConstructor(person_df, hh_df)
    tax_units_df = constructor.create_rule_based_units()
    
    print(f"Tax units created: {len(tax_units_df)}")
    
    # Debug: Show the tax units DataFrame structure
    if len(tax_units_df) > 0:
        print("\nTax units DataFrame columns:", list(tax_units_df.columns))
        print("\nTax units sample:")
        # Only try to print columns that exist
        columns_to_show = []
        for col in ['filing_status', 'num_dependents', 'primary_filer_id', 'secondary_filer_id']:
            if col in tax_units_df.columns:
                columns_to_show.append(col)
        if columns_to_show:
            print(tax_units_df[columns_to_show].to_string())
        else:
            print("No standard tax unit columns found in DataFrame")
            print("Available columns:", list(tax_units_df.columns))
    
    # Analyze by filing status
    if len(tax_units_df) > 0 and 'filing_status' in tax_units_df.columns:
        status_counts = tax_units_df['filing_status'].value_counts().to_dict()
    else:
        status_counts = {}
    
    if len(tax_units_df) > 0 and 'num_dependents' in tax_units_df.columns:
        total_dependents = tax_units_df['num_dependents'].sum()
    else:
        total_dependents = 0
    
    print("\nTax units by filing status:")
    for status, count in status_counts.items():
        print(f"  - {status}: {count}")
    
    print(f"\nTotal dependents assigned: {total_dependents}")
    
    # Verify all adults are assigned
    assigned_adults = set()
    tax_unit_assignments = {}
    
    if not tax_units_df.empty:
        print("\n=== Detailed Tax Unit Assignments ===")
        for idx, unit in tax_units_df.iterrows():
            # Safely get filing status
            filing_status = unit.get('filing_status', 'unknown')
            print(f"\nTax Unit {idx} ({filing_status}):")
            
            # Handle primary filer if the column exists
            if 'primary_filer_id' in unit:
                primary = unit['primary_filer_id']
                if pd.notna(primary) and primary != '':
                    assigned_adults.add(str(primary))  # Ensure string ID
                    tax_unit_assignments[str(primary)] = (idx, 'primary')
                    print(f"  - Primary filer: {primary}")
            
            # Handle secondary filer if the column exists
            if 'secondary_filer_id' in unit:
                secondary = unit['secondary_filer_id']
                if pd.notna(secondary) and secondary != '':
                    assigned_adults.add(str(secondary))  # Ensure string ID
                    tax_unit_assignments[str(secondary)] = (idx, 'secondary')
                    print(f"  - Secondary filer: {secondary}")
            
            # Show dependents if any
            if 'dependents' in unit and unit['dependents']:
                print(f"  - Dependents: {unit['dependents']}")
            
            # Show income if available
            if 'income' in unit:
                print(f"  - Income: ${unit['income']:,.2f}")
            
            # Handle dependents
            deps = unit.get('dependents', [])
            if isinstance(deps, str):
                deps = eval(deps) if deps.startswith('[') else []
            print(f"  - Dependents: {len(deps) if deps else 0}")
            if deps:
                for dep in deps[:5]:  # Show first 5 dependents
                    print(f"    - {dep}")
                if len(deps) > 5:
                    print(f"    - ... and {len(deps) - 5} more")
    
    # Get all adults in the test data
    all_adults = set(adults.index)
    unassigned = all_adults - assigned_adults
    
    print(f"\n=== Adult Assignment Summary ===")
    print(f"Total adults in test data: {len(all_adults)}")
    print(f"Adults assigned to tax units: {len(assigned_adults)}")
    print(f"Adults not assigned to any tax unit: {len(unassigned)}")
    
    if unassigned:
        print("\nUnassigned adults (SERIALNO_SPORDER):")
        for person_id in sorted(list(unassigned)):
            person = person_df.loc[person_id]
            print(f"  - {person_id}: Age {person['AGEP']}, "
                  f"Sex {'M' if person['SEX'] == 1 else 'F'}, "
                  f"Relationship: {person['RELSHIPP']}, "
                  f"Marital: {person['MAR']}")
    else:
        print("✓ All adults successfully assigned to tax units")
    
    # Expected results validation
    print(f"\n=== Expected vs Actual Results ===")
    expected = {
        'joint': 1,  # HH1: married couple
        'head_of_household': 1,  # HH2: single parent
        'single': 5,  # HH3: 2 unrelated adults, HH4: 1 single adult, HH5: 2 adults (parent + adult child file separately)
        'total_units': 7,
        'total_dependents': 3  # HH1: 2 children, HH2: 1 child
    }
    
    actual = {
        'joint': status_counts.get('joint', 0),
        'head_of_household': status_counts.get('head_of_household', 0),
        'single': status_counts.get('single', 0),
        'total_units': len(tax_units_df),
        'total_dependents': total_dependents
    }
    
    print("Expected results:")
    for key, value in expected.items():
        print(f"  - {key}: {value}")
    
    print("\nActual results:")
    for key, value in actual.items():
        print(f"  - {key}: {value}")
    
    # Check if results match expectations
    print(f"\n=== Validation Summary ===")
    all_correct = True
    for key in expected:
        if expected[key] != actual[key]:
            print(f"  ❌ {key}: expected {expected[key]}, got {actual[key]}")
            all_correct = False
        else:
            print(f"  ✓ {key}: {actual[key]} (correct)")
    
    if all_correct and len(unassigned) == 0:
        print(f"\n🎉 SUCCESS: All validation checks passed!")
        print("The tax unit construction improvements are working correctly.")
    else:
        print(f"\n⚠️  Some validation checks failed. Review the results above.")
    
    return tax_units_df

if __name__ == "__main__":
    validate_tax_units()
