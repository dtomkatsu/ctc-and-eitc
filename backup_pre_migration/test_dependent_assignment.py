#!/usr/bin/env python3
"""
Test script to validate dependent assignment in tax unit construction.
"""
import sys
import os
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.tax.units.constructor import TaxUnitConstructor
from src.tax.units.dependencies import identify_dependents

def create_test_data():
    """Create test data for dependent assignment testing."""
    # Household data
    hh_data = {
        'SERIALNO': ['HH1', 'HH1', 'HH1', 'HH1', 'HH2', 'HH2', 'HH2', 'HH3', 'HH3', 'HH3', 'HH3', 'HH3'],
        'SPORDER': [1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5],
        'AGEP': [40, 38, 12, 8, 35, 10, 5, 45, 43, 20, 18, 16],  # Ages
        'SEX': [1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1],  # 1=Male, 2=Female
        'RELSHIPP': [20, 21, 22, 22, 20, 22, 22, 20, 21, 22, 22, 22],  # 20=Householder, 21=Spouse, 22=Child
        'MAR': [1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],  # 1=Married, 0=Not married
        'CIT': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 1=Born in US
        'SCHL': [21, 20, 12, 10, 22, 8, 6, 21, 19, 15, 16, 14],  # Educational attainment
        'PINCP': [80000, 60000, 0, 0, 90000, 0, 0, 100000, 0, 5000, 0, 0],  # Income
        'ADJINC': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Adjustment factor
        'SEMP': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Self-employment income
        'RAC1P': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Race
        'HISP': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Not Hispanic
        'MSP': [1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],  # Married, spouse present
        'SSIP': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # No SSI income
        'SSP': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # No Social Security income
        'RETP': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # No retirement income
        'OIP': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # No other income
        'WAGP': [80000, 60000, 0, 0, 90000, 0, 0, 100000, 0, 5000, 0, 0],  # Wages/salary
        'INTP': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # No interest income
        'PAP': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # No public assistance income
    }
    
    # Create person DataFrame
    person_df = pd.DataFrame(hh_data)
    
    # Create household DataFrame with additional required columns
    hh_df = pd.DataFrame({
        'SERIALNO': ['HH1', 'HH2', 'HH3'],
        'HINCP': [140000, 90000, 105000],  # Household income
        'NP': [4, 3, 5],  # Number of persons in household
        'HHT': [1, 1, 1],  # Family household
        'TEN': [1, 1, 1],  # Owned with mortgage
        'HUPAC': [1, 1, 1],  # House or apartment
        'ACCESSINET': [1, 1, 1],  # Has internet access
        'WORKSTAT': [1, 1, 1],  # Worked last week
        'VEHICLES': [2, 1, 2]  # Number of vehicles
    })
    
    return person_df, hh_df

def test_dependent_assignment():
    """Test dependent assignment in tax unit construction."""
    print("Testing dependent assignment...")
    
    # Create test data
    person_df, hh_df = create_test_data()
    
    # Create tax unit constructor
    constructor = TaxUnitConstructor(person_df, hh_df)
    
    # Create tax units
    tax_units = constructor.create_rule_based_units()
    
    # Print results
    print("\nTax Units Created:")
    if not tax_units.empty:
        print(tax_units[['filer_id', 'filing_status', 'num_dependents', 'dependents']].to_string())
    
    # Verify dependent assignment
    print("\nVerifying dependent assignment:")
    
    # Get all person IDs from the constructor's processed data
    person_ids = set(constructor.person_df.index)
    
    # Get all dependents assigned to tax units
    assigned_dependents = set()
    if not tax_units.empty:
        for _, unit in tax_units.iterrows():
            if isinstance(unit['dependents'], list):
                assigned_dependents.update(unit['dependents'])
    
    # Get expected dependents (all children in the data)
    children = constructor.person_df[constructor.person_df['AGEP'] < 18]
    expected_dependents = set(children.index)
    
    # Check for unassigned dependents
    unassigned = expected_dependents - assigned_dependents
    if unassigned:
        print(f"❌ Found {len(unassigned)} unassigned dependents: {unassigned}")
    else:
        print("✅ All expected dependents were assigned to tax units")
    
    # Check for duplicate assignments
    all_deps = []
    if not tax_units.empty:
        for _, unit in tax_units.iterrows():
            if isinstance(unit['dependents'], list):
                all_deps.extend(unit['dependents'])
    
    duplicate_deps = set([x for x in all_deps if all_deps.count(x) > 1])
    if duplicate_deps:
        print(f"❌ Found {len(duplicate_deps)} dependents assigned to multiple tax units: {duplicate_deps}")
    else:
        print("✅ No dependents are assigned to multiple tax units")
    
    # Print detailed tax unit information
    print("\nDetailed Tax Unit Information:")
    if not tax_units.empty:
        for _, unit in tax_units.iterrows():
            print(f"\nTax Unit ID: {unit['filer_id']}")
            print(f"  Filing Status: {unit['filing_status']}")
            print(f"  Number of Dependents: {unit['num_dependents']}")
            print(f"  Dependents: {unit['dependents']}")
            print(f"  Income: ${unit['income']:,.2f}" if 'income' in unit else "  Income: Not available")

if __name__ == "__main__":
    test_dependent_assignment()
