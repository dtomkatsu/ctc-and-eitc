#!/usr/bin/env python3
"""
Test script for hybrid weighting implementation in tax unit constructor.
"""

import sys
import os
sys.path.append('/Users/dtomkatsu/CascadeProjects/ctc-and-eitc')

import pandas as pd
import numpy as np
from src.tax.units.constructor import TaxUnitConstructor

def create_test_data():
    """Create sample test data to verify hybrid weighting."""
    
    # Create sample person data
    person_data = {
        'SERIALNO': ['HH1', 'HH1', 'HH1', 'HH2', 'HH2'],
        'SPORDER': [1, 2, 3, 1, 2],
        'RELSHIPP': [20, 21, 22, 20, 22],  # Householder, Spouse, Child, Householder, Child
        'AGEP': [35, 33, 8, 28, 5],
        'PINCP': [50000, 45000, 0, 30000, 0],
        'PWGTP': [18.5, 19.2, 18.8, 22.1, 21.9],  # Person weights
        'ADJINC': [1.02, 1.02, 1.02, 1.01, 1.01],
        'MAR': [1, 1, 5, 5, 5],  # Married, Married, Never married, Never married, Never married
        'SEX': [1, 2, 1, 2, 1]   # Male, Female, Male, Female, Male
    }
    
    person_df = pd.DataFrame(person_data)
    person_df.index = [f"{row['SERIALNO']}_{row['SPORDER']}" for _, row in person_df.iterrows()]
    
    # Create sample household data
    hh_data = {
        'SERIALNO': ['HH1', 'HH2'],
        'WGTP': [20.0, 23.5],  # Household weights
        'NP': [3, 2],          # Number of persons
        'HINCP': [95000, 30000] # Household income
    }
    
    hh_df = pd.DataFrame(hh_data)
    # Keep SERIALNO as a column, don't set as index
    
    return person_df, hh_df

def test_hybrid_weighting():
    """Test the hybrid weighting implementation."""
    
    print("Testing Hybrid Weighting Implementation")
    print("=" * 50)
    
    # Create test data
    person_df, hh_df = create_test_data()
    
    print("Sample Data:")
    print("\nPerson DataFrame:")
    print(person_df[['SERIALNO', 'RELSHIPP', 'AGEP', 'PINCP', 'PWGTP', 'MAR']])
    
    print("\nHousehold DataFrame:")
    print(hh_df[['WGTP', 'NP', 'HINCP']])
    
    # Initialize constructor
    constructor = TaxUnitConstructor(person_df, hh_df, batch_size=10, num_processes=1)
    
    # Test the _calculate_hybrid_weight method directly
    print("\n" + "=" * 50)
    print("Testing _calculate_hybrid_weight method:")
    
    # Test case 1: Joint filer
    hh_weight = 20.0
    person_weights = [18.5, 19.2, 18.8]  # Two adults + one child
    filing_status = 'joint'
    
    hybrid_weight = constructor._calculate_hybrid_weight(hh_weight, person_weights, filing_status)
    
    print(f"\nTest Case 1 - Joint Filer:")
    print(f"  Household weight: {hh_weight}")
    print(f"  Person weights: {person_weights}")
    print(f"  Sum of person weights: {sum(person_weights)}")
    print(f"  Average of HH and person weights: {(hh_weight + sum(person_weights)) / 2}")
    print(f"  Filing status: {filing_status}")
    print(f"  Adjustment factor: 1.00 (joint)")
    print(f"  Final hybrid weight: {hybrid_weight}")
    
    # Test case 2: Single filer
    hh_weight = 23.5
    person_weights = [22.1, 21.9]  # Single adult + child
    filing_status = 'single'
    
    hybrid_weight = constructor._calculate_hybrid_weight(hh_weight, person_weights, filing_status)
    
    print(f"\nTest Case 2 - Single Filer:")
    print(f"  Household weight: {hh_weight}")
    print(f"  Person weights: {person_weights}")
    print(f"  Sum of person weights: {sum(person_weights)}")
    print(f"  Average of HH and person weights: {(hh_weight + sum(person_weights)) / 2}")
    print(f"  Filing status: {filing_status}")
    print(f"  Adjustment factor: 0.96 (single)")
    print(f"  Final hybrid weight: {hybrid_weight}")
    
    # Test case 3: Head of Household
    filing_status = 'head_of_household'
    hybrid_weight = constructor._calculate_hybrid_weight(hh_weight, person_weights, filing_status)
    
    print(f"\nTest Case 3 - Head of Household:")
    print(f"  Same inputs as Test Case 2")
    print(f"  Filing status: {filing_status}")
    print(f"  Adjustment factor: 0.77 (head_of_household)")
    print(f"  Final hybrid weight: {hybrid_weight}")
    
    # Test minimum weight enforcement
    print(f"\nTest Case 4 - Minimum Weight Enforcement:")
    very_low_weights = [0.05, 0.03]
    hybrid_weight = constructor._calculate_hybrid_weight(0.02, very_low_weights, 'single')
    print(f"  Very low weights: HH=0.02, Person=[0.05, 0.03]")
    print(f"  Calculated weight before minimum: {(0.02 + sum(very_low_weights)) / 2 * 0.96}")
    print(f"  Final hybrid weight (with 0.1 minimum): {hybrid_weight}")
    
    print("\n" + "=" * 50)
    print("Hybrid weighting implementation test completed!")
    
    return True

if __name__ == "__main__":
    test_hybrid_weighting()
