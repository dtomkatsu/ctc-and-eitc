#!/usr/bin/env python3
"""
Test complete tax unit construction pipeline with hybrid weighting.
"""

import sys
import os
sys.path.append('/Users/dtomkatsu/CascadeProjects/ctc-and-eitc')

import pandas as pd
import numpy as np
from src.tax.units.constructor import TaxUnitConstructor

def create_comprehensive_test_data():
    """Create comprehensive test data with multiple household types."""
    
    # Create sample person data with various household structures
    person_data = {
        'SERIALNO': ['HH1', 'HH1', 'HH1', 'HH2', 'HH2', 'HH3', 'HH4', 'HH4', 'HH4'],
        'SPORDER': [1, 2, 3, 1, 2, 1, 1, 2, 3],
        'RELSHIPP': [20, 21, 22, 20, 22, 20, 20, 21, 22],  # Various relationships
        'AGEP': [35, 33, 8, 28, 5, 45, 40, 38, 12],
        'PINCP': [50000, 45000, 0, 30000, 0, 60000, 55000, 50000, 0],
        'PWGTP': [18.5, 19.2, 18.8, 22.1, 21.9, 25.3, 20.1, 19.8, 20.5],  # Person weights
        'ADJINC': [1.02, 1.02, 1.02, 1.01, 1.01, 1.03, 1.02, 1.02, 1.02],
        'MAR': [1, 1, 5, 5, 5, 5, 1, 1, 5],  # Marital status
        'SEX': [1, 2, 1, 2, 1, 1, 1, 2, 2],   # Sex
        'SCHL': [21, 20, 1, 18, 1, 22, 21, 19, 8],  # School enrollment level
        'ESR': [1, 1, 6, 1, 6, 1, 1, 1, 6],  # Employment status
        'WKHP': [40, 35, 0, 30, 0, 45, 40, 38, 0],  # Hours worked per week
        'PERNP': [48000, 43000, 0, 29000, 0, 58000, 53000, 48000, 0],  # Person earnings
        'INTP': [500, 200, 0, 100, 0, 800, 600, 400, 0],  # Interest income
        'RETP': [0, 0, 0, 0, 0, 5000, 0, 0, 0],  # Retirement income
        'SSIP': [0, 0, 0, 0, 0, 0, 0, 0, 0],  # SSI income
        'SSP': [0, 0, 0, 0, 0, 0, 0, 0, 0],  # Social Security income
        'WAGP': [49500, 44800, 0, 29700, 0, 59300, 54200, 49200, 0],  # Wage income
        'SEMP': [0, 0, 0, 0, 0, 0, 0, 0, 0],  # Self-employment income
        'POVPIP': [250, 250, 250, 180, 180, 320, 280, 280, 280],  # Poverty ratio
        'FPOWSP': [0, 0, 0, 0, 0, 0, 0, 0, 0]  # Family poverty status
    }
    
    person_df = pd.DataFrame(person_data)
    person_df.index = [f"{row['SERIALNO']}_{row['SPORDER']}" for _, row in person_df.iterrows()]
    
    # Create sample household data
    hh_data = {
        'SERIALNO': ['HH1', 'HH2', 'HH3', 'HH4'],
        'WGTP': [20.0, 23.5, 25.0, 21.5],  # Household weights
        'NP': [3, 2, 1, 3],                # Number of persons
        'HINCP': [95000, 30000, 60000, 105000] # Household income
    }
    
    hh_df = pd.DataFrame(hh_data)
    
    return person_df, hh_df

def test_full_pipeline():
    """Test the complete tax unit construction pipeline with hybrid weights."""
    
    print("Testing Complete Tax Unit Construction Pipeline with Hybrid Weights")
    print("=" * 70)
    
    # Create test data
    person_df, hh_df = create_comprehensive_test_data()
    
    print("Input Data Summary:")
    print(f"- {len(person_df)} persons across {len(hh_df)} households")
    print(f"- Household types: Joint filers, Single parents, Single adults")
    
    print("\nPerson DataFrame:")
    print(person_df[['SERIALNO', 'RELSHIPP', 'AGEP', 'PINCP', 'PWGTP', 'MAR']].to_string())
    
    print("\nHousehold DataFrame:")
    print(hh_df[['SERIALNO', 'WGTP', 'NP', 'HINCP']].to_string())
    
    # Initialize constructor
    constructor = TaxUnitConstructor(person_df, hh_df, batch_size=10, num_processes=1)
    
    # Process tax units
    print("\n" + "=" * 70)
    print("Processing Tax Units...")
    
    tax_units_df = constructor.create_rule_based_units(parallel=False)
    tax_units = tax_units_df.to_dict('records')
    
    print(f"\nCreated {len(tax_units)} tax units")
    
    # Analyze results
    print("\n" + "=" * 70)
    print("Tax Unit Analysis:")
    
    filing_status_counts = {}
    total_hybrid_weight = 0
    total_hh_weight = 0
    total_person_weight = 0
    
    for i, unit in enumerate(tax_units):
        filing_status = unit.get('filing_status', 'unknown')
        filing_status_counts[filing_status] = filing_status_counts.get(filing_status, 0) + 1
        
        hybrid_weight = unit.get('weight', 0)
        hh_weight = unit.get('hh_weight', 0)
        person_weight_sum = unit.get('person_weight_sum', 0)
        
        total_hybrid_weight += hybrid_weight
        total_hh_weight += hh_weight
        total_person_weight += person_weight_sum
        
        print(f"\nTax Unit {i+1}:")
        print(f"  Filing Status: {filing_status}")
        print(f"  Income: ${unit.get('income', 0):,.2f}")
        print(f"  Dependents: {unit.get('num_dependents', 0)}")
        print(f"  Household Weight: {hh_weight:.2f}")
        print(f"  Person Weight Sum: {person_weight_sum:.2f}")
        print(f"  Hybrid Weight: {hybrid_weight:.2f}")
        
        if hh_weight > 0 and person_weight_sum > 0:
            avg_weight = (hh_weight + person_weight_sum) / 2
            print(f"  Average Weight: {avg_weight:.2f}")
            
            # Determine expected adjustment factor
            status_factors = {
                'single': 0.96,
                'joint': 1.00,
                'head_of_household': 0.77,
                'married_filing_separately': 0.73
            }
            expected_factor = status_factors.get(filing_status, 1.0)
            expected_hybrid = avg_weight * expected_factor
            print(f"  Expected Hybrid (avg * {expected_factor}): {expected_hybrid:.2f}")
            print(f"  Matches Expected: {'✓' if abs(hybrid_weight - expected_hybrid) < 0.01 else '✗'}")
    
    print(f"\n" + "=" * 70)
    print("Summary Statistics:")
    print(f"Filing Status Distribution:")
    for status, count in filing_status_counts.items():
        print(f"  {status}: {count} units")
    
    print(f"\nWeight Totals:")
    print(f"  Total Household Weight: {total_hh_weight:.2f}")
    print(f"  Total Person Weight: {total_person_weight:.2f}")
    print(f"  Total Hybrid Weight: {total_hybrid_weight:.2f}")
    
    # Validate hybrid weight calculation
    print(f"\nHybrid Weight Validation:")
    expected_avg = (total_hh_weight + total_person_weight) / 2
    print(f"  Expected average before adjustments: {expected_avg:.2f}")
    print(f"  Actual hybrid total: {total_hybrid_weight:.2f}")
    print(f"  Difference due to filing status adjustments: {total_hybrid_weight - expected_avg:.2f}")
    
    print("\n" + "=" * 70)
    print("Pipeline test completed successfully! ✓")
    
    return tax_units

if __name__ == "__main__":
    test_full_pipeline()
