"""
Smoke tests for tax unit construction - focus on critical paths only.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / '..' / 'src'))

from tax.units import TaxUnitConstructor, FILING_STATUS

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

def test_basic_initialization():
    """Test that the constructor initializes without errors."""
    # Create minimal test data
    person_df = pd.DataFrame([create_test_person()])
    hh_df = pd.DataFrame([create_test_household()])
    
    # Initialize the constructor
    constructor = TaxUnitConstructor(person_df, hh_df)
    
    # Basic assertions
    assert constructor is not None
    assert 'person_id' in constructor.person_df.columns
    assert 'total_income' in constructor.person_df.columns

def test_single_adult_household():
    """Test single adult creates exactly one tax unit."""
    # Single adult household
    hh_df = pd.DataFrame([create_test_household(NP=1, HINCP=50000)])
    
    person_df = pd.DataFrame([
        create_test_person(
            SPORDER='1',
            RELSHIPP=20,  # Reference person
            MAR=5,  # Never married
            PINCP=50000,
            WAGP=50000
        )
    ])
    
    # Create tax units
    constructor = TaxUnitConstructor(person_df, hh_df)
    tax_units = constructor.create_rule_based_units()
    
    # Verify results
    assert len(tax_units) == 1
    assert tax_units.iloc[0]['filing_status'] == FILING_STATUS['SINGLE']
    assert tax_units.iloc[0]['num_dependents'] == 0

def test_income_calculation():
    """Test income calculations, including negative incomes."""
    # Create test data with different income scenarios
    # Note: Using raw income values (before ADJINC is applied)
    hh_df = pd.DataFrame([create_test_household(NP=2, HINCP=80000)])
    
    # Person 1: Positive income from wages
    # Person 2: Small negative income (business loss) - not enough to trigger MFS
    person_df = pd.DataFrame([
        create_test_person(
            SPORDER='1',
            RELSHIPP=20,  # Reference person
            MAR=1,  # Married
            PINCP=50000,  # Reduced from 60000
            WAGP=50000,  # Wages
            SERIALNO='1',
            ADJINC=1000000  # ADJINC of 1,000,000 = multiplier of 1.0
        ),
        create_test_person(
            SPORDER='2',
            RELSHIPP=21,  # Spouse
            MAR=1,  # Married
            PINCP=-2000,  # Reduced from -10000 to avoid MFS trigger
            SEMP=-2000,  # Self-employment loss
            SERIALNO='1',
            SEX=2,  # Different sex for spouse
            ADJINC=1000000  # ADJINC of 1,000,000 = multiplier of 1.0
        )
    ])
    
    # Create tax units
    constructor = TaxUnitConstructor(person_df, hh_df)
    tax_units = constructor.create_rule_based_units()
    
    # Verify results
    assert len(tax_units) == 1
    assert tax_units.iloc[0]['filing_status'] == FILING_STATUS['JOINT']
    
    # Get the raw income values (before ADJINC is applied)
    primary_income = 50000  # From WAGP
    secondary_income = -2000  # From SEMP
    expected_total_income = primary_income + secondary_income  # 48000
    
    # Verify the calculated total income matches expected (with ADJINC applied)
    # Since ADJINC is 1.0 (1000000/1000000), the values should be the same
    assert tax_units.iloc[0]['total_income'] == expected_total_income
    
    # Verify individual income calculations with ADJINC applied
    calc_primary_income = constructor._calculate_income(person_df.iloc[0])
    calc_secondary_income = constructor._calculate_income(person_df.iloc[1])
    
    # The calculated incomes should match the raw values since ADJINC is 1.0
    assert calc_primary_income == primary_income
    assert calc_secondary_income == secondary_income
    assert calc_primary_income + calc_secondary_income == expected_total_income
    
    # Test with a different ADJINC value
    person_df['ADJINC'] = 2000000  # ADJINC of 2.0 should double the income
    calc_primary_income = constructor._calculate_income(person_df.iloc[0])
    calc_secondary_income = constructor._calculate_income(person_df.iloc[1])
    assert calc_primary_income == 100000  # 50000 * 2.0
    assert calc_secondary_income == -4000  # -2000 * 2.0

def test_married_couple_joint():
    """Test married couple creates joint return."""
    # Create household with married couple
    hh_df = pd.DataFrame([create_test_household(NP=2, HINCP=80000)])
    
    person_df = pd.DataFrame([
        create_test_person(
            SPORDER='1',
            RELSHIPP=20,  # Reference person
            MAR=1,  # Married
            PINCP=50000,
            WAGP=50000,
            ADJINC=1000000  # ADJINC of 1,000,000 = multiplier of 1.0
        ),
        create_test_person(
            SPORDER='2',
            RELSHIPP=21,  # Spouse
            MAR=1,  # Married
            PINCP=30000,
            WAGP=30000,
            SEX=2,  # Female
            ADJINC=1000000  # ADJINC of 1,000,000 = multiplier of 1.0
        )
    ])
    
    # Create tax units
    constructor = TaxUnitConstructor(person_df, hh_df)
    tax_units = constructor.create_rule_based_units()
    
    # Verify results
    assert len(tax_units) == 1
    assert tax_units.iloc[0]['filing_status'] == FILING_STATUS['JOINT']
    
    # Verify total income (50000 + 30000 = 80000)
    # With ADJINC=1.0, the total should be unchanged
    assert tax_units.iloc[0]['total_income'] == 80000
    
    # Verify individual income calculations with ADJINC applied
    calc_primary_income = constructor._calculate_income(person_df.iloc[0])
    calc_secondary_income = constructor._calculate_income(person_df.iloc[1])
    
    assert calc_primary_income == 50000
    assert calc_secondary_income == 30000
    assert calc_primary_income + calc_secondary_income == 80000

if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, '-v'])
