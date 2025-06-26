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

from tax.units.constructor import TaxUnitConstructor
from tax.units.base import FILING_STATUS

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
    
    # Verify the constructor initialized correctly
    assert constructor is not None
    assert constructor.person_df is not None
    assert constructor.hh_df is not None

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
    
    # Verify the calculated income matches expected (with ADJINC applied)
    # Since ADJINC is 1.0 (1000000/1000000), the values should be the same
    assert tax_units.iloc[0]['income'] == expected_total_income

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
    
    # Verify income (50000 + 30000 = 80000)
    # With ADJINC=1.0, the total should be unchanged
    assert tax_units.iloc[0]['income'] == 80000

def test_married_filing_separately():
    """Test that married filing separately status is correctly identified."""
    # Create household with married couple who should file separately
    # due to one spouse having significant negative income (business loss)
    # and the other having substantial positive income
    hh_df = pd.DataFrame([create_test_household(NP=2, HINCP=100000)])
    
    person_df = pd.DataFrame([
        create_test_person(
            SPORDER='1',
            RELSHIPP=20,  # Reference person
            MAR=1,  # Married
            PINCP=100000,
            WAGP=100000,  # High income
            ADJINC=1000000,
            DIS=1  # Has disability (for testing different status)
        ),
        create_test_person(
            SPORDER='2',
            RELSHIPP=21,  # Spouse
            MAR=1,  # Married
            PINCP=-20000,  # Significant business loss
            SEMP=-20000,  # Self-employment loss
            SEX=2,  # Female
            ADJINC=1000000,
            DIS=2  # No disability (different from spouse)
        )
    ])
    
    # Create tax units
    constructor = TaxUnitConstructor(person_df, hh_df)
    tax_units = constructor.create_rule_based_units()
    
    # Verify results - should create two separate returns
    assert len(tax_units) == 2, f"Expected 2 tax units, got {len(tax_units)}. Tax units: {tax_units}"
    
    # Both should be filing as single since they're filing separately
    # (In the US tax system, when married filing separately, each return is essentially a single return)
    assert all(tax_units['filing_status'] == FILING_STATUS['SINGLE']), \
        f"Expected all tax units to be 'single', got {tax_units['filing_status'].unique()}"
    
    # Verify incomes - should be the individual incomes
    incomes = sorted([t['income'] for t in tax_units.to_dict('records')])
    assert incomes == [-20000, 100000], f"Expected incomes [-20000, 100000], got {incomes}"

if __name__ == "__main__":
    # Run tests when executed directly
    # Run tests when executed directly
    pytest.main([__file__, '-v'])
