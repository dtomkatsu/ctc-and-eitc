"""
Tests for the income module.
"""

import pytest
import pandas as pd
import numpy as np
from tax.units.income import calculate_person_income, calculate_household_income, calculate_tax_unit_income, calculate_agi

# Test data
TEST_PERSON = pd.Series({
    'WAGP': 50000,     # Wages
    'SEMP': 10000,     # Self-employment income
    'INTP': 500,       # Interest income
    'RETP': 1000,      # Retirement income
    'SSP': 12000,      # Social Security benefits (85% taxable)
    'SSIP': 0,        # SSI (not taxable)
    'PAP': 0,         # Public assistance (not taxable)
    'OIP': 500,       # Other income
    'ADJINC': 1000000, # Adjustment factor (1.0 for testing)
    'STLPAY': 2000,    # Student loan interest paid
    'IRAP': 3000,      # IRA contributions
    'HSP': 2000,       # HSA contributions
    'EDUEXPP': 200     # Educator expenses
})

TEST_HOUSEHOLD = pd.DataFrame([
    {'WAGP': 50000, 'SEMP': 10000, 'INTP': 500, 'RETP': 1000, 'SSP': 12000, 'SSIP': 0, 'PAP': 0, 'OIP': 500, 'ADJINC': 1000000},
    {'WAGP': 30000, 'SEMP': 0, 'INTP': 200, 'RETP': 500, 'SSP': 0, 'SSIP': 0, 'PAP': 0, 'OIP': 0, 'ADJINC': 1000000},
    {'WAGP': 0, 'SEMP': 0, 'INTP': 100, 'RETP': 0, 'SSP': 5000, 'SSIP': 6000, 'PAP': 0, 'OIP': 0, 'ADJINC': 1000000}
])

class TestIncomeCalculations:
    """Test cases for income calculation functions."""
    
    def test_calculate_person_income(self):
        """Test calculation of a person's total income."""
        # Expected: WAGP + SEMP + INTP + RETP + (SSP * 0.85) + OIP
        expected = 50000 + 10000 + 500 + 1000 + (12000 * 0.85) + 500
        result = calculate_person_income(TEST_PERSON)
        assert np.isclose(result, expected)
    
    def test_calculate_household_income(self):
        """Test calculation of total household income."""
        # Sum of all members' incomes
        person1 = 50000 + 10000 + 500 + 1000 + (12000 * 0.85) + 500
        person2 = 30000 + 200 + 500
        person3 = 100 + (5000 * 0.85)
        expected = person1 + person2 + person3
        
        result = calculate_household_income(TEST_HOUSEHOLD)
        assert np.isclose(result, expected)
    
    def test_calculate_tax_unit_income(self):
        """Test calculation of tax unit income."""
        # For a tax unit with two people
        tax_unit = TEST_HOUSEHOLD.iloc[:2].copy()
        person1 = 50000 + 10000 + 500 + 1000 + (12000 * 0.85) + 500
        person2 = 30000 + 200 + 500
        expected = person1 + person2
        
        result = calculate_tax_unit_income(tax_unit)
        assert np.isclose(result, expected)
    
    def test_calculate_agi(self):
        """Test calculation of Adjusted Gross Income."""
        # Total income - deductions
        total_income = 50000 + 10000 + 500 + 1000 + (12000 * 0.85) + 500
        deductions = min(2000, 2500) + 3000 + 2000 + min(200, 250)  # Student loan + IRA + HSA + Educator
        expected = total_income - deductions
        
        result = calculate_agi(TEST_PERSON)
        assert np.isclose(result, expected)
    
    def test_income_with_adjustment(self):
        """Test income calculation with ADJINC adjustment."""
        person = TEST_PERSON.copy()
        person['ADJINC'] = 1100000  # 1.1 adjustment
        
        base_income = 50000 + 10000 + 500 + 1000 + (12000 * 0.85) + 500
        expected = base_income * 1.1
        
        result = calculate_person_income(person)
        assert np.isclose(result, expected)
