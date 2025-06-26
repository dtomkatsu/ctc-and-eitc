"""
Income calculation utilities.

This module provides functions for calculating different types of income
for tax purposes.
"""

from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np

def calculate_person_income(person: pd.Series) -> float:
    """
    Calculate total income for a single person.
    
    Args:
        person: Series containing person data
        
    Returns:
        float: Total income
        
    Note:
        This function uses individual income components (WAGP, SEMP, etc.)
        rather than PINCP to avoid potential double-counting issues.
    """
    # Initialize total income
    total_income = 0.0
    
    # Add up all income components
    # Wages, salaries, tips
    total_income += float(person.get('WAGP', 0) or 0)
    
    # Self-employment income (can be negative)
    total_income += float(person.get('SEMP', 0) or 0)
    
    # Interest income
    total_income += float(person.get('INTP', 0) or 0)
    
    # Retirement income
    total_income += float(person.get('RETP', 0) or 0)
    
    # Social Security benefits (only taxable portion)
    total_income += float(person.get('SSP', 0) or 0) * 0.85  # 85% is taxable
    
    # Supplemental Security Income (SSI) is not taxable
    # total_income += float(person.get('SSIP', 0) or 0)  # Excluded
    
    # Public assistance income (usually not taxable)
    # total_income += float(person.get('PAP', 0) or 0)  # Excluded
    
    # Other income
    total_income += float(person.get('OIP', 0) or 0)
    
    # Apply ADJINC (adjustment factor for income)
    adjinc = float(person.get('ADJINC', 1000000) or 1000000) / 1000000.0
    total_income *= adjinc
    
    return total_income

def calculate_household_income(household: pd.DataFrame) -> float:
    """
    Calculate total income for a household.
    
    Args:
        household: DataFrame containing all persons in a household
        
    Returns:
        float: Total household income
    """
    if household.empty:
        return 0.0
        
    # Calculate income for each person and sum
    return household.apply(calculate_person_income, axis=1).sum()

def calculate_tax_unit_income(tax_unit: pd.DataFrame) -> float:
    """
    Calculate total income for a tax filing unit.
    
    Args:
        tax_unit: DataFrame containing persons in a tax filing unit
        
    Returns:
        float: Total tax unit income
    """
    if tax_unit.empty:
        return 0.0
        
    # Calculate income for each person and sum
    return tax_unit.apply(calculate_person_income, axis=1).sum()

def calculate_agi(person_or_household: Union[pd.Series, pd.DataFrame]) -> float:
    """
    Calculate Adjusted Gross Income (AGI) for a person or household.
    
    Args:
        person_or_household: Series (for one person) or DataFrame (for multiple)
        
    Returns:
        float: AGI
    """
    if isinstance(person_or_household, pd.DataFrame):
        # For multiple people, sum their individual AGIs
        return person_or_household.apply(calculate_agi, axis=1).sum()
    
    # For a single person, AGI is similar to total income but with some adjustments
    # This is a simplified version - actual AGI calculation would be more complex
    total_income = calculate_person_income(person_or_household)
    
    # Common above-the-line deductions (simplified)
    deductions = 0.0
    
    # Student loan interest deduction (up to $2,500)
    if 'STLPAY' in person_or_household:
        deductions += min(float(person_or_household.get('STLPAY', 0) or 0), 2500)
    
    # IRA contributions (traditional, deductible)
    if 'IRAP' in person_or_household:
        deductions += float(person_or_household.get('IRAP', 0) or 0)
    
    # Health savings account deduction
    if 'HSP' in person_or_household:
        deductions += float(person_or_household.get('HSP', 0) or 0)
    
    # Educator expenses (up to $250)
    if 'EDUEXPP' in person_or_household:
        deductions += min(float(person_or_household.get('EDUEXPP', 0) or 0), 250)
    
    return max(0, total_income - deductions)
