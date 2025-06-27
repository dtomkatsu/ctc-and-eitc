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
    
    # Other income
    total_income += float(person.get('OIP', 0) or 0)
    
    # Apply ADJINC (adjustment factor for income)
    adjinc = float(person.get('ADJINC', 1000000) or 1000000) / 1000000.0
    total_income *= adjinc
    
    return total_income

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
