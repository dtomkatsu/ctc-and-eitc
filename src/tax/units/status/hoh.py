"""
Head of Household status determination.

This module contains logic for determining if a taxpayer qualifies for
Head of Household filing status.
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple, Union

def is_head_of_household(
    person: pd.Series, 
    person_data: pd.DataFrame
) -> bool:
    """
    Determine if a person qualifies as Head of Household.
    
    Args:
        person: The person's data
        person_data: Full person data for reference
        
    Returns:
        bool: True if qualifies as Head of Household
    """
    # Must be unmarried or considered unmarried on the last day of the year
    if not _is_unmarried(person, person_data):
        return False
        
    # Must have a qualifying person (usually a child) living with them
    if not _has_qualifying_person(person, person_data):
        return False
        
    # Must have paid more than half the cost of keeping up a home
    if not _paid_half_home_cost(person, person_data):
        return False
        
    return True

def _is_unmarried(person: pd.Series, person_data: pd.DataFrame) -> bool:
    """Check if a person is considered unmarried."""
    # Considered unmarried if:
    # 1. Single, divorced, or legally separated
    if person.get('MAR') in [3, 4, 5, 6]:  # Separated, divorced, widowed, never married
        return True
        
    # 2. Married but living apart from spouse for last 6 months of year
    # This is a simplification - would need more detailed data
    
    return False

def _has_qualifying_person(person: pd.Series, person_data: pd.DataFrame) -> bool:
    """Check if the person has at least one qualifying person."""
    household_id = person.get('SERIALNO')
    if not household_id:
        return False
        
    # Get all people in the same household
    household = person_data[person_data['SERIALNO'] == household_id]
    
    # Check each household member (excluding self)
    for _, member in household[household.index != person.name].iterrows():
        if _is_qualifying_child(member, person, person_data) or \
           _is_qualifying_relative(member, person, person_data):
            return True
            
    return False

def _is_qualifying_child(
    child: pd.Series, 
    potential_hoh: pd.Series,
    person_data: pd.DataFrame
) -> bool:
    """Check if a person is a qualifying child of the potential HOH."""
    # Relationship to filer (child, stepchild, foster child, etc.)
    rel = str(child.get('RELSHIPP', ''))
    if rel not in ['00', '02', '03', '04', '05']:  # Not a child relationship
        return False
        
    # Age test
    age = child.get('AGEP', 0)
    if age >= 19 and (age >= 24 or child.get('SCHL', 0) < 16):
        return False
        
    # Residency test - must live with filer for more than half the year
    # Assuming same household means they live together
    
    # Support test - cannot provide more than half of their own support
    # This is a simplification - would need more detailed data
    
    # Not filing a joint return (unless only to claim refund)
    if child.get('MAR') == 1:  # Married
        return False
        
    return True

def _is_qualifying_relative(
    relative: pd.Series, 
    potential_hoh: pd.Series,
    person_data: pd.DataFrame
) -> bool:
    """Check if a person is a qualifying relative of the potential HOH."""
    # Can't be a qualifying child of the filer or anyone else
    if _is_qualifying_child(relative, potential_hoh, person_data):
        return False
        
    # Relationship test - must be a relative or have lived with filer all year
    rel = str(relative.get('RELSHIPP', ''))
    if rel not in ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17']:
        return False
        
    # Gross income test
    income = _calculate_income(relative)
    if income >= 4300:  # 2023 amount, should be configurable
        return False
        
    # Support test - filer must provide more than half of support
    # This is a simplification - would need more detailed data
    
    # Not filing a joint return (unless only to claim refund)
    if relative.get('MAR') == 1:  # Married
        return False
        
    return True

def _paid_half_home_cost(person: pd.Series, person_data: pd.DataFrame) -> bool:
    """Check if the person paid more than half the cost of keeping up a home."""
    household_id = person.get('SERIALNO')
    if not household_id:
        return False
        
    # Get household costs (simplified)
    # In reality would need rent/mortgage, utilities, insurance, etc.
    household = person_data[person_data['SERIALNO'] == household_id]
    
    # Simplified: Assume income is proxy for contribution
    person_income = _calculate_income(person)
    total_income = household.apply(_calculate_income, axis=1).sum()
    
    return person_income > (total_income - person_income)  # More than half

def _calculate_income(person: pd.Series) -> float:
    """Calculate total income for a person."""
    income = 0.0
    for col in ['WAGP', 'SEMP', 'INTP', 'RETP', 'SSP', 'SSIP', 'PAP', 'OIP']:
        income += float(person.get(col, 0) or 0)
    return income
