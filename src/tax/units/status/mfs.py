"""
Married Filing Separately status determination.

This module contains logic for determining if a taxpayer qualifies for
Married Filing Separately status.
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import random

def is_married_filing_separately(
    person1: pd.Series, 
    person2: pd.Series, 
    person_data: pd.DataFrame
) -> bool:
    """
    Determine if a married person should file as Married Filing Separately.
    
    Args:
        person1: First person's data
        person2: Second person's data (spouse)
        person_data: Full person data for reference
        
    Returns:
        bool: True if they should file separately
    """
    # Must be married to each other
    if not _are_married(person1, person2):
        return False
    
    # Get incomes
    income1 = _calculate_income(person1)
    income2 = _calculate_income(person2)
    
    # Check for large income disparity
    if income1 > 0 and income2 > 0:
        ratio = max(income1, income2) / min(income1, income2)
        if ratio > 10:  # One earns 10x more than the other
            return True
    
    # One spouse has significant negative income (business losses)
    if (income1 < -5000 and income2 > 50000) or (income2 < -5000 and income1 > 50000):
        return True
    
    # Different disability status (medical expense deduction reasons)
    if 'DIS' in person1 and 'DIS' in person2:
        if person1.get('DIS', 2) != person2.get('DIS', 2):
            if person1.get('DIS') == 1 or person2.get('DIS') == 1:
                return True
    
    # Different citizenship/immigration status
    if 'CIT' in person1 and 'CIT' in person2:
        if person1.get('CIT', 0) != person2.get('CIT', 0):
            if person1.get('CIT', 0) >= 4 or person2.get('CIT', 0) >= 4:
                return True
    
    # Public assistance income (can affect eligibility)
    pap1 = float(person1.get('PAP', 0) or 0)
    pap2 = float(person2.get('PAP', 0) or 0)
    if (pap1 > 0 and pap2 == 0) or (pap2 > 0 and pap1 == 0):
        return True
    
    # Student loan considerations
    intp1 = float(person1.get('INTP', 0) or 0)
    intp2 = float(person2.get('INTP', 0) or 0)
    if (intp1 > 10000 and income2 < 30000) or (intp2 > 10000 and income1 < 30000):
        return True
    
    # Self-employment income differences
    semp1 = float(person1.get('SEMP', 0) or 0)
    semp2 = float(person2.get('SEMP', 0) or 0)
    if (abs(semp1) > 50000 and semp2 == 0) or (abs(semp2) > 50000 and semp1 == 0):
        return True
    
    # Random assignment to reach target percentage (about 8.6% of married couples)
    random.seed(int(str(person1.get('SERIALNO', '0')) + str(person1.get('SPORDER', '0'))))
    if random.random() < 0.02:  # 2% random assignment
        return True
    
    return False

def _are_married(person1: pd.Series, person2: pd.Series) -> bool:
    """Check if two people are married to each other."""
    # Both must be marked as married
    if person1.get('MAR') != 1 or person2.get('MAR') != 1:
        return False
        
    # Check relationship codes
    rel1 = str(person1.get('RELSHIPP', ''))
    rel2 = str(person2.get('RELSHIPP', ''))
    
    # Should be reference person (20) and spouse (21) or vice versa
    return (rel1 == '20' and rel2 == '21') or (rel1 == '21' and rel2 == '20')

def _calculate_income(person: pd.Series) -> float:
    """Calculate total income for a person."""
    income = 0.0
    for col in ['WAGP', 'SEMP', 'INTP', 'RETP', 'SSP', 'SSIP', 'PAP', 'OIP']:
        income += float(person.get(col, 0) or 0)
    return income
