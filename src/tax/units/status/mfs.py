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
    
    # Strong indicators for filing separately:
    
    # 1. Large income disparity (one spouse earns significantly more)
    if income1 > 0 and income2 > 0:
        ratio = max(income1, income2) / min(income1, income2)
        if ratio > 15:  # One earns 15x more than the other
            return True
    
    # 2. One spouse has significant negative income (business losses)
    if (income1 < -10000 and income2 > 75000) or (income2 < -10000 and income1 > 75000):
        return True
    
    # 3. One spouse has very high medical expenses (disability indicator)
    if 'DIS' in person1 and 'DIS' in person2:
        if person1.get('DIS', 2) != person2.get('DIS', 2):
            if person1.get('DIS') == 1 or person2.get('DIS') == 1:
                # Only if there's also significant income to make itemizing worthwhile
                if max(income1, income2) > 100000:
                    return True
    
    # 4. Different citizenship status with tax implications
    if 'CIT' in person1 and 'CIT' in person2:
        cit1 = person1.get('CIT', 0)
        cit2 = person2.get('CIT', 0)
        # If one is non-citizen and the other is citizen, and there's significant income
        if (cit1 >= 4 and cit2 < 4) or (cit2 >= 4 and cit1 < 4):
            if max(income1, income2) > 50000:
                return True
    
    # 5. Significant self-employment income differences
    semp1 = float(person1.get('SEMP', 0) or 0)
    semp2 = float(person2.get('SEMP', 0) or 0)
    if (abs(semp1) > 75000 and abs(semp2) < 10000) or (abs(semp2) > 75000 and abs(semp1) < 10000):
        return True
    
    # 6. One spouse has significant investment income
    intp1 = float(person1.get('INTP', 0) or 0)
    intp2 = float(person2.get('INTP', 0) or 0)
    if (intp1 > 25000 and intp2 < 5000) or (intp2 > 25000 and intp1 < 5000):
        return True
    
    # 7. Age difference suggesting different life stages/tax situations
    age1 = person1.get('AGEP', 0)
    age2 = person2.get('AGEP', 0)
    age_diff = abs(age1 - age2)
    if age_diff > 20 and max(income1, income2) > 80000:
        return True
    
    # 8. One spouse receiving significant public assistance
    pap1 = float(person1.get('PAP', 0) or 0)
    pap2 = float(person2.get('PAP', 0) or 0)
    if (pap1 > 5000 and pap2 == 0 and income2 > 40000) or (pap2 > 5000 and pap1 == 0 and income1 > 40000):
        return True
    
    # 9. Educational differences that might affect tax credits
    schl1 = person1.get('SCHL', 0)
    schl2 = person2.get('SCHL', 0)
    # If one is in graduate school (SCHL 20-21) and the other is not
    if ((schl1 >= 20 and schl2 < 16) or (schl2 >= 20 and schl1 < 16)) and max(income1, income2) > 60000:
        return True
    
    # Use a deterministic but pseudo-random approach based on household characteristics
    try:
        # Try to extract numeric part from SERIALNO
        serialno_str = str(person1.get('SERIALNO', '0'))
        # Extract digits only
        numeric_part = ''.join(filter(str.isdigit, serialno_str))
        if numeric_part:
            seed_base = int(numeric_part[-4:]) if len(numeric_part) >= 4 else int(numeric_part)
        else:
            seed_base = hash(serialno_str) % 10000
    except (ValueError, TypeError):
        seed_base = hash(str(person1.get('SERIALNO', '0'))) % 10000
    
    seed_value = seed_base + int(person1.get('SPORDER', 0)) + int(person2.get('SPORDER', 0))
    random.seed(seed_value)
    if random.random() < 0.01:  # 1% random assignment (reduced from 2%)
        return True
    
    return False

def _are_married(person1: pd.Series, person2: pd.Series) -> bool:
    """Check if two people are married to each other."""
    # Both must be marked as married
    if person1.get('MAR') != 1 or person2.get('MAR') != 1:
        return False
        
    # Check relationship codes
    rel1 = person1.get('RELSHIPP', 0)
    rel2 = person2.get('RELSHIPP', 0)
    
    # Should be reference person (1) and spouse (2) or vice versa
    return (rel1 == 1 and rel2 == 2) or (rel1 == 2 and rel2 == 1)

def _calculate_income(person: pd.Series) -> float:
    """Calculate total income for a person."""
    # Use PINCP (total person income) if available, otherwise sum components
    pincp = person.get('PINCP', 0)
    if pincp and pincp > 0:
        # Apply adjustment factor
        adjinc = person.get('ADJINC', 1.0)
        if adjinc and adjinc > 0:
            return float(pincp) * float(adjinc)
        return float(pincp)
    
    # Fallback to summing individual components
    income = 0.0
    for col in ['WAGP', 'SEMP', 'INTP', 'RETP', 'SSP', 'SSIP', 'PAP', 'OIP']:
        value = person.get(col, 0)
        if value:
            income += float(value)
    
    # Apply adjustment factor
    adjinc = person.get('ADJINC', 1.0)
    if adjinc and adjinc > 0:
        income *= float(adjinc)
    
    return income
