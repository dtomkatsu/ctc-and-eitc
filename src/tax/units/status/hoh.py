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
    """Check if person is unmarried for HOH purposes."""
    marital_status = person.get('MAR', 0)
    
    # Unmarried includes: never married (5), divorced (3), separated (4), widowed (2)
    if marital_status in [2, 3, 4, 5]:
        return True
    
    # If married (1), check if spouse is present in household
    if marital_status == 1:
        person_rel = person.get('RELSHIPP', 0)
        
        # If person is householder (20), look for spouse (21)
        if person_rel == 20:
            spouse_present = any(person_data['RELSHIPP'] == 21)
            return not spouse_present
        
        # If person is spouse (21), look for householder (20)  
        if person_rel == 21:
            householder_present = any(person_data['RELSHIPP'] == 20)
            return not householder_present
    
    return False

def _has_qualifying_person(person: pd.Series, person_data: pd.DataFrame) -> bool:
    """Check if person has a qualifying person for HOH."""
    household_id = person.get('SERIALNO')
    if not household_id:
        return False
        
    household = person_data[person_data['SERIALNO'] == household_id]
    person_rel = person.get('RELSHIPP', 0)
    
    # Check for qualifying children or relatives
    for _, other_person in household.iterrows():
        if other_person.name == person.name:
            continue
            
        other_rel = other_person.get('RELSHIPP', 0)
        other_age = other_person.get('AGEP', 0)
        
        # Qualifying child: biological/adopted/step child (22-24) under 19, 
        # or under 24 if full-time student, or any age if disabled
        if other_rel in [22, 23, 24]:  # Child relationships
            if other_age < 19:
                return True
            # TODO: Add student and disability checks when available in data
            
        # Grandchild (25) can also be qualifying child
        if other_rel == 25 and other_age < 19:
            return True
            
        # Foster child (34)
        if other_rel == 34 and other_age < 19:
            return True
            
        # Qualifying relative: other relatives who meet income test
        # This includes parents (27), siblings (26), grandparents (28), etc.
        if other_rel in [26, 27, 28, 30]:  # Various relative relationships
            other_income = other_person.get('PINCP', 0) or 0
            # Qualifying relative must have gross income less than exemption amount (~$4,700)
            if other_income < 5000:  # Approximate threshold
                return True
    
    return False

def _is_qualifying_child(
    child: pd.Series, 
    potential_hoh: pd.Series,
    person_data: pd.DataFrame
) -> bool:
    """Check if a person is a qualifying child of the potential HOH."""
    # Relationship to filer (child, stepchild, foster child, etc.)
    rel = child.get('RELSHIPP', 0)
    
    # PUMS relationship codes for children:
    # 3 = Biological son or daughter
    # 4 = Adopted son or daughter  
    # 5 = Stepson or stepdaughter
    # 6 = Brother or sister
    # 7 = Father or mother
    # 8 = Grandchild
    # 9 = Parent-in-law
    # 10 = Son-in-law or daughter-in-law
    # 11 = Other relative
    # 12 = Roomer, boarder
    # 13 = Housemate, roommate
    # 14 = Unmarried partner
    # 15 = Foster child
    # 16 = Other nonrelative
    # 17 = Institutionalized group quarters person
    
    if rel not in [3, 4, 5, 6, 8, 15]:  # Child relationships
        return False
        
    # Age test - must be under 19, or under 24 if full-time student, or any age if disabled
    age = child.get('AGEP', 0)
    if age >= 19:
        # Check if full-time student under 24
        if age < 24:
            school_level = child.get('SCHL', 0)
            # SCHL codes 16-21 indicate college/graduate school
            if school_level < 16:  # Not in college
                return False
        else:
            # Over 24, must be disabled
            disability = child.get('DIS', 2)
            if disability != 1:  # Not disabled
                return False
        
    # Residency test - must live with filer for more than half the year
    # Assuming same household means they live together (PUMS limitation)
    
    # Support test - cannot provide more than half of their own support
    child_income = _calculate_income(child)
    if child_income > 4300:  # 2023 threshold, should be configurable
        return False
        
    # Not filing a joint return (unless only to claim refund)
    if child.get('MAR') == 1:  # Married
        # In PUMS, we can't easily determine if it's just for refund
        # So we'll be conservative and exclude married children
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
    rel = relative.get('RELSHIPP', 0)
    
    # Valid relationships for qualifying relative
    valid_rels = [3, 4, 5, 6, 7, 8, 9, 10, 11, 15]  # Various relative relationships
    if rel not in valid_rels:
        return False
        
    # Gross income test
    income = _calculate_income(relative)
    if income >= 4300:  # 2023 amount, should be configurable
        return False
        
    # Support test - filer must provide more than half of support
    # This is simplified - in reality would need detailed expense data
    hoh_income = _calculate_income(potential_hoh)
    if hoh_income <= income:  # Simplified support test
        return False
        
    # Not filing a joint return (unless only to claim refund)
    if relative.get('MAR') == 1:  # Married
        return False
        
    return True

def _paid_half_home_cost(person: pd.Series, person_data: pd.DataFrame) -> bool:
    """Check if the person paid more than half the cost of keeping up a home."""
    household_id = person.get('SERIALNO')
    if not household_id:
        return False
        
    # Get household members
    household = person_data[person_data['SERIALNO'] == household_id]
    
    # For PUMS data, we'll use income as a proxy for contribution to household costs
    person_income = _calculate_income(person)
    
    # Calculate total adult income in household
    total_adult_income = 0
    adult_count = 0
    
    for _, member in household.iterrows():
        if member.get('AGEP', 0) >= 18:  # Adults
            member_income = _calculate_income(member)
            total_adult_income += member_income
            adult_count += 1
    
    # Person must contribute more than half of total adult income
    if total_adult_income > 0:
        contribution_ratio = person_income / total_adult_income
        return contribution_ratio > 0.5
    
    # If no income data, assume they pay half if they're the householder
    return person.get('RELSHIPP', 0) == 1

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
