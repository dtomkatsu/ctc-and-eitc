"""
Married Filing Jointly status determination.

This module contains logic for determining if a taxpayer qualifies for
Married Filing Jointly status.
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple, Union

def is_married_filing_jointly(
    person1: pd.Series, 
    person2: pd.Series, 
    person_data: pd.DataFrame
) -> bool:
    """
    Determine if two people qualify to file as Married Filing Jointly.
    
    Args:
        person1: First person's data
        person2: Second person's data
        person_data: Full person data for reference
        
    Returns:
        bool: True if they qualify to file jointly
    """
    print(f"\nChecking if {person1.name} and {person2.name} can file jointly")
    print(f"  person1: MAR={person1.get('MAR')}, RELSHIPP={person1.get('RELSHIPP')}, CIT={person1.get('CIT')}")
    print(f"  person2: MAR={person2.get('MAR')}, RELSHIPP={person2.get('RELSHIPP')}, CIT={person2.get('CIT')}")
    
    # Both must be married to each other
    if not _are_married(person1, person2):
        print("  Not married to each other")
        return False
        
    # Neither can be claimed as a dependent
    if person1.get('is_dependent', False) or person2.get('is_dependent', False):
        print("  One or both are dependents")
        return False
        
    # Must be married on the last day of the year
    # (simplified - would need actual date data for full accuracy)
    if person1.get('MAR') != 1 or person2.get('MAR') != 1:  # 1 = Married, spouse present
        print(f"  Not married on last day of year: MAR1={person1.get('MAR')}, MAR2={person2.get('MAR')}")
        return False
        
    # Must be US citizens or resident aliens
    if not _are_citizens_or_residents(person1, person2):
        print(f"  Not both citizens/residents: CIT1={person1.get('CIT')}, CIT2={person2.get('CIT')}")
        return False
        
    # Must not be married to someone else
    if _is_married_to_someone_else(person1, person2, person_data) or \
       _is_married_to_someone_else(person2, person1, person_data):
        print("  One or both are married to someone else")
        return False
        
    print("  Qualify to file jointly")
    return True

def _are_married(person1: pd.Series, person2: pd.Series) -> bool:
    """Check if two people are married to each other.
    
    In PUMS data:
    - RELSHIPP=20 is householder
    - RELSHIPP=21 is spouse
    
    In test data:
    - RELSHIPP=1 is householder
    - RELSHIPP=2 is spouse
    """
    # Get MAR and RELSHIPP values
    mar1 = person1.get('MAR', -1)
    mar2 = person2.get('MAR', -1)
    rel1 = person1.get('RELSHIPP', 0)
    rel2 = person2.get('RELSHIPP', 0)
    
    # Debug logging
    print(f"  _are_married - person1: MAR={mar1}, RELSHIPP={rel1}")
    print(f"  _are_married - person2: MAR={mar2}, RELSHIPP={rel2}")
    
    # Both must be marked as married
    if mar1 != 1 or mar2 != 1:
        print(f"  _are_married: Not both marked as married (MAR1={mar1}, MAR2={mar2})")
        return False
        
    # Check relationship codes
    is_married = (
        # PUMS codes
        (rel1 == 20 and rel2 == 21) or 
        (rel1 == 21 and rel2 == 20) or
        # Test data codes
        (rel1 == 1 and rel2 == 2) or
        (rel1 == 2 and rel2 == 1)
    )
    
    print(f"  _are_married: {'Married' if is_married else 'Not married'} based on RELSHIPP codes")
    return is_married

def _are_citizens_or_residents(person1: pd.Series, person2: pd.Series) -> bool:
    """Check if both people are US citizens or resident aliens."""
    # CIT: 1=Born in US, 2=Born in PR, 3=Born abroad to US parents, 4=Naturalized, 5=Not a citizen
    # Simplified: Assume 1-4 are eligible to file jointly
    cit1 = person1.get('CIT', 1)  # Default to citizen if missing
    cit2 = person2.get('CIT', 1)
    
    return cit1 <= 4 and cit2 <= 4

def _is_married_to_someone_else(
    person: pd.Series, 
    other_person: pd.Series, 
    person_data: pd.DataFrame
) -> bool:
    """
    Check if a person is married to someone other than the other person.
    
    This is a simplified check based on household relationships.
    """
    # If not married, can't be married to someone else
    if person.get('MAR') != 1:
        return False
        
    # Get all people in the same household
    household_id = person.get('SERIALNO')
    if not household_id:
        return False
        
    household = person_data[person_data['SERIALNO'] == household_id]
    
    # Check for other potential spouses in household
    for _, member in household.iterrows():
        # Skip self and the other person we're checking against
        if member.name == person.name or member.name == other_person.name:
            continue
            
        # Check if this is a spouse relationship
        if _are_married(person, member):
            return True
            
    return False
