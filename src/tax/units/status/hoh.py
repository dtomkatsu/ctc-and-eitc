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
    import logging
    logger = logging.getLogger(__name__)
    
    person_id = person.name
    
    # Must be unmarried or considered unmarried on the last day of the year
    if not _is_unmarried(person, person_data):
        logger.debug(f"Person {person_id} failed HoH: not unmarried (MAR={person.get('MAR', 'N/A')})")
        return False
        
    # Must have a qualifying person (usually a child) living with them
    if not _has_qualifying_person(person, person_data):
        logger.debug(f"Person {person_id} failed HoH: no qualifying person")
        return False
        
    # Must have paid more than half the cost of keeping up a home
    if not _paid_half_home_cost(person, person_data):
        logger.debug(f"Person {person_id} failed HoH: did not pay half home cost")
        return False
    
    logger.debug(f"Person {person_id} qualifies as Head of Household")
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
    import logging
    logger = logging.getLogger(__name__)
    
    household_id = person.get('SERIALNO')
    person_id = person.name
    
    if not household_id:
        logger.debug(f"Person {person_id} has no SERIALNO")
        return False
        
    household = person_data[person_data['SERIALNO'] == household_id]
    person_rel = person.get('RELSHIPP', 0)
    
    logger.debug(f"Person {person_id} checking for qualifying persons in household {household_id} with {len(household)} members")
    
    qualifying_found = False
    
    # Check for qualifying children or relatives
    for _, other_person in household.iterrows():
        if other_person.name == person.name:
            continue
            
        other_rel = other_person.get('RELSHIPP', 0)
        other_age = other_person.get('AGEP', 0)
        other_id = other_person.name
        
        logger.debug(f"  Checking person {other_id}: age={other_age}, rel={other_rel}")
        
        # Qualifying child: biological/adopted/step child (22-24)
        if other_rel in [22, 23, 24]:  # Child relationships
            if other_age < 19:
                logger.debug(f"  Person {other_id} qualifies as child under 19")
                qualifying_found = True
                return True
            # Check if full-time student under 24
            if other_age < 24:
                school_level = other_person.get('SCHL', 0)
                logger.debug(f"    Child {other_id} age {other_age}, school level {school_level}")
                # SCHL codes 16-21 indicate college/graduate school
                if school_level >= 16:  # In college
                    logger.debug(f"  Person {other_id} qualifies as student child")
                    qualifying_found = True
                    return True
            # Check if disabled (any age)
            if other_person.get('DIS', 2) == 1:  # 1 = With a disability
                logger.debug(f"  Person {other_id} qualifies as disabled child")
                qualifying_found = True
                return True
            logger.debug(f"    Child {other_id} does not qualify (age {other_age}, not student/disabled)")
                    
        # Grandchild (25) can also be qualifying child
        elif other_rel == 25:
            if other_age < 19:
                logger.debug(f"  Person {other_id} qualifies as grandchild under 19")
                qualifying_found = True
                return True
            # Check if full-time student under 24
            if other_age < 24:
                school_level = other_person.get('SCHL', 0)
                if school_level >= 16:  # In college
                    logger.debug(f"  Person {other_id} qualifies as student grandchild")
                    qualifying_found = True
                    return True
            # Check if disabled (any age)
            if other_person.get('DIS', 2) == 1:
                logger.debug(f"  Person {other_id} qualifies as disabled grandchild")
                qualifying_found = True
                return True
            logger.debug(f"    Grandchild {other_id} does not qualify (age {other_age})")
                
        # Foster child (34)
        elif other_rel == 34:
            if other_age < 19:
                logger.debug(f"  Person {other_id} qualifies as foster child under 19")
                qualifying_found = True
                return True
            # Check if full-time student under 24
            if other_age < 24:
                school_level = other_person.get('SCHL', 0)
                if school_level >= 16:  # In college
                    logger.debug(f"  Person {other_id} qualifies as student foster child")
                    qualifying_found = True
                    return True
            # Check if disabled (any age)
            if other_person.get('DIS', 2) == 1:
                logger.debug(f"  Person {other_id} qualifies as disabled foster child")
                qualifying_found = True
                return True
            logger.debug(f"    Foster child {other_id} does not qualify (age {other_age})")
            
        # Qualifying relative: other relatives who meet income test
        # This includes parents (27), siblings (26), grandparents (28), etc.
        elif other_rel in [26, 27, 28, 30]:  # Various relative relationships
            other_income = other_person.get('PINCP', 0) or 0
            # Apply ADJINC adjustment to income
            adjinc = other_person.get('ADJINC', 1.0)
            if adjinc and adjinc > 0:
                other_income = other_income * adjinc
            logger.debug(f"    Relative {other_id} income: {other_income}")
            # Qualifying relative must have gross income less than exemption amount (~$4,700)
            if other_income < 5000:  # Approximate threshold
                logger.debug(f"  Person {other_id} qualifies as low-income relative")
                qualifying_found = True
                return True
            logger.debug(f"    Relative {other_id} income too high: {other_income}")
        else:
            logger.debug(f"    Person {other_id} has non-qualifying relationship: {other_rel}")
    
    if not qualifying_found:
        logger.debug(f"Person {person_id} has no qualifying persons in household")
    
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
    """Check if the person paid more than half the cost of keeping up a home.
    
    For PUMS data, we'll use a realistic approach that considers:
    1. Householder status (primary indicator)
    2. Income contribution in multi-adult households
    3. Presence of dependents (indicates responsibility for household)
    """
    import logging
    logger = logging.getLogger(__name__)
    
    person_id = person.name
    person_rel = person.get('RELSHIPP', 0)
    
    # If person is the householder (RELSHIPP=20), they are almost certainly paying housing costs
    if person_rel == 20:
        logger.debug(f"Person {person_id} qualifies for home cost test as householder")
        return True
        
    household_id = person.get('SERIALNO')
    if not household_id:
        logger.debug(f"Person {person_id} has no SERIALNO")
        return False
        
    # Get household members
    household = person_data[person_data['SERIALNO'] == household_id]
    
    # Count number of adults in household
    adults = household[household['AGEP'] >= 18]
    num_adults = len(adults)
    
    # If only one adult, they must be paying the costs
    if num_adults == 1:
        logger.debug(f"Person {person_id} qualifies as only adult in household")
        return True
        
    # Check if this person has qualifying dependents in the household
    has_qualifying_dependents = False
    qualifying_dependents = []
    
    for _, other_person in household.iterrows():
        if other_person.name == person.name:
            continue
            
        other_rel = other_person.get('RELSHIPP', 0)
        other_age = other_person.get('AGEP', 0)
        
        # Check for children/dependents of this person
        if other_rel in [22, 23, 24, 25, 34]:  # Child, grandchild, foster relationships
            if other_age < 19 or (other_age < 24 and other_person.get('SCHL', 0) >= 16) or other_person.get('DIS', 2) == 1:
                has_qualifying_dependents = True
                qualifying_dependents.append(other_person.name)
    
    logger.debug(f"Person {person_id} has {len(qualifying_dependents)} qualifying dependents: {qualifying_dependents}")
    
    # If this person has qualifying dependents, be extremely lenient with cost test
    if has_qualifying_dependents:
        person_income = _calculate_income(person)
        total_adult_income = sum(_calculate_income(adult) for _, adult in adults.iterrows())
        
        logger.debug(f"Person {person_id} income: {person_income}, total adult income: {total_adult_income}")
        
        # For households with 2 adults where one has dependents
        if num_adults == 2:
            if total_adult_income > 0:
                contribution_ratio = person_income / total_adult_income
                logger.debug(f"Person {person_id} contribution ratio: {contribution_ratio:.3f}")
                # Extremely lenient threshold for parents with dependents (5%)
                if contribution_ratio >= 0.05:
                    logger.debug(f"Person {person_id} qualifies with {contribution_ratio:.1%} contribution (2 adults with dependents)")
                    return True
                else:
                    logger.debug(f"Person {person_id} failed home cost test: {contribution_ratio:.1%} < 5% (2 adults with dependents)")
                    return False
            else:
                # If no income data, assume parent with dependents pays costs
                logger.debug(f"Person {person_id} qualifies due to no income data but has dependents")
                return True
        
        # For households with 3+ adults where one has dependents
        elif num_adults >= 3:
            if total_adult_income > 0:
                contribution_ratio = person_income / total_adult_income
                logger.debug(f"Person {person_id} contribution ratio: {contribution_ratio:.3f}")
                # Extremely lenient threshold for parents with dependents (3%)
                if contribution_ratio >= 0.03:
                    logger.debug(f"Person {person_id} qualifies with {contribution_ratio:.1%} contribution (3+ adults with dependents)")
                    return True
                else:
                    logger.debug(f"Person {person_id} failed home cost test: {contribution_ratio:.1%} < 3% (3+ adults with dependents)")
                    return False
            else:
                # If no income data, assume parent with dependents pays costs
                logger.debug(f"Person {person_id} qualifies due to no income data but has dependents")
                return True
    
    # For people without qualifying dependents, check if they might still qualify
    # based on relationship and income contribution
    person_income = _calculate_income(person)
    total_adult_income = sum(_calculate_income(adult) for _, adult in adults.iterrows())
    
    if total_adult_income > 0:
        contribution_ratio = person_income / total_adult_income
        logger.debug(f"Person {person_id} (no dependents) contribution ratio: {contribution_ratio:.3f}")
        
        # For 2 adults without dependents, use moderate threshold
        if num_adults == 2 and contribution_ratio >= 0.35:
            logger.debug(f"Person {person_id} qualifies with {contribution_ratio:.1%} contribution (2 adults, no dependents)")
            return True
        
        # For 3+ adults without dependents, check if they contribute more than fair share
        elif num_adults >= 3:
            fair_share = 1.0 / num_adults
            if contribution_ratio > fair_share * 1.2:  # 20% above fair share
                logger.debug(f"Person {person_id} qualifies with {contribution_ratio:.1%} vs fair share {fair_share:.1%}")
                return True
    
    logger.debug(f"Person {person_id} failed home cost test")
    return False

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
