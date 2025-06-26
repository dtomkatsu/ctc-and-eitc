"""
Relationship detection and management.

This module provides functions for identifying relationships between
household members for tax purposes.
"""

from typing import Dict, List, Optional, Set, Tuple, Union
import pandas as pd
import numpy as np

def identify_relationships(household: pd.DataFrame) -> Dict[str, List[Tuple[str, str]]]:
    """
    Identify all relationships within a household.
    
    Args:
        household: DataFrame containing all persons in a household
        
    Returns:
        dict: Mapping from relationship type to list of (person1_id, person2_id) tuples
    """
    if household.empty:
        return {}
        
    relationships = {
        'spouses': [],
        'parent_child': [],
        'siblings': [],
        'other_relative': [],
        'unrelated': []
    }
    
    # Get all person IDs in the household
    person_ids = household.index.tolist()
    
    # Check all pairs of people in the household
    for i, id1 in enumerate(person_ids):
        person1 = household.loc[id1]
        
        for j in range(i + 1, len(person_ids)):
            id2 = person_ids[j]
            person2 = household.loc[id2]
            
            # Check for spousal relationship
            if _are_spouses(person1, person2):
                relationships['spouses'].append((id1, id2))
                continue
                
            # Check for parent-child relationship
            if _is_parent_child(person1, person2):
                relationships['parent_child'].append((id1, id2))
                continue
                
            # Check for siblings
            if _are_siblings(person1, person2, household):
                relationships['siblings'].append((id1, id2))
                continue
                
            # Check for other family relationships
            if _are_related(person1, person2):
                relationships['other_relative'].append((id1, id2))
            else:
                relationships['unrelated'].append((id1, id2))
    
    return relationships

def _are_spouses(person1: pd.Series, person2: pd.Series) -> bool:
    """
    Check if two people are spouses.
    
    Args:
        person1: First person's data
        person2: Second person's data
        
    Returns:
        bool: True if they are spouses
    """
    # Both must be marked as married
    if person1.get('MAR') != 1 or person2.get('MAR') != 1:
        return False
        
    # Check relationship codes
    rel1 = str(person1.get('RELSHIPP', ''))
    rel2 = str(person2.get('RELSHIPP', ''))
    
    # Should be reference person (20) and spouse (21) or vice versa
    return (rel1 == '20' and rel2 == '21') or (rel1 == '21' and rel2 == '20')

def _is_parent_child(parent: pd.Series, child: pd.Series) -> bool:
    """
    Check if one person is the parent of another.
    
    Args:
        parent: Potential parent's data
        child: Potential child's data
        
    Returns:
        bool: True if parent-child relationship exists
    """
    # Check if child is actually a child by age
    if child.get('AGEP', 0) >= 18 and not _is_student(child):
        return False
        
    # Check relationship codes
    parent_rel = str(parent.get('RELSHIPP', ''))
    child_rel = str(child.get('RELSHIPP', ''))
    
    # Parent is reference person (20), child is child (00, 02, 03, 04, 05)
    if parent_rel == '20' and child_rel in ['00', '02', '03', '04', '05']:
        return True
        
    # Could add more sophisticated checks here based on other fields
    
    return False

def _are_siblings(
    person1: pd.Series, 
    person2: pd.Series, 
    household: pd.DataFrame
) -> bool:
    """
    Check if two people are siblings.
    
    Args:
        person1: First person's data
        person2: Second person's data
        household: Full household data for reference
        
    Returns:
        bool: True if they are siblings
    """
    # Must have the same parents in the household
    # This is a simplified check - would need more complete family relationship data
    
    # Check if both are children of the reference person
    rel1 = str(person1.get('RELSHIPP', ''))
    rel2 = str(person2.get('RELSHIPP', ''))
    
    if rel1 in ['00', '02', '03', '04', '05'] and \
       rel2 in ['00', '02', '03', '04', '05']:
        return True
        
    return False

def _are_related(person1: pd.Series, person2: pd.Series) -> bool:
    """
    Check if two people are related in some way.
    
    Args:
        person1: First person's data
        person2: Second person's data
        
    Returns:
        bool: True if they are related
    """
    # Check if they have the same last name (if available)
    if 'NAME_LAST' in person1 and 'NAME_LAST' in person2:
        if person1['NAME_LAST'] and person2['NAME_LAST']:
            if person1['NAME_LAST'] == person2['NAME_LAST']:
                return True
    
    # Could add more sophisticated relationship checking here
    
    return False

def _is_student(person: pd.Series) -> bool:
    """
    Check if a person is a student.
    
    Args:
        person: Person's data
        
    Returns:
        bool: True if the person is a student
    """
    # Check school enrollment
    if 'SCH' in person:
        return person['SCH'] == 1  # 1 = Yes, in school
        
    # Check age and education level
    age = person.get('AGEP', 0)
    education = person.get('SCHL', 0)
    
    # In college or graduate school
    if education >= 16 and age <= 24:
        return True
        
    return False
