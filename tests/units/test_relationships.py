"""
Tests for the relationships module.
"""

import pytest
import pandas as pd
import numpy as np
from tax.units.relationships import identify_relationships, _are_spouses, _is_parent_child, _are_related

# Test data
TEST_HOUSEHOLD = pd.DataFrame([
    # Reference person (20) - married, 40yo
    {'SERIALNO': '1', 'SPORDER': '1', 'AGEP': 40, 'SEX': 1, 'MAR': 1, 'RELSHIPP': '20', 'NAME_LAST': 'Smith'},
    # Spouse (21) - 38yo
    {'SERIALNO': '1', 'SPORDER': '2', 'AGEP': 38, 'SEX': 2, 'MAR': 1, 'RELSHIPP': '21', 'NAME_LAST': 'Smith'},
    # Child (00) - 10yo
    {'SERIALNO': '1', 'SPORDER': '3', 'AGEP': 10, 'SEX': 1, 'MAR': 0, 'RELSHIPP': '00', 'NAME_LAST': 'Smith'},
    # Stepchild (02) - 15yo
    {'SERIALNO': '1', 'SPORDER': '4', 'AGEP': 15, 'SEX': 2, 'MAR': 0, 'RELSHIPP': '02', 'NAME_LAST': 'Johnson'},
    # Unrelated individual - 30yo
    {'SERIALNO': '1', 'SPORDER': '5', 'AGEP': 30, 'SEX': 1, 'MAR': 0, 'RELSHIPP': '16', 'NAME_LAST': 'Williams'}
])

# Set index for easier access
TEST_HOUSEHOLD['person_id'] = TEST_HOUSEHOLD['SERIALNO'] + '_' + TEST_HOUSEHOLD['SPORDER']
TEST_HOUSEHOLD.set_index('person_id', inplace=True)

class TestRelationships:
    """Test cases for relationship identification."""
    
    def test_identify_relationships(self):
        """Test identification of all relationships in a household."""
        relationships = identify_relationships(TEST_HOUSEHOLD)
        
        # Should have identified 1 spousal relationship
        assert len(relationships['spouses']) == 1
        
        # Should have identified parent-child relationships
        assert len(relationships['parent_child']) > 0
        
        # Reference person should be parent of children
        ref_person = '1_1'
        children = [rel[1] for rel in relationships['parent_child'] if rel[0] == ref_person]
        assert len(children) >= 2  # At least 2 children
        
        # Should have some unrelated individuals
        assert len(relationships['unrelated']) > 0
    
    def test_are_spouses_positive(self):
        """Test positive case for spousal relationship."""
        person1 = TEST_HOUSEHOLD.loc['1_1']  # Reference person
        person2 = TEST_HOUSEHOLD.loc['1_2']  # Spouse
        
        assert _are_spouses(person1, person2) is True
    
    def test_are_spouses_negative(self):
        """Test negative case for spousal relationship."""
        person1 = TEST_HOUSEHOLD.loc['1_1']  # Reference person
        person3 = TEST_HOUSEHOLD.loc['1_3']  # Child
        
        assert _are_spouses(person1, person3) is False
    
    def test_is_parent_child_positive(self):
        """Test positive case for parent-child relationship."""
        parent = TEST_HOUSEHOLD.loc['1_1']  # Reference person
        child = TEST_HOUSEHOLD.loc['1_3']   # Child
        
        assert _is_parent_child(parent, child) is True
    
    def test_is_parent_child_negative(self):
        """Test negative case for parent-child relationship."""
        person1 = TEST_HOUSEHOLD.loc['1_1']  # Reference person
        person5 = TEST_HOUSEHOLD.loc['1_5']  # Unrelated individual
        
        assert _is_parent_child(person1, person5) is False
    
    def test_are_related_positive(self):
        """Test positive case for related individuals."""
        person1 = TEST_HOUSEHOLD.loc['1_1']  # Reference person
        person3 = TEST_HOUSEHOLD.loc['1_3']  # Child (same last name)
        
        assert _are_related(person1, person3) is True
    
    def test_are_related_negative(self):
        """Test negative case for related individuals."""
        person1 = TEST_HOUSEHOLD.loc['1_1']  # Reference person
        person5 = TEST_HOUSEHOLD.loc['1_5']  # Unrelated individual (different last name)
        
        assert _are_related(person1, person5) is False
