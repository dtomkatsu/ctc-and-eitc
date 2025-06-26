"""
Tests for the Head of Household status determination.
"""

import pytest
import pandas as pd
import numpy as np
from tax.units.status.hoh import is_head_of_household, _is_unmarried, _has_qualifying_person, _paid_half_home_cost

# Test data
def create_test_household():
    """Create a test household with various members."""
    # Create household members
    members = [
        # Potential HOH - unmarried, has qualifying child
        {'SERIALNO': '1', 'SPORDER': '1', 'AGEP': 35, 'SEX': 1, 'MAR': 5, 'RELSHIPP': '20', 'HINCP': 40000, 'PAP': 0},
        # Qualifying child
        {'SERIALNO': '1', 'SPORDER': '2', 'AGEP': 10, 'SEX': 1, 'MAR': 0, 'RELSHIPP': '00', 'HINCP': 0, 'PAP': 0},
        # Non-qualifying person (adult)
        {'SERIALNO': '1', 'SPORDER': '3', 'AGEP': 25, 'SEX': 2, 'MAR': 0, 'RELSHIPP': '16', 'HINCP': 20000, 'PAP': 0},
        # Qualifying relative (elderly parent)
        {'SERIALNO': '1', 'SPORDER': '4', 'AGEP': 70, 'SEX': 2, 'MAR': 3, 'RELSHIPP': '03', 'HINCP': 5000, 'PAP': 0}
    ]
    
    # Create DataFrame and set index
    df = pd.DataFrame(members)
    df['person_id'] = df['SERIALNO'] + '_' + df['SPORDER'].astype(str)
    df.set_index('person_id', inplace=True)
    
    return df

class TestHeadOfHousehold:
    """Test cases for Head of Household status determination."""
    
    def test_is_unmarried_positive(self):
        """Test that unmarried individuals are correctly identified."""
        hh_df = create_test_household()
        potential_hoh = hh_df.loc['1_1']  # MAR=5 (Never married)
        
        assert _is_unmarried(potential_hoh, hh_df) is True
    
    def test_is_unmarried_negative(self):
        """Test that married individuals are not identified as unmarried."""
        hh_df = create_test_household()
        married_person = hh_df.loc['1_1'].copy()
        married_person['MAR'] = 1  # Married
        
        assert _is_unmarried(married_person, hh_df) is False
    
    def test_has_qualifying_person_positive_child(self):
        """Test identification of qualifying child."""
        hh_df = create_test_household()
        potential_hoh = hh_df.loc['1_1']  # Has a qualifying child (1_2)
        
        assert _has_qualifying_person(potential_hoh, hh_df) is True
    
    def test_has_qualifying_person_positive_relative(self):
        """Test identification of qualifying relative."""
        hh_df = create_test_household()
        # Create a person with a qualifying relative (elderly parent)
        person = hh_df.loc['1_3'].copy()  # Adult non-relative
        person['RELSHIPP'] = '01'  # Make them related to the elderly parent
        
        assert _has_qualifying_person(person, hh_df) is True
    
    def test_has_qualifying_person_negative(self):
        """Test case with no qualifying persons."""
        hh_df = create_test_household()
        # Create a person with no qualifying dependents
        person = hh_df.loc['1_3'].copy()  # Adult non-relative
        
        assert _has_qualifying_person(person, hh_df) is False
    
    def test_paid_half_home_cost_positive(self):
        """Test that person paying >50% of home costs is identified."""
        hh_df = create_test_household()
        potential_hoh = hh_df.loc['1_1']  # Makes $40k out of $65k total
        
        assert _paid_half_home_cost(potential_hoh, hh_df) is True
    
    def test_paid_half_home_cost_negative(self):
        """Test that person not paying >50% of home costs is not identified."""
        hh_df = create_test_household()
        person = hh_df.loc['1_3']  # Makes $20k out of $65k total
        
        assert _paid_half_home_cost(person, hh_df) is False
    
    def test_is_head_of_household_positive(self):
        """Test a valid Head of Household scenario."""
        hh_df = create_test_household()
        potential_hoh = hh_df.loc['1_1']  # Meets all criteria
        
        assert is_head_of_household(potential_hoh, hh_df) is True
    
    def test_is_head_of_household_negative_married(self):
        """Test that married individuals cannot be HOH."""
        hh_df = create_test_household()
        married_person = hh_df.loc['1_1'].copy()
        married_person['MAR'] = 1  # Married
        
        assert is_head_of_household(married_person, hh_df) is False
    
    def test_is_head_of_household_negative_no_qualifying_person(self):
        """Test that person with no qualifying dependents cannot be HOH."""
        hh_df = create_test_household()
        person = hh_df.loc['1_3']  # No qualifying dependents
        
        assert is_head_of_household(person, hh_df) is False
