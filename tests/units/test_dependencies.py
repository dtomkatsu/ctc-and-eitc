"""
Tests for the dependencies module.
"""

import pytest
import pandas as pd
import numpy as np
from tax.units.dependencies import identify_dependents, _is_qualifying_child, _is_qualifying_relative

# Test data
def create_test_household():
    """Create a test household with various members."""
    # Create household members
    members = [
        # Primary filer (adult)
        {'SERIALNO': '1', 'SPORDER': '1', 'AGEP': 35, 'SEX': 1, 'MAR': 5, 'RELSHIPP': '20', 'SCHL': 16, 'SCH': 0, 'WAGP': 50000},
        # Spouse
        {'SERIALNO': '1', 'SPORDER': '2', 'AGEP': 33, 'SEX': 2, 'MAR': 1, 'RELSHIPP': '21', 'SCHL': 16, 'SCH': 0, 'WAGP': 45000},
        # Qualifying child 1 (under 19)
        {'SERIALNO': '1', 'SPORDER': '3', 'AGEP': 10, 'SEX': 1, 'MAR': 0, 'RELSHIPP': '00', 'SCHL': 3, 'SCH': 1, 'WAGP': 0},
        # Qualifying child 2 (student under 24)
        {'SERIALNO': '1', 'SPORDER': '4', 'AGEP': 20, 'SEX': 2, 'MAR': 0, 'RELSHIPP': '00', 'SCHL': 16, 'SCH': 1, 'WAGP': 5000},
        # Non-qualifying person (adult)
        {'SERIALNO': '1', 'SPORDER': '5', 'AGEP': 25, 'SEX': 1, 'MAR': 0, 'RELSHIPP': '16', 'SCHL': 16, 'SCH': 0, 'WAGP': 30000},
        # Qualifying relative (elderly parent)
        {'SERIALNO': '1', 'SPORDER': '6', 'AGEP': 70, 'SEX': 2, 'MAR': 3, 'RELSHIPP': '03', 'SCHL': 10, 'SCH': 0, 'WAGP': 10000},
        # Foster child
        {'SERIALNO': '1', 'SPORDER': '7', 'AGEP': 12, 'SEX': 1, 'MAR': 0, 'RELSHIPP': '05', 'SCHL': 4, 'SCH': 1, 'WAGP': 0}
    ]
    
    # Create DataFrame and set index
    df = pd.DataFrame(members)
    df['person_id'] = df['SERIALNO'] + '_' + df['SPORDER'].astype(str)
    df.set_index('person_id', inplace=True)
    
    return df

class TestDependencies:
    """Test cases for dependency identification."""
    
    def test_identify_dependents(self):
        """Test identification of all dependents in a household."""
        hh_df = create_test_household()
        dependents = identify_dependents(hh_df)
        
        # Primary filer should have dependents
        assert '1_1' in dependents
        assert len(dependents['1_1']) > 0
        
        # Should have identified all qualifying children and relatives
        # Primary filer's dependents should include children (1_3, 1_4, 1_7) and possibly relative (1_6)
        assert '1_3' in dependents['1_1']  # Child under 19
        assert '1_4' in dependents['1_1']  # Student under 24
        assert '1_7' in dependents['1_1']  # Foster child
        
        # Spouse might also be able to claim some dependents
        assert '1_2' in dependents
    
    def test_is_qualifying_child_positive(self):
        """Test identification of a qualifying child."""
        hh_df = create_test_household()
        parent = hh_df.loc['1_1']  # Primary filer
        child = hh_df.loc['1_3']   # Child under 19
        
        assert _is_qualifying_child(child, parent, hh_df) is True
    
    def test_is_qualifying_child_negative_age(self):
        """Test that a too-old non-student is not a qualifying child."""
        hh_df = create_test_household()
        parent = hh_df.loc['1_1']
        
        # Create a 25-year-old non-student
        old_child = hh_df.loc['1_4'].copy()
        old_child['AGEP'] = 25
        old_child['SCH'] = 0
        
        assert _is_qualifying_child(old_child, parent, hh_df) is False
    
    def test_is_qualifying_child_negative_relationship(self):
        """Test that unrelated individuals are not qualifying children."""
        hh_df = create_test_household()
        parent = hh_df.loc['1_1']  # Primary filer
        unrelated = hh_df.loc['1_5']  # Unrelated individual
        
        assert _is_qualifying_child(unrelated, parent, hh_df) is False
    
    def test_is_qualifying_relative_positive(self):
        """Test identification of a qualifying relative."""
        hh_df = create_test_household()
        filer = hh_df.loc['1_1']  # Primary filer
        relative = hh_df.loc['1_6']  # Elderly parent
        
        # Make relative's income low enough to qualify
        relative['WAGP'] = 4000
        
        assert _is_qualifying_relative(relative, filer, hh_df) is True
    
    def test_is_qualifying_relative_negative_income(self):
        """Test that high-income individuals are not qualifying relatives."""
        hh_df = create_test_household()
        filer = hh_df.loc['1_1']
        relative = hh_df.loc['1_6']
        
        # Make relative's income too high
        relative['WAGP'] = 5000  # Above the threshold
        
        assert _is_qualifying_relative(relative, filer, hh_df) is False
    
    def test_is_qualifying_relative_negative_relationship(self):
        """Test that unrelated individuals are not qualifying relatives."""
        hh_df = create_test_household()
        filer = hh_df.loc['1_1']
        unrelated = hh_df.loc['1_5']  # Unrelated individual
        
        assert _is_qualifying_relative(unrelated, filer, hh_df) is False
