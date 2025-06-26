"""
Tests for the Married Filing Jointly status determination.
"""

import pytest
import pandas as pd
import numpy as np
from tax.units.status.mfj import is_married_filing_jointly, _are_married, _are_citizens_or_residents

# Test data
def create_test_person(serialno, sporder, age, married=True, relshipp='20', cit=1):
    """Helper function to create a test person."""
    return pd.Series({
        'SERIALNO': str(serialno),
        'SPORDER': str(sporder),
        'AGEP': age,
        'MAR': 1 if married else 0,
        'RELSHIPP': relshipp,
        'CIT': cit,
        'is_dependent': False
    })

class TestMarriedFilingJointly:
    """Test cases for Married Filing Jointly status determination."""
    
    def test_are_married_positive(self):
        """Test that a married couple is correctly identified."""
        # Create a married couple
        person1 = create_test_person('1', '1', 40, married=True, relshipp='20')  # Reference person
        person2 = create_test_person('1', '2', 38, married=True, relshipp='21')  # Spouse
        
        assert _are_married(person1, person2) is True
    
    def test_are_married_negative_not_married(self):
        """Test that non-married individuals are not identified as married."""
        person1 = create_test_person('1', '1', 40, married=False, relshipp='20')
        person2 = create_test_person('1', '2', 38, married=False, relshipp='21')
        
        assert _are_married(person1, person2) is False
    
    def test_are_married_negative_wrong_relationship(self):
        """Test that people with wrong relationship codes are not identified as married."""
        person1 = create_test_person('1', '1', 40, married=True, relshipp='20')
        person2 = create_test_person('1', '3', 15, married=False, relshipp='00')  # Child
        
        assert _are_married(person1, person2) is False
    
    def test_are_citizens_or_residents_positive(self):
        """Test that citizens and residents are correctly identified."""
        # Test all valid CIT codes (1-4)
        for cit in [1, 2, 3, 4]:
            person1 = create_test_person('1', '1', 40, cit=cit)
            person2 = create_test_person('1', '2', 38, cit=cit)
            assert _are_citizens_or_residents(person1, person2) is True
    
    def test_are_citizens_or_residents_negative(self):
        """Test that non-citizens/non-residents are correctly identified."""
        # CIT=5 means not a citizen
        person1 = create_test_person('1', '1', 40, cit=5)
        person2 = create_test_person('1', '2', 38, cit=1)
        assert _are_citizens_or_residents(person1, person2) is False
    
    def test_is_married_filing_jointly_positive(self):
        """Test a valid married filing jointly couple."""
        person1 = create_test_person('1', '1', 40, married=True, relshipp='20')
        person2 = create_test_person('1', '2', 38, married=True, relshipp='21')
        
        # Create a minimal person dataframe for testing
        person_df = pd.DataFrame([person1, person2])
        
        assert is_married_filing_jointly(person1, person2, person_df) is True
    
    def test_is_married_filing_jointly_negative_dependent(self):
        """Test that dependents cannot file jointly."""
        person1 = create_test_person('1', '1', 40, married=True, relshipp='20')
        person2 = create_test_person('1', '2', 38, married=True, relshipp='21')
        person2['is_dependent'] = True  # One is a dependent
        
        person_df = pd.DataFrame([person1, person2])
        
        assert is_married_filing_jointly(person1, person2, person_df) is False
    
    def test_is_married_filing_jointly_negative_not_married(self):
        """Test that non-married individuals cannot file jointly."""
        person1 = create_test_person('1', '1', 40, married=False, relshipp='20')
        person2 = create_test_person('1', '2', 38, married=False, relshipp='21')
        
        person_df = pd.DataFrame([person1, person2])
        
        assert is_married_filing_jointly(person1, person2, person_df) is False
    
    def test_is_married_filing_jointly_negative_not_citizens(self):
        """Test that non-citizens cannot file jointly."""
        person1 = create_test_person('1', '1', 40, married=True, relshipp='20', cit=5)
        person2 = create_test_person('1', '2', 38, married=True, relshipp='21', cit=5)
        
        person_df = pd.DataFrame([person1, person2])
        
        assert is_married_filing_jointly(person1, person2, person_df) is False
