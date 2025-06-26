"""
Implementation of SingleFiler for single tax filers.
"""

from typing import Dict, List, Optional, Union
import pandas as pd
from .base import BaseFiler

class SingleFiler(BaseFiler):
    """
    Represents a single tax filer.
    
    This class handles the specific logic for single filers,
    including income calculation and eligibility rules.
    """
    
    def get_filing_status(self) -> str:
        """Return the filing status for this filer."""
        return 'single'
    
    def calculate_income(self, person_data: pd.DataFrame) -> float:
        """
        Calculate total income for a single filer.
        
        Args:
            person_data: DataFrame containing person data
            
        Returns:
            float: Total income
        """
        if not self.members:
            return 0.0
            
        # For single filers, just return the income of the primary member
        member_id = self.members[0]
        member_data = person_data.loc[member_id]
        
        # Sum up all income components
        income = 0.0
        for col in ['WAGP', 'SEMP', 'INTP', 'RETP', 'SSP', 'SSIP', 'PAP', 'OIP']:
            if col in person_data.columns:
                income += float(member_data.get(col, 0) or 0)
                
        return income
    
    def is_eligible(self, person_data: pd.DataFrame) -> bool:
        """
        Check if this filer is eligible to file as single.
        
        Args:
            person_data: DataFrame containing person data
            
        Returns:
            bool: True if eligible, False otherwise
        """
        if len(self.members) != 1:
            return False
            
        member_id = self.members[0]
        member_data = person_data.loc[member_id]
        
        # Must be at least 19 years old (or 24 if a student)
        age = member_data.get('AGEP', 0)
        is_student = member_data.get('SCHL', 0) >= 16  # Approximate student status
        
        if age < 19 or (is_student and age < 24):
            return False
            
        # Cannot be claimed as a dependent
        if member_data.get('is_dependent', False):
            return False
            
        return True
