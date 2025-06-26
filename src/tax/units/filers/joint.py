"""
Implementation of JointFiler for married filing jointly.
"""

from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from .base import BaseFiler

class JointFiler(BaseFiler):
    """
    Represents a married couple filing jointly.
    
    This class handles the specific logic for joint filers,
    including income calculation and eligibility rules.
    """
    
    def __init__(self, filer_id: str, primary_id: str, secondary_id: str, **kwargs):
        """
        Initialize a joint filer with primary and secondary filers.
        
        Args:
            filer_id: Unique identifier for this filer
            primary_id: ID of the primary filer
            secondary_id: ID of the secondary (spouse) filer
            **kwargs: Additional attributes
        """
        super().__init__(filer_id, **kwargs)
        self.primary_id = primary_id
        self.secondary_id = secondary_id
        self.members = [primary_id, secondary_id]
    
    def get_filing_status(self) -> str:
        """Return the filing status for this filer."""
        return 'joint'
    
    def calculate_income(self, person_data: pd.DataFrame) -> float:
        """
        Calculate total income for a joint filer.
        
        Args:
            person_data: DataFrame containing person data
            
        Returns:
            float: Combined income of both spouses
        """
        if not self.members:
            return 0.0
            
        total_income = 0.0
        
        # Calculate income for each member
        for member_id in self.members:
            if member_id not in person_data.index:
                continue
                
            member_data = person_data.loc[member_id]
            
            # Sum up all income components
            member_income = 0.0
            for col in ['WAGP', 'SEMP', 'INTP', 'RETP', 'SSP', 'SSIP', 'PAP', 'OIP']:
                if col in person_data.columns:
                    member_income += float(member_data.get(col, 0) or 0)
            
            total_income += member_income
                
        return total_income
    
    def is_eligible(self, person_data: pd.DataFrame) -> bool:
        """
        Check if this filer is eligible to file jointly.
        
        Args:
            person_data: DataFrame containing person data
            
        Returns:
            bool: True if eligible, False otherwise
        """
        # Must have exactly two members
        if len(self.members) != 2:
            return False
            
        # Both members must exist in person_data
        if not all(member_id in person_data.index for member_id in self.members):
            return False
            
        primary = person_data.loc[self.primary_id]
        secondary = person_data.loc[self.secondary_id]
        
        # Both must be married to each other
        if not self._are_married(primary, secondary):
            return False
            
        # Neither can be claimed as a dependent
        if primary.get('is_dependent', False) or secondary.get('is_dependent', False):
            return False
            
        return True
    
    def _are_married(self, person1: pd.Series, person2: pd.Series) -> bool:
        """
        Check if two people are married to each other.
        
        Args:
            person1: First person's data
            person2: Second person's data
            
        Returns:
            bool: True if married to each other
        """
        # Both must be marked as married
        if person1.get('MAR') != 1 or person2.get('MAR') != 1:
            return False
            
        # Check relationship codes (simplified)
        rel1 = str(person1.get('RELSHIPP', ''))
        rel2 = str(person2.get('RELSHIPP', ''))
        
        # Should be reference person (20) and spouse (21) or vice versa
        return (rel1 == '20' and rel2 == '21') or (rel1 == '21' and rel2 == '20')
