"""
Implementation of SeparateFiler for married filing separately.
"""

from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from .base import BaseFiler

class SeparateFiler(BaseFiler):
    """
    Represents a married person filing separately.
    
    This class handles the specific logic for married filing separately,
    including income calculation and eligibility rules.
    """
    
    def __init__(self, filer_id: str, **kwargs):
        """
        Initialize a separate filer.
        
        Args:
            filer_id: Unique identifier for this filer
            **kwargs: Additional filer-specific attributes
        """
        super().__init__(filer_id, **kwargs)
        self.spouse_id = kwargs.get('spouse_id')
    
    def get_filing_status(self) -> str:
        """Return the filing status for this filer."""
        return 'separate'
    
    def calculate_income(self, person_data: pd.DataFrame) -> float:
        """
        Calculate total income for a separate filer.
        
        Args:
            person_data: DataFrame containing person data
            
        Returns:
            float: Total income of this filer only
        """
        if not self.members:
            return 0.0
            
        # For separate filers, only include their own income
        member_id = self.members[0]  # Separate filers have only one member
        if member_id not in person_data.index:
            return 0.0
            
        member_data = person_data.loc[member_id]
        
        # Sum up all income components
        income = 0.0
        for col in ['WAGP', 'SEMP', 'INTP', 'RETP', 'SSP', 'SSIP', 'PAP', 'OIP']:
            if col in person_data.columns:
                income += float(member_data.get(col, 0) or 0)
                
        return income
    
    def is_eligible(self, person_data: pd.DataFrame) -> bool:
        """
        Check if this filer is eligible to file as married filing separately.
        
        Args:
            person_data: DataFrame containing person data
            
        Returns:
            bool: True if eligible, False otherwise
        """
        if len(self.members) != 1:
            return False
            
        member_id = self.members[0]
        if member_id not in person_data.index:
            return False
            
        member_data = person_data.loc[member_id]
        
        # Must be married
        if member_data.get('MAR') != 1:  # 1 = Married, spouse present
            return False
            
        # Cannot be claimed as a dependent
        if member_data.get('is_dependent', False):
            return False
            
        # Must meet age requirements
        age = member_data.get('AGEP', 0)
        if age < 19:  # Basic age requirement
            return False
            
        # If a student, must be at least 24
        is_student = member_data.get('SCHL', 0) >= 16  # Approximate student status
        if is_student and age < 24:
            return False
            
        return True
    
    def should_file_separately(self, person_data: pd.DataFrame) -> bool:
        """
        Determine if this person should file separately from their spouse.
        
        Args:
            person_data: DataFrame containing person data
            
        Returns:
            bool: True if should file separately
        """
        if not self.spouse_id or self.spouse_id not in person_data.index:
            return False
            
        member_id = self.members[0]
        member_data = person_data.loc[member_id]
        spouse_data = person_data.loc[self.spouse_id]
        
        # Get incomes
        member_income = self.calculate_income(pd.DataFrame([member_data]))
        spouse_income = self.calculate_income(pd.DataFrame([spouse_data]))
        
        # Check for large income disparity
        if member_income > 0 and spouse_income > 0:
            ratio = max(member_income, spouse_income) / min(member_income, spouse_income)
            if ratio > 10:  # One earns 10x more than the other
                return True
        
        # One spouse has significant negative income (business losses)
        if (member_income < -5000 and spouse_income > 50000) or \
           (spouse_income < -5000 and member_income > 50000):
            return True
            
        # Different disability status (medical expense deduction reasons)
        if 'DIS' in person_data.columns:
            if member_data.get('DIS', 2) != spouse_data.get('DIS', 2):
                if member_data.get('DIS') == 1 or spouse_data.get('DIS') == 1:
                    return True
                    
        # Different citizenship/immigration status
        if 'CIT' in person_data.columns:
            if member_data.get('CIT', 0) != spouse_data.get('CIT', 0):
                if member_data.get('CIT', 0) >= 4 or spouse_data.get('CIT', 0) >= 4:
                    return True
                    
        # Public assistance income (can affect eligibility)
        pap1 = float(member_data.get('PAP', 0) or 0)
        pap2 = float(spouse_data.get('PAP', 0) or 0)
        if (pap1 > 0 and pap2 == 0) or (pap2 > 0 and pap1 == 0):
            return True
            
        # Student loan considerations
        intp1 = float(member_data.get('INTP', 0) or 0)
        intp2 = float(spouse_data.get('INTP', 0) or 0)
        if (intp1 > 10000 and spouse_income < 30000) or \
           (intp2 > 10000 and member_income < 30000):
            return True
            
        # Self-employment income differences
        semp1 = float(member_data.get('SEMP', 0) or 0)
        semp2 = float(spouse_data.get('SEMP', 0) or 0)
        if (abs(semp1) > 50000 and semp2 == 0) or (abs(semp2) > 50000 and semp1 == 0):
            return True
            
        # Random assignment to reach target percentage (about 8.6% of married couples)
        import random
        random.seed(int(member_id.split('_')[-1]))  # For reproducibility
        if random.random() < 0.02:  # 2% random assignment
            return True
            
        return False
