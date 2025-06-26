"""
Implementation of HeadOfHouseholdFiler for head of household filers.
"""

from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from .base import BaseFiler

class HeadOfHouseholdFiler(BaseFiler):
    """
    Represents a head of household filer.
    
    This class handles the specific logic for head of household filers,
    including income calculation and eligibility rules.
    """
    
    def get_filing_status(self) -> str:
        """Return the filing status for this filer."""
        return 'head_of_household'
    
    def calculate_income(self, person_data: pd.DataFrame) -> float:
        """
        Calculate total income for a head of household.
        
        Args:
            person_data: DataFrame containing person data
            
        Returns:
            float: Total income of this filer and dependents
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
        Check if this filer is eligible to file as head of household.
        
        Args:
            person_data: DataFrame containing person data
            
        Returns:
            bool: True if eligible, False otherwise
        """
        if not self.members:
            return False
            
        # Must have at least one qualifying person (usually a child)
        if not self._has_qualifying_person(person_data):
            return False
            
        # Must be unmarried or considered unmarried on the last day of the year
        if not self._is_unmarried(person_data):
            return False
            
        # Must have paid more than half the cost of keeping up a home
        if not self._paid_half_home_cost(person_data):
            return False
            
        return True
    
    def _has_qualifying_person(self, person_data: pd.DataFrame) -> bool:
        """
        Check if the filer has at least one qualifying person.
        
        Args:
            person_data: DataFrame containing person data
            
        Returns:
            bool: True if has qualifying person
        """
        if not self.members:
            return False
            
        filer_id = self.members[0]  # First member is the filer
        if filer_id not in person_data.index:
            return False
            
        filer_data = person_data.loc[filer_id]
        
        # Check if any other household members are qualifying children or relatives
        household_id = filer_data.get('SERIALNO')
        if not household_id:
            return False
            
        # Get all people in the same household
        household = person_data[person_data['SERIALNO'] == household_id]
        
        # Check each household member (excluding self)
        for _, member in household[household.index != filer_id].iterrows():
            if self._is_qualifying_child(member, filer_data, person_data) or \
               self._is_qualifying_relative(member, filer_data, person_data):
                return True
                
        return False
    
    def _is_qualifying_child(
        self, 
        member: pd.Series, 
        filer: pd.Series,
        person_data: pd.DataFrame
    ) -> bool:
        """
        Check if a person is a qualifying child of the filer.
        
        Args:
            member: The potential qualifying child
            filer: The potential head of household
            person_data: Full person data for reference
            
        Returns:
            bool: True if qualifies as a child
        """
        # Relationship to filer (child, stepchild, foster child, etc.)
        rel = str(member.get('RELSHIPP', ''))
        if rel not in ['00', '02', '03', '04', '05']:  # Not a child relationship
            return False
            
        # Age test
        age = member.get('AGEP', 0)
        if age >= 19 and (age >= 24 or member.get('SCHL', 0) < 16):
            return False
            
        # Residency test - must live with filer for more than half the year
        # Assuming same household means they live together
        
        # Support test - cannot provide more than half of their own support
        # This is a simplification - in reality would need more detailed data
        
        # Not filing a joint return (unless only to claim refund)
        if member.get('MAR') == 1:  # Married
            return False
            
        return True
    
    def _is_qualifying_relative(
        self, 
        member: pd.Series, 
        filer: pd.Series,
        person_data: pd.DataFrame
    ) -> bool:
        """
        Check if a person is a qualifying relative of the filer.
        
        Args:
            member: The potential qualifying relative
            filer: The potential head of household
            person_data: Full person data for reference
            
        Returns:
            bool: True if qualifies as a relative
        """
        # Can't be a qualifying child of the filer or anyone else
        if self._is_qualifying_child(member, filer, person_data):
            return False
            
        # Relationship test - must be a relative or have lived with filer all year
        rel = str(member.get('RELSHIPP', ''))
        if rel not in ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17']:
            return False
            
        # Gross income test
        income = self.calculate_income(pd.DataFrame([member]))
        if income >= 4300:  # 2023 amount, should be configurable
            return False
            
        # Support test - filer must provide more than half of support
        # This is a simplification - would need more detailed data
        
        # Not a qualifying child of another taxpayer
        # Would need to check against all other potential filers
        
        # Not filing a joint return (unless only to claim refund)
        if member.get('MAR') == 1:  # Married
            return False
            
        return True
    
    def _is_unmarried(self, person_data: pd.DataFrame) -> bool:
        """
        Check if the filer is considered unmarried.
        
        Args:
            person_data: DataFrame containing person data
            
        Returns:
            bool: True if considered unmarried
        """
        if not self.members:
            return False
            
        filer_id = self.members[0]
        if filer_id not in person_data.index:
            return False
            
        filer_data = person_data.loc[filer_id]
        
        # Considered unmarried if:
        # 1. Single, divorced, or legally separated
        if filer_data.get('MAR') in [3, 4, 5, 6]:  # Separated, divorced, widowed, never married
            return True
            
        # 2. Married but living apart from spouse for last 6 months of year
        # This is a simplification - would need more detailed data
        
        return False
    
    def _paid_half_home_cost(self, person_data: pd.DataFrame) -> bool:
        """
        Check if the filer paid more than half the cost of keeping up a home.
        
        Args:
            person_data: DataFrame containing person data
            
        Returns:
            bool: True if paid more than half
        """
        if not self.members:
            return False
            
        filer_id = self.members[0]
        if filer_id not in person_data.index:
            return False
            
        filer_data = person_data.loc[filer_id]
        household_id = filer_data.get('SERIALNO')
        
        # Get household costs (simplified)
        # In reality would need rent/mortgage, utilities, insurance, etc.
        household = person_data[person_data['SERIALNO'] == household_id]
        
        # Simplified: Assume income is proxy for contribution
        filer_income = self.calculate_income(pd.DataFrame([filer_data]))
        total_income = self.calculate_income(household)
        
        return filer_income > (total_income - filer_income)  # More than half
