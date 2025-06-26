"""
Base class for all tax filer types.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import pandas as pd

class BaseFiler(ABC):
    """
    Abstract base class for all tax filer types.
    
    This class defines the common interface and functionality
    that all tax filer types must implement.
    """
    
    def __init__(self, filer_id: str, **kwargs):
        """
        Initialize a new tax filer.
        
        Args:
            filer_id: Unique identifier for this filer
            **kwargs: Additional filer-specific attributes
        """
        self.filer_id = filer_id
        self.filing_status = self.get_filing_status()
        self.members = []  # List of person_ids in this filing unit
        self.attributes = kwargs
    
    @abstractmethod
    def get_filing_status(self) -> str:
        """
        Return the filing status for this filer.
        
        Returns:
            str: Filing status (e.g., 'single', 'joint', etc.)
        """
        pass
    
    @abstractmethod
    def calculate_income(self, person_data: pd.DataFrame) -> float:
        """
        Calculate the total income for this filer.
        
        Args:
            person_data: DataFrame containing person data
            
        Returns:
            float: Total income
        """
        pass
    
    def add_member(self, person_id: str) -> None:
        """
        Add a person to this filing unit.
        
        Args:
            person_id: ID of the person to add
        """
        if person_id not in self.members:
            self.members.append(person_id)
    
    def to_dict(self) -> Dict:
        """
        Convert the filer to a dictionary representation.
        
        Returns:
            dict: Dictionary representation of the filer
        """
        return {
            'filer_id': self.filer_id,
            'filing_status': self.filing_status,
            'members': self.members,
            **self.attributes
        }
