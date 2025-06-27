"""
Tax Unit Constructor

This module provides the main TaxUnitConstructor class that uses the modular components
to construct tax units from PUMS data.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

# Import modular components
from .status import is_married_filing_jointly, is_married_filing_separately, is_head_of_household
from .income import calculate_tax_unit_income
from .dependencies import identify_dependents
from .utils import setup_logging, validate_input_data, create_person_id

# Configure logging
logger = logging.getLogger(__name__)


class TaxUnitConstructor:
    """
    Main class for constructing tax units from PUMS data using modular components.
    
    This class orchestrates the tax unit construction process by leveraging
    specialized modules for different aspects of the process.
    """
    
    def __init__(self, person_df: pd.DataFrame, hh_df: pd.DataFrame):
        """
        Initialize the TaxUnitConstructor with person and household data.
        
        Args:
            person_df: DataFrame containing person-level PUMS data
            hh_df: DataFrame containing household-level PUMS data
        """
        self.person_df = person_df.copy()
        self.hh_df = hh_df.copy()
        self.tax_units = None
        
        # Setup logging
        setup_logging()
        
        # Validate inputs
        is_valid, error_msg = validate_input_data(self.person_df, self.hh_df)
        if not is_valid:
            raise ValueError(f"Invalid input data: {error_msg}")
        
        # Preprocess data
        self._preprocess_data()
    
    def _preprocess_data(self) -> None:
        """Preprocess the input data for tax unit construction."""
        # Create person_id from SERIALNO and SPORDER
        self.person_df['person_id'] = create_person_id(self.person_df)
        
        # Set person_id as the index
        self.person_df.set_index('person_id', inplace=True)
        
        # Add is_adult flag
        self.person_df['is_adult'] = self.person_df['AGEP'] >= 18
        self.person_df['is_child'] = ~self.person_df['is_adult']
        
        # Merge household data
        self.person_df = self.person_df.merge(
            self.hh_df[['SERIALNO', 'HINCP']], 
            on='SERIALNO', 
            how='left'
        )
    
    def create_rule_based_units(self) -> pd.DataFrame:
        """
        Create tax units using rule-based approach.
        
        Returns:
            DataFrame containing the constructed tax units
        """
        logger.info("Creating tax units using rule-based approach...")
        
        # Placeholder for tax units
        tax_units = []
        
        # Process each household
        for hh_id, hh_group in self.person_df.groupby('SERIALNO'):
            # Skip households with no adults
            if not hh_group['is_adult'].any():
                continue
                
            # Process household members into tax units
            household_units = self._process_household(hh_group)
            tax_units.extend(household_units)
        
        # Convert to DataFrame
        if tax_units:
            self.tax_units = pd.DataFrame(tax_units)
            return self.tax_units
        return pd.DataFrame()
    
    def _process_household(self, hh_group: pd.DataFrame) -> List[dict]:
        """
        Process a single household and return a list of tax units.
        
        Args:
            hh_group: DataFrame containing all persons in a household
            
        Returns:
            List of tax unit dictionaries
        """
        tax_units = []
        processed = set()  # Track processed person IDs
        
        # Get all adults in the household
        adults = hh_group[hh_group['is_adult']].copy()
        
        # First, identify and process joint filers
        joint_filers = self._identify_joint_filers(adults, hh_group)
        for adult1_id, adult2_id in joint_filers:
            if adult1_id not in processed and adult2_id not in processed:
                adult1 = adults.loc[adult1_id]
                adult2 = adults.loc[adult2_id]
                
                # Check if they should file separately
                file_separately = self._should_file_separately(adult1, adult2, hh_group)
                
                if file_separately:
                    # Create separate tax units for each
                    tax_unit1 = self._create_single_filer(adult1, hh_group, hh_group.iloc[0])
                    tax_unit2 = self._create_single_filer(adult2, hh_group, hh_group.iloc[0])
                    
                    if tax_unit1:
                        tax_units.append(tax_unit1)
                        processed.add(adult1_id)
                    if tax_unit2:
                        tax_units.append(tax_unit2)
                        processed.add(adult2_id)
                else:
                    # Create joint tax unit
                    tax_unit = self._create_joint_filer(adult1, adult2, hh_group, hh_group.iloc[0])
                    if tax_unit:
                        tax_units.append(tax_unit)
                        processed.add(adult1_id)
                        processed.add(adult2_id)
        
        # Process remaining adults as single filers
        remaining_adults = adults[~adults.index.isin(processed)]
        for _, adult in remaining_adults.iterrows():
            tax_unit = self._create_single_filer(adult, hh_group, hh_group.iloc[0])
            if tax_unit:
                tax_units.append(tax_unit)
                processed.add(adult.name)
        
        return tax_units
    
    def _identify_joint_filers(self, adults: pd.DataFrame, hh_members: pd.DataFrame) -> List[Tuple[str, str]]:
        """
        Identify potential joint filers within a household.
        
        Args:
            adults: DataFrame of adult household members
            hh_members: All members of the household
            
        Returns:
            List of (person_id1, person_id2) tuples for potential joint filers
        """
        joint_filers = []
        processed = set()
        
        # Convert to list of (id, series) for easier iteration
        adult_list = [(idx, row) for idx, row in adults.iterrows()]
        
        for i, (id1, person1) in enumerate(adult_list):
            if id1 in processed:
                continue
                
            for j in range(i + 1, len(adult_list)):
                id2, person2 = adult_list[j]
                
                if id2 in processed:
                    continue
                
                # Use the status module to check for joint filing status
                if is_married_filing_jointly(person1, person2, hh_members):
                    joint_filers.append((id1, id2))
                    processed.update([id1, id2])
                    break
        
        return joint_filers
    
    def _should_file_separately(self, adult1: pd.Series, adult2: pd.Series, 
                              hh_members: pd.DataFrame) -> bool:
        """
        Determine if a married couple should file separately.
        
        Args:
            adult1: First spouse
            adult2: Second spouse
            hh_members: All household members
            
        Returns:
            bool: True if should file separately, False otherwise
        """
        return is_married_filing_separately(adult1, adult2, hh_members)
    
    def _create_single_filer(self, adult: pd.Series, hh_members: pd.DataFrame, 
                           hh_data: pd.Series) -> Optional[dict]:
        """
        Create a tax unit for a single filer.
        
        Args:
            adult: The adult who is the primary filer
            hh_members: All members of the household
            hh_data: Household-level data
            
        Returns:
            Dictionary containing tax unit information or None if not valid
        """
        # Identify dependents
        dependents = identify_dependents(hh_members)
        adult_dependents = dependents.get(adult.name, [])
        
        # Check for Head of Household status
        is_hoh = is_head_of_household(adult, hh_members)
        
        # Calculate income
        income = calculate_tax_unit_income(pd.concat([pd.DataFrame([adult]), 
                                                    hh_members.loc[adult_dependents]]))
        
        # Create tax unit
        tax_unit = {
            'filer_id': adult.name,
            'filing_status': 'head_of_household' if is_hoh else 'single',
            'income': income,
            'num_dependents': len(adult_dependents),
            'dependents': adult_dependents,
            'hh_id': adult['SERIALNO']
        }
        
        return tax_unit
    
    def _create_joint_filer(self, adult1: pd.Series, adult2: pd.Series, 
                           hh_members: pd.DataFrame, hh_data: pd.Series) -> Optional[dict]:
        """
        Create a tax unit for a joint filer.
        
        Args:
            adult1: First adult in the joint filing couple
            adult2: Second adult in the joint filing couple
            hh_members: All members of the household
            hh_data: Household-level data
            
        Returns:
            Dictionary containing tax unit information or None if not valid
        """
        # Identify dependents for both adults
        dependents = identify_dependents(hh_members)
        adult1_deps = dependents.get(adult1.name, [])
        adult2_deps = dependents.get(adult2.name, [])
        
        # Combine dependents, removing any duplicates
        all_dependents = list(set(adult1_deps + adult2_deps))
        
        # Calculate combined income
        members = pd.concat([
            pd.DataFrame([adult1, adult2]),
            hh_members.loc[all_dependents]
        ])
        income = calculate_tax_unit_income(members)
        
        # Create tax unit
        tax_unit = {
            'filer_id': f"{adult1.name}_{adult2.name}",
            'filing_status': 'joint',
            'primary_filer_id': adult1.name,
            'secondary_filer_id': adult2.name,
            'income': income,
            'num_dependents': len(all_dependents),
            'dependents': all_dependents,
            'hh_id': adult1['SERIALNO']
        }
        
        return tax_unit
