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
        logger.info(f"Processing household {hh_group['SERIALNO'].iloc[0]} with {len(hh_group)} members")
        
        # Skip empty households
        if hh_group.empty:
            logger.debug("Skipping empty household")
            return []
            
        # Get household data
        hh_id = hh_group['SERIALNO'].iloc[0]
        hh_data = self.hh_df[self.hh_df['SERIALNO'] == hh_id].iloc[0] if not self.hh_df.empty else {}
        
        # Identify all adults (potential filers)
        adults = hh_group[hh_group['AGEP'] >= 18].copy()
        logger.debug(f"Found {len(adults)} adults in household {hh_id}")
        
        # If no adults, skip this household
        if adults.empty:
            logger.debug(f"No adults in household {hh_id}, skipping")
            return []
            
        # First pass: identify all dependents in the household
        logger.debug("Identifying all dependents in household")
        dependents = identify_dependents(hh_group)
        
        # Log the initial dependent assignments
        for filer_id, deps in dependents.items():
            if deps:
                logger.debug(f"Initial dependent assignment - Filer {filer_id} has {len(deps)} dependents")
                for dep_id in deps:
                    dep = hh_group.loc[dep_id]
                    logger.debug(f"  - Dependent {dep_id} (Age: {dep['AGEP']}, REL: {dep.get('RELSHIPP', 'N/A')}, MAR: {dep.get('MAR', 'N/A')})")
        
        # Track which dependents have been claimed
        claimed_dependents = set()
        # Track which adults have been processed in joint filers
        processed_adults = set()
        tax_units = []
        
        # Identify potential joint filers and MFS filers
        print("Identifying potential joint filers and MFS filers")
        joint_filers, mfs_filers = self._identify_joint_filers(adults, hh_group)
        print(f"CONSTRUCTOR: Found {len(joint_filers)} joint filer pairs and {len(mfs_filers)} MFS filer pairs")
        
        # Debug: Print details of identified filers
        for i, (id1, id2) in enumerate(joint_filers):
            print(f"  Joint filer pair {i+1}: {id1} and {id2}")
        for i, (id1, id2) in enumerate(mfs_filers):
            print(f"  MFS filer pair {i+1}: {id1} and {id2}")
            person1 = adults.loc[id1]
            person2 = adults.loc[id2]
            print(f"    {id1}: MAR={person1.get('MAR')}, RELSHIPP={person1.get('RELSHIPP')}, CIT={person1.get('CIT')}, WAGP={person1.get('WAGP')}")
            print(f"    {id2}: MAR={person2.get('MAR')}, RELSHIPP={person2.get('RELSHIPP')}, CIT={person2.get('CIT')}, WAGP={person2.get('WAGP')}")
        
        # Process MFS filers first (they file as single)
        for adult1_id, adult2_id in mfs_filers:
            adult1 = adults.loc[adult1_id]
            adult2 = adults.loc[adult2_id]
            
            logger.info(f"Processing MFS filers: {adult1_id} and {adult2_id}")
            
            # Get dependents for each adult
            deps1 = set(dependents.get(adult1_id, []))
            deps2 = set(dependents.get(adult2_id, []))
            
            # Remove already claimed dependents
            available_deps1 = [d for d in deps1 if d not in claimed_dependents]
            available_deps2 = [d for d in deps2 if d not in claimed_dependents]
            
            logger.debug(f"  {adult1_id} has {len(deps1)} potential dependents, {len(available_deps1)} available")
            logger.debug(f"  {adult2_id} has {len(deps2)} potential dependents, {len(available_deps2)} available")
            
            # Create separate tax units for each MFS filer
            tax_unit1 = self._create_single_filer(adult1, hh_group, hh_data, available_deps1, filing_status='married_separate')
            if tax_unit1:
                logger.info(f"  Created MFS tax unit for {adult1_id} with {len(tax_unit1['dependents'])} dependents")
                tax_units.append(tax_unit1)
                claimed_dependents.update(tax_unit1['dependents'])
                
            tax_unit2 = self._create_single_filer(adult2, hh_group, hh_data, available_deps2, filing_status='married_separate')
            if tax_unit2:
                logger.info(f"  Created MFS tax unit for {adult2_id} with {len(tax_unit2['dependents'])} dependents")
                tax_units.append(tax_unit2)
                claimed_dependents.update(tax_unit2['dependents'])
                
            # Mark both adults as processed
            processed_adults.update([adult1_id, adult2_id])
        
        # Process joint filers
        for adult1_id, adult2_id in joint_filers:
            adult1 = adults.loc[adult1_id]
            adult2 = adults.loc[adult2_id]
            
            logger.info(f"Processing joint filers: {adult1_id} and {adult2_id}")
            
            # Get dependents for this couple (union of both adults' dependents)
            deps1 = set(dependents.get(adult1_id, []))
            deps2 = set(dependents.get(adult2_id, []))
            all_deps = deps1.union(deps2)
            
            # Remove already claimed dependents
            available_deps = [d for d in all_deps if d not in claimed_dependents]
            
            logger.debug(f"  Potential dependents before filtering: {len(all_deps)}")
            logger.debug(f"  Available dependents after filtering: {len(available_deps)}")
            
            # Create joint tax unit
            tax_unit = self._create_joint_filer(adult1, adult2, hh_group, hh_data, available_deps)
            if tax_unit:
                logger.info(f"  Created joint tax unit with {len(tax_unit['dependents'])} dependents")
                tax_units.append(tax_unit)
                claimed_dependents.update(tax_unit['dependents'])
                # Mark both adults as processed
                processed_adults.update([adult1_id, adult2_id])
                logger.debug(f"  Updated claimed dependents: {claimed_dependents}")
        
        # Process remaining adults as single filers
        remaining_adults = adults[~adults.index.isin(processed_adults)]
        logger.debug(f"Processing {len(remaining_adults)} remaining adults as single filers")
        
        for _, adult in remaining_adults.iterrows():
            logger.info(f"Processing single filer: {adult.name}")
            
            # Get dependents for this adult
            adult_deps = set(dependents.get(adult.name, []))
            
            # Remove already claimed dependents
            available_deps = [d for d in adult_deps if d not in claimed_dependents]
            
            logger.debug(f"  Potential dependents before filtering: {len(adult_deps)}")
            logger.debug(f"  Available dependents after filtering: {len(available_deps)}")
            
            # Create single tax unit
            tax_unit = self._create_single_filer(adult, hh_group, hh_data, available_deps)
            if tax_unit:
                logger.info(f"  Created single tax unit with {len(tax_unit['dependents'])} dependents")
                tax_units.append(tax_unit)
                claimed_dependents.update(tax_unit['dependents'])
                logger.debug(f"  Updated claimed dependents: {claimed_dependents}")
        
        # Log any unclaimed dependents
        all_dependents = set()
        for deps in dependents.values():
            all_dependents.update(deps)
        unclaimed = all_dependents - claimed_dependents
        if unclaimed:
            logger.warning(f"Household {hh_id} has {len(unclaimed)} unclaimed dependents: {unclaimed}")
        
        logger.info(f"Completed processing household {hh_id}. Created {len(tax_units)} tax units")
        return tax_units
    
    def _identify_joint_filers(self, adults: pd.DataFrame, hh_members: pd.DataFrame) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        Identify potential joint filers and married filing separately couples within a household.
        
        Args:
            adults: DataFrame of adult household members
            hh_members: All members of the household
            
        Returns:
            Tuple of (joint_filers, mfs_filers) where each is a list of (person_id1, person_id2) tuples
        """
        joint_filers = []
        mfs_filers = []
        processed = set()
        
        # Convert to list of (id, series) for easier iteration
        adult_list = [(idx, row) for idx, row in adults.iterrows()]
        
        logger.debug(f"Checking {len(adult_list)} adults for potential joint/MFS filers")
        
        for i, (id1, person1) in enumerate(adult_list):
            if id1 in processed:
                logger.debug(f"Skipping already processed adult {id1}")
                continue
                
            for j in range(i + 1, len(adult_list)):
                id2, person2 = adult_list[j]
                
                if id2 in processed:
                    logger.debug(f"Skipping already processed adult {id2}")
                    continue
                
                logger.debug(f"Checking if {id1} (MAR={person1.get('MAR')}, RELSHIPP={person1.get('RELSHIPP')}) and {id2} (MAR={person2.get('MAR')}, RELSHIPP={person2.get('RELSHIPP')}) are married")
                
                # First check if they should file as MFS (this covers all married couples)
                mfs = is_married_filing_separately(person1, person2, hh_members)
                if mfs:
                    mfs_filers.append((id1, id2))
                    logger.debug(f"  Married couple {id1} and {id2} will file separately")
                    # Mark both as processed
                    processed.update([id1, id2])
                    break
                
                # If not MFS, check if they can file jointly
                elif is_married_filing_jointly(person1, person2, hh_members):
                    joint_filers.append((id1, id2))
                    logger.debug(f"  Identified joint filers: {id1} and {id2}")
                    # Mark both as processed
                    processed.update([id1, id2])
                    break
        
        return joint_filers, mfs_filers
    
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
        # 1. Check if either spouse is a non-resident alien
        if adult1.get('CIT', 0) == 5 or adult2.get('CIT', 0) == 5:  # 5 = Not a citizen
            logger.debug("One spouse is a nonresident alien, using Married Filing Separately status")
            return True
            
        # 2. Check if spouses have significantly different incomes
        income1 = self._calculate_income(adult1)
        income2 = self._calculate_income(adult2)
        
        if income1 > 0 and income2 > 0:
            ratio = max(income1, income2) / min(income1, income2)
            if ratio > 10:
                logger.debug(f"Significant income difference (ratio: {ratio:.1f}), considering separate filing")
                return True
                
        # 3. Check for significant self-employment income differences
        semp1 = float(adult1.get('SEMP', 0) or 0)
        semp2 = float(adult2.get('SEMP', 0) or 0)
        
        if (abs(semp1) > 50000 and semp2 == 0) or (abs(semp2) > 50000 and semp1 == 0):
            logger.debug("Significant self-employment income difference, considering separate filing")
            return True
            
        # 4. Randomly assign some couples to file separately to match real-world distribution
        import random
        random.seed(int(adult1.get('SERIALNO', '0')[-4:]) + int(adult1.get('SPORDER', 0)))
        if random.random() < 0.02:  # Approximately 2% of couples file separately
            logger.debug("Randomly assigned to file separately")
            return True
            
        return False
        
    def _calculate_income(self, person: pd.Series) -> float:
        """
        Calculate total income for a person, adjusted by ADJINC factor.
        
        Args:
            person: Series containing person's data with income fields
            
        Returns:
            float: Total income adjusted by ADJINC factor
        """
        # Get ADJINC factor (default to 1.0 if not present)
        # ADJINC values in PUMS data are already the adjustment factors (around 1.0-1.2)
        adjinc = float(person.get('ADJINC', 1.0))
        
        # Use PINCP (total person income) which already includes all income sources
        # This avoids double-counting that would occur if we summed individual components
        total_income = float(person.get('PINCP', 0) or 0)
        
        # Apply ADJINC adjustment
        total_income *= adjinc
        
        return total_income

    def _create_joint_filer(self, adult1: pd.Series, adult2: pd.Series, 
                           hh_members: pd.DataFrame, hh_data: pd.Series, 
                           available_deps: List[str] = None) -> Optional[dict]:
        """
        Create a tax unit for a joint filer.
        
        Args:
            adult1: First adult in the joint filing couple
            adult2: Second adult in the joint filing couple
            hh_members: All members of the household
            hh_data: Household-level data
            available_deps: List of available dependent person_ids that can be claimed
            
        Returns:
            Dictionary containing tax unit information or None if not valid
        """
        if available_deps is None:
            available_deps = []
        
        logger.debug(f"Creating joint filer tax unit for {adult1.name} and {adult2.name} with {len(available_deps)} available dependents")
        
        # Filter available dependents to only include those actually in the household
        valid_dependents = [d for d in available_deps if d in hh_members.index]
        
        # Combine all members for income calculation
        members_to_include = [adult1, adult2]
        if valid_dependents:
            for dep_id in valid_dependents:
                members_to_include.append(hh_members.loc[dep_id])
        
        # Create DataFrame from Series objects
        income_df = pd.DataFrame(members_to_include)
        income = calculate_tax_unit_income(income_df)
        
        # Create tax unit
        tax_unit = {
            'filer_id': f"{hh_data['SERIALNO']}_joint_{adult1.name}_{adult2.name}",
            'SERIALNO': hh_data['SERIALNO'],
            'filing_status': 'joint',
            'primary_filer_id': adult1.name,
            'secondary_filer_id': adult2.name,
            'income': income,
            'num_dependents': len(valid_dependents),
            'dependents': valid_dependents,
            'hh_id': adult1['SERIALNO']
        }
        
        logger.debug(f"Created joint tax unit: {tax_unit}")
        return tax_unit

    def _create_single_filer(self, adult: pd.Series, hh_members: pd.DataFrame, 
                           hh_data: pd.Series, available_deps: List[str] = None,
                           filing_status: str = None) -> Optional[dict]:
        """
        Create a tax unit for a single filer.
        
        Args:
            adult: The adult who is the primary filer
            hh_members: All members of the household
            hh_data: Household-level data
            available_deps: List of available dependent person_ids that can be claimed
            
        Returns:
            Dictionary containing tax unit information or None if not valid
        """
        if available_deps is None:
            available_deps = []
            
        logger.debug(f"Creating single filer tax unit for {adult.name} with {len(available_deps)} available dependents")
        
        # Filter available dependents to only include those actually in the household
        valid_dependents = [d for d in available_deps if d in hh_members.index]
        
        # Determine filing status if not provided
        if filing_status is None:
            filing_status = 'single'
            
            # Check if this person is married but filing separately
            is_married_separate = False
            for _, other_adult in hh_members.iterrows():
                if other_adult.name != adult.name and other_adult.get('AGEP', 0) >= 18:
                    if is_married_filing_separately(adult, other_adult, hh_members):
                        filing_status = 'married_filing_separate'
                        break
            
            # Check for Head of Household status if not married filing separately
            if filing_status != 'married_filing_separate' and valid_dependents:
                person_data = hh_members.copy()
                person_data['SERIALNO'] = hh_data.get('SERIALNO', '')
                
                if is_head_of_household(adult, person_data):
                    filing_status = 'head_of_household'
                    logger.debug(f"Person {adult.name} qualifies as Head of Household")
        
        # Calculate income (include dependents in the calculation)
        members_to_include = [adult]
        if valid_dependents:
            for dep_id in valid_dependents:
                members_to_include.append(hh_members.loc[dep_id])
        
        # Create DataFrame from Series objects
        income_df = pd.DataFrame(members_to_include)
        income = calculate_tax_unit_income(income_df)
        
        # Create tax unit
        tax_unit = {
            'filer_id': f"{hh_data.get('SERIALNO', '')}_{filing_status}_{adult.name}",
            'filing_status': filing_status,
            'income': income,
            'num_dependents': len(valid_dependents),
            'dependents': valid_dependents,
            'hh_id': adult['SERIALNO']
        }
        
        logger.debug(f"Created tax unit: {tax_unit}")
        return tax_unit
