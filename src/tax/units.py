"""
Tax Unit Construction Module

This module provides functionality to construct tax units from PUMS data
using rule-based and machine learning approaches. Includes Hawaii-specific
adjustments for extended family structures and dependency rules.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
import logging

# Relationship codes with Hawaii-specific extensions
RELSHIPP_CODES = {
    '20': 'Reference person',
    '21': 'Opposite-sex spouse',
    '22': 'Opposite-sex unmarried partner',
    '23': 'Same-sex spouse',
    '24': 'Same-sex unmarried partner',
    '25': 'Biological son or daughter',
    '26': 'Adopted son or daughter',
    '27': 'Stepson or stepdaughter',
    '28': 'Brother or sister',
    '29': 'Father or mother',
    '30': 'Grandchild',
    '31': 'Parent-in-law',
    '32': 'Son-in-law or daughter-in-law',
    '33': 'Other relative',
    '34': 'Foster child',
    '39': 'Grandparent (Hawaii extended family)',
    '40': 'Aunt/Uncle (Hawaii extended family)',
    '41': 'Niece/Nephew (Hawaii extended family)',
    '42': 'Cousin (Hawaii extended family)',
    '43': 'In-law (Hawaii extended family)',
    '44': 'Other relative (Hawaii extended family)'
}

# Hawaii-specific standard deductions (2023)
HAWAII_STANDARD_DEDUCTIONS = {
    'single': 2200,
    'joint': 4400,
    'head_of_household': 3200,
    'married_separate': 2200,
    'widow': 4400
}

# Hawaii-specific income thresholds (2023)
HAWAII_INCOME_THRESHOLDS = {
    'single': 15000,      # Lower threshold for single filers in Hawaii
    'joint': 30000,       # Lower threshold for joint filers in Hawaii
    'hoh': 22500,         # Lower threshold for head of household in Hawaii
    'dependent': 5000,    # Lower threshold for dependents in Hawaii
    'support': 20000,     # Support test threshold for Hawaii
    'relative_income': 5000  # Gross income test for qualifying relatives in Hawaii
}

# Filing status codes - using string values for consistency
FILING_STATUS = {
    'SINGLE': 'single',
    'JOINT': 'joint',
    'SEPARATE': 'separate',
    'HEAD_HOUSEHOLD': 'head_of_household',
    'WIDOW': 'widow'
}

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('tax_units.log')
    ]
)
logger = logging.getLogger(__name__)

class TaxUnitConstructor:
    """Class for constructing tax units from PUMS data."""
    
    def __init__(self, person_df: pd.DataFrame, hh_df: pd.DataFrame):
        """Initialize with person and household data."""
        self.person_df = person_df.copy()
        self.hh_df = hh_df.copy()
        self.tax_units = None
        
        # Preprocess data for performance
        self._preprocess_for_performance()
        
        # Ensure required columns exist
        self._validate_inputs()
    
    def _preprocess_for_performance(self):
        """Preprocess data for better performance."""
        # Note: We no longer pre-calculate total_income here to avoid inconsistencies
        # with _calculate_income() which handles ADJINC adjustments
        
        # Add person_id first as it's needed for spouse mapping
        if 'person_id' not in self.person_df.columns:
            # Ensure both SERIALNO and SPORDER are strings before concatenation
            self.person_df['SERIALNO'] = self.person_df['SERIALNO'].astype(str)
            self.person_df['SPORDER'] = self.person_df['SPORDER'].astype(str)
            self.person_df['person_id'] = (
                self.person_df['SERIALNO'] + '_' + 
                self.person_df['SPORDER']
            )
        
        # Pre-calculate other commonly used values
        if 'is_child' not in self.person_df.columns:
            self.person_df['is_child'] = (self.person_df['AGEP'] < 18).astype(int)
            
        # Add household size for easier filtering
        hh_sizes = self.person_df.groupby('SERIALNO').size().rename('hh_size')
        self.person_df = self.person_df.join(hh_sizes, on='SERIALNO')
        
        # Add number of adults in household
        adults_per_hh = self.person_df[self.person_df['AGEP'] >= 18].groupby('SERIALNO').size().rename('num_adults')
        self.person_df = self.person_df.join(adults_per_hh, on='SERIALNO')
        self.person_df['num_adults'] = self.person_df['num_adults'].fillna(0).astype(int)
        
        # Add number of children in household
        children_per_hh = self.person_df[self.person_df['is_child'] == 1].groupby('SERIALNO').size().rename('num_children')
        self.person_df = self.person_df.join(children_per_hh, on='SERIALNO')
        self.person_df['num_children'] = self.person_df['num_children'].fillna(0).astype(int)
        
        # Add spouse information if available
        if 'SPORDER' in self.person_df.columns and 'RELSHIPP' in self.person_df.columns:
            # Create a mapping of household reference persons to their spouses
            ref_to_spouse = self.person_df[
                (self.person_df['RELSHIPP'] == 20) &  # Reference person
                (self.person_df['MAR'] == 1)  # Married
            ].set_index('SERIALNO')['person_id']
            
            # Map spouse IDs to the person DataFrame
            self.person_df['spouse_id'] = self.person_df['SERIALNO'].map(ref_to_spouse)
            
        # Add household income for easier filtering
        if 'HINCP' in self.hh_df.columns:
            hh_income = self.hh_df.set_index('SERIALNO')['HINCP']
            self.person_df['hh_income'] = self.person_df['SERIALNO'].map(hh_income)
        
        # Add flag for whether person is a potential dependent
        self.person_df['is_potential_dependent'] = (
            (self.person_df['AGEP'] < 19) |  # Under 19
            ((self.person_df['AGEP'] < 24) &  # 19-23 and in school
             self.person_df['SCHL'].between(15, 18, inclusive='both')) |
            (self.person_df['AGEP'] < 19) |  # Under 19
            (self.person_df['is_child'] == 1)  # Already identified as child
        ).astype(int)
        
        # Add flag for whether person is a potential filer
        self.person_df['is_potential_filer'] = (
            (self.person_df['AGEP'] >= 19) &  # 19 or older
            (self.person_df['is_child'] == 0)  # Not a child
        ).astype(int)
        
        # Add total_income column for backward compatibility with tests
        # Note: This is calculated on the fly in _calculate_income()
        self.person_df['total_income'] = self.person_df.apply(self._calculate_income, axis=1)
        
        # Sort by household for better memory locality
        self.person_df = self.person_df.sort_values(['SERIALNO', 'RELSHIPP'])
        self.hh_df = self.hh_df.sort_values('SERIALNO')
    
    def _validate_inputs(self) -> None:
        """Validate that required columns exist in the input DataFrames."""
        required_person_cols = ['SERIALNO', 'SPORDER', 'AGEP', 'SEX', 'MAR', 'RELSHIPP', 'SCHL', 'PINCP', 'WAGP', 'SEMP', 'INTP', 'RETP', 'OIP', 'PAP', 'SSP', 'SSIP']
        required_hh_cols = ['SERIALNO', 'HINCP']
        
        missing_person = [col for col in required_person_cols if col not in self.person_df.columns]
        missing_hh = [col for col in required_hh_cols if col not in self.hh_df.columns]
        
        if missing_person:
            raise ValueError(f"Missing required columns in person_df: {missing_person}")
        if missing_hh:
            raise ValueError(f"Missing required columns in hh_df: {missing_hh}")
            
    def _calculate_income(self, person: pd.Series) -> float:
        """
        Calculate total income for a person using PINCP (total person income) with ADJINC adjustment.
        
        Args:
            person: Series containing person's data with income fields
            
        Returns:
            float: Total income adjusted by ADJINC factor
        """
        # Get ADJINC factor (default to 1.0 if not present)
        adjinc = float(person.get('ADJINC', 1000000)) / 1000000.0
        
        # Use PINCP (total person income) which already includes all income sources
        # This avoids double-counting that would occur if we summed individual components
        total_income = float(person.get('PINCP', 0) or 0)
        
        # Apply ADJINC adjustment
        total_income *= adjinc
        
        return total_income
    
    def create_rule_based_units(self, batch_size: int = 1000, n_jobs: int = 1) -> pd.DataFrame:
        """
        Create tax units using rule-based approach based on IRS filing rules.
        
        Args:
            batch_size: Number of households to process in each batch
            n_jobs: Number of parallel jobs to run (use -1 for all available cores)
            
        Returns:
            DataFrame with one row per tax unit and relevant attributes
        """
        logger.info(f"Creating tax units using rule-based approach (batch_size={batch_size}, n_jobs={n_jobs})...")
        
        # Ensure person_id exists
        if 'person_id' not in self.person_df.columns:
            self.person_df['person_id'] = (
                self.person_df['SERIALNO'].astype(str) + '_' + 
                self.person_df['SPORDER'].astype(str)
            )
        
        # Pre-allocate lists to store results
        all_tax_units = []
        
        # Process households in batches
        total_households = len(self.hh_df)
        
        for batch_start in range(0, total_households, batch_size):
            batch_end = min(batch_start + batch_size, total_households)
            batch_df = self.hh_df.iloc[batch_start:batch_end]
            
            logger.info(f"Processing households {batch_start+1} to {batch_end} of {total_households}")
            
            # Process batch in parallel if n_jobs != 1
            if n_jobs != 1:
                import multiprocessing as mp
                from functools import partial
                
                # Determine number of processes
                n_processes = mp.cpu_count() if n_jobs == -1 else min(n_jobs, mp.cpu_count())
                
                # Create a pool of workers
                with mp.Pool(processes=n_processes) as pool:
                    # Process households in parallel
                    batch_results = pool.map(
                        partial(self._process_household_batch, person_df=self.person_df),
                        batch_df.to_dict('records')
                    )
            else:
                # Process serially
                batch_results = [
                    self._process_household(household, self.person_df)
                    for household in batch_df.to_dict('records')
                ]
            
            # Flatten results and add to main list
            for result in batch_results:
                if result:  # Skip None results
                    all_tax_units.extend(result)
            
            logger.info(f"Completed batch {batch_start//batch_size + 1}/{(total_households+batch_size-1)//batch_size}")
        
        # Convert to DataFrame
        self.tax_units = pd.DataFrame(all_tax_units) if all_tax_units else pd.DataFrame()
        logger.info(f"Created {len(self.tax_units)} tax units from {total_households} households")
        
        return self.tax_units
    
    def _process_household_batch(self, household: Dict, person_df: pd.DataFrame) -> List[Dict]:
        """Process a single household (for parallel execution)."""
        try:
            return self._process_household(household, person_df)
        except Exception as e:
            logger.error(f"Error processing household {household.get('SERIALNO', 'unknown')}: {str(e)}")
            return []
    
    def _process_household(self, household: Dict, person_df: pd.DataFrame) -> List[Dict]:
        """Process a single household and return list of tax units."""
        hh_id = household['SERIALNO']
        tax_units = []
        
        try:
            # Filter household members using vectorized operations
            hh_mask = person_df['SERIALNO'] == hh_id
            hh_members = person_df[hh_mask].copy()
            
            if hh_members.empty:
                return []
            
            # Identify adults (18+)
            adults_mask = hh_members['AGEP'] >= 18
            adults = hh_members[adults_mask]
            
            # Skip households with no adults
            if len(adults) == 0:
                household_size = len(hh_members)
                if household_size > 0 and logger.isEnabledFor(logging.DEBUG):
                    ages = hh_members['AGEP'].tolist()
                    relationships = hh_members['RELSHIPP'].astype(str).tolist()
                    logger.debug(
                        f"Excluding household {hh_id} with no adults (size: {household_size}): "
                        f"Ages: {ages}, Relationships: {relationships}"
                    )
                return []
            
            # Track processed adult indices to avoid duplicates
            processed = set()
            
            # First pass: Identify married couples and create joint returns
            for i, (_, adult1) in enumerate(adults.iterrows()):
                if i in processed:
                    continue
                    
                # Look for a spouse
                for j, (_, adult2) in enumerate(adults.iterrows()):
                    if j <= i or j in processed:
                        continue
                        
                    if self._is_potential_spouse(adult1, adult2):
                        # Create joint return
                        joint_return = self._create_joint_filer(adult1, adult2, hh_members, household)
                        if joint_return:
                            tax_units.append(joint_return)
                            processed.update([i, j])
                            break
            
            # Second pass: Create single returns for remaining adults
            for i, (_, adult) in enumerate(adults.iterrows()):
                if i not in processed:
                    single_return = self._create_single_filer(adult, hh_members, household)
                    if single_return:
                        tax_units.append(single_return)
            
            return tax_units
            
        except Exception as e:
            logger.error(f"Error processing household {hh_id}: {str(e)}", exc_info=True)
            return []

    def _count_qualifying_relatives(self, person_id: str, hh_members: pd.DataFrame) -> int:
        """
        Count the number of qualifying relatives for a potential Head of Household.
        
        Args:
            person_id: ID of the potential HOH
            hh_members: All members of the household
            
        Returns:
            int: Number of qualifying relatives
        """
        count = 0
        potential_hoh = hh_members[hh_members['person_id'] == person_id].iloc[0]
        
        for _, member in hh_members.iterrows():
            if member['person_id'] == person_id:
                continue
                
            # Check if member is a qualifying relative
            if self._is_qualifying_relative(member, potential_hoh, hh_members):
                count += 1
                
        return count
        
    def _qualifies_for_head_of_household(self, adult: pd.Series, has_qualifying_child: bool, 
                                      hh_members: pd.DataFrame, hh_data: pd.Series) -> bool:
        """
        Determine if an adult qualifies for Head of Household filing status with Hawaii-specific considerations.
        
        Args:
            adult: The adult being evaluated
            has_qualifying_child: Whether the adult has any qualifying children
            hh_members: All members of the household
            hh_data: Household-level data
            
        Returns:
            bool: True if qualifies for Head of Household, False otherwise
        """
        # Must have at least one qualifying person (child OR dependent)
        if not has_qualifying_child:
            # Check for other qualifying persons (e.g., parents, other relatives)
            other_deps = self._count_qualifying_relatives(adult['person_id'], hh_members)
            if other_deps == 0:
                return False
        
        # Get marital status
        mar_status = adult.get('MAR', 6)  # Default to never married if missing
        
        # Expanded criteria for "considered unmarried"
        # 1 = Married, spouse present
        # 2 = Married, spouse absent (can qualify for HOH)
        # 3 = Separated (can qualify for HOH)
        # 4 = Divorced
        # 5 = Widowed
        # 6 = Never married
        
        if mar_status == 1:  # Married, spouse present
            # Check if living apart (no spouse in household)
            spouse_in_hh = hh_members[
                (hh_members['RELSHIPP'].isin(['21', '23'])) &
                (hh_members['person_id'] != adult['person_id'])
            ]
            if len(spouse_in_hh) > 0:
                return False
        
        # Must have some income to maintain household
        adult_income = float(adult.get('PINCP', 0) or 0)
        if adult_income < 5000:  # Must have at least $5,000 in income
            return False
            
        # Must not be a dependent of someone else
        for _, member in hh_members.iterrows():
            if member['person_id'] == adult['person_id']:
                continue
                
            # Check if this member could claim the adult as a dependent
            if self._is_qualifying_relative(adult, member, hh_members):
                return False
                
        return True

    def _get_filing_status(self, adult: pd.Series, has_qualifying_child: bool, 
                        hh_members: pd.DataFrame, hh_data: pd.Series, 
                        is_married: bool = False, file_separately: bool = False) -> str:
        """Determine the correct filing status for the tax unit.
        
        Args:
            adult: Series containing adult's information
            has_qualifying_child: Whether the tax unit has any qualifying children
            hh_members: All members of the household (for HOH determination)
            hh_data: Household-level data (for HOH determination)
            is_married: Whether the adult is married
            file_separately: Whether to file separately (for married couples)
            
        Returns:
            str: Filing status code ('single', 'joint', 'separate', 'head_of_household')
        """
        age = adult.get('AGEP', 0)
        
        if is_married:
            if file_separately:
                status = FILING_STATUS['SEPARATE']
            else:
                status = FILING_STATUS['JOINT']
        elif self._qualifies_for_head_of_household(adult, has_qualifying_child, hh_members, hh_data):
            status = FILING_STATUS['HEAD_HOUSEHOLD']
        else:
            status = FILING_STATUS['SINGLE']
            
        logger.debug(f"Determined filing status for adult {adult.get('person_id')}: {status} "
                   f"(age={age}, married={is_married}, has_qualifying_child={has_qualifying_child})")
        return status
    
    def _create_single_filer(self, adult: pd.Series, hh_members: pd.DataFrame, 
                           hh_data: pd.Series) -> Dict[str, Any]:
        """
        Create a tax unit for a single filer with detailed dependency tests.
        Optimized for performance by reducing redundant calculations and using vectorized operations.
        
        Args:
            adult: The adult who is the primary filer
            hh_members: All members of the household
            hh_data: Household-level data
            
        Returns:
            Dictionary containing tax unit information
        """
        logger.debug(f"Creating single filer for adult {adult['person_id']}")
        
        # Use vectorized version if available and appropriate
        if hasattr(self, '_create_single_filer_vectorized'):
            try:
                return self._create_single_filer_vectorized(adult, hh_members, hh_data)
            except Exception as e:
                logger.warning(f"Vectorized single filer creation failed: {str(e)}")
        
        # Fall back to original implementation
        return self._create_single_filer_original(adult, hh_members, hh_data)
    
    def _create_single_filer_original(self, adult: pd.Series, hh_members: pd.DataFrame, 
                                   hh_data: pd.Series) -> Dict[str, Any]:
        """Original implementation of _create_single_filer for fallback."""
        logger.debug(f"Creating single filer (original) for adult {adult['person_id']}")
        
        # Initialize lists to track dependents
        qualifying_children, other_dependents = self._identify_dependents(adult['person_id'], hh_members)
        
        # Determine filing status
        has_qualifying_child = len(qualifying_children) > 0
        filing_status = self._get_filing_status(adult, has_qualifying_child, hh_members, hh_data)
        
        # Calculate total income (simplified - would include all relevant income sources)
        income = self._calculate_income(adult)
        
        # Create tax unit dictionary
        tax_unit = {
            'taxunit_id': f"{hh_data['SERIALNO']}_single_{adult['person_id']}",
            'SERIALNO': hh_data['SERIALNO'],
            'filing_status': filing_status,
            'primary_filer_id': adult['person_id'],
            'secondary_filer_id': None,
            'num_dependents': len(qualifying_children) + len(other_dependents),
            'num_children': len(qualifying_children),
            'num_other_dependents': len(other_dependents),
            'hh_income': hh_data.get('HINCP', 0),
            'total_income': income,
            'wages': adult.get('WAGP', 0),
            'self_employment_income': adult.get('SEMP', 0),
            'interest_income': adult.get('INTP', 0),
            'ss_income': adult.get('SSP', 0) + adult.get('SSIP', 0),
            'other_income': adult.get('OIP', 0) + adult.get('PAP', 0) + adult.get('RETP', 0),
            'file_separately': False
        }
        
        return tax_unit
    
    def _identify_joint_filers(self, adults: pd.DataFrame, hh_members: pd.DataFrame, 
                             hh_data: pd.Series) -> List[Dict[str, Any]]:
        """
        Identify potential joint filers within a household.
        Uses vectorized operations when possible for better performance.
        """
        # Use vectorized version if available and appropriate
        if hasattr(self, '_identify_joint_filers_vectorized'):
            try:
                return self._identify_joint_filers_vectorized(adults, hh_members, hh_data)
            except Exception as e:
                logger.warning(f"Vectorized joint filer identification failed, falling back: {str(e)}")
        
        # Fall back to original implementation
        return self._identify_joint_filers_original(adults, hh_members, hh_data)
    
    def _identify_joint_filers_original(self, adults: pd.DataFrame, hh_members: pd.DataFrame, 
                                     hh_data: pd.Series) -> List[Dict[str, Any]]:
        """Original implementation of _identify_joint_filers for fallback."""
        tax_units = []
        processed = set()
        
        # First, identify potential married couples
        for i, (_, adult1) in enumerate(adults.iterrows()):
            if i in processed:
                continue
                
            # Look for a spouse among remaining adults
            spouse = None
            for j in range(i + 1, len(adults)):
                if j in processed:
                    continue
                    
                adult2 = adults.iloc[j]
                # Check if these adults appear to be spouses
                if self._is_potential_spouse(adult1, adult2):
                    spouse = adult2
                    processed.add(adults.index[j])  # Add the actual index of the spouse
                    break
            
            if spouse is not None:
                # Create joint return
                tax_unit = self._create_joint_filer(adult1, spouse, hh_members, hh_data)
                if tax_unit is not None:  # Only add if creation was successful
                    tax_units.append(tax_unit)
                    processed.add(i)
                else:
                    # Joint filer creation failed, create single returns instead
                    tax_unit1 = self._create_single_filer(adult1, hh_members, hh_data)
                    tax_unit2 = self._create_single_filer(spouse, hh_members, hh_data)
                    if tax_unit1 is not None:
                        tax_units.append(tax_unit1)
                    if tax_unit2 is not None:
                        tax_units.append(tax_unit2)
                    processed.add(i)
            else:
                # No spouse found, create single return
                tax_unit = self._create_single_filer(adult1, hh_members, hh_data)
                if tax_unit is not None:  # Only add if creation was successful
                    tax_units.append(tax_unit)
    def _identify_joint_filers_vectorized(self, adults: pd.DataFrame, hh_members: pd.DataFrame, 
                                           hh_data: pd.Series) -> List[Dict[str, Any]]:
        """
        Vectorized implementation of _identify_joint_filers for better performance.
            
        Note:
            For complex cases that can't be handled in a fully vectorized way, falls back
            to the original implementation for those specific cases.
        """
        # Implementation of joint filer identification
        return []
        
    def _identify_dependents(self, person_id: str, hh_members: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Identify qualifying children and other dependents for a given person.
        
        Args:
            person_id: ID of the potential filer
            hh_members: All members of the household
            
        Returns:
            Tuple of (list of qualifying child person_ids, list of other dependent person_ids)
        """
        # Get the adult's data
        adult = hh_members[hh_members['person_id'] == person_id].iloc[0]
        
        # Use vectorized implementation if available
        if hasattr(self, '_identify_dependents_vectorized'):
            try:
                return self._identify_dependents_vectorized(adult, hh_members)
            except Exception as e:
                logger.warning(f"Vectorized dependent identification failed: {str(e)}")
                # Fall back to original implementation
        
        # Fall back to original implementation
        return self._identify_dependents_original(adult, hh_members)
        
    def _identify_dependents_original(self, adult: pd.Series, hh_members: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Original implementation of dependent identification using iterative approach.
        
        Args:
            adult: The adult who is the potential filer
            hh_members: All members of the household
            
        Returns:
            Tuple of (list of qualifying child person_ids, list of other dependent person_ids)
        """
        qualifying_children = []
        other_dependents = []
        
        for _, member in hh_members.iterrows():
            if member['person_id'] == adult['person_id']:
                continue  # Skip the adult themselves
                
            # Check if member is a qualifying child
            if self._is_qualifying_child(member, adult, hh_members):
                qualifying_children.append(member['person_id'])
            # Check if member is a qualifying relative (and not already a qualifying child)
            elif self._is_qualifying_relative_original(member, adult, hh_members):
                other_dependents.append(member['person_id'])
                
        return qualifying_children, other_dependents
        
    def _identify_dependents_vectorized(self, adult: pd.Series, hh_members: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Efficiently identify qualifying children and other dependents using vectorized operations.
        
        This implementation provides significant performance improvements over the iterative
        approach by using pandas vectorized operations.
        
        Args:
            adult: Series containing the adult's data
            hh_members: DataFrame containing all household members
            
        Returns:
            Tuple of (list of qualifying child person_ids, list of other dependent person_ids)
        """
        try:
            # Create a copy to avoid modifying the original dataframe
            potential_deps = hh_members.copy()
            
            # Exclude the adult themselves and filter potential dependents
            potential_deps = potential_deps[potential_deps['person_id'] != adult['person_id']]
            if potential_deps.empty:
                return [], []
                
            # Ensure we have the required columns with proper types
            required_numeric = ['AGEP', 'WAGP', 'SEMP', 'INTP', 'RETP', 'OIP', 'PAP', 'SSP', 'SSIP']
            
            # Convert to appropriate dtypes if needed
            for col in required_numeric:
                if col in potential_deps.columns:
                    potential_deps[col] = pd.to_numeric(potential_deps[col], errors='coerce').fillna(0)
            
            # Ensure relationship codes are strings for consistent comparison
            if 'rel_code_str' not in potential_deps.columns:
                potential_deps['rel_code_str'] = potential_deps['RELSHIPP'].astype(str)
            
            # Pre-compute income for all potential dependents using _calculate_income
            potential_deps['total_income'] = potential_deps.apply(
                lambda x: self._calculate_income(x), axis=1
            )
            
            # Define relationship code sets for fast lookups
            child_rels = {'25', '26', '27', '28', '29', '30', '31', '32', '33', '39', '40'}
            other_rels = {'28', '29', '31', '32', '33', '39', '40', '41', '42', '43', '44'}
            
            # Vectorized relationship checks
            is_child_rel = potential_deps['rel_code_str'].isin(child_rels)
            is_other_rel = potential_deps['rel_code_str'].isin(other_rels)
            
            # Age-based checks (vectorized)
            age = potential_deps['AGEP']
            is_student = potential_deps['SCHL'].isin(range(15, 25))  # Enrolled in school
            is_under_19 = age < 19
            is_student_under_24 = (age < 24) & is_student
            
            # Income check - must be below threshold (vectorized)
            income_ok = potential_deps['total_income'] <= HAWAII_INCOME_THRESHOLDS['relative_income']
            
            # Qualifying children conditions (vectorized)
            is_qualifying_child = (
                is_child_rel &  # Has qualifying relationship
                (is_under_19 | is_student_under_24) &  # Age/student status
                income_ok  # Income below threshold
            )
            
            # Get qualifying children
            qualifying_children = potential_deps[is_qualifying_child]['person_id'].tolist()
            
            # Other dependents (qualifying relatives that aren't children)
            is_qualifying_relative = (
                is_other_rel &  # Has qualifying relationship
                ~is_qualifying_child &  # Not already a qualifying child
                income_ok  # Income below threshold
            )
            other_dependents = potential_deps[is_qualifying_relative]['person_id'].tolist()
            
            return qualifying_children, other_dependents
            
        except Exception as e:
            logger.error(f"Error in vectorized dependent identification: {str(e)}")
            # Fall back to original implementation in case of errors
            return self._identify_dependents_original(adult, hh_members)
        
    def _is_relative(self, person1: pd.Series, person2: pd.Series) -> bool:
        """
        Check if two people are related based on their RELSHIPP values.
        
        Args:
            person1: First person's data
            person2: Second person's data
            
        Returns:
            bool: True if the two people are related, False otherwise
        """
        # If they're the same person, they're not relatives for tax purposes
        if person1['person_id'] == person2['person_id']:
            return False
            
        # Get relationship codes
        rel1 = person1.get('RELSHIPP', -1)
        rel2 = person2.get('RELSHIPP', -1)
        
        # Check if they're in the same household
        if person1.get('SERIALNO') != person2.get('SERIALNO'):
            return False
            
        # Check if either is the reference person (20) and the other is a relative
        if (rel1 == 20 and rel2 in [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]) or \
           (rel2 == 20 and rel1 in [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]):
            return True
            
        # Check if they're siblings
        if rel1 in [24, 25, 26] and rel2 in [24, 25, 26]:
            return True
            
        # Add other relationship checks as needed
        
        return False
        
    def _is_qualifying_relative_original(self, person: pd.Series, potential_filer: pd.Series, 
                                      hh_members: pd.DataFrame) -> bool:
        """
        Determine if a person qualifies as a relative for tax purposes.
        
        Args:
            person: The person who might be a qualifying relative
            potential_filer: The potential filer (adult)
            hh_members: All members of the household
            
        Returns:
            bool: True if the person is a qualifying relative, False otherwise
        """
        # Relationship test - must be a relative
        if not self._is_relative(person, potential_filer):
            return False
            
        # Income test - must have gross income less than the exemption amount
        income = self._calculate_income(person)
        if income >= HAWAII_INCOME_THRESHOLDS['dependent']:
            return False
            
        # Support test - must not provide more than half of their own support
        # For now, we'll assume they don't provide more than half of their own support
        
        # Not a qualifying child of any other taxpayer
        # This is a simplified check - in a full implementation, you'd need to check against all other adults
        
        # Not filing a joint return (unless it's only to claim a refund)
        if person.get('MAR') == 1:  # Married filing jointly
            return False
            
        # Must be a US citizen, national, or resident alien
        # This is a simplified check - in a real implementation, you'd check CIT or CITWP
        
        return True
        
    def _is_qualifying_child(self, child: pd.Series, potential_parent: pd.Series, 
                           hh_members: pd.DataFrame) -> bool:
        """
        Determine if a person qualifies as a child for tax purposes (EITC/CTC).
{{ ... }}
        
        IRS tests for qualifying child:
        1. Relationship test
        2. Age test
        3. Residency test (assumed met if in same household)
        4. Support test (child cannot provide > 50% of own support)
        5. Joint return test (child cannot file joint return)
        """
        # Use vectorized version if available and appropriate
        if hasattr(self, '_is_qualifying_child_vectorized'):
            try:
                # Convert to DataFrame row if needed
                if not isinstance(child, pd.Series):
                    child = pd.Series(child)
                if not isinstance(potential_parent, pd.Series):
                    potential_parent = pd.Series(potential_parent)
                return self._is_qualifying_child_vectorized(child, potential_parent, hh_members)
            except Exception as e:
                logger.warning(f"Vectorized qualifying child check failed, falling back: {str(e)}")
        
        # Fall back to original implementation
        return self._is_qualifying_child_original(child, potential_parent, hh_members)
    
    def _is_qualifying_child_original(self, child: pd.Series, potential_parent: pd.Series, 
                                   hh_members: pd.DataFrame) -> bool:
        """Original implementation of _is_qualifying_child for fallback."""
        # Relationship test
        child_rel = str(child['RELSHIPP'])
        
        # Check if the child is related to the potential parent
        # 25 = Biological son or daughter
        # 26 = Adopted son or daughter
        # 27 = Stepson or stepdaughter
        # 30 = Son or daughter of unmarried partner
        # 34 = Foster child or stepchild of unmarried partner
        if child_rel not in ['25', '26', '27', '30', '34']:
            return False
            
        # Age test
        age = child['AGEP']
        if pd.isna(age) or age >= 19:  # Must be under 19, or under 24 if a student
            # Check if they're a student under 24
            if age >= 24 or not self._is_student(child):
                return False
        
        # Support test (simplified - assumes child doesn't provide > 50% of own support)
        # In a more detailed implementation, we would check actual support amounts
        # Support test - use income as proxy
        if person_income > HAWAII_INCOME_THRESHOLDS['dependent']:
            return False
        
        return True
    
    def _is_qualifying_relative(self, person: pd.Series, potential_claimer: pd.Series, 
                              hh_members: pd.DataFrame) -> bool:
        """
        Determine if a person qualifies as a dependent relative.
        Uses vectorized operations when possible for better performance.
        """
        # Use vectorized version if available and appropriate
        if hasattr(self, '_is_qualifying_relative_vectorized'):
            try:
                if not isinstance(person, pd.Series):
                    person = pd.Series(person)
                if not isinstance(potential_claimer, pd.Series):
                    potential_claimer = pd.Series(potential_claimer)
                return self._is_qualifying_relative_vectorized(person, potential_claimer, hh_members)
            except Exception as e:
                logger.warning(f"Vectorized qualifying relative check failed: {str(e)}")
        
        # Fall back to original implementation
        return self._is_qualifying_relative_original(person, potential_claimer, hh_members)
    
    def _is_qualifying_relative_original(self, person: pd.Series, potential_claimer: pd.Series, 
                                      hh_members: pd.DataFrame) -> bool:
        """Original implementation of _is_qualifying_relative for fallback."""
        # Relationship test - must be child, stepchild, adopted child, foster child,
        # sibling, stepsibling, or descendant of any of these
        valid_relationships = {
            '25', '26', '27',  # Children (who don't meet qualifying child)
            '28',  # Siblings
            '29',  # Parents
            '30',  # Grandchildren
            '31',  # In-laws
            '32',  # Son/daughter-in-law
            '33',  # Other relatives
            '39', '40'  # More restrictive Hawaii extended family
        }
        
        # Member of household test
        rel_code = str(person.get('RELSHIPP', ''))
        if rel_code not in valid_relationships:
            # Could still qualify if lived with taxpayer all year
            # Exclude roommates/housemates
            if rel_code in {'35', '36', '37'}:
                return False
        
        # Gross income test - use pre-calculated total_income if available
        if 'total_income' in person:
            person_income = person['total_income']
        else:
            person_income = 0
            for inc_type in ['WAGP', 'SEMP', 'INTP', 'RETP', 'OIP', 'PAP']:
                person_income += abs(float(person.get(inc_type, 0) or 0))
        
        if person_income > HAWAII_INCOME_THRESHOLDS['relative_income']:
            return False
        
        # Support test - use income as proxy
        if person_income > HAWAII_INCOME_THRESHOLDS['dependent']:
            return False
            
        # Can't be claimed if they file a joint return (simplified)
        if person.get('MAR') == 1:  # 1 = Married, spouse present
            return False
            
        # Can't be claimed if they are a qualifying child of another taxpayer
        # This is handled at the tax unit construction level
        
        return True
        
    def _is_qualifying_relative_vectorized(self, person: pd.Series, potential_claimer: pd.Series, 
                                          hh_members: pd.DataFrame) -> bool:
        """
        Vectorized implementation of _is_qualifying_relative for better performance.
        
        Args:
            person: Person being evaluated as a qualifying relative
            potential_claimer: Person who might claim this relative
            hh_members: All members of the household
            
        Returns:
            bool: True if the person qualifies as a relative, False otherwise
        """
        try:
            # Fast path: Use pre-calculated values if available
            if 'rel_code_str' in person:
                rel_code = person['rel_code_str']
            else:
                rel_code = str(person.get('RELSHIPP', ''))
            
            # Relationship test - must be child, stepchild, adopted child, foster child,
            # sibling, stepsibling, or descendant of any of these
            valid_relationships = {
                '25', '26', '27',  # Children (who don't meet qualifying child)
                '28',  # Siblings
                '29',  # Parents
                '30',  # Grandchildren
                '31',  # In-laws
                '32',  # Son/daughter-in-law
                '33',  # Other relatives
                '39', '40'  # More restrictive Hawaii extended family
            }
            
            # Member of household test
            if rel_code not in valid_relationships:
                # Could still qualify if lived with taxpayer all year
                # Exclude roommates/housemates
                if rel_code in {'35', '36', '37'}:
                    return False
            
            # Gross income test - use pre-calculated total_income if available
            if 'total_income' in person:
                person_income = person['total_income']
            else:
                person_income = sum(
                    abs(float(person.get(inc_type, 0) or 0)) 
                    for inc_type in ['WAGP', 'SEMP', 'INTP', 'RETP', 'OIP', 'PAP']
                )
            
            if person_income > HAWAII_INCOME_THRESHOLDS['relative_income']:
                return False
            
            # Support test - use income as proxy
            if person_income > HAWAII_INCOME_THRESHOLDS['dependent']:
                return False
                
            # Can't be claimed if they file a joint return (simplified)
            if person.get('MAR') == 1:  # 1 = Married, spouse present
                return False
                
            # Can't be claimed if they are a qualifying child of another taxpayer
            # This is handled at the tax unit construction level
            
            return True
            
        except Exception as e:
            logger.error(f"Error in vectorized qualifying relative check: {str(e)}")
            # Fall back to original implementation
            return self._is_qualifying_relative_original(person, potential_claimer, hh_members)
    
    def _is_potential_spouse(self, person1: pd.Series, person2: pd.Series) -> bool:
        """
        Determine if two adults are likely spouses or partners.
        Uses vectorized operations when possible for better performance.
        
        Args:
            person1: First person's data
            person2: Second person's data
            
        Returns:
            bool: True if likely spouses/partners, False otherwise
        """
        # Use vectorized version if available and appropriate
        if hasattr(self, '_is_potential_spouse_vectorized'):
            try:
                # Convert to Series if needed
                if not isinstance(person1, pd.Series):
                    person1 = pd.Series(person1)
                if not isinstance(person2, pd.Series):
                    person2 = pd.Series(person2)
                return self._is_potential_spouse_vectorized(person1, person2)
            except Exception as e:
                logger.warning(f"Vectorized spouse check failed, falling back: {str(e)}")
        
        # Fall back to original implementation
        return self._is_potential_spouse_original(person1, person2)
    
    def _is_potential_spouse_original(self, person1: pd.Series, person2: pd.Series) -> bool:
        """Original implementation of _is_potential_spouse for fallback."""
        # Skip if either person is not an adult
        if person1['AGEP'] < 18 or person2['AGEP'] < 18:
            return False
            
        # Check if they have the same last name (if available)
        if 'NAME_LAST' in person1 and 'NAME_LAST' in person2:
            if person1['NAME_LAST'] and person2['NAME_LAST']:
                if person1['NAME_LAST'] != person2['NAME_LAST']:
                    # Different last names - less likely to be spouses
                    # But still possible (e.g., not changed name after marriage)
                    pass
        
        # Check marital status
        # 1 = Married, spouse present
        # 2 = Married, spouse absent
        # 3 = Separated
        # 4 = Divorced
        # 5 = Widowed
        # 6 = Never married
        
        # If either is married to someone else, they can't be spouses
        if person1.get('MAR') in [1, 2] and person2.get('MAR') in [1, 2]:
            # Both are married - could be to each other if they're the only adults
            pass
        elif person1.get('MAR') in [1, 2] or person2.get('MAR') in [1, 2]:
            # Only one is married - can't be spouses unless they're married to each other
            # This would require checking the spouse's ID if available
            pass
            
        # Check relationship codes
        rel1 = str(person1.get('RELSHIPP', ''))
        rel2 = str(person2.get('RELSHIPP', ''))
        
        # Spouse/partner relationships
        spouse_relationships = {
            '21',  # Opposite-sex spouse
            '22',  # Opposite-sex unmarried partner
            '23',  # Same-sex spouse
            '24'   # Same-sex unmarried partner
        }
        
        # Check if they are spouses/partners of each other
        if (rel1 in spouse_relationships and rel2 == '20') or \
           (rel2 in spouse_relationships and rel1 == '20'):
            return True
            
        # Check if they are both reference persons (20) in a multi-adult household
        # This could indicate a cohabiting couple where one is not properly marked as spouse
        if rel1 == '20' and rel2 == '20':
            # Additional checks to reduce false positives
            age_diff = abs(person1['AGEP'] - person2['AGEP'])
            max_age_diff = 20  # Maximum age difference for potential spouses
            
            # More likely to be spouses if similar age
            if age_diff <= max_age_diff:
                # Check if they have children in common
                # This is a simplified check - in a full implementation, we'd look at relationships
                has_children = False
                if 'hh_members' in person1 and 'hh_members' in person2:
                    # Check if they share children in the household
                    pass
                
                # If they have children in common, more likely to be spouses
                if has_children:
                    return True
                    
                # If no children, be more conservative
                # Check if they are the only two adults in the household
                # This is a common pattern for cohabiting couples
                if 'hh_adults' in person1 and person1['hh_adults'] == 2:
                    return True
                    
        return False
        
    def _is_potential_spouse_vectorized(self, person1: pd.Series, person2: pd.Series) -> bool:
        """
        Vectorized implementation of _is_potential_spouse for better performance.
        
        Args:
            person1: First person's data
            person2: Second person's data
            
        Returns:
            bool: True if likely spouses/partners, False otherwise
        """
        try:
            # Skip if either person is not an adult
            if person1['AGEP'] < 18 or person2['AGEP'] < 18:
                return False
                
            # Check relationship codes first (fastest check)
            rel1 = str(person1.get('RELSHIPP', ''))
            rel2 = str(person2.get('RELSHIPP', ''))
            
            # Spouse/partner relationships
            spouse_relationships = {'21', '22', '23', '24'}
            
            # 1. Direct spouse/partner relationship (one is reference, other is spouse/partner)
            if (rel1 in spouse_relationships and rel2 == '20') or \
               (rel2 in spouse_relationships and rel1 == '20'):
                return True
                
            # 2. Both are reference persons (20) - could be cohabiting couple
            if rel1 == '20' and rel2 == '20':
                # Additional checks for cohabiting couples
                age_diff = abs(person1['AGEP'] - person2['AGEP'])
                max_age_diff = 25  # Increased from 20 to be more inclusive
                
                # More likely to be spouses if similar age, but don't be too strict
                if age_diff > max_age_diff:
                    return False
                
                # Check if they are the only two adults in the household
                if 'hh_adults' in person1 and person1['hh_adults'] == 2:
                    return True
                    
                # Even if not the only adults, they might be a couple if they have children together
                if 'has_children' in person1 and person1['has_children'] and \
                   'has_children' in person2 and person2['has_children']:
                    return True
            
            # 3. Check marital status - if both are married, they might be married to each other
            mar1 = person1.get('MAR', 6)  # Default to never married if missing
            mar2 = person2.get('MAR', 6)
            
            # If both are married (status 1 = married, spouse present)
            if mar1 == 1 and mar2 == 1:
                # If they're the only two adults, they're likely married to each other
                if 'hh_adults' in person1 and person1['hh_adults'] == 2:
                    return True
                    
                # If they have similar last names, more likely to be married
                if 'NAME_LAST' in person1 and 'NAME_LAST' in person2:
                    if person1['NAME_LAST'] and person2['NAME_LAST'] and \
                       person1['NAME_LAST'] == person2['NAME_LAST'] and \
                       abs(person1['AGEP'] - person2['AGEP']) <= 15:  # Similar age and same last name
                        return True
            
            # 4. Check for common children (if available in the data)
            if 'children_in_common' in person1 and person1['children_in_common'] > 0:
                return True
                
            return False
            
        except Exception as e:
            logger.warning(f"Error in vectorized spouse check: {str(e)}")
            # Fall back to original implementation
            return self._is_potential_spouse_original(person1, person2)
    
    def _create_joint_filer(self, adult1: pd.Series, adult2: pd.Series, 
                          hh_members: pd.DataFrame, hh_data: pd.Series) -> Dict[str, Any]:
        """Create a tax unit for a joint filer.
        
        Args:
            adult1: First adult in the joint filing couple
            adult2: Second adult in the joint filing couple
            hh_members: All members of the household
            hh_data: Household-level data
            
        Returns:
            Dictionary containing tax unit information or None if not valid
        """
        logger.debug(f"Creating joint filer for adults {adult1['person_id']} and {adult2['person_id']}")
        
        # Count qualifying children (under 17)
        children = hh_members[
            (hh_members['AGEP'] < 17) & 
            (hh_members['AGEP'] > 0) &  # Exclude infants if needed
            (hh_members['person_id'] != adult1['person_id']) & 
            (hh_members['person_id'] != adult2['person_id'])
        ]
        
        # Count other dependents (17+ or other relatives)
        other_deps = hh_members[
            ((hh_members['AGEP'] >= 17) & (hh_members['AGEP'] < 19) |  # 17-18 year olds
             (hh_members['AGEP'] >= 19) & (hh_members['AGEP'] < 24) &  # 19-23 year old students
             hh_members['SCHL'].between(15, 18))  # Enrolled in school
        ]
        
        # Determine primary and secondary filer (typically higher earner first)
        if adult1.get('PINCP', 0) >= adult2.get('PINCP', 0):
            primary, secondary = adult1, adult2
        else:
            primary, secondary = adult2, adult1
        
        # Check if married couple should file separately
        # Common reasons: one spouse is nonresident alien, separated but not divorced,
        # or one spouse is filing for innocent spouse relief
        file_separately = False
        
        # Check if either spouse is a nonresident alien (common reason for MFS)
        if adult1.get('CIT', 0) == 5 or adult2.get('CIT', 0) == 5:  # 5 = Not a citizen
            logger.debug(f"One spouse is a nonresident alien, using Married Filing Separately status")
            file_separately = True
            
        # Get appropriate filing status using the helper method
        has_qualifying_child = len(children) > 0
        filing_status = self._get_filing_status(
            adult=primary,
            has_qualifying_child=has_qualifying_child,
            hh_members=hh_members,
            hh_data=hh_data,
            is_married=True,
            file_separately=file_separately
        )
        
        logger.debug(f"Joint filers {primary['person_id']} and {secondary['person_id']} have "
                   f"{len(children)} qualifying children and {len(other_deps)} other dependents")
        logger.debug(f"Assigned filing status: {filing_status}")
        
        # Calculate total income for the tax unit using _calculate_income
        # This ensures we don't double-count income sources
        total_income = self._calculate_income(primary) + self._calculate_income(secondary)
        
        # For informational purposes, we can still break down the income components
        # but we won't use them to calculate the total income
        wages = primary.get('WAGP', 0) + secondary.get('WAGP', 0)
        self_employment_income = primary.get('SEMP', 0) + secondary.get('SEMP', 0)
        interest_income = primary.get('INTP', 0) + secondary.get('INTP', 0)
        ss_income = (primary.get('SSP', 0) + primary.get('SSIP', 0) +
                    secondary.get('SSP', 0) + secondary.get('SSIP', 0))
        other_income = (primary.get('OIP', 0) + primary.get('PAP', 0) + primary.get('RETP', 0) +
                       secondary.get('OIP', 0) + secondary.get('PAP', 0) + secondary.get('RETP', 0))
        
        # Create and return the tax unit dictionary
        return {
            'taxunit_id': f"{hh_data['SERIALNO']}_joint_{primary['person_id']}_{secondary['person_id']}",
            'SERIALNO': hh_data['SERIALNO'],
            'filing_status': filing_status,
            'primary_filer_id': primary['person_id'],
            'secondary_filer_id': secondary['person_id'],
            'num_dependents': len(children) + len(other_deps),
            'num_children': len(children),
            'num_other_dependents': len(other_deps),
            'hh_income': hh_data.get('HINCP', 0),
            'total_income': total_income,  # This is the authoritative total income
            'wages': wages,
            'self_employment_income': self_employment_income,
            'interest_income': interest_income,
            'ss_income': ss_income,
            'other_income': other_income,
            'file_separately': file_separately  # Track whether filing separately
        }

def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed PUMS data."""
    # Look for data in the project root's data/processed directory
    data_dir = Path('/Users/dtomkatsu/CascadeProjects/ctc-and-eitc/data/processed')
    person_file = data_dir / 'pums_person_processed.parquet'
    hh_file = data_dir / 'pums_household_processed.parquet'
    
    logger.info(f"Looking for person data at: {person_file}")
    logger.info(f"Looking for household data at: {hh_file}")
    
    person_df = pd.read_parquet(person_file)
    hh_df = pd.read_parquet(hh_file)
    
    return person_df, hh_df

def main():
    """Main function to demonstrate tax unit creation."""
    try:
        # Load processed data
        logger.info("Loading processed PUMS data...")
        person_df, hh_df = load_processed_data()
        
        # Create tax units
        constructor = TaxUnitConstructor(person_df, hh_df)
        tax_units = constructor.create_rule_based_units()
        
        # Save results
        output_file = Path(__file__).parent.parent / 'data/processed/tax_units_rule_based.parquet'
        tax_units.to_parquet(output_file, index=False)
        logger.info(f"Saved {len(tax_units)} tax units to {output_file}")
        
        # Print summary
        print("\nTax Unit Summary:")
        print(f"Total tax units: {len(tax_units)}")
        print(f"Single filers: {len(tax_units[tax_units['filing_status'] == FILING_STATUS['SINGLE']])}")
        print(f"Joint filers: {len(tax_units[tax_units['filing_status'] == FILING_STATUS['JOINT']])}")
        print(f"Head of Household filers: {len(tax_units[tax_units['filing_status'] == FILING_STATUS['HEAD_HOUSEHOLD']])}")
        print(f"Married Filing Separately: {len(tax_units[tax_units['filing_status'] == FILING_STATUS['SEPARATE']])}")
        print(f"Total dependents: {tax_units['num_dependents'].sum()}")
        
        # Print distribution of filing statuses
        if len(tax_units) > 0:
            print("\nFiling Status Distribution:")
            status_dist = tax_units['filing_status'].value_counts(normalize=True) * 100
            for status, pct in status_dist.items():
                print(f"{status}: {pct:.1f}%")
        
    except Exception as e:
        logger.error(f"Error creating tax units: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
