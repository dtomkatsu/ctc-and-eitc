"""
Tax Unit Constructor

This module provides the main TaxUnitConstructor class that uses the modular components
to construct tax units from PUMS data with performance optimizations including batch processing,
parallel execution, and memory efficiency.
"""

import logging
import time
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Union, Any, Generator, Set
import pandas as pd
import numpy as np
from tqdm import tqdm

# Import modular components
from .status import is_married_filing_jointly, is_married_filing_separately, is_head_of_household
from .income import calculate_tax_unit_income
from .dependencies import identify_dependents
from .utils import setup_logging, validate_input_data, create_person_id
from .validation import TaxUnitValidator, ValidationIssue, ValidationSeverity

# Constants for optimization
DEFAULT_BATCH_SIZE = 1000
DEFAULT_NUM_PROCESSES = max(1, mp.cpu_count() - 1)

# Configure logging
logger = logging.getLogger(__name__)


class TaxUnitConstructor:
    """
    Main class for constructing tax units from PUMS data using modular components.
    
    This class orchestrates the tax unit construction process by leveraging
    specialized modules for different aspects of the process.
    """
    
    def __init__(self, person_df: pd.DataFrame, hh_df: pd.DataFrame, 
                 batch_size: int = None, num_processes: int = None,
                 progress_bar: bool = True):
        """
        Initialize the TaxUnitConstructor with person and household data.
        
        Args:
            person_df: DataFrame containing person-level PUMS data
            hh_df: DataFrame containing household-level PUMS data
            batch_size: Number of households to process in each batch (default: 1000)
            num_processes: Number of parallel processes to use (default: CPU count - 1)
            progress_bar: Whether to show a progress bar during processing
        """
        self.person_df = person_df.copy()
        self.hh_df = hh_df.copy()
        self.tax_units = None
        
        # Performance settings
        self.batch_size = batch_size or DEFAULT_BATCH_SIZE
        self.num_processes = num_processes or DEFAULT_NUM_PROCESSES
        self.progress_bar = progress_bar and (len(person_df) > 1000)  # Only show for large datasets
        
        # Setup logging
        setup_logging()
        logger.info(f"Initialized TaxUnitConstructor with batch_size={self.batch_size}, "
                   f"num_processes={self.num_processes}")
        
        # Validate inputs
        is_valid, error_msg = validate_input_data(self.person_df, self.hh_df)
        if not is_valid:
            raise ValueError(f"Invalid input data: {error_msg}")
        
        # Preprocess data
        self._preprocess_data()
    
    def _preprocess_data(self) -> None:
        """Preprocess the input data for tax unit construction."""
        # Check if person_id already exists as index
        if self.person_df.index.name == 'person_id':
            # Person IDs already set up correctly, don't recreate them
            logger.debug("Person IDs already exist as index, preserving them")
        else:
            # Create person_id from SERIALNO and SPORDER
            self.person_df['person_id'] = create_person_id(self.person_df)
            
            # Set person_id as the index
            self.person_df.set_index('person_id', inplace=True)
        
        # Add is_adult flag
        self.person_df['is_adult'] = self.person_df['AGEP'] >= 18
        self.person_df['is_child'] = ~self.person_df['is_adult']
        
        # Merge household data - preserve the index
        original_index = self.person_df.index
        self.person_df = self.person_df.merge(
            self.hh_df[['SERIALNO', 'HINCP']], 
            on='SERIALNO', 
            how='left'
        )
        # Restore the original index
        self.person_df.index = original_index
    
    def _process_household_batch(self, batch: List[Tuple[str, pd.DataFrame]]) -> Tuple[List[dict], List[dict]]:
        """
        Process a batch of households.
        
        Args:
            batch: List of (household_id, household_data) tuples
            
        Returns:
            Tuple of (tax_units, validation_issues) for the batch
        """
        batch_units = []
        batch_issues = []
        
        for hh_id, hh_group in batch:
            try:
                logger.debug(f"Processing household {hh_id} in batch")
                household_units = self._process_household(hh_group)
                
                if household_units:
                    # Validate individual tax units
                    for unit in household_units:
                        issues = TaxUnitValidator.validate_tax_unit(unit)
                        batch_issues.extend(issues)
                    
                    # Validate household coverage
                    coverage_issues = TaxUnitValidator.validate_household_coverage(
                        household_units, hh_group
                    )
                    batch_issues.extend(coverage_issues)
                    
                    batch_units.extend(household_units)
                
            except Exception as e:
                error_msg = f"Error processing household {hh_id}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                batch_issues.append({
                    'household_id': hh_id,
                    'error': error_msg,
                    'severity': 'ERROR'
                })
        
        return batch_units, batch_issues
    
    def _create_batches(self, data: pd.DataFrame) -> List[List[Tuple[str, pd.DataFrame]]]:
        """
        Split data into batches for parallel processing.
        
        Args:
            data: DataFrame containing person data with SERIALNO column
            
        Returns:
            List of batches, where each batch is a list of (household_id, household_data) tuples
        """
        # Group by household
        households = list(data.groupby('SERIALNO'))
        
        # Create batches
        batches = []
        for i in range(0, len(households), self.batch_size):
            batch = households[i:i + self.batch_size]
            batches.append(batch)
        
        logger.info(f"Created {len(batches)} batches with up to {self.batch_size} households each")
        return batches
    
    def create_rule_based_units(self, parallel: bool = True) -> pd.DataFrame:
        """
        Create tax units using rule-based approach with optional parallel processing.
        
        This method processes households in batches, with each batch optionally processed
        in parallel. It handles the following types of tax units:
        - Married filing jointly
        - Married filing separately
        - Head of household
        - Single filers
        
        Args:
            parallel: Whether to use parallel processing (default: True)
            
        Returns:
            DataFrame containing the constructed tax units with validation results
            
        Raises:
            ValueError: If input data is invalid or processing fails
        """
        start_time = time.time()
        logger.info(f"Creating tax units using rule-based approach (parallel={parallel}, "
                   f"processes={self.num_processes}, batch_size={self.batch_size})...")
        
        tax_units = []
        validation_issues = []
        
        # Create batches of households
        batches = self._create_batches(self.person_df)
        
        # Process batches
        if parallel and self.num_processes > 1:
            # Process batches in parallel
            with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                # Submit all batches
                future_to_batch = {
                    executor.submit(self._process_household_batch, batch): i 
                    for i, batch in enumerate(batches)
                }
                
                # Process results as they complete
                for future in (tqdm(as_completed(future_to_batch), 
                                 total=len(batches),
                                 desc="Processing households",
                                 disable=not self.progress_bar) 
                             if self.progress_bar else as_completed(future_to_batch)):
                    
                    batch_units, batch_issues = future.result()
                    tax_units.extend(batch_units)
                    validation_issues.extend(batch_issues)
        else:
            # Process batches sequentially
            for batch in (tqdm(batches, desc="Processing households", disable=not self.progress_bar) 
                         if self.progress_bar else batches):
                batch_units, batch_issues = self._process_household_batch(batch)
                tax_units.extend(batch_units)
                validation_issues.extend(batch_issues)
        
        # Log validation issues
        if validation_issues:
            self._log_validation_issues(validation_issues)
        
        # Calculate performance metrics
        end_time = time.time()
        processing_time = end_time - start_time
        units_per_second = len(tax_units) / processing_time if processing_time > 0 else 0
        
        # Log performance summary
        logger.info(f"Processed {len(tax_units):,} tax units from {len(self.person_df['SERIALNO'].unique()):,} "
                  f"households in {processing_time:.2f} seconds ({units_per_second:.1f} units/sec)")
        
        # Convert to DataFrame with optimized dtypes
        if tax_units:
            # Create DataFrame with optimized dtypes
            self.tax_units = pd.DataFrame(tax_units)
            
            # Optimize column dtypes
            self._optimize_dataframe_dtypes()
            
            # Add validation summary as an attribute
            def get_severity(issue):
                if hasattr(issue, 'severity'):
                    return issue.severity
                elif isinstance(issue, dict):
                    return issue.get('severity')
                else:
                    return 'INFO'
            
            error_count = sum(1 for issue in validation_issues if get_severity(issue) == 'ERROR')
            warning_count = sum(1 for issue in validation_issues if get_severity(issue) == 'WARNING')
            
            self.validation_summary = {
                'total_units': len(tax_units),
                'total_households': len(self.person_df['SERIALNO'].unique()),
                'processing_time_seconds': processing_time,
                'units_per_second': units_per_second,
                'validation_issues': len(validation_issues),
                'validation_errors': error_count,
                'validation_warnings': warning_count,
                'has_errors': error_count > 0,
                'batch_size': self.batch_size,
                'num_processes': self.num_processes if parallel and self.num_processes > 1 else 1
            }
            
            logger.info(f"Validation summary: {len(tax_units):,} tax units created, "
                      f"{error_count} errors, {warning_count} warnings")
        else:
            # Empty result with proper column dtypes
            self.tax_units = pd.DataFrame({
                'filer_id': pd.Series(dtype='string'),
                'filing_status': pd.CategoricalDtype(categories=[
                    'single', 'married_filing_jointly', 
                    'married_filing_separately', 'head_of_household'
                ]),
                'income': pd.Series(dtype='float64'),
                'num_dependents': pd.Series(dtype='int32'),
                'dependents': pd.Series(dtype='object'),  # List of strings
                'hh_id': pd.Series(dtype='string'),
                'primary_filer_id': pd.Series(dtype='string'),
                'secondary_filer_id': pd.Series(dtype='string')
            })
            
            self.validation_summary = {
                'total_units': 0,
                'processing_time_seconds': processing_time,
                'units_per_second': 0,
                'validation_issues': len(validation_issues),
                'validation_errors': sum(1 for issue in validation_issues 
                                      if getattr(issue, 'severity', '') == 'ERROR'),
                'validation_warnings': sum(1 for issue in validation_issues 
                                        if getattr(issue, 'severity', '') == 'WARNING'),
                'has_errors': True,
                'batch_size': self.batch_size,
                'num_processes': self.num_processes if parallel and self.num_processes > 1 else 1
            }
            
        return self.tax_units
    
    def _optimize_dataframe_dtypes(self) -> None:
        """
        Optimize the data types of the tax_units DataFrame to reduce memory usage.
        
        This method should be called after creating the tax_units DataFrame.
        """
        if self.tax_units is None or self.tax_units.empty:
            return
            
        # Convert string columns to categorical where appropriate
        for col in self.tax_units.columns:
            # Skip non-string columns
            if not pd.api.types.is_string_dtype(self.tax_units[col]):
                continue
                
            # For string columns with low cardinality, use categorical
            num_unique = len(self.tax_units[col].unique())
            num_rows = len(self.tax_units)
            
            if 1 < num_unique < (num_rows * 0.5):  # Less than 50% unique values
                self.tax_units[col] = self.tax_units[col].astype('category')
                
        # Downcast numeric columns
        for col in self.tax_units.select_dtypes(include=['int64', 'float64']).columns:
            if pd.api.types.is_integer_dtype(self.tax_units[col]):
                # Downcast integers
                self.tax_units[col] = pd.to_numeric(self.tax_units[col], downcast='integer')
            else:
                # Downcast floats
                self.tax_units[col] = pd.to_numeric(self.tax_units[col], downcast='float')
        
        # Log memory savings
        mem_before = self.tax_units.memory_usage(deep=True).sum() / 1024**2  # MB
        mem_after = self.tax_units.memory_usage(deep=True).sum() / 1024**2  # MB
        logger.info(f"Optimized DataFrame memory usage: {mem_before:.2f}MB -> {mem_after:.2f}MB "
                  f"({((mem_before - mem_after) / mem_before * 100):.1f}% reduction)")
    
    def _log_validation_issues(self, issues: List[Dict[str, Any]]) -> None:
        """
        Log validation issues at appropriate severity levels.
        
        Args:
            issues: List of validation issues
        """
        if not issues:
            return
            
        # Handle both dict and ValidationIssue objects
        def get_severity(issue):
            if hasattr(issue, 'severity'):
                return issue.severity
            elif isinstance(issue, dict):
                return issue.get('severity')
            else:
                return 'INFO'
        
        def get_message(issue):
            if hasattr(issue, 'message'):
                return issue.message
            elif isinstance(issue, dict):
                return issue.get('message', str(issue))
            else:
                return str(issue)
        
        error_count = sum(1 for issue in issues if get_severity(issue) == 'ERROR')
        warning_count = sum(1 for issue in issues if get_severity(issue) == 'WARNING')
        info_count = len(issues) - error_count - warning_count
        
        logger.info(f"Validation summary: {error_count} errors, {warning_count} warnings, {info_count} info messages")
        
        # Log errors first
        for issue in (i for i in issues if get_severity(i) == 'ERROR'):
            logger.error(f"Validation error: {get_message(issue)}")
            
        # Then warnings
        for issue in (i for i in issues if get_severity(i) == 'WARNING'):
            logger.warning(f"Validation warning: {get_message(issue)}")
            
        # Finally info messages
        for issue in (i for i in issues if get_severity(i) not in ['ERROR', 'WARNING']):
            logger.info(f"Validation info: {get_message(issue)}")

    def _process_household(self, hh_group: pd.DataFrame) -> List[dict]:
        """
        Process a single household and return a list of tax units using vectorized operations.
        
        This method identifies different types of tax units within a household:
        - Married couples (filing jointly or separately)
        - Head of household filers
        - Single filers
        
        Optimized for performance with:
        - Vectorized operations instead of loops
        - Efficient data structures (sets, dictionaries)
        - Batch processing where possible
        
        Args:
            hh_group: DataFrame containing all persons in a household
            
        Returns:
            List of tax unit dictionaries, where each dictionary contains:
            - filer_id: Unique identifier for the tax unit
            - filing_status: One of 'single', 'married_filing_jointly', 
                           'married_filing_separately', 'head_of_household'
            - income: Total income for the tax unit
            - num_dependents: Number of dependents claimed
            - dependents: List of dependent person IDs
            - hh_id: Household identifier
            - primary_filer_id: ID of the primary filer
            - secondary_filer_id: ID of the secondary filer (for joint returns)
        """
        # Skip empty households immediately
        if hh_group.empty:
            logger.debug("Skipping empty household")
            return []

        # Pre-compute household data and cache
        hh_id = hh_group['SERIALNO'].iloc[0]
        logger.info(f"Processing household {hh_id} with {len(hh_group)} members")
        
        # Use vectorized operation to get household data
        hh_data = self.hh_df[self.hh_df['SERIALNO'] == hh_id].iloc[0] if not self.hh_df.empty else {}
        
        # Vectorized adult identification
        adults_mask = hh_group['AGEP'] >= 18
        adults = hh_group[adults_mask].copy()
        
        if adults.empty:
            logger.debug(f"No adults in household {hh_id}, skipping")
            return []
            
        # Vectorized dependent identification
        logger.debug("Identifying all dependents in household using vectorized operations")
        dependents = identify_dependents(hh_group)
        
        # Convert dependents to a more efficient structure
        dependents = {k: set(v) for k, v in dependents.items()}
        all_dependents = set().union(*dependents.values()) if dependents else set()
        
        # Log initial assignments if debug level
        if logger.isEnabledFor(logging.DEBUG):
            for filer_id, deps in dependents.items():
                if deps:
                    logger.debug(f"Initial dependent assignment - Filer {filer_id} has {len(deps)} dependents")
        
        # Track state with sets for O(1) lookups
        claimed_dependents = set()
        processed_adults = set()
        tax_units = []
        
        # Identify potential joint filers and MFS filers using vectorized operations
        logger.debug("Identifying potential joint and MFS filers")
        joint_filers, mfs_filers = self._identify_joint_filers(adults, hh_group)
        logger.debug(f"Found {len(joint_filers)} joint filer pairs and {len(mfs_filers)} MFS filer pairs")
        
        # Process MFS filers first (they file as single)
        for adult1_id, adult2_id in mfs_filers:
            if adult1_id in processed_adults or adult2_id in processed_adults:
                continue
                
            adult1 = adults.loc[adult1_id]
            adult2 = adults.loc[adult2_id]
            
            logger.info(f"Processing MFS filers: {adult1_id} and {adult2_id}")
            
            # Get available dependents for each adult using set operations
            deps1 = dependents.get(adult1_id, set()) - claimed_dependents
            deps2 = dependents.get(adult2_id, set()) - claimed_dependents
            
            logger.debug(f"  {adult1_id} has {len(deps1)} available dependents")
            logger.debug(f"  {adult2_id} has {len(deps2)} available dependents")
            
            # Create tax units for each MFS filer
            for adult_id, deps in [(adult1_id, deps1), (adult2_id, deps2)]:
                tax_unit = self._create_single_filer(
                    adults.loc[adult_id], 
                    hh_group, 
                    hh_data, 
                    list(deps), 
                    filing_status='married_filing_separate'
                )
                
                if tax_unit:
                    num_deps = len(tax_unit['dependents'])
                    logger.info(f"  Created MFS tax unit for {adult_id} with {num_deps} dependents")
                    tax_units.append(tax_unit)
                    claimed_dependents.update(tax_unit['dependents'])
            
            # Mark both adults as processed
            processed_adults.update([adult1_id, adult2_id])
        
        # Process joint filers with optimized set operations
        for adult1_id, adult2_id in joint_filers:
            if adult1_id in processed_adults or adult2_id in processed_adults:
                continue
                
            adult1 = adults.loc[adult1_id]
            adult2 = adults.loc[adult2_id]
            
            logger.info(f"Processing joint filers: {adult1_id} and {adult2_id}")
            
            # Get all available dependents using set operations
            deps1 = dependents.get(adult1_id, set())
            deps2 = dependents.get(adult2_id, set())
            available_deps = list((deps1 | deps2) - claimed_dependents)
            
            logger.debug(f"  Potential dependents: {len(deps1 | deps2)}, Available: {len(available_deps)}")
            
            # Create joint tax unit
            tax_unit = self._create_joint_filer(adult1, adult2, hh_group, hh_data, available_deps)
            if tax_unit:
                num_deps = len(tax_unit['dependents'])
                logger.info(f"  Created joint tax unit with {num_deps} dependents")
                tax_units.append(tax_unit)
                claimed_dependents.update(tax_unit['dependents'])
                processed_adults.update([adult1_id, adult2_id])
        
        # Process remaining adults as single or head of household filers
        remaining_adult_ids = set(adults.index) - processed_adults
        logger.info(f"Processing {len(remaining_adult_ids)} remaining adults as single/HoH filers")
        
        # Create a list to store potential HoH filers
        potential_hoh = []
        
        # First pass: Identify potential HoH filers
        for adult_id in remaining_adult_ids:
            adult_deps = dependents.get(adult_id, set()) - claimed_dependents
            if adult_deps:  # Only consider adults with unclaimed dependents
                potential_hoh.append((adult_id, adult_deps))
        
        # Sort potential HoH by number of dependents (most first)
        potential_hoh.sort(key=lambda x: len(x[1]), reverse=True)
        
        # Process HoH filers
        for adult_id, deps in potential_hoh:
            if adult_id in processed_adults:
                continue
                
            tax_unit = self._create_single_filer(
                adults.loc[adult_id], 
                hh_group, 
                hh_data, 
                list(deps)
            )
            
            if tax_unit and tax_unit['filing_status'] == 'head_of_household':
                num_deps = len(tax_unit['dependents'])
                logger.info(f"Created HoH tax unit for {adult_id} with {num_deps} dependents")
                tax_units.append(tax_unit)
                claimed_dependents.update(tax_unit['dependents'])
                processed_adults.add(adult_id)
        
        # Process remaining adults as single filers (vectorized where possible)
        remaining_adult_ids = set(adults.index) - processed_adults
        for adult_id in remaining_adult_ids:
            adult = adults.loc[adult_id]
            deps = list(dependents.get(adult_id, set()) - claimed_dependents)
            
            tax_unit = self._create_single_filer(
                adult,
                hh_group,
                hh_data,
                deps,
                'single'
            )
            
            if tax_unit:
                num_deps = len(tax_unit['dependents'])
                logger.debug(f"Created single filer tax unit for {adult_id} with {num_deps} dependents")
                tax_units.append(tax_unit)
                claimed_dependents.update(tax_unit['dependents'])
                processed_adults.add(adult_id)
        
        # Assign any remaining unclaimed dependents
        unclaimed_deps = all_dependents - claimed_dependents
        
        if unclaimed_deps:
            logger.warning(f"Household {hh_id} has {len(unclaimed_deps)} unassigned dependents")
            
            # Create a list of (tax_unit_idx, dependent_id) pairs that can be claimed
            assignments = []
            
            for dep_id in unclaimed_deps:
                dependent = hh_group.loc[dep_id]
                
                # Find the first tax unit that can claim this dependent
                for idx, tax_unit in enumerate(tax_units):
                    if self._can_claim_dependent(tax_unit, dependent, hh_group):
                        assignments.append((idx, dep_id))
                        break
            
            # Apply all valid assignments
            for idx, dep_id in assignments:
                tax_units[idx]['dependents'].append(dep_id)
                tax_units[idx]['num_dependents'] += 1
                claimed_dependents.add(dep_id)
                logger.info(f"Assigned unclaimed dependent {dep_id} to tax unit {tax_units[idx]['filer_id']}")
        
        # Final validation and logging
        unassigned_adults = set(adults.index) - processed_adults
        num_assigned_deps = len(claimed_dependents)
        
        logger.info(
            f"Household {hh_id} summary: "
            f"{len(tax_units)} tax units, "
            f"{num_assigned_deps}/{len(all_dependents)} dependents assigned, "
            f"{len(unassigned_adults)}/{len(adults)} adults unassigned"
        )
        
        if unassigned_adults:
            logger.warning(f"  WARNING: {len(unassigned_adults)} adults not assigned to any tax unit")
        
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
        
        # Get all adult IDs
        adult_ids = list(adults.index)
        
        # Look for married couples using PUMS relationship codes
        # RELSHIPP values: 20=Householder, 21=Spouse, 22-24=Children
        householder = None
        
        # First find the householder (RELSHIPP == 20)
        for id1 in adult_ids:
            person = adults.loc[id1]
            if person.get('RELSHIPP') == 20:  # Householder
                householder = (id1, person)
                logger.debug(f"Found householder: {id1}")
                break
        
        # If we found a householder, look for their spouse
        if householder:
            id1, person1 = householder
            if person1.get('MAR') == 1:  # 1 = Married
                # Look for spouse (RELSHIPP == 21)
                for id2 in adult_ids:
                    if id2 == id1:
                        continue
                    person2 = adults.loc[id2]
                    if person2.get('RELSHIPP') == 21:  # Spouse
                        # Check if they should file separately
                        if self._should_file_separately(person1, person2, hh_members):
                            mfs_filers.append((id1, id2))
                            logger.debug(f"  Identified MFS filers: {id1} and {id2}")
                        else:
                            joint_filers.append((id1, id2))
                            logger.debug(f"  Identified joint filers: {id1} and {id2}")
                        processed.update([id1, id2])
                        break
        
        # Look for other potential married couples (not householder/spouse)
        for i, id1 in enumerate(adult_ids):
            if id1 in processed:
                continue
                
            person1 = adults.loc[id1]
            
            # Skip if not married
            if person1.get('MAR') != 1:  # 1 = Married
                continue
                
            # Look for potential spouse
            for j in range(i + 1, len(adult_ids)):
                id2 = adult_ids[j]
                if id2 in processed:
                    continue
                    
                person2 = adults.loc[id2]
                
                # Check if they're both married and around the same age
                if (person2.get('MAR') == 1 and 
                    abs(person1.get('AGEP', 0) - person2.get('AGEP', 0)) <= 10):
                    
                    # Check if they should file separately
                    if self._should_file_separately(person1, person2, hh_members):
                        mfs_filers.append((id1, id2))
                        logger.debug(f"  Identified MFS filers: {id1} and {id2}")
                    else:
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
        import hashlib
        
        # Create a stable seed from SERIALNO and SPORDER
        serialno = str(adult1.get('SERIALNO', '0'))
        sporder = str(adult1.get('SPORDER', 0))
        seed_string = f"{serialno}_{sporder}"
        seed = int(hashlib.md5(seed_string.encode()).hexdigest()[:8], 16)
        
        random.seed(seed)
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
        
        # Create tax unit with proper string IDs
        tax_unit = {
            'filer_id': f"{hh_data['SERIALNO']}_joint_{adult1.name}_{adult2.name}",
            'SERIALNO': str(hh_data['SERIALNO']),  # Ensure SERIALNO is string
            'filing_status': 'joint',
            'primary_filer_id': str(adult1.name),   # Convert to string
            'secondary_filer_id': str(adult2.name), # Convert to string
            'income': income,
            'num_dependents': len(valid_dependents),
            'dependents': [str(d) for d in valid_dependents],  # Ensure dependents are strings
            'hh_id': str(adult1['SERIALNO'])  # Ensure hh_id is string
        }
        
        logger.debug(f"Created joint tax unit: {tax_unit}")
        return tax_unit

    def _can_claim_dependent(self, tax_unit: dict, dependent: pd.Series, hh_members: pd.DataFrame) -> bool:
        """
        Check if a tax unit can claim a dependent.
        
        Args:
            tax_unit: The tax unit dictionary
            dependent: The dependent's data as a Series
            hh_members: All household members
            
        Returns:
            bool: True if the tax unit can claim the dependent
        """
        # Get the primary filer - handle missing primary_filer_id
        primary_filer_id = tax_unit.get('primary_filer_id')
        if not primary_filer_id or primary_filer_id not in hh_members.index:
            logger.warning(f"Primary filer ID {primary_filer_id} not found in household members")
            return False
            
        primary_filer = hh_members.loc[primary_filer_id]
        
        # Check relationship using correct PUMS codes
        rel_to_primary = dependent.get('RELSHIPP', 0)
        
        # Direct child/stepchild/adopted child relationship
        # PUMS codes: 22=Biological child, 23=Adopted child, 24=Stepchild
        if rel_to_primary in [22, 23, 24]:  
            return True
            
        # Grandchild relationship
        if rel_to_primary == 25:  # Grandchild
            return True
            
        # Other qualifying relative relationships
        if rel_to_primary in [26, 27, 28, 29, 30]:  # Siblings, parents, grandparents, in-laws, other relatives
            # Check if they lived together all year
            # (In PUMS, if they're in the same household, we assume they lived together all year)
            return True
            
        return False
        
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
            filing_status: Explicit filing status to use (overrides automatic determination)
            
        Returns:
            Dictionary containing tax unit information or None if not valid
        """
        if available_deps is None:
            available_deps = []
        
        # Skip if this adult is already in a tax unit
        if 'in_tax_unit' in adult and adult['in_tax_unit']:
            logger.debug(f"Adult {adult.name} is already in a tax unit, skipping")
            return None
            
        # Filter available dependents to only include those actually in the household
        # and not already in another tax unit
        valid_dependents = []
        logger.debug(f"Processing available_deps: {available_deps}")
        logger.debug(f"Household members index: {list(hh_members.index)}")
        
        for dep_id in available_deps:
            # Ensure dep_id is a string
            dep_id_str = str(dep_id)
            logger.debug(f"Checking dependent ID: {dep_id} (type: {type(dep_id)}) -> {dep_id_str}")
            
            if dep_id_str in hh_members.index:
                dep = hh_members.loc[dep_id_str]
                if 'in_tax_unit' not in dep or not dep['in_tax_unit']:
                    valid_dependents.append(dep_id_str)
                    logger.debug(f"Added valid dependent: {dep_id_str}")
            else:
                logger.warning(f"Dependent ID {dep_id_str} not found in household members index")
        
        # Calculate income (include dependents in the calculation)
        members_to_include = []
        
        # Add the adult (convert Series to dict if needed)
        if isinstance(adult, pd.Series):
            members_to_include.append(adult.to_dict())
        else:
            members_to_include.append(adult)
        
        # Add dependents if any
        if valid_dependents:
            for dep_id in valid_dependents:
                try:
                    dep_data = hh_members.loc[dep_id]
                    if isinstance(dep_data, pd.Series):
                        members_to_include.append(dep_data.to_dict())
                    else:
                        members_to_include.append(dep_data)
                except KeyError as e:
                    logger.error(f"Error accessing dependent {dep_id}: {e}")
                    logger.error(f"Available household member IDs: {list(hh_members.index)}")
                    continue
        
        try:
            income = calculate_tax_unit_income(pd.DataFrame(members_to_include))
        except Exception as e:
            logger.error(f"Error calculating income for tax unit: {e}")
            income = 0.0
        
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
        # Note: Don't require valid_dependents here because HoH qualification 
        # should be independent of dependent assignment issues
        if filing_status != 'married_filing_separate':
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
            'hh_id': adult['SERIALNO'],
            'primary_filer_id': str(adult.name)  # Add primary filer ID
        }
        
        logger.debug(f"Created tax unit: {tax_unit}")
        return tax_unit
