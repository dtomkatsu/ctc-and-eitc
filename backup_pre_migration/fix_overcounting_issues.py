#!/usr/bin/env python3
"""
Comprehensive script to identify and fix overcounting issues in tax unit construction.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import logging
from collections import defaultdict
from src.data.pums_loader import PUMSDataLoader
from src.tax.units.constructor import TaxUnitConstructor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_data_integrity():
    """Analyze data integrity issues that could cause overcounting."""
    logger.info("Analyzing data integrity...")
    
    # Load data
    pums_loader = PUMSDataLoader()
    person_df, hh_df = pums_loader.load_data()
    
    logger.info(f"Loaded {len(person_df):,} persons and {len(hh_df):,} households")
    
    # Check for households without persons
    person_households = set(person_df['SERIALNO'].unique())
    hh_households = set(hh_df['SERIALNO'].unique())
    
    households_without_persons = hh_households - person_households
    persons_without_households = person_households - hh_households
    
    logger.info(f"Households without persons: {len(households_without_persons):,}")
    logger.info(f"Persons without households: {len(persons_without_households):,}")
    
    if households_without_persons:
        logger.warning(f"Found {len(households_without_persons)} households without person records")
        # Sample some to understand the pattern
        sample_hh_without_persons = list(households_without_persons)[:5]
        for hh_id in sample_hh_without_persons:
            hh_data = hh_df[hh_df['SERIALNO'] == hh_id].iloc[0]
            logger.info(f"  Household {hh_id}: NP={hh_data.get('NP', 'N/A')}, WGTP={hh_data.get('WGTP', 'N/A')}")
    
    # Check for duplicate person IDs
    person_df['person_id'] = person_df['SERIALNO'].astype(str) + '_' + person_df['SPORDER'].astype(str)
    duplicate_persons = person_df['person_id'].duplicated().sum()
    logger.info(f"Duplicate person IDs: {duplicate_persons}")
    
    # Check for zero-weight households
    zero_weight_hh = (hh_df['WGTP'] == 0).sum()
    logger.info(f"Zero-weight households: {zero_weight_hh:,}")
    
    return person_df, hh_df, households_without_persons

def test_tax_unit_construction_integrity(person_df, hh_df, households_without_persons):
    """Test tax unit construction for integrity issues."""
    logger.info("Testing tax unit construction integrity...")
    
    # Filter out households without persons to prevent processing empty households
    valid_households = set(person_df['SERIALNO'].unique())
    hh_df_filtered = hh_df[hh_df['SERIALNO'].isin(valid_households)].copy()
    
    logger.info(f"Filtered households: {len(hh_df_filtered):,} (removed {len(hh_df) - len(hh_df_filtered):,} without persons)")
    
    # Create constructor with filtered data
    constructor = TaxUnitConstructor(person_df, hh_df_filtered, batch_size=500, num_processes=1)
    
    # Track household processing
    processed_households = set()
    
    # Override the _process_household method to track processing
    original_process_household = constructor._process_household
    
    def tracked_process_household(hh_group):
        hh_id = hh_group['SERIALNO'].iloc[0]
        if hh_id in processed_households:
            logger.error(f"DUPLICATE PROCESSING: Household {hh_id} processed multiple times!")
        processed_households.add(hh_id)
        return original_process_household(hh_group)
    
    constructor._process_household = tracked_process_household
    
    # Create tax units
    tax_units_df = constructor.create_rule_based_units(parallel=False)
    
    logger.info(f"Created {len(tax_units_df):,} tax units from {len(processed_households):,} households")
    
    # Check for duplicate processing
    expected_households = len(person_df['SERIALNO'].unique())
    if len(processed_households) != expected_households:
        logger.error(f"Household processing mismatch: processed {len(processed_households)}, expected {expected_households}")
    
    # Check for duplicate tax unit IDs
    duplicate_tax_units = tax_units_df['filer_id'].duplicated().sum()
    if duplicate_tax_units > 0:
        logger.error(f"Found {duplicate_tax_units} duplicate tax unit IDs!")
    
    # Analyze filing status distribution
    filing_status_counts = tax_units_df['filing_status'].value_counts()
    logger.info("Filing status distribution:")
    for status, count in filing_status_counts.items():
        pct = count / len(tax_units_df) * 100
        logger.info(f"  {status}: {count:,} ({pct:.1f}%)")
    
    return tax_units_df, processed_households

def validate_household_coverage(person_df, tax_units_df):
    """Validate that all persons are properly assigned to tax units."""
    logger.info("Validating household coverage...")
    
    # Get all person IDs from original data
    all_person_ids = set(person_df['SERIALNO'].astype(str) + '_' + person_df['SPORDER'].astype(str))
    
    # Get all person IDs from tax units (primary filers + dependents)
    covered_person_ids = set()
    
    for _, tax_unit in tax_units_df.iterrows():
        # Add primary filer
        if pd.notna(tax_unit['primary_filer_id']):
            covered_person_ids.add(str(tax_unit['primary_filer_id']))
        
        # Add secondary filer if exists
        if pd.notna(tax_unit.get('secondary_filer_id')):
            covered_person_ids.add(str(tax_unit['secondary_filer_id']))
        
        # Add dependents
        dependents = tax_unit['dependents']
        if pd.notna(dependents) and dependents is not None:
            if isinstance(dependents, (list, tuple)):
                if len(dependents) > 0:  # Check length instead of truth value
                    covered_person_ids.update(str(dep_id) for dep_id in dependents)
            elif isinstance(dependents, str) and dependents:
                covered_person_ids.add(str(dependents))
    
    # Find uncovered persons
    uncovered_persons = all_person_ids - covered_person_ids
    overcovered_persons = covered_person_ids - all_person_ids
    
    logger.info(f"Total persons in data: {len(all_person_ids):,}")
    logger.info(f"Persons covered by tax units: {len(covered_person_ids):,}")
    logger.info(f"Uncovered persons: {len(uncovered_persons):,}")
    logger.info(f"Overcovered persons (shouldn't exist): {len(overcovered_persons):,}")
    
    if uncovered_persons:
        logger.warning(f"Found {len(uncovered_persons)} uncovered persons")
        # Sample some uncovered persons
        sample_uncovered = list(uncovered_persons)[:5]
        for person_id in sample_uncovered:
            serialno, sporder = person_id.split('_')
            person_data = person_df[(person_df['SERIALNO'] == serialno) & 
                                   (person_df['SPORDER'] == int(sporder))]
            if not person_data.empty:
                person = person_data.iloc[0]
                logger.info(f"  Uncovered person {person_id}: Age={person['AGEP']}, Relationship={person['RELSHIPP']}")
    
    if overcovered_persons:
        logger.error(f"Found {len(overcovered_persons)} overcovered persons - this indicates a bug!")

def apply_weights_correctly(tax_units_df, hh_df):
    """Apply PUMS weights correctly to tax units."""
    logger.info("Applying PUMS weights correctly...")
    
    # Merge household weights
    tax_units_with_weights = tax_units_df.merge(
        hh_df[['SERIALNO', 'WGTP']], 
        left_on='hh_id', 
        right_on='SERIALNO', 
        how='left'
    )
    
    # Check for missing weights
    missing_weights = tax_units_with_weights['WGTP'].isna().sum()
    if missing_weights > 0:
        logger.warning(f"Found {missing_weights} tax units with missing weights")
    
    # Exclude zero-weight households (group quarters)
    zero_weight_units = (tax_units_with_weights['WGTP'] == 0).sum()
    logger.info(f"Zero-weight tax units (group quarters): {zero_weight_units:,}")
    
    # Filter to positive weights only
    weighted_tax_units = tax_units_with_weights[tax_units_with_weights['WGTP'] > 0].copy()
    
    logger.info(f"Tax units with positive weights: {len(weighted_tax_units):,}")
    
    # Calculate weighted totals
    total_weighted_units = weighted_tax_units['WGTP'].sum()
    logger.info(f"Total weighted tax units: {total_weighted_units:,}")
    
    # Weighted filing status distribution
    weighted_filing_status = weighted_tax_units.groupby('filing_status')['WGTP'].sum()
    logger.info("Weighted filing status distribution:")
    for status, weight in weighted_filing_status.items():
        pct = weight / total_weighted_units * 100
        logger.info(f"  {status}: {weight:,.0f} ({pct:.1f}%)")
    
    return weighted_tax_units

def main():
    """Main function to identify and fix overcounting issues."""
    logger.info("Starting comprehensive overcounting analysis and fix...")
    
    # Step 1: Analyze data integrity
    person_df, hh_df, households_without_persons = analyze_data_integrity()
    
    # Step 2: Test tax unit construction integrity
    tax_units_df, processed_households = test_tax_unit_construction_integrity(
        person_df, hh_df, households_without_persons
    )
    
    # Step 3: Validate household coverage
    validate_household_coverage(person_df, tax_units_df)
    
    # Step 4: Apply weights correctly
    weighted_tax_units = apply_weights_correctly(tax_units_df, hh_df)
    
    logger.info("Overcounting analysis complete!")
    
    return {
        'person_df': person_df,
        'hh_df': hh_df,
        'tax_units_df': tax_units_df,
        'weighted_tax_units': weighted_tax_units,
        'households_without_persons': households_without_persons,
        'processed_households': processed_households
    }

if __name__ == "__main__":
    results = main()
