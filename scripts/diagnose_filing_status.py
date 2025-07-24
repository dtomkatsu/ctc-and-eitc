#!/usr/bin/env python3
"""
Diagnostic script to analyze filing status assignment in detail.

This script examines the tax units and provides detailed breakdowns of:
1. Why individuals are classified as HOH vs Single
2. Distribution of qualifying children
3. Income patterns for HOH vs Single filers
4. Relationship patterns in households
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data():
    """Load tax units and PUMS data."""
    try:
        # Load tax units
        tax_units_path = Path('src/data/processed/tax_units_rule_based.parquet')
        tax_units = pd.read_parquet(tax_units_path)
        logger.info(f"Loaded {len(tax_units)} tax units")
        
        # Load person data
        person_path = Path('data/processed/pums_person_processed.parquet')
        person_df = pd.read_parquet(person_path)
        logger.info(f"Loaded {len(person_df)} person records")
        
        # Load household data
        hh_path = Path('data/processed/pums_household_processed.parquet')
        hh_df = pd.read_parquet(hh_path)
        logger.info(f"Loaded {len(hh_df)} household records")
        
        return tax_units, person_df, hh_df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None, None

def analyze_filing_status_distribution(tax_units, person_df):
    """Analyze the distribution of filing statuses."""
    logger.info("Analyzing filing status distribution...")
    
    # Get filing status counts
    status_counts = tax_units['filing_status'].value_counts()
    logger.info(f"Filing status distribution:\n{status_counts}")
    
    # Calculate weighted counts
    if 'weight' in tax_units.columns:
        weighted_counts = tax_units.groupby('filing_status')['weight'].sum()
        logger.info(f"Weighted filing status distribution:\n{weighted_counts}")
    
    return status_counts

def analyze_hoh_characteristics(tax_units, person_df, hh_df):
    """Analyze characteristics of Head of Household filers."""
    logger.info("Analyzing Head of Household characteristics...")
    
    hoh_units = tax_units[tax_units['filing_status'] == 'head_of_household'].copy()
    single_units = tax_units[tax_units['filing_status'] == 'single'].copy()
    
    logger.info(f"HOH units: {len(hoh_units)}")
    logger.info(f"Single units: {len(single_units)}")
    
    # Analyze number of dependents
    logger.info("\nDependents distribution for HOH filers:")
    dependents_dist = hoh_units['num_dependents'].value_counts().sort_index()
    logger.info(f"{dependents_dist}")
    
    logger.info("\nDependents distribution for Single filers:")
    single_dependents_dist = single_units['num_dependents'].value_counts().sort_index()
    logger.info(f"{single_dependents_dist}")
    
    # Analyze income patterns
    logger.info(f"\nHOH median income: ${hoh_units['income'].median():,.0f}")
    logger.info(f"Single median income: ${single_units['income'].median():,.0f}")
    
    # Analyze household income vs personal income for HOH
    hoh_with_hh = hoh_units.merge(hh_df[['SERIALNO', 'HINCP']], on='SERIALNO', how='left')
    hoh_with_hh['income_ratio'] = hoh_with_hh['income'] / hoh_with_hh['HINCP'].fillna(1)
    
    logger.info(f"\nHOH personal income as % of household income:")
    logger.info(f"  Mean: {hoh_with_hh['income_ratio'].mean():.2%}")
    logger.info(f"  Median: {hoh_with_hh['income_ratio'].median():.2%}")
    logger.info(f"  % with >50% of HH income: {(hoh_with_hh['income_ratio'] > 0.5).mean():.1%}")
    
    return hoh_units, single_units

def analyze_marital_status(tax_units, person_df):
    # Analyze marital status patterns
    logger.info("\nAnalyzing marital status patterns...")
    
    # Get marital status for primary filers
    logger.info(f"Sample primary_filer_id values: {tax_units['primary_filer_id'].head()}")
    logger.info(f"Person df index: {person_df.index[:5]}")
    
    # Try to merge with person data to get marital status
    try:
        # First check if we can merge on the index (person_id)
        merged = tax_units.merge(
            person_df[['MAR']], 
            left_on='primary_filer_id', 
            right_index=True, 
            how='left',
            suffixes=('', '_primary')
        )
        
        # For joint filers, get marital status of second adult
        joint_filers = merged[merged['filing_status'] == 'joint'].copy()
        if not joint_filers.empty and 'secondary_filer_id' in joint_filers.columns:
            merged = merged.merge(
                person_df[['MAR']], 
                left_on='secondary_filer_id', 
                right_index=True, 
                how='left',
                suffixes=('', '_secondary')
            )
            
            # Check for consistency in marital status
            joint_filers = merged[merged['filing_status'] == 'joint']
            if 'MAR' in joint_filers.columns and 'MAR_secondary' in joint_filers.columns:
                inconsistent = joint_filers[joint_filers['MAR'] != joint_filers['MAR_secondary']]
                logger.info(f"Found {len(inconsistent)} joint filers with inconsistent marital status")
    except Exception as e:
        logger.error(f"Error analyzing marital status: {str(e)}")
    
    # Simple count by marital status for primary filers
    if 'MAR' in merged.columns:
        logger.info("\nMarital status distribution for primary filers:")
        logger.info(merged['MAR'].value_counts(dropna=False))

def analyze_household_composition(tax_units, person_df, hh_df):
    """Analyze household composition patterns."""
    logger.info("Analyzing household composition...")
    
    # Get household sizes
    hh_sizes = person_df.groupby('SERIALNO').size().reset_index(name='hh_size')
    
    # Merge with tax units
    units_with_size = tax_units.merge(hh_sizes, on='SERIALNO', how='left')
    
    # Analyze by filing status
    for status in ['single', 'head_of_household', 'joint']:
        status_units = units_with_size[units_with_size['filing_status'] == status]
        if len(status_units) > 0:
            logger.info(f"\n{status.upper()} filers household size distribution:")
            size_dist = status_units['hh_size'].value_counts().sort_index()
            logger.info(f"{size_dist}")

def main():
    """Main diagnostic function."""
    logger.info("Starting filing status diagnostic analysis...")
    
    # Load data
    tax_units, person_df, hh_df = load_data()
    if tax_units is None:
        return 1
    
    # Run analyses
    analyze_filing_status_distribution(tax_units, person_df)
    analyze_hoh_characteristics(tax_units, person_df, hh_df)
    analyze_marital_status(tax_units, person_df)
    analyze_household_composition(tax_units, person_df, hh_df)
    
    logger.info("Diagnostic analysis complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
