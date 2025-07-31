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
    # Use hh_id for merging since SERIALNO might not be available in tax units
    merge_col = 'hh_id' if 'hh_id' in hoh_units.columns else 'SERIALNO'
    hh_merge_col = 'SERIALNO' if 'SERIALNO' in hh_df.columns else 'hh_id'
    
    logger.info(f"Merging HOH units using {merge_col} with household data using {hh_merge_col}")
    
    hoh_with_hh = hoh_units.merge(
        hh_df[['SERIALNO', 'HINCP']], 
        left_on=merge_col, 
        right_on=hh_merge_col, 
        how='left'
    )
    
    # Debug income values
    logger.info(f"\nDEBUG - Income analysis for HOH:")
    logger.info(f"  Sample HOH tax unit income: {hoh_with_hh['income'].head()}")
    logger.info(f"  Sample household income (HINCP): {hoh_with_hh['HINCP'].head()}")
    logger.info(f"  HOH income stats: mean=${hoh_with_hh['income'].mean():.0f}, median=${hoh_with_hh['income'].median():.0f}")
    logger.info(f"  HH income stats: mean=${hoh_with_hh['HINCP'].mean():.0f}, median=${hoh_with_hh['HINCP'].median():.0f}")
    
    # Calculate income ratio with better handling of edge cases
    # Only calculate ratio where both values are positive and reasonable
    valid_mask = (hoh_with_hh['income'] > 0) & (hoh_with_hh['HINCP'] > 0) & (hoh_with_hh['HINCP'] < 1e6)
    
    if valid_mask.sum() > 0:
        hoh_with_hh['income_ratio'] = hoh_with_hh['income'] / hoh_with_hh['HINCP']
        valid_ratios = hoh_with_hh.loc[valid_mask, 'income_ratio']
        
        logger.info(f"\nHOH personal income as % of household income ({valid_mask.sum()} valid cases):")
        logger.info(f"  Mean: {valid_ratios.mean():.2%}")
        logger.info(f"  Median: {valid_ratios.median():.2%}")
        logger.info(f"  % with >50% of HH income: {(valid_ratios > 0.5).mean():.1%}")
    else:
        logger.info(f"\nHOH personal income as % of household income: No valid cases found for ratio calculation")
    
    return hoh_units, single_units

def analyze_marital_status(tax_units, person_df):
    # Analyze marital status patterns
    logger.info("\nAnalyzing marital status patterns...")
    
    # The primary_filer_id in tax_units contains constructed IDs like "2019GQ0000071_1"
    # We need to extract the actual person index from these IDs
    # The format appears to be SERIALNO_SPORDER, but we need to map back to person DataFrame
    
    logger.info(f"Sample primary_filer_id values: {tax_units['primary_filer_id'].head()}")
    logger.info(f"Person df shape: {person_df.shape}")
    logger.info(f"Person df columns with SERIAL/SPORDER: SERIALNO={person_df['SERIALNO'].nunique()}, SPORDER range={person_df['SPORDER'].min()}-{person_df['SPORDER'].max()}")
    
    # Create a person lookup using SERIALNO + SPORDER combination
    person_df_lookup = person_df.copy()
    person_df_lookup['person_lookup_id'] = person_df_lookup['SERIALNO'].astype(str) + '_' + person_df_lookup['SPORDER'].astype(str)
    
    # Extract the lookup ID from primary_filer_id (remove any prefixes/suffixes if needed)
    tax_units_copy = tax_units.copy()
    
    # Debug: check if we can find matches
    sample_lookup_ids = person_df_lookup['person_lookup_id'].head(10).tolist()
    sample_filer_ids = tax_units_copy['primary_filer_id'].head(10).tolist()
    logger.info(f"Sample person lookup IDs: {sample_lookup_ids}")
    logger.info(f"Sample filer IDs: {sample_filer_ids}")
    
    try:
        # Try to merge using the person lookup
        merged = tax_units_copy.merge(
            person_df_lookup[['person_lookup_id', 'MAR']],
            left_on='primary_filer_id',
            right_on='person_lookup_id',
            how='left'
        )
        
        # Check merge success
        successful_merges = merged['MAR'].notna().sum()
        logger.info(f"Successfully merged {successful_merges}/{len(tax_units_copy)} tax units with marital status")
        
        if successful_merges > 0:
            logger.info("\nMarital status distribution for primary filers:")
            mar_counts = merged['MAR'].value_counts(dropna=False).sort_index()
            logger.info(mar_counts)
            
            # Add marital status labels for clarity
            mar_labels = {1: 'Married', 2: 'Widowed', 3: 'Divorced', 4: 'Separated', 5: 'Never married'}
            logger.info("\nMarital status with labels:")
            for code, count in mar_counts.items():
                if pd.notna(code):
                    label = mar_labels.get(int(code), f'Unknown ({int(code)})')
                    logger.info(f"  {label} ({int(code)}): {count}")
                else:
                    logger.info(f"  Missing/NaN: {count}")
        else:
            logger.warning("No successful merges found - primary_filer_id format may not match person data structure")
            
            # Try alternative approach: direct index matching if primary_filer_id is numeric
            try:
                # Check if primary_filer_id can be converted to int (direct index)
                numeric_ids = pd.to_numeric(tax_units_copy['primary_filer_id'], errors='coerce')
                valid_numeric = numeric_ids.notna().sum()
                
                if valid_numeric > 0:
                    logger.info(f"Trying direct index matching for {valid_numeric} numeric IDs")
                    tax_units_numeric = tax_units_copy[numeric_ids.notna()].copy()
                    tax_units_numeric['numeric_id'] = numeric_ids[numeric_ids.notna()].astype(int)
                    
                    merged_alt = tax_units_numeric.merge(
                        person_df[['MAR']],
                        left_on='numeric_id',
                        right_index=True,
                        how='left'
                    )
                    
                    alt_successful = merged_alt['MAR'].notna().sum()
                    logger.info(f"Alternative approach: merged {alt_successful}/{len(tax_units_numeric)} units")
                    
                    if alt_successful > 0:
                        logger.info("\nMarital status distribution (alternative approach):")
                        logger.info(merged_alt['MAR'].value_counts(dropna=False).sort_index())
                        
            except Exception as alt_e:
                logger.error(f"Alternative approach failed: {alt_e}")
            
    except Exception as e:
        logger.error(f"Error analyzing marital status: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def analyze_household_composition(tax_units, person_df, hh_df):
    """Analyze household composition patterns."""
    logger.info("Analyzing household composition...")
    
    # Debug the merge columns
    logger.info(f"Tax units columns: {list(tax_units.columns)}")
    logger.info(f"Tax units SERIALNO sample: {tax_units.get('SERIALNO', tax_units.get('hh_id', 'NOT_FOUND')).head()}")
    logger.info(f"Person df SERIALNO sample: {person_df['SERIALNO'].head()}")
    
    # Get household sizes from person data
    hh_sizes = person_df.groupby('SERIALNO').size().reset_index(name='hh_size')
    logger.info(f"Household sizes calculated: {len(hh_sizes)} households")
    
    # Determine which column to use for merging
    # Check if SERIALNO has valid values, otherwise use hh_id
    if 'SERIALNO' in tax_units.columns and tax_units['SERIALNO'].notna().sum() > 0:
        merge_col = 'SERIALNO'
    else:
        merge_col = 'hh_id'
    
    logger.info(f"Using merge column: {merge_col}")
    
    try:
        # Merge with tax units
        if merge_col in tax_units.columns:
            units_with_size = tax_units.merge(hh_sizes, left_on=merge_col, right_on='SERIALNO', how='left')
            logger.info(f"Merged {len(units_with_size)} tax units with household size data")
            
            # Check merge success
            successful_merges = units_with_size['hh_size'].notna().sum()
            logger.info(f"Successfully merged {successful_merges}/{len(tax_units)} tax units with household sizes")
            
            # Analyze by filing status
            for status in ['single', 'head_of_household', 'joint', 'married_filing_separate']:
                status_units = units_with_size[units_with_size['filing_status'] == status]
                if len(status_units) > 0:
                    logger.info(f"\n{status.upper()} filers household size distribution:")
                    size_dist = status_units['hh_size'].value_counts().sort_index()
                    logger.info(f"{size_dist}")
                else:
                    logger.info(f"\n{status.upper()} filers household size distribution: No filers found")
        else:
            logger.error(f"Cannot find merge column. Available columns: {list(tax_units.columns)}")
            
    except Exception as e:
        logger.error(f"Error in household composition analysis: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

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
