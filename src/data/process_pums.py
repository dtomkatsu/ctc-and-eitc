#!/usr/bin/env python3
"""
Process and clean PUMS data for CTC/EITC analysis.

This script loads the raw PUMS data, applies cleaning and preprocessing steps,
and saves the processed data for further analysis.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('process_pums.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
# Use absolute path to ensure reliability
PROJECT_ROOT = Path(__file__).parent.parent.parent  # src -> ctc-and-eitc
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw/pums"
PROCESSED_DIR = DATA_DIR / "processed"

# Create directories if they don't exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Define data types for columns to ensure proper loading
DTYPES = {
    # Person-level data types
    'SERIALNO': str,
    'SPORDER': 'int8',
    'PUMA': str,
    'AGEP': 'int8',
    'SEX': 'int8',
    'HISP': 'int16',
    'RAC1P': 'int8',
    'SCHL': 'int8',
    'DIS': 'int8',
    'WAGP': 'float32',
    'SSIP': 'float32',
    'RETP': 'float32',
    'SSP': 'float32',
    'INTP': 'float32',
    'PAP': 'float32',
    'OIP': 'float32',
    'PWGTP': 'int32',
    'NP': 'int8',
    'MAR': 'int8',
    'NOC': 'int8',
    'PINCP': 'float32',
    'RELSHIPP': 'int8',
    'SEMP': 'float32',
    'ADJINC': 'float32'
}

def load_pums_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the raw PUMS person and household data.
    
    Returns:
        Tuple containing person and household DataFrames
    """
    logger.info("Loading PUMS data...")
    
    # Load person data
    person_file = RAW_DIR / 'psam_p15.csv'
    person_df = pd.read_csv(person_file, dtype=DTYPES)
    logger.info(f"Loaded {len(person_df)} person records")
    
    # Load household data
    hh_file = RAW_DIR / 'psam_h15.csv'
    hh_df = pd.read_csv(hh_file, dtype={
        'SERIALNO': str,
        'PUMA': str,
        'HINCP': 'float32',
        'WGTP': 'int32',
        'NP': 'int8',
        'ADJINC': 'float32'  # Add ADJINC to household data
    })
    logger.info(f"Loaded {len(hh_df)} household records")
    
    return person_df, hh_df

def clean_person_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess person-level data."""
    logger.info("Cleaning person data...")
    
    # Convert income variables to 2023 dollars using ADJINC
    income_cols = ['WAGP', 'SSIP', 'RETP', 'SSP', 'INTP', 'PAP', 'OIP', 'SEMP', 'PINCP']
    for col in income_cols:
        if col in df.columns:
            df[col] = df[col] * (df['ADJINC'] / 1e6)  # ADJINC is in millionths
    
    # Create age groups
    bins = [0, 5, 13, 18, 25, 35, 45, 55, 65, 75, 125]
    labels = ['0-5', '6-13', '14-18', '19-25', '26-35', '36-45', '46-55', '56-65', '66-75', '76+']
    df['age_group'] = pd.cut(df['AGEP'], bins=bins, labels=labels, right=False)
    
    # Create flag for children (under 18)
    df['is_child'] = (df['AGEP'] < 18).astype(int)
    
    # Create flag for working-age adults (18-64)
    df['is_working_age'] = ((df['AGEP'] >= 18) & (df['AGEP'] <= 64)).astype(int)
    
    # Create flag for seniors (65+)
    df['is_senior'] = (df['AGEP'] >= 65).astype(int)
    
    # Recode sex
    df['sex'] = df['SEX'].map({1: 'Male', 2: 'Female'}).astype('category')
    
    # Recode Hispanic origin
    hisp_map = {
        1: 'Not Hispanic',
        2: 'Mexican',
        3: 'Puerto Rican',
        4: 'Cuban',
        5: 'Dominican',
        6: 'Central American',
        7: 'South American',
        8: 'Other Hispanic',
        9: 'Not Hispanic'
    }
    df['hispanic'] = df['HISP'].map(hisp_map).fillna('Not Hispanic').astype('category')
    
    # Recode race
    race_map = {
        1: 'White',
        2: 'Black',
        3: 'Native American',
        4: 'Alaska Native',
        5: 'Native American',
        6: 'Asian',
        7: 'Pacific Islander',
        8: 'Other',
        9: 'Multiracial'
    }
    df['race'] = df['RAC1P'].map(race_map).astype('category')
    
    # Create a simplified race/ethnicity variable
    df['race_ethnicity'] = np.where(
        df['hispanic'] != 'Not Hispanic',
        'Hispanic',
        df['race'].astype(str)
    )
    df['race_ethnicity'] = pd.Categorical(df['race_ethnicity'])
    
    # Recode disability status
    df['has_disability'] = (df['DIS'] == 1).astype(int)
    
    # Recode marital status
    marital_map = {
        1: 'Married',
        2: 'Widowed',
        3: 'Divorced',
        4: 'Separated',
        5: 'Never married',
        6: 'Under 15'
    }
    df['marital_status'] = df['MAR'].map(marital_map).astype('category')
    
    return df

def clean_household_data(df: pd.DataFrame, person_df: pd.DataFrame = None) -> pd.DataFrame:
    """Clean and preprocess household-level data."""
    logger.info("Cleaning household data...")
    
    # If ADJINC is not in household data but is in person data, get it from there
    if 'ADJINC' not in df.columns and person_df is not None and 'ADJINC' in person_df.columns:
        # Get the first ADJINC value for each household
        adjinc_by_household = person_df.groupby('SERIALNO')['ADJINC'].first()
        df = df.merge(adjinc_by_household, on='SERIALNO', how='left')
    
    # Convert household income to 2023 dollars if ADJINC is available
    if 'ADJINC' in df.columns:
        df['HINCP'] = df['HINCP'] * (1000000 / df['ADJINC'])  # ADJINC is in millionths
    
    # Create income categories
    income_bins = [-np.inf, 15000, 30000, 50000, 75000, 100000, 150000, 200000, np.inf]
    income_labels = [
        'Less than $15k',
        '$15k to $30k',
        '$30k to $50k',
        '$50k to $75k',
        '$75k to $100k',
        '$100k to $150k',
        '$150k to $200k',
        '$200k+'
    ]
    df['income_group'] = pd.cut(
        df['HINCP'],
        bins=income_bins,
        labels=income_labels,
        right=False
    )
    
    # Create household size categories
    df['hh_size_group'] = pd.cut(
        df['NP'],
        bins=[0, 1, 2, 3, 4, 5, np.inf],
        labels=['1', '2', '3', '4', '5', '6+'],
        right=False
    )
    
    return df

def merge_person_household(person_df: pd.DataFrame, hh_df: pd.DataFrame) -> pd.DataFrame:
    """Merge person and household data."""
    logger.info("Merging person and household data...")
    
    # Merge person data with household data
    merged_df = pd.merge(
        person_df,
        hh_df[['SERIALNO', 'HINCP', 'income_group', 'hh_size_group']],
        on='SERIALNO',
        how='left'
    )
    
    return merged_df

def save_processed_data(person_df: pd.DataFrame, hh_df: pd.DataFrame) -> None:
    """Save processed data to parquet files."""
    logger.info("Saving processed data...")
    
    # Save person data
    person_file = PROCESSED_DIR / 'pums_person_processed.parquet'
    person_df.to_parquet(person_file, index=False)
    logger.info(f"Saved processed person data to {person_file}")
    
    # Save household data
    hh_file = PROCESSED_DIR / 'pums_household_processed.parquet'
    hh_df.to_parquet(hh_file, index=False)
    logger.info(f"Saved processed household data to {hh_file}")

def main():
    """Main function to process PUMS data."""
    try:
        # Load data
        person_df, hh_df = load_pums_data()
        
        # Clean data
        person_clean = clean_person_data(person_df)
        hh_clean = clean_household_data(hh_df, person_df)  # Pass person_df to access ADJINC
        
        # Merge person and household data
        merged_df = merge_person_household(person_clean, hh_clean)
        
        # Save processed data
        save_processed_data(merged_df, hh_clean)
        
        logger.info("Data processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error processing PUMS data: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
