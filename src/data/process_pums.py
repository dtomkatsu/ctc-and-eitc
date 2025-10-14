#!/usr/bin/env python3
"""
Process and clean PUMS data for CTC/EITC analysis.

This script loads the raw PUMS data, applies cleaning and preprocessing steps,
and saves the processed data for further analysis.
"""

import os
import pandas as pd
import logging
import os
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np
logging
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

# Data type (1-year or 5-year)
PUMS_TYPE = '5yr'  # Using 5-year data (2018-2022)

# Create directories if they don't exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Define data types for columns to ensure proper loading
# Using 'Int64' (capital I) for nullable integer types to handle NA values
DTYPES = {
    # Person-level data types
    'SERIALNO': str,
    'SPORDER': 'Int64',
    'PUMA': str,
    'AGEP': 'Int64',
    'SEX': 'Int64',
    'HISP': 'Int64',
    'RAC1P': 'Int64',
    'SCHL': 'Int64',
    'DIS': 'Int64',
    'WAGP': 'float64',  # Using float64 for better precision with monetary values
    'SSIP': 'float64',
    'RETP': 'float64',
    'SSP': 'float64',
    'INTP': 'float64',
    'PAP': 'float64',
    'OIP': 'float64',
    'PWGTP': 'Int64',
    'NP': 'Int64',
    'MAR': 'Int64',
    'NOC': 'Int64',
    'PINCP': 'float64',
    'RELSHIPP': 'Int64',
    'SEMP': 'float64',
    'ADJINC': 'float64',
    # Additional columns that might be in the 5-year data
    'MIL': 'Int64',
    'MIG': 'Int64',
    'MILITARY': 'Int64',
    'MSP': 'Int64',
    'HICOV': 'Int64',
    'HINS1': 'Int64',
    'HINS2': 'Int64',
    'HINS3': 'Int64',
    'HINS4': 'Int64'
}

def load_pums_data(pums_type: str = PUMS_TYPE) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the raw PUMS person and household data.
    
    Args:
        pums_type: '1yr' or '5yr' for 1-year or 5-year PUMS data
        
    Returns:
        Tuple containing person and household DataFrames
    """
    logger.info(f"Loading PUMS {pums_type} data...")
    
    # For 5-year data, we'll use the existing CSV files
    ext = 'csv'
    
    try:
        # Load person data
        person_file = RAW_DIR / f'psam_p15.{ext}'
        if ext == 'parquet':
            person_df = pd.read_parquet(person_file)
        else:
            person_df = pd.read_csv(person_file, dtype=DTYPES)
        logger.info(f"Loaded {len(person_df)} person records from {person_file}")
        
        # Load household data
        hh_file = RAW_DIR / f'psam_h15.{ext}'
        if ext == 'parquet':
            hh_df = pd.read_parquet(hh_file)
        else:
            hh_df = pd.read_csv(hh_file, dtype={
                'SERIALNO': str,
                'PUMA': str,
                'HINCP': 'float32',
                'WGTP': 'int32',
                'NP': 'int8',
                'ADJINC': 'float32'
            })
        logger.info(f"Loaded {len(hh_df)} household records from {hh_file}")
        
        return person_df, hh_df
        
    except FileNotFoundError as e:
        logger.error(f"Error loading PUMS data: {e}")
        raise FileNotFoundError(
            f"Could not find PUMS data files. Expected CSV files: {person_file}, {hh_file}"
        )
    
    return person_df, hh_df

def clean_person_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess person-level data."""
    logger.info("Cleaning person data...")
    
    # Convert income variables to 2023 dollars using ADJINC
    income_cols = ['WAGP', 'SSIP', 'RETP', 'SSP', 'INTP', 'PAP', 'OIP', 'SEMP', 'PINCP']
    for col in income_cols:
        if col in df.columns:
            df[col] = df[col] * df['ADJINC']  # ADJINC is the adjustment factor to convert to 2023 dollars
    
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
        df['HINCP'] = df['HINCP'] * df['ADJINC']  # ADJINC is the adjustment factor to convert to 2023 dollars
    
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

def save_processed_data(
    person_df: pd.DataFrame, 
    hh_df: pd.DataFrame, 
    merged_df: Optional[pd.DataFrame] = None,
    pums_type: str = '5yr',
    data_year: int = 2022
) -> None:
    """
    Save processed data to parquet files.
    
    Args:
        person_df: Processed person-level data
        hh_df: Processed household-level data
        merged_df: Optional merged person-household data
        pums_type: Type of PUMS data ('1yr' or '5yr')
        data_year: Reference year for the data
    """
    logger.info("Saving processed data...")
    
    # Create a subdirectory for the data year and type
    output_dir = PROCESSED_DIR / f"pums_{pums_type}_{data_year}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save person data
    person_file = output_dir / 'pums_person_processed.parquet'
    person_df.to_parquet(person_file, index=False)
    logger.info(f"Saved processed person data to {person_file}")
    
    # Save household data
    hh_file = output_dir / 'pums_household_processed.parquet'
    hh_df.to_parquet(hh_file, index=False)
    logger.info(f"Saved processed household data to {hh_file}")
    
    # Save merged data if provided
    if merged_df is not None:
        merged_file = output_dir / 'pums_merged_processed.parquet'
        merged_df.to_parquet(merged_file, index=False)
        logger.info(f"Saved merged data to {merged_file}")
    
    # Also save a copy with a fixed name for backward compatibility
    if pums_type == '5yr' and data_year == 2022:
        for df, name in [(person_df, 'pums_person_processed.parquet'),
                        (hh_df, 'pums_household_processed.parquet'),
                        (merged_df, 'pums_merged_processed.parquet')]:
            if df is not None:
                df.to_parquet(PROCESSED_DIR / name, index=False)

def main(pums_type: str = '5yr', data_year: int = 2022) -> None:
    """
    Main function to process PUMS data.
    
    Args:
        pums_type: '1yr' or '5yr' for 1-year or 5-year PUMS data
        data_year: Reference year for the PUMS data
    """
    try:
        logger.info(f"Starting PUMS {pums_type} data processing for {data_year}...")
        
        # Load the data
        person_df, hh_df = load_pums_data(pums_type)
        
        # Clean and process the data
        person_df = clean_person_data(person_df)
        hh_df = clean_household_data(hh_df, person_df)
        
        # Merge person and household data
        merged_df = merge_person_household(person_df, hh_df)
        
        # Add year and pums_type information to the data
        for df in [person_df, hh_df, merged_df]:
            if df is not None:
                df['pums_year'] = data_year
                df['pums_type'] = pums_type
    
        # Save the processed data with metadata
        save_processed_data(
            person_df=person_df,
            hh_df=hh_df,
            merged_df=merged_df,
            pums_type=pums_type,
            data_year=data_year
        )
        
        logger.info(f"PUMS {pums_type} data processing completed successfully!")
        logger.info(f"- Processed {len(person_df):,} person records")
        logger.info(f"- Processed {len(hh_df):,} household records")
        logger.info(f"- Merged data contains {len(merged_df):,} records")
        
    except Exception as e:
        logger.error(f"Error processing PUMS {pums_type} data: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process PUMS data')
    parser.add_argument('--pums-type', 
                      type=str, 
                      default=PUMS_TYPE,
                      choices=['1yr', '5yr'],
                      help='Type of PUMS data to process (1yr or 5yr)')
    parser.add_argument('--data-year',
                      type=int,
                      default=2022,
                      help='Reference year for the PUMS data')
    args = parser.parse_args()
    
    main(pums_type=args.pums_type, data_year=args.data_year)
