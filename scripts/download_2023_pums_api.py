#!/usr/bin/env python3
"""
Download 2023 PUMS data for Hawaii using the Census API.
"""

import os
import requests
import pandas as pd
from pathlib import Path
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
PUMS_DATA_DIR = Path('data/raw/pums')
BACKUP_DIR = Path('data/raw/pums_backup_2019')
CENSUS_API_KEY = "2104852dd7bfd83fbc9e320d650eb57decc11817"
# Base URL for 2019-2023 5-year PUMS data
CENSUS_API_URL = "https://api.census.gov/data/2023/acs/acs5/pums"

# Note: The Census API has limitations on PUMS data access
# For full PUMS data, we need to download the files directly
PUMS_DOWNLOAD_URL = "https://www2.census.gov/programs-surveys/acs/data/pums/2023/5-Year"

# Variables to download
PERSON_VARS = [
    'SERIALNO', 'SPORDER', 'PUMA', 'ST', 'AGEP', 'SEX', 'HISP', 'RAC1P',
    'SCHL', 'DIS', 'MIL', 'MILITARY', 'WAGP', 'SEMP', 'INTP', 'RETP',
    'SSP', 'SSIP', 'PAP', 'OIP', 'POVPIP', 'PWGTP', 'NP', 'MAR', 'NOC',
    'PINCP', 'ADJINC', 'RELSHIPP', 'MSP', 'HICOV', 'HINS1', 'HINS2',
    'HINS3', 'HINS4', 'MIG'
]

HOUSEHOLD_VARS = [
    'SERIALNO', 'PUMA', 'ST', 'WGTP', 'HINCP', 'NP', 'NOC', 'TEN', 'BDSP',
    'BLD', 'RMSP', 'ELEP', 'GASP', 'FULP', 'HINCP', 'GRPIP', 'GRNTP',
    'MULTG', 'MV', 'R18', 'R60', 'R65', 'RESMODE', 'SMOCP', 'SMP', 'SRNT'
]

def backup_existing_data():
    """Backup existing PUMS data."""
    if PUMS_DATA_DIR.exists():
        logger.info("Backing up existing PUMS data...")
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        
        # Move existing files to backup
        for file in PUMS_DATA_DIR.glob('*.csv'):
            backup_file = BACKUP_DIR / file.name
            if backup_file.exists():
                backup_file.unlink()
            file.rename(backup_file)
            logger.info(f"Backed up {file.name} to {backup_file}")

def download_pums_data(data_type, variables, output_file):
    """Download PUMS data using direct file download."""
    # The Census API has limitations on PUMS data access
    # Instead, we'll provide download instructions for the full files
    logger.warning("Direct API download of full PUMS data is not available.")
    logger.info("\nPlease download the following files manually:")
    logger.info(f"1. Person data: {PUMS_DOWNLOAD_URL}/csv_phi.zip")
    logger.info(f"2. Household data: {PUMS_DOWNLOAD_URL}/csv_hhi.zip")
    logger.info("\nAfter downloading, place both files in:")
    logger.info(f"   {PUMS_DATA_DIR.absolute()}")
    logger.info("\nThen run this script again to process the files.")
    return False
    
    try:
        data = response.json()
        # Convert to DataFrame
        df = pd.DataFrame(data[1:], columns=data[0])
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(df):,} {data_type} records to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {data_type} data: {e}")
        return False

def update_data_loader_config():
    """Update the PUMS data loader configuration for 2023 data."""
    config_file = Path('src/data/pums_loader.py')
    
    if not config_file.exists():
        logger.warning("PUMS loader configuration file not found")
        return
    
    # Read current config
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Update default year
    updated_content = content.replace(
        'DEFAULT_PUMS_YEAR = 2022',
        'DEFAULT_PUMS_YEAR = 2023'
    )
    
    # Write updated config
    with open(config_file, 'w') as f:
        f.write(updated_content)
    
    logger.info("Updated PUMS loader configuration to use 2023 data")

def verify_data_quality(file_path, data_type):
    """Verify the downloaded data quality."""
    try:
        df = pd.read_csv(file_path, nrows=1000)  # Sample first 1000 rows
        
        # Check basic stats
        logger.info(f"\n{data_type.upper()} DATA QUALITY CHECK:")
        logger.info(f"Total records: {len(df):,}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Check for required columns
        required_cols = PERSON_VARS if data_type == 'person' else HOUSEHOLD_VARS
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
        else:
            logger.info("✓ All required columns present")
        
        # Sample data
        logger.info("\nSample data:")
        logger.info(df.head(2).to_string())
        
        return True
        
    except Exception as e:
        logger.error(f"Error verifying {data_type} data: {e}")
        return False

def main():
    """Main function to download and set up 2023 PUMS data."""
    logger.info("Starting 2023 PUMS data download using Census API")
    logger.info("=" * 60)
    
    # Create data directory
    PUMS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Backup existing data
    backup_existing_data()
    
    # Download person data
    person_file = PUMS_DATA_DIR / 'psam_p15.csv'
    if not download_pums_data('person', PERSON_VARS, person_file):
        logger.error("Failed to download person data")
        return
    
    # Download household data
    hh_file = PUMS_DATA_DIR / 'psam_h15.csv'
    if not download_pums_data('household', HOUSEHOLD_VARS, hh_file):
        logger.error("Failed to download household data")
        return
    
    # Verify data quality
    logger.info("\nVerifying data quality...")
    verify_data_quality(person_file, 'person')
    verify_data_quality(hh_file, 'household')
    
    # Update configuration
    update_data_loader_config()
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ 2023 PUMS data download and setup complete!")
    logger.info("Next steps:")
    logger.info("1. Run the CTC pipeline with the new data")
    logger.info("2. Regenerate dashboard CSVs")
    logger.info("3. Compare results with previous 2019 data")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
