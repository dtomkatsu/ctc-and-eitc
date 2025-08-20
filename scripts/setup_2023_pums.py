#!/usr/bin/env python3
"""
Setup 2023 PUMS data for Hawaii.

This script provides instructions and helper functions to download and process
the 2023 PUMS data for Hawaii.
"""

import os
import zipfile
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
PUMS_DATA_DIR = Path('data/raw/pums')
BACKUP_DIR = Path('data/raw/pums_backup_2019')

# File names
PERSON_ZIP = 'csv_phi.zip'
HOUSEHOLD_ZIP = 'csv_hhi.zip'
PERSON_CSV = 'psam_p15.csv'
HOUSEHOLD_CSV = 'psam_h15.csv'

def print_download_instructions():
    """Print instructions for downloading PUMS data."""
    print("\n" + "="*80)
    print("INSTRUCTIONS FOR DOWNLOADING 2023 PUMS DATA")
    print("="*80)
    print("\n1. Download the following files from the Census Bureau website:")
    print("   a. Person data (csv_phi.zip)")
    print("   b. Household data (csv_hhi.zip)")
    print("\n   Direct links (copy and paste into your browser):")
    print("   - https://www2.census.gov/programs-surveys/acs/data/pums/2023/5-Year/csv_phi.zip")
    print("   - https://www2.census.gov/programs-surveys/acs/data/pums/2023/5-Year/csv_hhi.zip")
    print("\n2. Save both .zip files to this directory:")
    print(f"   {PUMS_DATA_DIR.absolute()}")
    print("\n3. Run this script again to process the files")
    print("\n" + "="*80 + "\n")

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

def extract_hawaii_data():
    """Extract Hawaii data from downloaded zip files."""
    person_zip = PUMS_DATA_DIR / PERSON_ZIP
    household_zip = PUMS_DATA_DIR / HOUSEHOLD_ZIP
    
    # Check if zip files exist
    if not person_zip.exists() or not household_zip.exists():
        logger.error("Required zip files not found!")
        print_download_instructions()
        return False
    
    # Backup existing data
    backup_existing_data()
    
    # Extract person data
    logger.info(f"Extracting {person_zip}...")
    with zipfile.ZipFile(person_zip, 'r') as zip_ref:
        # Find Hawaii file (starts with 'psam_p15')
        hi_file = next((f for f in zip_ref.namelist() if f.startswith('psam_p15')), None)
        if hi_file:
            zip_ref.extract(hi_file, PUMS_DATA_DIR)
            # Rename to standard name
            (PUMS_DATA_DIR / hi_file).rename(PUMS_DATA_DIR / PERSON_CSV)
            logger.info(f"Extracted person data to {PERSON_CSV}")
    
    # Extract household data
    logger.info(f"Extracting {household_zip}...")
    with zipfile.ZipFile(household_zip, 'r') as zip_ref:
        # Find Hawaii file (starts with 'psam_h15')
        hi_file = next((f for f in zip_ref.namelist() if f.startswith('psam_h15')), None)
        if hi_file:
            zip_ref.extract(hi_file, PUMS_DATA_DIR)
            # Rename to standard name
            (PUMS_DATA_DIR / hi_file).rename(PUMS_DATA_DIR / HOUSEHOLD_CSV)
            logger.info(f"Extracted household data to {HOUSEHOLD_CSV}")
    
    # Clean up zip files
    person_zip.unlink()
    household_zip.unlink()
    
    return True

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

def verify_data():
    """Verify the extracted data."""
    person_file = PUMS_DATA_DIR / PERSON_CSV
    household_file = PUMS_DATA_DIR / HOUSEHOLD_CSV
    
    if not person_file.exists() or not household_file.exists():
        logger.error("Required CSV files not found!")
        return False
    
    # Check person data
    logger.info("Verifying person data...")
    try:
        df_person = pd.read_csv(person_file, nrows=5)
        logger.info(f"Person data sample (first 5 rows):\n{df_person}")
    except Exception as e:
        logger.error(f"Error reading person data: {e}")
        return False
    
    # Check household data
    logger.info("Verifying household data...")
    try:
        df_household = pd.read_csv(household_file, nrows=5)
        logger.info(f"Household data sample (first 5 rows):\n{df_household}")
    except Exception as e:
        logger.error(f"Error reading household data: {e}")
        return False
    
    return True

def main():
    """Main function to set up 2023 PUMS data."""
    print("\n" + "="*80)
    print("SETUP 2023 PUMS DATA FOR HAWAII")
    print("="*80 + "\n")
    
    # Ensure data directory exists
    PUMS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Show download instructions first
    print_download_instructions()
    
    # Check if zip files exist
    person_zip = PUMS_DATA_DIR / PERSON_ZIP
    household_zip = PUMS_DATA_DIR / HOUSEHOLD_ZIP
    
    if person_zip.exists() and household_zip.exists():
        print("\n" + "="*80)
        print("PROCESSING DOWNLOADED FILES...")
        print("="*80 + "\n")
        
        # Backup existing data
        backup_existing_data()
        
        # Extract data from zip files
        if extract_hawaii_data() and verify_data():
            update_data_loader_config()
            print("\n" + "="*80)
            print("✓ 2023 PUMS DATA SETUP COMPLETE!")
            print("="*80)
            print("\nNext steps:")
            print("1. Run the CTC pipeline with the new data")
            print("2. Regenerate dashboard CSVs")
            print("3. Compare results with previous 2019 data")
            print("\n" + "="*80 + "\n")
    
    # Check if data is already set up
    person_file = PUMS_DATA_DIR / PERSON_CSV
    household_file = PUMS_DATA_DIR / HOUSEHOLD_CSV
    
    if person_file.exists() and household_file.exists():
        print("\n" + "="*80)
        print("2023 PUMS DATA IS ALREADY SET UP!")
        print("="*80 + "\n")
        if verify_data():
            print("✓ Data verification successful!")
            print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
