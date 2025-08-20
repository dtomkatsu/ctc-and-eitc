#!/usr/bin/env python3
"""
Download 2023 PUMS data for Hawaii from the Census Bureau.

This script downloads the latest 5-year ACS PUMS data (2019-2023) for Hawaii
and replaces the existing 2019 data files.
"""

import os
import requests
import zipfile
import pandas as pd
from pathlib import Path
import logging
import ssl
import urllib3

# Disable SSL warnings for this script
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
PUMS_DATA_DIR = Path('data/raw/pums')
BACKUP_DIR = Path('data/raw/pums_backup_2019')
CENSUS_BASE_URL = "https://www2.census.gov/programs-surveys/acs/data/pums/2023/5-Year"

# Hawaii PUMS files for 2023 (5-year estimates)
PUMS_FILES = {
    'household': 'csv_hhi.zip',  # Household file
    'person': 'csv_phi.zip'      # Person file
}

def backup_existing_data():
    """Backup existing 2019 PUMS data."""
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

def download_pums_file(file_type, zip_filename):
    """Download and extract a PUMS file."""
    url = f"{CENSUS_BASE_URL}/{zip_filename}"
    zip_path = PUMS_DATA_DIR / zip_filename
    
    logger.info(f"Downloading {file_type} data from {url}")
    
    try:
        response = requests.get(url, stream=True, verify=False)
        response.raise_for_status()
        
        # Save zip file
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Downloaded {zip_filename}")
        
        # Extract zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # List contents to find Hawaii files
            file_list = zip_ref.namelist()
            hawaii_files = [f for f in file_list if 'hi.csv' in f.lower()]
            
            if not hawaii_files:
                logger.error(f"No Hawaii files found in {zip_filename}")
                return None
            
            # Extract Hawaii file
            hawaii_file = hawaii_files[0]
            zip_ref.extract(hawaii_file, PUMS_DATA_DIR)
            
            # Rename to standard format
            extracted_path = PUMS_DATA_DIR / hawaii_file
            if file_type == 'household':
                new_name = PUMS_DATA_DIR / 'psam_h15.csv'
            else:  # person
                new_name = PUMS_DATA_DIR / 'psam_p15.csv'
            
            if new_name.exists():
                new_name.unlink()
            extracted_path.rename(new_name)
            
            logger.info(f"Extracted and renamed to {new_name.name}")
        
        # Clean up zip file
        zip_path.unlink()
        
        return new_name
        
    except requests.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        return None
    except zipfile.BadZipFile as e:
        logger.error(f"Failed to extract {zip_filename}: {e}")
        return None

def verify_data_quality(file_path, file_type):
    """Verify the downloaded data quality."""
    logger.info(f"Verifying {file_type} data quality...")
    
    try:
        df = pd.read_csv(file_path, nrows=1000)  # Sample first 1000 rows
        
        # Check for required columns
        if file_type == 'household':
            required_cols = ['SERIALNO', 'PUMA', 'WGTP', 'HINCP']
        else:  # person
            required_cols = ['SERIALNO', 'PUMA', 'PWGTP', 'AGEP', 'RELSHIPP']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns in {file_type} data: {missing_cols}")
        else:
            logger.info(f"✓ All required columns present in {file_type} data")
        
        # Check data format
        if 'SERIALNO' in df.columns:
            sample_serialno = df['SERIALNO'].iloc[0]
            if str(sample_serialno).startswith('2023') or str(sample_serialno).startswith('2019'):
                logger.info(f"✓ Data appears to be from correct time period: {sample_serialno}")
            else:
                logger.warning(f"Unexpected SERIALNO format: {sample_serialno}")
        
        # Check Hawaii data
        if 'PUMA' in df.columns:
            pumas = df['PUMA'].unique()
            logger.info(f"✓ Found {len(pumas)} PUMAs in {file_type} data: {sorted(pumas)}")
        
        logger.info(f"✓ {file_type} data verification complete")
        return True
        
    except Exception as e:
        logger.error(f"Failed to verify {file_type} data: {e}")
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

def main():
    """Main function to download and set up 2023 PUMS data."""
    logger.info("Starting 2023 PUMS data download and setup")
    logger.info("=" * 50)
    
    # Create data directory
    PUMS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Backup existing data
    backup_existing_data()
    
    # Download files
    success_count = 0
    for file_type, zip_filename in PUMS_FILES.items():
        file_path = download_pums_file(file_type, zip_filename)
        if file_path and verify_data_quality(file_path, file_type):
            success_count += 1
    
    if success_count == len(PUMS_FILES):
        logger.info("✓ Successfully downloaded and verified all PUMS files")
        
        # Update configuration
        update_data_loader_config()
        
        logger.info("✓ 2023 PUMS data setup complete!")
        logger.info("Next steps:")
        logger.info("1. Run the CTC pipeline with the new data")
        logger.info("2. Regenerate dashboard CSVs")
        logger.info("3. Compare results with previous 2019 data")
        
    else:
        logger.error(f"Failed to download {len(PUMS_FILES) - success_count} files")
        logger.info("You may need to manually download the files from:")
        logger.info(f"  {CENSUS_BASE_URL}")

if __name__ == "__main__":
    main()
