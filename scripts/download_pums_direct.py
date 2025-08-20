#!/usr/bin/env python3
"""
Directly download 2023 PUMS data for Hawaii.
"""

import os
import sys
import requests
import zipfile
import pandas as pd
from pathlib import Path
import logging
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
PUMS_DATA_DIR = Path('data/raw/pums')
BACKUP_DIR = Path('data/raw/pums_backup_2019')

# File URLs
PUMS_FILES = {
    'person': {
        'url': 'https://www2.census.gov/programs-surveys/acs/data/pums/2023/5-Year/csv_phi.zip',
        'zip_name': 'csv_phi.zip',
        'extract_name': 'psam_p15.csv'
    },
    'household': {
        'url': 'https://www2.census.gov/programs-surveys/acs/data/pums/2023/5-Year/csv_hhi.zip',
        'zip_name': 'csv_hhi.zip',
        'extract_name': 'psam_h15.csv'
    }
}

def create_session():
    """Create a requests session with retry logic."""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def download_file(url, local_path):
    """Download a file with progress tracking."""
    session = create_session()
    
    # Stream the download to handle large files
    with session.get(url, stream=True, verify=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        # Create parent directory if it doesn't exist
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress
        with open(local_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Show progress
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rDownloading: {downloaded/1024/1024:.1f}MB / {total_size/1024/1024:.1f}MB ({percent:.1f}%)", end='')
            print()  # New line after progress
    
    return local_path

def extract_zip(zip_path, extract_to, expected_file):
    """Extract a zip file and find the Hawaii data file."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Find the Hawaii file (starts with psam_p15 or psam_h15)
        for file in zip_ref.namelist():
            if file.startswith(expected_file.split('.')[0]):
                logger.info(f"Extracting {file}...")
                zip_ref.extract(file, extract_to)
                # Rename to standard name
                extracted_file = extract_to / file
                target_file = extract_to / expected_file
                if extracted_file != target_file:
                    if target_file.exists():
                        target_file.unlink()
                    extracted_file.rename(target_file)
                return target_file
    return None

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
    print("\n" + "="*80)
    print("DOWNLOADING 2023 PUMS DATA FOR HAWAII")
    print("="*80 + "\n")
    
    # Ensure data directory exists
    PUMS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Backup existing data
    backup_existing_data()
    
    # Download and process each file
    for file_type, file_info in PUMS_FILES.items():
        zip_path = PUMS_DATA_DIR / file_info['zip_name']
        csv_path = PUMS_DATA_DIR / file_info['extract_name']
        
        # Skip if already extracted
        if csv_path.exists():
            logger.info(f"{file_type.capitalize()} data already exists at {csv_path}")
            continue
        
        # Download the file
        logger.info(f"Downloading {file_type} data from {file_info['url']}")
        try:
            download_file(file_info['url'], zip_path)
            
            # Extract the file
            logger.info(f"Extracting {file_type} data...")
            extracted_file = extract_zip(zip_path, PUMS_DATA_DIR, file_info['extract_name'])
            
            if extracted_file and extracted_file.exists():
                logger.info(f"Successfully extracted {file_type} data to {extracted_file}")
                # Remove the zip file after extraction
                zip_path.unlink()
            else:
                logger.error(f"Failed to extract {file_type} data from {zip_path}")
                
        except Exception as e:
            logger.error(f"Error processing {file_type} data: {e}")
            if zip_path.exists():
                zip_path.unlink()  # Clean up partial downloads
            continue
    
    # Update configuration
    update_data_loader_config()
    
    print("\n" + "="*80)
    print("✓ 2023 PUMS DATA DOWNLOAD AND SETUP COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("1. Run the CTC pipeline with the new data")
    print("2. Regenerate dashboard CSVs")
    print("3. Compare results with previous 2019 data")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
