#!/usr/bin/env python3
"""
Download PUMS data for CTC and EITC Analysis

This script downloads Public Use Microdata Sample (PUMS) data from the U.S. Census Bureau
for Hawaii, focusing on variables relevant for Child Tax Credit (CTC) and Earned Income
Tax Credit (EITC) analysis.

Usage:
    python scripts/download_pums.py [--year YYYY] [--state XX] [--api-key KEY]

Example:
    python scripts/download_pums.py --year 2022 --state 15 --api-key YOUR_CENSUS_API_KEY
"""

import os
import sys
import argparse
import logging
import requests
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_pums.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_YEAR = 2023  # 2023 5-year PUMS data
DEFAULT_STATE = '15'  # Hawaii FIPS code
DEFAULT_DATA_DIR = Path("data/raw/pums")
DEFAULT_API_KEY = os.getenv('CENSUS_API_KEY')

# Maximum number of variables per API request (Census API has limits)
MAX_VARS_PER_REQUEST = 45

class PUMSDownloader:
    """Handles downloading PUMS data from the Census API."""
    
    def __init__(self, year: int = DEFAULT_YEAR, 
                 state: str = DEFAULT_STATE,
                 data_dir: Path = DEFAULT_DATA_DIR,
                 api_key: Optional[str] = DEFAULT_API_KEY):
        """Initialize the PUMS downloader.
        
        Args:
            year: Year of data to download
            state: State FIPS code
            data_dir: Directory to save downloaded files
            api_key: Census API key (or None to use environment variable)
        """
        self.year = year
        self.state = state
        self.data_dir = Path(data_dir)
        self.api_key = api_key
        
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Base URL for the Census PUMS API
        self.base_url = f"https://api.census.gov/data/{self.year}/acs/acs5/pums"
        
        # Define variables needed for CTC/EITC analysis
        # Using 2023 5-year PUMS variable names
        # Note: Using only the most essential variables to avoid API errors
        self.person_vars = [
            # Identification
            'SERIALNO', 'SPORDER', 'PUMA', 'ST',
            
            # Basic demographics
            'AGEP', 'SEX', 'HISP', 'RAC1P', 'SCHL',
            
            # Income sources
            'WAGP', 'SEMP', 'INTP', 'RETP', 'SSP', 'SSIP',
            
            # Family/household
            'MSP', 'NOC', 'PINCP', 'RELSHIPP',
            
            # Weights and basic adjustments
            'PWGTP', 'ADJINC'
        ]
        
        self.household_vars = [
            'SERIALNO', 'PUMA', 'ST', 'HINCP', 'ADJINC', 'WGTP', 'NP', 'TEN'
        ]
    
    def _make_api_request(self, dataset_type: str, variables: List[str]) -> Optional[pd.DataFrame]:
        """Make a request to the Census PUMS API.
        
        Args:
            dataset_type: 'person' or 'household'
            variables: List of variables to request
            
        Returns:
            DataFrame with the requested data, or None if the request failed
        """
        # Always include SERIALNO for joining
        if 'SERIALNO' not in variables:
            variables = ['SERIALNO'] + variables
        
        # Split variables into chunks to avoid URL length limits
        chunks = [variables[i:i + MAX_VARS_PER_REQUEST] 
                 for i in range(0, len(variables), MAX_VARS_PER_REQUEST)]
        
        result_df = None
        
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Fetching {dataset_type} data chunk {i}/{len(chunks)} with {len(chunk)} variables...")
            
            params = {
                'get': ','.join(chunk),
                'for': f'state:{self.state}',
                'key': self.api_key
            }
            
            try:
                response = requests.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                
                # Parse JSON response
                data = response.json()
                headers = data[0]
                rows = data[1:]
                
                # Create DataFrame for this chunk
                chunk_df = pd.DataFrame(rows, columns=headers)
                
                # Convert numeric columns
                for col in chunk_df.columns:
                    if col != 'SERIALNO':
                        try:
                            chunk_df[col] = pd.to_numeric(chunk_df[col], errors='ignore')
                        except:
                            pass
                
                # Merge with previous chunks
                if result_df is None:
                    result_df = chunk_df
                else:
                    # Merge on SERIALNO and state
                    merge_cols = list(set(result_df.columns) & set(chunk_df.columns))
                    result_df = pd.merge(result_df, chunk_df, on=merge_cols, how='outer')
                
                logger.info(f"Retrieved {len(chunk_df)} records")
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching data: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    logger.error(f"Status code: {e.response.status_code}")
                    logger.error(f"Response: {e.response.text[:500]}")
                continue
        
        return result_df
    
    def download_person_data(self) -> Optional[pd.DataFrame]:
        """Download person-level PUMS data."""
        logger.info(f"Downloading person data for {self.state} ({self.year})...")
        return self._make_api_request('person', self.person_vars)
    
    def download_household_data(self) -> Optional[pd.DataFrame]:
        """Download household-level PUMS data."""
        logger.info(f"Downloading household data for {self.state} ({self.year})...")
        return self._make_api_request('household', self.household_vars)
    
    def save_data(self, df: pd.DataFrame, filename: str) -> Path:
        """Save DataFrame to a CSV file.
        
        Args:
            df: DataFrame to save
            filename: Output filename (will be saved in data_dir)
            
        Returns:
            Path to the saved file
        """
        if df is None or df.empty:
            logger.warning(f"No data to save for {filename}")
            return None
            
        filepath = self.data_dir / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} records to {filepath}")
        return filepath
    
    def download_all(self) -> bool:
        """Download both person and household data.
        
        Returns:
            True if both downloads were successful, False otherwise
        """
        if not self.api_key:
            logger.warning("No Census API key provided. Using public access which has rate limits.")
        
        success = True
        
        # Download person data
        person_df = self.download_person_data()
        if person_df is not None:
            self.save_data(person_df, f"psam_p{self.state}.csv")
        else:
            success = False
            logger.error("Failed to download person data")
        
        # Download household data
        household_df = self.download_household_data()
        if household_df is not None:
            self.save_data(household_df, f"psam_h{self.state}.csv")
        else:
            success = False
            logger.error("Failed to download household data")
        
        return success

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Download PUMS data for CTC/EITC analysis')
    parser.add_argument('--year', type=int, default=DEFAULT_YEAR,
                       help=f'Year of data to download (default: {DEFAULT_YEAR})')
    parser.add_argument('--state', type=str, default=DEFAULT_STATE,
                       help=f'State FIPS code (default: {DEFAULT_STATE} for Hawaii)')
    parser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR,
                       help=f'Directory to save data (default: {DEFAULT_DATA_DIR})')
    parser.add_argument('--api-key', type=str, default=DEFAULT_API_KEY,
                       help='Census API key (default: CENSUS_API_KEY environment variable)')
    return parser.parse_args()

def main():
    """Main function to run the script."""
    args = parse_args()
    
    downloader = PUMSDownloader(
        year=args.year,
        state=args.state,
        data_dir=args.data_dir,
        api_key=args.api_key or DEFAULT_API_KEY
    )
    
    success = downloader.download_all()
    
    if success:
        logger.info("\nDownload completed successfully!")
        logger.info(f"Files saved to: {args.data_dir}")
    else:
        logger.error("\nDownload completed with errors. Check the log for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
