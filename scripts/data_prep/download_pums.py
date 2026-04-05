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

import io
import os
import sys
import time
import argparse
import logging
import zipfile
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
DEFAULT_YEAR = 2024  # 2024 5-year PUMS data (released Dec 2025)
DEFAULT_STATE = '15'  # Hawaii FIPS code
DEFAULT_DATA_DIR = Path("data/raw/pums")
DEFAULT_API_KEY = os.getenv('CENSUS_API_KEY')

# Maximum number of variables per API request (Census API has limits)
MAX_VARS_PER_REQUEST = 45

class PUMSDownloader:
    """Handles downloading PUMS data from the Census Bureau.

    Supports two download methods:
    - 'ftp': Download full PUMS files from Census FTP (all ~290 columns).
             This is the recommended method as the model requires columns
             not available through the API.
    - 'api': Download a subset of variables via the Census API (~24 columns).
             Only useful for quick checks; insufficient for the full model.
    """

    # Census FTP base URL for PUMS files
    FTP_BASE_URL = "https://www2.census.gov/programs-surveys/acs/data/pums"

    # FIPS code to lowercase state abbreviation (used in FTP filenames)
    FIPS_TO_ABBREV = {
        '15': 'hi',   # Hawaii
    }

    def __init__(self, year: int = DEFAULT_YEAR,
                 state: str = DEFAULT_STATE,
                 data_dir: Path = DEFAULT_DATA_DIR,
                 api_key: Optional[str] = DEFAULT_API_KEY,
                 method: str = 'ftp'):
        """Initialize the PUMS downloader.

        Args:
            year: Year of data to download
            state: State FIPS code
            data_dir: Directory to save downloaded files
            api_key: Census API key (required for 'api' method)
            method: Download method — 'ftp' (full data) or 'api' (subset)
        """
        self.year = year
        self.state = state
        self.data_dir = Path(data_dir)
        self.api_key = api_key
        self.method = method

        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Base URL for the Census PUMS API
        self.base_url = f"https://api.census.gov/data/{self.year}/acs/acs5/pums"
        
        # Person variables required for tax credit analysis
        self.person_vars = [
            # Identification and basic demographics
            'SERIALNO', 'PUMA', 'AGEP', 'SEX', 'HISP',
            'RAC1P', 'SCHL', 'DIS', 
            
            # Income sources
            'WAGP', 'SSIP', 'RETP', 'SSP', 'INTP', 'PAP', 'OIP',
            
            # Family/household
            'PWGTP', 'NP', 'MAR', 'NOC', 'PINCP', 'RELSHIPP', 'SPORDER',
            'SEMP', 'ADJINC'
        ]
        
        # Household variables required for tax credit analysis
        self.household_vars = [
            'SERIALNO', 'PUMA', 'HINCP', 'WGTP', 'NP'
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
        failed_chunks = []
        max_retries = 3

        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Fetching {dataset_type} data chunk {i}/{len(chunks)} with {len(chunk)} variables...")

            params = {
                'get': ','.join(chunk),
                'for': f'state:{self.state}',
                'key': self.api_key
            }

            chunk_success = False
            for attempt in range(max_retries + 1):
                try:
                    response = requests.get(self.base_url, params=params, timeout=30)

                    # Handle 429 rate-limit before raise_for_status
                    if response.status_code == 429:
                        if attempt < max_retries:
                            retry_after = int(response.headers.get('Retry-After', 2 * (2 ** attempt)))
                            wait = min(retry_after, 60)
                            logger.warning(
                                f"Rate limited (429) on chunk {i}, attempt {attempt + 1}/{max_retries + 1}. "
                                f"Retrying in {wait}s..."
                            )
                            time.sleep(wait)
                            continue
                        else:
                            logger.error(f"Rate limited on chunk {i} after {max_retries + 1} attempts")
                            break

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
                            except (ValueError, TypeError):
                                pass

                    # Merge with previous chunks
                    if result_df is None:
                        result_df = chunk_df
                    else:
                        result_df = pd.merge(result_df, chunk_df, on='SERIALNO', how='outer')

                    logger.info(f"Retrieved {len(chunk_df)} records")
                    chunk_success = True
                    break  # Success — exit retry loop

                except requests.exceptions.RequestException as e:
                    logger.error(f"Error fetching data: {e}")
                    if hasattr(e, 'response') and e.response is not None:
                        logger.error(f"Status code: {e.response.status_code}")
                        logger.error(f"Response: {e.response.text[:500]}")
                    break  # Non-429 errors: don't retry

            if not chunk_success:
                failed_chunks.append(i)

        if failed_chunks:
            logger.error(
                f"Failed {len(failed_chunks)}/{len(chunks)} chunks for {dataset_type}. "
                f"Returning None to avoid incomplete data."
            )
            return None

        return result_df

    def _download_ftp(self, file_type: str) -> Optional[pd.DataFrame]:
        """Download full PUMS file from Census FTP (all columns).

        Args:
            file_type: 'person' or 'household'

        Returns:
            DataFrame with all PUMS columns, or None on failure.
        """
        state_abbr = self.FIPS_TO_ABBREV.get(self.state)
        if not state_abbr:
            logger.error(
                f"No state abbreviation mapping for FIPS '{self.state}'. "
                f"Supported: {self.FIPS_TO_ABBREV}"
            )
            return None

        prefix = 'p' if file_type == 'person' else 'h'
        zip_name = f"csv_{prefix}{state_abbr}.zip"
        url = f"{self.FTP_BASE_URL}/{self.year}/5-Year/{zip_name}"

        logger.info(f"Downloading {file_type} PUMS from {url} ...")

        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            total_bytes = len(response.content)
            logger.info(f"Downloaded {total_bytes / 1_048_576:.1f} MB")

            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                csv_files = [f for f in zf.namelist() if f.lower().endswith('.csv')]
                if not csv_files:
                    logger.error(f"No CSV files found in {zip_name}")
                    return None

                csv_name = csv_files[0]
                logger.info(f"Extracting {csv_name} from {zip_name}")

                with zf.open(csv_name) as csv_file:
                    df = pd.read_csv(csv_file, dtype={'SERIALNO': str})

            logger.info(
                f"Loaded {file_type} PUMS: {len(df)} rows, {len(df.columns)} columns"
            )
            return df

        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                logger.error(
                    f"{file_type} PUMS not found at {url}. "
                    f"The {self.year} 5-Year ACS may not be published yet."
                )
            else:
                logger.error(f"HTTP error downloading {file_type} PUMS: {e}")
            return None
        except zipfile.BadZipFile:
            logger.error(f"Downloaded file from {url} is not a valid ZIP archive")
            return None
        except Exception as e:
            logger.error(f"Error downloading {file_type} PUMS from FTP: {e}")
            return None

    def download_person_data(self) -> Optional[pd.DataFrame]:
        """Download person-level PUMS data."""
        logger.info(f"Downloading person data for {self.state} ({self.year}) via {self.method}...")
        if self.method == 'ftp':
            return self._download_ftp('person')
        return self._make_api_request('person', self.person_vars)

    def download_household_data(self) -> Optional[pd.DataFrame]:
        """Download household-level PUMS data."""
        logger.info(f"Downloading household data for {self.state} ({self.year}) via {self.method}...")
        if self.method == 'ftp':
            return self._download_ftp('household')
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
    
    def _validate_download(self, df: pd.DataFrame, dataset_type: str) -> bool:
        """Validate that a downloaded DataFrame has the required schema.

        Args:
            df: DataFrame to validate
            dataset_type: 'person' or 'household'

        Returns:
            True if validation passes, False otherwise
        """
        if dataset_type == 'person':
            required_cols = {'SERIALNO', 'AGEP', 'WAGP', 'PINCP', 'RELSHIPP', 'SPORDER'}
            min_rows = 1000
        elif dataset_type == 'household':
            required_cols = {'SERIALNO', 'HINCP', 'WGTP', 'NP'}
            min_rows = 500
        else:
            logger.warning(f"Unknown dataset_type '{dataset_type}', skipping validation")
            return True

        missing = required_cols - set(df.columns)
        if missing:
            logger.error(
                f"PUMS {dataset_type} validation failed: missing columns {missing}. "
                f"Got columns: {sorted(df.columns)}"
            )
            return False

        if len(df) < min_rows:
            logger.error(
                f"PUMS {dataset_type} validation failed: only {len(df)} rows "
                f"(expected >= {min_rows} for Hawaii)"
            )
            return False

        logger.info(
            f"PUMS {dataset_type} validation passed: {len(df)} rows, "
            f"all {len(required_cols)} required columns present"
        )
        return True

    def download_all(self) -> bool:
        """Download both person and household data.

        Returns:
            True if both downloads were successful, False otherwise
        """
        if self.method == 'api' and not self.api_key:
            logger.warning("No Census API key provided. Using public access which has rate limits.")

        success = True

        # Download person data
        person_df = self.download_person_data()
        if person_df is not None and self._validate_download(person_df, 'person'):
            self.save_data(person_df, f"psam_p{self.state}.csv")
        else:
            success = False
            logger.error("Failed to download or validate person data")

        # Download household data
        household_df = self.download_household_data()
        if household_df is not None and self._validate_download(household_df, 'household'):
            self.save_data(household_df, f"psam_h{self.state}.csv")
        else:
            success = False
            logger.error("Failed to download or validate household data")

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
    parser.add_argument('--method', choices=['ftp', 'api'], default='ftp',
                       help='Download method: ftp (full data, ~290 cols) or api (subset, ~24 cols)')
    return parser.parse_args()

def main():
    """Main function to run the script."""
    args = parse_args()

    downloader = PUMSDownloader(
        year=args.year,
        state=args.state,
        data_dir=args.data_dir,
        api_key=args.api_key or DEFAULT_API_KEY,
        method=args.method,
    )

    logger.info(f"Method: {args.method.upper()}, Year: {args.year}, State: {args.state}")
    success = downloader.download_all()

    if success:
        logger.info("\nDownload completed successfully!")
        logger.info(f"Files saved to: {args.data_dir}")
    else:
        logger.error("\nDownload completed with errors. Check the log for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
