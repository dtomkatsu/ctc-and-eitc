#!/usr/bin/env python3
"""
Download B07001 (Migration) table only with chunking fix.

This script specifically targets the B07001 table that failed in the main
download due to too many variables (96) in a single API request.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from scripts.data_prep.download_acs_tables import ACSTableDownloader
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Download B07001 table for all years 2015-2023."""
    
    # Initialize downloader
    api_key = '2104852dd7bfd83fbc9e320d650eb57decc11817'
    output_dir = Path('data/raw/acs_1yr')
    
    downloader = ACSTableDownloader(api_key, output_dir)
    
    logger.info("="*60)
    logger.info("B07001 (Migration) Table Download")
    logger.info("="*60)
    
    table_code = 'B07001'
    results = {}
    
    for year in range(2015, 2024):
        if year == 2020:  # Skip 2020 - no 1-year estimates due to COVID
            continue
            
        logger.info(f"\nDownloading {table_code} for {year}...")
        
        df = downloader.download_table(table_code, year)
        
        if df is not None:
            results[year] = df
            
            # Save individual file
            filename = f"{table_code}_{year}.csv"
            filepath = output_dir / 'additional' / filename
            df.to_csv(filepath, index=False)
            logger.info(f"Saved to {filepath}")
        else:
            logger.warning(f"Failed to download {table_code} for {year}")
    
    # Create combined file if we have any data
    if results:
        import pandas as pd
        combined = pd.concat(results.values(), ignore_index=True)
        combined = combined.sort_values('year')
        
        filename = f"{table_code}_2015_2023_combined.csv"
        filepath = output_dir / 'additional' / filename
        combined.to_csv(filepath, index=False)
        logger.info(f"Saved combined file: {filepath}")
        
        logger.info(f"\nSUCCESS: Downloaded {table_code} for {len(results)} years")
        logger.info(f"Years: {sorted(results.keys())}")
        logger.info(f"Combined shape: {combined.shape}")
    else:
        logger.error(f"FAILED: No data downloaded for {table_code}")

if __name__ == '__main__':
    main()
