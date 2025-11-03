#!/usr/bin/env python3
"""
Parse DOTAX Historical Individual Income Tax Tables

This script extracts key data from DOTAX historical spreadsheets (2015-2021)
to create a comprehensive time series of:
- Returns by filing status
- AGI by filing status
- Tax liability by filing status (before and after credits)
- Revenue trends over time
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_table_5a(file_path: Path, year: int) -> pd.DataFrame:
    """
    Parse Table 5A - Returns by Filing Status.
    
    Contains:
    - Number of returns by filing status
    - Hawaii AGI (positive and negative)
    - Tax liability before credits
    - Tax liability after credits
    """
    logger.info(f"Parsing Table 5A from {year} file...")
    
    try:
        # Read the table with proper header
        df = pd.read_excel(file_path, sheet_name='5A', header=3)
        
        # Clean up the data structure
        # First row contains the actual column headers
        df = pd.read_excel(file_path, sheet_name='5A', header=4)
        
        # The filing status is in first column
        df = df.rename(columns={df.columns[0]: 'filing_status'})
        
        # Extract the data rows (before "TOTAL" row)
        data_rows = []
        for idx, row in df.iterrows():
            filing_status = str(row['filing_status']).strip()
            
            # Stop at TOTAL row
            if filing_status.upper() == 'TOTAL':
                break
            
            # Skip NaN or empty rows
            if pd.isna(filing_status) or filing_status == 'nan':
                continue
                
            data_rows.append(row)
        
        if not data_rows:
            logger.warning(f"No data rows found in Table 5A for year {year}")
            return None
            
        result_df = pd.DataFrame(data_rows)
        
        # Add year column
        result_df['year'] = year
        
        # Try to extract key metrics (column positions may vary by year)
        # Typically: Filing Status, Returns, %, AGI+, %, AGI-, %, Tax Before Credits, %, Tax After Credits, %
        
        logger.info(f"  Found {len(result_df)} filing status categories")
        logger.info(f"  Columns: {result_df.columns.tolist()}")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error parsing Table 5A for year {year}: {e}")
        return None


def parse_table_2(file_path: Path, year: int) -> pd.DataFrame:
    """
    Parse Table 2 - Returns by AGI Class.
    
    Contains income bracket distribution data.
    """
    logger.info(f"Parsing Table 2 from {year} file...")
    
    try:
        df = pd.read_excel(file_path, sheet_name='2', header=None)
        
        # Find the header row (usually contains "AGI Class" or similar)
        header_row = None
        for idx, row in df.iterrows():
            if any('AGI' in str(cell) for cell in row if pd.notna(cell)):
                header_row = idx
                break
        
        if header_row is not None:
            df = pd.read_excel(file_path, sheet_name='2', skiprows=header_row+1)
            df['year'] = year
            logger.info(f"  Found {len(df)} AGI brackets")
            return df
        else:
            logger.warning(f"Could not find header row in Table 2 for year {year}")
            return None
            
    except Exception as e:
        logger.error(f"Error parsing Table 2 for year {year}: {e}")
        return None


def parse_all_years(data_dir: Path) -> dict:
    """
    Parse DOTAX data for all available years.
    
    Returns:
        Dictionary with dataframes for each table type across all years
    """
    results = {
        'table_5a': [],  # Returns by filing status
        'table_2': []    # Returns by AGI class
    }
    
    # Find all year files
    year_files = sorted(data_dir.glob('*indinc_tables.xlsx'))
    
    logger.info(f"Found {len(year_files)} year files")
    
    for file_path in year_files:
        # Extract year from filename (e.g., "2021indinc_tables.xlsx" -> 2021)
        year = int(file_path.stem[:4])
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing year {year}: {file_path.name}")
        logger.info(f"{'='*80}")
        
        # Parse Table 5A (filing status)
        df_5a = parse_table_5a(file_path, year)
        if df_5a is not None:
            results['table_5a'].append(df_5a)
        
        # Parse Table 2 (AGI brackets)
        df_2 = parse_table_2(file_path, year)
        if df_2 is not None:
            results['table_2'].append(df_2)
    
    # Combine all years
    for key in results:
        if results[key]:
            results[key] = pd.concat(results[key], ignore_index=True)
            logger.info(f"\nCombined {key}: {len(results[key])} total rows across all years")
        else:
            results[key] = None
            logger.warning(f"\nNo data found for {key}")
    
    return results


def save_parsed_data(results: dict, output_dir: Path):
    """Save parsed data to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for table_name, df in results.items():
        if df is not None and not df.empty:
            output_file = output_dir / f'dotax_historical_{table_name}.csv'
            df.to_csv(output_file, index=False)
            logger.info(f"Saved {table_name} to {output_file}")


def main():
    """Main parsing workflow."""
    logger.info("="*80)
    logger.info("DOTAX HISTORICAL DATA PARSER")
    logger.info("="*80)
    
    # Set paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / 'data' / 'raw' / 'dotax_historical_data'
    output_dir = project_root / 'data' / 'processed' / 'dotax_historical'
    
    # Parse all years
    results = parse_all_years(data_dir)
    
    # Save results
    save_parsed_data(results, output_dir)
    
    # Display summary
    logger.info("\n" + "="*80)
    logger.info("PARSING COMPLETE - SUMMARY")
    logger.info("="*80)
    
    for table_name, df in results.items():
        if df is not None:
            logger.info(f"\n{table_name}:")
            logger.info(f"  Total rows: {len(df)}")
            logger.info(f"  Years: {sorted(df['year'].unique())}")
            logger.info(f"  Columns: {df.columns.tolist()}")


if __name__ == "__main__":
    main()
