#!/usr/bin/env python3
"""
Download ACS 1-Year Estimates for Hawaii (2015-2024, excluding 2020)

This script downloads key demographic and economic tables from the Census Bureau's
American Community Survey (ACS) to support ensemble growth projections for the
Hawaii tax model. These tables, combined with BLS OES wage data, enable
sophisticated income and demographic growth modeling.

Tables Downloaded:
Critical Tables (5):
- B19001: Household Income Distribution
- B19013: Median Household Income
- B19019: Median Household Income by Household Type
- B12001: Marital Status by Sex and Age
- B01001: Sex by Age

Additional Tables (6):
- B11001: Household Type
- B09002: Own Children Under 18 Years by Family Type
- B07001: Geographic Mobility in Past Year
- B23025: Employment Status
- B25003: Tenure (Owner/Renter)
- B25077: Median Home Value

Usage:
    python scripts/data_prep/download_acs_tables.py
    
    # Custom years or API key
    python scripts/data_prep/download_acs_tables.py --start-year 2018 --end-year 2023
    python scripts/data_prep/download_acs_tables.py --api-key YOUR_KEY
"""

import requests
import pandas as pd
import argparse
import logging
from pathlib import Path
import time
from typing import List, Dict, Optional
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ACSTableDownloader:
    """Download ACS 1-year estimate tables from the Census Bureau API."""

    # Table definitions
    CRITICAL_TABLES = {
        'B19001': 'Income Distribution',
        'B19013': 'Median Income',
        'B19019': 'Income by Household Type',
        'B12001': 'Marital Status',
        'B01001': 'Age/Sex Distribution',
    }

    ADDITIONAL_TABLES = {
        'B11001': 'Household Type',
        'B09002': 'Children by Family Type',
        'B07001': 'Migration',
        'B23025': 'Employment',
        'B25003': 'Homeownership',
        'B25077': 'Home Value',
    }

    BASE_URL = 'https://api.census.gov/data/{year}/acs/acs1'
    HAWAII_FIPS = '15'  # State FIPS code for Hawaii

    # Years with 1-year ACS data now available again (including 2020 release)
    AVAILABLE_YEARS = {
        2015,
        2016,
        2017,
        2018,
        2019,
        2021,
        2022,
        2023,
        2024,
    }

    def __init__(self, api_key: str, output_dir: Path, available_years: Optional[set[int]] = None):
        """Initialize downloader."""

        self.api_key = api_key
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Allow caller override of available years (helps when new releases appear)
        if available_years is None:
            self.available_years = self.AVAILABLE_YEARS
        else:
            self.available_years = available_years

        # Create subdirectories
        (self.output_dir / 'critical').mkdir(exist_ok=True)
        (self.output_dir / 'additional').mkdir(exist_ok=True)
        (self.output_dir / 'metadata').mkdir(exist_ok=True)

    def get_table_variables(self, table_code: str, year: int) -> List[str]:
        """Fetch all variables for a given table."""

        url = f'https://api.census.gov/data/{year}/acs/acs1/variables.json'

        try:
            response = requests.get(url)
            response.raise_for_status()
            variables = response.json()['variables']

            # Filter for this table
            table_vars = [var for var in variables.keys() if var.startswith(f'{table_code}_')]

            logger.info("Found %d variables for %s (%d)", len(table_vars), table_code, year)
            return table_vars

        except Exception as exc:  # pragma: no cover - network error path
            logger.warning(
                "Could not fetch variables for %s (%d): %s", table_code, year, exc
            )
            # Fallback: request all variables with wildcard
            return [f'{table_code}_*']
    
    def download_table(
        self, 
        table_code: str, 
        year: int,
        get_all_vars: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Download a single ACS table.
        
        Args:
            table_code: ACS table code (e.g., 'B19001')
            year: Year of data
            get_all_vars: Whether to fetch all variables in table
            
        Returns:
            DataFrame with table data, or None if failed
        """
        logger.info(f"Downloading {table_code} for {year}...")
        
        # Build request URL
        url = self.BASE_URL.format(year=year)
        
        # Get variables for this table
        if get_all_vars:
            variables = self.get_table_variables(table_code, year)
            if not variables:
                logger.error(f"No variables found for {table_code}")
                return None
            
            # Census API has limit on number of variables per request
            # Use smaller chunks for problematic tables like B07001
            chunk_size = 30 if table_code == 'B07001' else 50
            var_chunks = [variables[i:i+chunk_size] for i in range(0, len(variables), chunk_size)]
            logger.info(f"Split {len(variables)} variables into {len(var_chunks)} chunks of max {chunk_size}")
        else:
            var_chunks = [[f'{table_code}_001E']]  # Just get estimate for concept
        
        all_data = []
        
        max_retries = 3
        for i, chunk in enumerate(var_chunks):
            logger.info(f"Downloading chunk {i+1}/{len(var_chunks)} ({len(chunk)} variables)")

            params = {
                'get': ','.join(chunk + ['NAME']),
                'for': 'state:15',  # Hawaii
                'key': self.api_key
            }

            try:
                # Retry loop for 429 rate-limiting
                response = None
                for attempt in range(max_retries + 1):
                    response = requests.get(url, params=params)
                    if response.status_code == 429:
                        if attempt < max_retries:
                            retry_after = int(response.headers.get('Retry-After', 2 * (2 ** attempt)))
                            wait = min(retry_after, 60)
                            logger.warning(
                                f"Rate limited (429) on {table_code} chunk {i+1}, "
                                f"attempt {attempt + 1}/{max_retries + 1}. Retrying in {wait}s..."
                            )
                            time.sleep(wait)
                            continue
                        else:
                            logger.error(f"Rate limited on {table_code} chunk {i+1} after {max_retries + 1} attempts")
                            break
                    break  # Not 429 — proceed to normal handling

                response.raise_for_status()
                data = response.json()

                # Convert to DataFrame
                df = pd.DataFrame(data[1:], columns=data[0])
                all_data.append(df)

                # Rate limiting - longer for problematic tables
                sleep_time = 1.0 if table_code == 'B07001' else 0.5
                time.sleep(sleep_time)

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    logger.warning(f"Table {table_code} not available for {year}")
                    return None
                elif e.response.status_code == 400:
                    logger.error(f"HTTP 400 error for {table_code} chunk {i+1}: {e}")
                    logger.error(f"Chunk variables: {chunk[:5]}...{chunk[-5:] if len(chunk) > 5 else chunk}")
                    # Continue with other chunks
                    continue
                else:
                    logger.error(f"HTTP error downloading {table_code} ({year}): {e}")
                    return None
            except Exception as e:
                logger.error(f"Error downloading {table_code} ({year}) chunk {i+1}: {e}")
                continue
        
        if not all_data:
            logger.error(f"No data chunks successfully downloaded for {table_code} ({year})")
            return None
            
        # Combine chunks
        result = pd.concat(all_data, axis=1)
        # Remove duplicate columns
        result = result.loc[:, ~result.columns.duplicated()]
        
        # Add metadata columns
        result['year'] = year
        result['table_code'] = table_code
        
        logger.info(f"Successfully downloaded {table_code} ({year}): {result.shape}")
        return result
    
    def download_all_tables(
        self,
        start_year: int = 2015,
        end_year: int = 2024
    ) -> Dict[str, Dict[int, pd.DataFrame]]:
        """Download all tables for the requested year range."""

        all_tables = {**self.CRITICAL_TABLES, **self.ADDITIONAL_TABLES}
        results: Dict[str, Dict[int, pd.DataFrame]] = {}

        # Clamp requested years to the set of known 1-year releases
        requested_years = [
            year
            for year in range(start_year, end_year + 1)
            if year in self.available_years
        ]

        if not requested_years:
            raise ValueError(
                f"No ACS 1-year data available between {start_year} and {end_year}."
            )

        total_downloads = len(all_tables) * len(requested_years)
        current = 0

        for table_code, description in all_tables.items():
            logger.info(f"\n{'='*60}")
            logger.info("Table %s: %s", table_code, description)
            logger.info(f"{'='*60}")

            results[table_code] = {}

            for year in requested_years:
                current += 1
                logger.info("Progress: %d/%d", current, total_downloads)

                df = self.download_table(table_code, year)

                if df is not None:
                    results[table_code][year] = df

                    # Save individual file
                    category = 'critical' if table_code in self.CRITICAL_TABLES else 'additional'
                    filename = f"{table_code}_{year}.csv"
                    filepath = self.output_dir / category / filename
                    df.to_csv(filepath, index=False)
                    logger.info("Saved to %s", filepath)

                # Rate limiting between tables
                time.sleep(1)

        return results

    def create_summary_report(
        self,
        results: Dict[str, Dict[int, pd.DataFrame]]
    ) -> pd.DataFrame:
        """Create a summary report describing downloaded tables."""

        summary_data = []

        for table_code, years_data in results.items():
            all_tables = {**self.CRITICAL_TABLES, **self.ADDITIONAL_TABLES}
            description = all_tables.get(table_code, 'Unknown')

            for year, df in years_data.items():
                summary_data.append({
                    'table_code': table_code,
                    'description': description,
                    'year': year,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'category': 'critical' if table_code in self.CRITICAL_TABLES else 'additional'
                })

        summary_df = pd.DataFrame(summary_data)

        # Save summary
        summary_path = self.output_dir / 'metadata' / 'download_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        logger.info("\nSummary saved to %s", summary_path)

        # Print summary statistics
        logger.info("\n" + "=" * 60)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("=" * 60)
        logger.info("Total tables requested: %d", len(results))
        logger.info("Total successful downloads: %d", len(summary_df))
        logger.info("\nBy Category:")
        logger.info(summary_df.groupby('category').size())
        logger.info("\nBy Year:")
        logger.info(summary_df.groupby('year').size())

        return summary_df

    def create_combined_files(self, results: Dict[str, Dict[int, pd.DataFrame]]):
        """Create combined files containing all years for each table."""

        logger.info("\nCreating combined time-series files...")

        for table_code, years_data in results.items():
            if not years_data:
                continue

            # Combine all years
            combined = pd.concat(years_data.values(), ignore_index=True)

            # Sort by year and reset index
            combined = combined.sort_values('year').reset_index(drop=True)

            # Determine year range for filename
            available_years = sorted(years_data.keys())
            start_year = available_years[0]
            end_year = available_years[-1]

            # Save
            category = 'critical' if table_code in self.CRITICAL_TABLES else 'additional'
            filename = f"{table_code}_{start_year}_{end_year}_combined.csv"
            filepath = self.output_dir / category / filename
            combined.to_csv(filepath, index=False)
            logger.info("Saved combined file: %s", filepath)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Download ACS 1-year estimates for Hawaii growth projections'
    )
    parser.add_argument(
        '--api-key',
        default='2104852dd7bfd83fbc9e320d650eb57decc11817',
        help='Census API key (default: provided key)'
    )
    parser.add_argument(
        '--start-year',
        type=int,
        default=2015,
        help='First year to download (default: 2015)'
    )
    parser.add_argument(
        '--end-year',
        type=int,
        default=2023,
        help='Last year to download (default: 2023)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/raw/acs_1yr'),
        help='Output directory for downloaded tables'
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("ACS Table Downloader")
    logger.info("="*60)
    logger.info(f"Years: {args.start_year}-{args.end_year}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Tables: {len(ACSTableDownloader.CRITICAL_TABLES) + len(ACSTableDownloader.ADDITIONAL_TABLES)}")
    
    # Initialize downloader
    downloader = ACSTableDownloader(args.api_key, args.output_dir)
    
    # Download all tables
    results = downloader.download_all_tables(args.start_year, args.end_year)
    
    # Create summary report
    summary = downloader.create_summary_report(results)
    
    # Create combined files
    downloader.create_combined_files(results)
    
    logger.info("\n" + "="*60)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("="*60)
    logger.info(f"Files saved to: {args.output_dir}")
    logger.info("\nNext steps:")
    logger.info("1. Review download_summary.csv in metadata folder")
    logger.info("2. Check for any missing tables or years")
    logger.info("3. Run ensemble projection script to combine with BLS OES data")


if __name__ == '__main__':
    main()
