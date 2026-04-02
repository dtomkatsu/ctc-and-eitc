#!/usr/bin/env python3
"""
Harmonize BLS OES data across years (2009-2024) for Hawaii ensemble growth projections.

This script processes raw BLS OES Excel files into a consistent time-series dataset
for occupation-specific wage growth modeling. The output feeds into ensemble
projection models that combine ACS demographic trends with BLS wage trajectories.

Key Features:
- Processes 16 years of BLS OES data (2009-2024)
- Standardizes column names and data types across years
- Filters for Hawaii state data only
- Creates occupation-specific wage growth trajectories
- Handles missing data and format changes over time
- Outputs both wide and long formats for different modeling needs

Usage:
    python scripts/data_prep/harmonize_bls_oes_years.py
    python scripts/data_prep/harmonize_bls_oes_years.py --start-year 2015 --end-year 2024
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BLSOESHarmonizer:
    """Harmonize BLS OES data across multiple years for Hawaii."""
    
    def __init__(self, data_dir: Path, output_dir: Path):
        """Initialize harmonizer with data directories."""
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Standard column mapping from BLS to our internal format
        self.column_mapping = {
            'AREA': 'area_code',
            'AREA_TITLE': 'area_title', 
            'OCC_CODE': 'occupation_code',
            'OCC_TITLE': 'occupation_title',
            'TOT_EMP': 'employment',
            'EMP_PRSE': 'employment_rse',
            'A_MEAN': 'mean_wage_annual',
            'MEAN_PRSE': 'mean_wage_rse',
            'A_MEDIAN': 'median_wage_annual',
            'A_PCT10': 'pct10_wage_annual',
            'A_PCT25': 'pct25_wage_annual', 
            'A_PCT75': 'pct75_wage_annual',
            'A_PCT90': 'pct90_wage_annual',
            'H_MEAN': 'mean_wage_hourly',
            'H_MEDIAN': 'median_wage_hourly',
            'H_PCT10': 'pct10_wage_hourly',
            'H_PCT25': 'pct25_wage_hourly',
            'H_PCT75': 'pct75_wage_hourly', 
            'H_PCT90': 'pct90_wage_hourly',
            'NAICS': 'industry_code',
            'NAICS_TITLE': 'industry_title',
            'O_GROUP': 'occupation_group',
            'I_GROUP': 'industry_group',
            'PRIM_STATE': 'state_code'
        }
        
        # Core columns we need for analysis
        self.required_columns = {
            'occupation_code', 'occupation_title', 'employment', 
            'mean_wage_annual', 'median_wage_annual'
        }
        
        # Wage columns for growth calculations
        self.wage_columns = [
            'mean_wage_annual', 'median_wage_annual',
            'pct10_wage_annual', 'pct25_wage_annual',
            'pct75_wage_annual', 'pct90_wage_annual'
        ]

    def find_bls_files(self, start_year: int = 2009, end_year: int = 2024) -> Dict[int, Path]:
        """Find BLS OES files for the specified year range."""
        
        files_by_year = {}
        
        for year in range(start_year, end_year + 1):
            # Try different file naming patterns
            patterns = [
                f'state_{year}.xls',
                f'state_{year}.xlsx', 
                f'state_M{year}_dl.xls',
                f'state_M{year}_dl.xlsx'
            ]
            
            for pattern in patterns:
                file_path = self.data_dir / pattern
                if file_path.exists():
                    files_by_year[year] = file_path
                    logger.info(f"Found BLS OES file for {year}: {file_path.name}")
                    break
            else:
                logger.warning(f"No BLS OES file found for year {year}")
        
        logger.info(f"Found {len(files_by_year)} BLS OES files covering years: {sorted(files_by_year.keys())}")
        return files_by_year

    def parse_bls_file(self, file_path: Path, year: int) -> pd.DataFrame:
        """Parse a single BLS OES Excel file."""
        
        logger.info(f"Parsing BLS OES file: {file_path.name}")
        
        try:
            # Read Excel file - try different sheet names
            sheet_names = [0, 'State', 'state', 'All Data', 'Data']
            df = None
            
            # Pick engine based on extension
            engine = 'xlrd' if str(file_path).endswith('.xls') else 'openpyxl'

            for sheet in sheet_names:
                try:
                    df = pd.read_excel(
                        file_path,
                        sheet_name=sheet,
                        engine=engine,
                        na_values=['#', '*', '**', '***', 'N/A', ''],
                        keep_default_na=True
                    )
                    logger.debug(f"Successfully read sheet: {sheet}")
                    break
                except Exception as e:
                    logger.debug(f"Failed to read sheet {sheet}: {e}")
                    continue
            
            if df is None:
                raise ValueError(f"Could not read any sheet from {file_path}")
            
            # Normalize column names to uppercase for consistent filtering
            df.columns = [c.upper() for c in df.columns]

            # Filter for Hawaii data — column layout differs across years:
            #   2009-2018: AREA (numeric), ST (HI), STATE (Hawaii)
            #   2019: area_title (→ AREA_TITLE after upper())
            #   2020+: AREA_TITLE
            hawaii_mask = pd.Series(False, index=df.index)
            for col, pattern in [
                ('AREA_TITLE', 'Hawaii'),
                ('STATE', 'Hawaii'),
                ('ST', 'HI'),
                ('AREA', '15'),
            ]:
                if col in df.columns:
                    m = df[col].astype(str).str.contains(pattern, case=False, na=False)
                    if m.any():
                        hawaii_mask |= m
                        break
            
            if not hawaii_mask.any():
                logger.warning(f"No Hawaii data found in {file_path.name}")
                return pd.DataFrame()
            
            df = df[hawaii_mask].copy()
            
            # Rename columns using mapping
            df = df.rename(columns=self.column_mapping)
            
            # Add year and state
            df['year'] = year
            df['state'] = 'HI'
            
            # Clean and convert wage columns
            for col in self.wage_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Clean employment
            if 'employment' in df.columns:
                df[col] = pd.to_numeric(df['employment'], errors='coerce')
            
            # Filter out summary rows and invalid occupation codes
            if 'occupation_code' in df.columns:
                # Remove rows with missing or invalid occupation codes
                df = df[df['occupation_code'].notna()]
                df = df[df['occupation_code'].astype(str).str.len() >= 2]
                
                # Remove summary rows (usually have '00-0000' pattern or 'Total' in title)
                summary_mask = (
                    df['occupation_code'].astype(str).str.endswith('-0000') |
                    df['occupation_title'].str.contains('Total', case=False, na=False) |
                    df['occupation_title'].str.contains('All Occupations', case=False, na=False)
                )
                df = df[~summary_mask]
            
            logger.info(f"Parsed {len(df):,} occupation records for Hawaii ({year})")
            return df
            
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return pd.DataFrame()

    def harmonize_occupation_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Harmonize occupation codes across years to handle SOC revisions."""
        
        if 'occupation_code' not in df.columns:
            return df
        
        # SOC code revisions and mappings
        # Major revisions occurred in 2010 and 2018
        soc_mappings = {
            # Map old codes to new codes for consistency
            # This is a simplified mapping - full crosswalks are available from BLS
            '15-1199': '15-1299',  # Computer occupations, all other
            '29-1199': '29-1299',  # Health diagnosing practitioners, all other
            '43-9199': '43-9299',  # Office and administrative support workers, all other
        }
        
        df = df.copy()
        df['occupation_code_original'] = df['occupation_code']
        
        # Apply mappings
        for old_code, new_code in soc_mappings.items():
            mask = df['occupation_code'] == old_code
            if mask.any():
                df.loc[mask, 'occupation_code'] = new_code
                logger.info(f"Mapped {mask.sum()} records from {old_code} to {new_code}")
        
        return df

    def calculate_growth_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate wage growth rates by occupation across years."""
        
        logger.info("Calculating occupation-specific wage growth rates")
        
        # Sort by occupation and year
        df = df.sort_values(['occupation_code', 'year'])
        
        growth_data = []
        
        for occ_code in df['occupation_code'].unique():
            occ_data = df[df['occupation_code'] == occ_code].copy()
            
            if len(occ_data) < 2:
                continue  # Need at least 2 years for growth calculation
            
            occ_data = occ_data.sort_values('year')
            
            # Calculate year-over-year growth rates
            for col in self.wage_columns:
                if col in occ_data.columns:
                    occ_data[f'{col}_growth'] = occ_data[col].pct_change(fill_method=None)
            
            # Calculate compound annual growth rate (CAGR)
            first_year = occ_data['year'].iloc[0]
            last_year = occ_data['year'].iloc[-1]
            years_span = last_year - first_year
            
            if years_span > 0:
                for col in self.wage_columns:
                    if col in occ_data.columns:
                        first_val = occ_data[col].iloc[0]
                        last_val = occ_data[col].iloc[-1]
                        
                        if pd.notna(first_val) and pd.notna(last_val) and first_val > 0:
                            cagr = (last_val / first_val) ** (1 / years_span) - 1
                            occ_data[f'{col}_cagr'] = cagr
            
            growth_data.append(occ_data)
        
        if growth_data:
            result = pd.concat(growth_data, ignore_index=True)
            logger.info(f"Calculated growth rates for {len(result['occupation_code'].unique())} occupations")
            return result
        else:
            return df

    def create_occupation_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create occupation-level summary statistics across all years."""
        
        logger.info("Creating occupation summary statistics")
        
        summary_stats = []
        
        for occ_code in df['occupation_code'].unique():
            occ_data = df[df['occupation_code'] == occ_code]
            
            if occ_data.empty:
                continue
            
            # Get most recent occupation title
            latest_title = occ_data.loc[occ_data['year'].idxmax(), 'occupation_title']
            
            # Calculate summary statistics
            stats = {
                'occupation_code': occ_code,
                'occupation_title': latest_title,
                'years_available': len(occ_data),
                'year_range': f"{occ_data['year'].min()}-{occ_data['year'].max()}",
                'avg_employment': occ_data['employment'].mean(),
                'avg_mean_wage': occ_data['mean_wage_annual'].mean(),
                'avg_median_wage': occ_data['median_wage_annual'].mean(),
            }
            
            # Calculate growth rates if we have multiple years
            if len(occ_data) >= 2:
                first_wage = occ_data.loc[occ_data['year'].idxmin(), 'mean_wage_annual']
                last_wage = occ_data.loc[occ_data['year'].idxmax(), 'mean_wage_annual']
                years_span = occ_data['year'].max() - occ_data['year'].min()
                
                if pd.notna(first_wage) and pd.notna(last_wage) and first_wage > 0 and years_span > 0:
                    total_growth = (last_wage - first_wage) / first_wage
                    annual_growth = (last_wage / first_wage) ** (1 / years_span) - 1
                    
                    stats.update({
                        'total_wage_growth': total_growth,
                        'annual_wage_growth': annual_growth
                    })
            
            summary_stats.append(stats)
        
        summary_df = pd.DataFrame(summary_stats)
        logger.info(f"Created summary for {len(summary_df)} occupations")
        
        return summary_df

    def process_all_years(
        self, 
        start_year: int = 2009, 
        end_year: int = 2024
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Process all BLS OES files and create harmonized dataset."""
        
        logger.info(f"Processing BLS OES data for years {start_year}-{end_year}")
        
        # Find all available files
        files_by_year = self.find_bls_files(start_year, end_year)
        
        if not files_by_year:
            raise ValueError("No BLS OES files found in the specified directory")
        
        # Process each file
        all_data = []
        
        for year, file_path in sorted(files_by_year.items()):
            df = self.parse_bls_file(file_path, year)
            
            if not df.empty:
                df = self.harmonize_occupation_codes(df)
                all_data.append(df)
        
        if not all_data:
            raise ValueError("No valid data found in any BLS OES files")
        
        # Combine all years
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined data: {len(combined_df):,} records across {len(files_by_year)} years")
        
        # Calculate growth rates
        combined_df = self.calculate_growth_rates(combined_df)
        
        # Create occupation summary
        summary_df = self.create_occupation_summary(combined_df)
        
        return combined_df, summary_df

    def save_outputs(self, combined_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
        """Save processed data in multiple formats."""
        
        logger.info("Saving processed BLS OES data")
        
        # Save main timeseries data
        timeseries_path = self.output_dir / 'bls_oes_timeseries.parquet'
        combined_df.to_parquet(timeseries_path, index=False)
        logger.info(f"Saved timeseries data: {timeseries_path}")
        
        # Save CSV for inspection
        csv_path = self.output_dir / 'bls_oes_timeseries.csv'
        combined_df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV snapshot: {csv_path}")
        
        # Save occupation summary
        summary_path = self.output_dir / 'bls_oes_occupation_summary.parquet'
        summary_df.to_parquet(summary_path, index=False)
        logger.info(f"Saved occupation summary: {summary_path}")
        
        summary_csv_path = self.output_dir / 'bls_oes_occupation_summary.csv'
        summary_df.to_csv(summary_csv_path, index=False)
        logger.info(f"Saved summary CSV: {summary_csv_path}")
        
        # Create metadata file
        metadata = {
            'processing_date': pd.Timestamp.now().isoformat(),
            'total_records': int(len(combined_df)),
            'total_occupations': int(len(combined_df['occupation_code'].unique())),
            'year_range': f"{int(combined_df['year'].min())}-{int(combined_df['year'].max())}",
            'years_available': [int(y) for y in sorted(combined_df['year'].unique())],
            'wage_columns': self.wage_columns,
            'data_source': 'BLS Occupational Employment Statistics (OES)',
            'geographic_scope': 'Hawaii (State FIPS: 15)',
            'notes': 'Harmonized across years with SOC code mapping and growth rate calculations'
        }
        
        metadata_path = self.output_dir / 'bls_oes_metadata.json'
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata: {metadata_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Harmonize BLS OES data across years for Hawaii ensemble projections"
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data/external/bls_oes'),
        help='Directory containing raw BLS OES Excel files'
    )
    parser.add_argument(
        '--output-dir', 
        type=Path,
        default=Path('data/processed'),
        help='Directory to save processed outputs'
    )
    parser.add_argument(
        '--start-year',
        type=int,
        default=2009,
        help='First year to process'
    )
    parser.add_argument(
        '--end-year',
        type=int, 
        default=2024,
        help='Last year to process'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable debug logging'
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=" * 80)
    logger.info("BLS OES HARMONIZATION FOR HAWAII ENSEMBLE PROJECTIONS")
    logger.info("=" * 80)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Year range: {args.start_year}-{args.end_year}")
    
    # Initialize harmonizer
    harmonizer = BLSOESHarmonizer(args.data_dir, args.output_dir)
    
    # Process all years
    try:
        combined_df, summary_df = harmonizer.process_all_years(args.start_year, args.end_year)
        
        # Save outputs
        harmonizer.save_outputs(combined_df, summary_df)
        
        # Print summary
        logger.info("=" * 80)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total records processed: {len(combined_df):,}")
        logger.info(f"Unique occupations: {len(combined_df['occupation_code'].unique()):,}")
        logger.info(f"Years covered: {sorted(combined_df['year'].unique())}")
        logger.info(f"Average wage (all occupations): ${combined_df['mean_wage_annual'].mean():,.0f}")
        
        # Show top occupations by employment
        top_employment = combined_df.groupby('occupation_code').agg({
            'occupation_title': 'first',
            'employment': 'mean'
        }).nlargest(5, 'employment')
        
        logger.info("\nTop 5 occupations by employment:")
        for occ_code, row in top_employment.iterrows():
            logger.info(f"  {occ_code}: {row['occupation_title']} ({row['employment']:,.0f})")
        
        logger.info(f"\nFiles saved to: {args.output_dir}")
        logger.info("Next step: Use data/processed/bls_oes_timeseries.parquet for ensemble modeling")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


if __name__ == '__main__':
    main()
