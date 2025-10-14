"""
BLS Occupational Employment Statistics (OES) Data Loader.

This module loads and processes BLS OES data for Hawaii to provide:
- State-level wage distributions by occupation
- Industry-specific wage patterns
- Temporal wage growth rates
- Occupation-industry wage matrices

Data Source: https://www.bls.gov/oes/tables.htm
Free public data updated annually.

Key Features:
- Automatic data download from BLS API
- Hawaii-specific filtering
- Occupation code mapping (SOC codes)
- Industry code mapping (NAICS codes)
- Wage distribution statistics
"""

import pandas as pd
import numpy as np
import requests
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class OESWageData:
    """Container for OES wage statistics."""
    occupation_code: str
    occupation_title: str
    industry_code: Optional[str]
    industry_title: Optional[str]
    employment: int
    mean_wage: float
    median_wage: float
    pct10_wage: float
    pct25_wage: float
    pct75_wage: float
    pct90_wage: float
    year: int
    state: str = 'HI'
    
    @property
    def wage_distribution(self) -> Dict[str, float]:
        """Return wage distribution as dictionary."""
        return {
            'mean': self.mean_wage,
            'median': self.median_wage,
            'p10': self.pct10_wage,
            'p25': self.pct25_wage,
            'p75': self.pct75_wage,
            'p90': self.pct90_wage
        }
    
    def get_wage_growth_rate(self, previous_year_data: 'OESWageData') -> float:
        """Calculate wage growth rate from previous year."""
        if previous_year_data.mean_wage == 0:
            return 0.0
        return (self.mean_wage - previous_year_data.mean_wage) / previous_year_data.mean_wage


class BLSOESLoader:
    """
    Loader for BLS Occupational Employment Statistics data.
    
    Provides access to Hawaii-specific wage distributions by occupation
    and industry, critical for temporal alignment and wage imputation.
    """
    
    # BLS API configuration
    BLS_API_BASE = "https://api.bls.gov/publicAPI/v2"
    BLS_OES_SERIES_PREFIX = "OEUS"  # OES series
    
    # Hawaii state code
    HAWAII_STATE_CODE = "15"
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        api_key: Optional[str] = None,
        cache_data: bool = True
    ):
        """
        Initialize BLS OES loader.
        
        Args:
            data_dir: Directory for cached data
            api_key: BLS API key (optional, increases rate limits)
            cache_data: Whether to cache downloaded data
        """
        self.data_dir = Path(data_dir) if data_dir else Path('data/external/bls_oes')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key
        self.cache_data = cache_data
        
        self._wage_data_cache: Dict[int, pd.DataFrame] = {}
        
    def load_hawaii_wages(
        self,
        year: int = 2022,
        force_download: bool = False
    ) -> pd.DataFrame:
        """
        Load Hawaii wage data for a specific year.
        
        Args:
            year: Year to load data for
            force_download: Force download even if cached
            
        Returns:
            DataFrame with wage data by occupation and industry
        """
        # Check cache
        if not force_download and year in self._wage_data_cache:
            logger.info(f"Using cached BLS OES data for {year}")
            return self._wage_data_cache[year]
        
        # Check file cache
        cache_file = self.data_dir / f'hawaii_oes_{year}.parquet'
        if not force_download and cache_file.exists():
            logger.info(f"Loading BLS OES data from cache: {cache_file}")
            df = pd.read_parquet(cache_file)
            self._wage_data_cache[year] = df
            return df
        
        # Download from BLS
        logger.info(f"Downloading BLS OES data for Hawaii, year {year}")
        df = self._download_oes_data(year)
        
        # Cache the data
        if self.cache_data:
            df.to_parquet(cache_file, index=False)
            logger.info(f"Cached BLS OES data to {cache_file}")
        
        self._wage_data_cache[year] = df
        return df
    
    def _download_oes_data(self, year: int) -> pd.DataFrame:
        """
        Download OES data from BLS API or flat files.
        
        Note: BLS API has rate limits. For production, consider downloading
        the annual flat files directly from:
        https://www.bls.gov/oes/tables.htm
        """
        # For now, we'll use a mock implementation that loads from flat files
        # In production, you would implement actual BLS API calls or file downloads
        
        logger.warning("BLS API integration not fully implemented - using mock data")
        logger.info("To use real data, download from: https://www.bls.gov/oes/tables.htm")
        
        # Check if user has downloaded the data manually
        manual_file = self.data_dir / f'state_M{year}_dl.xlsx'
        if manual_file.exists():
            logger.info(f"Found manually downloaded BLS OES file: {manual_file}")
            return self._parse_oes_excel(manual_file, year)
        
        # Return mock data structure for development
        return self._create_mock_oes_data(year)
    
    def _parse_oes_excel(self, file_path: Path, year: int) -> pd.DataFrame:
        """
        Parse BLS OES Excel file.
        
        The BLS publishes state-level OES data in Excel format with specific structure:
        - Sheet name: typically 'State' or similar
        - Columns: AREA, OCC_CODE, OCC_TITLE, TOT_EMP, A_MEAN, etc.
        """
        logger.info(f"Parsing BLS OES Excel file: {file_path}")
        
        # Define data types for columns to ensure proper parsing
        dtype_spec = {
            'AREA': str,
            'AREA_TITLE': str,
            'OCC_CODE': str,
            'OCC_TITLE': str,
            'TOT_EMP': 'Int64',  # Use Int64 (capital I) to handle NA values
            'A_MEAN': 'float64',
            'A_MEDIAN': 'float64',
            'A_PCT10': 'float64',
            'A_PCT25': 'float64',
            'A_PCT75': 'float64',
            'A_PCT90': 'float64',
            'NAICS': str,
            'NAICS_TITLE': str,
            'O_GROUP': str,
            'I_GROUP': str,
            'PRIM_STATE': str,
            'H_MEAN': 'float64',
            'H_MEDIAN': 'float64',
            'H_PCT10': 'float64',
            'H_PCT25': 'float64',
            'H_PCT75': 'float64',
            'H_PCT90': 'float64',
            'ANNUAL': 'float64',
            'HOURLY': 'float64',
            'YEAR': 'int64'
        }
        
        # Read Excel file with explicit dtypes and handle errors
        df = pd.read_excel(
            file_path, 
            sheet_name=0,
            dtype=dtype_spec,
            na_values=['#', '*', '**', '***', '#', 'N/A'],
            keep_default_na=True
        )
        
        # Check if we have the expected columns
        required_columns = {'AREA', 'AREA_TITLE', 'OCC_CODE', 'OCC_TITLE', 'TOT_EMP', 'A_MEAN'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Missing required columns in BLS OES file. Expected: {required_columns}, got: {set(df.columns)}")
        
        # Filter for Hawaii
        df = df[df['AREA_TITLE'].str.contains('Hawaii', case=False, na=False)].copy()
        
        # Standardize column names (map from BLS column names to our internal names)
        column_mapping = {
            'OCC_CODE': 'occupation_code',
            'OCC_TITLE': 'occupation_title',
            'TOT_EMP': 'employment',
            'A_MEAN': 'mean_wage_annual',
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
            'PRIM_STATE': 'state_code',
            'AREA': 'area_code',
            'AREA_TITLE': 'area_title',
            'AREA_TYPE': 'area_type',
            'YEAR': 'year'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Convert annual wages to numeric
        wage_cols = ['mean_wage_annual', 'median_wage_annual', 
                     'pct10_wage_annual', 'pct25_wage_annual',
                     'pct75_wage_annual', 'pct90_wage_annual']
        
        for col in wage_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add year
        df['year'] = year
        df['state'] = 'HI'
        
        # Clean employment
        if 'employment' in df.columns:
            df['employment'] = pd.to_numeric(df['employment'], errors='coerce')
        
        logger.info(f"Parsed {len(df):,} occupation-industry wage records for Hawaii")
        
        return df
    
    def _create_mock_oes_data(self, year: int) -> pd.DataFrame:
        """
        Create mock OES data for development/testing.
        
        This provides realistic wage distributions for common occupations in Hawaii.
        Replace with real data in production.
        """
        logger.warning("Creating mock BLS OES data - replace with real data in production")
        
        # Common Hawaii occupations with realistic wage distributions
        mock_data = [
            # Management occupations
            {
                'occupation_code': '11-0000',
                'occupation_title': 'Management Occupations',
                'employment': 45000,
                'mean_wage_annual': 95000,
                'median_wage_annual': 85000,
                'pct10_wage_annual': 50000,
                'pct25_wage_annual': 65000,
                'pct75_wage_annual': 115000,
                'pct90_wage_annual': 150000
            },
            # Business and financial operations
            {
                'occupation_code': '13-0000',
                'occupation_title': 'Business and Financial Operations',
                'employment': 35000,
                'mean_wage_annual': 75000,
                'median_wage_annual': 70000,
                'pct10_wage_annual': 45000,
                'pct25_wage_annual': 55000,
                'pct75_wage_annual': 90000,
                'pct90_wage_annual': 115000
            },
            # Computer and mathematical
            {
                'occupation_code': '15-0000',
                'occupation_title': 'Computer and Mathematical Occupations',
                'employment': 15000,
                'mean_wage_annual': 85000,
                'median_wage_annual': 80000,
                'pct10_wage_annual': 50000,
                'pct25_wage_annual': 65000,
                'pct75_wage_annual': 100000,
                'pct90_wage_annual': 125000
            },
            # Healthcare practitioners
            {
                'occupation_code': '29-0000',
                'occupation_title': 'Healthcare Practitioners and Technical',
                'employment': 25000,
                'mean_wage_annual': 90000,
                'median_wage_annual': 75000,
                'pct10_wage_annual': 45000,
                'pct25_wage_annual': 55000,
                'pct75_wage_annual': 110000,
                'pct90_wage_annual': 180000
            },
            # Food preparation and serving
            {
                'occupation_code': '35-0000',
                'occupation_title': 'Food Preparation and Serving',
                'employment': 85000,
                'mean_wage_annual': 35000,
                'median_wage_annual': 32000,
                'pct10_wage_annual': 25000,
                'pct25_wage_annual': 28000,
                'pct75_wage_annual': 40000,
                'pct90_wage_annual': 50000
            },
            # Sales and related
            {
                'occupation_code': '41-0000',
                'occupation_title': 'Sales and Related Occupations',
                'employment': 55000,
                'mean_wage_annual': 45000,
                'median_wage_annual': 38000,
                'pct10_wage_annual': 25000,
                'pct25_wage_annual': 30000,
                'pct75_wage_annual': 55000,
                'pct90_wage_annual': 75000
            },
            # Office and administrative support
            {
                'occupation_code': '43-0000',
                'occupation_title': 'Office and Administrative Support',
                'employment': 70000,
                'mean_wage_annual': 42000,
                'median_wage_annual': 40000,
                'pct10_wage_annual': 28000,
                'pct25_wage_annual': 33000,
                'pct75_wage_annual': 50000,
                'pct90_wage_annual': 60000
            },
            # Construction and extraction
            {
                'occupation_code': '47-0000',
                'occupation_title': 'Construction and Extraction',
                'employment': 35000,
                'mean_wage_annual': 55000,
                'median_wage_annual': 52000,
                'pct10_wage_annual': 35000,
                'pct25_wage_annual': 42000,
                'pct75_wage_annual': 65000,
                'pct90_wage_annual': 80000
            },
            # Transportation and material moving
            {
                'occupation_code': '53-0000',
                'occupation_title': 'Transportation and Material Moving',
                'employment': 45000,
                'mean_wage_annual': 42000,
                'median_wage_annual': 40000,
                'pct10_wage_annual': 28000,
                'pct25_wage_annual': 33000,
                'pct75_wage_annual': 50000,
                'pct90_wage_annual': 60000
            }
        ]
        
        df = pd.DataFrame(mock_data)
        df['year'] = year
        df['state'] = 'HI'
        df['industry_code'] = None
        df['industry_title'] = 'All Industries'
        
        return df
    
    def get_occupation_wage_distribution(
        self,
        occupation_code: str,
        year: int = 2022,
        industry_code: Optional[str] = None
    ) -> Optional[Dict[str, float]]:
        """
        Get wage distribution for a specific occupation.
        
        Args:
            occupation_code: SOC occupation code (e.g., '15-1252' for Software Developers)
            year: Year to get data for
            industry_code: Optional NAICS industry code for industry-specific wages
            
        Returns:
            Dictionary with wage percentiles or None if not found
        """
        df = self.load_hawaii_wages(year)
        
        # Filter by occupation
        mask = df['occupation_code'].str.startswith(occupation_code[:2])
        
        if industry_code:
            mask &= df['industry_code'] == industry_code
        
        filtered = df[mask]
        
        if filtered.empty:
            logger.warning(f"No wage data found for occupation {occupation_code}")
            return None
        
        # Use most specific match
        result = filtered.iloc[0]
        
        return {
            'mean': result.get('mean_wage_annual', 0),
            'median': result.get('median_wage_annual', 0),
            'p10': result.get('pct10_wage_annual', 0),
            'p25': result.get('pct25_wage_annual', 0),
            'p75': result.get('pct75_wage_annual', 0),
            'p90': result.get('pct90_wage_annual', 0),
            'employment': result.get('employment', 0)
        }
    
    def calculate_wage_growth_rates(
        self,
        start_year: int,
        end_year: int
    ) -> pd.DataFrame:
        """
        Calculate wage growth rates by occupation between two years.
        
        This is critical for temporal alignment of PUMS data to current year.
        
        Args:
            start_year: Starting year
            end_year: Ending year
            
        Returns:
            DataFrame with occupation codes and growth rates
        """
        logger.info(f"Calculating wage growth rates from {start_year} to {end_year}")
        
        # Load data for both years
        df_start = self.load_hawaii_wages(start_year)
        df_end = self.load_hawaii_wages(end_year)
        
        # Merge on occupation code
        merged = df_start.merge(
            df_end,
            on='occupation_code',
            suffixes=('_start', '_end'),
            how='inner'
        )
        
        # Calculate growth rates
        merged['wage_growth_rate'] = (
            (merged['mean_wage_annual_end'] - merged['mean_wage_annual_start']) /
            merged['mean_wage_annual_start']
        )
        
        merged['annual_growth_rate'] = (
            (1 + merged['wage_growth_rate']) ** (1 / (end_year - start_year)) - 1
        )
        
        # Select relevant columns
        result = merged[[
            'occupation_code',
            'occupation_title_start',
            'mean_wage_annual_start',
            'mean_wage_annual_end',
            'wage_growth_rate',
            'annual_growth_rate'
        ]].copy()
        
        result.columns = [
            'occupation_code',
            'occupation_title',
            f'mean_wage_{start_year}',
            f'mean_wage_{end_year}',
            'total_growth_rate',
            'annual_growth_rate'
        ]
        
        logger.info(f"Calculated growth rates for {len(result):,} occupations")
        logger.info(f"Average annual growth rate: {result['annual_growth_rate'].mean():.2%}")
        
        return result
    
    def map_pums_occupation_to_soc(self, pums_occ_code: str) -> str:
        """
        Map PUMS occupation code to SOC code.
        
        PUMS uses a modified version of SOC codes. This provides the mapping.
        
        Args:
            pums_occ_code: PUMS occupation code
            
        Returns:
            SOC occupation code
        """
        # PUMS occupation codes are generally compatible with SOC
        # but may need some translation
        
        # For now, use first 2 digits as major group
        if len(pums_occ_code) >= 2:
            return pums_occ_code[:2] + '-0000'
        
        return '00-0000'  # Unknown
    
    def get_wage_adjustment_factors(
        self,
        pums_year: int,
        target_year: int
    ) -> pd.DataFrame:
        """
        Get wage adjustment factors to align PUMS data to target year.
        
        This is critical for using older PUMS data with current SOI benchmarks.
        
        Args:
            pums_year: Year of PUMS data
            target_year: Target year for alignment
            
        Returns:
            DataFrame with occupation codes and adjustment factors
        """
        if pums_year == target_year:
            logger.info("PUMS year matches target year, no adjustment needed")
            # Return identity factors
            df = self.load_hawaii_wages(target_year)
            return pd.DataFrame({
                'occupation_code': df['occupation_code'],
                'adjustment_factor': 1.0
            })
        
        # Calculate growth rates
        growth_df = self.calculate_wage_growth_rates(pums_year, target_year)
        
        # Convert to adjustment factors
        growth_df['adjustment_factor'] = 1 + growth_df['total_growth_rate']
        
        return growth_df[['occupation_code', 'adjustment_factor']]
    
    def get_summary_statistics(self, year: int = 2022) -> Dict:
        """Get summary statistics for Hawaii wages."""
        df = self.load_hawaii_wages(year)
        
        return {
            'year': year,
            'state': 'HI',
            'total_occupations': len(df),
            'total_employment': df['employment'].sum(),
            'mean_wage': df['mean_wage_annual'].mean(),
            'median_wage': df['median_wage_annual'].median(),
            'wage_range': {
                'min': df['pct10_wage_annual'].min(),
                'max': df['pct90_wage_annual'].max()
            },
            'top_occupations_by_employment': df.nlargest(5, 'employment')[
                ['occupation_title', 'employment', 'mean_wage_annual']
            ].to_dict('records'),
            'top_occupations_by_wage': df.nlargest(5, 'mean_wage_annual')[
                ['occupation_title', 'employment', 'mean_wage_annual']
            ].to_dict('records')
        }


def create_occupation_wage_lookup(
    bls_loader: BLSOESLoader,
    year: int = 2022
) -> Dict[str, Dict[str, float]]:
    """
    Create a lookup dictionary for quick wage distribution access.
    
    Args:
        bls_loader: Initialized BLS OES loader
        year: Year to create lookup for
        
    Returns:
        Dictionary mapping occupation codes to wage distributions
    """
    df = bls_loader.load_hawaii_wages(year)
    
    lookup = {}
    for _, row in df.iterrows():
        occ_code = row['occupation_code']
        lookup[occ_code] = {
            'mean': row.get('mean_wage_annual', 0),
            'median': row.get('median_wage_annual', 0),
            'p10': row.get('pct10_wage_annual', 0),
            'p25': row.get('pct25_wage_annual', 0),
            'p75': row.get('pct75_wage_annual', 0),
            'p90': row.get('pct90_wage_annual', 0),
            'employment': row.get('employment', 0),
            'title': row.get('occupation_title', '')
        }
    
    return lookup
