"""
National SOI Public Use File (PUF) Data Loader.

This module loads and processes the IRS SOI Public Use File to provide:
- Detailed income relationships and tax patterns
- High-income household characteristics
- Income source correlations
- Tax calculation patterns

Data Source: https://www.irs.gov/statistics/soi-tax-stats-individual-public-use-microdata-files
Free public microdata updated annually (with 3-year lag).

Key Features:
- Comprehensive income and deduction data
- Actual tax return patterns
- Income source relationships
- High-income oversampling
- Statistical matching templates
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import logging
from dataclasses import dataclass
import requests

logger = logging.getLogger(__name__)


@dataclass
class SOIPUFRecord:
    """Container for SOI PUF tax return data."""
    agi: float
    wages: float
    interest: float
    dividends: float
    business_income: float
    capital_gains: float
    ira_distributions: float
    pensions: float
    rental_income: float
    filing_status: str
    exemptions: int
    itemized_deductions: float
    tax_liability: float
    credits: float
    weight: float
    
    @property
    def total_investment_income(self) -> float:
        """Total investment income."""
        return self.interest + self.dividends + self.capital_gains
    
    @property
    def total_business_income(self) -> float:
        """Total business income."""
        return self.business_income + self.rental_income
    
    @property
    def wage_share(self) -> float:
        """Wage income as share of AGI."""
        return self.wages / self.agi if self.agi > 0 else 0


class NationalSOIPUFLoader:
    """
    Loader for IRS SOI Public Use File.
    
    Provides detailed income patterns and tax relationships from actual
    tax returns, critical for modeling complex income sources.
    """
    
    # IRS SOI PUF URLs
    SOI_PUF_BASE_URL = "https://www.irs.gov/pub/irs-soi"
    
    # Variable mappings (SOI PUF uses specific codes)
    VARIABLE_MAP = {
        # Income variables
        'MARS': 'filing_status',  # Marital status
        'XTOT': 'exemptions',  # Total exemptions
        'A00100': 'agi',  # Adjusted Gross Income
        'A00200': 'wages',  # Wages and salaries
        'A00300': 'interest',  # Taxable interest
        'A00600': 'dividends',  # Ordinary dividends
        'A00650': 'qualified_dividends',  # Qualified dividends
        'A00900': 'business_income',  # Business income/loss
        'A01000': 'capital_gains',  # Capital gains/losses
        'A01400': 'ira_distributions',  # IRA distributions
        'A01700': 'pensions',  # Pensions and annuities
        'A02300': 'unemployment',  # Unemployment compensation
        'A02500': 'social_security',  # Social security benefits
        'A03300': 'rental_income',  # Rental income/loss
        
        # Deductions
        'A04470': 'itemized_deductions',  # Total itemized deductions
        'A19700': 'state_local_tax',  # State and local taxes
        'A19300': 'mortgage_interest',  # Mortgage interest
        'A19800': 'charitable_contributions',  # Charitable contributions
        
        # Tax and credits
        'A06500': 'income_tax',  # Income tax before credits
        'A07100': 'total_tax',  # Total tax liability
        'A11070': 'ctc',  # Child Tax Credit
        'A11580': 'eitc',  # Earned Income Tax Credit
        'A85530': 'premium_tax_credit',  # Premium Tax Credit
        
        # Weight
        'S006': 'weight'  # Sample weight
    }
    
    # Filing status codes
    FILING_STATUS_MAP = {
        1: 'single',
        2: 'joint',
        3: 'mfs',
        4: 'hoh',
        5: 'widow'
    }
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        cache_data: bool = True
    ):
        """
        Initialize National SOI PUF loader.
        
        Args:
            data_dir: Directory for cached data
            cache_data: Whether to cache downloaded data
        """
        self.data_dir = Path(data_dir) if data_dir else Path('data/external/soi_puf')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_data = cache_data
        
        self._puf_data_cache: Dict[int, pd.DataFrame] = {}
    
    def load_puf_data(
        self,
        year: int = 2019,  # Most recent available
        force_download: bool = False
    ) -> pd.DataFrame:
        """
        Load SOI PUF data for a specific year.
        
        Args:
            year: Year to load data for (note: 3-year lag)
            force_download: Force download even if cached
            
        Returns:
            DataFrame with tax return data
        """
        # Check cache
        if not force_download and year in self._puf_data_cache:
            logger.info(f"Using cached SOI PUF data for {year}")
            return self._puf_data_cache[year]
        
        # Check file cache
        cache_file = self.data_dir / f'soi_puf_{year}.parquet'
        if not force_download and cache_file.exists():
            logger.info(f"Loading SOI PUF data from cache: {cache_file}")
            df = pd.read_parquet(cache_file)
            self._puf_data_cache[year] = df
            return df
        
        # Download from IRS
        logger.info(f"Downloading SOI PUF data for year {year}")
        df = self._download_puf_data(year)
        
        # Cache the data
        if self.cache_data:
            df.to_parquet(cache_file, index=False)
            logger.info(f"Cached SOI PUF data to {cache_file}")
        
        self._puf_data_cache[year] = df
        return df
    
    def _download_puf_data(self, year: int) -> pd.DataFrame:
        """
        Download SOI PUF data from IRS.
        
        Note: IRS publishes PUF as CSV files. For production, implement
        actual download logic.
        """
        logger.warning("SOI PUF download not fully implemented - using mock data")
        logger.info(f"To use real data, download from: {self.SOI_PUF_BASE_URL}")
        logger.info(f"File format: {year}puf.csv")
        
        # Check if user has downloaded the data manually
        manual_file = self.data_dir / f'{year}puf.csv'
        if manual_file.exists():
            logger.info(f"Found manually downloaded SOI PUF file: {manual_file}")
            return self._parse_puf_csv(manual_file, year)
        
        # Return mock data for development
        return self._create_mock_puf_data(year)
    
    def _parse_puf_csv(self, file_path: Path, year: int) -> pd.DataFrame:
        """
        Parse SOI PUF CSV file.
        
        The PUF file contains detailed tax return data with specific variable codes.
        """
        logger.info(f"Parsing SOI PUF CSV file: {file_path}")
        
        # Read CSV
        df = pd.read_csv(file_path)
        
        # Rename columns using variable map
        available_vars = {k: v for k, v in self.VARIABLE_MAP.items() if k in df.columns}
        df = df.rename(columns=available_vars)
        
        # Map filing status
        if 'filing_status' in df.columns:
            df['filing_status'] = df['filing_status'].map(self.FILING_STATUS_MAP)
        
        # Convert to numeric
        numeric_cols = ['agi', 'wages', 'interest', 'dividends', 'business_income',
                       'capital_gains', 'ira_distributions', 'pensions', 'rental_income',
                       'itemized_deductions', 'income_tax', 'total_tax', 'weight']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add year
        df['year'] = year
        
        # Calculate derived variables
        df['total_investment_income'] = (
            df.get('interest', 0) + 
            df.get('dividends', 0) + 
            df.get('capital_gains', 0)
        )
        
        df['total_business_income'] = (
            df.get('business_income', 0) + 
            df.get('rental_income', 0)
        )
        
        if 'agi' in df.columns and 'wages' in df.columns:
            df['wage_share'] = df['wages'] / df['agi'].replace(0, np.nan)
            df['wage_share'] = df['wage_share'].fillna(0).clip(0, 1)
        
        logger.info(f"Parsed {len(df):,} SOI PUF tax returns")
        logger.info(f"Weighted returns: {df['weight'].sum():,.0f}")
        
        return df
    
    def _create_mock_puf_data(self, year: int) -> pd.DataFrame:
        """
        Create mock SOI PUF data for development/testing.
        
        This provides realistic tax return patterns with proper income relationships.
        Replace with real data in production.
        """
        logger.warning("Creating mock SOI PUF data - replace with real data in production")
        
        np.random.seed(42)
        n_samples = 10000
        
        # Filing status distribution
        filing_status = np.random.choice(
            ['single', 'joint', 'hoh', 'mfs'],
            size=n_samples,
            p=[0.51, 0.36, 0.10, 0.03]
        )
        
        # AGI distribution (log-normal with high-income oversampling)
        # SOI PUF oversamples high-income returns
        agi_base = np.random.lognormal(mean=10.5, sigma=1.2, size=n_samples)
        
        # Oversample high incomes
        high_income_mask = np.random.random(n_samples) < 0.2
        agi_base[high_income_mask] = np.random.lognormal(mean=12.5, sigma=0.8, 
                                                          size=high_income_mask.sum())
        
        agi = agi_base
        
        # Wages (primary income source for most)
        wage_share = np.random.beta(8, 2, n_samples)
        wage_share = np.where(agi > 200000, 
                             np.random.beta(5, 3, n_samples),  # High earners: more diverse
                             wage_share)
        wages = agi * wage_share
        
        # Investment income (increases with AGI)
        investment_prob = np.minimum(0.8, agi / 200000)
        has_investment = np.random.random(n_samples) < investment_prob
        
        interest = np.where(has_investment, 
                           agi * np.random.beta(1, 20, n_samples) * 0.02,
                           0)
        
        dividends = np.where(has_investment,
                            agi * np.random.beta(1, 15, n_samples) * 0.03,
                            0)
        
        capital_gains = np.where(has_investment & (np.random.random(n_samples) < 0.3),
                                agi * np.random.beta(1, 10, n_samples) * 0.10,
                                0)
        
        # Business income (some taxpayers)
        has_business = np.random.random(n_samples) < 0.15
        business_income = np.where(has_business,
                                  agi * np.random.beta(2, 5, n_samples) * 0.30,
                                  0)
        
        # Rental income (some taxpayers)
        has_rental = np.random.random(n_samples) < 0.10
        rental_income = np.where(has_rental,
                                agi * np.random.beta(1, 10, n_samples) * 0.15,
                                0)
        
        # Retirement income
        has_retirement = np.random.random(n_samples) < 0.25
        ira_distributions = np.where(has_retirement,
                                     agi * np.random.beta(2, 8, n_samples) * 0.20,
                                     0)
        
        pensions = np.where(has_retirement,
                           agi * np.random.beta(2, 6, n_samples) * 0.25,
                           0)
        
        # Exemptions
        exemptions = np.where(filing_status == 'joint',
                             np.random.choice([2, 3, 4, 5], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
                             np.random.choice([1, 2, 3], n_samples, p=[0.6, 0.3, 0.1]))
        
        # Itemized deductions (more common for high income)
        itemizes = np.random.random(n_samples) < np.minimum(0.7, agi / 150000)
        itemized_deductions = np.where(itemizes,
                                      agi * np.random.beta(2, 10, n_samples) * 0.15,
                                      0)
        
        # Tax liability (simplified calculation)
        taxable_income = np.maximum(0, agi - itemized_deductions - exemptions * 4000)
        income_tax = taxable_income * 0.15  # Simplified
        
        # Credits
        ctc = np.where((exemptions > 2) & (agi < 200000),
                      (exemptions - 2) * 2000,
                      0)
        
        eitc = np.where((agi < 50000) & (exemptions > 1),
                       np.minimum(6000, agi * 0.10),
                       0)
        
        total_tax = np.maximum(0, income_tax - ctc - eitc)
        
        # Weights (SOI PUF uses complex weighting)
        # High-income returns have lower weights (oversampled)
        base_weight = 1000
        weight = np.where(agi > 200000,
                         base_weight / 10,
                         base_weight)
        
        # Create DataFrame
        df = pd.DataFrame({
            'filing_status': filing_status,
            'exemptions': exemptions,
            'agi': agi,
            'wages': wages,
            'interest': interest,
            'dividends': dividends,
            'business_income': business_income,
            'capital_gains': capital_gains,
            'ira_distributions': ira_distributions,
            'pensions': pensions,
            'rental_income': rental_income,
            'itemized_deductions': itemized_deductions,
            'income_tax': income_tax,
            'total_tax': total_tax,
            'ctc': ctc,
            'eitc': eitc,
            'weight': weight,
            'year': year
        })
        
        # Calculate derived variables
        df['total_investment_income'] = df['interest'] + df['dividends'] + df['capital_gains']
        df['total_business_income'] = df['business_income'] + df['rental_income']
        df['wage_share'] = df['wages'] / df['agi'].replace(0, np.nan)
        df['wage_share'] = df['wage_share'].fillna(0).clip(0, 1)
        
        return df
    
    def get_income_relationships(
        self,
        year: int = 2019,
        income_bracket: Optional[Tuple[float, float]] = None
    ) -> pd.DataFrame:
        """
        Get income source relationships for statistical matching.
        
        This shows how different income sources correlate, critical for
        imputing missing income components.
        
        Args:
            year: Year to analyze
            income_bracket: Optional (min, max) AGI bracket
            
        Returns:
            Correlation matrix of income sources
        """
        df = self.load_puf_data(year)
        
        # Filter by income bracket
        if income_bracket:
            df = df[(df['agi'] >= income_bracket[0]) & (df['agi'] <= income_bracket[1])]
            logger.info(f"Filtered to AGI ${income_bracket[0]:,.0f} - ${income_bracket[1]:,.0f}")
        
        # Select income variables
        income_vars = ['wages', 'interest', 'dividends', 'business_income',
                      'capital_gains', 'ira_distributions', 'pensions', 'rental_income']
        
        available_vars = [v for v in income_vars if v in df.columns]
        
        # Calculate correlation matrix
        corr_matrix = df[available_vars].corr()
        
        return corr_matrix
    
    def get_income_composition_by_agi(
        self,
        year: int = 2019,
        n_brackets: int = 10
    ) -> pd.DataFrame:
        """
        Get income composition by AGI bracket.
        
        Shows how income sources change with income level.
        
        Args:
            year: Year to analyze
            n_brackets: Number of AGI brackets
            
        Returns:
            DataFrame with income composition by bracket
        """
        df = self.load_puf_data(year)
        
        # Create AGI brackets
        df['agi_bracket'] = pd.qcut(df['agi'], q=n_brackets, labels=False, duplicates='drop')
        
        # Calculate statistics by bracket
        income_cols = ['wages', 'interest', 'dividends', 'business_income',
                      'capital_gains', 'ira_distributions', 'pensions', 'rental_income']
        
        available_cols = [c for c in income_cols if c in df.columns]
        
        grouped = df.groupby('agi_bracket').agg({
            'agi': ['min', 'max', 'mean', 'median'],
            'weight': 'sum',
            **{col: ['mean', 'median'] for col in available_cols}
        })
        
        # Calculate shares
        for col in available_cols:
            grouped[(col, 'share')] = grouped[(col, 'mean')] / grouped[('agi', 'mean')]
        
        return grouped
    
    def create_matching_template(
        self,
        year: int = 2019,
        matching_vars: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create a template for statistical matching.
        
        This provides donor records for matching to PUMS recipients.
        
        Args:
            year: Year to create template for
            matching_vars: Variables to include in template
            
        Returns:
            DataFrame suitable for statistical matching
        """
        df = self.load_puf_data(year)
        
        if matching_vars is None:
            matching_vars = [
                'filing_status', 'agi', 'wages', 'exemptions',
                'total_investment_income', 'total_business_income'
            ]
        
        # Select matching variables
        available_vars = [v for v in matching_vars if v in df.columns]
        template = df[available_vars + ['weight']].copy()
        
        # Add ID
        template['donor_id'] = range(len(template))
        
        logger.info(f"Created matching template with {len(template):,} donor records")
        
        return template
    
    def get_high_income_patterns(
        self,
        year: int = 2019,
        income_threshold: float = 200000
    ) -> Dict[str, any]:
        """
        Get patterns for high-income returns.
        
        Critical for addressing PUMS undercounting of high-income households.
        
        Args:
            year: Year to analyze
            income_threshold: AGI threshold for "high income"
            
        Returns:
            Dictionary with high-income statistics
        """
        df = self.load_puf_data(year)
        
        high_income = df[df['agi'] >= income_threshold]
        all_returns = df
        
        stats = {
            'threshold': income_threshold,
            'n_returns': len(high_income),
            'pct_of_returns': len(high_income) / len(all_returns),
            'weighted_returns': high_income['weight'].sum(),
            'mean_agi': high_income['agi'].mean(),
            'median_agi': high_income['agi'].median(),
            'income_composition': {
                'wage_share': (high_income['wages'] / high_income['agi']).mean(),
                'investment_share': (high_income['total_investment_income'] / high_income['agi']).mean(),
                'business_share': (high_income['total_business_income'] / high_income['agi']).mean()
            },
            'filing_status_distribution': high_income['filing_status'].value_counts(normalize=True).to_dict(),
            'mean_exemptions': high_income['exemptions'].mean(),
            'pct_itemizing': (high_income['itemized_deductions'] > 0).mean()
        }
        
        return stats
    
    def get_summary_statistics(self, year: int = 2019) -> Dict:
        """Get summary statistics for SOI PUF data."""
        df = self.load_puf_data(year)
        
        return {
            'year': year,
            'sample_size': len(df),
            'weighted_returns': df['weight'].sum(),
            'mean_agi': df['agi'].mean(),
            'median_agi': df['agi'].median(),
            'filing_status_distribution': df['filing_status'].value_counts(normalize=True).to_dict(),
            'income_composition': {
                'wage_share': (df['wages'] / df['agi']).mean(),
                'investment_share': (df['total_investment_income'] / df['agi']).mean(),
                'business_share': (df['total_business_income'] / df['agi']).mean()
            },
            'high_income_stats': {
                'pct_over_100k': (df['agi'] > 100000).mean(),
                'pct_over_200k': (df['agi'] > 200000).mean(),
                'pct_over_500k': (df['agi'] > 500000).mean()
            }
        }


def match_pums_to_soi_puf(
    pums_data: pd.DataFrame,
    soi_puf_loader: NationalSOIPUFLoader,
    year: int = 2019,
    matching_vars: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Match PUMS tax units to SOI PUF donors for income imputation.
    
    This is a convenience function that combines the SOI PUF loader with
    the statistical matching framework.
    
    Args:
        pums_data: PUMS tax unit data
        soi_puf_loader: Initialized SOI PUF loader
        year: Year to match
        matching_vars: Variables to match on
        
    Returns:
        PUMS data enhanced with SOI PUF income patterns
    """
    from .statistical_matching import StatisticalMatcher, MatchingConfig, MatchingMethod
    
    # Load SOI PUF template
    soi_template = soi_puf_loader.create_matching_template(year, matching_vars)
    
    # Configure matching
    if matching_vars is None:
        matching_vars = ['filing_status', 'agi', 'wages']
    
    config = MatchingConfig(
        method=MatchingMethod.PROPENSITY_SCORE,
        matching_variables=matching_vars,
        caliper=0.25,
        use_weights=True
    )
    
    # Perform matching
    matcher = StatisticalMatcher(config)
    matched_data, match_results = matcher.match(
        recipients=pums_data,
        donors=soi_template,
        recipient_id_col='tax_unit_id',
        donor_id_col='donor_id'
    )
    
    logger.info(f"Matched {len(matched_data):,} PUMS records to SOI PUF donors")
    
    return matched_data
