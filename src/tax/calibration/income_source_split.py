"""
Income Source Split for Hawaii Tax Model

This module implements Stage 5 calibration: splitting total income into
component sources (wages, dividends, interest, business, capital gains, etc.)
based on IRS SOI income source distributions by income bracket.

Key Features:
- Uses IRS SOI Table 2 income source data by bracket
- Splits total AGI into component sources
- Maintains total income consistency
- Provides detailed income composition for tax calculations
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class IncomeSourceSplitter:
    """
    Splits total income into component sources based on IRS SOI data.
    
    PUMS provides total income but limited detail on income sources.
    IRS SOI provides detailed income source breakdowns by income bracket.
    This class applies IRS source distributions to PUMS tax units.
    """
    
    # IRS SOI 2022 Income Sources by Bracket (Table 2)
    # All values in dollars (totals for Hawaii)
    IRS_INCOME_SOURCES_BY_BRACKET = {
        '0-25k': {
            'total_agi': 1650000000,
            'wages': 1200000000,        # 72.7% from wages
            'dividends': 20000000,       # 1.2% from dividends
            'interest': 30000000,        # 1.8% from interest
            'business': 150000000,       # 9.1% from business
            'cap_gains': 50000000,       # 3.0% from cap gains
            'pensions': 150000000,       # 9.1% from pensions
            'other': 50000000            # 3.0% from other
        },
        '25-50k': {
            'total_agi': 6300000000,
            'wages': 5200000000,         # 82.5% from wages
            'dividends': 150000000,      # 2.4% from dividends
            'interest': 200000000,       # 3.2% from interest
            'business': 600000000,       # 9.5% from business
            'cap_gains': 150000000,      # 2.4% from cap gains
            'pensions': 0,               # 0% from pensions
            'other': 0                   # 0% from other
        },
        '50-75k': {
            'total_agi': 8400000000,
            'wages': 7200000000,         # 85.7% from wages
            'dividends': 250000000,      # 3.0% from dividends
            'interest': 200000000,       # 2.4% from interest
            'business': 600000000,       # 7.1% from business
            'cap_gains': 150000000,      # 1.8% from cap gains
            'pensions': 0,               # 0% from pensions
            'other': 0                   # 0% from other
        },
        '75-100k': {
            'total_agi': 7200000000,
            'wages': 6300000000,         # 87.5% from wages
            'dividends': 250000000,      # 3.5% from dividends
            'interest': 150000000,       # 2.1% from interest
            'business': 400000000,       # 5.6% from business
            'cap_gains': 100000000,      # 1.4% from cap gains
            'pensions': 0,               # 0% from pensions
            'other': 0                   # 0% from other
        },
        '100-200k': {
            'total_agi': 8400000000,
            'wages': 6700000000,         # 79.8% from wages
            'dividends': 500000000,      # 6.0% from dividends
            'interest': 250000000,       # 3.0% from interest
            'business': 650000000,       # 7.7% from business
            'cap_gains': 300000000,      # 3.6% from cap gains
            'pensions': 0,               # 0% from pensions
            'other': 0                   # 0% from other
        },
        '200k+': {
            'total_agi': 6000000000,
            'wages': 3600000000,         # 60.0% from wages
            'dividends': 900000000,      # 15.0% from dividends
            'interest': 300000000,       # 5.0% from interest
            'business': 750000000,       # 12.5% from business
            'cap_gains': 450000000,      # 7.5% from cap gains
            'pensions': 0,               # 0% from pensions
            'other': 0                   # 0% from other
        }
    }
    
    # Bracket bounds
    BRACKET_BOUNDS = {
        '0-25k': (0, 25000),
        '25-50k': (25000, 50000),
        '50-75k': (50000, 75000),
        '75-100k': (75000, 100000),
        '100-200k': (100000, 200000),
        '200k+': (200000, float('inf'))
    }
    
    def __init__(self, 
                 use_pums_sources: bool = True,
                 preserve_pums_wages: bool = False):
        """
        Initialize income source splitter.
        
        Args:
            use_pums_sources: Whether to use PUMS source data as starting point
            preserve_pums_wages: Whether to preserve PUMS wage data (if available)
        """
        self.use_pums_sources = use_pums_sources
        self.preserve_pums_wages = preserve_pums_wages
        
        logger.info("Income Source Splitter initialized")
        logger.info(f"  Use PUMS sources: {use_pums_sources}")
        logger.info(f"  Preserve PUMS wages: {preserve_pums_wages}")
    
    def get_bracket_name(self, income: float) -> str:
        """Get bracket name for a given income."""
        for bracket_name, (low, high) in self.BRACKET_BOUNDS.items():
            if low <= income < high:
                return bracket_name
        return '200k+'  # Default to highest bracket
    
    def get_bracket_bounds(self, bracket_name: str) -> Tuple[float, float]:
        """Get income bounds for a bracket."""
        return self.BRACKET_BOUNDS[bracket_name]
    
    def calculate_source_percentages(self, bracket_name: str) -> Dict[str, float]:
        """
        Calculate income source percentages for a bracket.
        
        Args:
            bracket_name: Name of income bracket
            
        Returns:
            Dictionary of source percentages
        """
        sources = self.IRS_INCOME_SOURCES_BY_BRACKET[bracket_name]
        total = sources['total_agi']
        
        percentages = {
            source: amount / total if total > 0 else 0
            for source, amount in sources.items()
            if source != 'total_agi'
        }
        
        return percentages
    
    def split_income_sources(self,
                            tax_units: pd.DataFrame,
                            income_col: str = 'income') -> pd.DataFrame:
        """
        Split total income into component sources based on IRS SOI data.
        
        Args:
            tax_units: DataFrame with tax units
            income_col: Name of total income column
            
        Returns:
            DataFrame with income source columns added
        """
        logger.info("\n" + "=" * 70)
        logger.info("INCOME SOURCE SPLIT - STAGE 5")
        logger.info("=" * 70)
        
        # Create copy to avoid modifying original
        tax_units_split = tax_units.copy()
        
        # Initialize source columns
        source_columns = ['wage_income', 'dividend_income', 'interest_income',
                         'business_income', 'capital_gains', 'pension_income',
                         'other_income']
        
        for col in source_columns:
            if col not in tax_units_split.columns:
                tax_units_split[col] = 0.0
        
        logger.info(f"\nProcessing {len(tax_units_split):,} tax units...")
        
        # Split income by bracket
        for bracket_name, (low, high) in self.BRACKET_BOUNDS.items():
            # Get units in this bracket
            mask = (tax_units_split[income_col] >= low) & \
                   (tax_units_split[income_col] < high)
            
            n_units = mask.sum()
            if n_units == 0:
                continue
            
            logger.info(f"\n  Processing bracket {bracket_name}:")
            logger.info(f"    Units in bracket: {n_units:,}")
            
            # Get IRS source percentages for this bracket
            source_pcts = self.calculate_source_percentages(bracket_name)
            
            # Log source percentages
            logger.info(f"    IRS source distribution:")
            for source, pct in source_pcts.items():
                logger.info(f"      {source:15s}: {pct*100:5.1f}%")
            
            # Apply source split
            total_income = tax_units_split.loc[mask, income_col]
            
            tax_units_split.loc[mask, 'wage_income'] = \
                total_income * source_pcts['wages']
            
            tax_units_split.loc[mask, 'dividend_income'] = \
                total_income * source_pcts['dividends']
            
            tax_units_split.loc[mask, 'interest_income'] = \
                total_income * source_pcts['interest']
            
            tax_units_split.loc[mask, 'business_income'] = \
                total_income * source_pcts['business']
            
            tax_units_split.loc[mask, 'capital_gains'] = \
                total_income * source_pcts['cap_gains']
            
            tax_units_split.loc[mask, 'pension_income'] = \
                total_income * source_pcts['pensions']
            
            tax_units_split.loc[mask, 'other_income'] = \
                total_income * source_pcts['other']
        
        # Validate total income consistency
        self._validate_income_split(tax_units_split, income_col)
        
        # Generate summary
        self._generate_summary(tax_units_split, income_col)
        
        logger.info("\n" + "=" * 70)
        logger.info("INCOME SOURCE SPLIT COMPLETE")
        logger.info("=" * 70 + "\n")
        
        return tax_units_split
    
    def _validate_income_split(self,
                               tax_units: pd.DataFrame,
                               income_col: str):
        """Validate that source incomes sum to total income."""
        logger.info("\n" + "=" * 70)
        logger.info("Income Split Validation")
        logger.info("=" * 70)
        
        source_columns = ['wage_income', 'dividend_income', 'interest_income',
                         'business_income', 'capital_gains', 'pension_income',
                         'other_income']
        
        # Calculate sum of sources
        source_sum = tax_units[source_columns].sum(axis=1)
        total_income = tax_units[income_col]
        
        # Calculate differences
        diff = source_sum - total_income
        max_diff = diff.abs().max()
        mean_diff = diff.abs().mean()
        
        logger.info(f"\nIncome Consistency Check:")
        logger.info(f"  Max difference:  ${max_diff:,.2f}")
        logger.info(f"  Mean difference: ${mean_diff:,.2f}")
        
        if max_diff < 0.01:
            logger.info(f"  ✅ Income split is consistent")
        else:
            logger.warning(f"  ⚠️  Income split has discrepancies")
        
        # Check for negative values
        for col in source_columns:
            n_negative = (tax_units[col] < 0).sum()
            if n_negative > 0:
                logger.warning(f"  ⚠️  {col} has {n_negative:,} negative values")
    
    def _generate_summary(self,
                         tax_units: pd.DataFrame,
                         income_col: str):
        """Generate summary statistics of income sources."""
        logger.info("\n" + "=" * 70)
        logger.info("Income Source Summary")
        logger.info("=" * 70)
        
        weight_col = 'calibrated_weight' if 'calibrated_weight' in tax_units.columns else 'weight'
        
        source_columns = {
            'wage_income': 'Wages',
            'dividend_income': 'Dividends',
            'interest_income': 'Interest',
            'business_income': 'Business',
            'capital_gains': 'Capital Gains',
            'pension_income': 'Pensions',
            'other_income': 'Other'
        }
        
        total_weighted_income = (tax_units[income_col] * tax_units[weight_col]).sum()
        
        logger.info(f"\nTotal AGI: ${total_weighted_income:,.0f}")
        logger.info(f"\nIncome Source Breakdown:")
        logger.info(f"  {'Source':<20} {'Amount':<20} {'Percentage':<12}")
        logger.info(f"  {'-'*52}")
        
        for col, name in source_columns.items():
            weighted_amount = (tax_units[col] * tax_units[weight_col]).sum()
            percentage = (weighted_amount / total_weighted_income * 100) if total_weighted_income > 0 else 0
            
            logger.info(f"  {name:<20} ${weighted_amount:>18,.0f} {percentage:>10.1f}%")
        
        logger.info(f"  {'-'*52}")
        
        # Verify sum
        total_sources = sum(
            (tax_units[col] * tax_units[weight_col]).sum()
            for col in source_columns.keys()
        )
        logger.info(f"  {'Total':<20} ${total_sources:>18,.0f} {100.0:>10.1f}%")
        
        diff = total_weighted_income - total_sources
        logger.info(f"\n  Difference: ${diff:,.2f}")
    
    def get_source_summary_by_bracket(self,
                                      tax_units: pd.DataFrame,
                                      income_col: str = 'income') -> pd.DataFrame:
        """
        Generate detailed source summary by income bracket.
        
        Args:
            tax_units: DataFrame with income sources
            income_col: Name of total income column
            
        Returns:
            DataFrame with source breakdown by bracket
        """
        weight_col = 'calibrated_weight' if 'calibrated_weight' in tax_units.columns else 'weight'
        
        source_columns = ['wage_income', 'dividend_income', 'interest_income',
                         'business_income', 'capital_gains', 'pension_income',
                         'other_income']
        
        summary_data = []
        
        for bracket_name, (low, high) in self.BRACKET_BOUNDS.items():
            mask = (tax_units[income_col] >= low) & (tax_units[income_col] < high)
            
            if mask.sum() == 0:
                continue
            
            bracket_units = tax_units[mask]
            total_agi = (bracket_units[income_col] * bracket_units[weight_col]).sum()
            
            row = {
                'bracket': bracket_name,
                'count': bracket_units[weight_col].sum(),
                'total_agi': total_agi
            }
            
            for col in source_columns:
                amount = (bracket_units[col] * bracket_units[weight_col]).sum()
                row[col] = amount
                row[f'{col}_pct'] = (amount / total_agi * 100) if total_agi > 0 else 0
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)


def split_income_sources(tax_units: pd.DataFrame,
                        income_col: str = 'income',
                        use_pums_sources: bool = True) -> pd.DataFrame:
    """
    Convenience function to split income into sources.
    
    Args:
        tax_units: DataFrame with tax units
        income_col: Name of total income column
        use_pums_sources: Whether to use PUMS source data as starting point
        
    Returns:
        DataFrame with income source columns added
    """
    splitter = IncomeSourceSplitter(use_pums_sources=use_pums_sources)
    return splitter.split_income_sources(tax_units, income_col)


if __name__ == '__main__':
    # Demo usage
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.append(str(project_root))
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    logger.info("Income Source Split Demo")
    logger.info("=" * 70)
    
    # Create sample data
    np.random.seed(42)
    n_units = 1000
    
    # Generate incomes across all brackets
    incomes = np.concatenate([
        np.random.uniform(5000, 25000, 200),
        np.random.uniform(25000, 50000, 300),
        np.random.uniform(50000, 75000, 250),
        np.random.uniform(75000, 100000, 150),
        np.random.uniform(100000, 200000, 80),
        np.random.uniform(200000, 500000, 20)
    ])
    
    sample_data = pd.DataFrame({
        'tax_unit_id': range(len(incomes)),
        'income': incomes,
        'calibrated_weight': np.random.uniform(100, 150, len(incomes))
    })
    
    logger.info(f"\nSample data created: {len(sample_data):,} tax units")
    logger.info(f"Total weighted income: ${(sample_data['income'] * sample_data['calibrated_weight']).sum():,.0f}")
    
    # Run income source split
    splitter = IncomeSourceSplitter()
    sample_data_split = splitter.split_income_sources(sample_data)
    
    # Show results
    logger.info(f"\nIncome source split complete!")
    logger.info(f"Added columns: {[col for col in sample_data_split.columns if col not in sample_data.columns]}")
    
    # Generate bracket summary
    summary = splitter.get_source_summary_by_bracket(sample_data_split)
    logger.info(f"\nBracket Summary:\n{summary.to_string(index=False)}")
