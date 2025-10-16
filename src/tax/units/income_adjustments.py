"""
Post-processing module to apply AGI adjustments and calculate taxable income.

This module provides functions to:
1. Load tax units with total income
2. Apply AGI adjustments (IRA, student loans, etc.)
3. Calculate taxable income (after standard deduction)
4. Create versions for different use cases:
   - Original: total_income (for general analysis)
   - AGI version: agi (for tax revenue estimates)
   - Taxable version: taxable_income (for SOI comparison)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Dict, Tuple

from tax.adjustments.agi_adjustments import AGIAdjustmentEstimator
from tax.units.taxable_income import TaxableIncomeCalculator

logger = logging.getLogger(__name__)


def detect_self_employment(row: pd.Series) -> bool:
    """
    Detect if a tax unit has self-employment income.
    
    Args:
        row: Tax unit row
        
    Returns:
        True if has self-employment income
    """
    # Check if we have SE income data
    if 'has_self_employment' in row:
        return bool(row['has_self_employment'])
    
    # Heuristic: assume ~10% of filers with income >$50K are self-employed
    # This is a rough estimate based on IRS SOI data
    if row.get('income', 0) > 50000:
        # Use a deterministic hash to be consistent
        import hashlib
        seed_str = f"{row.get('filer_id', 'unknown')}_{row.get('income', 0)}"
        hash_val = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
        return (hash_val % 100) < 10  # 10% probability
    
    return False


def extract_age(row: pd.Series) -> Optional[int]:
    """
    Extract age from tax unit data.
    
    Args:
        row: Tax unit row
        
    Returns:
        Age of primary filer, or None if not available
    """
    # Try different possible age fields
    if 'age' in row and pd.notna(row['age']):
        return int(row['age'])
    
    if 'primary_filer' in row and isinstance(row['primary_filer'], dict):
        age = row['primary_filer'].get('AGEP')
        if age is not None:
            return int(age)
    
    # Age estimation based on income (rough heuristic)
    income = row.get('income', 0)
    if income < 25000:
        return 25  # Young worker
    elif income < 50000:
        return 35  # Mid-career
    elif income < 100000:
        return 45  # Established career
    else:
        return 50  # Peak earning years
    
    return None


def apply_agi_adjustments(
    tax_units: pd.DataFrame,
    income_col: str = 'income',
    filing_status_col: str = 'filing_status',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Apply AGI adjustments to tax units.
    
    This creates a copy of the tax units with additional columns:
    - agi: Adjusted Gross Income (total income - adjustments)
    - agi_adjustments: Total amount of adjustments
    - agi_adjustment_details: Breakdown of adjustments (dict)
    
    Args:
        tax_units: DataFrame with tax units
        income_col: Column name for total income
        filing_status_col: Column name for filing status
        verbose: Whether to print progress
        
    Returns:
        DataFrame with AGI columns added
    """
    if verbose:
        logger.info(f"Applying AGI adjustments to {len(tax_units):,} tax units")
    
    result_df = tax_units.copy()
    estimator = AGIAdjustmentEstimator()
    
    # Calculate AGI for each tax unit
    agis = []
    adjustments = []
    adjustment_details = []
    
    for idx, row in tax_units.iterrows():
        total_income = row[income_col]
        filing_status = row[filing_status_col]
        age = extract_age(row)
        is_se = detect_self_employment(row)
        
        # Calculate adjustments
        adj_detail = estimator.estimate_total_adjustments(
            total_income, age, filing_status, is_se
        )
        
        agi = total_income - adj_detail['total']
        agi = max(0, agi)  # AGI can't be negative
        
        agis.append(agi)
        adjustments.append(adj_detail['total'])
        adjustment_details.append(adj_detail)
    
    # Add columns
    result_df['agi'] = agis
    result_df['agi_adjustments'] = adjustments
    result_df['agi_adjustment_details'] = adjustment_details
    
    if verbose:
        avg_income = result_df[income_col].mean()
        avg_agi = result_df['agi'].mean()
        avg_adj = result_df['agi_adjustments'].mean()
        
        logger.info(f"Average total income: ${avg_income:,.0f}")
        logger.info(f"Average AGI: ${avg_agi:,.0f}")
        logger.info(f"Average adjustments: ${avg_adj:,.0f} ({(avg_adj/avg_income)*100:.2f}%)")
        logger.info(f"AGI is {(avg_agi/avg_income)*100:.1f}% of total income")
    
    return result_df


def apply_taxable_income_calculation(
    tax_units: pd.DataFrame,
    income_col: str = 'income',
    filing_status_col: str = 'filing_status',
    tax_year: int = 2022,
    apply_agi: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Calculate taxable income for tax units.
    
    This creates a copy with additional columns:
    - agi: Adjusted Gross Income (if apply_agi=True)
    - agi_adjustments: Total AGI adjustments
    - standard_deduction: Standard deduction amount
    - taxable_income: Taxable income (AGI - standard deduction)
    
    Args:
        tax_units: DataFrame with tax units
        income_col: Column name for total income
        filing_status_col: Column name for filing status
        tax_year: Tax year for standard deduction (2022 or 2024)
        apply_agi: Whether to apply AGI adjustments
        verbose: Whether to print progress
        
    Returns:
        DataFrame with taxable income columns added
    """
    if verbose:
        logger.info(f"Calculating taxable income for {len(tax_units):,} tax units")
        logger.info(f"Tax year: {tax_year}, Apply AGI adjustments: {apply_agi}")
    
    # First apply AGI adjustments if requested
    if apply_agi:
        result_df = apply_agi_adjustments(tax_units, income_col, filing_status_col, verbose=False)
        income_for_deduction = 'agi'
    else:
        result_df = tax_units.copy()
        result_df['agi'] = result_df[income_col]
        result_df['agi_adjustments'] = 0
        income_for_deduction = income_col
    
    # Apply standard deduction
    calculator = TaxableIncomeCalculator(tax_year=tax_year)
    
    standard_deductions = []
    taxable_incomes = []
    
    for idx, row in result_df.iterrows():
        filing_status = row[filing_status_col]
        income_after_agi = row[income_for_deduction]
        
        # Get standard deduction
        std_ded = calculator.get_standard_deduction(filing_status)
        standard_deductions.append(std_ded)
        
        # Calculate taxable income
        taxable = income_after_agi - std_ded
        taxable_incomes.append(max(0, taxable))
    
    result_df['standard_deduction'] = standard_deductions
    result_df['taxable_income'] = taxable_incomes
    
    if verbose:
        avg_income = result_df[income_col].mean()
        avg_agi = result_df['agi'].mean()
        avg_std_ded = result_df['standard_deduction'].mean()
        avg_taxable = result_df['taxable_income'].mean()
        
        logger.info(f"Average total income:      ${avg_income:>12,.0f}")
        logger.info(f"Average AGI:               ${avg_agi:>12,.0f} ({(avg_agi/avg_income)*100:.1f}%)")
        logger.info(f"Average standard deduction:${avg_std_ded:>12,.0f}")
        logger.info(f"Average taxable income:    ${avg_taxable:>12,.0f} ({(avg_taxable/avg_income)*100:.1f}%)")
        logger.info(f"Total reduction:           ${avg_income - avg_taxable:>12,.0f} ({((avg_income-avg_taxable)/avg_income)*100:.1f}%)")
    
    return result_df


def create_income_versions(
    tax_units: pd.DataFrame,
    tax_year: int = 2022,
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Create different versions of tax units for different use cases.
    
    Returns three versions:
    1. 'original': Total income (for general analysis)
    2. 'agi': With AGI calculated (for tax revenue estimates)
    3. 'taxable': With taxable income (for SOI comparison)
    
    Args:
        tax_units: Original tax units DataFrame
        tax_year: Tax year for calculations
        verbose: Whether to print progress
        
    Returns:
        Dictionary with three DataFrames:
        - 'original': Original data
        - 'agi': With AGI adjustments
        - 'taxable': With taxable income
    """
    if verbose:
        print("="*80)
        print("CREATING INCOME VERSIONS FOR TAX UNITS")
        print("="*80)
        print(f"\nOriginal tax units: {len(tax_units):,}")
        print(f"Tax year: {tax_year}\n")
    
    versions = {}
    
    # Version 1: Original (no changes)
    versions['original'] = tax_units.copy()
    if verbose:
        print("✓ Created 'original' version (total income)")
    
    # Version 2: AGI (for tax revenue estimates)
    versions['agi'] = apply_agi_adjustments(tax_units, verbose=verbose)
    if verbose:
        print("✓ Created 'agi' version (with AGI adjustments)")
    
    # Version 3: Taxable (for SOI comparison)
    versions['taxable'] = apply_taxable_income_calculation(
        tax_units, tax_year=tax_year, apply_agi=True, verbose=verbose
    )
    if verbose:
        print("✓ Created 'taxable' version (with standard deduction)")
    
    if verbose:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        for name, df in versions.items():
            income_col = 'income' if name == 'original' else ('agi' if name == 'agi' else 'taxable_income')
            avg_income = df[income_col].mean()
            total_income = (df[income_col] * df['weight']).sum()
            
            print(f"\n{name.upper()}:")
            print(f"  Income column: {income_col}")
            print(f"  Average income: ${avg_income:,.0f}")
            print(f"  Total weighted income: ${total_income:,.0f}")
    
    return versions


def load_and_prepare_tax_units(
    file_path: Optional[Path] = None,
    use_version: str = 'agi',
    tax_year: int = 2022,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load tax units and prepare with income adjustments.
    
    Args:
        file_path: Path to tax units parquet file (if None, finds latest)
        use_version: Which version to return ('original', 'agi', or 'taxable')
        tax_year: Tax year for calculations
        verbose: Whether to print progress
        
    Returns:
        DataFrame with requested income version
    """
    # Find latest tax units file if not specified
    if file_path is None:
        from pathlib import Path
        data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'processed'
        files = sorted(data_dir.glob('tax_units_regenerated_*.parquet'))
        if not files:
            raise FileNotFoundError("No tax units files found")
        file_path = files[-1]
    
    if verbose:
        print(f"Loading tax units from: {file_path.name}")
    
    # Load original tax units
    tax_units = pd.read_parquet(file_path)
    
    if verbose:
        print(f"Loaded {len(tax_units):,} tax units")
        print(f"Total weighted: {tax_units['weight'].sum():,.0f}\n")
    
    # Create versions
    versions = create_income_versions(tax_units, tax_year=tax_year, verbose=verbose)
    
    # Return requested version
    if use_version not in versions:
        raise ValueError(f"Unknown version '{use_version}'. Choose from: {list(versions.keys())}")
    
    return versions[use_version]


if __name__ == '__main__':
    # Example usage
    print("\n" + "="*80)
    print("INCOME ADJUSTMENTS MODULE - EXAMPLE USAGE")
    print("="*80)
    
    # Load and create all versions
    versions = create_income_versions(
        pd.read_parquet('data/processed/tax_units_regenerated_20251015_085131.parquet'),
        tax_year=2022,
        verbose=True
    )
    
    print("\n" + "="*80)
    print("USAGE EXAMPLES")
    print("="*80)
    
    print("\n# For general analysis (use total income):")
    print("tax_units = load_and_prepare_tax_units(use_version='original')")
    
    print("\n# For tax revenue estimates (use AGI):")
    print("tax_units = load_and_prepare_tax_units(use_version='agi')")
    print("# Then use 'agi' column for tax calculations")
    
    print("\n# For SOI comparison (use taxable income):")
    print("tax_units = load_and_prepare_tax_units(use_version='taxable')")
    print("# Then use 'taxable_income' column for bracket comparison")
    
    print("\n" + "="*80)
