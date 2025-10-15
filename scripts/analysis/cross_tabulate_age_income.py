#!/usr/bin/env python3
"""
Cross-Tabulate Tax Units by Age Group and Income Bracket

This script analyzes the 2022 PUMS tax units to show:
1. Distribution of filers by age group and income bracket
2. How age-specific population growth affects each income bracket
3. Where the 65+ growth actually appears in the income distribution

Usage:
    python scripts/analysis/cross_tabulate_age_income.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Age-specific population growth (2022 → 2024)
AGE_SPECIFIC_POPULATION_GROWTH = {
    'Under 18': -0.0187,
    '18-24': -0.0165,
    '25-34': -0.0165,
    '35-44': -0.0165,
    '45-54': -0.0165,
    '55-64': -0.0165,
    '65+': 0.1029
}


def get_age_group(age: int) -> str:
    """Map age to age group."""
    if age < 18:
        return 'Under 18'
    elif age < 25:
        return '18-24'
    elif age < 35:
        return '25-34'
    elif age < 45:
        return '35-44'
    elif age < 55:
        return '45-54'
    elif age < 65:
        return '55-64'
    else:
        return '65+'


def get_income_bracket(income: float) -> str:
    """Map income to bracket."""
    if income < 25000:
        return '0-25k'
    elif income < 50000:
        return '25-50k'
    elif income < 75000:
        return '50-75k'
    elif income < 100000:
        return '75-100k'
    elif income < 200000:
        return '100-200k'
    else:
        return '200k+'


def load_tax_units_with_age():
    """Load tax units and merge with age data."""
    logger.info("Loading tax units...")
    
    # Find most recent tax units file
    processed_dir = project_root / 'data' / 'processed'
    tax_unit_files = list(processed_dir.glob('hawaii_ctc_full_population_*.parquet'))
    
    if not tax_unit_files:
        raise FileNotFoundError("No tax units file found in data/processed/")
    
    latest_file = max(tax_unit_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"  Using: {latest_file.name}")
    
    tax_units = pd.read_parquet(latest_file)
    logger.info(f"  Loaded {len(tax_units):,} tax units")
    
    # Load person-level data to get primary filer age
    logger.info("  Loading age data from PUMS...")
    pums_file = project_root / 'data' / 'raw' / 'pums' / 'psam_p15.csv'
    persons = pd.read_csv(pums_file, usecols=['SERIALNO', 'SPORDER', 'AGEP'])
    
    # Get primary filer (SPORDER == 1) age for each household
    primary_filers = persons[persons['SPORDER'] == 1][['SERIALNO', 'AGEP']].copy()
    primary_filers.columns = ['hh_id', 'primary_age']
    
    # Convert hh_id to string to match tax units
    primary_filers['hh_id'] = primary_filers['hh_id'].astype(str)
    
    # Merge age data with tax units
    tax_units = tax_units.merge(primary_filers, on='hh_id', how='left')
    
    # Fill missing ages with median
    missing_age = tax_units['primary_age'].isna().sum()
    if missing_age > 0:
        median_age = tax_units['primary_age'].median()
        logger.warning(f"  ⚠️  {missing_age} tax units missing age data, using median: {median_age:.0f}")
        tax_units['primary_age'] = tax_units['primary_age'].fillna(median_age)
    
    return tax_units

def create_cross_tabulation(tax_units):
    """Create cross-tabulation of age groups and income brackets."""
    logger.info("\nCreating cross-tabulation...")
    
    # Determine weight column
    # Note: PWGTP gives 527k filers (83% of SOI target of 635k)
    # This gap is due to tax unit construction coverage, not weight column choice
    weight_col = 'calibrated_weight' if 'calibrated_weight' in tax_units.columns else 'PWGTP'
    
    # Determine income column
    if 'total_income' in tax_units.columns:
        income_col = 'total_income'
    elif 'income' in tax_units.columns:
        income_col = 'income'
    else:
        raise ValueError("No income column found in tax units")
    
    logger.info(f"  Using weight column: {weight_col}")
    logger.info(f"  Using income column: {income_col}")
    
    # Add age group and income bracket
    tax_units['age_group'] = tax_units['primary_age'].apply(get_age_group)
    tax_units['income_bracket'] = tax_units[income_col].apply(get_income_bracket)
    
    # Create cross-tabulation for 2022
    logger.info("\n" + "="*80)
    logger.info("2022 BASELINE: Tax Units by Age Group and Income Bracket")
    logger.info("="*80)
    
    age_groups = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    income_brackets = ['0-25k', '25-50k', '50-75k', '75-100k', '100-200k', '200k+']
    
    # Create pivot table
    pivot_2022 = tax_units.pivot_table(
        values=weight_col,
        index='income_bracket',
        columns='age_group',
        aggfunc='sum',
        fill_value=0
    )
    
    # Reorder columns and rows
    pivot_2022 = pivot_2022.reindex(columns=age_groups, fill_value=0)
    pivot_2022 = pivot_2022.reindex(income_brackets, fill_value=0)
    
    # Add totals
    pivot_2022['TOTAL'] = pivot_2022.sum(axis=1)
    pivot_2022.loc['TOTAL'] = pivot_2022.sum(axis=0)
    
    # Print table
    print("\n" + pivot_2022.to_string())
    
    # Apply population growth
    logger.info("\n" + "="*80)
    logger.info("2024 PROJECTION: Tax Units by Age Group and Income Bracket")
    logger.info("="*80)
    
    # Create copy for 2024
    tax_units_2024 = tax_units.copy()
    tax_units_2024['population_growth_rate'] = tax_units_2024['age_group'].map(AGE_SPECIFIC_POPULATION_GROWTH)
    tax_units_2024[weight_col] = tax_units_2024[weight_col] * (1 + tax_units_2024['population_growth_rate'])
    
    # Create pivot table for 2024
    pivot_2024 = tax_units_2024.pivot_table(
        values=weight_col,
        index='income_bracket',
        columns='age_group',
        aggfunc='sum',
        fill_value=0
    )
    
    # Reorder columns and rows
    pivot_2024 = pivot_2024.reindex(columns=age_groups, fill_value=0)
    pivot_2024 = pivot_2024.reindex(income_brackets, fill_value=0)
    
    # Add totals
    pivot_2024['TOTAL'] = pivot_2024.sum(axis=1)
    pivot_2024.loc['TOTAL'] = pivot_2024.sum(axis=0)
    
    # Print table
    print("\n" + pivot_2024.to_string())
    
    # Calculate changes
    logger.info("\n" + "="*80)
    logger.info("CHANGE (2022 → 2024): Absolute Change in Filers")
    logger.info("="*80)
    
    pivot_change = pivot_2024 - pivot_2022
    print("\n" + pivot_change.to_string())
    
    # Calculate percentage changes
    logger.info("\n" + "="*80)
    logger.info("CHANGE (2022 → 2024): Percentage Change")
    logger.info("="*80)
    
    pivot_pct_change = ((pivot_2024 - pivot_2022) / pivot_2022 * 100).round(1)
    pivot_pct_change = pivot_pct_change.replace([np.inf, -np.inf], 0)
    print("\n" + pivot_pct_change.applymap(lambda x: f"{x:+.1f}%").to_string())
    
    # Summary by income bracket
    logger.info("\n" + "="*80)
    logger.info("SUMMARY: Income Bracket Changes")
    logger.info("="*80)
    
    summary = pd.DataFrame({
        '2022 Filers': pivot_2022['TOTAL'][:-1],
        '2024 Filers': pivot_2024['TOTAL'][:-1],
        'Change': pivot_change['TOTAL'][:-1],
        'Pct Change': pivot_pct_change['TOTAL'][:-1]
    })
    
    print("\n" + summary.to_string())
    
    # Summary by age group
    logger.info("\n" + "="*80)
    logger.info("SUMMARY: Age Group Changes")
    logger.info("="*80)
    
    summary_age = pd.DataFrame({
        '2022 Filers': pivot_2022.loc['TOTAL'][:-1],
        '2024 Filers': pivot_2024.loc['TOTAL'][:-1],
        'Change': pivot_change.loc['TOTAL'][:-1],
        'Pct Change': pivot_pct_change.loc['TOTAL'][:-1]
    })
    
    print("\n" + summary_age.to_string())
    
    # Save to CSV
    output_dir = project_root / 'analysis_results' / 'calibration'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pivot_2022.to_csv(output_dir / 'age_income_crosstab_2022.csv')
    pivot_2024.to_csv(output_dir / 'age_income_crosstab_2024.csv')
    pivot_change.to_csv(output_dir / 'age_income_crosstab_change.csv')
    
    logger.info(f"\n✅ Cross-tabulation saved to: {output_dir}/")
    
    return pivot_2022, pivot_2024, pivot_change


def main():
    """Main execution."""
    logger.info("="*80)
    logger.info("CROSS-TABULATION: Age Group × Income Bracket")
    logger.info("="*80)
    
    # Load data
    tax_units = load_tax_units_with_age()
    
    # Create cross-tabulation
    pivot_2022, pivot_2024, pivot_change = create_cross_tabulation(tax_units)
    
    logger.info("\n" + "="*80)
    logger.info("✅ Analysis Complete")
    logger.info("="*80)


if __name__ == '__main__':
    main()
