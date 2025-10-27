#!/usr/bin/env python3
"""
Compare 2026 Hawaii tax revenue under 2017 vs 2024 tax policy.

This script projects 2026 income using the ensemble model, then calculates
tax liability under two scenarios:
1. 2017 tax brackets and standard deductions (pre-TCJA era)
2. 2024 tax brackets and standard deductions (post-TCJA era)

Shows the revenue impact of Hawaii's tax policy changes.
"""

from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.projection import EnsembleProjector
from src.tax.brackets import load_tax_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_projected_2026_incomes() -> pd.DataFrame:
    """Load 2026 projected tax units from ensemble model."""
    
    data_dir = Path('data/processed/projections')
    ensemble_file = data_dir / 'tax_units_2026_ensemble.parquet'
    
    if not ensemble_file.exists():
        raise FileNotFoundError(
            f"2026 ensemble projections not found at {ensemble_file}. "
            "Run scripts/projection/project_to_2026.py first."
        )
    
    logger.info(f"Loading 2026 projected incomes from {ensemble_file}")
    tax_units = pd.read_parquet(ensemble_file)
    
    logger.info(f"Loaded {len(tax_units):,} tax units")
    logger.info(f"Total projected income: ${tax_units['income'].sum():,.0f}")
    
    return tax_units


def calculate_taxes_with_policy(
    tax_units: pd.DataFrame,
    year: int,
    policy_name: str,
    tax_calculator
) -> pd.DataFrame:
    """Calculate taxes using specific year's policy."""
    
    logger.info(f"Calculating taxes with {policy_name} policy (year {year})...")
    
    # Prepare data
    units = tax_units.copy()
    
    # Map filing status to Hawaii format
    filing_status_map = {
        'single': 'Single_Married_Separate',
        'married_filing_jointly': 'Joint_Surviving_Spouse',
        'married_filing_separate': 'Single_Married_Separate',
        'head_of_household': 'Head_of_Household',
        'qualifying_widow': 'Joint_Surviving_Spouse'
    }
    
    units['hi_filing_status'] = units['filing_status'].map(filing_status_map)
    
    # Handle missing filing statuses
    missing_status = units['hi_filing_status'].isna().sum()
    if missing_status > 0:
        logger.warning(f"Found {missing_status} units with missing filing status")
        units['hi_filing_status'] = units['hi_filing_status'].fillna('Single_Married_Separate')
    
    # Calculate taxes using specified year's brackets/deductions
    results = tax_calculator.calculate_tax_for_dataframe(
        units,
        year=year,
        income_col='income',
        filing_status_col='hi_filing_status'
    )
    
    # Add policy identifier
    results[f'{policy_name}_tax_liability'] = results['hi_tax_tax_liability']
    results[f'{policy_name}_effective_rate'] = results['hi_tax_effective_rate']
    results[f'{policy_name}_standard_deduction'] = results['hi_tax_standard_deduction']
    results[f'{policy_name}_taxable_income'] = results['hi_tax_taxable_income']
    
    logger.info(f"{policy_name} policy results:")
    logger.info(f"  Total tax liability: ${results[f'{policy_name}_tax_liability'].sum():,.0f}")
    logger.info(f"  Average effective rate: {results[f'{policy_name}_effective_rate'].mean() * 100:.2f}%")
    logger.info(f"  Median tax liability: ${results[f'{policy_name}_tax_liability'].median():,.0f}")
    
    return results


def analyze_policy_differences(results_2017: pd.DataFrame, results_2024: pd.DataFrame) -> pd.DataFrame:
    """Analyze differences between 2017 and 2024 tax policies."""
    
    logger.info("Analyzing policy differences...")
    
    # Combine results
    comparison = pd.DataFrame({
        'metric': [
            'Total Tax Revenue',
            'Average Tax per Filer',
            'Median Tax per Filer',
            'Average Effective Rate',
            'Average Standard Deduction',
            'Average Taxable Income',
            'Total Taxable Income'
        ],
        '2017_policy': [
            results_2017['policy_2017_tax_liability'].sum(),
            results_2017['policy_2017_tax_liability'].mean(),
            results_2017['policy_2017_tax_liability'].median(),
            results_2017['policy_2017_effective_rate'].mean() * 100,
            results_2017['policy_2017_standard_deduction'].mean(),
            results_2017['policy_2017_taxable_income'].mean(),
            results_2017['policy_2017_taxable_income'].sum()
        ],
        '2024_policy': [
            results_2024['policy_2024_tax_liability'].sum(),
            results_2024['policy_2024_tax_liability'].mean(),
            results_2024['policy_2024_tax_liability'].median(),
            results_2024['policy_2024_effective_rate'].mean() * 100,
            results_2024['policy_2024_standard_deduction'].mean(),
            results_2024['policy_2024_taxable_income'].mean(),
            results_2024['policy_2024_taxable_income'].sum()
        ]
    })
    
    # Calculate differences
    comparison['difference'] = comparison['2024_policy'] - comparison['2017_policy']
    comparison['pct_change'] = (comparison['2024_policy'] / comparison['2017_policy'] - 1) * 100
    
    return comparison


def analyze_by_income_group(results_2017: pd.DataFrame, results_2024: pd.DataFrame) -> pd.DataFrame:
    """Analyze policy impact by income group."""
    
    logger.info("Analyzing by income groups...")
    
    # Define income groups
    def get_income_group(income):
        if income < 25000:
            return 'Under $25K'
        elif income < 50000:
            return '$25K-$50K'
        elif income < 100000:
            return '$50K-$100K'
        elif income < 200000:
            return '$100K-$200K'
        else:
            return 'Over $200K'
    
    # Add income groups
    results_2017['income_group'] = results_2017['income'].apply(get_income_group)
    results_2024['income_group'] = results_2024['income'].apply(get_income_group)
    
    # Group analysis
    groups_2017 = results_2017.groupby('income_group').agg({
        'policy_2017_tax_liability': ['count', 'sum', 'mean'],
        'policy_2017_effective_rate': 'mean',
        'income': 'mean'
    }).round(2)
    
    groups_2024 = results_2024.groupby('income_group').agg({
        'policy_2024_tax_liability': ['count', 'sum', 'mean'],
        'policy_2024_effective_rate': 'mean',
        'income': 'mean'
    }).round(2)
    
    # Flatten column names
    groups_2017.columns = ['count', 'total_tax_2017', 'avg_tax_2017', 'avg_rate_2017', 'avg_income']
    groups_2024.columns = ['count', 'total_tax_2024', 'avg_tax_2024', 'avg_rate_2024', 'avg_income']
    
    # Combine
    income_analysis = pd.concat([groups_2017, groups_2024[['total_tax_2024', 'avg_tax_2024', 'avg_rate_2024']]], axis=1)
    
    # Calculate differences
    income_analysis['tax_difference'] = income_analysis['total_tax_2024'] - income_analysis['total_tax_2017']
    income_analysis['tax_pct_change'] = (income_analysis['total_tax_2024'] / income_analysis['total_tax_2017'] - 1) * 100
    income_analysis['rate_difference'] = income_analysis['avg_rate_2024'] - income_analysis['avg_rate_2017']
    
    return income_analysis


def analyze_by_filing_status(results_2017: pd.DataFrame, results_2024: pd.DataFrame) -> pd.DataFrame:
    """Analyze policy impact by filing status."""
    
    logger.info("Analyzing by filing status...")
    
    # Group by filing status
    status_2017 = results_2017.groupby('filing_status').agg({
        'policy_2017_tax_liability': ['count', 'sum', 'mean'],
        'policy_2017_effective_rate': 'mean',
        'policy_2017_standard_deduction': 'mean'
    }).round(2)
    
    status_2024 = results_2024.groupby('filing_status').agg({
        'policy_2024_tax_liability': ['count', 'sum', 'mean'],
        'policy_2024_effective_rate': 'mean',
        'policy_2024_standard_deduction': 'mean'
    }).round(2)
    
    # Flatten column names
    status_2017.columns = ['count', 'total_tax_2017', 'avg_tax_2017', 'avg_rate_2017', 'std_deduction_2017']
    status_2024.columns = ['count', 'total_tax_2024', 'avg_tax_2024', 'avg_rate_2024', 'std_deduction_2024']
    
    # Combine
    status_analysis = pd.concat([
        status_2017, 
        status_2024[['total_tax_2024', 'avg_tax_2024', 'avg_rate_2024', 'std_deduction_2024']]
    ], axis=1)
    
    # Calculate differences
    status_analysis['tax_difference'] = status_analysis['total_tax_2024'] - status_analysis['total_tax_2017']
    status_analysis['tax_pct_change'] = (status_analysis['total_tax_2024'] / status_analysis['total_tax_2017'] - 1) * 100
    status_analysis['deduction_increase'] = status_analysis['std_deduction_2024'] - status_analysis['std_deduction_2017']
    
    return status_analysis


def main():
    """Main analysis workflow."""
    
    logger.info("=" * 80)
    logger.info("HAWAII TAX POLICY COMPARISON: 2017 vs 2024 BRACKETS/DEDUCTIONS")
    logger.info("=" * 80)
    
    # Load 2026 projected incomes
    tax_units_2026 = load_projected_2026_incomes()
    
    # Initialize tax calculator
    tax_calculator = load_tax_data()
    logger.info("Initialized Hawaii tax calculator")
    
    # Calculate taxes under 2017 policy
    logger.info("\n" + "=" * 80)
    logger.info("SCENARIO 1: 2017 TAX POLICY (PRE-TCJA ERA)")
    logger.info("=" * 80)
    
    results_2017 = calculate_taxes_with_policy(
        tax_units_2026, 
        year=2017, 
        policy_name='policy_2017',
        tax_calculator=tax_calculator
    )
    
    # Calculate taxes under 2024 policy
    logger.info("\n" + "=" * 80)
    logger.info("SCENARIO 2: 2024 TAX POLICY (POST-TCJA ERA)")
    logger.info("=" * 80)
    
    results_2024 = calculate_taxes_with_policy(
        tax_units_2026, 
        year=2024, 
        policy_name='policy_2024',
        tax_calculator=tax_calculator
    )
    
    # Overall comparison
    logger.info("\n" + "=" * 80)
    logger.info("POLICY COMPARISON ANALYSIS")
    logger.info("=" * 80)
    
    overall_comparison = analyze_policy_differences(results_2017, results_2024)
    print("\nOVERALL COMPARISON:")
    print(overall_comparison.to_string(index=False))
    
    # Income group analysis
    income_analysis = analyze_by_income_group(results_2017, results_2024)
    print("\nBY INCOME GROUP:")
    print(income_analysis.to_string())
    
    # Filing status analysis
    status_analysis = analyze_by_filing_status(results_2017, results_2024)
    print("\nBY FILING STATUS:")
    print(status_analysis.to_string())
    
    # Save results
    logger.info("\n" + "=" * 80)
    logger.info("SAVING RESULTS")
    logger.info("=" * 80)
    
    output_dir = Path('data/processed/policy_analysis')
    output_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    results_2017.to_parquet(output_dir / 'tax_units_2026_with_2017_policy.parquet')
    results_2024.to_parquet(output_dir / 'tax_units_2026_with_2024_policy.parquet')
    
    # Save comparisons
    overall_comparison.to_csv(output_dir / 'policy_comparison_2017_vs_2024.csv', index=False)
    income_analysis.to_csv(output_dir / 'policy_comparison_by_income_group.csv')
    status_analysis.to_csv(output_dir / 'policy_comparison_by_filing_status.csv')
    
    logger.info(f"Results saved to {output_dir}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    
    revenue_2017 = results_2017['policy_2017_tax_liability'].sum()
    revenue_2024 = results_2024['policy_2024_tax_liability'].sum()
    revenue_difference = revenue_2024 - revenue_2017
    pct_change = (revenue_2024 / revenue_2017 - 1) * 100
    
    logger.info(f"\n2026 Tax Revenue Projections (Sample):")
    logger.info(f"  With 2017 policy: ${revenue_2017:,.0f}")
    logger.info(f"  With 2024 policy: ${revenue_2024:,.0f}")
    logger.info(f"  Difference: ${revenue_difference:,.0f} ({pct_change:+.1f}%)")
    
    # Scale to full population
    scaling_factor = 700000 / len(tax_units_2026)  # Estimated Hawaii filers / sample size
    
    scaled_2017 = revenue_2017 * scaling_factor / 1_000_000  # Convert to millions
    scaled_2024 = revenue_2024 * scaling_factor / 1_000_000
    scaled_diff = scaled_2024 - scaled_2017
    
    logger.info(f"\nScaled to Full Hawaii Population:")
    logger.info(f"  With 2017 policy: ${scaled_2017:.0f}M")
    logger.info(f"  With 2024 policy: ${scaled_2024:.0f}M")
    logger.info(f"  Revenue impact: ${scaled_diff:+.0f}M ({pct_change:+.1f}%)")
    
    if pct_change < 0:
        logger.info(f"\n💰 Tax cuts reduced revenue by ${abs(scaled_diff):.0f}M")
    else:
        logger.info(f"\n📈 Tax increases raised revenue by ${scaled_diff:.0f}M")


if __name__ == '__main__':
    main()
