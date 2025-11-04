#!/usr/bin/env python3
"""
Project Hawaii tax revenue to 2026 using ensemble growth model.

This script:
1. Loads PUMS 2023 tax units (base year)
2. Projects income forward to 2026 using ensemble model (ACS + BLS)
3. Applies 2026 Hawaii tax brackets and standard deductions
4. Calculates projected 2026 tax revenue
5. Compares to baseline (simple inflation) approach

Usage:
    python scripts/projection/project_to_2026.py
    python scripts/projection/project_to_2026.py --acs-weight 0.5 --bls-weight 0.5
"""

from __future__ import annotations

import argparse
import json
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


def load_pums_persons(pums_dir: Path) -> pd.DataFrame:
    """Load PUMS person data with occupation codes."""
    
    person_file = pums_dir / 'psam_p15.csv'
    
    logger.info(f"Loading PUMS person data from {person_file}")
    
    # Load only needed columns for efficiency
    cols_needed = ['SERIALNO', 'SPORDER', 'SOCP', 'WAGP', 'PINCP', 'AGEP', 'PWGTP']
    
    persons = pd.read_csv(person_file, usecols=cols_needed)
    
    logger.info(f"Loaded {len(persons):,} persons")
    logger.info(f"  Workers with SOCP: {persons['SOCP'].notna().sum():,}")
    
    return persons


def load_tax_units(tax_units_file: Path) -> pd.DataFrame:
    """Load tax units from processed file."""
    
    logger.info(f"Loading tax units from {tax_units_file}")
    
    tax_units = pd.read_parquet(tax_units_file)
    
    logger.info(f"Loaded {len(tax_units):,} tax units")
    logger.info(f"  Filing statuses: {tax_units['filing_status'].value_counts().to_dict()}")
    logger.info(f"  Total income: ${tax_units['income'].sum():,.0f}")
    
    return tax_units


def calculate_2026_taxes(
    tax_units: pd.DataFrame,
    tax_calculator
) -> pd.DataFrame:
    """Calculate 2026 Hawaii state income taxes."""
    
    logger.info("Calculating 2026 Hawaii state income taxes...")
    
    # Prepare for tax calculation
    tax_units = tax_units.copy()
    
    # Map filing status to Hawaii format
    filing_status_map = {
        'single': 'single',
        'married_filing_jointly': 'married_filing_jointly',
        'married_filing_separate': 'married_filing_separately',
        'head_of_household': 'head_of_household',
        'qualifying_widow': 'married_filing_jointly'
    }
    
    tax_units['hi_filing_status'] = tax_units['filing_status'].map(filing_status_map)
    
    # Check for missing filing statuses
    missing_status = tax_units['hi_filing_status'].isna().sum()
    if missing_status > 0:
        logger.warning(f"Found {missing_status} tax units with missing filing status")
        logger.warning(f"Original statuses: {tax_units[tax_units['hi_filing_status'].isna()]['filing_status'].value_counts().to_dict()}")
        # Fill missing with 'single' as default
        tax_units['hi_filing_status'] = tax_units['hi_filing_status'].fillna('single')
    
    # Calculate taxes using 2026 brackets
    results = tax_calculator.calculate_tax_for_dataframe(
        tax_units,
        year=2026,
        income_col='income',
        filing_status_col='hi_filing_status'
    )
    
    logger.info("Tax calculation complete:")
    logger.info(f"  Total tax liability: ${results['hi_tax_tax_liability'].sum():,.0f}")
    logger.info(f"  Average effective rate: {results['hi_tax_effective_rate'].mean() * 100:.2f}%")
    logger.info(f"  Median tax liability: ${results['hi_tax_tax_liability'].median():,.0f}")
    
    return results


def calculate_baseline_projection(
    tax_units: pd.DataFrame,
    target_year: int,
    base_year: int = 2023,
    inflation_rate: float = 0.025
) -> pd.DataFrame:
    """Calculate baseline projection using simple inflation."""
    
    years_forward = target_year - base_year
    inflation_factor = (1 + inflation_rate) ** years_forward
    
    baseline = tax_units.copy()
    baseline['income_original'] = baseline['income']
    baseline['income'] = baseline['income'] * inflation_factor
    baseline['projection_method'] = 'baseline_inflation'
    baseline['growth_factor'] = inflation_factor
    
    logger.info(f"Baseline projection: {inflation_rate:.1%} annual inflation")
    logger.info(f"  Growth factor: {inflation_factor:.3f} over {years_forward} years")
    
    return baseline


def compare_projections(
    ensemble_results: pd.DataFrame,
    baseline_results: pd.DataFrame
) -> pd.DataFrame:
    """Compare ensemble vs baseline projection results."""
    
    comparison = pd.DataFrame({
        'metric': [
            'Total Income',
            'Avg Income',
            'Median Income',
            'Total Tax Liability',
            'Avg Effective Rate',
            'Total Revenue'
        ],
        'baseline': [
            baseline_results['income'].sum(),
            baseline_results['income'].mean(),
            baseline_results['income'].median(),
            baseline_results['hi_tax_tax_liability'].sum(),
            baseline_results['hi_tax_effective_rate'].mean(),
            baseline_results['hi_tax_tax_liability'].sum()
        ],
        'ensemble': [
            ensemble_results['income'].sum(),
            ensemble_results['income'].mean(),
            ensemble_results['income'].median(),
            ensemble_results['hi_tax_tax_liability'].sum(),
            ensemble_results['hi_tax_effective_rate'].mean(),
            ensemble_results['hi_tax_tax_liability'].sum()
        ]
    })
    
    comparison['difference'] = comparison['ensemble'] - comparison['baseline']
    comparison['pct_diff'] = (comparison['ensemble'] / comparison['baseline'] - 1) * 100
    
    return comparison


def main(args: argparse.Namespace) -> None:
    """Main projection workflow."""
    
    logger.info("=" * 80)
    logger.info("HAWAII TAX REVENUE PROJECTION TO 2026")
    logger.info("=" * 80)
    
    # Define paths
    data_dir = Path('data')
    processed_dir = data_dir / 'processed'
    pums_dir = data_dir / 'raw' / 'pums'
    acs_dir = processed_dir / 'acs_timeseries'
    
    # Load data
    logger.info("\n" + "=" * 80)
    logger.info("LOADING DATA")
    logger.info("=" * 80)
    
    tax_units = load_tax_units(processed_dir / 'tax_units_original.parquet')
    pums_persons = load_pums_persons(pums_dir)
    
    # Load BLS OES data
    bls_summary = pd.read_parquet(processed_dir / 'bls_oes_occupation_summary.parquet')
    logger.info(f"Loaded BLS OES data: {len(bls_summary)} occupations")
    
    # Initialize tax calculator
    tax_calculator = load_tax_data()
    logger.info("Initialized Hawaii tax calculator")
    
    # Note: tax_units_original.parquet is already calibrated to $2,814M (7.1% below target)
    # This is the best achievable with current SOI income calibration
    
    # Initialize ensemble projector
    logger.info("\n" + "=" * 80)
    logger.info("INITIALIZING ENSEMBLE PROJECTOR")
    logger.info("=" * 80)
    
    projector = EnsembleProjector(
        acs_timeseries_dir=acs_dir,
        bls_oes_data=bls_summary,
        base_year=args.base_year,
        acs_weight=args.acs_weight,
        bls_weight=args.bls_weight
    )
    
    # Project income to 2026
    logger.info("\n" + "=" * 80)
    logger.info("ENSEMBLE PROJECTION TO 2026")
    logger.info("=" * 80)
    
    ensemble_projected = projector.project_income(
        tax_units=tax_units,
        pums_persons=pums_persons,
        target_year=2026
    )
    
    # Calculate 2026 taxes (ensemble)
    logger.info("\n" + "=" * 80)
    logger.info("CALCULATING 2026 TAXES (ENSEMBLE)")
    logger.info("=" * 80)
    
    ensemble_results = calculate_2026_taxes(ensemble_projected, tax_calculator)
    
    # Baseline projection for comparison
    logger.info("\n" + "=" * 80)
    logger.info("BASELINE PROJECTION (SIMPLE INFLATION)")
    logger.info("=" * 80)
    
    baseline_projected = calculate_baseline_projection(
        tax_units,
        target_year=2026,
        base_year=args.base_year,
        inflation_rate=args.inflation_rate
    )
    
    baseline_results = calculate_2026_taxes(baseline_projected, tax_calculator)
    
    # Compare results
    logger.info("\n" + "=" * 80)
    logger.info("PROJECTION COMPARISON")
    logger.info("=" * 80)
    
    comparison = compare_projections(ensemble_results, baseline_results)
    
    print("\n" + comparison.to_string(index=False))
    
    # Get projection summary
    summary = projector.get_projection_summary(ensemble_results)
    
    # Save results
    logger.info("\n" + "=" * 80)
    logger.info("SAVING RESULTS")
    logger.info("=" * 80)
    
    output_dir = processed_dir / 'projections'
    output_dir.mkdir(exist_ok=True)
    
    # Save projected tax units
    ensemble_results.to_parquet(output_dir / 'tax_units_2026_ensemble.parquet')
    baseline_results.to_parquet(output_dir / 'tax_units_2026_baseline.parquet')
    logger.info(f"Saved projected tax units to {output_dir}")
    
    # Save comparison
    comparison.to_csv(output_dir / 'projection_comparison_2026.csv', index=False)
    logger.info(f"Saved comparison to {output_dir / 'projection_comparison_2026.csv'}")
    
    # Save summary
    with open(output_dir / 'projection_summary_2026.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Saved summary to {output_dir / 'projection_summary_2026.json'}")
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("PROJECTION COMPLETE")
    logger.info("=" * 80)
    
    ensemble_revenue = ensemble_results['hi_tax_tax_liability'].sum()
    baseline_revenue = baseline_results['hi_tax_tax_liability'].sum()
    
    logger.info(f"\n2026 Projected Tax Revenue:")
    logger.info(f"  Ensemble model: ${ensemble_revenue:,.0f}")
    logger.info(f"  Baseline (inflation): ${baseline_revenue:,.0f}")
    logger.info(f"  Difference: ${ensemble_revenue - baseline_revenue:,.0f} ({(ensemble_revenue/baseline_revenue - 1) * 100:+.1f}%)")
    
    logger.info(f"\nEnsemble Model Match Quality:")
    if 'match_quality' in summary:
        for level, stats in summary['match_quality'].items():
            logger.info(f"  {level}: {stats['count']:,} ({stats['percent']:.1f}%)")
    
    logger.info(f"\nAverage confidence score: {summary.get('avg_confidence', 0):.2f}")
    logger.info(f"Average annual growth: {summary.get('avg_annual_growth', 0) * 100:.2f}%")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Project Hawaii tax revenue to 2026 using ensemble model"
    )
    parser.add_argument(
        '--base-year',
        type=int,
        default=2023,
        help='Base year for PUMS data (default: 2023)'
    )
    parser.add_argument(
        '--acs-weight',
        type=float,
        default=0.4,
        help='Weight for ACS aggregate growth (default: 0.4)'
    )
    parser.add_argument(
        '--bls-weight',
        type=float,
        default=0.6,
        help='Weight for BLS occupation-specific growth (default: 0.6)'
    )
    parser.add_argument(
        '--inflation-rate',
        type=float,
        default=0.025,
        help='Annual inflation rate for baseline comparison (default: 0.025)'
    )
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
