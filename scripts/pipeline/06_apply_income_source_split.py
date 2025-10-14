#!/usr/bin/env python3
"""
Apply Income Source Split to Tax Units

This script implements Stage 5 of the calibration process:
- Loads tax units from Stage 4 (high-income enhancement)
- Splits total income into component sources using IRS SOI data
- Validates income consistency
- Generates detailed source breakdown reports

Usage:
    python scripts/pipeline/06_apply_income_source_split.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
import logging
from datetime import datetime

from src.tax.calibration.income_source_split import IncomeSourceSplitter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_tax_units(input_path: str) -> pd.DataFrame:
    """Load tax units from previous calibration stage."""
    logger.info(f"Loading tax units from: {input_path}")
    
    if input_path.endswith('.parquet'):
        tax_units = pd.read_parquet(input_path)
    elif input_path.endswith('.csv'):
        tax_units = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path}")
    
    logger.info(f"  Loaded {len(tax_units):,} tax units")
    
    # Check for required columns
    weight_col = 'calibrated_weight' if 'calibrated_weight' in tax_units.columns else 'weight'
    total_income = (tax_units['income'] * tax_units[weight_col]).sum()
    logger.info(f"  Total weighted income: ${total_income:,.0f}")
    
    return tax_units


def save_results(tax_units: pd.DataFrame, 
                output_path: str):
    """Save tax units with income sources."""
    logger.info(f"\nSaving tax units with income sources to: {output_path}")
    
    # Save main results
    if output_path.endswith('.parquet'):
        tax_units.to_parquet(output_path, index=False)
    elif output_path.endswith('.csv'):
        tax_units.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported file format: {output_path}")
    
    logger.info(f"  ✅ Saved {len(tax_units):,} tax units")
    
    # List new columns
    source_columns = ['wage_income', 'dividend_income', 'interest_income',
                     'business_income', 'capital_gains', 'pension_income',
                     'other_income']
    
    logger.info(f"  ✅ Added {len(source_columns)} income source columns:")
    for col in source_columns:
        if col in tax_units.columns:
            logger.info(f"     - {col}")


def generate_comparison_report(tax_units: pd.DataFrame,
                               output_dir: str):
    """Generate detailed income source breakdown report."""
    logger.info("\nGenerating income source breakdown report...")
    
    weight_col = 'calibrated_weight' if 'calibrated_weight' in tax_units.columns else 'weight'
    
    # Overall source breakdown
    source_columns = {
        'wage_income': 'Wages',
        'dividend_income': 'Dividends',
        'interest_income': 'Interest',
        'business_income': 'Business Income',
        'capital_gains': 'Capital Gains',
        'pension_income': 'Pensions',
        'other_income': 'Other Income'
    }
    
    total_agi = (tax_units['income'] * tax_units[weight_col]).sum()
    
    summary_data = []
    for col, name in source_columns.items():
        if col in tax_units.columns:
            amount = (tax_units[col] * tax_units[weight_col]).sum()
            percentage = (amount / total_agi * 100) if total_agi > 0 else 0
            
            summary_data.append({
                'Income Source': name,
                'Total Amount': f"${amount:,.0f}",
                'Percentage': f"{percentage:.1f}%"
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary
    summary_path = Path(output_dir) / 'income_source_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"  ✅ Saved summary to: {summary_path}")
    
    # Generate by-bracket breakdown
    splitter = IncomeSourceSplitter()
    bracket_summary = splitter.get_source_summary_by_bracket(tax_units)
    
    bracket_path = Path(output_dir) / 'income_source_by_bracket.csv'
    bracket_summary.to_csv(bracket_path, index=False)
    logger.info(f"  ✅ Saved bracket breakdown to: {bracket_path}")
    
    # Print summary to console
    logger.info("\n" + "=" * 80)
    logger.info("INCOME SOURCE BREAKDOWN SUMMARY")
    logger.info("=" * 80)
    logger.info(f"\nTotal AGI: ${total_agi:,.0f}")
    logger.info("\n" + summary_df.to_string(index=False))
    logger.info("\n" + "=" * 80)
    
    return summary_df


def validate_income_sources(tax_units: pd.DataFrame):
    """Validate income source split."""
    logger.info("\n" + "=" * 80)
    logger.info("INCOME SOURCE VALIDATION")
    logger.info("=" * 80)
    
    weight_col = 'calibrated_weight' if 'calibrated_weight' in tax_units.columns else 'weight'
    
    source_columns = ['wage_income', 'dividend_income', 'interest_income',
                     'business_income', 'capital_gains', 'pension_income',
                     'other_income']
    
    # Check 1: Sum of sources equals total income
    source_sum = tax_units[source_columns].sum(axis=1)
    total_income = tax_units['income']
    
    diff = (source_sum - total_income).abs()
    max_diff = diff.max()
    mean_diff = diff.mean()
    n_mismatches = (diff > 0.01).sum()
    
    logger.info(f"\n1. Income Consistency Check:")
    logger.info(f"   Max difference:     ${max_diff:,.2f}")
    logger.info(f"   Mean difference:    ${mean_diff:,.2f}")
    logger.info(f"   Units with >$0.01:  {n_mismatches:,}")
    
    if max_diff < 1.0:
        logger.info(f"   ✅ Income split is consistent")
    else:
        logger.warning(f"   ⚠️  Income split has discrepancies")
    
    # Check 2: No negative values
    logger.info(f"\n2. Negative Value Check:")
    has_negatives = False
    for col in source_columns:
        n_negative = (tax_units[col] < 0).sum()
        if n_negative > 0:
            logger.warning(f"   ⚠️  {col}: {n_negative:,} negative values")
            has_negatives = True
    
    if not has_negatives:
        logger.info(f"   ✅ No negative values found")
    
    # Check 3: Weighted totals match IRS targets
    logger.info(f"\n3. IRS Target Comparison:")
    
    total_weighted_income = (tax_units['income'] * tax_units[weight_col]).sum()
    irs_total = sum(
        IncomeSourceSplitter.IRS_INCOME_SOURCES_BY_BRACKET[b]['total_agi']
        for b in IncomeSourceSplitter.IRS_INCOME_SOURCES_BY_BRACKET.keys()
    )
    
    # Scale IRS total to DOTAX (634,956 / 610,000)
    dotax_scale = 634956 / 610000
    irs_total_scaled = irs_total * dotax_scale
    
    ratio = total_weighted_income / irs_total_scaled
    logger.info(f"   PUMS Total:    ${total_weighted_income:,.0f}")
    logger.info(f"   IRS Total:     ${irs_total_scaled:,.0f}")
    logger.info(f"   Ratio:         {ratio:.3f}")
    
    if 0.95 <= ratio <= 1.05:
        logger.info(f"   ✅ Within 5% of IRS target")
    else:
        logger.warning(f"   ⚠️  Outside 5% of IRS target")
    
    logger.info("\n" + "=" * 80)


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("INCOME SOURCE SPLIT - STAGE 5")
    logger.info("=" * 80)
    
    # Configuration
    input_path = "src/data/processed/tax_units_high_income_enhanced.parquet"
    output_path = "src/data/processed/tax_units_income_sources.parquet"
    output_dir = "analysis_results/calibration"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load tax units
    tax_units = load_tax_units(input_path)
    
    # Initialize splitter
    logger.info("\nInitializing Income Source Splitter...")
    splitter = IncomeSourceSplitter(
        use_pums_sources=True,      # Use PUMS source data if available
        preserve_pums_wages=False   # Don't preserve PUMS wages (use IRS)
    )
    
    # Apply income source split
    logger.info("\nApplying income source split...")
    tax_units_split = splitter.split_income_sources(
        tax_units,
        income_col='income'
    )
    
    # Validate results
    validate_income_sources(tax_units_split)
    
    # Generate comparison report
    summary = generate_comparison_report(tax_units_split, output_dir)
    
    # Save results
    save_results(tax_units_split, output_path)
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("INCOME SOURCE SPLIT COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\n✅ Input:  {input_path}")
    logger.info(f"✅ Output: {output_path}")
    logger.info(f"✅ Report: {output_dir}/income_source_summary.csv")
    logger.info(f"✅ Breakdown: {output_dir}/income_source_by_bracket.csv")
    
    weight_col = 'calibrated_weight' if 'calibrated_weight' in tax_units_split.columns else 'weight'
    total_agi = (tax_units_split['income'] * tax_units_split[weight_col]).sum()
    logger.info(f"\nTotal AGI: ${total_agi:,.0f}")
    
    # Show top 3 income sources
    source_columns = {
        'wage_income': 'Wages',
        'dividend_income': 'Dividends',
        'interest_income': 'Interest',
        'business_income': 'Business',
        'capital_gains': 'Capital Gains',
        'pension_income': 'Pensions',
        'other_income': 'Other'
    }
    
    source_amounts = {
        name: (tax_units_split[col] * tax_units_split[weight_col]).sum()
        for col, name in source_columns.items()
        if col in tax_units_split.columns
    }
    
    top_sources = sorted(source_amounts.items(), key=lambda x: x[1], reverse=True)[:3]
    
    logger.info(f"\nTop 3 Income Sources:")
    for i, (name, amount) in enumerate(top_sources, 1):
        pct = (amount / total_agi * 100) if total_agi > 0 else 0
        logger.info(f"  {i}. {name}: ${amount:,.0f} ({pct:.1f}%)")
    
    logger.info("\n" + "=" * 80 + "\n")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"\n❌ Error during income source split: {e}", exc_info=True)
        sys.exit(1)
