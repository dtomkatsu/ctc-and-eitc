#!/usr/bin/env python3
"""
Apply Wage Growth Adjustment to Tax Units

This script implements Phase 1 of growth forecasting:
- Adjusts 2022 PUMS wage data to 2024 using bracket-specific growth rates
- Lower income brackets see higher wage growth (catch-up effect)
- Higher income brackets see moderate wage growth
- Updates total income to reflect wage changes
- Generates detailed growth rate reports

Usage:
    python scripts/pipeline/07_apply_wage_growth_adjustment.py
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Bracket-specific wage growth rates (2022 → 2024)
# Based on BLS OES data showing differential growth by income level
WAGE_GROWTH_BY_BRACKET = {
    '0-25k': 0.1450,      # 14.5% - Minimum wage increases, service sector recovery
    '25-50k': 0.1300,     # 13.0% - Strong growth in lower-middle income jobs
    '50-75k': 0.1150,     # 11.5% - Above-average growth
    '75-100k': 0.1050,    # 10.5% - Near-average growth
    '100-200k': 0.0950,   # 9.5% - Below-average growth
    '200k+': 0.0850       # 8.5% - Slowest growth (already high earners)
}

# Overall employment-weighted average: 11.38%
OVERALL_WAGE_GROWTH = 0.1138

# Bracket bounds (matching income source split)
BRACKET_BOUNDS = {
    '0-25k': (0, 25000),
    '25-50k': (25000, 50000),
    '50-75k': (50000, 75000),
    '75-100k': (75000, 100000),
    '100-200k': (100000, 200000),
    '200k+': (200000, float('inf'))
}


def get_bracket_name(income: float) -> str:
    """Get bracket name for a given income level."""
    for bracket_name, (low, high) in BRACKET_BOUNDS.items():
        if low <= income < high:
            return bracket_name
    return '200k+'  # Default to highest bracket


def get_wage_growth_rate(income: float) -> float:
    """Get wage growth rate for a given income level."""
    bracket = get_bracket_name(income)
    return WAGE_GROWTH_BY_BRACKET[bracket]


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
    
    if 'income' in tax_units.columns:
        total_income = (tax_units['income'] * tax_units[weight_col]).sum()
        logger.info(f"  Total weighted income: ${total_income:,.0f}")
    
    if 'wage_income' in tax_units.columns:
        total_wages = (tax_units['wage_income'] * tax_units[weight_col]).sum()
        logger.info(f"  Total weighted wages: ${total_wages:,.0f}")
    
    return tax_units


def apply_bracket_specific_wage_growth(tax_units: pd.DataFrame) -> pd.DataFrame:
    """Apply bracket-specific wage growth rates to tax units."""
    logger.info("\nApplying bracket-specific wage growth adjustment...")
    
    # Make a copy
    tax_units = tax_units.copy()
    
    # Check if wage_income exists, if not create it from total income
    if 'wage_income' not in tax_units.columns:
        logger.warning("  ⚠️  'wage_income' column not found. Estimating from total income...")
        # Assume 75% of income is wages (conservative estimate)
        tax_units['wage_income'] = tax_units['income'] * 0.75
        logger.info(f"  Created wage_income column (75% of total income)")
    
    # Save original wage income
    tax_units['wage_income_2022'] = tax_units['wage_income']
    
    # Calculate growth rate for each tax unit based on income
    logger.info("  Calculating bracket-specific growth rates...")
    tax_units['wage_growth_rate'] = tax_units['income'].apply(get_wage_growth_rate)
    tax_units['income_bracket'] = tax_units['income'].apply(get_bracket_name)
    
    # Apply growth rates
    logger.info("  Applying wage adjustments...")
    tax_units['wage_income'] = tax_units['wage_income_2022'] * (1 + tax_units['wage_growth_rate'])
    
    # Update total income
    wage_increase = tax_units['wage_income'] - tax_units['wage_income_2022']
    tax_units['income'] = tax_units['income'] + wage_increase
    
    # Log bracket-level statistics
    weight_col = 'calibrated_weight' if 'calibrated_weight' in tax_units.columns else 'weight'
    
    logger.info("\n  Wage Growth by Income Bracket:")
    logger.info(f"  {'Bracket':<12} {'Count':>8} {'Growth Rate':>12} {'Wage 2022':>15} {'Wage 2024':>15} {'Increase':>15}")
    logger.info("  " + "-" * 90)
    
    for bracket in ['0-25k', '25-50k', '50-75k', '75-100k', '100-200k', '200k+']:
        mask = tax_units['income_bracket'] == bracket
        if mask.sum() > 0:
            count = mask.sum()
            growth_rate = WAGE_GROWTH_BY_BRACKET[bracket]
            wage_2022 = (tax_units.loc[mask, 'wage_income_2022'] * tax_units.loc[mask, weight_col]).sum()
            wage_2024 = (tax_units.loc[mask, 'wage_income'] * tax_units.loc[mask, weight_col]).sum()
            increase = wage_2024 - wage_2022
            
            logger.info(f"  {bracket:<12} {count:>8,} {growth_rate:>11.1%} ${wage_2022:>14,.0f} ${wage_2024:>14,.0f} ${increase:>14,.0f}")
    
    # Overall statistics
    total_wage_2022 = (tax_units['wage_income_2022'] * tax_units[weight_col]).sum()
    total_wage_2024 = (tax_units['wage_income'] * tax_units[weight_col]).sum()
    total_increase = total_wage_2024 - total_wage_2022
    overall_growth = (total_wage_2024 / total_wage_2022 - 1) if total_wage_2022 > 0 else 0
    
    logger.info("  " + "-" * 90)
    logger.info(f"  {'TOTAL':<12} {len(tax_units):>8,} {overall_growth:>11.1%} ${total_wage_2022:>14,.0f} ${total_wage_2024:>14,.0f} ${total_increase:>14,.0f}")
    
    return tax_units


def save_results(tax_units: pd.DataFrame, 
                output_path: str,
                output_dir: str):
    """Save tax units with wage adjustments."""
    logger.info(f"\nSaving tax units with wage adjustments to: {output_path}")
    
    # Save main results
    if output_path.endswith('.parquet'):
        tax_units.to_parquet(output_path, index=False)
    elif output_path.endswith('.csv'):
        tax_units.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported file format: {output_path}")
    
    logger.info(f"  ✅ Saved {len(tax_units):,} tax units")
    
    # Save growth rate summary
    summary_data = []
    for bracket, growth_rate in WAGE_GROWTH_BY_BRACKET.items():
        summary_data.append({
            'Income Bracket': bracket,
            'Growth Rate': f"{growth_rate:.1%}",
            'Adjustment Factor': f"{1 + growth_rate:.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = Path(output_dir) / 'wage_growth_rates_by_bracket.csv'
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"  ✅ Saved growth rate summary to: {summary_path}")


def validate_results(tax_units: pd.DataFrame):
    """Validate wage adjustment results."""
    logger.info("\n" + "=" * 80)
    logger.info("WAGE ADJUSTMENT VALIDATION")
    logger.info("=" * 80)
    
    weight_col = 'calibrated_weight' if 'calibrated_weight' in tax_units.columns else 'weight'
    
    # Check 1: All units have wage adjustments
    n_adjusted = (tax_units['wage_income'] != tax_units['wage_income_2022']).sum()
    pct_adjusted = n_adjusted / len(tax_units) * 100
    
    logger.info(f"\n1. Adjustment Coverage:")
    logger.info(f"   Units adjusted:     {n_adjusted:,} ({pct_adjusted:.1f}%)")
    
    if pct_adjusted > 95:
        logger.info(f"   ✅ Nearly all units adjusted")
    else:
        logger.warning(f"   ⚠️  Some units not adjusted")
    
    # Check 2: Growth rates are reasonable
    individual_growth = (
        (tax_units['wage_income'] - tax_units['wage_income_2022']) /
        tax_units['wage_income_2022'].replace(0, np.nan)
    )
    
    logger.info(f"\n2. Individual Growth Rate Distribution:")
    logger.info(f"   Min:                {individual_growth.min():>8.2%}")
    logger.info(f"   25th percentile:    {individual_growth.quantile(0.25):>8.2%}")
    logger.info(f"   Median:             {individual_growth.median():>8.2%}")
    logger.info(f"   75th percentile:    {individual_growth.quantile(0.75):>8.2%}")
    logger.info(f"   Max:                {individual_growth.max():>8.2%}")
    
    # Check 3: Total returns maintained
    total_returns = tax_units[weight_col].sum()
    logger.info(f"\n3. Total Returns Check:")
    logger.info(f"   Total returns:      {total_returns:,.0f}")
    logger.info(f"   Expected (DOTAX):   634,956")
    
    ratio = total_returns / 634956
    if 0.99 <= ratio <= 1.01:
        logger.info(f"   ✅ Total returns maintained")
    else:
        logger.warning(f"   ⚠️  Total returns changed (ratio: {ratio:.3f})")
    
    logger.info("\n" + "=" * 80)


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("WAGE GROWTH ADJUSTMENT - PHASE 1 (2022 → 2024)")
    logger.info("BRACKET-SPECIFIC GROWTH RATES")
    logger.info("=" * 80)
    
    # Configuration - try multiple possible input paths
    possible_inputs = [
        "src/data/processed/tax_units_income_sources.parquet",
        "data/processed/tax_units_soi_calibrated.parquet",
        "src/data/processed/tax_units_irs_based.parquet"
    ]
    
    input_path = None
    for path in possible_inputs:
        if Path(path).exists():
            input_path = path
            logger.info(f"\n✅ Found input file: {path}")
            break
    
    if input_path is None:
        logger.error(f"\n❌ No input file found. Tried:")
        for path in possible_inputs:
            logger.error(f"   - {path}")
        logger.error("\nPlease run previous pipeline stages first:")
        logger.error("  1. python scripts/pipeline/01_construct_tax_units.py")
        logger.error("  2. python scripts/pipeline/02_apply_soi_calibration.py")
        logger.error("  3. python scripts/pipeline/04_apply_irs_bracket_calibration.py")
        logger.error("  4. python scripts/pipeline/05_apply_high_income_enhancement.py")
        logger.error("  5. python scripts/pipeline/06_apply_income_source_split.py")
        sys.exit(1)
    
    output_path = "src/data/processed/tax_units_2024_adjusted.parquet"
    output_dir = "analysis_results/calibration"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load tax units
    tax_units = load_tax_units(input_path)
    
    # Show growth rate configuration
    logger.info("\nBracket-Specific Wage Growth Rates (2022 → 2024):")
    logger.info(f"  {'Bracket':<12} {'Growth Rate':>12} {'Rationale'}")
    logger.info("  " + "-" * 70)
    logger.info(f"  {'0-25k':<12} {WAGE_GROWTH_BY_BRACKET['0-25k']:>11.1%}  Minimum wage increases, service recovery")
    logger.info(f"  {'25-50k':<12} {WAGE_GROWTH_BY_BRACKET['25-50k']:>11.1%}  Strong lower-middle income growth")
    logger.info(f"  {'50-75k':<12} {WAGE_GROWTH_BY_BRACKET['50-75k']:>11.1%}  Above-average growth")
    logger.info(f"  {'75-100k':<12} {WAGE_GROWTH_BY_BRACKET['75-100k']:>11.1%}  Near-average growth")
    logger.info(f"  {'100-200k':<12} {WAGE_GROWTH_BY_BRACKET['100-200k']:>11.1%}  Below-average growth")
    logger.info(f"  {'200k+':<12} {WAGE_GROWTH_BY_BRACKET['200k+']:>11.1%}  Slowest growth (high earners)")
    logger.info(f"\n  Overall employment-weighted average: {OVERALL_WAGE_GROWTH:.2%}")
    
    # Apply wage adjustment
    tax_units_adjusted = apply_bracket_specific_wage_growth(tax_units)
    
    # Validate results
    validate_results(tax_units_adjusted)
    
    # Save results
    save_results(tax_units_adjusted, output_path, output_dir)
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("WAGE GROWTH ADJUSTMENT COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\n✅ Input:  {input_path}")
    logger.info(f"✅ Output: {output_path}")
    logger.info(f"✅ Growth rates: {output_dir}/wage_growth_rates_by_bracket.csv")
    
    weight_col = 'calibrated_weight' if 'calibrated_weight' in tax_units_adjusted.columns else 'weight'
    total_wage_2022 = (tax_units_adjusted['wage_income_2022'] * tax_units_adjusted[weight_col]).sum()
    total_wage_2024 = (tax_units_adjusted['wage_income'] * tax_units_adjusted[weight_col]).sum()
    total_increase = total_wage_2024 - total_wage_2022
    
    logger.info(f"\nTotal Wage Income:")
    logger.info(f"  2022: ${total_wage_2022:,.0f}")
    logger.info(f"  2024: ${total_wage_2024:,.0f}")
    logger.info(f"  Increase: ${total_increase:,.0f} ({total_increase/total_wage_2022:.1%})")
    
    logger.info("\n" + "=" * 80 + "\n")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"\n❌ Error during wage adjustment: {e}", exc_info=True)
        sys.exit(1)
