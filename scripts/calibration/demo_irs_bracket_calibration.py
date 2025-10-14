#!/usr/bin/env python3
"""
Demo: IRS SOI Bracket Calibration

This script demonstrates the IRS SOI bracket calibration process with
synthetic data to show how the algorithm works.

The calibration process:
1. Adjusts weights to match IRS bracket counts (scaled to DOTAX total)
2. Re-normalizes to maintain DOTAX total returns
3. Adjusts average income within brackets to match IRS targets
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from src.tax.calibration.irs_bracket_calibration import IRSBracketCalibrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def create_synthetic_tax_units(n_units: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Create synthetic tax unit data for demonstration.
    
    This creates a realistic distribution of tax units with:
    - Log-normal income distribution
    - Varying weights
    - Different filing statuses
    """
    np.random.seed(seed)
    
    # Generate incomes with log-normal distribution
    # Mean around $60k, with long right tail
    incomes = np.random.lognormal(mean=10.8, sigma=1.0, size=n_units)
    
    # Generate weights (representing how many actual filers each unit represents)
    weights = np.random.uniform(100, 300, size=n_units)
    
    # Generate filing statuses
    filing_statuses = np.random.choice(
        ['single', 'married_filing_jointly', 'head_of_household', 'married_filing_separately'],
        size=n_units,
        p=[0.53, 0.34, 0.11, 0.02]
    )
    
    # Create DataFrame
    tax_units = pd.DataFrame({
        'tax_unit_id': range(n_units),
        'income': incomes,
        'weight': weights,
        'filing_status': filing_statuses
    })
    
    return tax_units


def print_distribution_summary(tax_units: pd.DataFrame, 
                               weight_col: str = 'weight',
                               title: str = "Distribution Summary"):
    """Print summary statistics of the tax unit distribution."""
    logger.info(f"\n{'=' * 70}")
    logger.info(f"{title}")
    logger.info(f"{'=' * 70}")
    
    total_returns = tax_units[weight_col].sum()
    logger.info(f"\nTotal Returns: {total_returns:,.0f}")
    
    # Income statistics
    weighted_income = (tax_units['income'] * tax_units[weight_col]).sum()
    avg_income = weighted_income / total_returns
    logger.info(f"Average Income: ${avg_income:,.0f}")
    logger.info(f"Total AGI: ${weighted_income:,.0f}")
    
    # Bracket distribution
    calibrator = IRSBracketCalibrator()
    tax_units['bracket'] = tax_units['income'].apply(calibrator.assign_bracket)
    
    logger.info(f"\nIncome Bracket Distribution:")
    logger.info(f"  {'Bracket':<12} {'Count':>14} {'Percentage':>12} {'Avg Income':>14}")
    logger.info("  " + "-" * 55)
    
    for bracket_name in ['0-25k', '25-50k', '50-75k', '75-100k', '100-200k', '200k+']:
        bracket_units = tax_units[tax_units['bracket'] == bracket_name]
        count = bracket_units[weight_col].sum()
        pct = (count / total_returns) * 100
        
        if len(bracket_units) > 0:
            avg = (bracket_units['income'] * bracket_units[weight_col]).sum() / count
            logger.info(f"  {bracket_name:<12} {count:>14,.0f} {pct:>11.1f}% ${avg:>13,.0f}")
        else:
            logger.info(f"  {bracket_name:<12} {0:>14,} {0:>11.1f}% ${'N/A':>13}")


def plot_comparison(tax_units_before: pd.DataFrame,
                   tax_units_after: pd.DataFrame,
                   output_path: str = None):
    """Create visualization comparing before/after calibration."""
    calibrator = IRSBracketCalibrator()
    
    # Assign brackets
    tax_units_before['bracket'] = tax_units_before['income'].apply(calibrator.assign_bracket)
    tax_units_after['bracket'] = tax_units_after['income'].apply(calibrator.assign_bracket)
    
    # Calculate counts
    brackets = ['0-25k', '25-50k', '50-75k', '75-100k', '100-200k', '200k+']
    before_counts = []
    after_counts = []
    target_counts = []
    
    for bracket in brackets:
        before_counts.append(
            tax_units_before[tax_units_before['bracket'] == bracket]['weight'].sum()
        )
        after_counts.append(
            tax_units_after[tax_units_after['bracket'] == bracket]['calibrated_weight'].sum()
        )
        target_counts.append(calibrator.target_counts[bracket])
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Bracket counts
    x = np.arange(len(brackets))
    width = 0.25
    
    ax1.bar(x - width, before_counts, width, label='Before Calibration', alpha=0.8)
    ax1.bar(x, after_counts, width, label='After Calibration', alpha=0.8)
    ax1.bar(x + width, target_counts, width, label='IRS/DOTAX Target', alpha=0.8)
    
    ax1.set_xlabel('Income Bracket', fontsize=12)
    ax1.set_ylabel('Number of Returns', fontsize=12)
    ax1.set_title('Tax Returns by Income Bracket', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(brackets, rotation=45)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Average income by bracket
    before_avgs = []
    after_avgs = []
    target_avgs = []
    
    for bracket in brackets:
        before_bracket = tax_units_before[tax_units_before['bracket'] == bracket]
        after_bracket = tax_units_after[tax_units_after['bracket'] == bracket]
        
        if len(before_bracket) > 0:
            before_avg = (before_bracket['income'] * before_bracket['weight']).sum() / \
                        before_bracket['weight'].sum()
            before_avgs.append(before_avg)
        else:
            before_avgs.append(0)
        
        if len(after_bracket) > 0:
            after_avg = (after_bracket['income'] * after_bracket['calibrated_weight']).sum() / \
                       after_bracket['calibrated_weight'].sum()
            after_avgs.append(after_avg)
        else:
            after_avgs.append(0)
        
        target_avgs.append(calibrator.IRS_BRACKETS[bracket]['avg'])
    
    ax2.bar(x - width, before_avgs, width, label='Before Calibration', alpha=0.8)
    ax2.bar(x, after_avgs, width, label='After Calibration', alpha=0.8)
    ax2.bar(x + width, target_avgs, width, label='IRS Target', alpha=0.8)
    
    ax2.set_xlabel('Income Bracket', fontsize=12)
    ax2.set_ylabel('Average Income ($)', fontsize=12)
    ax2.set_title('Average Income by Bracket', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(brackets, rotation=45)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"\n✅ Saved comparison plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Main demonstration function."""
    logger.info("=" * 70)
    logger.info("IRS SOI BRACKET CALIBRATION DEMO")
    logger.info("=" * 70)
    
    # Create output directory
    output_dir = Path("analysis_results/calibration/demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Create synthetic data
    logger.info("\nStep 1: Creating synthetic tax unit data...")
    tax_units = create_synthetic_tax_units(n_units=5000)
    logger.info(f"  ✅ Created {len(tax_units):,} synthetic tax units")
    
    # Show initial distribution
    print_distribution_summary(tax_units, weight_col='weight', 
                              title="BEFORE CALIBRATION")
    
    # Step 2: Initialize calibrator
    logger.info("\nStep 2: Initializing IRS Bracket Calibrator...")
    calibrator = IRSBracketCalibrator(
        max_weight_adjustment=3.0,
        max_income_adjustment=0.3,
        apply_income_adjustment=True
    )
    
    # Step 3: Apply calibration
    logger.info("\nStep 3: Applying calibration...")
    tax_units_calibrated = calibrator.calibrate(tax_units.copy())
    
    # Show calibrated distribution
    print_distribution_summary(tax_units_calibrated, weight_col='calibrated_weight',
                              title="AFTER CALIBRATION")
    
    # Step 4: Generate comparison report
    logger.info("\nStep 4: Generating comparison report...")
    summary = calibrator.get_calibration_summary(tax_units, tax_units_calibrated)
    
    logger.info("\n" + "=" * 70)
    logger.info("CALIBRATION SUMMARY")
    logger.info("=" * 70)
    logger.info("\n" + summary.to_string(index=False))
    
    # Save summary
    summary_path = output_dir / "calibration_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info(f"\n✅ Saved summary to: {summary_path}")
    
    # Step 5: Create visualization
    logger.info("\nStep 5: Creating visualization...")
    plot_path = output_dir / "calibration_comparison.png"
    plot_comparison(tax_units, tax_units_calibrated, output_path=str(plot_path))
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("DEMO COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\n✅ Summary: {summary_path}")
    logger.info(f"✅ Plot:    {plot_path}")
    logger.info("\nKey Takeaways:")
    logger.info("  • Weights adjusted to match IRS bracket counts")
    logger.info("  • Incomes adjusted to match IRS average incomes")
    logger.info("  • Total returns maintained at DOTAX target")
    logger.info("  • All adjustments bounded to prevent distortions")
    logger.info("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"\n❌ Error during demo: {e}", exc_info=True)
        sys.exit(1)
