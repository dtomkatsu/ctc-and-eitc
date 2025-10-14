#!/usr/bin/env python3
"""
Demo: High-Income Enhancement

This script demonstrates the high-income enhancement process with
synthetic data to show how the algorithm addresses PUMS undersampling
of wealthy households.

The enhancement process:
1. Identifies gap between PUMS and IRS high-income counts
2. Generates synthetic records using Pareto distribution
3. Matches IRS average income and percentile floors
4. Re-calibrates to maintain DOTAX total
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

from src.tax.calibration.high_income_enhancement import HighIncomeEnhancer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def create_synthetic_tax_units_with_gap(n_units: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Create synthetic tax unit data with underrepresented high-income households.
    
    This simulates the PUMS problem where high earners are undersampled.
    """
    np.random.seed(seed)
    
    # Generate incomes with insufficient high earners (typical PUMS issue)
    # Use log-normal but truncate the right tail
    incomes = np.random.lognormal(mean=10.5, sigma=1.0, size=n_units)
    
    # Artificially reduce high-income representation (simulate PUMS undersampling)
    high_income_mask = incomes > 200000
    n_high = high_income_mask.sum()
    
    # Remove 60% of high-income units to simulate undersampling
    remove_indices = np.random.choice(
        np.where(high_income_mask)[0],
        size=int(n_high * 0.6),
        replace=False
    )
    
    # Create DataFrame without removed high-income units
    keep_mask = np.ones(n_units, dtype=bool)
    keep_mask[remove_indices] = False
    
    incomes = incomes[keep_mask]
    n_units = len(incomes)
    
    # Generate weights
    weights = np.random.uniform(120, 140, size=n_units)
    
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
        'calibrated_weight': weights,
        'filing_status': filing_statuses
    })
    
    return tax_units


def print_distribution_summary(tax_units: pd.DataFrame, 
                               weight_col: str = 'calibrated_weight',
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
    
    # High-income statistics
    high_income_mask = tax_units['income'] > 200000
    high_income_count = tax_units.loc[high_income_mask, weight_col].sum()
    high_income_pct = (high_income_count / total_returns) * 100
    
    if high_income_mask.sum() > 0:
        high_income_avg = (
            tax_units.loc[high_income_mask, 'income'] * 
            tax_units.loc[high_income_mask, weight_col]
        ).sum() / high_income_count
        
        logger.info(f"\nHigh-Income (>$200k):")
        logger.info(f"  Count:      {high_income_count:>12,.0f} ({high_income_pct:.1f}%)")
        logger.info(f"  Avg Income: ${high_income_avg:>11,.0f}")
    
    # Percentile statistics
    sorted_units = tax_units.sort_values('income')
    cumulative_weight = sorted_units[weight_col].cumsum()
    total_weight = sorted_units[weight_col].sum()
    
    for pct in [90, 95, 99]:
        pct_idx = (cumulative_weight >= (pct/100) * total_weight).idxmax()
        pct_income = sorted_units.loc[pct_idx, 'income']
        logger.info(f"  P{pct}:        ${pct_income:>11,.0f}")
    
    # Synthetic record count
    if 'is_synthetic' in tax_units.columns:
        n_synthetic = tax_units['is_synthetic'].sum()
        synthetic_weight = tax_units.loc[tax_units['is_synthetic'], weight_col].sum()
        logger.info(f"\nSynthetic Records:")
        logger.info(f"  Count:      {n_synthetic:>12,}")
        logger.info(f"  Weight:     {synthetic_weight:>12,.0f}")


def plot_comparison(tax_units_before: pd.DataFrame,
                   tax_units_after: pd.DataFrame,
                   output_path: str = None):
    """Create visualization comparing before/after enhancement."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    weight_col = 'calibrated_weight'
    
    # Plot 1: Income distribution (log scale)
    ax1 = axes[0, 0]
    
    # Before
    before_incomes = np.repeat(tax_units_before['income'].values, 
                               tax_units_before[weight_col].astype(int).values)
    ax1.hist(before_incomes[before_incomes > 0], bins=50, alpha=0.6, 
             label='Before Enhancement', edgecolor='black')
    
    # After
    after_incomes = np.repeat(tax_units_after['income'].values,
                             tax_units_after[weight_col].astype(int).values)
    ax1.hist(after_incomes[after_incomes > 0], bins=50, alpha=0.6,
             label='After Enhancement', edgecolor='black')
    
    ax1.set_xlabel('Income ($)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Income Distribution (All Returns)', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: High-income focus (>$200k)
    ax2 = axes[0, 1]
    
    before_high = tax_units_before[tax_units_before['income'] > 200000]
    after_high = tax_units_after[tax_units_after['income'] > 200000]
    
    if len(before_high) > 0:
        before_high_incomes = np.repeat(before_high['income'].values,
                                        before_high[weight_col].astype(int).values)
        ax2.hist(before_high_incomes, bins=30, alpha=0.6,
                label=f'Before ({len(before_high):,} records)', edgecolor='black')
    
    if len(after_high) > 0:
        after_high_incomes = np.repeat(after_high['income'].values,
                                       after_high[weight_col].astype(int).values)
        ax2.hist(after_high_incomes, bins=30, alpha=0.6,
                label=f'After ({len(after_high):,} records)', edgecolor='black')
    
    ax2.axvline(400000, color='red', linestyle='--', linewidth=2, label='IRS Avg ($400k)')
    ax2.axvline(650000, color='orange', linestyle='--', linewidth=2, label='Top 1% Floor ($650k)')
    
    ax2.set_xlabel('Income ($)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('High-Income Distribution (>$200k)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Plot 3: Cumulative distribution
    ax3 = axes[1, 0]
    
    # Before
    before_sorted = tax_units_before.sort_values('income')
    before_cumsum = before_sorted[weight_col].cumsum()
    before_total = before_sorted[weight_col].sum()
    ax3.plot(before_sorted['income'], before_cumsum / before_total * 100,
            label='Before Enhancement', linewidth=2)
    
    # After
    after_sorted = tax_units_after.sort_values('income')
    after_cumsum = after_sorted[weight_col].cumsum()
    after_total = after_sorted[weight_col].sum()
    ax3.plot(after_sorted['income'], after_cumsum / after_total * 100,
            label='After Enhancement', linewidth=2)
    
    ax3.axhline(99, color='red', linestyle='--', alpha=0.5, label='99th Percentile')
    ax3.axvline(200000, color='gray', linestyle='--', alpha=0.5, label='$200k Threshold')
    
    ax3.set_xlabel('Income ($)', fontsize=12)
    ax3.set_ylabel('Cumulative Percentage', fontsize=12)
    ax3.set_title('Cumulative Income Distribution', fontsize=14, fontweight='bold')
    ax3.set_xscale('log')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Plot 4: Count comparison by bracket
    ax4 = axes[1, 1]
    
    brackets = [
        (0, 25000, '$0-25k'),
        (25000, 50000, '$25-50k'),
        (50000, 75000, '$50-75k'),
        (75000, 100000, '$75-100k'),
        (100000, 200000, '$100-200k'),
        (200000, float('inf'), '$200k+')
    ]
    
    before_counts = []
    after_counts = []
    labels = []
    
    for low, high, label in brackets:
        before_mask = (tax_units_before['income'] >= low) & (tax_units_before['income'] < high)
        after_mask = (tax_units_after['income'] >= low) & (tax_units_after['income'] < high)
        
        before_counts.append(tax_units_before.loc[before_mask, weight_col].sum())
        after_counts.append(tax_units_after.loc[after_mask, weight_col].sum())
        labels.append(label)
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax4.bar(x - width/2, before_counts, width, label='Before', alpha=0.8)
    ax4.bar(x + width/2, after_counts, width, label='After', alpha=0.8)
    
    ax4.set_xlabel('Income Bracket', fontsize=12)
    ax4.set_ylabel('Number of Returns', fontsize=12)
    ax4.set_title('Returns by Income Bracket', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=45)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
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
    logger.info("HIGH-INCOME ENHANCEMENT DEMO")
    logger.info("=" * 70)
    
    # Create output directory
    output_dir = Path("analysis_results/calibration/demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Create synthetic data with high-income gap
    logger.info("\nStep 1: Creating synthetic tax unit data with high-income gap...")
    tax_units = create_synthetic_tax_units_with_gap(n_units=5000)
    logger.info(f"  ✅ Created {len(tax_units):,} synthetic tax units")
    logger.info(f"  ⚠️  Artificially reduced high-income representation (simulating PUMS)")
    
    # Show initial distribution
    print_distribution_summary(tax_units, weight_col='calibrated_weight', 
                              title="BEFORE ENHANCEMENT")
    
    # Step 2: Initialize enhancer
    logger.info("\nStep 2: Initializing High-Income Enhancer...")
    enhancer = HighIncomeEnhancer(
        pareto_shape=2.3,
        pareto_scale=50000,
        verify_targets=True
    )
    
    # Step 3: Apply enhancement
    logger.info("\nStep 3: Applying high-income enhancement...")
    tax_units_enhanced = enhancer.enhance(tax_units.copy())
    
    # Show enhanced distribution
    print_distribution_summary(tax_units_enhanced, weight_col='calibrated_weight',
                              title="AFTER ENHANCEMENT")
    
    # Step 4: Create visualization
    logger.info("\nStep 4: Creating visualization...")
    plot_path = output_dir / "high_income_enhancement_comparison.png"
    plot_comparison(tax_units, tax_units_enhanced, output_path=str(plot_path))
    
    # Step 5: Generate detailed report
    logger.info("\nStep 5: Generating detailed report...")
    
    report_data = []
    
    # Total returns
    before_total = tax_units['calibrated_weight'].sum()
    after_total = tax_units_enhanced['calibrated_weight'].sum()
    report_data.append({
        'Metric': 'Total Returns',
        'Before': f"{before_total:,.0f}",
        'After': f"{after_total:,.0f}",
        'Target': '634,956',
        'Change': f"{((after_total - before_total) / before_total * 100):+.1f}%"
    })
    
    # High-income count
    before_high = tax_units[tax_units['income'] > 200000]
    after_high = tax_units_enhanced[tax_units_enhanced['income'] > 200000]
    
    before_high_count = before_high['calibrated_weight'].sum()
    after_high_count = after_high['calibrated_weight'].sum()
    
    report_data.append({
        'Metric': 'High-Income Count (>$200k)',
        'Before': f"{before_high_count:,.0f}",
        'After': f"{after_high_count:,.0f}",
        'Target': '15,000',
        'Change': f"{((after_high_count - before_high_count) / before_high_count * 100):+.1f}%"
    })
    
    # High-income average
    before_high_avg = (before_high['income'] * before_high['calibrated_weight']).sum() / before_high_count if before_high_count > 0 else 0
    after_high_avg = (after_high['income'] * after_high['calibrated_weight']).sum() / after_high_count if after_high_count > 0 else 0
    
    report_data.append({
        'Metric': 'High-Income Avg (>$200k)',
        'Before': f"${before_high_avg:,.0f}",
        'After': f"${after_high_avg:,.0f}",
        'Target': '$400,000',
        'Change': f"{((after_high_avg - before_high_avg) / before_high_avg * 100):+.1f}%"
    })
    
    # Synthetic records
    if 'is_synthetic' in tax_units_enhanced.columns:
        n_synthetic = tax_units_enhanced['is_synthetic'].sum()
        synthetic_weight = tax_units_enhanced.loc[tax_units_enhanced['is_synthetic'], 'calibrated_weight'].sum()
        report_data.append({
            'Metric': 'Synthetic Records Added',
            'Before': '0',
            'After': f"{n_synthetic:,}",
            'Target': f"{synthetic_weight:,.0f} weighted",
            'Change': 'N/A'
        })
    
    report_df = pd.DataFrame(report_data)
    
    # Save report
    report_path = output_dir / "enhancement_report.csv"
    report_df.to_csv(report_path, index=False)
    logger.info(f"  ✅ Saved report to: {report_path}")
    
    # Print report
    logger.info("\n" + "=" * 70)
    logger.info("ENHANCEMENT REPORT")
    logger.info("=" * 70)
    logger.info("\n" + report_df.to_string(index=False))
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("DEMO COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\n✅ Report: {report_path}")
    logger.info(f"✅ Plot:   {plot_path}")
    logger.info("\nKey Takeaways:")
    logger.info("  • Identified gap in high-income representation")
    logger.info("  • Generated synthetic records using Pareto distribution")
    logger.info("  • Matched IRS average income and percentile floors")
    logger.info("  • Maintained DOTAX total returns")
    logger.info("  • Improved high-income accuracy for revenue estimates")
    logger.info("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"\n❌ Error during demo: {e}", exc_info=True)
        sys.exit(1)
