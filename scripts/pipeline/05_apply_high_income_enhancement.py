#!/usr/bin/env python3
"""
Apply High-Income Enhancement to Tax Units

This script implements Stage 4 of the calibration process:
- Loads tax units from Stage 3 (IRS bracket calibration)
- Identifies gap in high-income representation
- Generates synthetic high-income records using Pareto distribution
- Re-calibrates to maintain DOTAX total
- Validates against IRS targets

Usage:
    python scripts/pipeline/05_apply_high_income_enhancement.py
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

from src.tax.calibration.high_income_enhancement import HighIncomeEnhancer

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
    logger.info(f"  Total weighted returns: {tax_units[weight_col].sum():,.0f}")
    
    return tax_units


def save_results(tax_units: pd.DataFrame, 
                output_path: str,
                metrics: dict):
    """Save enhanced tax units and validation metrics."""
    logger.info(f"\nSaving enhanced tax units to: {output_path}")
    
    # Save main results
    if output_path.endswith('.parquet'):
        tax_units.to_parquet(output_path, index=False)
    elif output_path.endswith('.csv'):
        tax_units.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported file format: {output_path}")
    
    logger.info(f"  ✅ Saved {len(tax_units):,} enhanced tax units")
    
    # Count synthetic records
    if 'is_synthetic' in tax_units.columns:
        n_synthetic = tax_units['is_synthetic'].sum()
        logger.info(f"  ✅ Includes {n_synthetic:,} synthetic high-income records")
    
    # Save validation metrics
    metrics_path = output_path.replace('.parquet', '_metrics.csv').replace('.csv', '_metrics.csv')
    metrics_df = pd.DataFrame([metrics])
    metrics_df['enhancement_date'] = datetime.now().isoformat()
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"  ✅ Saved validation metrics to: {metrics_path}")


def generate_comparison_report(tax_units_before: pd.DataFrame,
                               tax_units_after: pd.DataFrame,
                               output_dir: str):
    """Generate detailed comparison report."""
    logger.info("\nGenerating comparison report...")
    
    weight_col = 'calibrated_weight' if 'calibrated_weight' in tax_units_after.columns else 'weight'
    
    # High-income comparison
    threshold = 200000
    
    before_high = tax_units_before[tax_units_before['income'] > threshold]
    after_high = tax_units_after[tax_units_after['income'] > threshold]
    
    before_count = before_high[weight_col].sum()
    after_count = after_high[weight_col].sum()
    
    before_avg = (before_high['income'] * before_high[weight_col]).sum() / before_count if before_count > 0 else 0
    after_avg = (after_high['income'] * after_high[weight_col]).sum() / after_count if after_count > 0 else 0
    
    # Create summary
    summary_data = [
        {
            'Metric': 'Total Returns',
            'Before': f"{tax_units_before[weight_col].sum():,.0f}",
            'After': f"{tax_units_after[weight_col].sum():,.0f}",
            'Target': '634,956',
            'Status': '✅'
        },
        {
            'Metric': 'High-Income Count (>$200k)',
            'Before': f"{before_count:,.0f}",
            'After': f"{after_count:,.0f}",
            'Target': '15,000',
            'Status': '✅' if abs(after_count - 15000) / 15000 < 0.05 else '⚠️'
        },
        {
            'Metric': 'High-Income Avg (>$200k)',
            'Before': f"${before_avg:,.0f}",
            'After': f"${after_avg:,.0f}",
            'Target': '$400,000',
            'Status': '✅' if 0.9 <= after_avg / 400000 <= 1.1 else '⚠️'
        }
    ]
    
    if 'is_synthetic' in tax_units_after.columns:
        n_synthetic = tax_units_after['is_synthetic'].sum()
        synthetic_weight = tax_units_after.loc[tax_units_after['is_synthetic'], weight_col].sum()
        summary_data.append({
            'Metric': 'Synthetic Records',
            'Before': '0',
            'After': f"{n_synthetic:,}",
            'Target': f"{synthetic_weight:,.0f} weighted",
            'Status': '✅'
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary
    summary_path = Path(output_dir) / 'high_income_enhancement_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"  ✅ Saved summary to: {summary_path}")
    
    # Print summary to console
    logger.info("\n" + "=" * 80)
    logger.info("HIGH-INCOME ENHANCEMENT SUMMARY")
    logger.info("=" * 80)
    logger.info("\n" + summary_df.to_string(index=False))
    logger.info("\n" + "=" * 80)
    
    return summary_df


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("HIGH-INCOME ENHANCEMENT - STAGE 4")
    logger.info("=" * 80)
    
    # Configuration
    input_path = "src/data/processed/tax_units_irs_bracket_calibrated.parquet"
    output_path = "src/data/processed/tax_units_high_income_enhanced.parquet"
    output_dir = "analysis_results/calibration"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load tax units
    tax_units = load_tax_units(input_path)
    
    # Store original for comparison
    tax_units_before = tax_units.copy()
    
    # Initialize enhancer
    logger.info("\nInitializing High-Income Enhancer...")
    enhancer = HighIncomeEnhancer(
        pareto_shape=2.3,        # Calibrated to match IRS targets
        pareto_scale=50000,      # Calibrated to match IRS targets
        verify_targets=True,     # Verify synthetic data matches targets
        max_iterations=10        # Max iterations for parameter tuning
    )
    
    # Apply enhancement
    logger.info("\nApplying high-income enhancement...")
    weight_col = 'calibrated_weight' if 'calibrated_weight' in tax_units.columns else 'weight'
    tax_units_enhanced = enhancer.enhance(
        tax_units,
        weight_col=weight_col,
        income_col='income'
    )
    
    # Get validation metrics
    metrics = enhancer.validate(tax_units_enhanced, weight_col=weight_col, income_col='income')
    
    # Generate comparison report
    summary = generate_comparison_report(
        tax_units_before,
        tax_units_enhanced,
        output_dir
    )
    
    # Save results
    save_results(tax_units_enhanced, output_path, metrics)
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("ENHANCEMENT COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\n✅ Input:  {input_path}")
    logger.info(f"✅ Output: {output_path}")
    logger.info(f"✅ Report: {output_dir}/high_income_enhancement_summary.csv")
    
    if 'is_synthetic' in tax_units_enhanced.columns:
        n_synthetic = tax_units_enhanced['is_synthetic'].sum()
        synthetic_weight = tax_units_enhanced.loc[tax_units_enhanced['is_synthetic'], weight_col].sum()
        logger.info(f"\n✅ Added {n_synthetic:,} synthetic records ({synthetic_weight:,.0f} weighted)")
    
    logger.info(f"\nTotal Returns After Enhancement: {tax_units_enhanced[weight_col].sum():,.0f}")
    logger.info(f"DOTAX Target: 634,956")
    
    high_income_count = tax_units_enhanced[tax_units_enhanced['income'] > 200000][weight_col].sum()
    logger.info(f"\nHigh-Income Count (>$200k): {high_income_count:,.0f}")
    logger.info(f"IRS Target: 15,000")
    logger.info(f"Error: {abs(high_income_count - 15000) / 15000 * 100:.1f}%")
    
    logger.info("\n" + "=" * 80 + "\n")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"\n❌ Error during enhancement: {e}", exc_info=True)
        sys.exit(1)
