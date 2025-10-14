#!/usr/bin/env python3
"""
Apply IRS SOI Bracket Calibration to Tax Units

This script implements Stage 3 of the calibration process:
- Loads tax units from Stage 2 (DOTAX calibration)
- Applies IRS SOI bracket matching for counts and averages
- Saves calibrated results with validation metrics

Usage:
    python scripts/pipeline/04_apply_irs_bracket_calibration.py
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

from src.tax.calibration.irs_bracket_calibration import IRSBracketCalibrator

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
    logger.info(f"  Total weighted returns: {tax_units['weight'].sum():,.0f}")
    
    return tax_units


def save_results(tax_units: pd.DataFrame, 
                output_path: str,
                metrics: dict):
    """Save calibrated tax units and validation metrics."""
    logger.info(f"\nSaving calibrated tax units to: {output_path}")
    
    # Save main results
    if output_path.endswith('.parquet'):
        tax_units.to_parquet(output_path, index=False)
    elif output_path.endswith('.csv'):
        tax_units.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported file format: {output_path}")
    
    logger.info(f"  ✅ Saved {len(tax_units):,} calibrated tax units")
    
    # Save validation metrics
    metrics_path = output_path.replace('.parquet', '_metrics.csv').replace('.csv', '_metrics.csv')
    metrics_df = pd.DataFrame([metrics])
    metrics_df['calibration_date'] = datetime.now().isoformat()
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"  ✅ Saved validation metrics to: {metrics_path}")


def generate_comparison_report(tax_units_before: pd.DataFrame,
                               tax_units_after: pd.DataFrame,
                               calibrator: IRSBracketCalibrator,
                               output_dir: str):
    """Generate detailed comparison report."""
    logger.info("\nGenerating comparison report...")
    
    # Get summary
    summary = calibrator.get_calibration_summary(
        tax_units_before, 
        tax_units_after,
        income_col='income'
    )
    
    # Save summary
    summary_path = Path(output_dir) / 'irs_bracket_calibration_summary.csv'
    summary.to_csv(summary_path, index=False)
    logger.info(f"  ✅ Saved summary to: {summary_path}")
    
    # Print summary to console
    logger.info("\n" + "=" * 80)
    logger.info("IRS BRACKET CALIBRATION SUMMARY")
    logger.info("=" * 80)
    logger.info("\n" + summary.to_string(index=False))
    logger.info("\n" + "=" * 80)
    
    return summary


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("IRS SOI BRACKET CALIBRATION - STAGE 3")
    logger.info("=" * 80)
    
    # Configuration
    input_path = "src/data/processed/tax_units_dotax_calibrated.parquet"
    output_path = "src/data/processed/tax_units_irs_bracket_calibrated.parquet"
    output_dir = "analysis_results/calibration"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load tax units
    tax_units = load_tax_units(input_path)
    
    # Store original for comparison
    tax_units_before = tax_units.copy()
    
    # Initialize calibrator
    logger.info("\nInitializing IRS Bracket Calibrator...")
    calibrator = IRSBracketCalibrator(
        max_weight_adjustment=3.0,      # Allow up to 3x weight adjustments
        max_income_adjustment=0.3,      # Allow ±30% income adjustments
        apply_income_adjustment=True    # Adjust incomes within brackets
    )
    
    # Apply calibration
    logger.info("\nApplying IRS SOI bracket calibration...")
    tax_units_calibrated = calibrator.calibrate(
        tax_units,
        weight_col='calibrated_weight' if 'calibrated_weight' in tax_units.columns else 'weight',
        income_col='income'
    )
    
    # Get validation metrics
    metrics = calibrator.validate(tax_units_calibrated, income_col='income')
    
    # Generate comparison report
    summary = generate_comparison_report(
        tax_units_before,
        tax_units_calibrated,
        calibrator,
        output_dir
    )
    
    # Save results
    save_results(tax_units_calibrated, output_path, metrics)
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("CALIBRATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\n✅ Input:  {input_path}")
    logger.info(f"✅ Output: {output_path}")
    logger.info(f"✅ Report: {output_dir}/irs_bracket_calibration_summary.csv")
    logger.info(f"\nTotal Returns After Calibration: {tax_units_calibrated['calibrated_weight'].sum():,.0f}")
    logger.info(f"DOTAX Target: {calibrator.DOTAX_TOTAL_RETURNS:,}")
    logger.info(f"Average Bracket Error: {metrics.get('avg_bracket_error_pct', 0):.2f}%")
    logger.info("\n" + "=" * 80 + "\n")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"\n❌ Error during calibration: {e}", exc_info=True)
        sys.exit(1)
