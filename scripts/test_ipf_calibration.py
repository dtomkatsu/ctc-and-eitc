#!/usr/bin/env python3
"""
Test and Compare IPF vs Sequential Calibration

This script demonstrates the difference between:
1. Sequential calibration (each step can undermine previous steps)
2. IPF calibration (iteratively adjusts all targets simultaneously)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
import logging

from src.tax.calibration import apply_ipf_calibration, apply_systematic_calibration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def create_test_dataset(n_units: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic test dataset."""
    np.random.seed(seed)
    
    # Generate realistic distributions
    agi = np.random.lognormal(10.5, 1.0, n_units)
    
    filing_status = np.random.choice(
        ['single', 'married_filing_jointly', 'head_of_household', 'married_filing_separately'],
        n_units,
        p=[0.55, 0.30, 0.12, 0.03]  # Initial distribution (off-target)
    )
    
    # Tax liability roughly proportional to AGI with some noise
    hi_state_tax = agi * 0.08 * np.random.lognormal(0, 0.3, n_units)
    
    weight = np.ones(n_units) * 300  # Initial weight
    
    df = pd.DataFrame({
        'agi': agi,
        'filing_status': filing_status,
        'hi_state_tax': hi_state_tax,
        'weight': weight,
        'num_dependents': np.random.choice([0, 1, 2, 3], n_units, p=[0.5, 0.2, 0.2, 0.1]),
        'num_adults': np.ones(n_units, dtype=int),
        'total_deductions': np.random.uniform(0, 10000, n_units),
        'hi_taxable_income': agi * 0.9,
    })
    
    return df


def calculate_metrics(df: pd.DataFrame, name: str) -> dict:
    """Calculate calibration quality metrics."""
    from src.tax.calibration.ipf_orchestrator import IPFCalibrationOrchestrator
    
    orchestrator = IPFCalibrationOrchestrator()
    
    filer_results = orchestrator._validate_filer_counts(df)
    tax_results = orchestrator._validate_tax_totals(df)
    status_results = orchestrator._validate_filing_status(df)
    
    return {
        'name': name,
        'filer_chi_squared': filer_results['chi_squared'],
        'filer_max_dev': filer_results['max_deviation'],
        'tax_chi_squared': tax_results['chi_squared'],
        'tax_max_dev': tax_results['max_deviation'],
        'status_chi_squared': status_results['chi_squared'],
        'status_max_dev': status_results['max_deviation'],
    }


def main():
    """Main comparison test."""
    logger.info("="*80)
    logger.info("IPF VS SEQUENTIAL CALIBRATION COMPARISON")
    logger.info("="*80)
    
    # Create test dataset
    logger.info("\nCreating test dataset...")
    df = create_test_dataset(n_units=2000)
    
    logger.info(f"Dataset: {len(df)} tax units")
    logger.info(f"Total weighted filers: {df['weight'].sum():,.0f}")
    logger.info(f"Total weighted tax: ${(df['hi_state_tax'] * df['weight']).sum() / 1_000_000:,.1f}M")
    
    # Get baseline metrics
    baseline_metrics = calculate_metrics(df, "Baseline (uncalibrated)")
    
    # Test 1: Sequential Calibration
    logger.info("\n" + "="*80)
    logger.info("TEST 1: SEQUENTIAL CALIBRATION")
    logger.info("="*80)
    
    df_sequential = df.copy()
    df_sequential = apply_systematic_calibration(
        df_sequential,
        max_iterations=5,
        tolerance=0.05
    )
    
    sequential_metrics = calculate_metrics(df_sequential, "Sequential Calibration")
    
    # Test 2: IPF Calibration
    logger.info("\n" + "="*80)
    logger.info("TEST 2: IPF CALIBRATION")
    logger.info("="*80)
    
    df_ipf = df.copy()
    df_ipf = apply_ipf_calibration(
        df_ipf,
        max_iterations=20,
        tolerance=0.02,
        calibrate_filer_counts=True,
        calibrate_tax_totals=True,
        calibrate_filing_status=True
    )
    
    ipf_metrics = calculate_metrics(df_ipf, "IPF Calibration")
    
    # Compare results
    logger.info("\n" + "="*80)
    logger.info("COMPARISON RESULTS")
    logger.info("="*80)
    
    metrics_list = [baseline_metrics, sequential_metrics, ipf_metrics]
    
    logger.info("\n1. FILER COUNT ACCURACY:")
    logger.info(f"{'Method':<25} {'Chi-Squared':>15} {'Max Deviation':>15}")
    logger.info("─" * 60)
    for m in metrics_list:
        logger.info(f"{m['name']:<25} {m['filer_chi_squared']:>15.2f} {m['filer_max_dev']:>14.1%}")
    
    logger.info("\n2. TAX TOTAL ACCURACY:")
    logger.info(f"{'Method':<25} {'Chi-Squared':>15} {'Max Deviation':>15}")
    logger.info("─" * 60)
    for m in metrics_list:
        logger.info(f"{m['name']:<25} {m['tax_chi_squared']:>15.2f} {m['tax_max_dev']:>14.1%}")
    
    logger.info("\n3. FILING STATUS ACCURACY:")
    logger.info(f"{'Method':<25} {'Chi-Squared':>15} {'Max Deviation':>15}")
    logger.info("─" * 60)
    for m in metrics_list:
        logger.info(f"{m['name']:<25} {m['status_chi_squared']:>15.2f} {m['status_max_dev']:>14.1%}")
    
    # Overall winner
    logger.info("\n" + "="*80)
    logger.info("WINNER")
    logger.info("="*80)
    
    # Lower chi-squared is better
    filer_winner = "IPF" if ipf_metrics['filer_chi_squared'] < sequential_metrics['filer_chi_squared'] else "Sequential"
    tax_winner = "IPF" if ipf_metrics['tax_chi_squared'] < sequential_metrics['tax_chi_squared'] else "Sequential"
    status_winner = "IPF" if ipf_metrics['status_chi_squared'] < sequential_metrics['status_chi_squared'] else "Sequential"
    
    logger.info(f"\nFiler Count Accuracy: {filer_winner} calibration")
    logger.info(f"Tax Total Accuracy: {tax_winner} calibration")
    logger.info(f"Filing Status Accuracy: {status_winner} calibration")
    
    if filer_winner == tax_winner == status_winner == "IPF":
        logger.info("\n✅ IPF calibration is superior across all metrics!")
    elif filer_winner == tax_winner == status_winner == "Sequential":
        logger.info("\n✅ Sequential calibration is superior across all metrics!")
    else:
        logger.info("\n⚠️  Mixed results - each method has strengths")
    
    logger.info("\n" + "="*80)
    logger.info("KEY INSIGHT")
    logger.info("="*80)
    logger.info("\nIPF Calibration Advantages:")
    logger.info("  • Simultaneously adjusts multiple targets")
    logger.info("  • Handles overlapping constraints better")
    logger.info("  • Converges to optimal solution iteratively")
    logger.info("  • No step undermines previous adjustments")
    logger.info("\nSequential Calibration Advantages:")
    logger.info("  • Simpler to understand and debug")
    logger.info("  • Faster for non-overlapping targets")
    logger.info("  • More control over adjustment order")
    logger.info("  • Can incorporate domain knowledge in sequencing")


if __name__ == '__main__':
    main()
