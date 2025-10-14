#!/usr/bin/env python3
"""
Test IPF Calibration on PUMS Data

This script demonstrates how IPF adjusts PUMS weights to match 2022 DOTAX benchmarks.
It compares results across different calibration methods:
- No calibration (raw PUMS)
- Simple overall adjustment
- Filing status-specific adjustment
- IPF (multi-dimensional)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tax.units.soi_calibration import SOICalibrator, calibrate_to_soi_benchmarks
from tax.calibration.ipf_calibration import IPFCalibrator, create_benchmarks_from_dotax
from tax.calibration.dotax_soi_parser import DOTAXSOIParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_sample_tax_units(limit: int = None) -> pd.DataFrame:
    """Load processed tax units from PUMS data."""
    
    # Try multiple possible file locations
    possible_paths = [
        Path("data/processed/tax_units_rule_based.parquet"),
        Path("data/processed/tax_units_with_geography.parquet"),
        Path("data/processed/tax_units_soi_calibrated.parquet"),
        Path("src/data/processed/tax_units_rule_based.parquet"),
    ]
    
    tax_units_path = None
    for path in possible_paths:
        if path.exists():
            tax_units_path = path
            break
    
    if tax_units_path is None:
        logger.error("No tax units file found. Tried:")
        for path in possible_paths:
            logger.error(f"  - {path}")
        logger.info("Please run tax unit construction first.")
        sys.exit(1)
    
    logger.info(f"Loading tax units from {tax_units_path}...")
    tax_units = pd.read_parquet(tax_units_path)
    
    if limit:
        tax_units = tax_units.head(limit)
    
    logger.info(f"Loaded {len(tax_units):,} tax units")
    
    # Ensure required columns exist
    if 'weight' not in tax_units.columns:
        if 'hh_weight' in tax_units.columns:
            tax_units['weight'] = tax_units['hh_weight']
        elif 'PWGTP' in tax_units.columns:
            tax_units['weight'] = tax_units['PWGTP']
        elif 'WGTP' in tax_units.columns:
            tax_units['weight'] = tax_units['WGTP']
        else:
            logger.warning("No weight column found, using uniform weights")
            tax_units['weight'] = 1.0
    
    return tax_units


def compare_calibration_methods(tax_units: pd.DataFrame):
    """Compare different calibration methods."""
    
    logger.info("\n" + "="*80)
    logger.info("COMPARING CALIBRATION METHODS")
    logger.info("="*80)
    
    # Load DOTAX benchmarks
    logger.info("\nLoading DOTAX benchmarks...")
    parser = DOTAXSOIParser()
    resident_totals = parser.get_resident_totals()
    
    dotax_benchmarks = {
        'total_returns': resident_totals['total_returns']
    }
    
    logger.info(f"DOTAX target: {dotax_benchmarks['total_returns']:,} returns")
    
    # Original (uncalibrated)
    original_total = tax_units['weight'].sum()
    logger.info(f"\nOriginal PUMS total: {original_total:,.0f}")
    logger.info(f"Overcount: {original_total - dotax_benchmarks['total_returns']:,.0f} ({(original_total/dotax_benchmarks['total_returns'] - 1)*100:+.1f}%)")
    
    # Test each method
    methods = ['overall', 'filing_status', 'income_bracket', 'ipf']
    results = {}
    
    for method in methods:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing method: {method.upper()}")
        logger.info(f"{'='*80}")
        
        try:
            calibrated = calibrate_to_soi_benchmarks(
                tax_units=tax_units,
                dotax_benchmarks=dotax_benchmarks,
                method=method
            )
            
            calibrated_total = calibrated['weight_calibrated'].sum()
            
            results[method] = {
                'total': calibrated_total,
                'error': calibrated_total - dotax_benchmarks['total_returns'],
                'error_pct': (calibrated_total / dotax_benchmarks['total_returns'] - 1) * 100,
                'dataframe': calibrated
            }
            
            logger.info(f"\n{method.upper()} Results:")
            logger.info(f"  Calibrated total: {calibrated_total:,.0f}")
            logger.info(f"  Error: {results[method]['error']:+,.0f} ({results[method]['error_pct']:+.2f}%)")
            
        except Exception as e:
            logger.error(f"Error with method {method}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary comparison
    logger.info("\n" + "="*80)
    logger.info("SUMMARY COMPARISON")
    logger.info("="*80)
    logger.info(f"\n{'Method':<20} {'Total':>15} {'Error':>15} {'Error %':>10}")
    logger.info("-"*80)
    logger.info(f"{'Target (DOTAX)':<20} {dotax_benchmarks['total_returns']:>15,} {0:>15} {0:>10.2f}%")
    logger.info(f"{'Original (PUMS)':<20} {original_total:>15,.0f} {original_total - dotax_benchmarks['total_returns']:>+15,.0f} {(original_total/dotax_benchmarks['total_returns'] - 1)*100:>+10.2f}%")
    
    for method, result in results.items():
        logger.info(
            f"{method.upper():<20} {result['total']:>15,.0f} "
            f"{result['error']:>+15,.0f} {result['error_pct']:>+10.2f}%"
        )
    
    logger.info("="*80)
    
    return results


def analyze_ipf_convergence(tax_units: pd.DataFrame):
    """Analyze IPF convergence in detail."""
    
    logger.info("\n" + "="*80)
    logger.info("IPF CONVERGENCE ANALYSIS")
    logger.info("="*80)
    
    # Create benchmarks
    benchmarks = create_benchmarks_from_dotax()
    
    # Initialize IPF calibrator
    ipf_calibrator = IPFCalibrator(
        benchmarks=benchmarks,
        max_iterations=100,
        tolerance=0.001,
        damping_factor=0.7
    )
    
    # Run calibration
    calibrated = ipf_calibrator.calibrate(tax_units)
    
    # Plot convergence history
    logger.info("\nConvergence History:")
    logger.info(f"{'Iteration':<12} {'Max Rel Change':<18} {'Total Weight':<15}")
    logger.info("-"*50)
    
    for entry in ipf_calibrator.convergence_history:
        logger.info(
            f"{entry['iteration']:<12} {entry['max_rel_change']:<18.6f} {entry['total_weight']:<15,.0f}"
        )
    
    return calibrated, ipf_calibrator


def main():
    """Main execution."""
    
    logger.info("="*80)
    logger.info("IPF CALIBRATION TEST")
    logger.info("="*80)
    
    # Load tax units
    tax_units = load_sample_tax_units()
    
    # Compare calibration methods
    results = compare_calibration_methods(tax_units)
    
    # Analyze IPF convergence
    if 'ipf' in results:
        logger.info("\n\nRunning detailed IPF analysis...")
        calibrated, ipf_calibrator = analyze_ipf_convergence(tax_units)
        
        # Save calibrated results
        output_path = Path("data/processed/tax_units_ipf_calibrated.parquet")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        calibrated.to_parquet(output_path)
        logger.info(f"\nSaved IPF-calibrated tax units to {output_path}")
    
    logger.info("\n" + "="*80)
    logger.info("TEST COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
