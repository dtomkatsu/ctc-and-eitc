#!/usr/bin/env python3
"""
Simple IPF Calibration Demo

Demonstrates how IPF adjusts 5-year PUMS weights (2018-2022 average) 
to match 2022 DOTAX benchmarks.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tax.calibration.ipf_calibration import IPFCalibrator, create_benchmarks_from_dotax
from tax.calibration.dotax_soi_parser import DOTAXSOIParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_synthetic_pums_data() -> pd.DataFrame:
    """
    Create synthetic PUMS-like data for demonstration.
    
    This simulates the 5-year PUMS data structure with filing statuses
    and income that needs to be reweighted to 2022.
    """
    np.random.seed(42)
    
    # Create 1000 synthetic tax units
    n = 1000
    
    # Filing status distribution (roughly matching PUMS patterns)
    filing_statuses = np.random.choice(
        ['single', 'joint', 'hoh', 'mfs'],
        size=n,
        p=[0.55, 0.28, 0.12, 0.05]  # PUMS tends to overcount singles
    )
    
    # Income distribution (log-normal)
    incomes = np.random.lognormal(mean=10.8, sigma=0.8, size=n)
    
    # PUMS weights (representing 5-year average, needs adjustment to 2022)
    # These sum to more than the 2022 target
    weights = np.random.uniform(800, 1200, size=n)
    
    df = pd.DataFrame({
        'tax_unit_id': range(n),
        'filing_status': filing_statuses,
        'income': incomes,
        'weight': weights
    })
    
    logger.info(f"Created {len(df)} synthetic tax units")
    logger.info(f"Total weight (5-year average): {df['weight'].sum():,.0f}")
    
    return df


def main():
    """Main demonstration."""
    
    print("\n" + "="*80)
    print("IPF CALIBRATION DEMONSTRATION")
    print("="*80)
    print("\nThis demonstrates how IPF adjusts 5-year PUMS weights (2018-2022)")
    print("to represent 2022 specifically by matching DOTAX benchmarks.")
    print("="*80)
    
    # Step 1: Load DOTAX 2022 benchmarks
    print("\n[1] Loading DOTAX 2022 Benchmarks...")
    print("-" * 80)
    
    try:
        benchmarks = create_benchmarks_from_dotax()
        
        print(f"\nDOTAX 2022 Targets:")
        print(f"  Total returns: {benchmarks['total_returns']:,}")
        print(f"\n  Filing Status Distribution:")
        for status, count in benchmarks['filing_status_distribution'].items():
            pct = (count / benchmarks['total_returns']) * 100
            print(f"    {status:<10}: {count:>7,} ({pct:>5.1f}%)")
        
        print(f"\n  Income Brackets: {len(benchmarks['income_bracket_distribution'])} brackets")
        
    except Exception as e:
        logger.error(f"Could not load DOTAX benchmarks: {e}")
        logger.info("Using simplified benchmarks for demo...")
        
        # Fallback to simplified benchmarks
        benchmarks = {
            'total_returns': 635117,
            'filing_status_distribution': {
                'single': 323910,
                'joint': 228642,
                'hoh': 60971,
                'mfs': 21594
            }
        }
        
        print(f"\nSimplified Benchmarks (2022 targets):")
        print(f"  Total returns: {benchmarks['total_returns']:,}")
        for status, count in benchmarks['filing_status_distribution'].items():
            pct = (count / benchmarks['total_returns']) * 100
            print(f"    {status:<10}: {count:>7,} ({pct:>5.1f}%)")
    
    # Step 2: Create synthetic PUMS data
    print("\n[2] Creating Synthetic PUMS Data (5-year average 2018-2022)...")
    print("-" * 80)
    
    pums_data = create_synthetic_pums_data()
    
    print(f"\nBefore Calibration:")
    print(f"  Total weighted units: {pums_data['weight'].sum():,.0f}")
    print(f"  Target (2022):        {benchmarks['total_returns']:,}")
    print(f"  Overcount:            {pums_data['weight'].sum() - benchmarks['total_returns']:+,.0f}")
    
    # Filing status distribution before
    print(f"\n  Filing Status Distribution (Before):")
    for status in ['single', 'joint', 'hoh', 'mfs']:
        count = pums_data[pums_data['filing_status'] == status]['weight'].sum()
        pct = (count / pums_data['weight'].sum()) * 100
        if status in benchmarks['filing_status_distribution']:
            target_pct = (benchmarks['filing_status_distribution'][status] / benchmarks['total_returns']) * 100
            print(f"    {status:<10}: {count:>10,.0f} ({pct:>5.1f}%) | Target: {target_pct:>5.1f}%")
        else:
            print(f"    {status:<10}: {count:>10,.0f} ({pct:>5.1f}%) | Target: N/A")
    
    # Step 3: Apply IPF calibration
    print("\n[3] Applying IPF Calibration...")
    print("-" * 80)
    
    ipf_calibrator = IPFCalibrator(
        benchmarks=benchmarks,
        max_iterations=50,
        tolerance=0.001,
        damping_factor=0.7
    )
    
    calibrated = ipf_calibrator.calibrate(pums_data)
    
    # Step 4: Show results
    print("\n[4] Results After IPF Calibration...")
    print("-" * 80)
    
    print(f"\nAfter Calibration:")
    print(f"  Total weighted units: {calibrated['weight_calibrated'].sum():,.0f}")
    print(f"  Target (2022):        {benchmarks['total_returns']:,}")
    print(f"  Difference:           {calibrated['weight_calibrated'].sum() - benchmarks['total_returns']:+,.0f}")
    print(f"  Error:                {((calibrated['weight_calibrated'].sum() / benchmarks['total_returns']) - 1) * 100:+.2f}%")
    
    # Filing status distribution after
    print(f"\n  Filing Status Distribution (After):")
    print(f"  {'Status':<12} {'Calibrated':>12} {'Target':>12} {'Diff %':>10}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
    
    for status in ['single', 'joint', 'hoh', 'mfs']:
        actual = calibrated[calibrated['filing_status'] == status]['weight_calibrated'].sum()
        if status in benchmarks['filing_status_distribution']:
            target = benchmarks['filing_status_distribution'][status]
            diff_pct = ((actual / target) - 1) * 100
            print(f"  {status:<12} {actual:>12,.0f} {target:>12,} {diff_pct:>+9.2f}%")
        else:
            print(f"  {status:<12} {actual:>12,.0f} {'N/A':>12} {'N/A':>10}")
    
    # Convergence info
    print(f"\n  Convergence:")
    print(f"    Iterations: {calibrated['calibration_iterations'].iloc[0]}")
    print(f"    Converged:  {calibrated['calibration_converged'].iloc[0]}")
    
    # Step 5: Key insight
    print("\n[5] Key Insight...")
    print("-" * 80)
    print("\n✓ IPF successfully adjusted PUMS weights to match 2022 benchmarks!")
    print("\nWhat changed:")
    print("  • PUMS records stayed the same (same households, same characteristics)")
    print("  • Only the weights changed to match 2022 totals")
    print("  • Result: 5-year PUMS data now represents 2022 specifically")
    
    print("\nWhy this matters:")
    print("  • 5-year PUMS has larger sample size (more reliable)")
    print("  • IPF adjusts it to match 2022 administrative data (more accurate)")
    print("  • Best of both worlds: large sample + accurate totals")
    
    # Save results
    output_path = Path("data/processed/demo_ipf_calibrated.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    calibrated.to_parquet(output_path)
    print(f"\n✓ Saved calibrated data to: {output_path}")
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
