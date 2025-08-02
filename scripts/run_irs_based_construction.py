"""
Run tax unit construction with IRS-based filing status logic and compare to SOI.
"""
import sys
import os
import logging
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.pums_loader import PUMSDataLoader
from src.tax.units.constructor import TaxUnitConstructor
from src.tax.units.utils import setup_logging

# Set up logging
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting tax unit construction with IRS-based filing status...")
    
    # Initialize data loader
    data_loader = PUMSDataLoader()
    
    # Load data
    logger.info("Loading PUMS data...")
    person_df, hh_df = data_loader.load_data()
    
    # Initialize tax unit constructor
    logger.info("Initializing tax unit constructor...")
    constructor = TaxUnitConstructor(person_df, hh_df)
    
    # Run tax unit construction
    logger.info("Running tax unit construction...")
    tax_units = constructor.create_rule_based_units(parallel=True)
    
    # Save results
    output_dir = Path("src/data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "tax_units_irs_based.parquet"
    tax_units.to_parquet(output_file)
    logger.info(f"Saved tax units to {output_file}")
    
    # Print summary
    total_units = len(tax_units)
    status_counts = tax_units['filing_status'].value_counts()
    status_pct = tax_units['filing_status'].value_counts(normalize=True) * 100
    
    print("\nTax Unit Construction Results:")
    print("=" * 50)
    print(f"Total tax units: {total_units:,}")
    print("\nFiling Status Distribution:")
    for status, count in status_counts.items():
        print(f"  {status}: {count:,} ({status_pct[status]:.1f}%)")
    
    # Compare to SOI targets
    print("\nComparison to SOI Benchmarks:")
    print("-" * 50)
    print(f"{'Status':<25} {'PUMS Count':>12} {'PUMS %':>10} {'SOI Target %':>12}")
    print("-" * 60)
    
    # Map actual filing status names to SOI targets
    status_mapping = {
        'single': 'single',
        'married_filing_jointly': 'joint',
        'head_of_household': 'head_of_household', 
        'married_filing_separately': 'married_filing_separate'
    }
    
    soi_targets = {
        'single': 51.0,
        'joint': 36.0,
        'head_of_household': 9.6,
        'married_filing_separate': 3.4
    }
    
    for actual_status, soi_status in status_mapping.items():
        count = status_counts.get(actual_status, 0)
        pct = status_pct.get(actual_status, 0.0)
        target = soi_targets[soi_status]
        gap = pct - target
        gap_str = f"({gap:+.1f}pp)" if gap != 0 else ""
        print(f"{soi_status:<25} {count:>12,} {pct:>9.1f}% {target:>11.1f}% {gap_str}")
    
    # Calculate joint filer capture rate
    total_married = len(person_df[person_df['MAR'] == 1]) / 2  # Approximate number of married couples
    joint_count = status_counts.get('married_filing_jointly', 0)
    joint_capture_rate = (joint_count / total_married) * 100 if total_married > 0 else 0
    
    print(f"\nJoint Filer Capture Rate: {joint_capture_rate:.1f}% of married couples")
    
    return tax_units

if __name__ == "__main__":
    tax_units = main()
