#!/usr/bin/env python3
"""
Test the joint filer fix by running tax unit construction and checking results.
"""

import pandas as pd
import logging
from pathlib import Path
from src.tax.units import TaxUnitConstructor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_joint_filer_fix():
    """Test the joint filer fix."""
    
    # Load processed data
    data_dir = Path("data/processed")
    person_file = data_dir / "pums_person_processed.parquet"
    hh_file = data_dir / "pums_household_processed.parquet"
    
    logger.info("Loading processed PUMS data...")
    person_df = pd.read_parquet(person_file)
    hh_df = pd.read_parquet(hh_file)
    
    # Create constructor
    constructor = TaxUnitConstructor(person_df, hh_df)
    
    # Construct tax units
    logger.info("Constructing tax units...")
    tax_units_df = constructor.create_rule_based_units()
    
    # Analyze results
    logger.info(f"\nTax Unit Construction Results:")
    logger.info(f"Total tax units created: {len(tax_units_df)}")
    
    # Count by filing status
    filing_status_counts = tax_units_df['filing_status'].value_counts().to_dict()
    
    total_units = len(tax_units_df)
    logger.info(f"\nFiling Status Distribution:")
    for status, count in sorted(filing_status_counts.items()):
        pct = count / total_units * 100
        logger.info(f"  {status}: {count:,} ({pct:.1f}%)")
    
    # Save results for validation
    output_dir = Path("src/data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "tax_units_rule_based.parquet"
    tax_units_df.to_parquet(output_file)
    logger.info(f"\nSaved tax units to {output_file}")
    
    # Quick validation against expected proportions
    expected_joint_pct = 36.0  # Hawaii DOTAX 2022
    actual_joint_pct = filing_status_counts.get('joint', 0) / total_units * 100
    
    logger.info(f"\nQuick Validation:")
    logger.info(f"Expected joint filers (Hawaii DOTAX 2022): {expected_joint_pct}%")
    logger.info(f"Actual joint filers: {actual_joint_pct:.1f}%")
    logger.info(f"Difference: {actual_joint_pct - expected_joint_pct:+.1f} percentage points")
    
    if actual_joint_pct > 5:  # Significant improvement from 0.3%
        logger.info("✅ Joint filer fix appears successful!")
    else:
        logger.info("❌ Joint filer count still too low")
    
    return tax_units_df

if __name__ == "__main__":
    test_joint_filer_fix()
