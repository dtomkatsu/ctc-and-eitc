#!/usr/bin/env python3
"""
Test ML validator integration during tax unit construction.
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tax.units.constructor import TaxUnitConstructor
from src.tax.units.validation.ml_validator import MLTaxUnitValidator
from src.tax.units.utils import setup_logging

# Configure logging
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_sample_data():
    """Load sample PUMS data with diverse household structures."""
    data_dir = Path("data/raw/pums")
    
    # Load person and household data using correct file names
    person_df = pd.read_csv(data_dir / "psam_p15.csv")
    hh_df = pd.read_csv(data_dir / "psam_h15.csv")
    
    # Get household sizes to ensure diversity
    hh_sizes = person_df.groupby('SERIALNO').size()
    
    # Select diverse household types (20 households total)
    diverse_hh_ids = []
    diverse_hh_ids.extend(hh_sizes[hh_sizes == 1].head(5).index.tolist())  # Single-person
    diverse_hh_ids.extend(hh_sizes[hh_sizes == 2].head(5).index.tolist())  # Couples
    diverse_hh_ids.extend(hh_sizes[hh_sizes == 3].head(5).index.tolist())  # Small families
    diverse_hh_ids.extend(hh_sizes[hh_sizes >= 4].head(5).index.tolist()) # Larger families
    
    # Filter data to selected households
    hh_df = hh_df[hh_df['SERIALNO'].isin(diverse_hh_ids)]
    person_df = person_df[person_df['SERIALNO'].isin(diverse_hh_ids)]
    
    # Log household structure
    final_sizes = person_df.groupby('SERIALNO').size()
    size_dist = final_sizes.value_counts().sort_index()
    
    logger.info(f"Sample data: {len(person_df)} persons in {len(hh_df)} households")
    logger.info("Household size distribution:")
    for size, count in size_dist.items():
        logger.info(f"  {size} persons: {count} households")
    
    return person_df, hh_df

def analyze_validation_results(tax_units):
    """Analyze and display validation results."""
    flagged_units = [tu for tu in tax_units if tu.get('validation_flags')]
    
    # Count by filing status
    status_counts = {}
    for tu in tax_units:
        status = tu.get('filing_status', 'unknown')
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Count flags by type
    flag_types = {}
    for tu in flagged_units:
        for flag in tu['validation_flags']:
            flag_type = flag.get('code', 'UNKNOWN')
            flag_types[flag_type] = flag_types.get(flag_type, 0) + 1
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("VALIDATION RESULTS")
    logger.info("="*50)
    logger.info(f"Total tax units: {len(tax_units)}")
    logger.info(f"Units with validation flags: {len(flagged_units)} ({len(flagged_units)/len(tax_units):.1%})")
    
    logger.info("\nFiling Status Distribution:")
    for status, count in sorted(status_counts.items()):
        logger.info(f"- {status}: {count} units ({count/len(tax_units):.1%})")
    
    if flag_types:
        logger.info("\nFlag Types:")
        for flag_type, count in flag_types.items():
            logger.info(f"- {flag_type}: {count} occurrences")
    
    # Show detailed examples of flagged units
    if flagged_units:
        logger.info("\n" + "-"*50)
        logger.info("EXAMPLES OF FLAGGED TAX UNITS")
        logger.info("-"*50)
        
        for i, tu in enumerate(flagged_units[:3]):  # Show up to 3 examples
            logger.info(f"\nExample {i+1}:")
            logger.info(f"- Filing Status: {tu.get('filing_status', 'unknown')}")
            logger.info(f"- Members: {tu.get('num_dependents', 0) + 1}")
            logger.info(f"- Income: ${tu.get('income', 0):,.2f}")
            
            logger.info("Validation Flags:")
            for flag in tu['validation_flags']:
                logger.info(f"  - {flag.get('code', 'UNKNOWN')}: {flag.get('message', 'No message')}")
                logger.info(f"    Confidence: {flag.get('confidence', 0):.1%}")
    
    logger.info("\n" + "="*50)

if __name__ == "__main__":
    # Load ML model
    model_path = "models/tax_unit_validator.joblib"
    if not os.path.exists(model_path):
        logger.error(f"ML model not found at {model_path}")
        sys.exit(1)
    
    # Load sample data
    logger.info("Loading sample data...")
    person_df, hh_df = load_sample_data()
    
    # Initialize tax unit constructor with ML validation
    logger.info("Initializing tax unit constructor with ML validation...")
    constructor = TaxUnitConstructor(
        person_df=person_df,
        hh_df=hh_df,
        ml_model_path=model_path,
        batch_size=100,
        num_processes=os.cpu_count() - 1 if os.cpu_count() > 1 else 1
    )
    
    # Create tax units with ML validation
    logger.info("Creating tax units with ML validation...")
    tax_units_df = constructor.create_rule_based_units()
    
    # Convert to list of tax unit dictionaries
    tax_units = tax_units_df.to_dict('records')
    
    # Analyze and display results
    analyze_validation_results(tax_units)
