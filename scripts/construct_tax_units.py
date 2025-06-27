#!/usr/bin/env python3
"""
Script to construct tax units from PUMS data using the TaxUnitConstructor.
"""
import os
import sys
import logging
import pandas as pd
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.pums_loader import load_pums_data
from src.tax.units.constructor import TaxUnitConstructor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main function to construct tax units from PUMS data."""
    try:
        logger.info("Starting tax unit construction...")
        
        # Load PUMS data
        logger.info("Loading PUMS data...")
        person_df, hh_df = load_pums_data()
        
        # Initialize the constructor
        logger.info("Initializing TaxUnitConstructor...")
        constructor = TaxUnitConstructor(person_df, hh_df)
        
        # Create tax units
        logger.info("Creating tax units...")
        tax_units = constructor.create_rule_based_units()
        
        # Save the results
        output_dir = Path("src/data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "tax_units_rule_based.parquet"
        logger.info(f"Saving tax units to {output_path}...")
        tax_units.to_parquet(output_path, index=False)
        
        logger.info(f"Successfully created {len(tax_units):,} tax units")
        
    except Exception as e:
        logger.error(f"Error constructing tax units: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
