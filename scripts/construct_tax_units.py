#!/usr/bin/env python3
"""
Script to construct tax units from PUMS data using the TaxUnitConstructor.
Processes data in batches to handle large datasets efficiently.
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.pums_loader import PUMSDataLoader
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

def process_batch(
    pums_loader: PUMSDataLoader,
    batch_size: int = 1000,
    batch_num: Optional[int] = None,
    total_households: Optional[int] = None
) -> Tuple[pd.DataFrame, int]:
    """Process a batch of households and return tax units.
    
    Args:
        pums_loader: Initialized PUMSDataLoader instance
        batch_size: Number of households to process in this batch
        batch_num: Current batch number (for logging)
        total_households: Total number of households (for progress tracking)
        
    Returns:
        Tuple of (tax_units_df, num_households_processed)
    """
    # Load a batch of household data
    hh_batch = pums_loader.load_households_batch(batch_size=batch_size)
    if hh_batch.empty:
        return pd.DataFrame(), 0
    
    # Get person data for these households
    person_batch = pums_loader.load_persons_for_households(hh_batch['SERIALNO'].unique())
    
    # Initialize constructor with this batch
    constructor = TaxUnitConstructor(person_batch, hh_batch)
    
    # Process this batch
    tax_units = constructor.create_rule_based_units()
    
    # Log progress
    batch_info = f"Batch {batch_num}" if batch_num is not None else "Batch"
    total_info = f" of {total_households}" if total_households else ""
    logger.info(
        f"{batch_info}: Processed {len(hh_batch):,} households, "
        f"created {len(tax_units):,} tax units{total_info}"
    )
    
    return tax_units, len(hh_batch)

def main():
    """Main function to construct tax units from PUMS data in batches."""
    try:
        logger.info("Starting batch processing of tax unit construction...")
        
        # Initialize data loader
        pums_loader = PUMSDataLoader()
        
        # Get total number of households for progress tracking
        total_households = pums_loader.get_total_households()
        logger.info(f"Found {total_households:,} total households to process")
        
        # Process in batches
        batch_size = 1000  # Adjust based on available memory
        all_tax_units = []
        batch_num = 0
        processed_households = 0
        
        while processed_households < total_households:
            batch_num += 1
            logger.info(f"\nProcessing batch {batch_num} (households {processed_households+1:,}-{min(processed_households+batch_size, total_households):,})")
            
            # Process this batch
            tax_units_batch, batch_processed = process_batch(
                pums_loader=pums_loader,
                batch_size=batch_size,
                batch_num=batch_num,
                total_households=total_households
            )
            
            if batch_processed == 0:
                break
                
            # Add to results
            if not tax_units_batch.empty:
                all_tax_units.append(tax_units_batch)
            
            # Update progress
            processed_households += batch_processed
            logger.info(f"Progress: {processed_households:,}/{total_households:,} households ({processed_households/total_households:.1%})")
        
        # Combine all batches
        if all_tax_units:
            tax_units = pd.concat(all_tax_units, ignore_index=True)
            
            # Save the results
            output_dir = Path("src/data/processed")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / "tax_units_rule_based.parquet"
            logger.info(f"Saving {len(tax_units):,} tax units to {output_path}...")
            tax_units.to_parquet(output_path, index=False)
            
            # Save a sample for testing
            sample_path = output_dir / "tax_units_sample.parquet"
            sample_size = min(1000, len(tax_units))
            tax_units.sample(n=sample_size, random_state=42).to_parquet(sample_path, index=False)
            
            logger.info(f"Successfully created {len(tax_units):,} tax units")
            logger.info(f"Sample of {sample_size} tax units saved to {sample_path}")
        else:
            logger.warning("No tax units were created")
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
