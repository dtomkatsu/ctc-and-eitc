#!/usr/bin/env python3
"""
Calculate hybrid weights for tax units (simplified version).
"""
import logging
import pandas as pd
from pathlib import Path
from src.tax.units.constructor import TaxUnitConstructor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# File paths
DATA_DIR = Path('data/raw/pums')
PERSON_FILE = DATA_DIR / 'psam_p15.csv'
HOUSEHOLD_FILE = DATA_DIR / 'psam_h15.csv'

# Configuration
BATCH_SIZE = 1000  # Process households in batches to manage memory

def process_batch(persons_batch, households_batch):
    """Process a batch of households and return tax units with hybrid weights."""
    try:
        constructor = TaxUnitConstructor()
        tax_units = constructor.create_rule_based_units(
            persons_batch, 
            households_batch
        )
        
        if not tax_units:
            return {
                'total_hybrid_weight': 0,
                'total_household_weight': 0,
                'total_person_weight': 0,
                'num_units': 0,
                'status_counts': {}
            }
            
        # Calculate summary statistics
        total_hybrid = sum(tu.get('weight', 0) for tu in tax_units)
        total_hh = sum(tu.get('hh_weight', 0) for tu in tax_units)
        total_person = sum(tu.get('person_weight_sum', 0) for tu in tax_units)
        
        # Count by status
        status_counts = {}
        for tu in tax_units:
            status = tu.get('filing_status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            'total_hybrid_weight': total_hybrid,
            'total_household_weight': total_hh,
            'total_person_weight': total_person,
            'num_units': len(tax_units),
            'status_counts': status_counts
        }
        
    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        return {
            'total_hybrid_weight': 0,
            'total_household_weight': 0,
            'total_person_weight': 0,
            'num_units': 0,
            'status_counts': {}
        }

def main():
    """Main function to process all data and calculate hybrid weight totals."""
    logger.info("Starting hybrid weight calculation...")
    
    # Initialize totals
    totals = {
        'total_hybrid_weight': 0,
        'total_household_weight': 0,
        'total_person_weight': 0,
        'num_units': 0,
        'status_counts': {}
    }
    
    # Read household data
    logger.info("Reading household data...")
    households = pd.read_csv(HOUSEHOLD_FILE, dtype={'SERIALNO': str})
    
    # Process in batches
    logger.info("Processing data in batches...")
    for i, chunk in enumerate(pd.read_csv(PERSON_FILE, chunksize=10000, dtype={'SERIALNO': str})):
        # Get unique household IDs in this chunk
        hh_ids = chunk['SERIALNO'].unique()
        
        # Get corresponding household data
        hh_batch = households[households['SERIALNO'].isin(hh_ids)].copy()
        
        # Skip if no households in this chunk
        if len(hh_batch) == 0:
            continue
            
        # Process batch
        logger.info(f"Processing batch {i+1} with {len(hh_batch)} households...")
        batch_result = process_batch(chunk, hh_batch)
        
        # Update totals
        totals['total_hybrid_weight'] += batch_result['total_hybrid_weight']
        totals['total_household_weight'] += batch_result['total_household_weight']
        totals['total_person_weight'] += batch_result['total_person_weight']
        totals['num_units'] += batch_result['num_units']
        
        # Update status counts
        for status, count in batch_result['status_counts'].items():
            totals['status_counts'][status] = totals['status_counts'].get(status, 0) + count
    
    # Print summary
    print("\n=== Hybrid Weight Summary ===")
    print(f"Total Hybrid Weight: {totals['total_hybrid_weight']:,.2f}")
    print(f"Total Household Weight: {totals['total_household_weight']:,.2f}")
    print(f"Total Person Weight: {totals['total_person_weight']:,.2f}")
    print(f"Total Tax Units: {totals['num_units']:,}")
    
    print("\n=== By Filing Status ===")
    for status, count in sorted(totals['status_counts'].items()):
        print(f"{status}: {count:,} units")
    
    logger.info("Hybrid weight calculation complete.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("An error occurred during hybrid weight calculation")
        raise
