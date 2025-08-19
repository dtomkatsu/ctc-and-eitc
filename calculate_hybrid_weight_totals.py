#!/usr/bin/env python3
"""
Calculate total hybrid weights for tax units.
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
STATE_CODE = 15  # Hawaii
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
            return pd.DataFrame()
            
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(tax_units)
        
        # Calculate summary statistics
        stats = {
            'total_hybrid_weight': df['weight'].sum(),
            'total_household_weight': df['hh_weight'].sum(),
            'total_person_weight': df['person_weight_sum'].sum(),
            'num_units': len(df)
        }
        
        # Add filing status breakdown
        for status in df['filing_status'].unique():
            status_df = df[df['filing_status'] == status]
            stats[f'weight_{status}'] = status_df['weight'].sum()
            stats[f'count_{status}'] = len(status_df)
            
        return pd.DataFrame([stats])
        
    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        return pd.DataFrame()

def main():
    """Main function to process all data and calculate hybrid weight totals."""
    logger.info("Starting hybrid weight calculation...")
    
    # Read household data
    logger.info("Reading household data...")
    households = pd.read_csv(HOUSEHOLD_FILE, dtype={'SERIALNO': str})
    
    # Read person data in chunks
    logger.info("Processing person data in chunks...")
    results = []
    
    # Process in batches to manage memory
    for i, chunk in enumerate(pd.read_csv(PERSON_FILE, chunksize=10000, dtype={'SERIALNO': str})):
        # Merge with household data to get state filter
        chunk = chunk.merge(households[['SERIALNO', 'ST']], on='SERIALNO', how='left')
        
        # Filter for Hawaii
        chunk = chunk[chunk['ST'] == STATE_CODE]
        
        if len(chunk) == 0:
            continue
            
        # Get unique household IDs in this chunk
        hh_ids = chunk['SERIALNO'].unique()
        
        # Get corresponding household data
        hh_batch = households[households['SERIALNO'].isin(hh_ids)].copy()
        
        # Process batch
        logger.info(f"Processing batch {i+1} with {len(hh_batch)} households...")
        batch_result = process_batch(chunk, hh_batch)
        
        if not batch_result.empty:
            results.append(batch_result)
    
    # Combine results
    if results:
        final_results = pd.concat(results, ignore_index=True)
        
        # Calculate totals
        total_hybrid = final_results['total_hybrid_weight'].sum()
        total_hh = final_results['total_household_weight'].sum()
        total_persons = final_results['total_person_weight'].sum()
        total_units = final_results['num_units'].sum()
        
        # Calculate filing status breakdown
        status_weights = {}
        status_counts = {}
        for col in final_results.columns:
            if col.startswith('weight_'):
                status = col.replace('weight_', '')
                status_weights[status] = final_results[col].sum()
            elif col.startswith('count_'):
                status = col.replace('count_', '')
                status_counts[status] = int(final_results[col].sum())
        
        # Print summary
        print("\n=== Hybrid Weight Summary ===")
        print(f"Total Hybrid Weight: {total_hybrid:,.2f}")
        print(f"Total Household Weight: {total_hh:,.2f}")
        print(f"Total Person Weight: {total_persons:,.2f}")
        print(f"Total Tax Units: {total_units:,}")
        
        print("\n=== By Filing Status ===")
        for status in sorted(status_weights.keys()):
            print(f"{status.upper()}:")
            print(f"  Weight: {status_weights[status]:,.2f}")
            print(f"  Count:  {status_counts.get(status, 0):,}")
            
    else:
        logger.warning("No tax units were generated.")
        
    logger.info("Hybrid weight calculation complete.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("An error occurred during hybrid weight calculation")
        raise
