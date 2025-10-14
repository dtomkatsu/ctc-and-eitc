#!/usr/bin/env python3
"""
Test script for multiprocessing implementation in TaxUnitConstructor.

This script loads a sample of PUMS data and tests the performance of tax unit
construction with different numbers of worker processes.
"""

import os
import sys
import time
import logging
import argparse
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.pums_loader import PUMSDataLoader
from src.tax.units.constructor import TaxUnitConstructor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('tax_unit_construction.log')
    ]
)
logger = logging.getLogger(__name__)

def load_sample_data(sample_size=1000):
    """Load a sample of PUMS data for testing."""
    logger.info(f"Loading sample of {sample_size} households...")
    
    # Initialize data loader
    data_dir = os.path.join(project_root, 'data/raw/pums')
    loader = PUMSDataLoader(data_dir)
    
    # Load the data using the standard method first to ensure everything is set up
    logger.info("Loading full dataset...")
    try:
        # Try to load a small sample first to check if data is accessible
        test_households = loader.load_households_batch(batch_size=10, offset=0)
        if test_households.empty:
            logger.error("No household data found. Please check if PUMS data files exist in the data directory.")
            return None, None
            
        # Get total number of households
        total_households = loader.get_total_households()
        logger.info(f"Found {total_households} total households in dataset")
        
        # Calculate sample size (don't exceed total)
        sample_size = min(sample_size, total_households)
        
        # Load sample
        logger.info(f"Loading {sample_size} households...")
        households = loader.load_households_batch(batch_size=sample_size, offset=0)
        
        if households.empty:
            logger.error("No household data loaded. Check if the data files are in the correct format.")
            return None, None
            
        logger.info(f"Loading person data for {len(households)} households...")
        person_df = loader.load_persons_for_households(households['SERIALNO'].unique())
        
        if person_df.empty:
            logger.error("No person data loaded. Check if the data files are in the correct format.")
            return None, None
        
        logger.info(f"Loaded {len(households)} households and {len(person_df)} persons")
        return person_df, households
        
    except Exception as e:
        logger.error(f"Error loading sample data: {str(e)}", exc_info=True)
        return None, None

def test_multiprocessing(person_df, hh_df, n_jobs_list):
    """Test tax unit construction with different numbers of worker processes."""
    results = []
    
    for n_jobs in n_jobs_list:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing with n_jobs = {n_jobs}")
        logger.info(f"{'='*50}")
        
        # Create a fresh constructor for each test
        constructor = TaxUnitConstructor(
            person_df=person_df.copy(),
            hh_df=hh_df.copy()
        )
        
        # Time the operation
        start_time = time.time()
        
        try:
            tax_units = constructor.create_rule_based_units(n_jobs=n_jobs)
            elapsed = time.time() - start_time
            
            # Collect results
            result = {
                'n_jobs': n_jobs,
                'time_seconds': round(elapsed, 2),
                'n_tax_units': len(tax_units) if tax_units is not None else 0,
                'status': 'success'
            }
            logger.info(f"Completed in {elapsed:.2f} seconds. Created {result['n_tax_units']} tax units.")
            
        except Exception as e:
            elapsed = time.time() - start_time
            result = {
                'n_jobs': n_jobs,
                'time_seconds': round(elapsed, 2),
                'n_tax_units': 0,
                'status': f'error: {str(e)}'
            }
            logger.error(f"Failed after {elapsed:.2f} seconds: {str(e)}", exc_info=True)
        
        results.append(result)
    
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description='Test multiprocessing in TaxUnitConstructor')
    parser.add_argument('--sample-size', type=int, default=1000,
                       help='Number of households to sample (default: 1000)')
    parser.add_argument('--processes', type=int, nargs='+', default=[1, 2, 4, -1],
                       help='List of n_jobs values to test (default: [1, 2, 4, -1])')
    
    args = parser.parse_args()
    
    try:
        # Load sample data
        person_df, hh_df = load_sample_data(args.sample_size)
        
        # Run tests
        results = test_multiprocessing(person_df, hh_df, args.processes)
        
        # Print results
        print("\nTest Results:")
        print("-" * 50)
        print(results[['n_jobs', 'time_seconds', 'n_tax_units', 'status']].to_string(index=False))
        
        # Save results to CSV
        results.to_csv('multiprocessing_test_results.csv', index=False)
        logger.info("Test results saved to multiprocessing_test_results.csv")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
