#!/usr/bin/env python3
"""
Debug batch processing to find where household duplication is occurring.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import logging
from src.data.pums_loader import PUMSDataLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def debug_batch_duplication():
    print("Investigating batch processing duplication...")
    
    # Initialize data loader
    pums_loader = PUMSDataLoader()
    
    # Get total number of households
    total_households = pums_loader.get_total_households()
    print(f"Total households reported: {total_households:,}")
    
    # Load all data at once to check for duplicates
    person_df, hh_df = pums_loader.load_data()
    
    print(f"Loaded person data: {len(person_df):,} persons")
    print(f"Loaded household data: {len(hh_df):,} households")
    
    # Check for duplicate households in person data
    person_households = person_df['SERIALNO'].unique()
    print(f"Unique households in person data: {len(person_households):,}")
    
    # Check for duplicate households in household data
    hh_households = hh_df['SERIALNO'].unique()
    print(f"Unique households in household data: {len(hh_households):,}")
    
    # Check if there are duplicate SERIALNO values in person_df
    person_serialno_counts = person_df['SERIALNO'].value_counts()
    max_persons_per_hh = person_serialno_counts.max()
    print(f"Max persons per household: {max_persons_per_hh}")
    
    # Check if there are duplicate SERIALNO values in hh_df
    hh_duplicates = hh_df['SERIALNO'].duplicated().sum()
    print(f"Duplicate households in hh_df: {hh_duplicates}")
    
    if hh_duplicates > 0:
        print("Found duplicate households in household data!")
        duplicate_households = hh_df[hh_df['SERIALNO'].duplicated(keep=False)]['SERIALNO'].unique()
        print(f"Duplicate household IDs: {duplicate_households[:10]}...")  # Show first 10
    
    # Test batch loading specifically
    print(f"\nTesting batch loading...")
    
    # Reset the loader and test batch processing
    pums_loader = PUMSDataLoader()
    
    batch_size = 1000
    batch_num = 0
    all_households_seen = set()
    duplicate_households_found = set()
    
    while True:
        batch_num += 1
        print(f"Loading batch {batch_num}...")
        
        # Load a batch of household data
        hh_batch = pums_loader.load_households_batch(batch_size=batch_size)
        if hh_batch.empty:
            print("No more batches to process")
            break
        
        print(f"  Batch {batch_num}: {len(hh_batch)} households")
        
        # Check for duplicates within this batch
        batch_households = set(hh_batch['SERIALNO'].unique())
        batch_duplicates = len(hh_batch) - len(batch_households)
        if batch_duplicates > 0:
            print(f"  WARNING: {batch_duplicates} duplicate households within batch {batch_num}")
        
        # Check for duplicates across batches
        overlapping = batch_households.intersection(all_households_seen)
        if overlapping:
            print(f"  WARNING: {len(overlapping)} households already seen in previous batches")
            duplicate_households_found.update(overlapping)
            print(f"    Duplicate households: {list(overlapping)[:5]}...")  # Show first 5
        
        all_households_seen.update(batch_households)
        
        # Stop after a few batches for testing
        if batch_num >= 5:
            print("Stopping after 5 batches for testing...")
            break
    
    print(f"\nSummary:")
    print(f"Total unique households seen across batches: {len(all_households_seen):,}")
    print(f"Households that appeared in multiple batches: {len(duplicate_households_found):,}")
    
    if duplicate_households_found:
        print("FOUND THE BUG: Households are being duplicated across batches!")
        print(f"Example duplicate households: {list(duplicate_households_found)[:10]}")

if __name__ == "__main__":
    debug_batch_duplication()
