#!/usr/bin/env python3
"""
Script to check PINCP values in PUMS data.
"""
import pandas as pd
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Load person data
    person_file = Path("data/raw/pums/psam_p15.csv")
    logger.info(f"Loading person data from {person_file}")
    
    # Only load necessary columns to save memory
    usecols = ['SERIALNO', 'SPORDER', 'PINCP', 'ADJINC']
    
    # Load data in chunks to handle large files
    chunks = []
    for chunk in pd.read_csv(person_file, usecols=usecols, chunksize=10000):
        chunks.append(chunk)
        if len(chunks) > 10:  # Only load first 100,000 rows
            break
    
    df = pd.concat(chunks, ignore_index=True)
    
    # Calculate adjusted income
    df['ADJINC'] = df['ADJINC'].fillna(1_000_000) / 1_000_000
    df['ADJ_PINCP'] = df['PINCP'] * df['ADJINC']
    
    # Print summary statistics
    print("\n=== PINCP Summary ===")
    print(df['PINCP'].describe())
    
    print("\n=== ADJINC Summary ===")
    print(df['ADJINC'].describe())
    
    print("\n=== Adjusted PINCP Summary ===")
    print(df['ADJ_PINCP'].describe())
    
    # Plot distribution of PINCP
    plt.figure(figsize=(12, 6))
    
    # Plot raw PINCP (log scale)
    plt.subplot(1, 2, 1)
    df['PINCP'].clip(lower=1).hist(bins=50, log=True)
    plt.title('Distribution of PINCP (log scale)')
    plt.xlabel('PINCP')
    plt.ylabel('Count')
    
    # Plot adjusted PINCP (log scale)
    plt.subplot(1, 2, 2)
    df['ADJ_PINCP'].clip(lower=1).hist(bins=50, log=True)
    plt.title('Distribution of Adjusted PINCP (log scale)')
    plt.xlabel('Adjusted PINCP')
    
    plt.tight_layout()
    plt.savefig('pincp_distribution.png')
    print("\nSaved PINCP distribution plot to pincp_distribution.png")
    
    # Check for negative values
    print("\n=== Negative Incomes ===")
    neg_mask = df['ADJ_PINCP'] < 0
    print(f"Number of negative adjusted incomes: {neg_mask.sum()}")
    if neg_mask.any():
        print("\nSample of negative incomes:")
        print(df[neg_mask].head())

if __name__ == "__main__":
    main()
