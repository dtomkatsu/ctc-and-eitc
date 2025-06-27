#!/usr/bin/env python3
"""
Script to verify PUMS income and ADJINC values.
"""
import pandas as pd
import logging
from pathlib import Path
import numpy as np

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
    usecols = ['SERIALNO', 'SPORDER', 'PINCP', 'ADJINC', 'WAGP', 'SEMP', 'INTP', 'RETP', 'SSP', 'SSIP', 'PAP', 'OIP']
    
    # Load first 1000 rows for testing
    df = pd.read_csv(person_file, usecols=usecols, nrows=1000)
    
    # Calculate total income from components
    income_components = ['WAGP', 'SEMP', 'INTP', 'RETP', 'SSP', 'SSIP', 'PAP', 'OIP']
    df['sum_components'] = df[income_components].sum(axis=1)
    
    # Calculate ADJINC factor
    df['adjinc_factor'] = df['ADJINC'] / 1_000_000
    
    # Calculate adjusted income
    df['adjusted_income'] = df['PINCP'] * df['adjinc_factor']
    
    # Print summary statistics
    print("\n=== PINCP Summary ===")
    print(df['PINCP'].describe())
    
    print("\n=== ADJINC Values ===")
    print(df['ADJINC'].describe())
    print("\nADJINC value counts:")
    print(df['ADJINC'].value_counts().head())
    
    print("\n=== ADJINC Factor (ADJINC/1,000,000) ===")
    print(df['adjinc_factor'].describe())
    
    print("\n=== Adjusted Income (PINCP * ADJINC/1,000,000) ===")
    print(df['adjusted_income'].describe())
    
    # Print first 10 rows with income > 0
    print("\n=== Sample Rows with Income > 0 ===")
    sample = df[df['PINCP'] > 0].head(10)
    print(sample[['SERIALNO', 'SPORDER', 'PINCP', 'ADJINC', 'adjinc_factor', 'adjusted_income']])
    
    # Check for discrepancies between PINCP and sum of components
    print("\n=== Rows where |PINCP - sum_components| > $1 ===")
    diff = df[abs(df['PINCP'] - df['sum_components']) > 1]
    if not diff.empty:
        print(diff[['SERIALNO', 'SPORDER', 'PINCP', 'sum_components', 'ADJINC', 'adjusted_income']])
    else:
        print("No significant discrepancies found")

if __name__ == "__main__":
    main()
