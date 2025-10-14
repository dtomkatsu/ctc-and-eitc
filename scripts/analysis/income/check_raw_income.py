#!/usr/bin/env python3
"""
Script to check raw income values in PUMS data.
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
    usecols = [
        'SERIALNO', 'SPORDER', 'AGEP', 'PINCP', 'ADJINC', 'WAGP', 'SEMP',
        'INTP', 'RETP', 'SSP', 'SSIP', 'PAP', 'OIP', 'RELSHIPP', 'SEX', 'MAR'
    ]
    
    # Load first 1000 rows for testing
    df = pd.read_csv(person_file, usecols=usecols, nrows=1000)
    
    # Calculate total income from components
    income_components = ['WAGP', 'SEMP', 'INTP', 'RETP', 'SSP', 'SSIP', 'PAP', 'OIP']
    df['sum_components'] = df[income_components].sum(axis=1)
    
    # Apply ADJINC adjustment
    df['PINCP_adj'] = df['PINCP'] * (df['ADJINC'] / 1_000_000)
    df['sum_components_adj'] = df['sum_components'] * (df['ADJINC'] / 1_000_000)
    
    # Print summary statistics
    print("\n=== Raw Income Values ===")
    print(df[['PINCP', 'sum_components', 'PINCP_adj', 'sum_components_adj']].describe())
    
    # Print first 10 rows
    print("\n=== First 10 Rows ===")
    print(df[['SERIALNO', 'SPORDER', 'PINCP', 'sum_components', 'ADJINC', 'PINCP_adj', 'sum_components_adj']].head(10))
    
    # Check for discrepancies
    print("\n=== Rows where PINCP != sum_components ===")
    diff = df[abs(df['PINCP'] - df['sum_components']) > 1]
    print(diff[['SERIALNO', 'SPORDER', 'PINCP', 'sum_components', 'ADJINC']])

if __name__ == "__main__":
    main()
