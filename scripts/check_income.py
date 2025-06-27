#!/usr/bin/env python3
"""
Script to check income values in PUMS data.
"""
import pandas as pd
import logging
from pathlib import Path

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
    
    df = pd.read_csv(person_file, usecols=usecols)
    
    # Check ADJINC values
    print("\n=== ADJINC Values ===")
    print(df['ADJINC'].value_counts().head())
    
    # Look at some raw income values
    print("\n=== Sample Income Values ===")
    sample = df[df['PINCP'] > 0].sample(5, random_state=42)
    print(sample[['PINCP', 'ADJINC', 'WAGP', 'SEMP']])
    
    # Calculate adjusted income manually
    df['adjusted_income'] = df['PINCP'] * (df['ADJINC'] / 1_000_000)
    
    print("\n=== Adjusted Income Summary ===")
    print(df['adjusted_income'].describe())
    
    # Check if any income values are negative
    print("\n=== Negative Incomes ===")
    print(f"Number of negative incomes: {(df['adjusted_income'] < 0).sum()}")
    print("Negative income samples:")
    print(df[df['adjusted_income'] < 0].head())

if __name__ == "__main__":
    main()
