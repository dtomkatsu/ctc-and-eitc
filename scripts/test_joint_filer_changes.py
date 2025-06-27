#!/usr/bin/env python3
"""
Test script to verify the joint filer logic changes.
"""
import sys
import logging
import pandas as pd
from pathlib import Path

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

def analyze_filing_status(tax_units):
    """Analyze the distribution of filing statuses and other tax unit characteristics."""
    if len(tax_units) == 0:
        logger.error("No tax units to analyze")
        return
    
    # Check available columns
    available_columns = tax_units.columns.tolist()
    print("\nAvailable columns in tax_units:", available_columns)
    
    # Count by filing status if the column exists
    if 'filing_status' in tax_units.columns:
        status_counts = tax_units['filing_status'].value_counts().to_dict()
        total = len(tax_units)
        
        print("\n=== Filing Status Distribution ===")
        for status, count in status_counts.items():
            print(f"{status}: {count} ({(count/total)*100:.1f}%)")
        
        # Check for married filing separately
        if 'separate' in status_counts:
            print(f"\nMarried Filing Separately: {status_counts['separate']} ({(status_counts['separate']/total)*100:.1f}% of all filers)")
            if 'joint' in status_counts:
                total_married = status_counts['joint'] + status_counts['separate']
                if total_married > 0:
                    print(f"MFS as % of married couples: {(status_counts['separate']/total_married)*100:.1f}%")
    else:
        print("\nWarning: 'filing_status' column not found in tax_units")
    
    # Check income distribution
    print("\n=== Income Distribution ===")
    income_columns = [col for col in ['total_income', 'income', 'WAGP'] if col in tax_units.columns]
    if income_columns:
        for col in income_columns:
            print(f"\n{col}:")
            print(tax_units[col].describe())
    else:
        print("No income-related columns found in tax_units")
    
    # Check dependents if the column exists
    if 'num_dependents' in tax_units.columns:
        print("\n=== Dependents ===")
        print("Average dependents per tax unit:", tax_units['num_dependents'].mean())
        print("Max dependents:", tax_units['num_dependents'].max())
        print("Tax units with dependents:", len(tax_units[tax_units['num_dependents'] > 0]))
    else:
        print("\nWarning: 'num_dependents' column not found in tax_units")
    
    # Display first few rows for inspection
    print("\n=== Sample Tax Units ===")
    print(tax_units.head())

def main():
    """Main function to test the joint filer changes."""
    try:
        logger.info("Loading PUMS data...")
        loader = PUMSDataLoader()
        person_df, hh_df = loader.load_data()
        
        # Take a sample for testing (remove this for full run)
        sample_size = 10000  # Adjust as needed
        if len(person_df) > sample_size:
            logger.info(f"Taking a sample of {sample_size} persons for testing...")
            person_df = person_df.sample(n=sample_size, random_state=42)
            hh_ids = person_df['SERIALNO'].unique()
            hh_df = hh_df[hh_df['SERIALNO'].isin(hh_ids)]
        
        logger.info("Initializing TaxUnitConstructor...")
        constructor = TaxUnitConstructor(person_df, hh_df)
        
        logger.info("Creating tax units...")
        tax_units = constructor.create_rule_based_units()
        
        if tax_units is None or len(tax_units) == 0:
            logger.error("No tax units were created")
            return 1
            
        logger.info(f"Created {len(tax_units)} tax units")
        
        # Ensure filer_id is treated as a string
        tax_units['filer_id'] = tax_units['filer_id'].astype(str)
        
        # Save results for further analysis
        output_file = Path("data/processed/test_tax_units.parquet")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        tax_units.to_parquet(output_file, index=False)
        logger.info(f"Saved test tax units to {output_file}")
        
        # Analyze the results
        analyze_filing_status(tax_units)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in test: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
