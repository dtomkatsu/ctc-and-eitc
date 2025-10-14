"""Test script for CEX Data Loader."""

import sys
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the loader
from src.data.cex_loader import CEXLoader, CEXIncomeProfile

def test_cex_loader():
    """Test the CEX loader with real data."""
    print("Testing CEX Loader with real data...")
    
    # Initialize the loader
    data_dir = Path('data/external/cex')
    loader = CEXLoader(data_dir=data_dir)
    
    # Test loading family data
    print("\n1. Loading family data...")
    try:
        family_data = loader.load_family_data()
        print(f"✅ Success! Loaded {len(family_data):,} family records")
        print(f"Columns: {', '.join(family_data.columns[:10])}...")
        print(f"Sample data (first row):\n{family_data.iloc[0].head(10).to_dict()}")
    except Exception as e:
        print(f"❌ Error loading family data: {str(e)}")
        raise
    
    # Test loading income data
    print("\n2. Loading income data...")
    try:
        income_data = loader.load_income_data()
        print(f"✅ Success! Loaded {len(income_data):,} income records")
        print(f"Income columns: {[c for c in income_data.columns if 'income' in c.lower()]}")
        print(f"Total income stats:\n{income_data['total_income'].describe()}")
    except Exception as e:
        print(f"❌ Error loading income data: {str(e)}")
        raise
    
    # Test income composition by age group
    print("\n3. Income composition by age group:")
    try:
        age_comp = loader.get_income_composition_by_group(group_by='age_group')
        print(age_comp[['n', 'mean_income', 'median_income', 'top_sources']].head())
    except Exception as e:
        print(f"❌ Error getting income composition: {str(e)}")
        raise
    
    # Test income matching
    print("\n4. Testing income matching:")
    test_incomes = [25000, 50000, 100000, 200000]
    for income in test_incomes:
        try:
            comp = loader.match_income_composition(income)
            print(f"\nIncome ${income:,}:")
            for k, v in comp.items():
                if v > 0.01:  # Only show significant sources
                    print(f"  {k}: {v:.1%}")
        except Exception as e:
            print(f"❌ Error matching income ${income:,}: {str(e)}")
    
    # Test summary statistics
    print("\n5. Summary statistics:")
    try:
        stats = loader.get_summary_statistics()
        print(f"Year: {stats['year']}")
        print(f"Households: {stats['n_households']:,}")
        print(f"Mean income: ${stats['mean_income']:,.2f}")
        print(f"Median income: ${stats['median_income']:,.2f}")
        print(f"% with wage income: {stats['pct_with_wages']:.1f}%")
        print(f"% with investment income: {stats['pct_with_investment']:.1f}%")
    except Exception as e:
        print(f"❌ Error getting summary statistics: {str(e)}")
        raise
    
    print("\n✅ CEX loader tests completed successfully!")

if __name__ == "__main__":
    test_cex_loader()
