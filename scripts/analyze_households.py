#!/usr/bin/env python3
"""
Analyze household composition in the PUMS data, particularly focusing on households without adults.
"""
import pandas as pd
import logging
from pathlib import Path
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_data():
    """Load the processed PUMS data."""
    data_dir = Path(__file__).parent.parent / 'data' / 'processed'
    
    logger.info("Loading processed PUMS data...")
    person_file = data_dir / 'pums_person_processed.parquet'
    hh_file = data_dir / 'pums_household_processed.parquet'
    
    if not person_file.exists() or not hh_file.exists():
        logger.error("Processed data files not found. Please run process_pums.py first.")
        sys.exit(1)
    
    person_df = pd.read_parquet(person_file)
    hh_df = pd.read_parquet(hh_file)
    
    return person_df, hh_df

def analyze_households(person_df, hh_df):
    """Analyze household composition."""
    # Get household size and adult count
    hh_summary = person_df.groupby('SERIALNO').agg(
        hh_size=('SPORDER', 'size'),
        adult_count=('AGEP', lambda x: (x >= 18).sum()),
        child_count=('AGEP', lambda x: ((x < 18) & (x > 0)).sum()),
        has_foster=('RELSHIPP', lambda x: (x == 37).any())
    ).reset_index()
    
    # Merge with household data
    hh_summary = hh_summary.merge(hh_df[['SERIALNO', 'WGTP']], on='SERIALNO', how='left')
    
    # Categorize households
    hh_summary['category'] = 'Other'
    
    # Empty households
    empty_mask = hh_summary['hh_size'] == 0
    hh_summary.loc[empty_mask, 'category'] = 'Empty'
    
    # Households with children but no adults
    child_only_mask = (hh_summary['adult_count'] == 0) & (hh_summary['child_count'] > 0)
    hh_summary.loc[child_only_mask, 'category'] = 'Children only'
    
    # Households with foster children
    foster_mask = hh_summary['has_foster']
    hh_summary.loc[foster_mask, 'category'] = 'Has foster children'
    
    # Count households in each category
    category_counts = hh_summary.groupby('category').agg(
        count=('SERIALNO', 'size'),
        weighted_count=('WGTP', 'sum')
    ).reset_index()
    
    # Calculate percentages
    total = category_counts['count'].sum()
    total_weighted = category_counts['weighted_count'].sum()
    
    category_counts['pct'] = (category_counts['count'] / total * 100).round(2)
    category_counts['weighted_pct'] = (category_counts['weighted_count'] / total_weighted * 100).round(2)
    
    return category_counts, hh_summary

def main():
    """Main function."""
    try:
        # Load data
        person_df, hh_df = load_data()
        
        # Analyze households
        logger.info("Analyzing household composition...")
        category_counts, hh_summary = analyze_households(person_df, hh_df)
        
        # Print results
        print("\nHousehold Composition Analysis:")
        print("=" * 50)
        print("Category\tCount\t% of Total\tWeighted Count\tWeighted %")
        print("-" * 70)
        
        for _, row in category_counts.sort_values('count', ascending=False).iterrows():
            print(f"{row['category']:15} {row['count']:7,} {row['pct']:8.2f}% \
                {row['weighted_count']:12,.0f} {row['weighted_pct']:10.2f}%")
        
        # Detailed analysis of households without adults
        no_adults = hh_summary[hh_summary['adult_count'] == 0]
        if not no_adults.empty:
            print("\nHouseholds without adults (first 5):")
            print(no_adults[['SERIALNO', 'hh_size', 'child_count', 'has_foster']].head().to_string(index=False))
            
            # Save detailed info to file
            output_file = Path('households_without_adults_detailed.csv')
            no_adults.to_csv(output_file, index=False)
            print(f"\nDetailed information saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error analyzing households: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
