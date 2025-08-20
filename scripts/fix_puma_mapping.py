#!/usr/bin/env python3
"""
Fix PUMA mapping for tax units by ensuring all tax units have proper PUMA assignments.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fix_puma_mapping.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    # Load data
    logger.info("Loading tax units data...")
    tax_units_path = 'src/data/processed/tax_units_rule_based.parquet'
    tax_units = pd.read_parquet(tax_units_path)
    logger.info(f"Loaded {len(tax_units):,} tax units")

    # Load person data
    logger.info("Loading person data...")
    person_data = pd.read_parquet('data/processed/pums_person_processed.parquet')
    logger.info(f"Loaded {len(person_data):,} person records")

    # Create a mapping of SERIALNO to PUMA (using the first occurrence if multiple)
    logger.info("Creating SERIALNO to PUMA mapping...")
    serialno_to_puma = person_data.drop_duplicates('SERIALNO')[['SERIALNO', 'PUMA']].set_index('SERIALNO')['PUMA']
    
    # Map PUMA to tax units
    logger.info("Mapping PUMA to tax units...")
    tax_units['PUMA'] = tax_units['SERIALNO'].map(serialno_to_puma)
    
    # Check for any remaining missing PUMAs
    missing_puma = tax_units['PUMA'].isna().sum()
    if missing_puma > 0:
        logger.warning(f"{missing_puma:,} tax units still missing PUMA information")
        
        # Try to get PUMA from household data
        hh_data = person_data.drop_duplicates('SERIALNO')[['SERIALNO', 'PUMA']].set_index('SERIALNO')['PUMA']
        tax_units['PUMA'] = tax_units['PUMA'].fillna(tax_units['SERIALNO'].map(hh_data))
        
        # Check again
        still_missing = tax_units['PUMA'].isna().sum()
        if still_missing > 0:
            logger.warning(f"{still_missing:,} tax units still missing PUMA after second attempt")
    
    # Load PUMA to county mapping - use corrected mapping
    puma_mapping_path = 'data/crosswalks/hawaii_puma_counties_corrected.csv'
    if os.path.exists(puma_mapping_path):
        puma_df = pd.read_csv(puma_mapping_path)
        puma_to_county = dict(zip(puma_df['PUMA'].astype(str), puma_df['county']))
        print(f"Loaded corrected PUMA mapping with {len(puma_to_county)} entries")
    else:
        print(f"PUMA mapping file not found: {puma_mapping_path}")
        # Create basic mapping using actual Census PUMA codes
        puma_to_county = {
            '0100': 'Maui_Kalawao_Kauai',  # Combined county
            '0200': 'Hawaii',              # Hawaii County
            '0301': 'Honolulu', '0302': 'Honolulu', '0303': 'Honolulu', '0304': 'Honolulu',
            '0305': 'Honolulu', '0306': 'Honolulu', '0307': 'Honolulu', '0308': 'Honolulu'
        }
    
    tax_units['county'] = tax_units['PUMA'].apply(lambda x: puma_to_county.get(str(x), 'Unknown'))
    
    # Save the updated tax units with PUMA and county information
    output_path = 'data/processed/tax_units_with_geography.parquet'
    tax_units.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(tax_units):,} tax units with geography to {output_path}")
    
    # Print summary
    print("\nTax Unit Distribution by County:")
    print("=" * 50)
    county_stats = tax_units['county'].value_counts().reset_index()
    county_stats.columns = ['county', 'num_tax_units']
    county_stats['pct_of_total'] = (county_stats['num_tax_units'] / len(tax_units)) * 100
    print(county_stats.to_string(index=False, float_format='{:,.2f}'.format))
    
    # Print PUMA distribution
    print("\nPUMA Distribution:")
    print("=" * 50)
    puma_stats = tax_units['PUMA'].value_counts().reset_index()
    puma_stats.columns = ['PUMA', 'count']
    puma_stats['pct'] = (puma_stats['count'] / len(tax_units)) * 100
    print(puma_stats.to_string(index=False, float_format='{:,.2f}'.format))

if __name__ == "__main__":
    main()
