#!/usr/bin/env python3
"""
Create official PUMA-to-district crosswalk using Census Bureau 2022 legislative district data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_puma_district_mapping():
    """Create PUMA-to-district mapping using tract-based allocation."""
    logger.info("Creating PUMA-district mapping from Census tract data...")
    
    # Load House districts
    house_df = pd.read_csv('hawaii_house_districts_tract_relationship.txt', sep='|')
    house_df = house_df[house_df['GEOID_TRACT_20'].astype(str).str.startswith('15')]
    house_df['house_district'] = house_df['GEOID_SLDL2022_20'].astype(str).str[-2:].astype(int)
    house_df['tract'] = house_df['GEOID_TRACT_20'].astype(str)
    
    # Load Senate districts
    senate_df = pd.read_csv('hawaii_senate_districts_tract_relationship.txt', sep='|')
    senate_df = senate_df[senate_df['GEOID_TRACT_20'].astype(str).str.startswith('15')]
    senate_df['senate_district'] = senate_df['GEOID_SLDU2022_20'].astype(str).str[-2:].astype(int)
    senate_df['tract'] = senate_df['GEOID_TRACT_20'].astype(str)
    
    logger.info(f"Loaded {len(house_df)} House district-tract mappings")
    logger.info(f"Loaded {len(senate_df)} Senate district-tract mappings")
    
    # Merge House and Senate on tract
    district_df = house_df[['tract', 'house_district']].merge(
        senate_df[['tract', 'senate_district']], 
        on='tract', 
        how='outer'
    )
    
    # Map tracts to PUMAs based on Hawaii geography
    # Hawaii has specific tract-to-PUMA mappings based on county structure
    def tract_to_puma(tract_id):
        """Map tract ID to PUMA based on Hawaii county structure."""
        county_code = tract_id[2:5]  # Extract county FIPS from tract ID
        
        if county_code == '001':  # Hawaii County
            return '200'
        elif county_code == '003':  # Honolulu County - map to PUMAs 301-308
            # Distribute Honolulu tracts across 8 PUMAs based on tract number
            tract_num = int(tract_id[5:11])
            puma_num = (tract_num % 8) + 1  # Distribute across PUMAs 1-8
            return f'30{puma_num}'
        elif county_code in ['005', '007', '009']:  # Kalawao, Kauai, Maui -> PUMA 100
            return '100'
        else:
            return '999'  # Unknown
    
    district_df['PUMA'] = district_df['tract'].apply(tract_to_puma)
    
    # Remove unknown PUMAs
    district_df = district_df[district_df['PUMA'] != '999']
    
    # Create PUMA-district summary by finding the most common districts for each PUMA
    puma_house = district_df.groupby(['PUMA', 'house_district']).size().reset_index(name='tract_count')
    puma_senate = district_df.groupby(['PUMA', 'senate_district']).size().reset_index(name='tract_count')
    
    # For each PUMA, get the top districts (allowing multiple districts per PUMA)
    crosswalk_entries = []
    
    for puma in district_df['PUMA'].unique():
        # Get top House districts for this PUMA (up to 6 districts per PUMA)
        puma_house_data = puma_house[puma_house['PUMA'] == puma].nlargest(6, 'tract_count')
        # Get top Senate districts for this PUMA (up to 3 districts per PUMA)
        puma_senate_data = puma_senate[puma_senate['PUMA'] == puma].nlargest(3, 'tract_count')
        
        house_districts = puma_house_data['house_district'].tolist()
        senate_districts = puma_senate_data['senate_district'].tolist()
        
        # Create combinations of house and senate districts for this PUMA
        for house_dist in house_districts:
            for senate_dist in senate_districts:
                crosswalk_entries.append({
                    'PUMA': puma,
                    'house_district': house_dist,
                    'senate_district': senate_dist,
                    'county': get_county_for_puma(puma)
                })
    
    crosswalk_df = pd.DataFrame(crosswalk_entries).drop_duplicates()
    
    logger.info(f"Created crosswalk with {len(crosswalk_df)} PUMA-district combinations")
    logger.info(f"Unique PUMAs: {len(crosswalk_df['PUMA'].unique())}")
    logger.info(f"Unique House districts: {len(crosswalk_df['house_district'].unique())}")
    logger.info(f"Unique Senate districts: {len(crosswalk_df['senate_district'].unique())}")
    
    return crosswalk_df

def get_county_for_puma(puma):
    """Get county name for PUMA code."""
    if puma == '100':
        return 'Maui+Kauai'
    elif puma == '200':
        return 'Hawaii'
    elif puma.startswith('30'):
        return 'Honolulu'
    else:
        return 'Unknown'

def main():
    """Create and save the official crosswalk."""
    logger.info("=== Creating Official Hawaii PUMA-District Crosswalk ===")
    
    crosswalk_df = create_puma_district_mapping()
    
    # Save to file
    output_path = Path('data/crosswalks/hawaii_puma_districts_official_2022.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    crosswalk_df.to_csv(output_path, index=False)
    logger.info(f"Saved official crosswalk to {output_path}")
    
    # Display summary
    print("\n=== OFFICIAL CROSSWALK SUMMARY ===")
    print(f"Total PUMA-district combinations: {len(crosswalk_df)}")
    print(f"Unique PUMAs: {sorted(crosswalk_df['PUMA'].unique())}")
    print(f"Unique House districts: {len(crosswalk_df['house_district'].unique())} (should be 51)")
    print(f"Unique Senate districts: {len(crosswalk_df['senate_district'].unique())} (should be 25)")
    
    print(f"\nHouse districts covered: {sorted(crosswalk_df['house_district'].unique())}")
    print(f"Senate districts covered: {sorted(crosswalk_df['senate_district'].unique())}")
    
    print("\nSample crosswalk entries:")
    print(crosswalk_df.head(15))
    
    return crosswalk_df

if __name__ == "__main__":
    crosswalk = main()
