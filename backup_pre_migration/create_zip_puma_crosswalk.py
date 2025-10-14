#!/usr/bin/env python3
"""
Create ZIP-to-PUMA crosswalk for Hawaii using HUD and Census data sources.
"""

import pandas as pd
import numpy as np
import requests
import logging
from pathlib import Path
import zipfile
import io

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_hud_zip_county_crosswalk():
    """Download HUD USPS ZIP-to-County crosswalk."""
    logger.info("Downloading HUD ZIP-to-County crosswalk...")
    
    # HUD provides quarterly ZIP-to-County crosswalks
    hud_url = "https://www.huduser.gov/portal/datasets/usps_crosswalk.html"
    
    # For now, we'll use a direct link to the most recent crosswalk
    # This URL may need to be updated based on HUD's current data
    crosswalk_url = "https://www.huduser.gov/portal/datasets/usps/ZIP_COUNTY_122023.xlsx"
    
    try:
        logger.info(f"Attempting to download from: {crosswalk_url}")
        response = requests.get(crosswalk_url, timeout=30)
        response.raise_for_status()
        
        # Save to file
        output_path = Path('data/crosswalks/hud_zip_county_crosswalk.xlsx')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"Downloaded HUD crosswalk to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.warning(f"Could not download HUD crosswalk: {e}")
        logger.info("Will create manual Hawaii ZIP-County mapping...")
        return None

def create_manual_hawaii_zip_county():
    """Create manual ZIP-to-County mapping for Hawaii based on known ZIP ranges."""
    logger.info("Creating manual Hawaii ZIP-County mapping...")
    
    # Hawaii ZIP code ranges by county (approximate)
    zip_county_mapping = {
        # Honolulu County (Oahu)
        'honolulu': [
            range(96701, 96898),  # Main Oahu range
            [96813, 96814, 96815, 96816, 96817, 96818, 96819, 96820, 96821, 96822, 96823, 96825, 96826],
            [96801, 96802, 96803, 96804, 96805, 96806, 96807, 96808, 96809, 96810, 96811, 96812],
            [96850, 96860, 96898]
        ],
        # Hawaii County (Big Island)  
        'hawaii': [
            range(96704, 96785),  # Big Island range
            [96720, 96725, 96727, 96728, 96737, 96738, 96740, 96743, 96749, 96750, 96755, 96760, 96771, 96772, 96773, 96774, 96776, 96777, 96778, 96780, 96781, 96783, 96785]
        ],
        # Maui County
        'maui': [
            [96708, 96713, 96729, 96732, 96753, 96761, 96768, 96779, 96784, 96790, 96793]
        ],
        # Kalawao County (part of Maui for practical purposes)
        'kalawao': [
            [96742]  # Kalaupapa
        ],
        # Kauai County
        'kauai': [
            [96703, 96705, 96714, 96715, 96716, 96722, 96741, 96746, 96747, 96751, 96752, 96754, 96756, 96765, 96766, 96769, 96770]
        ]
    }
    
    # Create DataFrame
    zip_county_data = []
    
    for county, zip_groups in zip_county_mapping.items():
        for zip_group in zip_groups:
            if isinstance(zip_group, range):
                for zip_code in zip_group:
                    zip_county_data.append({
                        'ZIP': f"{zip_code:05d}",
                        'COUNTY': county.title(),
                        'STATE': 'HI'
                    })
            else:
                for zip_code in zip_group:
                    zip_county_data.append({
                        'ZIP': f"{zip_code:05d}",
                        'COUNTY': county.title(),
                        'STATE': 'HI'
                    })
    
    df = pd.DataFrame(zip_county_data)
    
    # Save manual crosswalk
    output_path = Path('data/crosswalks/hawaii_zip_county_manual.csv')
    df.to_csv(output_path, index=False)
    logger.info(f"Created manual Hawaii ZIP-County mapping: {output_path}")
    logger.info(f"Mapped {len(df)} ZIP codes across {df['COUNTY'].nunique()} counties")
    
    return df

def load_existing_puma_county_crosswalk():
    """Load our existing PUMA-to-County crosswalk."""
    logger.info("Loading existing PUMA-County crosswalk...")
    
    crosswalk_path = Path('data/crosswalks/hawaii_puma_districts_official_2022.csv')
    
    if not crosswalk_path.exists():
        logger.error(f"PUMA-County crosswalk not found: {crosswalk_path}")
        return None
    
    df = pd.read_csv(crosswalk_path)
    
    # Extract unique PUMA-County mappings
    puma_county = df[['PUMA', 'county']].drop_duplicates()
    
    logger.info(f"Loaded PUMA-County mappings: {len(puma_county)} unique combinations")
    logger.info("PUMA-County distribution:")
    print(puma_county.groupby('county')['PUMA'].count().to_string())
    
    return puma_county

def create_zip_puma_crosswalk():
    """Create comprehensive ZIP-to-PUMA crosswalk for Hawaii."""
    logger.info("Creating ZIP-to-PUMA crosswalk...")
    
    # Load ZIP-County mapping
    zip_county_df = create_manual_hawaii_zip_county()
    
    # Load PUMA-County mapping  
    puma_county_df = load_existing_puma_county_crosswalk()
    
    if puma_county_df is None:
        logger.error("Cannot create crosswalk without PUMA-County mapping")
        return None
    
    # Standardize county names for joining
    # Our PUMA crosswalk uses combined counties for some areas
    county_mapping = {
        'Honolulu': 'Honolulu',
        'Hawaii': 'Hawaii', 
        'Maui': 'Maui+Kalawao+Kauai',  # Combined in PUMA data
        'Kalawao': 'Maui+Kalawao+Kauai',  # Combined in PUMA data
        'Kauai': 'Maui+Kalawao+Kauai'   # Combined in PUMA data
    }
    
    # Map ZIP counties to PUMA counties
    zip_county_df['PUMA_COUNTY'] = zip_county_df['COUNTY'].map(county_mapping)
    
    # Join with PUMA data
    zip_puma_df = zip_county_df.merge(
        puma_county_df, 
        left_on='PUMA_COUNTY', 
        right_on='county',
        how='left'
    )
    
    # Clean up columns
    zip_puma_df = zip_puma_df[['ZIP', 'COUNTY', 'PUMA_COUNTY', 'PUMA']].rename(columns={
        'ZIP': 'zip_code',
        'COUNTY': 'actual_county', 
        'PUMA_COUNTY': 'puma_county',
        'PUMA': 'puma_code'
    })
    
    # Add PUMA code formatting (ensure 3 digits)
    zip_puma_df['puma_code'] = zip_puma_df['puma_code'].astype(str).str.zfill(3)
    
    logger.info(f"Created ZIP-PUMA crosswalk with {len(zip_puma_df)} ZIP codes")
    logger.info("Distribution by PUMA:")
    print(zip_puma_df.groupby(['puma_code', 'puma_county']).size().to_string())
    
    # Save crosswalk
    output_path = Path('data/crosswalks/hawaii_zip_puma_crosswalk.csv')
    zip_puma_df.to_csv(output_path, index=False)
    logger.info(f"Saved ZIP-PUMA crosswalk: {output_path}")
    
    return zip_puma_df

def validate_crosswalk_with_irs_data():
    """Validate the crosswalk against IRS SOI ZIP codes."""
    logger.info("Validating crosswalk against IRS SOI data...")
    
    # Load IRS ZIP codes
    irs_zip_path = Path('data/irs_soi/hawaii_zip_statistics.csv')
    if not irs_zip_path.exists():
        logger.warning("IRS ZIP statistics not found for validation")
        return
    
    irs_zips = pd.read_csv(irs_zip_path)
    irs_zip_set = set(irs_zips['zip_code'].astype(str))
    
    # Load our crosswalk
    crosswalk_path = Path('data/crosswalks/hawaii_zip_puma_crosswalk.csv')
    if not crosswalk_path.exists():
        logger.error("ZIP-PUMA crosswalk not found")
        return
    
    crosswalk_df = pd.read_csv(crosswalk_path)
    crosswalk_zip_set = set(crosswalk_df['zip_code'].astype(str))
    
    # Compare coverage
    irs_count = len(irs_zip_set)
    crosswalk_count = len(crosswalk_zip_set)
    overlap_count = len(irs_zip_set & crosswalk_zip_set)
    
    logger.info(f"IRS SOI ZIP codes: {irs_count}")
    logger.info(f"Crosswalk ZIP codes: {crosswalk_count}")
    logger.info(f"Overlap: {overlap_count} ({overlap_count/irs_count*100:.1f}% of IRS ZIPs)")
    
    # Identify missing ZIPs
    missing_in_crosswalk = irs_zip_set - crosswalk_zip_set
    missing_in_irs = crosswalk_zip_set - irs_zip_set
    
    if missing_in_crosswalk:
        logger.warning(f"ZIP codes in IRS but not in crosswalk: {len(missing_in_crosswalk)}")
        logger.info(f"Sample missing: {list(missing_in_crosswalk)[:10]}")
    
    if missing_in_irs:
        logger.info(f"ZIP codes in crosswalk but not in IRS: {len(missing_in_irs)}")
        logger.info(f"Sample extra: {list(missing_in_irs)[:10]}")
    
    return {
        'irs_zip_count': irs_count,
        'crosswalk_zip_count': crosswalk_count,
        'overlap_count': overlap_count,
        'coverage_pct': overlap_count/irs_count*100 if irs_count > 0 else 0
    }

def create_enhanced_crosswalk_with_districts():
    """Create enhanced crosswalk that includes legislative districts."""
    logger.info("Creating enhanced crosswalk with legislative districts...")
    
    # Load ZIP-PUMA crosswalk
    zip_puma_path = Path('data/crosswalks/hawaii_zip_puma_crosswalk.csv')
    if not zip_puma_path.exists():
        logger.error("ZIP-PUMA crosswalk not found")
        return None
    
    zip_puma_df = pd.read_csv(zip_puma_path)
    
    # Load PUMA-District crosswalk
    puma_district_path = Path('data/crosswalks/hawaii_puma_districts_official_2022.csv')
    if not puma_district_path.exists():
        logger.error("PUMA-District crosswalk not found")
        return None
    
    puma_district_df = pd.read_csv(puma_district_path)
    
    # Join to add district information
    enhanced_df = zip_puma_df.merge(
        puma_district_df[['PUMA', 'house_district', 'senate_district']],
        left_on='puma_code',
        right_on='PUMA',
        how='left'
    )
    
    # Clean up columns
    enhanced_df = enhanced_df.drop('PUMA', axis=1)
    
    logger.info(f"Enhanced crosswalk with {len(enhanced_df)} ZIP codes")
    logger.info("Sample enhanced crosswalk:")
    print(enhanced_df.head(10).to_string())
    
    # Save enhanced crosswalk
    output_path = Path('data/crosswalks/hawaii_zip_puma_district_crosswalk.csv')
    enhanced_df.to_csv(output_path, index=False)
    logger.info(f"Saved enhanced crosswalk: {output_path}")
    
    return enhanced_df

def main():
    """Main function to create comprehensive ZIP-PUMA crosswalk."""
    logger.info("Starting ZIP-PUMA crosswalk creation...")
    
    # Create base ZIP-PUMA crosswalk
    zip_puma_df = create_zip_puma_crosswalk()
    
    if zip_puma_df is None:
        logger.error("Failed to create ZIP-PUMA crosswalk")
        return
    
    # Validate against IRS data
    validation_results = validate_crosswalk_with_irs_data()
    
    # Create enhanced version with districts
    enhanced_df = create_enhanced_crosswalk_with_districts()
    
    logger.info("\n" + "="*60)
    logger.info("ZIP-PUMA CROSSWALK CREATION COMPLETE")
    logger.info("="*60)
    
    if validation_results:
        logger.info(f"Coverage of IRS ZIP codes: {validation_results['coverage_pct']:.1f}%")
        logger.info(f"Total ZIP codes mapped: {validation_results['crosswalk_zip_count']}")
    
    logger.info("\nFiles created:")
    logger.info("- data/crosswalks/hawaii_zip_county_manual.csv")
    logger.info("- data/crosswalks/hawaii_zip_puma_crosswalk.csv") 
    logger.info("- data/crosswalks/hawaii_zip_puma_district_crosswalk.csv")
    
    logger.info("\nNext steps:")
    logger.info("1. Use crosswalk to aggregate IRS SOI data by PUMA")
    logger.info("2. Compare IRS vs PUMS filing status distributions by PUMA")
    logger.info("3. Implement geographic calibration of tax unit construction")
    
    return {
        'zip_puma_crosswalk': zip_puma_df,
        'enhanced_crosswalk': enhanced_df,
        'validation': validation_results
    }

if __name__ == "__main__":
    results = main()
