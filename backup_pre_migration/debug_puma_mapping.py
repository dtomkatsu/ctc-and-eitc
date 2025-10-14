#!/usr/bin/env python3
"""
Debug PUMA mapping issue in district assignment.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_tax_units_puma_format():
    """Check PUMA format in tax units data."""
    logger.info("Checking tax units PUMA format...")
    
    tax_units_path = Path('src/data/processed/tax_units_weight_split_raked.parquet')
    if not tax_units_path.exists():
        logger.error(f"Tax units file not found: {tax_units_path}")
        return None
    
    df = pd.read_parquet(tax_units_path)
    logger.info(f"Loaded {len(df)} tax units")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Check for PUMA-related columns
    puma_cols = [col for col in df.columns if 'puma' in col.lower()]
    logger.info(f"PUMA-related columns: {puma_cols}")
    
    # Check the actual PUMA values
    if 'PUMA' in df.columns:
        puma_values = df['PUMA'].value_counts().head(10)
        logger.info("Top 10 PUMA values:")
        print(puma_values)
        
        # Check data types and format
        logger.info(f"PUMA data type: {df['PUMA'].dtype}")
        logger.info(f"Sample PUMA values: {df['PUMA'].head().tolist()}")
        
        return df['PUMA'].unique()
    else:
        logger.error("No PUMA column found in tax units")
        return None

def check_crosswalk_puma_format():
    """Check PUMA format in crosswalk data."""
    logger.info("Checking crosswalk PUMA format...")
    
    crosswalk_path = Path('data/crosswalks/hawaii_zip_puma_district_crosswalk.csv')
    if not crosswalk_path.exists():
        logger.error(f"Crosswalk file not found: {crosswalk_path}")
        return None
    
    df = pd.read_csv(crosswalk_path)
    logger.info(f"Loaded crosswalk with {len(df)} records")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Check PUMA values in crosswalk
    if 'puma_code' in df.columns:
        puma_values = df['puma_code'].value_counts()
        logger.info("PUMA values in crosswalk:")
        print(puma_values)
        
        logger.info(f"PUMA data type: {df['puma_code'].dtype}")
        logger.info(f"Sample PUMA values: {df['puma_code'].head().tolist()}")
        
        return df['puma_code'].unique()
    else:
        logger.error("No puma_code column found in crosswalk")
        return None

def create_corrected_mapping():
    """Create corrected PUMA-to-district mapping."""
    logger.info("Creating corrected PUMA-district mapping...")
    
    # Load the original PUMA-district crosswalk
    original_path = Path('data/crosswalks/hawaii_puma_districts_official_2022.csv')
    if not original_path.exists():
        logger.error(f"Original crosswalk not found: {original_path}")
        return None
    
    df = pd.read_csv(original_path)
    logger.info(f"Loaded original crosswalk with {len(df)} records")
    
    # Check PUMA format
    logger.info(f"Original PUMA values: {df['PUMA'].unique()}")
    
    # Create mapping with different PUMA formats to handle potential mismatches
    puma_district_map = {}
    
    for _, row in df.iterrows():
        puma = row['PUMA']
        house_dist = row['house_district']
        senate_dist = row['senate_district']
        
        # Try multiple PUMA formats
        puma_formats = [
            str(puma),           # Original format
            str(puma).zfill(3),  # 3-digit format
            f"{puma}.0",         # Float format
            int(puma) if pd.notna(puma) else None  # Integer format
        ]
        
        for puma_fmt in puma_formats:
            if puma_fmt is not None:
                puma_district_map[puma_fmt] = {
                    'house_district': house_dist,
                    'senate_district': senate_dist
                }
    
    logger.info(f"Created mapping with {len(puma_district_map)} PUMA format variations")
    
    # Show sample mappings
    sample_keys = list(puma_district_map.keys())[:10]
    logger.info("Sample mappings:")
    for key in sample_keys:
        logger.info(f"  {key}: {puma_district_map[key]}")
    
    return puma_district_map

def test_mapping_with_tax_units(puma_district_map):
    """Test the mapping with actual tax units data."""
    logger.info("Testing mapping with tax units...")
    
    tax_units_path = Path('src/data/processed/tax_units_weight_split_raked.parquet')
    if not tax_units_path.exists():
        logger.error(f"Tax units file not found: {tax_units_path}")
        return
    
    df = pd.read_parquet(tax_units_path)
    
    if 'PUMA' not in df.columns:
        logger.error("No PUMA column in tax units")
        return
    
    # Test different PUMA format conversions
    test_formats = [
        ('original', df['PUMA']),
        ('string', df['PUMA'].astype(str)),
        ('3digit', df['PUMA'].astype(str).str.zfill(3)),
        ('float_str', df['PUMA'].astype(str) + '.0')
    ]
    
    for format_name, puma_series in test_formats:
        matches = puma_series.map(lambda x: x in puma_district_map).sum()
        total = len(puma_series)
        logger.info(f"{format_name} format: {matches}/{total} matches ({matches/total*100:.1f}%)")
        
        if matches > 0:
            logger.info(f"  Sample successful mappings for {format_name}:")
            successful = puma_series[puma_series.map(lambda x: x in puma_district_map)]
            for puma in successful.head(3):
                mapping = puma_district_map[puma]
                logger.info(f"    {puma} -> House: {mapping['house_district']}, Senate: {mapping['senate_district']}")

def main():
    """Main debugging function."""
    logger.info("Starting PUMA mapping debug...")
    
    # Check formats in both datasets
    tax_units_pumas = check_tax_units_puma_format()
    crosswalk_pumas = check_crosswalk_puma_format()
    
    # Create corrected mapping
    corrected_mapping = create_corrected_mapping()
    
    if corrected_mapping:
        # Test with tax units
        test_mapping_with_tax_units(corrected_mapping)
        
        # Save corrected mapping
        output_path = Path('data/crosswalks/corrected_puma_district_mapping.json')
        with open(output_path, 'w') as f:
            json.dump(corrected_mapping, f, indent=2)
        logger.info(f"Saved corrected mapping: {output_path}")
    
    logger.info("Debug complete")

if __name__ == "__main__":
    main()
