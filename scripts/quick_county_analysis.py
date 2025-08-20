#!/usr/bin/env python3
"""
Quick analysis of tax units by county using PUMA mapping.
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
        logging.FileHandler('county_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

# Load data
logger.info("Loading tax units data...")
tax_units = pd.read_parquet('src/data/processed/tax_units_rule_based.parquet')
logger.info(f"Loaded {len(tax_units):,} tax units")

logger.info("Loading person data...")
person_data = pd.read_parquet('data/processed/pums_person_processed.parquet')
logger.info(f"Loaded {len(person_data):,} person records")

# Get unique SERIALNO-PUMA mapping
logger.info("Creating PUMA mapping...")
puma_mapping = person_data[['SERIALNO', 'PUMA']].drop_duplicates()
logger.info(f"Created mapping with {len(puma_mapping):,} unique SERIALNO-PUMA pairs")

# Merge with tax units
logger.info("Merging tax units with PUMA data...")
before_merge = len(tax_units)
tax_units = tax_units.merge(puma_mapping, on='SERIALNO', how='left')
after_merge = len(tax_units)

if before_merge != after_merge:
    logger.warning(f"Row count changed during merge - before: {before_merge}, after: {after_merge}")

# Check for missing PUMA values
missing_puma = tax_units['PUMA'].isna().sum()
if missing_puma > 0:
    logger.warning(f"{missing_puma:,} tax units are missing PUMA information")

# Load PUMA to county mapping - use corrected mapping
puma_mapping_path = 'data/crosswalks/hawaii_puma_counties_corrected.csv'
if not os.path.exists(puma_mapping_path):
    print(f"Warning: PUMA mapping file not found at {puma_mapping_path}")
    # Create a simple mapping based on actual Census PUMA codes
    def map_puma_to_county(puma_str):
        if pd.isna(puma_str):
            return 'Unknown'
        puma_str = str(puma_str).strip()
        # Handle both string and numeric PUMA codes
        if puma_str in ['100', '0100']:  # Maui + Kalawao + Kauai combined
            return 'Maui_Kalawao_Kauai'
        elif puma_str in ['200', '0200']:  # Hawaii County
            return 'Hawaii'  
        elif puma_str in ['301', '302', '303', '304', '305', '306', '307', '308',
                         '0301', '0302', '0303', '0304', '0305', '0306', '0307', '0308']:  # Honolulu
            return 'Honolulu'
        else:
            return 'Unknown'
else:
    puma_df = pd.read_csv(puma_mapping_path)
    puma_to_county = dict(zip(puma_df['PUMA'].astype(str), puma_df['county']))
    
    def map_puma_to_county(puma_str):
        if pd.isna(puma_str):
            return 'Unknown'
        puma_str = str(puma_str).strip()
        # Try exact match first
        if puma_str in puma_to_county:
            return puma_to_county[puma_str]
        # Try with leading zero
        puma_padded = puma_str.zfill(4)
        if puma_padded in puma_to_county:
            return puma_to_county[puma_padded]
        return 'Unknown'

tax_units['county'] = tax_units['PUMA'].apply(map_puma_to_county)

# Calculate basic statistics
county_stats = tax_units.groupby('county').agg(
    num_tax_units=('SERIALNO', 'count'),
    avg_income=('income', 'mean'),
    total_income=('income', 'sum'),
    avg_dependents=('num_dependents', 'mean')
).reset_index()

# Calculate population distribution
total_units = len(tax_units)
county_stats['pct_of_total'] = (county_stats['num_tax_units'] / total_units) * 100

# Format results
county_stats = county_stats.sort_values('num_tax_units', ascending=False)

print("\nTax Unit Distribution by County:")
print("=" * 50)
print(county_stats.to_string(index=False, float_format='{:,.2f}'.format))

# Calculate state totals
state_totals = {
    'county': 'STATE TOTAL',
    'num_tax_units': county_stats['num_tax_units'].sum(),
    'avg_income': (county_stats['total_income'].sum() / county_stats['num_tax_units'].sum()),
    'total_income': county_stats['total_income'].sum(),
    'avg_dependents': (county_stats['avg_dependents'] * county_stats['num_tax_units']).sum() / county_stats['num_tax_units'].sum(),
    'pct_of_total': 100.0
}

print("\nStatewide Totals:")
print("=" * 50)
for k, v in state_totals.items():
    if k == 'county':
        print(f"{k:>15}: {v}")
    elif isinstance(v, (int, np.integer)):
        print(f"{k:>15}: {v:,.0f}")
    else:
        print(f"{k:>15}: {v:,.2f}")
