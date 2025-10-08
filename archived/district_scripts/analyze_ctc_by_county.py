#!/usr/bin/env python3
"""
Analyze CTC results by county using PUMA mapping with PUMA 0100 imputation.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tax.credits.ctc import calculate_ctc
from src.analysis.puma_imputation import PUMA0100Imputer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('county_ctc_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load data
tax_units = pd.read_parquet('src/data/processed/tax_units_rule_based.parquet')
person_data = pd.read_parquet('data/processed/pums_person_processed.parquet')
crosswalk = pd.read_csv('data/crosswalks/hawaii_puma_districts_3digit.csv')

# Ensure consistent data types for merging
crosswalk['PUMA'] = crosswalk['PUMA'].astype(str)
person_data['PUMA'] = person_data['PUMA'].astype(str)

# Get unique mapping of PUMA to county
puma_to_county = crosswalk[['PUMA', 'county']].drop_duplicates()

# Merge tax units with person data to get PUMA
tax_units_with_puma = tax_units.merge(
    person_data[['SERIALNO', 'PUMA']].drop_duplicates(),
    on='SERIALNO',
    how='left'
)

# Load PUMA to county mapping - use corrected mapping
puma_mapping_path = 'data/crosswalks/hawaii_puma_counties_corrected.csv'
puma_df = pd.read_csv(puma_mapping_path)
puma_to_county = dict(zip(puma_df['PUMA'].astype(str), puma_df['county']))

# Merge with county mapping
tax_units_with_county = tax_units_with_puma.copy()
tax_units_with_county['county'] = tax_units_with_county['PUMA'].map(puma_to_county)

# Apply PUMA 0100 imputation
puma_0100_mask = tax_units_with_county['PUMA'].astype(str).isin(['100', '0100'])
if puma_0100_mask.any():
    logger.info(f"Found {puma_0100_mask.sum():,} tax units in PUMA 0100, applying imputation...")
    imputer = PUMA0100Imputer()
    tax_units_with_county = imputer.impute_counties(tax_units_with_county)
else:
    logger.info("No PUMA 0100 tax units found, skipping imputation")

# Calculate CTC for each tax unit
tax_units_with_ctc = []
for _, row in tax_units_with_county.iterrows():
    tax_unit = row.to_dict()
    ctc_result = calculate_ctc(tax_unit)
    tax_units_with_ctc.append({
        'SERIALNO': tax_unit['SERIALNO'],
        'county': tax_unit['county'],
        'ctc_total': ctc_result.get('ctc_total', 0),
        'ctc_refundable': ctc_result.get('ctc_refundable', 0),
        'ctc_nonrefundable': ctc_result.get('ctc_nonrefundable', 0),
        'num_children': len(ctc_result.get('qualifying_children', [])),
        'income': tax_unit.get('income', 0)
    })

# Convert to DataFrame
ctc_results = pd.DataFrame(tax_units_with_ctc)

# Group by county and calculate summary statistics
county_summary = ctc_results.groupby('county').agg({
    'SERIALNO': 'count',
    'ctc_total': 'sum',
    'ctc_refundable': 'sum',
    'ctc_nonrefundable': 'sum',
    'num_children': 'sum',
    'income': 'mean'
}).reset_index()

# Calculate per-child amounts
county_summary['avg_ctc_per_child'] = county_summary['ctc_total'] / county_summary['num_children']
county_summary['avg_refundable_per_child'] = county_summary['ctc_refundable'] / county_summary['num_children']
county_summary['avg_nonrefundable_per_child'] = county_summary['ctc_nonrefundable'] / county_summary['num_children']

# Format results
county_summary = county_summary.rename(columns={
    'SERIALNO': 'num_tax_units',
    'ctc_total': 'total_ctc',
    'ctc_refundable': 'total_refundable',
    'ctc_nonrefundable': 'total_nonrefundable',
    'income': 'avg_income'
})

# Reorder columns
county_summary = county_summary[[
    'county', 'num_tax_units', 'num_children', 'avg_income',
    'total_ctc', 'total_refundable', 'total_nonrefundable',
    'avg_ctc_per_child', 'avg_refundable_per_child', 'avg_nonrefundable_per_child'
]]

# Save results
output_dir = Path('output/county_analysis')
output_dir.mkdir(exist_ok=True, parents=True)
county_summary.to_csv(output_dir / 'ctc_by_county.csv', index=False)

print("\nCTC Analysis by County:")
print("=" * 50)
print(county_summary.to_string(index=False, float_format='{:,.2f}'.format))

# Calculate state totals
state_totals = county_summary.sum(numeric_only=True)
state_totals['avg_income'] = county_summary['avg_income'].mean()
state_totals['avg_ctc_per_child'] = county_summary['total_ctc'].sum() / county_summary['num_children'].sum()
state_totals['avg_refundable_per_child'] = county_summary['total_refundable'].sum() / county_summary['num_children'].sum()
state_totals['avg_nonrefundable_per_child'] = county_summary['total_nonrefundable'].sum() / county_summary['num_children'].sum()

print("\nStatewide Totals:")
print("=" * 50)
print(f"Total Tax Units: {state_totals['num_tax_units']:,.0f}")
print(f"Total Children: {state_totals['num_children']:,.0f}")
print(f"Average Income: ${state_totals['avg_income']:,.2f}")
print(f"Total CTC: ${state_totals['total_ctc']:,.2f}")
print(f"Total Refundable: ${state_totals['total_refundable']:,.2f}")
print(f"Total Non-refundable: ${state_totals['total_nonrefundable']:,.2f}")
print(f"Avg CTC per Child: ${state_totals['avg_ctc_per_child']:,.2f}")
print(f"Avg Refundable per Child: ${state_totals['avg_refundable_per_child']:,.2f}")
print(f"Avg Non-refundable per Child: ${state_totals['avg_nonrefundable_per_child']:,.2f}")
