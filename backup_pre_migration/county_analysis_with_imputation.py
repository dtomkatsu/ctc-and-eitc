"""
County-Level CTC Analysis with PUMA 0100 Imputation

This script performs county-level Child Tax Credit analysis for Hawaii,
including sophisticated imputation to separate Maui and Kauai counties
from the combined PUMA 0100.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from analysis.puma_imputation import impute_puma_0100_counties
from tax.credits.ctc import calculate_ctc

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('county_analysis_with_imputation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_tax_units_with_geography():
    """Load tax units with PUMA information."""
    # Try to load existing tax units with geography
    geography_path = 'data/processed/tax_units_with_geography.parquet'
    if os.path.exists(geography_path):
        logger.info(f"Loading tax units with geography from {geography_path}")
        return pd.read_parquet(geography_path)
    
    # Fallback to loading and merging data
    logger.info("Loading tax units and merging with person data for PUMA information")
    
    # Load tax units
    tax_units_path = 'data/processed/hawaii_tax_units_latest.parquet'
    if not os.path.exists(tax_units_path):
        raise FileNotFoundError(f"Tax units file not found: {tax_units_path}")
    
    tax_units = pd.read_parquet(tax_units_path)
    logger.info(f"Loaded {len(tax_units):,} tax units")
    
    # Load person data for PUMA information
    person_data_path = 'data/processed/hawaii_person_data_latest.parquet'
    if not os.path.exists(person_data_path):
        raise FileNotFoundError(f"Person data file not found: {person_data_path}")
    
    person_data = pd.read_parquet(person_data_path)
    logger.info(f"Loaded {len(person_data):,} person records")
    
    # Merge to get PUMA information
    puma_mapping = person_data[['SERIALNO', 'PUMA']].drop_duplicates()
    tax_units_with_puma = tax_units.merge(puma_mapping, on='SERIALNO', how='left')
    
    missing_puma = tax_units_with_puma['PUMA'].isna().sum()
    if missing_puma > 0:
        logger.warning(f"{missing_puma:,} tax units missing PUMA information")
    
    return tax_units_with_puma

def apply_county_mapping(tax_units_df):
    """Apply county mapping including imputation for PUMA 0100."""
    logger.info("Applying county mapping with PUMA 0100 imputation")
    
    # Load PUMA to county mapping
    puma_mapping_path = 'data/crosswalks/hawaii_puma_counties_corrected.csv'
    if not os.path.exists(puma_mapping_path):
        raise FileNotFoundError(f"PUMA mapping file not found: {puma_mapping_path}")
    
    puma_df = pd.read_csv(puma_mapping_path)
    puma_to_county = dict(zip(puma_df['PUMA'].astype(str), puma_df['county']))
    
    # Apply basic county mapping
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
    
    tax_units_df['county'] = tax_units_df['PUMA'].apply(map_puma_to_county)
    
    # Apply imputation for PUMA 0100 (Maui_Kalawao_Kauai)
    puma_0100_count = (tax_units_df['county'] == 'Maui_Kalawao_Kauai').sum()
    if puma_0100_count > 0:
        logger.info(f"Found {puma_0100_count:,} tax units in PUMA 0100 requiring imputation")
        tax_units_df = impute_puma_0100_counties(tax_units_df, validate_results=True)
    else:
        logger.info("No PUMA 0100 tax units found")
    
    return tax_units_df

def calculate_ctc_by_county(tax_units_df):
    """Calculate CTC estimates by county."""
    logger.info("Calculating CTC by county")
    
    # Calculate CTC for each tax unit
    ctc_results = []
    for idx, row in tax_units_df.iterrows():
        if idx % 10000 == 0:
            logger.info(f"Processing tax unit {idx:,} of {len(tax_units_df):,}")
        
        tax_unit = row.to_dict()
        ctc_result = calculate_ctc(tax_unit)
        
        ctc_results.append({
            'filer_id': tax_unit.get('filer_id'),
            'SERIALNO': tax_unit.get('SERIALNO'),
            'county': tax_unit.get('county'),
            'filing_status': tax_unit.get('filing_status'),
            'income': tax_unit.get('income', 0),
            'num_dependents': tax_unit.get('num_dependents', 0),
            'weight': tax_unit.get('weight', 1.0),
            'ctc_total': ctc_result.get('ctc_total', 0),
            'ctc_refundable': ctc_result.get('ctc_refundable', 0),
            'ctc_nonrefundable': ctc_result.get('ctc_nonrefundable', 0),
            'qualifying_children': ctc_result.get('qualifying_children', 0)
        })
    
    return pd.DataFrame(ctc_results)

def generate_county_summary(ctc_results_df):
    """Generate comprehensive county-level summary statistics."""
    logger.info("Generating county-level summary statistics")
    
    # Aggregate by county using weights
    county_summary = ctc_results_df.groupby('county').agg({
        'filer_id': 'count',  # Number of tax units
        'weight': 'sum',      # Total weighted population
        'income': lambda x: np.average(x, weights=ctc_results_df.loc[x.index, 'weight']),  # Weighted avg income
        'num_dependents': lambda x: np.average(x, weights=ctc_results_df.loc[x.index, 'weight']),  # Weighted avg dependents
        'ctc_total': lambda x: np.sum(x * ctc_results_df.loc[x.index, 'weight']),  # Total weighted CTC
        'ctc_refundable': lambda x: np.sum(x * ctc_results_df.loc[x.index, 'weight']),  # Total weighted refundable CTC
        'ctc_nonrefundable': lambda x: np.sum(x * ctc_results_df.loc[x.index, 'weight']),  # Total weighted non-refundable CTC
        'qualifying_children': lambda x: np.sum(x * ctc_results_df.loc[x.index, 'weight'])  # Total weighted qualifying children
    }).round(2)
    
    # Rename columns for clarity
    county_summary.columns = [
        'tax_units_count',
        'weighted_population',
        'avg_income',
        'avg_dependents',
        'total_ctc',
        'total_ctc_refundable',
        'total_ctc_nonrefundable',
        'total_qualifying_children'
    ]
    
    # Calculate per-capita and per-tax-unit metrics
    county_summary['ctc_per_capita'] = county_summary['total_ctc'] / county_summary['weighted_population']
    county_summary['ctc_per_tax_unit'] = county_summary['total_ctc'] / county_summary['tax_units_count']
    county_summary['pct_receiving_ctc'] = (
        ctc_results_df[ctc_results_df['ctc_total'] > 0]
        .groupby('county')['weight'].sum() / county_summary['weighted_population'] * 100
    ).fillna(0)
    
    return county_summary

def main():
    """Main analysis workflow."""
    logger.info("Starting county-level CTC analysis with PUMA 0100 imputation")
    
    try:
        # Step 1: Load tax units with geography
        tax_units = load_tax_units_with_geography()
        
        # Step 2: Apply county mapping with imputation
        tax_units_with_counties = apply_county_mapping(tax_units)
        
        # Step 3: Calculate CTC by county
        ctc_results = calculate_ctc_by_county(tax_units_with_counties)
        
        # Step 4: Generate county summary
        county_summary = generate_county_summary(ctc_results)
        
        # Step 5: Save results
        output_dir = Path('output/county_ctc_analysis')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        ctc_results.to_parquet(output_dir / 'ctc_results_by_tax_unit.parquet', index=False)
        county_summary.to_csv(output_dir / 'county_ctc_summary.csv')
        
        # Display results
        print("\n" + "="*80)
        print("HAWAII CHILD TAX CREDIT ANALYSIS BY COUNTY (2023)")
        print("="*80)
        print(county_summary.round(0))
        
        print(f"\nResults saved to {output_dir}/")
        
        # Log completion
        logger.info("County-level CTC analysis completed successfully")
        
        return county_summary
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
