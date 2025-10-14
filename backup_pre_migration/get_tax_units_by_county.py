#!/usr/bin/env python3
"""
Generate tax unit counts by county using the corrected PUMA mapping.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.pums_loader import PUMSDataLoader
from src.tax.units.constructor import TaxUnitConstructor

def load_data():
    """Load and process PUMS data with corrected county mapping."""
    print("Loading PUMS data with corrected county mapping...")
    
    # Point to the directory containing the 2023 PUMS data files using full path
    loader = PUMSDataLoader(data_dir=Path("/Users/dtomkatsu/CascadeProjects/ctc-and-eitc/data/raw/pums/"))
    person_df, hh_df = loader.load_data(year=2023)
    
    # Apply the corrected crosswalk
    crosswalk = [
        # Honolulu County (Oahu)
        {'PUMA': '00100', 'county': 'Honolulu'},
        {'PUMA': '00200', 'county': 'Honolulu'},
        {'PUMA': '00301', 'county': 'Honolulu'},
        {'PUMA': '00302', 'county': 'Honolulu'},
        {'PUMA': '00303', 'county': 'Honolulu'},
        {'PUMA': '00304', 'county': 'Honolulu'},
        
        # Hawaii County
        {'PUMA': '00305', 'county': 'Hawaii'},
        
        # Maui County
        {'PUMA': '00306', 'county': 'Maui'},
        {'PUMA': '00307', 'county': 'Maui'},
        
        # Kauai County
        {'PUMA': '00308', 'county': 'Kauai'},
    ]
    
    crosswalk_df = pd.DataFrame(crosswalk)
    crosswalk_df['PUMA'] = crosswalk_df['PUMA'].astype(str).str.zfill(5)
    
    # Merge with person data
    person_df['PUMA_str'] = person_df['PUMA'].astype(str).str.zfill(5)
    person_df = person_df.merge(crosswalk_df, left_on='PUMA_str', right_on='PUMA', how='left')
    
    # Create tax units using the corrected mapping
    print("Creating tax units...")
    constructor = TaxUnitConstructor(person_df, hh_df)
    tax_units_df = constructor.create_rule_based_units(parallel=False)
    
    # Merge back county information
    tax_units_df = tax_units_df.merge(
        person_df[['SERIALNO', 'county']].drop_duplicates(),
        on='SERIALNO',
        how='left'
    )
    
    return tax_units_df

def analyze_tax_units(tax_units_df):
    """Analyze tax units by county."""
    print("\n=== Tax Units by County ===")
    
    # Basic counts
    county_counts = tax_units_df['county'].value_counts().reset_index()
    county_counts.columns = ['County', 'Tax_Unit_Count']
    
    # Add weighted counts
    county_counts['Weighted_Count'] = county_counts['County'].map(
        tax_units_df.groupby('county')['weight'].sum()
    ).round(0).astype(int)
    
    # Calculate percentages
    total_weighted = county_counts['Weighted_Count'].sum()
    county_counts['Percentage'] = (county_counts['Weighted_Count'] / total_weighted * 100).round(1)
    
    # Format output
    county_counts = county_counts.sort_values('Weighted_Count', ascending=False)
    county_counts['Weighted_Count'] = county_counts['Weighted_Count'].apply(lambda x: f"{x:,}")
    county_counts['Percentage'] = county_counts['Percentage'].apply(lambda x: f"{x:.1f}%")
    
    print(county_counts[['County', 'Tax_Unit_Count', 'Weighted_Count', 'Percentage']].to_string(index=False))
    
    # Add breakdown by filing status
    print("\n=== Tax Units by County and Filing Status ===")
    if 'filing_status' in tax_units_df.columns:
        status_by_county = tax_units_df.groupby(['county', 'filing_status'])['weight'].sum().unstack().fillna(0).astype(int)
        status_by_county['Total'] = status_by_county.sum(axis=1)
        
        # Calculate percentages
        for col in status_by_county.columns:
            if col != 'Total':
                status_by_county[f"{col}_pct"] = (status_by_county[col] / status_by_county['Total'] * 100).round(1)
        
        print(status_by_county)
    else:
        print("Warning: Filing status not found in tax units data")
    
    return county_counts

def main():
    # Load data and create tax units
    tax_units_df = load_data()
    
    # Analyze and print results
    county_analysis = analyze_tax_units(tax_units_df)
    
    # Save results
    output_dir = Path('output/tax_units')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'tax_units_by_county_{timestamp}.csv'
    
    county_analysis.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
