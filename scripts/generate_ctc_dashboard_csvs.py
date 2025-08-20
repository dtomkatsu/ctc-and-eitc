#!/usr/bin/env python3
"""
Generate CTC dashboard CSVs in the same format as SNAP benefits files.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Configuration
OUTPUT_DIR = Path("CTC_dashboard_csvs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Input files
CTC_DATA_FILE = "data/processed/hawaii_ctc_full_population_20250819_132333.parquet"
PUMA_CROSSWALK = "data/crosswalks/hawaii_puma_districts.csv"

# Define column mappings from SNAP to CTC
COLUMN_MAPPING = {
    'NAME': 'NAME',
    'state': 'state',
    'geoid': 'geoid',
    'district': 'district',  # For legislative districts
    'total_households': 'total_households',
    'snap_households': 'ctc_households',  # Households receiving CTC
    'snap_households_adjusted': 'ctc_households_adjusted',  # Weighted count
    'snap_household_rate': 'ctc_household_rate',  # % of households receiving CTC
    'snap_benefit_monthly_per_household': 'ctc_benefit_annual_per_household',  # Annual CTC
    'snap_benefit_annual_per_household': 'ctc_benefit_annual_per_household',
    'snap_benefits_annual_total': 'ctc_benefits_annual_total',
    'median_income': 'median_income',
    'renter_occupied': 'renter_occupied',
    'total_housing_units': 'total_housing_units'
}

def load_ctc_data():
    """Load the CTC data from the processed data directory."""
    print(f"Loading CTC data from {CTC_DATA_FILE}...")
    if not os.path.exists(CTC_DATA_FILE):
        raise FileNotFoundError(f"Could not find CTC data at {CTC_DATA_FILE}")
    
    # Load the data
    df = pd.read_parquet(CTC_DATA_FILE)
    
    # Print available columns for debugging
    print("Available columns in the data:")
    print(df.columns.tolist())
    
    # Print unique counties and their counts
    if 'county' in df.columns:
        print("\nCounty distribution in the data:")
        print(df['county'].value_counts())
        
        # Print sample rows per county
        print("\nSample rows per county:")
        for county in df['county'].unique():
            sample = df[df['county'] == county].head(1)
            print(f"\n{county} county sample:")
            print(sample[['filer_id', 'filing_status', 'income', 'num_children', 'county']].to_string())
    
    # Map the data to our expected format
    # Use hh_id as the household identifier
    if 'hh_id' in df.columns:
        df['SERIALNO'] = df['hh_id']
    
    # Calculate total CTC amount
    if 'ctc_ctc_total' in df.columns:
        df['ctc'] = df['ctc_ctc_total']
    
    # Determine if household has children
    if 'num_children' in df.columns:
        df['has_children'] = df['num_children'] > 0
    
    # Ensure required columns exist with default values if missing
    required_columns = {
        'SERIALNO': 'unknown',  # Household identifier
        'PWGTP': 1,            # Person weight
        'PUMA': '15000',       # Default to Hawaii state
        'ctc': 0,              # CTC amount
        'has_children': False,  # Whether household has children
        'income': 0,           # Household income
        'county': 'Hawaii'     # County name
    }
    
    # Add missing columns with default values
    for col, default in required_columns.items():
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found, using default value: {default}")
            df[col] = default
    
    print(f"Loaded {len(df):,} records from {df['SERIALNO'].nunique():,} households")
    
    # Calculate total weighted households for context
    hh_weights = df.groupby('SERIALNO')['PWGTP'].first().sum()
    print(f"Total weighted households: {hh_weights:,}")
    print("Note: This excludes vacant housing units and group quarters with zero weights")
    
    print(f"Sample data:\n{df[['SERIALNO', 'PWGTP', 'ctc', 'has_children', 'income', 'county', 'PUMA']].head(2).to_string()}")
    return df

def generate_state_level(df):
    """Generate state-level CTC data."""
    print("Generating state-level data...")
    
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Group by household and calculate household-level metrics
    hh_df = df.groupby('SERIALNO').agg({
        'PWGTP': 'first',      # Use household weight
        'ctc': 'sum',          # Sum of CTC for household
        'has_children': 'any', # Any children in household
        'income': 'first'      # Household income
    }).reset_index()
    
    # Calculate metrics
    total_households = hh_df['PWGTP'].sum()
    
    # Calculate weighted counts
    ctc_households = (hh_df[hh_df['has_children']]['PWGTP'].sum() 
                      if 'has_children' in hh_df.columns else 0)
    
    # Calculate average CTC per household with children
    if 'ctc' in hh_df.columns and not hh_df[hh_df['ctc'] > 0].empty:
        avg_ctc_per_household = (hh_df[hh_df['ctc'] > 0]['ctc'] * 
                                hh_df[hh_df['ctc'] > 0]['PWGTP']).sum() / \
                               hh_df[hh_df['ctc'] > 0]['PWGTP'].sum()
    else:
        avg_ctc_per_household = 0
    
    # Calculate total CTC benefits
    total_ctc = (hh_df['ctc'] * hh_df['PWGTP']).sum()
    
    # Calculate median income
    median_income = hh_df['income'].median() if 'income' in hh_df.columns else 0
    
    ctc_household_rate = (ctc_households / total_households * 100 
                         if total_households > 0 else 0)
    
    # Create state-level DataFrame
    state_data = {
        'NAME': ['Hawaii'],
        'state': ['15'],  # FIPS code for Hawaii
        'geoid': ['15'],
        'total_households': [int(round(total_households))],
        'ctc_households': [int(round(ctc_households))],
        'ctc_households_adjusted': [int(round(ctc_households))],
        'ctc_household_rate': [round(ctc_household_rate, 2)],
        'ctc_benefit_annual_per_household': [round(avg_ctc_per_household, 2)],
        'ctc_benefits_annual_total': [int(round(total_ctc))],
        'median_income': [int(round(median_income))],
        'renter_occupied': [np.nan],  # Not available in this data
        'total_housing_units': [int(round(total_households))]  # Approximate
    }
    
    return pd.DataFrame(state_data)

def generate_county_level(df):
    """Generate county-level CTC data."""
    print("Generating county-level data...")
    
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Default to state-level data if no county/PUMA data is available
    if 'PUMA' not in df.columns and 'county' not in df.columns:
        print("Warning: No county/PUMA data found, using state as single county")
        return generate_state_level(df)
    
    # Map PUMA to county
    if 'PUMA' in df.columns and 'county' not in df.columns:
        df['county'] = df['PUMA'].map(puma_to_county)
    
    # Group by county
    group_by = 'county'
    
    # Group by the selected geography
    results = []
    
    # Get unique counties
    geos = df['county'].dropna().unique()
    print(f"Found {len(geos)} counties: {', '.join(geos)} for analysis")
    
    # Print record count per county
    print("\nRecord count by county:")
    print(df['county'].value_counts())
    
    # Hawaii 2020 PUMA to County mapping
    puma_to_county = {
        '00100': 'Honolulu',
        '00200': 'Honolulu',
        '00301': 'Hawaii',
        '00302': 'Hawaii',
        '00303': 'Hawaii',
        '00304': 'Hawaii',
        '00305': 'Maui',
        '00306': 'Maui',
        '00307': 'Maui',
        '00308': 'Kauai'
    }
    
    # County FIPS codes for Hawaii counties
    county_fips = {
        'Honolulu': '15003',
        'Hawaii': '15001',
        'Maui': '15009',
        'Kauai': '15007'
    }
    
    for geo in geos:
        geo_df = df[df['county'] == geo]
        geo_name = f"{geo} County"
        geo_id = county_fips.get(geo, '15000')  # Default to state FIPS if county not found
        
        # Group by household for this geographic unit
        hh_df = geo_df.groupby('SERIALNO').agg({
            'PWGTP': 'first',
            'ctc': 'sum',
            'has_children': 'any',
            'income': 'first'
        }).reset_index()
        
        # Calculate metrics
        total_households = hh_df['PWGTP'].sum()
        
        # Calculate weighted counts
        ctc_households = (hh_df[hh_df['has_children']]['PWGTP'].sum() 
                         if 'has_children' in hh_df.columns else 0)
        
        # Calculate average CTC per household with children
        if 'ctc' in hh_df and not hh_df[hh_df['ctc'] > 0].empty:
            avg_ctc = (hh_df[hh_df['ctc'] > 0]['ctc'] * 
                      hh_df[hh_df['ctc'] > 0]['PWGTP']).sum() / \
                     hh_df[hh_df['ctc'] > 0]['PWGTP'].sum()
        else:
            avg_ctc = 0
        
        total_ctc = (hh_df['ctc'] * hh_df['PWGTP']).sum()
        median_income = hh_df['income'].median() if 'income' in hh_df.columns else 0
        
        ctc_rate = (ctc_households / total_households * 100 
                   if total_households > 0 else 0)
        
        results.append({
            'NAME': geo_name,
            'state': '15',
            'geoid': geo_id,
            'total_households': int(round(total_households)),
            'ctc_households': int(round(ctc_households)),
            'ctc_households_adjusted': int(round(ctc_households)),
            'ctc_household_rate': round(ctc_rate, 2),
            'ctc_benefit_annual_per_household': round(avg_ctc, 2),
            'ctc_benefits_annual_total': int(round(total_ctc)),
            'median_income': int(round(median_income)),
            'renter_occupied': np.nan,
            'total_housing_units': int(round(total_households))
        })
    
    # If no results, return state-level data
    if not results:
        print("No county/PUMA data found, falling back to state-level data")
        return generate_state_level(df)
    
    # Convert to DataFrame
    return pd.DataFrame(results)

def generate_legislative_districts(df, chamber='house'):
    """Generate legislative district level CTC data."""
    print(f"Generating {chamber} district data...")
    
    # Check if we have the crosswalk file
    if not os.path.exists(PUMA_CROSSWALK):
        print(f"Warning: PUMA crosswalk file not found at {PUMA_CROSSWALK}")
        return pd.DataFrame(columns=COLUMN_MAPPING.keys())
    
    # For now, return an empty DataFrame with the right structure
    # In a full implementation, you would:
    # 1. Load the PUMA to district crosswalk
    # 2. Merge with the PUMS data
    # 3. Aggregate by district
    return pd.DataFrame(columns=COLUMN_MAPPING.keys())

def main():
    """Main function to generate all CTC dashboard CSVs."""
    try:
        # Load the data
        df = load_ctc_data()
        
        # Generate state-level data
        print("\nGenerating state-level data...")
        state_df = generate_state_level(df)
        
        # Generate county-level data
        print("\nGenerating county-level data...")
        county_df = generate_county_level(df)
        
        # Save the results
        OUTPUT_DIR.mkdir(exist_ok=True)
        
        state_csv = OUTPUT_DIR / "hawaii_state_ctc_2023.csv"
        county_csv = OUTPUT_DIR / "hawaii_county_ctc_2023.csv"
        
        print(f"\nSaving state data to {state_csv}")
        state_df.to_csv(state_csv, index=False)
        
        print(f"Saving county data to {county_csv}")
        county_df.to_csv(county_csv, index=False)
        
        # Print summary
        print("\nSummary of generated data:")
        print(f"- State file: {state_csv} ({os.path.getsize(state_csv):,} bytes)")
        print(f"- County file: {county_csv} ({os.path.getsize(county_csv):,} bytes)")
        
        # Print sample of state data
        print("\nSample state data:")
        print(state_df.head().to_string())
        
        # Print sample of county data
        print("\nSample county data:")
        print(county_df.head().to_string())
        
        print(f"\nSuccessfully generated CTC dashboard CSVs in {OUTPUT_DIR.absolute()}")
        return 0
        
    except Exception as e:
        print(f"\nError generating CTC dashboard CSVs: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
