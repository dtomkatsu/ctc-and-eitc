#!/usr/bin/env python3
"""
Diagnostic script to analyze Honolulu population count discrepancy.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

def analyze_puma_population():
    """Analyze population distribution by PUMA and county."""
    print("=== PUMA Population Analysis ===")
    
    # Load the processed PUMS data
    try:
        person_df = pd.read_parquet('data/processed/pums_person_processed.parquet')
        print(f"Loaded {len(person_df):,} persons from processed data")
    except FileNotFoundError:
        print("Processed data not found. Loading from raw PUMS data...")
        # Fallback to raw data loading
        from data.pums_loader import PUMSDataLoader
        loader = PUMSDataLoader()
        person_df, hh_df = loader.load_data(year=2023)
    
    print(f"\nTotal persons: {len(person_df):,}")
    print(f"Total households: {person_df['SERIALNO'].nunique():,}")
    
    # Analyze by PUMA
    puma_counts = person_df.groupby('PUMA').agg({
        'PWGTP': 'sum',  # Weighted population
        'SERIALNO': 'count'  # Sample size
    }).round(0).astype(int)
    
    puma_counts.columns = ['Weighted_Population', 'Sample_Size']
    puma_counts = puma_counts.sort_values('Weighted_Population', ascending=False)
    
    print(f"\n=== Population by PUMA (Top 10) ===")
    print(puma_counts.head(10))
    
    # CORRECTED crosswalk based on population analysis
    crosswalk_data = [
        # CORRECTED: Honolulu County gets 6 PUMAs (100, 200, 301, 302, 303, 304)
        {'PUMA': '00100', 'county': 'Honolulu', 'island': 'Oahu', 'region': 'West/Central Oahu'},
        {'PUMA': '00200', 'county': 'Honolulu', 'island': 'Oahu', 'region': 'East Honolulu/Windward'},
        {'PUMA': '00301', 'county': 'Honolulu', 'island': 'Oahu', 'region': 'Urban Honolulu West'},
        {'PUMA': '00302', 'county': 'Honolulu', 'island': 'Oahu', 'region': 'Urban Honolulu Central'},
        {'PUMA': '00303', 'county': 'Honolulu', 'island': 'Oahu', 'region': 'Urban Honolulu East'},
        {'PUMA': '00304', 'county': 'Honolulu', 'island': 'Oahu', 'region': 'North Shore/Central'},
        
        # Hawaii County gets 1 PUMA (305)
        {'PUMA': '00305', 'county': 'Hawaii', 'island': 'Hawaii', 'region': 'Hilo/Puna'},
        
        # Maui County gets 2 PUMAs (306, 307)
        {'PUMA': '00306', 'county': 'Maui', 'island': 'Maui', 'region': 'Central/West Maui'},
        {'PUMA': '00307', 'county': 'Maui', 'island': 'Molokai/Lanai', 'region': 'Molokai/Lanai'},
        
        # Kauai County gets 1 PUMA (308)
        {'PUMA': '00308', 'county': 'Kauai', 'island': 'Kauai', 'region': 'Kauai'},
    ]
    
    crosswalk_df = pd.DataFrame(crosswalk_data)
    crosswalk_df['PUMA'] = crosswalk_df['PUMA'].str.zfill(5)
    
    # Check what PUMAs actually exist in the data
    actual_pumas = set(person_df['PUMA'].astype(str).str.zfill(5).unique())
    crosswalk_pumas = set(crosswalk_df['PUMA'].unique())
    
    print(f"\n=== PUMA Mapping Analysis ===")
    print(f"PUMAs in data: {sorted(actual_pumas)}")
    print(f"PUMAs in crosswalk: {sorted(crosswalk_pumas)}")
    print(f"Missing from crosswalk: {sorted(actual_pumas - crosswalk_pumas)}")
    print(f"Extra in crosswalk: {sorted(crosswalk_pumas - actual_pumas)}")
    
    # Merge with crosswalk to get county assignments
    person_df['PUMA_str'] = person_df['PUMA'].astype(str).str.zfill(5)
    merged_df = person_df.merge(crosswalk_df, left_on='PUMA_str', right_on='PUMA', how='left')
    
    # Check for unmapped records
    unmapped = merged_df[merged_df['county'].isna()]
    if len(unmapped) > 0:
        print(f"\n=== UNMAPPED RECORDS ===")
        print(f"Unmapped persons: {len(unmapped):,}")
        print(f"Unmapped PUMAs: {sorted(unmapped['PUMA_str'].unique())}")
        
        unmapped_counts = unmapped.groupby('PUMA_str')['PWGTP'].sum().sort_values(ascending=False)
        print("Population by unmapped PUMA:")
        print(unmapped_counts)
    
    # County-level analysis
    county_counts = merged_df.groupby('county').agg({
        'PWGTP': 'sum',
        'SERIALNO': 'count'
    }).round(0).astype(int)
    
    county_counts.columns = ['Weighted_Population', 'Sample_Size']
    county_counts = county_counts.sort_values('Weighted_Population', ascending=False)
    
    print(f"\n=== Population by County ===")
    print(county_counts)
    
    # Expected Hawaii county populations (2023 estimates)
    expected_populations = {
        'Honolulu': 1016508,  # Should be ~70% of state
        'Hawaii': 200629,     # Big Island
        'Maui': 164221,       # Maui County
        'Kauai': 73298        # Kauai County
    }
    
    print(f"\n=== Comparison with Expected Populations ===")
    total_expected = sum(expected_populations.values())
    
    for county in expected_populations:
        if county in county_counts.index:
            actual = county_counts.loc[county, 'Weighted_Population']
            expected = expected_populations[county]
            pct_expected = expected / total_expected * 100
            pct_actual = actual / county_counts['Weighted_Population'].sum() * 100
            
            print(f"{county}:")
            print(f"  Expected: {expected:,} ({pct_expected:.1f}%)")
            print(f"  Actual:   {actual:,} ({pct_actual:.1f}%)")
            print(f"  Ratio:    {actual/expected:.2f}")
        else:
            print(f"{county}: NOT FOUND in data")
    
    return person_df, crosswalk_df

def check_puma_definitions():
    """Check official PUMA definitions for Hawaii."""
    print(f"\n=== Official Hawaii PUMA Definitions (2020 Census) ===")
    
    # Official 2020 PUMA definitions for Hawaii
    official_pumas = {
        '01501': 'Oahu--Honolulu City (Ewa, Waianae, Pearl City, Aiea)',
        '01502': 'Oahu--Honolulu City (Kalihi, Keeaumoku, Kalihi-Palama)', 
        '01503': 'Oahu--Honolulu City (Downtown, Chinatown, Nuuanu, Liliha)',
        '01504': 'Oahu--Honolulu City (Kaimuki, Kapahulu, Diamond Head, Kahala)',
        '01505': 'Oahu--Honolulu City (Manoa, Moiliili, McCully, Makiki)',
        '01506': 'Oahu--Honolulu City (Kailua, Kaneohe, Windward Coast)',
        '01507': 'Oahu--Honolulu City (North Shore, Central Oahu)',
        '01508': 'Hawaii County (Hilo, Puna)',
        '01509': 'Hawaii County (Kona, Kohala, Hamakua)',
        '01510': 'Maui County (Maui Island)',
        '01511': 'Maui County (Molokai, Lanai)',
        '01512': 'Kauai County'
    }
    
    for puma, description in official_pumas.items():
        print(f"  {puma}: {description}")
    
    return official_pumas

if __name__ == "__main__":
    person_df, crosswalk_df = analyze_puma_population()
    official_pumas = check_puma_definitions()
