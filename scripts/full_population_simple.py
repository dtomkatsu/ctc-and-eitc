"""
Simplified Full Population CTC Analysis

Sequential processing optimized for reliability and comprehensive results.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from tax.credits.ctc import calculate_ctc


def load_full_pums_data():
    """Load complete PUMS data."""
    print("Loading full PUMS dataset...")
    
    person_df = pd.read_parquet('data/processed/pums_person_processed.parquet')
    
    print(f"Loaded {len(person_df):,} persons from {person_df['SERIALNO'].nunique():,} households")
    print(f"Geographic coverage: {person_df['PUMA'].nunique()} PUMAs")
    
    return person_df


def create_hawaii_legislative_crosswalk():
    """Create detailed PUMA to legislative district crosswalk for Hawaii."""
    print("Creating Hawaii legislative district crosswalk...")
    
    crosswalk_data = [
        # Oahu PUMAs
        {'PUMA': '00100', 'county': 'Honolulu', 'island': 'Oahu', 'region': 'Urban Honolulu',
         'senate_district': 'SD12', 'house_district': 'HD23', 'senate_name': 'Downtown/Chinatown', 'house_name': 'Downtown/Nuuanu'},
        
        {'PUMA': '00200', 'county': 'Honolulu', 'island': 'Oahu', 'region': 'East Honolulu', 
         'senate_district': 'SD01', 'house_district': 'HD01', 'senate_name': 'Hawaii Kai/Portlock', 'house_name': 'Hawaii Kai'},
        
        # Big Island PUMAs  
        {'PUMA': '00301', 'county': 'Hawaii', 'island': 'Hawaii', 'region': 'Hilo',
         'senate_district': 'SD02', 'house_district': 'HD04', 'senate_name': 'Hilo/Hamakua', 'house_name': 'Hilo'},
        
        {'PUMA': '00302', 'county': 'Hawaii', 'island': 'Hawaii', 'region': 'Kona',
         'senate_district': 'SD03', 'house_district': 'HD05', 'senate_name': 'Kona/Kohala', 'house_name': 'North/South Kona'},
        
        {'PUMA': '00303', 'county': 'Hawaii', 'island': 'Hawaii', 'region': 'Puna',
         'senate_district': 'SD02', 'house_district': 'HD04', 'senate_name': 'Hilo/Hamakua', 'house_name': 'Hilo'},
        
        {'PUMA': '00304', 'county': 'Hawaii', 'island': 'Hawaii', 'region': 'Ka\'u',
         'senate_district': 'SD03', 'house_district': 'HD05', 'senate_name': 'Kona/Kohala', 'house_name': 'North/South Kona'},
        
        # Maui County PUMAs
        {'PUMA': '00305', 'county': 'Maui', 'island': 'Maui', 'region': 'West Maui',
         'senate_district': 'SD05', 'house_district': 'HD09', 'senate_name': 'West/South Maui', 'house_name': 'West Maui'},
        
        {'PUMA': '00306', 'county': 'Maui', 'island': 'Maui', 'region': 'Central Maui',
         'senate_district': 'SD06', 'house_district': 'HD10', 'senate_name': 'Central Maui', 'house_name': 'Wailuku/Waihee'},
        
        {'PUMA': '00307', 'county': 'Maui', 'island': 'Molokai', 'region': 'Molokai',
         'senate_district': 'SD05', 'house_district': 'HD13', 'senate_name': 'West/South Maui', 'house_name': 'Molokai/Lanai'},
        
        # Kauai PUMA
        {'PUMA': '00308', 'county': 'Kauai', 'island': 'Kauai', 'region': 'Kauai',
         'senate_district': 'SD07', 'house_district': 'HD14', 'senate_name': 'Kauai/Niihau', 'house_name': 'Kauai'},
    ]
    
    crosswalk_df = pd.DataFrame(crosswalk_data)
    crosswalk_df['PUMA'] = crosswalk_df['PUMA'].str.zfill(5)
    
    print(f"Created crosswalk for {len(crosswalk_df)} PUMAs")
    return crosswalk_df


def create_tax_units_efficient(person_df):
    """Create tax units efficiently using vectorized operations."""
    print("Creating tax units using efficient processing...")
    
    tax_units = []
    total_households = person_df['SERIALNO'].nunique()
    processed = 0
    
    for hh_id, hh_group in person_df.groupby('SERIALNO'):
        processed += 1
        
        # Progress indicator
        if processed % 5000 == 0:
            print(f"  Processed {processed:,}/{total_households:,} households ({processed/total_households*100:.1f}%)")
        
        # Basic household info
        adults = hh_group[hh_group['AGEP'] >= 18]
        children = hh_group[hh_group['AGEP'] < 18]
        
        if len(adults) == 0:
            continue
        
        # Filing status logic
        if len(adults) >= 2:
            married = adults[adults['MAR'] == 1]
            if len(married) >= 2:
                filing_status = 'married_filing_jointly'
            else:
                filing_status = 'head_of_household' if len(children) > 0 else 'single'
        else:
            filing_status = 'head_of_household' if len(children) > 0 else 'single'
        
        # Create dependents
        dependents = []
        for _, child in children.iterrows():
            if child['AGEP'] < 17:
                dependents.append({
                    'age': child['AGEP'],
                    'relationship': '22',
                    'citizenship': '1'
                })
        
        # Calculate income
        total_income = hh_group['PINCP'].fillna(0).sum()
        primary_adult = adults.iloc[0]
        
        tax_unit = {
            'filer_id': f"{hh_id}_1",
            'filing_status': filing_status,
            'income': max(0, total_income),
            'dependents': dependents,
            'num_dependents': len(dependents),
            'hh_id': hh_id,
            'PUMA': str(primary_adult['PUMA']).zfill(5),
            'state': 'Hawaii',
            'PWGTP': primary_adult['PWGTP'],
            'num_adults': len(adults),
            'num_children': len(children),
            'qualifying_children': len(dependents)
        }
        
        tax_units.append(tax_unit)
    
    tax_units_df = pd.DataFrame(tax_units)
    print(f"Created {len(tax_units_df):,} tax units")
    
    return tax_units_df


def calculate_ctc_efficient(tax_units_df):
    """Calculate CTC efficiently with progress tracking."""
    print(f"Calculating CTC for {len(tax_units_df):,} tax units...")
    
    results = []
    total_units = len(tax_units_df)
    
    for idx, tax_unit in tax_units_df.iterrows():
        # Progress indicator
        if (idx + 1) % 2000 == 0:
            print(f"  Processed {idx + 1:,}/{total_units:,} units ({(idx + 1)/total_units*100:.1f}%)")
        
        # Calculate CTC
        ctc_result = calculate_ctc(tax_unit.to_dict())
        
        # Combine results
        result = tax_unit.to_dict()
        for key, value in ctc_result.items():
            result[f'ctc_{key}'] = value
        
        results.append(result)
    
    return pd.DataFrame(results)


def comprehensive_analysis(tax_units_with_ctc, crosswalk_df):
    """Comprehensive analysis with legislative districts."""
    print("\n" + "="*80)
    print("COMPREHENSIVE HAWAII CTC ANALYSIS - FULL POPULATION")
    print("="*80)
    
    # Add legislative district information
    tax_units_with_ctc['PUMA'] = tax_units_with_ctc['PUMA'].astype(str).str.zfill(5)
    crosswalk_df['PUMA'] = crosswalk_df['PUMA'].astype(str).str.zfill(5)
    
    analysis_df = tax_units_with_ctc.merge(crosswalk_df, on='PUMA', how='left')
    
    # Overall statistics
    total_units = len(analysis_df)
    units_with_ctc = len(analysis_df[analysis_df['ctc_ctc_total'] > 0])
    total_ctc = analysis_df['ctc_ctc_total'].sum()
    weighted_total_ctc = (analysis_df['ctc_ctc_total'] * analysis_df['PWGTP']).sum()
    weighted_total_units = analysis_df['PWGTP'].sum()
    
    print(f"\nSTATEWIDE SUMMARY:")
    print(f"Total Tax Units: {total_units:,}")
    print(f"Units with CTC: {units_with_ctc:,}")
    print(f"CTC Participation Rate: {units_with_ctc/total_units*100:.1f}%")
    print(f"Total CTC Amount: ${total_ctc:,.2f}")
    print(f"Weighted Total CTC: ${weighted_total_ctc:,.2f}")
    print(f"Average CTC per Unit: ${total_ctc/total_units:.2f}")
    print(f"Weighted Average CTC: ${weighted_total_ctc/weighted_total_units:.2f}")
    if units_with_ctc > 0:
        print(f"Average CTC per Recipient: ${total_ctc/units_with_ctc:.2f}")
    
    # County-level analysis
    print(f"\nCOUNTY-LEVEL ANALYSIS:")
    print("-" * 25)
    
    for county in sorted(analysis_df['county'].dropna().unique()):
        subset = analysis_df[analysis_df['county'] == county]
        count = len(subset)
        ctc_recipients = len(subset[subset['ctc_ctc_total'] > 0])
        total_ctc_county = subset['ctc_ctc_total'].sum()
        weighted_ctc = (subset['ctc_ctc_total'] * subset['PWGTP']).sum()
        avg_ctc = subset['ctc_ctc_total'].mean()
        
        print(f"{county} County:")
        print(f"  Tax Units: {count:,}")
        print(f"  CTC Recipients: {ctc_recipients:,} ({ctc_recipients/count*100:.1f}%)")
        print(f"  Total CTC: ${total_ctc_county:,.2f}")
        print(f"  Weighted CTC: ${weighted_ctc:,.2f}")
        print(f"  Average CTC: ${avg_ctc:.2f}")
    
    # Senate district analysis
    print(f"\nSENATE DISTRICT ANALYSIS:")
    print("-" * 27)
    print(f"{'District':<12} {'Name':<25} {'Units':<8} {'Recipients':<12} {'Total CTC':<15}")
    print("-" * 85)
    
    for district in sorted(analysis_df['senate_district'].dropna().unique()):
        subset = analysis_df[analysis_df['senate_district'] == district]
        count = len(subset)
        recipients = len(subset[subset['ctc_ctc_total'] > 0])
        total_ctc_district = subset['ctc_ctc_total'].sum()
        district_name = subset['senate_name'].iloc[0] if len(subset) > 0 else 'Unknown'
        
        print(f"{district:<12} {district_name:<25} {count:>6} {recipients:>10} ${total_ctc_district:>12,.0f}")
    
    # House district analysis
    print(f"\nHOUSE DISTRICT ANALYSIS:")
    print("-" * 25)
    print(f"{'District':<12} {'Name':<25} {'Units':<8} {'Recipients':<12} {'Total CTC':<15}")
    print("-" * 85)
    
    for district in sorted(analysis_df['house_district'].dropna().unique()):
        subset = analysis_df[analysis_df['house_district'] == district]
        count = len(subset)
        recipients = len(subset[subset['ctc_ctc_total'] > 0])
        total_ctc_district = subset['ctc_ctc_total'].sum()
        district_name = subset['house_name'].iloc[0] if len(subset) > 0 else 'Unknown'
        
        print(f"{district:<12} {district_name:<25} {count:>6} {recipients:>10} ${total_ctc_district:>12,.0f}")
    
    return analysis_df


def save_results(analysis_df):
    """Save comprehensive results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\nSAVING RESULTS:")
    print("-" * 15)
    
    # Save full dataset
    full_file = f'data/processed/hawaii_ctc_full_population_{timestamp}.parquet'
    analysis_df.to_parquet(full_file)
    print(f"Full analysis: {full_file}")
    
    # County summary
    county_summary = analysis_df.groupby('county').agg({
        'ctc_ctc_total': ['count', 'sum', 'mean'],
        'ctc_qualifying_children': 'sum',
        'PWGTP': 'sum'
    }).round(2)
    county_file = f'data/processed/hawaii_ctc_by_county_{timestamp}.csv'
    county_summary.to_csv(county_file)
    print(f"County summary: {county_file}")
    
    # Senate district summary
    senate_summary = analysis_df.groupby(['senate_district', 'senate_name']).agg({
        'ctc_ctc_total': ['count', 'sum', 'mean'],
        'ctc_qualifying_children': 'sum',
        'PWGTP': 'sum'
    }).round(2)
    senate_file = f'data/processed/hawaii_ctc_by_senate_{timestamp}.csv'
    senate_summary.to_csv(senate_file)
    print(f"Senate districts: {senate_file}")
    
    # House district summary
    house_summary = analysis_df.groupby(['house_district', 'house_name']).agg({
        'ctc_ctc_total': ['count', 'sum', 'mean'],
        'ctc_qualifying_children': 'sum',
        'PWGTP': 'sum'
    }).round(2)
    house_file = f'data/processed/hawaii_ctc_by_house_{timestamp}.csv'
    house_summary.to_csv(house_file)
    print(f"House districts: {house_file}")


def main():
    """Run full population analysis."""
    print("Hawaii Full Population CTC Analysis")
    print("="*40)
    start_time = datetime.now()
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    person_df = load_full_pums_data()
    crosswalk_df = create_hawaii_legislative_crosswalk()
    
    # Create tax units
    tax_units_df = create_tax_units_efficient(person_df)
    
    if len(tax_units_df) == 0:
        print("No tax units created. Exiting.")
        return
    
    # Calculate CTC
    tax_units_with_ctc = calculate_ctc_efficient(tax_units_df)
    
    # Analysis
    analysis_df = comprehensive_analysis(tax_units_with_ctc, crosswalk_df)
    
    # Save results
    save_results(analysis_df)
    
    # Timing
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    print(f"\nCompleted in {total_time/60:.1f} minutes")
    print(f"Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
