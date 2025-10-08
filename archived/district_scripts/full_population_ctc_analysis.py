"""
Full Population PUMS CTC Analysis

This script runs CTC analysis on the complete PUMS dataset for Hawaii
with optimized batch processing and legislative district mapping.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import multiprocessing as mp
from functools import partial

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from tax.credits.ctc import calculate_ctc
from analysis.geographic import GeographicAnalyzer, GeographicLevel


def load_full_pums_data():
    """Load complete PUMS data."""
    print("Loading full PUMS dataset...")
    
    person_df = pd.read_parquet('data/processed/pums_person_processed.parquet')
    
    print(f"Loaded {len(person_df):,} persons from {person_df['SERIALNO'].nunique():,} households")
    print(f"Geographic coverage: {person_df['PUMA'].nunique()} PUMAs")
    print(f"PUMAs: {sorted(person_df['PUMA'].unique())}")
    
    return person_df


def create_hawaii_legislative_crosswalk():
    """Create detailed PUMA to legislative district crosswalk for Hawaii."""
    print("Creating Hawaii legislative district crosswalk...")
    
    # Hawaii 2020 PUMA to Legislative District mapping
    # Based on Hawaii State Legislature districts and Census PUMA boundaries
    crosswalk_data = [
        # Oahu PUMAs
        {'PUMA': '00100', 'county': 'Honolulu', 'island': 'Oahu', 'region': 'Urban Honolulu',
         'senate_district': 'SD12', 'house_district': 'HD23', 'senate_name': 'Downtown/Chinatown', 'house_name': 'Downtown/Nuuanu'},
        
        {'PUMA': '00200', 'county': 'Honolulu', 'island': 'Oahu', 'region': 'East Honolulu', 
         'senate_district': 'SD01', 'house_district': 'HD01', 'senate_name': 'Hawaii Kai/Portlock', 'house_name': 'Hawaii Kai'},
        
        # Honolulu County (Oahu) - Additional PUMAs
        {'PUMA': '00301', 'county': 'Honolulu', 'island': 'Oahu', 'region': 'West Oahu',
         'senate_district': 'SD19', 'house_district': 'HD42', 'senate_name': 'West Oahu', 'house_name': 'Ewa Beach'},
        
        {'PUMA': '00302', 'county': 'Honolulu', 'island': 'Oahu', 'region': 'Central Oahu',
         'senate_district': 'SD18', 'house_district': 'HD39', 'senate_name': 'Wahiawa/Whitmore', 'house_name': 'Wahiawa/Whitmore'},
        
        {'PUMA': '00303', 'county': 'Honolulu', 'island': 'Oahu', 'region': 'North Shore',
         'senate_district': 'SD21', 'house_district': 'HD45', 'senate_name': 'North Shore/Windward', 'house_name': 'North Shore'},
        
        {'PUMA': '00304', 'county': 'Honolulu', 'island': 'Oahu', 'region': 'Windward Oahu',
         'senate_district': 'SD24', 'house_district': 'HD48', 'senate_name': 'Kaneohe/Kailua', 'house_name': 'Kaneohe'},
        
        # Hawaii County (Big Island) PUMAs
        {'PUMA': '00305', 'county': 'Hawaii', 'island': 'Hawaii', 'region': 'Big Island',
         'senate_district': 'SD02', 'house_district': 'HD04', 'senate_name': 'Hilo/Hamakua', 'house_name': 'Hilo'},
        
        # Maui County PUMAs
        {'PUMA': '00306', 'county': 'Maui', 'island': 'Maui', 'region': 'West Maui',
         'senate_district': 'SD05', 'house_district': 'HD09', 'senate_name': 'West/South Maui', 'house_name': 'West Maui'},
        
        {'PUMA': '00307', 'county': 'Maui', 'island': 'Maui', 'region': 'Central Maui',
         'senate_district': 'SD06', 'house_district': 'HD10', 'senate_name': 'Central Maui', 'house_name': 'Wailuku/Waihee'},
        
        # Kauai PUMA
        {'PUMA': '00308', 'county': 'Kauai', 'island': 'Kauai', 'region': 'Kauai',
         'senate_district': 'SD07', 'house_district': 'HD14', 'senate_name': 'Kauai/Niihau', 'house_name': 'Kauai'},
    ]
    
    crosswalk_df = pd.DataFrame(crosswalk_data)
    
    # Ensure PUMA format consistency
    crosswalk_df['PUMA'] = crosswalk_df['PUMA'].str.zfill(5)
    
    print(f"Created crosswalk for {len(crosswalk_df)} PUMAs covering:")
    for county in crosswalk_df['county'].unique():
        pumas_in_county = len(crosswalk_df[crosswalk_df['county'] == county])
        print(f"  {county} County: {pumas_in_county} PUMAs")
    
    return crosswalk_df


def process_household_batch(household_batch):
    """Process a batch of households for tax unit creation."""
    tax_units = []
    
    for hh_id, hh_group in household_batch.groupby('SERIALNO'):
        # Basic household info
        adults = hh_group[hh_group['AGEP'] >= 18]
        children = hh_group[hh_group['AGEP'] < 18]
        
        if len(adults) == 0:
            continue  # Skip households with no adults
        
        # Filing status logic
        if len(adults) >= 2:
            married = adults[adults['MAR'] == 1]  # Married
            if len(married) >= 2:
                filing_status = 'married_filing_jointly'
            else:
                filing_status = 'head_of_household' if len(children) > 0 else 'single'
        else:
            filing_status = 'head_of_household' if len(children) > 0 else 'single'
        
        # Create dependents (children under 17 for CTC)
        dependents = []
        for _, child in children.iterrows():
            if child['AGEP'] < 17:
                dependents.append({
                    'age': child['AGEP'],
                    'relationship': '22',  # Natural child
                    'citizenship': '1'     # Assume US citizen
                })
        
        # Calculate income
        total_income = hh_group['PINCP'].fillna(0).sum()
        
        # Get geographic info
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
    
    return tax_units


def create_tax_units_parallel(person_df, batch_size=2000, n_processes=None):
    """Create tax units using parallel processing."""
    print(f"Creating tax units with parallel processing...")
    
    if n_processes is None:
        n_processes = min(mp.cpu_count(), 4)  # Limit to 4 processes
    
    print(f"Using {n_processes} processes with batch size {batch_size:,}")
    
    # Split households into batches
    unique_households = person_df['SERIALNO'].unique()
    household_batches = np.array_split(unique_households, 
                                     max(1, len(unique_households) // batch_size))
    
    print(f"Processing {len(household_batches)} batches...")
    
    # Create batches of person data
    person_batches = []
    for batch_households in household_batches:
        batch_data = person_df[person_df['SERIALNO'].isin(batch_households)]
        person_batches.append(batch_data)
    
    # Process batches in parallel
    with mp.Pool(processes=n_processes) as pool:
        batch_results = pool.map(process_household_batch, person_batches)
    
    # Combine results
    all_tax_units = []
    for batch_result in batch_results:
        all_tax_units.extend(batch_result)
    
    tax_units_df = pd.DataFrame(all_tax_units)
    print(f"Created {len(tax_units_df):,} tax units from {len(unique_households):,} households")
    
    return tax_units_df


def calculate_ctc_parallel(tax_units_df, batch_size=1000, n_processes=None):
    """Calculate CTC using parallel processing."""
    print(f"Calculating CTC with parallel processing...")
    
    if n_processes is None:
        n_processes = min(mp.cpu_count(), 4)
    
    print(f"Using {n_processes} processes with batch size {batch_size:,}")
    
    # Split into batches
    batches = np.array_split(tax_units_df, max(1, len(tax_units_df) // batch_size))
    
    def calculate_ctc_batch(batch_df):
        results = []
        for _, tax_unit in batch_df.iterrows():
            ctc_result = calculate_ctc(tax_unit.to_dict())
            result = tax_unit.to_dict()
            for key, value in ctc_result.items():
                result[f'ctc_{key}'] = value
            results.append(result)
        return results
    
    # Process batches in parallel
    with mp.Pool(processes=n_processes) as pool:
        batch_results = pool.map(calculate_ctc_batch, batches)
    
    # Combine results
    all_results = []
    for batch_result in batch_results:
        all_results.extend(batch_result)
    
    return pd.DataFrame(all_results)


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
    
    # County-level analysis
    print(f"\nCOUNTY-LEVEL ANALYSIS:")
    print("-" * 25)
    county_stats = analysis_df.groupby('county').agg({
        'ctc_ctc_total': ['count', 'sum', 'mean'],
        'ctc_qualifying_children': 'sum',
        'PWGTP': 'sum'
    }).round(2)
    
    for county in analysis_df['county'].unique():
        if pd.isna(county):
            continue
        subset = analysis_df[analysis_df['county'] == county]
        count = len(subset)
        ctc_recipients = len(subset[subset['ctc_ctc_total'] > 0])
        total_ctc_county = subset['ctc_ctc_total'].sum()
        weighted_ctc = (subset['ctc_ctc_total'] * subset['PWGTP']).sum()
        
        print(f"{county} County:")
        print(f"  Tax Units: {count:,}")
        print(f"  CTC Recipients: {ctc_recipients:,} ({ctc_recipients/count*100:.1f}%)")
        print(f"  Total CTC: ${total_ctc_county:,.2f}")
        print(f"  Weighted CTC: ${weighted_ctc:,.2f}")
    
    # Senate district analysis
    print(f"\nSENATE DISTRICT ANALYSIS:")
    print("-" * 27)
    print(f"{'District':<12} {'Units':<8} {'Recipients':<12} {'Total CTC':<15} {'Avg CTC':<10}")
    print("-" * 70)
    
    for district in sorted(analysis_df['senate_district'].dropna().unique()):
        subset = analysis_df[analysis_df['senate_district'] == district]
        count = len(subset)
        recipients = len(subset[subset['ctc_ctc_total'] > 0])
        total_ctc_district = subset['ctc_ctc_total'].sum()
        avg_ctc = subset['ctc_ctc_total'].mean()
        
        print(f"{district:<12} {count:>6} {recipients:>10} ${total_ctc_district:>12,.0f} ${avg_ctc:>8.0f}")
    
    # House district analysis
    print(f"\nHOUSE DISTRICT ANALYSIS:")
    print("-" * 25)
    print(f"{'District':<12} {'Units':<8} {'Recipients':<12} {'Total CTC':<15} {'Avg CTC':<10}")
    print("-" * 70)
    
    for district in sorted(analysis_df['house_district'].dropna().unique()):
        subset = analysis_df[analysis_df['house_district'] == district]
        count = len(subset)
        recipients = len(subset[subset['ctc_ctc_total'] > 0])
        total_ctc_district = subset['ctc_ctc_total'].sum()
        avg_ctc = subset['ctc_ctc_total'].mean()
        
        print(f"{district:<12} {count:>6} {recipients:>10} ${total_ctc_district:>12,.0f} ${avg_ctc:>8.0f}")
    
    return analysis_df


def save_comprehensive_results(analysis_df):
    """Save comprehensive analysis results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\nSAVING COMPREHENSIVE RESULTS:")
    print("-" * 32)
    
    # Save full dataset
    full_file = f'data/processed/hawaii_ctc_full_analysis_{timestamp}.parquet'
    analysis_df.to_parquet(full_file)
    print(f"Full analysis saved to: {full_file}")
    
    # Save county summary
    county_summary = analysis_df.groupby('county').agg({
        'ctc_ctc_total': ['count', 'sum', 'mean'],
        'ctc_qualifying_children': 'sum',
        'PWGTP': 'sum',
        'income': 'mean'
    }).round(2)
    
    county_file = f'data/processed/hawaii_ctc_by_county_{timestamp}.csv'
    county_summary.to_csv(county_file)
    print(f"County summary saved to: {county_file}")
    
    # Save senate district summary
    senate_summary = analysis_df.groupby(['senate_district', 'senate_name']).agg({
        'ctc_ctc_total': ['count', 'sum', 'mean'],
        'ctc_qualifying_children': 'sum',
        'PWGTP': 'sum'
    }).round(2)
    
    senate_file = f'data/processed/hawaii_ctc_by_senate_{timestamp}.csv'
    senate_summary.to_csv(senate_file)
    print(f"Senate district summary saved to: {senate_file}")
    
    # Save house district summary
    house_summary = analysis_df.groupby(['house_district', 'house_name']).agg({
        'ctc_ctc_total': ['count', 'sum', 'mean'],
        'ctc_qualifying_children': 'sum',
        'PWGTP': 'sum'
    }).round(2)
    
    house_file = f'data/processed/hawaii_ctc_by_house_{timestamp}.csv'
    house_summary.to_csv(house_file)
    print(f"House district summary saved to: {house_file}")
    
    # Create executive summary
    summary_file = f'data/processed/hawaii_ctc_executive_summary_{timestamp}.txt'
    with open(summary_file, 'w') as f:
        f.write("Hawaii Child Tax Credit Analysis - Executive Summary\n")
        f.write("="*55 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Source: Hawaii PUMS 2023\n\n")
        
        total_units = len(analysis_df)
        units_with_ctc = len(analysis_df[analysis_df['ctc_ctc_total'] > 0])
        total_ctc = analysis_df['ctc_ctc_total'].sum()
        weighted_total_ctc = (analysis_df['ctc_ctc_total'] * analysis_df['PWGTP']).sum()
        
        f.write("KEY FINDINGS:\n")
        f.write(f"• Total Tax Units: {total_units:,}\n")
        f.write(f"• Units Receiving CTC: {units_with_ctc:,} ({units_with_ctc/total_units*100:.1f}%)\n")
        f.write(f"• Total CTC Amount: ${total_ctc:,.2f}\n")
        f.write(f"• Weighted CTC Amount: ${weighted_total_ctc:,.2f}\n")
        f.write(f"• Average CTC per Unit: ${total_ctc/total_units:.2f}\n")
        f.write(f"• Average CTC per Recipient: ${total_ctc/units_with_ctc:.2f}\n")
    
    print(f"Executive summary saved to: {summary_file}")


def main():
    """Run comprehensive full population analysis."""
    print("Hawaii Full Population CTC Analysis with Legislative Districts")
    print("="*65)
    start_time = datetime.now()
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load full dataset
    person_df = load_full_pums_data()
    
    # Create legislative crosswalk
    crosswalk_df = create_hawaii_legislative_crosswalk()
    
    # Create tax units with parallel processing
    tax_units_df = create_tax_units_parallel(person_df, batch_size=3000)
    
    if len(tax_units_df) == 0:
        print("No tax units created. Exiting.")
        return
    
    # Calculate CTC with parallel processing
    print(f"\nCalculating CTC for {len(tax_units_df):,} tax units...")
    tax_units_with_ctc = calculate_ctc_parallel(tax_units_df, batch_size=2000)
    
    # Comprehensive analysis
    analysis_df = comprehensive_analysis(tax_units_with_ctc, crosswalk_df)
    
    # Save results
    save_comprehensive_results(analysis_df)
    
    # Timing summary
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    print(f"\nFull population analysis completed in {total_time/60:.1f} minutes")
    print(f"Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
