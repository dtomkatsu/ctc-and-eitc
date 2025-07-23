"""
Fast PUMS CTC Analysis

Simple, fast analysis using actual PUMS data with minimal processing.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from tax.credits.ctc import calculate_ctc


def load_and_prepare_data(sample_size=2000):
    """Load and prepare PUMS data quickly."""
    print(f"Loading PUMS data sample ({sample_size:,} households)...")
    
    # Load data
    person_df = pd.read_parquet('data/processed/pums_person_processed.parquet')
    
    # Sample households for speed
    unique_hh = person_df['SERIALNO'].unique()
    if len(unique_hh) > sample_size:
        sampled_hh = np.random.choice(unique_hh, size=sample_size, replace=False)
        person_df = person_df[person_df['SERIALNO'].isin(sampled_hh)]
    
    print(f"Working with {len(person_df):,} persons from {person_df['SERIALNO'].nunique():,} households")
    print(f"PUMAs covered: {sorted(person_df['PUMA'].unique())}")
    
    return person_df


def create_simple_tax_units(person_df):
    """Create tax units using simple, fast logic."""
    print("Creating tax units...")
    
    tax_units = []
    
    # Process each household
    for hh_id, hh_group in person_df.groupby('SERIALNO'):
        # Basic household info
        adults = hh_group[hh_group['AGEP'] >= 18]
        children = hh_group[hh_group['AGEP'] < 18]
        
        if len(adults) == 0:
            continue  # Skip households with no adults
        
        # Simple filing status logic
        if len(adults) >= 2:
            # Check for married couple
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
            if child['AGEP'] < 17:  # CTC qualifying age
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
            'PUMA': str(primary_adult['PUMA']).zfill(5),  # Ensure 5-digit format
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


def calculate_ctc_fast(tax_units_df):
    """Calculate CTC quickly."""
    print("Calculating CTC...")
    
    results = []
    
    for idx, tax_unit in tax_units_df.iterrows():
        # Calculate CTC
        ctc_result = calculate_ctc(tax_unit.to_dict())
        
        # Combine tax unit info with CTC results
        result = tax_unit.to_dict()
        for key, value in ctc_result.items():
            result[f'ctc_{key}'] = value
        
        results.append(result)
        
        # Progress indicator
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1:,}/{len(tax_units_df):,} tax units")
    
    return pd.DataFrame(results)


def analyze_results(tax_units_with_ctc):
    """Analyze CTC results."""
    print("\n" + "="*60)
    print("HAWAII PUMS CTC ANALYSIS RESULTS")
    print("="*60)
    
    # Basic statistics
    total_units = len(tax_units_with_ctc)
    units_with_ctc = len(tax_units_with_ctc[tax_units_with_ctc['ctc_ctc_total'] > 0])
    total_ctc = tax_units_with_ctc['ctc_ctc_total'].sum()
    total_children = tax_units_with_ctc['ctc_qualifying_children'].sum()
    
    print(f"\nOVERALL STATISTICS:")
    print(f"Sample Tax Units: {total_units:,}")
    print(f"Units with CTC: {units_with_ctc:,}")
    print(f"CTC Participation Rate: {units_with_ctc/total_units*100:.1f}%")
    print(f"Total CTC Amount: ${total_ctc:,.2f}")
    print(f"Average CTC per Unit: ${total_ctc/total_units:.2f}")
    if units_with_ctc > 0:
        print(f"Average CTC per Recipient: ${total_ctc/units_with_ctc:.2f}")
    print(f"Total Qualifying Children: {total_children:,}")
    
    # Filing status breakdown
    print(f"\nFILING STATUS BREAKDOWN:")
    for status in tax_units_with_ctc['filing_status'].unique():
        subset = tax_units_with_ctc[tax_units_with_ctc['filing_status'] == status]
        count = len(subset)
        ctc_recipients = len(subset[subset['ctc_ctc_total'] > 0])
        total_ctc_status = subset['ctc_ctc_total'].sum()
        avg_ctc = subset['ctc_ctc_total'].mean()
        
        print(f"  {status}:")
        print(f"    Units: {count:,} ({count/total_units*100:.1f}%)")
        print(f"    CTC Recipients: {ctc_recipients:,} ({ctc_recipients/count*100:.1f}%)")
        print(f"    Total CTC: ${total_ctc_status:,.2f}")
        print(f"    Average CTC: ${avg_ctc:.2f}")
    
    # Geographic analysis
    print(f"\nGEOGRAPHIC ANALYSIS BY PUMA:")
    print("-" * 40)
    
    puma_stats = tax_units_with_ctc.groupby('PUMA').agg({
        'ctc_ctc_total': ['count', 'sum', 'mean'],
        'ctc_qualifying_children': 'sum',
        'PWGTP': 'sum',
        'income': 'mean'
    }).round(2)
    
    # Calculate participation rates by PUMA
    puma_participation = tax_units_with_ctc.groupby('PUMA').apply(
        lambda x: len(x[x['ctc_ctc_total'] > 0]) / len(x) * 100
    ).round(1)
    
    print(f"{'PUMA':<8} {'Units':<8} {'Total CTC':<12} {'Avg CTC':<10} {'Part Rate':<10} {'Children':<10}")
    print("-" * 70)
    
    for puma in sorted(tax_units_with_ctc['PUMA'].unique()):
        subset = tax_units_with_ctc[tax_units_with_ctc['PUMA'] == puma]
        count = len(subset)
        total_ctc_puma = subset['ctc_ctc_total'].sum()
        avg_ctc = subset['ctc_ctc_total'].mean()
        participation = len(subset[subset['ctc_ctc_total'] > 0]) / len(subset) * 100
        children = subset['ctc_qualifying_children'].sum()
        
        print(f"{puma:<8} {count:>6} ${total_ctc_puma:>10,.0f} ${avg_ctc:>8.0f} {participation:>8.1f}% {children:>8}")
    
    # Weighted population estimates
    if 'PWGTP' in tax_units_with_ctc.columns:
        weighted_total_ctc = (tax_units_with_ctc['ctc_ctc_total'] * tax_units_with_ctc['PWGTP']).sum()
        weighted_total_units = tax_units_with_ctc['PWGTP'].sum()
        
        print(f"\nWEIGHTED POPULATION ESTIMATES:")
        print(f"Estimated Total Tax Units: {weighted_total_units:,.0f}")
        print(f"Estimated Total CTC Amount: ${weighted_total_ctc:,.2f}")
        print(f"Estimated Average CTC per Unit: ${weighted_total_ctc/weighted_total_units:.2f}")
        
        # Scale to full Hawaii population (approximate)
        scaling_factor = 32104 / total_units  # Total households / sample households
        estimated_statewide_ctc = total_ctc * scaling_factor
        
        print(f"\nESTIMATED STATEWIDE TOTALS (Scaled from Sample):")
        print(f"Estimated Total Tax Units: {total_units * scaling_factor:,.0f}")
        print(f"Estimated Total CTC Amount: ${estimated_statewide_ctc:,.2f}")
        print(f"Estimated Units with CTC: {units_with_ctc * scaling_factor:,.0f}")


def main():
    """Run fast PUMS CTC analysis."""
    print("Fast Hawaii PUMS CTC Analysis")
    print("="*35)
    start_time = datetime.now()
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    person_df = load_and_prepare_data(sample_size=2000)
    
    # Create tax units
    tax_units_df = create_simple_tax_units(person_df)
    
    if len(tax_units_df) == 0:
        print("No tax units created. Exiting.")
        return
    
    # Calculate CTC
    tax_units_with_ctc = calculate_ctc_fast(tax_units_df)
    
    # Analyze results
    analyze_results(tax_units_with_ctc)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'data/processed/fast_pums_ctc_{timestamp}.parquet'
    tax_units_with_ctc.to_parquet(output_file)
    print(f"\nResults saved to: {output_file}")
    
    # Timing
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    print(f"\nAnalysis completed in {total_time:.1f} seconds")
    print(f"Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
