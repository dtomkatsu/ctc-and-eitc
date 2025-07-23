"""
Optimized PUMS Data CTC Analysis

This script runs an optimized version of the CTC analysis with:
1. Sampling for quick results
2. Vectorized operations
3. Batch processing
4. Simplified tax unit construction
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
from analysis.geographic import GeographicAnalyzer, GeographicLevel, create_hawaii_puma_crosswalk


def load_pums_sample(sample_size=5000):
    """Load a sample of PUMS data for quick analysis."""
    print(f"Loading PUMS sample ({sample_size:,} households)...")
    
    person_df = pd.read_parquet('data/processed/pums_person_processed.parquet')
    hh_df = pd.read_parquet('data/processed/pums_household_processed.parquet')
    
    # Sample households
    if len(hh_df) > sample_size:
        sampled_hh = hh_df.sample(n=sample_size, random_state=42)
        sampled_persons = person_df[person_df['SERIALNO'].isin(sampled_hh['SERIALNO'])]
    else:
        sampled_hh = hh_df
        sampled_persons = person_df
    
    print(f"Sampled {len(sampled_persons):,} persons from {len(sampled_hh):,} households")
    print(f"Geographic coverage: {sampled_persons['PUMA'].nunique()} PUMAs")
    
    return sampled_persons, sampled_hh


def vectorized_tax_unit_construction(person_df, hh_df):
    """Fast, vectorized tax unit construction."""
    print("Creating tax units using vectorized approach...")
    
    # Prepare data
    person_df = person_df.copy()
    hh_df = hh_df.copy()
    
    # Ensure required columns
    person_df['SERIALNO'] = person_df['SERIALNO'].astype(str)
    hh_df['SERIALNO'] = hh_df['SERIALNO'].astype(str)
    
    # Add household info to persons
    person_df = person_df.merge(hh_df[['SERIALNO', 'HINCP', 'WGTP']], on='SERIALNO', how='left')
    
    # Create age groups
    person_df['is_adult'] = person_df['AGEP'] >= 18
    person_df['is_child'] = person_df['AGEP'] < 18
    
    # Group by household
    hh_groups = person_df.groupby('SERIALNO')
    
    # Vectorized household statistics
    hh_stats = hh_groups.agg({
        'is_adult': 'sum',
        'is_child': 'sum',
        'AGEP': 'count',
        'PINCP': 'sum',
        'PWGTP': 'first',
        'PUMA': 'first',
        'state': 'first',
        'HINCP': 'first'
    }).rename(columns={
        'is_adult': 'num_adults',
        'is_child': 'num_children',
        'AGEP': 'hh_size'
    })
    
    # Filter households with adults
    valid_hh = hh_stats[hh_stats['num_adults'] > 0]
    print(f"Processing {len(valid_hh):,} households with adults")
    
    # Create tax units (one per household for simplicity)
    tax_units = []
    
    for hh_id, stats in valid_hh.iterrows():
        # Get household members
        hh_members = person_df[person_df['SERIALNO'] == hh_id]
        adults = hh_members[hh_members['is_adult']]
        children = hh_members[hh_members['is_child']]
        
        # Determine filing status
        if len(adults) >= 2:
            # Check if married couple
            married_adults = adults[adults['MAR'].isin([1])]  # Married
            if len(married_adults) >= 2:
                filing_status = 'married_filing_jointly'
            else:
                filing_status = 'head_of_household' if len(children) > 0 else 'single'
        else:
            filing_status = 'head_of_household' if len(children) > 0 else 'single'
        
        # Create dependents list
        dependents = []
        for _, child in children.iterrows():
            if child['AGEP'] < 17:  # CTC age limit
                dependents.append({
                    'age': child['AGEP'],
                    'relationship': '22',  # Natural child
                    'citizenship': '1'     # US citizen
                })
        
        tax_unit = {
            'filer_id': f"{hh_id}_1",
            'filing_status': filing_status,
            'income': max(0, stats['PINCP']),
            'dependents': dependents,
            'num_dependents': len(dependents),
            'hh_id': hh_id,
            'PUMA': stats['PUMA'],
            'state': stats['state'],
            'PWGTP': stats['PWGTP'],
            'num_adults': stats['num_adults'],
            'num_children': stats['num_children']
        }
        
        tax_units.append(tax_unit)
    
    tax_units_df = pd.DataFrame(tax_units)
    print(f"Created {len(tax_units_df):,} tax units")
    
    return tax_units_df


def batch_ctc_calculation(tax_units_df, batch_size=1000):
    """Calculate CTC in batches for better performance."""
    print(f"Calculating CTC in batches of {batch_size:,}...")
    
    results = []
    total_batches = (len(tax_units_df) + batch_size - 1) // batch_size
    
    for i in range(0, len(tax_units_df), batch_size):
        batch = tax_units_df.iloc[i:i+batch_size].copy()
        
        # Calculate CTC for each tax unit in batch
        for idx, tax_unit in batch.iterrows():
            tax_unit_dict = tax_unit.to_dict()
            ctc_result = calculate_ctc(tax_unit_dict)
            
            # Add CTC results to tax unit
            for key, value in ctc_result.items():
                tax_unit_dict[f'ctc_{key}'] = value
            
            results.append(tax_unit_dict)
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed batch {i // batch_size + 1}/{total_batches}")
    
    return pd.DataFrame(results)


def quick_analysis(tax_units_with_ctc):
    """Quick analysis of CTC results."""
    print("\n" + "="*50)
    print("QUICK CTC ANALYSIS RESULTS")
    print("="*50)
    
    # Basic statistics
    total_units = len(tax_units_with_ctc)
    units_with_ctc = len(tax_units_with_ctc[tax_units_with_ctc['ctc_ctc_total'] > 0])
    total_ctc = tax_units_with_ctc['ctc_ctc_total'].sum()
    total_children = tax_units_with_ctc['ctc_qualifying_children'].sum()
    
    print(f"\nBASIC STATISTICS:")
    print(f"Total Tax Units: {total_units:,}")
    print(f"Units with CTC: {units_with_ctc:,}")
    print(f"CTC Participation Rate: {units_with_ctc/total_units*100:.1f}%")
    print(f"Total CTC Amount: ${total_ctc:,.2f}")
    print(f"Average CTC per Unit: ${total_ctc/total_units:.2f}")
    if units_with_ctc > 0:
        print(f"Average CTC per Recipient: ${total_ctc/units_with_ctc:.2f}")
    print(f"Total Qualifying Children: {total_children:,}")
    
    # Filing status breakdown
    print(f"\nFILING STATUS BREAKDOWN:")
    status_stats = tax_units_with_ctc.groupby('filing_status').agg({
        'ctc_ctc_total': ['count', 'sum', 'mean'],
        'ctc_qualifying_children': 'sum'
    }).round(2)
    
    for status in tax_units_with_ctc['filing_status'].unique():
        subset = tax_units_with_ctc[tax_units_with_ctc['filing_status'] == status]
        count = len(subset)
        total_ctc_status = subset['ctc_ctc_total'].sum()
        avg_ctc = subset['ctc_ctc_total'].mean()
        
        print(f"  {status}: {count:,} units, ${total_ctc_status:,.0f} total, ${avg_ctc:.0f} avg")
    
    # Weighted estimates
    if 'PWGTP' in tax_units_with_ctc.columns:
        weighted_total_ctc = (tax_units_with_ctc['ctc_ctc_total'] * tax_units_with_ctc['PWGTP']).sum()
        weighted_total_units = tax_units_with_ctc['PWGTP'].sum()
        
        print(f"\nWEIGHTED ESTIMATES (Population):")
        print(f"Estimated Total Tax Units: {weighted_total_units:,.0f}")
        print(f"Estimated Total CTC Amount: ${weighted_total_ctc:,.2f}")
        print(f"Estimated Average CTC per Unit: ${weighted_total_ctc/weighted_total_units:.2f}")


def quick_geographic_analysis(tax_units_with_ctc):
    """Quick geographic analysis."""
    print(f"\n{'='*50}")
    print("GEOGRAPHIC ANALYSIS")
    print("="*50)
    
    # PUMA-level analysis
    print(f"\nPUMA-LEVEL RESULTS:")
    print("-" * 20)
    
    puma_stats = tax_units_with_ctc.groupby('PUMA').agg({
        'ctc_ctc_total': ['count', 'sum', 'mean'],
        'ctc_qualifying_children': 'sum',
        'PWGTP': 'sum',
        'income': 'mean'
    }).round(2)
    
    # Flatten column names
    puma_stats.columns = ['_'.join(col).strip() for col in puma_stats.columns]
    
    # Calculate participation rates
    puma_participation = tax_units_with_ctc.groupby('PUMA').apply(
        lambda x: len(x[x['ctc_ctc_total'] > 0]) / len(x) * 100
    ).round(1)
    
    print(f"{'PUMA':<8} {'Units':<8} {'Total CTC':<12} {'Avg CTC':<10} {'Part Rate':<10} {'Avg Income':<12}")
    print("-" * 70)
    
    for puma in sorted(tax_units_with_ctc['PUMA'].unique()):
        stats = puma_stats.loc[puma]
        participation = puma_participation.loc[puma]
        
        print(f"{puma:<8} {stats['ctc_ctc_total_count']:>6.0f} "
              f"${stats['ctc_ctc_total_sum']:>10.0f} "
              f"${stats['ctc_ctc_total_mean']:>8.0f} "
              f"{participation:>8.1f}% "
              f"${stats['income_mean']:>10.0f}")
    
    return puma_stats


def scale_to_population(sample_results, sample_size, total_population):
    """Scale sample results to full population."""
    scaling_factor = total_population / sample_size
    
    print(f"\nSCALING TO FULL POPULATION:")
    print(f"Sample size: {sample_size:,}")
    print(f"Total population: {total_population:,}")
    print(f"Scaling factor: {scaling_factor:.2f}")
    
    # Scale key metrics
    sample_total_ctc = sample_results['ctc_ctc_total'].sum()
    estimated_total_ctc = sample_total_ctc * scaling_factor
    
    sample_units_with_ctc = len(sample_results[sample_results['ctc_ctc_total'] > 0])
    estimated_units_with_ctc = sample_units_with_ctc * scaling_factor
    
    print(f"\nESTIMATED STATEWIDE RESULTS:")
    print(f"Estimated Total CTC Amount: ${estimated_total_ctc:,.2f}")
    print(f"Estimated Units with CTC: {estimated_units_with_ctc:,.0f}")
    print(f"Estimated Total Tax Units: {len(sample_results) * scaling_factor:,.0f}")


def main():
    """Run optimized PUMS CTC analysis."""
    print("Optimized Hawaii PUMS CTC Analysis")
    print("="*40)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load sample data for quick analysis
    sample_size = 5000  # Adjust based on desired speed vs accuracy
    person_df, hh_df = load_pums_sample(sample_size)
    
    # Fast tax unit construction
    start_time = datetime.now()
    tax_units_df = vectorized_tax_unit_construction(person_df, hh_df)
    construction_time = (datetime.now() - start_time).total_seconds()
    print(f"Tax unit construction completed in {construction_time:.1f} seconds")
    
    if len(tax_units_df) == 0:
        print("No tax units created. Exiting.")
        return
    
    # Fast CTC calculation
    start_time = datetime.now()
    tax_units_with_ctc = batch_ctc_calculation(tax_units_df, batch_size=500)
    ctc_time = (datetime.now() - start_time).total_seconds()
    print(f"CTC calculations completed in {ctc_time:.1f} seconds")
    
    # Quick analysis
    quick_analysis(tax_units_with_ctc)
    
    # Geographic analysis
    puma_stats = quick_geographic_analysis(tax_units_with_ctc)
    
    # Scale to full population
    total_households = 32104  # From full PUMS data
    scale_to_population(tax_units_with_ctc, len(tax_units_df), total_households)
    
    # Save sample results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'data/processed/pums_ctc_sample_{timestamp}.parquet'
    tax_units_with_ctc.to_parquet(output_file)
    print(f"\nSample results saved to: {output_file}")
    
    total_time = construction_time + ctc_time
    print(f"\nTotal analysis time: {total_time:.1f} seconds")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
