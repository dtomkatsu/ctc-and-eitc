"""
Real PUMS Data CTC Analysis

This script runs the complete CTC analysis pipeline using actual PUMS data for Hawaii.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from tax.units.constructor import TaxUnitConstructor
from tax.credits.ctc import calculate_ctc_for_tax_units, get_ctc_summary_stats
from analysis.geographic import GeographicAnalyzer, GeographicLevel, create_hawaii_puma_crosswalk


def load_pums_data():
    """Load processed PUMS data."""
    print("Loading PUMS data...")
    
    person_df = pd.read_parquet('data/processed/pums_person_processed.parquet')
    hh_df = pd.read_parquet('data/processed/pums_household_processed.parquet')
    
    print(f"Loaded {len(person_df):,} persons from {len(hh_df):,} households")
    print(f"Geographic coverage: {person_df['PUMA'].nunique()} PUMAs")
    print(f"PUMAs: {sorted(person_df['PUMA'].unique())}")
    
    return person_df, hh_df


def prepare_data_for_constructor(person_df, hh_df):
    """Prepare data for the tax unit constructor."""
    print("Preparing data for tax unit construction...")
    
    # Ensure required columns exist
    required_person_cols = ['SERIALNO', 'SPORDER', 'AGEP', 'RELSHIPP', 'MAR', 'PWGTP']
    required_hh_cols = ['SERIALNO', 'WGTP', 'HINCP']
    
    missing_person = [col for col in required_person_cols if col not in person_df.columns]
    missing_hh = [col for col in required_hh_cols if col not in hh_df.columns]
    
    if missing_person:
        print(f"Warning: Missing person columns: {missing_person}")
    if missing_hh:
        print(f"Warning: Missing household columns: {missing_hh}")
    
    # Convert SPORDER to string if needed
    if 'SPORDER' in person_df.columns:
        person_df['SPORDER'] = person_df['SPORDER'].astype(str)
    
    # Ensure SERIALNO is string
    person_df['SERIALNO'] = person_df['SERIALNO'].astype(str)
    hh_df['SERIALNO'] = hh_df['SERIALNO'].astype(str)
    
    print("Data preparation completed")
    return person_df, hh_df


def construct_tax_units(person_df, hh_df):
    """Construct tax units from PUMS data."""
    print("Constructing tax units...")
    
    try:
        constructor = TaxUnitConstructor(person_df, hh_df)
        tax_units_df = constructor.create_rule_based_units()
        
        print(f"Successfully constructed {len(tax_units_df):,} tax units")
        
        if len(tax_units_df) > 0:
            print("Filing status distribution:")
            filing_status_counts = tax_units_df['filing_status'].value_counts()
            for status, count in filing_status_counts.items():
                print(f"  {status}: {count:,} ({count/len(tax_units_df)*100:.1f}%)")
        
        return tax_units_df
        
    except Exception as e:
        print(f"Error constructing tax units: {e}")
        print("Creating simplified tax units for analysis...")
        return create_simplified_tax_units(person_df, hh_df)


def create_simplified_tax_units(person_df, hh_df):
    """Create simplified tax units when full construction fails."""
    print("Creating simplified tax units...")
    
    # Group by household and create basic tax units
    tax_units = []
    
    for hh_id, hh_group in person_df.groupby('SERIALNO'):
        # Get household info
        hh_info = hh_df[hh_df['SERIALNO'] == hh_id].iloc[0] if len(hh_df[hh_df['SERIALNO'] == hh_id]) > 0 else {}
        
        # Get adults (18+)
        adults = hh_group[hh_group['AGEP'] >= 18]
        children = hh_group[hh_group['AGEP'] < 18]
        
        if len(adults) == 0:
            continue  # Skip households with no adults
        
        # Create one tax unit per household (simplified)
        primary_adult = adults.iloc[0]
        
        # Create dependents list
        dependents = []
        for _, child in children.iterrows():
            dependents.append({
                'age': child['AGEP'],
                'relationship': str(child.get('RELSHIPP', '22')),  # Default to natural child
                'citizenship': '1'  # Assume US citizen for simplicity
            })
        
        # Calculate household income
        total_income = hh_group['PINCP'].fillna(0).sum()
        
        # Determine filing status (simplified)
        if len(adults) >= 2:
            filing_status = 'married_filing_jointly'
        elif len(children) > 0:
            filing_status = 'head_of_household'
        else:
            filing_status = 'single'
        
        tax_unit = {
            'filer_id': f"{hh_id}_1",
            'filing_status': filing_status,
            'income': total_income,
            'dependents': dependents,
            'num_dependents': len(dependents),
            'hh_id': hh_id,
            'PUMA': primary_adult.get('PUMA', ''),
            'state': primary_adult.get('state', 'Hawaii'),
            'PWGTP': primary_adult.get('PWGTP', 1),
            'primary_filer_id': primary_adult.name,
            'secondary_filer_id': adults.iloc[1].name if len(adults) > 1 else None
        }
        
        tax_units.append(tax_unit)
    
    tax_units_df = pd.DataFrame(tax_units)
    print(f"Created {len(tax_units_df):,} simplified tax units")
    
    return tax_units_df


def analyze_ctc_results(tax_units_with_ctc):
    """Analyze CTC results and provide summary."""
    print("\n" + "="*60)
    print("CTC ANALYSIS RESULTS")
    print("="*60)
    
    # Overall statistics
    total_units = len(tax_units_with_ctc)
    units_with_ctc = len(tax_units_with_ctc[tax_units_with_ctc['ctc_ctc_total'] > 0])
    total_ctc = tax_units_with_ctc['ctc_ctc_total'].sum()
    total_children = tax_units_with_ctc['ctc_qualifying_children'].sum()
    
    print(f"\nOVERALL STATISTICS:")
    print(f"Total Tax Units: {total_units:,}")
    print(f"Units with CTC: {units_with_ctc:,}")
    print(f"CTC Participation Rate: {units_with_ctc/total_units*100:.1f}%")
    print(f"Total CTC Amount: ${total_ctc:,.2f}")
    print(f"Average CTC per Unit: ${total_ctc/total_units:.2f}")
    if units_with_ctc > 0:
        print(f"Average CTC per Recipient: ${total_ctc/units_with_ctc:.2f}")
    print(f"Total Qualifying Children: {total_children:,}")
    
    # Weighted statistics (using PWGTP)
    if 'PWGTP' in tax_units_with_ctc.columns:
        weighted_total_ctc = (tax_units_with_ctc['ctc_ctc_total'] * tax_units_with_ctc['PWGTP']).sum()
        weighted_total_units = tax_units_with_ctc['PWGTP'].sum()
        
        print(f"\nWEIGHTED STATISTICS (Population Estimates):")
        print(f"Estimated Total Tax Units: {weighted_total_units:,.0f}")
        print(f"Estimated Total CTC Amount: ${weighted_total_ctc:,.2f}")
        print(f"Estimated Average CTC per Unit: ${weighted_total_ctc/weighted_total_units:.2f}")


def run_geographic_analysis(tax_units_with_ctc):
    """Run geographic analysis at multiple levels."""
    print(f"\n{'='*60}")
    print("GEOGRAPHIC ANALYSIS")
    print("="*60)
    
    # Create analyzer
    crosswalk = create_hawaii_puma_crosswalk()
    analyzer = GeographicAnalyzer(tax_units_with_ctc, crosswalk)
    
    # State-level analysis
    print(f"\nSTATE-LEVEL ANALYSIS:")
    print("-" * 25)
    state_analysis = analyzer.analyze_ctc_by_geography(GeographicLevel.STATE, weight_col='PWGTP')
    
    for _, row in state_analysis.iterrows():
        print(f"State: {row['state']}")
        print(f"  Total CTC Amount: ${row['total_ctc_amount']:,.2f}")
        print(f"  Average CTC per Unit: ${row['average_ctc_per_unit']:.2f}")
        print(f"  Participation Rate: {row['ctc_participation_rate']:.1f}%")
        print(f"  Total Tax Units: {row['weighted_tax_units']:,.0f}")
    
    # PUMA-level analysis
    print(f"\nPUMA-LEVEL ANALYSIS:")
    print("-" * 22)
    puma_analysis = analyzer.analyze_ctc_by_geography(GeographicLevel.PUMA, weight_col='PWGTP')
    puma_analysis = puma_analysis.sort_values('total_ctc_amount', ascending=False)
    
    print(f"{'PUMA':<8} {'Total CTC':<15} {'Avg CTC':<10} {'Units':<8} {'Part Rate':<10}")
    print("-" * 60)
    
    for _, row in puma_analysis.iterrows():
        print(f"{row['PUMA']:<8} ${row['total_ctc_amount']:>12,.0f} "
              f"${row['average_ctc_per_unit']:>8.0f} "
              f"{row['weighted_tax_units']:>6.0f} "
              f"{row['ctc_participation_rate']:>8.1f}%")
    
    # Legislative district analysis
    print(f"\nLEGISLATIVE DISTRICT ANALYSIS:")
    print("-" * 32)
    
    try:
        senate_analysis = analyzer.analyze_ctc_by_geography(GeographicLevel.SENATE_DISTRICT, weight_col='PWGTP')
        house_analysis = analyzer.analyze_ctc_by_geography(GeographicLevel.HOUSE_DISTRICT, weight_col='PWGTP')
        
        print("Senate Districts:")
        for _, row in senate_analysis.iterrows():
            print(f"  {row['senate_district']}: ${row['total_ctc_amount']:,.0f} total, "
                  f"{row['weighted_tax_units']:,.0f} units")
        
        print("\nHouse Districts:")
        for _, row in house_analysis.iterrows():
            print(f"  {row['house_district']}: ${row['total_ctc_amount']:,.0f} total, "
                  f"{row['weighted_tax_units']:,.0f} units")
                  
    except ValueError as e:
        print(f"District analysis not available: {e}")
    
    return puma_analysis


def save_results(tax_units_with_ctc, puma_analysis):
    """Save analysis results."""
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print("="*60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save tax units with CTC
    output_file = f'data/processed/pums_tax_units_with_ctc_{timestamp}.parquet'
    tax_units_with_ctc.to_parquet(output_file)
    print(f"Tax units with CTC saved to: {output_file}")
    
    # Save PUMA analysis
    puma_file = f'data/processed/pums_ctc_by_puma_{timestamp}.csv'
    puma_analysis.to_csv(puma_file, index=False)
    print(f"PUMA analysis saved to: {puma_file}")
    
    # Create summary report
    summary_file = f'data/processed/pums_ctc_summary_{timestamp}.txt'
    with open(summary_file, 'w') as f:
        f.write("Hawaii PUMS CTC Analysis Summary\n")
        f.write("="*40 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Tax Units: {len(tax_units_with_ctc):,}\n")
        f.write(f"Units with CTC: {len(tax_units_with_ctc[tax_units_with_ctc['ctc_ctc_total'] > 0]):,}\n")
        f.write(f"Total CTC Amount: ${tax_units_with_ctc['ctc_ctc_total'].sum():,.2f}\n")
        f.write(f"Average CTC per Unit: ${tax_units_with_ctc['ctc_ctc_total'].mean():.2f}\n")
    
    print(f"Summary report saved to: {summary_file}")


def main():
    """Run the complete PUMS CTC analysis."""
    print("Hawaii PUMS Data CTC Analysis")
    print("="*40)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    person_df, hh_df = load_pums_data()
    
    # Prepare data
    person_df, hh_df = prepare_data_for_constructor(person_df, hh_df)
    
    # Construct tax units
    tax_units_df = construct_tax_units(person_df, hh_df)
    
    if len(tax_units_df) == 0:
        print("No tax units created. Exiting.")
        return
    
    # Calculate CTC
    print(f"\nCalculating CTC for {len(tax_units_df):,} tax units...")
    tax_units_with_ctc = calculate_ctc_for_tax_units(tax_units_df)
    print("CTC calculations completed")
    
    # Analyze results
    analyze_ctc_results(tax_units_with_ctc)
    
    # Geographic analysis
    puma_analysis = run_geographic_analysis(tax_units_with_ctc)
    
    # Save results
    save_results(tax_units_with_ctc, puma_analysis)
    
    print(f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
