"""
Full Pipeline Example: From Tax Unit Construction to Geographic CTC Analysis

This example demonstrates the complete pipeline:
1. Load and process PUMS data
2. Construct tax units
3. Calculate CTC benefits
4. Perform geographic analysis at multiple levels
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from tax.units.constructor import TaxUnitConstructor
from tax.credits.ctc import calculate_ctc_for_tax_units, get_ctc_summary_stats
from analysis.geographic import GeographicAnalyzer, GeographicLevel, create_hawaii_puma_crosswalk


def load_sample_pums_data():
    """Load sample PUMS data for demonstration."""
    # In practice, this would load actual PUMS data
    # For this example, we'll create realistic sample data
    
    np.random.seed(42)
    
    # Create household data
    households = []
    persons = []
    
    pumas = ['00100', '00200', '00300', '00400', '00500']
    
    for hh_id in range(1, 101):  # 100 households
        puma = np.random.choice(pumas)
        
        # Household characteristics
        hh_income = max(20000, np.random.normal(65000, 25000))
        hh_size = np.random.choice([1, 2, 3, 4, 5], p=[0.3, 0.3, 0.2, 0.15, 0.05])
        
        household = {
            'SERIALNO': f'HH{hh_id:03d}',
            'PUMA': puma,
            'HINCP': hh_income,
            'WGTP': np.random.randint(50, 200),
            'NP': hh_size,
            'state': 'Hawaii',
            'ADJINC': np.random.uniform(1.0, 1.2)
        }
        households.append(household)
        
        # Create persons in household
        for person_id in range(1, hh_size + 1):
            if person_id == 1:
                # Householder
                age = np.random.randint(25, 65)
                relationship = 20  # Householder
                married = np.random.choice([1, 2, 5], p=[0.4, 0.4, 0.2])  # Married, divorced, never married
            elif person_id == 2 and married == 1:
                # Spouse
                age = np.random.randint(22, 62)
                relationship = 21  # Spouse
                married = 1
            else:
                # Child or other
                if np.random.random() < 0.8:  # 80% chance of being a child
                    age = np.random.randint(0, 18)
                    relationship = 22  # Natural child
                    married = 5  # Never married
                else:
                    age = np.random.randint(18, 30)
                    relationship = 22  # Adult child
                    married = np.random.choice([1, 5], p=[0.3, 0.7])
            
            # Income (adults only)
            if age >= 18:
                income = max(0, np.random.normal(45000, 20000))
            else:
                income = 0
            
            person = {
                'SERIALNO': f'HH{hh_id:03d}',
                'SPORDER': person_id,
                'PUMA': puma,
                'AGEP': age,
                'SEX': np.random.choice([1, 2]),
                'RELSHIPP': relationship,
                'MAR': married,
                'CIT': np.random.choice([1, 2, 3, 4], p=[0.7, 0.1, 0.1, 0.1]),  # Citizenship
                'WAGP': income,
                'PINCP': income,
                'PWGTP': np.random.randint(50, 200),
                'state': 'Hawaii',
                'ADJINC': household['ADJINC']
            }
            persons.append(person)
    
    return pd.DataFrame(households), pd.DataFrame(persons)


def main():
    """Run the full pipeline example."""
    print("Full Pipeline: Tax Unit Construction → CTC Calculation → Geographic Analysis")
    print("=" * 80)
    
    # Step 1: Load PUMS data
    print("Step 1: Loading PUMS Data")
    print("-" * 25)
    hh_df, person_df = load_sample_pums_data()
    print(f"Loaded {len(hh_df)} households and {len(person_df)} persons")
    print(f"Geographic coverage: {person_df['PUMA'].nunique()} PUMAs in {person_df['state'].nunique()} state(s)")
    print()
    
    # Step 2: Construct tax units
    print("Step 2: Constructing Tax Units")
    print("-" * 30)
    constructor = TaxUnitConstructor(person_df, hh_df)
    tax_units_df = constructor.create_rule_based_units()
    
    print(f"Constructed {len(tax_units_df)} tax units")
    print("Filing status distribution:")
    filing_status_counts = tax_units_df['filing_status'].value_counts()
    for status, count in filing_status_counts.items():
        print(f"  {status}: {count} ({count/len(tax_units_df)*100:.1f}%)")
    print()
    
    # Step 3: Calculate CTC
    print("Step 3: Calculating Child Tax Credit")
    print("-" * 34)
    tax_units_with_ctc = calculate_ctc_for_tax_units(tax_units_df)
    
    # Get CTC summary statistics
    ctc_stats = get_ctc_summary_stats(tax_units_with_ctc)
    print("CTC Summary Statistics:")
    for key, value in ctc_stats.items():
        if isinstance(value, float):
            if 'amount' in key or 'average' in key:
                print(f"  {key}: ${value:,.2f}")
            else:
                print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value:,}")
    print()
    
    # Step 4: Geographic Analysis
    print("Step 4: Geographic Analysis")
    print("-" * 27)
    
    # Create PUMA crosswalk for district analysis
    crosswalk = create_hawaii_puma_crosswalk()
    analyzer = GeographicAnalyzer(tax_units_with_ctc, crosswalk)
    
    # 4a. State-level analysis
    print("4a. State-Level Analysis:")
    state_results = analyzer.analyze_ctc_by_geography(GeographicLevel.STATE)
    print(f"  Total CTC Amount: ${state_results['total_ctc_amount'].iloc[0]:,.2f}")
    print(f"  Average CTC per Unit: ${state_results['average_ctc_per_unit'].iloc[0]:,.2f}")
    print(f"  Participation Rate: {state_results['ctc_participation_rate'].iloc[0]:.1f}%")
    print()
    
    # 4b. PUMA-level analysis
    print("4b. PUMA-Level Analysis:")
    puma_results = analyzer.analyze_ctc_by_geography(GeographicLevel.PUMA)
    print(f"  Number of PUMAs: {len(puma_results)}")
    print(f"  Highest Average CTC: ${puma_results['average_ctc_per_unit'].max():,.2f}")
    print(f"  Lowest Average CTC: ${puma_results['average_ctc_per_unit'].min():,.2f}")
    print(f"  Geographic Variation (CV): {puma_results['average_ctc_per_unit'].std()/puma_results['average_ctc_per_unit'].mean()*100:.1f}%")
    print()
    
    # 4c. Legislative district analysis
    print("4c. Legislative District Analysis:")
    try:
        senate_results = analyzer.analyze_ctc_by_geography(GeographicLevel.SENATE_DISTRICT)
        house_results = analyzer.analyze_ctc_by_geography(GeographicLevel.HOUSE_DISTRICT)
        
        print(f"  Senate Districts: {len(senate_results)}")
        print(f"  Average CTC per Senate District: ${senate_results['total_ctc_amount'].mean():,.2f}")
        print(f"  House Districts: {len(house_results)}")
        print(f"  Average CTC per House District: ${house_results['total_ctc_amount'].mean():,.2f}")
        
    except ValueError as e:
        print(f"  District analysis not available: {e}")
    print()
    
    # Step 5: Detailed Results
    print("Step 5: Detailed Geographic Results")
    print("-" * 35)
    
    print("Top 3 PUMAs by Total CTC Amount:")
    top_pumas = analyzer.get_top_areas(GeographicLevel.PUMA, 'total_ctc_amount', 3)
    for idx, row in top_pumas.iterrows():
        print(f"  PUMA {row['PUMA']}: ${row['total_ctc_amount']:,.2f} "
              f"(avg: ${row['average_ctc_per_unit']:,.2f}, {row['total_tax_units']} units)")
    print()
    
    print("PUMAs by CTC Participation Rate:")
    by_participation = analyzer.compare_geographic_areas(GeographicLevel.PUMA, 'ctc_participation_rate')
    for idx, row in by_participation.iterrows():
        print(f"  PUMA {row['PUMA']}: {row['ctc_participation_rate']:.1f}% "
              f"({row['units_with_ctc']}/{row['total_tax_units']} units)")
    print()
    
    # Step 6: Export Results
    print("Step 6: Exporting Results")
    print("-" * 23)
    
    # Save tax units with CTC calculations
    output_file = 'data/processed/tax_units_with_ctc.parquet'
    tax_units_with_ctc.to_parquet(output_file)
    print(f"Tax units with CTC saved to: {output_file}")
    
    # Save geographic analysis results
    puma_output = 'data/processed/ctc_analysis_by_puma.csv'
    puma_results.to_csv(puma_output, index=False)
    print(f"PUMA-level analysis saved to: {puma_output}")
    
    if 'senate_results' in locals():
        senate_output = 'data/processed/ctc_analysis_by_senate_district.csv'
        senate_results.to_csv(senate_output, index=False)
        print(f"Senate district analysis saved to: {senate_output}")
    
    print()
    print("Pipeline completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
