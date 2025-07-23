"""
Example: Geographic Analysis of Child Tax Credit (CTC)

This example demonstrates how to analyze CTC at various geographic levels
including state, county (PUMA), and legislative districts.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from tax.credits.ctc import calculate_ctc_for_tax_units
from analysis.geographic import (
    GeographicAnalyzer, 
    GeographicLevel, 
    create_hawaii_puma_crosswalk,
    analyze_ctc_by_all_levels
)


def create_sample_geographic_data():
    """Create sample tax units with geographic information for demonstration."""
    np.random.seed(42)  # For reproducible results
    
    # Hawaii PUMA codes
    pumas = ['00100', '00200', '00300', '00400', '00500']
    puma_names = ['Honolulu Urban', 'Honolulu Suburban', 'Big Island', 'Maui', 'Kauai']
    
    data = []
    
    for i, (puma, puma_name) in enumerate(zip(pumas, puma_names)):
        # Create 50 tax units per PUMA
        for j in range(50):
            # Vary income by PUMA (urban areas tend to have higher income)
            base_income = 40000 + (i * 15000) + np.random.normal(0, 20000)
            base_income = max(15000, base_income)  # Minimum income
            
            # Vary number of children
            num_children = np.random.choice([0, 1, 2, 3], p=[0.3, 0.3, 0.3, 0.1])
            
            # Create dependents
            dependents = []
            for k in range(num_children):
                age = np.random.randint(0, 18)
                dependents.append({
                    'age': age,
                    'relationship': '22',  # Natural child
                    'citizenship': '1'     # US citizen
                })
            
            # Random filing status
            filing_status = np.random.choice([
                'single', 'married_filing_jointly', 'head_of_household'
            ], p=[0.4, 0.4, 0.2])
            
            tax_unit = {
                'filer_id': f'{puma}_{j+1}',
                'filing_status': filing_status,
                'income': base_income,
                'dependents': dependents,
                'num_dependents': len(dependents),
                'hh_id': f'HH_{puma}_{j+1}',
                'state': 'Hawaii',
                'PUMA': puma,
                'puma_name': puma_name,
                'PWGTP': np.random.randint(50, 200)  # Person weight
            }
            
            data.append(tax_unit)
    
    return pd.DataFrame(data)


def main():
    """Run geographic CTC analysis example."""
    print("Geographic Analysis of Child Tax Credit (CTC)")
    print("=" * 50)
    
    # Create sample data
    print("Creating sample tax units with geographic data...")
    tax_units = create_sample_geographic_data()
    print(f"Created {len(tax_units)} tax units across {tax_units['PUMA'].nunique()} PUMAs\n")
    
    # Calculate CTC for all tax units
    print("Calculating CTC for all tax units...")
    tax_units_with_ctc = calculate_ctc_for_tax_units(tax_units)
    print("CTC calculations completed.\n")
    
    # Initialize geographic analyzer
    crosswalk = create_hawaii_puma_crosswalk()
    analyzer = GeographicAnalyzer(tax_units_with_ctc, crosswalk)
    
    # 1. State-level analysis
    print("1. STATE-LEVEL ANALYSIS")
    print("-" * 25)
    state_analysis = analyzer.analyze_ctc_by_geography(GeographicLevel.STATE)
    print(state_analysis.round(2))
    print()
    
    # 2. PUMA-level analysis
    print("2. PUMA-LEVEL ANALYSIS")
    print("-" * 22)
    puma_analysis = analyzer.analyze_ctc_by_geography(GeographicLevel.PUMA)
    
    # Add PUMA names for better readability
    puma_names = tax_units[['PUMA', 'puma_name']].drop_duplicates()
    puma_analysis = puma_analysis.merge(puma_names, on='PUMA', how='left')
    
    # Display key metrics
    display_cols = ['puma_name', 'total_tax_units', 'ctc_participation_rate', 
                   'average_ctc_per_unit', 'total_ctc_amount', 'average_income']
    print(puma_analysis[display_cols].round(2))
    print()
    
    # 3. Legislative district analysis
    print("3. LEGISLATIVE DISTRICT ANALYSIS")
    print("-" * 32)
    
    try:
        # Senate districts
        senate_analysis = analyzer.analyze_ctc_by_geography(GeographicLevel.SENATE_DISTRICT)
        print("Senate Districts:")
        senate_display = ['senate_district', 'total_tax_units', 'average_ctc_per_unit', 'total_ctc_amount']
        print(senate_analysis[senate_display].round(2))
        print()
        
        # House districts  
        house_analysis = analyzer.analyze_ctc_by_geography(GeographicLevel.HOUSE_DISTRICT)
        print("House Districts:")
        house_display = ['house_district', 'total_tax_units', 'average_ctc_per_unit', 'total_ctc_amount']
        print(house_analysis[house_display].round(2))
        print()
        
    except ValueError as e:
        print(f"District analysis not available: {e}\n")
    
    # 4. Comparative analysis
    print("4. COMPARATIVE ANALYSIS")
    print("-" * 22)
    
    # Top PUMAs by total CTC amount
    top_pumas = analyzer.get_top_areas(GeographicLevel.PUMA, 'total_ctc_amount', 3)
    print("Top 3 PUMAs by Total CTC Amount:")
    top_pumas_display = top_pumas.merge(puma_names, on='PUMA', how='left')
    print(top_pumas_display[['puma_name', 'total_ctc_amount', 'average_ctc_per_unit']].round(2))
    print()
    
    # PUMAs by participation rate
    top_participation = analyzer.get_top_areas(GeographicLevel.PUMA, 'ctc_participation_rate', 3)
    print("Top 3 PUMAs by CTC Participation Rate:")
    top_participation_display = top_participation.merge(puma_names, on='PUMA', how='left')
    print(top_participation_display[['puma_name', 'ctc_participation_rate', 'units_with_ctc']].round(2))
    print()
    
    # 5. Summary statistics
    print("5. SUMMARY STATISTICS")
    print("-" * 19)
    
    total_units = len(tax_units_with_ctc)
    total_ctc = tax_units_with_ctc['ctc_ctc_total'].sum()
    units_with_ctc = len(tax_units_with_ctc[tax_units_with_ctc['ctc_ctc_total'] > 0])
    
    print(f"Total Tax Units: {total_units:,}")
    print(f"Units with CTC: {units_with_ctc:,}")
    print(f"Statewide Participation Rate: {units_with_ctc/total_units*100:.1f}%")
    print(f"Total CTC Amount: ${total_ctc:,.2f}")
    print(f"Average CTC per Unit: ${total_ctc/total_units:.2f}")
    print(f"Average CTC per Recipient: ${total_ctc/units_with_ctc:.2f}")
    
    # Geographic variation
    puma_avg_ctc = puma_analysis['average_ctc_per_unit']
    print(f"\nGeographic Variation (by PUMA):")
    print(f"Highest Average CTC: ${puma_avg_ctc.max():.2f}")
    print(f"Lowest Average CTC: ${puma_avg_ctc.min():.2f}")
    print(f"Coefficient of Variation: {puma_avg_ctc.std()/puma_avg_ctc.mean()*100:.1f}%")


if __name__ == "__main__":
    main()
