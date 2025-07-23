"""
Simplified Geographic Analysis Example

This example demonstrates geographic CTC analysis using existing processed tax units.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from tax.credits.ctc import calculate_ctc_for_tax_units, get_ctc_summary_stats
from analysis.geographic import GeographicAnalyzer, GeographicLevel, create_hawaii_puma_crosswalk


def load_existing_tax_units():
    """Load existing processed tax units and add geographic information."""
    try:
        # Try to load existing tax units
        tax_units = pd.read_parquet('data/processed/test_tax_units.parquet')
        print(f"Loaded {len(tax_units)} existing tax units")
        
        # Add geographic information if missing
        if 'PUMA' not in tax_units.columns:
            # Assign random PUMAs for demonstration
            pumas = ['00100', '00200', '00300', '00400', '00500']
            tax_units['PUMA'] = np.random.choice(pumas, size=len(tax_units))
            tax_units['state'] = 'Hawaii'
            tax_units['PWGTP'] = np.random.randint(50, 200, size=len(tax_units))
        
        return tax_units
        
    except FileNotFoundError:
        print("No existing tax units found. Creating sample data...")
        return create_sample_tax_units()


def create_sample_tax_units():
    """Create sample tax units with proper structure."""
    np.random.seed(42)
    
    pumas = ['00100', '00200', '00300', '00400', '00500']
    puma_names = ['Honolulu Urban', 'Honolulu Suburban', 'Big Island', 'Maui', 'Kauai']
    
    data = []
    
    for i in range(200):  # Create 200 tax units
        puma = np.random.choice(pumas)
        
        # Create dependents with proper structure
        num_children = np.random.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])
        dependents = []
        
        for j in range(num_children):
            age = np.random.randint(0, 18)
            dependents.append({
                'age': age,
                'relationship': '22',  # Natural child
                'citizenship': '1'     # US citizen
            })
        
        # Create tax unit
        tax_unit = {
            'filer_id': f'TU_{i+1:03d}',
            'filing_status': np.random.choice([
                'single', 'married_filing_jointly', 'head_of_household'
            ], p=[0.4, 0.4, 0.2]),
            'income': max(15000, np.random.normal(55000, 25000)),
            'dependents': dependents,
            'num_dependents': len(dependents),
            'hh_id': f'HH_{i+1:03d}',
            'PUMA': puma,
            'state': 'Hawaii',
            'PWGTP': np.random.randint(50, 200)
        }
        
        data.append(tax_unit)
    
    return pd.DataFrame(data)


def main():
    """Run simplified geographic analysis."""
    print("Simplified Geographic CTC Analysis")
    print("=" * 40)
    
    # Load tax units
    print("Loading tax units...")
    tax_units = load_existing_tax_units()
    print(f"Working with {len(tax_units)} tax units")
    print()
    
    # Calculate CTC
    print("Calculating CTC...")
    tax_units_with_ctc = calculate_ctc_for_tax_units(tax_units)
    print("CTC calculations completed")
    print()
    
    # Basic CTC statistics
    print("Overall CTC Statistics:")
    print("-" * 22)
    total_ctc = tax_units_with_ctc['ctc_ctc_total'].sum()
    units_with_ctc = len(tax_units_with_ctc[tax_units_with_ctc['ctc_ctc_total'] > 0])
    
    print(f"Total Tax Units: {len(tax_units_with_ctc):,}")
    print(f"Units with CTC: {units_with_ctc:,}")
    print(f"Participation Rate: {units_with_ctc/len(tax_units_with_ctc)*100:.1f}%")
    print(f"Total CTC Amount: ${total_ctc:,.2f}")
    print(f"Average CTC per Unit: ${total_ctc/len(tax_units_with_ctc):.2f}")
    if units_with_ctc > 0:
        print(f"Average CTC per Recipient: ${total_ctc/units_with_ctc:.2f}")
    print()
    
    # Geographic analysis
    print("Geographic Analysis:")
    print("-" * 19)
    
    # Create analyzer
    crosswalk = create_hawaii_puma_crosswalk()
    analyzer = GeographicAnalyzer(tax_units_with_ctc, crosswalk)
    
    # State-level analysis
    print("State Level:")
    state_analysis = analyzer.analyze_ctc_by_geography(GeographicLevel.STATE)
    for _, row in state_analysis.iterrows():
        print(f"  {row['state']}: ${row['total_ctc_amount']:,.2f} total, "
              f"${row['average_ctc_per_unit']:.2f} avg, "
              f"{row['ctc_participation_rate']:.1f}% participation")
    print()
    
    # PUMA-level analysis
    print("PUMA Level:")
    puma_analysis = analyzer.analyze_ctc_by_geography(GeographicLevel.PUMA)
    puma_analysis = puma_analysis.sort_values('total_ctc_amount', ascending=False)
    
    for _, row in puma_analysis.iterrows():
        print(f"  PUMA {row['PUMA']}: ${row['total_ctc_amount']:,.2f} total, "
              f"${row['average_ctc_per_unit']:.2f} avg, "
              f"{row['total_tax_units']} units")
    print()
    
    # Legislative district analysis
    print("Legislative Districts:")
    try:
        senate_analysis = analyzer.analyze_ctc_by_geography(GeographicLevel.SENATE_DISTRICT)
        print("Senate Districts:")
        for _, row in senate_analysis.iterrows():
            print(f"  {row['senate_district']}: ${row['total_ctc_amount']:,.2f} total, "
                  f"{row['total_tax_units']} units")
        print()
        
        house_analysis = analyzer.analyze_ctc_by_geography(GeographicLevel.HOUSE_DISTRICT)
        print("House Districts:")
        for _, row in house_analysis.iterrows():
            print(f"  {row['house_district']}: ${row['total_ctc_amount']:,.2f} total, "
                  f"{row['total_tax_units']} units")
        
    except ValueError as e:
        print(f"District analysis not available: {e}")
    
    print()
    print("Analysis completed!")


if __name__ == "__main__":
    main()
