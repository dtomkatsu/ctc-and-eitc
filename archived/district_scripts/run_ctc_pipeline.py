#!/usr/bin/env python3
"""
Run the full CTC estimation pipeline with geographic analysis.
"""
import sys
import os
import pandas as pd
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

from src.tax.units.constructor import TaxUnitConstructor
from src.tax.credits.ctc import calculate_ctc_for_tax_units, get_ctc_summary_stats
from src.analysis.geographic import GeographicAnalyzer, GeographicLevel
from src.tax.units.status.irs_based import calibrate_to_soi_totals

# Configuration
OUTPUT_DIR = 'output/ctc_estimates'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_pums_data():
    """Load PUMS data from the data directory."""
    # Define file paths
    pums_dir = 'data/raw/pums'
    person_file = os.path.join(pums_dir, 'psam_p15.csv')
    hh_file = os.path.join(pums_dir, 'psam_h15.csv')
    
    # Load person data
    print("Loading person data...")
    persons = pd.read_csv(person_file, dtype={'SERIALNO': str})
    
    # Load household data
    print("Loading household data...")
    households = pd.read_csv(hh_file, dtype={'SERIALNO': str})
    
    # Check if we have data
    if len(households) == 0 or len(persons) == 0:
        print("Warning: No data found in the input files")
        return pd.DataFrame(), pd.DataFrame()
        
    # Filter for Hawaii (if state column exists)
    if 'state' in households.columns:
        households = households[households['state'] == 'Hawaii']
    elif 'ST' in households.columns:
        households = households[households['ST'] == '15']  # Hawaii's FIPS code
    
    # Only filter persons if we have households
    if len(households) > 0:
        persons = persons[persons['SERIALNO'].isin(households['SERIALNO'])]
    
    print(f"Loaded {len(persons):,} persons and {len(households):,} households")
    return persons, households

def main():
    """Run the full CTC estimation pipeline."""
    print("Starting CTC Estimation Pipeline")
    print("=" * 50)
    
    # 1. Load data
    print("\n1. Loading PUMS data...")
    persons, households = load_pums_data()
    
    # 2. Construct tax units
    print("\n2. Constructing tax units...")
    constructor = TaxUnitConstructor(persons, households)
    tax_units = constructor.create_rule_based_units()
    
    if isinstance(tax_units, list):
        tax_units = pd.DataFrame(tax_units)
    
    print(f"Created {len(tax_units):,} tax units")
    
    # 2.5 Apply SOI calibration to match expected filing status distribution
    print("\n2.5 Applying SOI calibration to match expected filing status distribution...")
    calibrated_units = calibrate_to_soi_totals(
        tax_units,
        target_single_pct=0.51,    # 51% single
        target_joint_pct=0.36,     # 36% joint
        target_hoh_pct=0.096,      # 9.6% head of household
        target_mfs_pct=0.034       # 3.4% married filing separately
    )
    
    # Count units by filing status after calibration
    status_counts = pd.Series([unit['filing_status'] for unit in calibrated_units]).value_counts()
    print("  - Filing status distribution after SOI calibration:")
    for status, count in status_counts.items():
        print(f"    - {status}: {count} units ({(count/len(calibrated_units)):.1%})")
    
    # 3. Calculate CTC
    print("\n3. Calculating CTC benefits...")
    tax_units = calculate_ctc_for_tax_units(calibrated_units)
    
    # 4. Geographic analysis (if we have data)
    if len(tax_units) > 0:
        print("\n4. Performing geographic analysis...")
        
        # Initialize geographic analyzer with just the tax units (other data is loaded internally)
        geo_analyzer = GeographicAnalyzer(tax_units)
        
        # Analyze at different geographic levels
        for level in [GeographicLevel.STATE, GeographicLevel.PUMA]:
            print(f"\nAnalyzing at {level.name} level...")
            try:
                results = geo_analyzer.analyze(level)
                
                # Save results
                filename = f"ctc_estimates_{level.name.lower()}.csv"
                results.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)
                print(f"  - Saved results to {os.path.join(OUTPUT_DIR, filename)}")
            except Exception as e:
                print(f"  - Error analyzing {level.name} level: {e}")
                continue
    
    # 5. Generate summary statistics
    print("\n5. Generating summary statistics...")
    summary = get_ctc_summary_stats(tax_units)
    
    # Save summary
    summary_file = os.path.join(OUTPUT_DIR, 'ctc_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("CTC Program Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total Tax Units: {len(tax_units):,}\n")
        f.write(f"Total Children Benefiting: {summary['total_children_benefiting']:,.0f}\n")
        f.write(f"Total CTC Amount: ${summary['total_ctc_amount']:,.0f}\n")
        f.write(f"Average CTC per Family: ${summary['avg_ctc_per_family']:,.2f}\n")
        f.write(f"Percent of Families with Children Receiving CTC: {summary['pct_families_with_children_receiving']:.1f}%\n")
    
    print(f"\nPipeline completed. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
