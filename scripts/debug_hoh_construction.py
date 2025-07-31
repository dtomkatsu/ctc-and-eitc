#!/usr/bin/env python3
"""
Debug why qualified HoH adults aren't making it into final tax units.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import logging
from src.data.pums_loader import PUMSDataLoader
from src.tax.units.status.hoh import is_head_of_household
from src.tax.units.constructor import TaxUnitConstructor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def debug_hoh_construction():
    print("Loading PUMS data...")
    loader = PUMSDataLoader()
    person_df, hh_df = loader.load_data()
    
    # Focus on a sample of households with both adults and children
    households_with_children = set(person_df[person_df['AGEP'] < 18]['SERIALNO'].unique())
    households_with_adults = set(person_df[person_df['AGEP'] >= 18]['SERIALNO'].unique())
    households_with_both = list(households_with_children.intersection(households_with_adults))
    
    # Take first 50 households for detailed analysis
    sample_households = households_with_both[:50]
    print(f"Analyzing {len(sample_households)} sample households...")
    
    # Find adults who qualify for HoH in these households
    qualified_hoh_adults = []
    for hh_id in sample_households:
        hh_members = person_df[person_df['SERIALNO'] == hh_id]
        adults = hh_members[hh_members['AGEP'] >= 18]
        
        for _, adult in adults.iterrows():
            if is_head_of_household(adult, person_df):
                qualified_hoh_adults.append({
                    'person_id': adult.name,
                    'household': hh_id,
                    'age': adult.get('AGEP'),
                    'mar': adult.get('MAR'),
                    'rel': adult.get('RELSHIPP'),
                    'income': adult.get('PINCP', 0)
                })
    
    print(f"Found {len(qualified_hoh_adults)} adults who qualify for HoH in sample households")
    
    # Now construct tax units for these households and see what happens
    constructor = TaxUnitConstructor(person_df, hh_df)
    
    # Process each sample household individually
    hoh_results = []
    for hh_id in sample_households:
        hh_members = person_df[person_df['SERIALNO'] == hh_id]
        
        # Process this household
        tax_units = constructor._process_household(hh_members)
        
        # Check what filing statuses were created
        filing_statuses = [unit['filing_status'] for unit in tax_units]
        hoh_count = filing_statuses.count('head_of_household')
        
        # Find qualified HoH adults in this household
        qualified_in_hh = [adult for adult in qualified_hoh_adults if adult['household'] == hh_id]
        
        hoh_results.append({
            'household': hh_id,
            'qualified_hoh_adults': len(qualified_in_hh),
            'created_hoh_units': hoh_count,
            'all_filing_statuses': filing_statuses,
            'qualified_adults': qualified_in_hh
        })
    
    # Analyze results
    print(f"\n{'='*60}")
    print(f"HOH CONSTRUCTION DEBUG RESULTS")
    print(f"{'='*60}")
    
    total_qualified = sum(result['qualified_hoh_adults'] for result in hoh_results)
    total_created = sum(result['created_hoh_units'] for result in hoh_results)
    
    print(f"Total qualified HoH adults in sample: {total_qualified}")
    print(f"Total HoH tax units created: {total_created}")
    print(f"Conversion rate: {total_created/total_qualified*100:.1f}%" if total_qualified > 0 else "N/A")
    
    # Show detailed breakdown
    print(f"\nDetailed household breakdown:")
    mismatch_count = 0
    for result in hoh_results:
        if result['qualified_hoh_adults'] != result['created_hoh_units']:
            mismatch_count += 1
            print(f"  HH {result['household']}: {result['qualified_hoh_adults']} qualified → {result['created_hoh_units']} created")
            print(f"    Filing statuses: {result['all_filing_statuses']}")
            print(f"    Qualified adults: {result['qualified_adults']}")
    
    print(f"\nHouseholds with mismatches: {mismatch_count}/{len(sample_households)}")
    
    # Show filing status distribution
    all_statuses = []
    for result in hoh_results:
        all_statuses.extend(result['all_filing_statuses'])
    
    status_counts = {}
    for status in all_statuses:
        status_counts[status] = status_counts.get(status, 0) + 1
    
    print(f"\nFiling status distribution in sample:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

if __name__ == "__main__":
    debug_hoh_construction()
