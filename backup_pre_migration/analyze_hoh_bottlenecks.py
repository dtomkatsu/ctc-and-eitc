#!/usr/bin/env python3
"""
Analyze Head of Household bottlenecks to understand why HoH rates are still low.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import logging
from src.data.pums_loader import PUMSDataLoader
from src.tax.units.status.hoh import is_head_of_household, _is_unmarried, _has_qualifying_person, _paid_half_home_cost

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def analyze_hoh_bottlenecks():
    print("Loading PUMS data...")
    loader = PUMSDataLoader()
    person_df, hh_df = loader.load_data()
    
    print(f"Loaded {len(person_df)} persons and {len(hh_df)} households")
    
    # Focus on households with both adults and children
    households_with_children = set(person_df[person_df['AGEP'] < 18]['SERIALNO'].unique())
    households_with_adults = set(person_df[person_df['AGEP'] >= 18]['SERIALNO'].unique())
    households_with_both = households_with_children.intersection(households_with_adults)
    
    print(f"Found {len(households_with_both)} households with both adults and children")
    
    # Analyze all adults in these households
    adults_in_mixed_households = person_df[
        (person_df['AGEP'] >= 18) & 
        (person_df['SERIALNO'].isin(households_with_both))
    ].copy()
    
    print(f"Found {len(adults_in_mixed_households)} adults in households with children")
    
    # Test each adult for HoH qualification and track reasons for failure
    results = {
        'total_adults': 0,
        'unmarried_pass': 0,
        'unmarried_fail': 0,
        'qualifying_person_pass': 0,
        'qualifying_person_fail': 0,
        'home_cost_pass': 0,
        'home_cost_fail': 0,
        'final_hoh_qualify': 0,
        'marital_status_breakdown': {},
        'relationship_breakdown': {}
    }
    
    sample_failures = {
        'unmarried_failures': [],
        'qualifying_person_failures': [],
        'home_cost_failures': []
    }
    
    for _, adult in adults_in_mixed_households.iterrows():
        results['total_adults'] += 1
        
        # Track marital status
        mar_status = adult.get('MAR', 'Unknown')
        results['marital_status_breakdown'][mar_status] = results['marital_status_breakdown'].get(mar_status, 0) + 1
        
        # Track relationship
        rel_status = adult.get('RELSHIPP', 'Unknown')
        results['relationship_breakdown'][rel_status] = results['relationship_breakdown'].get(rel_status, 0) + 1
        
        # Test each step of HoH qualification
        is_unmarried = _is_unmarried(adult, person_df)
        if is_unmarried:
            results['unmarried_pass'] += 1
        else:
            results['unmarried_fail'] += 1
            if len(sample_failures['unmarried_failures']) < 5:
                sample_failures['unmarried_failures'].append({
                    'person_id': adult.name,
                    'age': adult.get('AGEP'),
                    'mar': adult.get('MAR'),
                    'rel': adult.get('RELSHIPP')
                })
            continue
        
        has_qualifying_person = _has_qualifying_person(adult, person_df)
        if has_qualifying_person:
            results['qualifying_person_pass'] += 1
        else:
            results['qualifying_person_fail'] += 1
            if len(sample_failures['qualifying_person_failures']) < 5:
                sample_failures['qualifying_person_failures'].append({
                    'person_id': adult.name,
                    'age': adult.get('AGEP'),
                    'mar': adult.get('MAR'),
                    'rel': adult.get('RELSHIPP'),
                    'household': adult.get('SERIALNO')
                })
            continue
        
        paid_half_cost = _paid_half_home_cost(adult, person_df)
        if paid_half_cost:
            results['home_cost_pass'] += 1
        else:
            results['home_cost_fail'] += 1
            if len(sample_failures['home_cost_failures']) < 5:
                sample_failures['home_cost_failures'].append({
                    'person_id': adult.name,
                    'age': adult.get('AGEP'),
                    'mar': adult.get('MAR'),
                    'rel': adult.get('RELSHIPP'),
                    'household': adult.get('SERIALNO')
                })
            continue
        
        # If we get here, they qualify for HoH
        results['final_hoh_qualify'] += 1
    
    # Print analysis results
    print(f"\n{'='*60}")
    print(f"HEAD OF HOUSEHOLD BOTTLENECK ANALYSIS")
    print(f"{'='*60}")
    
    print(f"\nTotal adults in households with children: {results['total_adults']}")
    print(f"\nStep-by-step qualification results:")
    print(f"  1. Unmarried test:")
    print(f"     - Pass: {results['unmarried_pass']} ({results['unmarried_pass']/results['total_adults']*100:.1f}%)")
    print(f"     - Fail: {results['unmarried_fail']} ({results['unmarried_fail']/results['total_adults']*100:.1f}%)")
    
    if results['unmarried_pass'] > 0:
        print(f"  2. Qualifying person test (of {results['unmarried_pass']} unmarried):")
        print(f"     - Pass: {results['qualifying_person_pass']} ({results['qualifying_person_pass']/results['unmarried_pass']*100:.1f}%)")
        print(f"     - Fail: {results['qualifying_person_fail']} ({results['qualifying_person_fail']/results['unmarried_pass']*100:.1f}%)")
    
    if results['qualifying_person_pass'] > 0:
        print(f"  3. Home cost test (of {results['qualifying_person_pass']} with qualifying persons):")
        print(f"     - Pass: {results['home_cost_pass']} ({results['home_cost_pass']/results['qualifying_person_pass']*100:.1f}%)")
        print(f"     - Fail: {results['home_cost_fail']} ({results['home_cost_fail']/results['qualifying_person_pass']*100:.1f}%)")
    
    print(f"\nFinal HoH qualifications: {results['final_hoh_qualify']} ({results['final_hoh_qualify']/results['total_adults']*100:.1f}%)")
    
    print(f"\nMarital status breakdown:")
    for status, count in sorted(results['marital_status_breakdown'].items()):
        print(f"  MAR {status}: {count} ({count/results['total_adults']*100:.1f}%)")
    
    print(f"\nRelationship breakdown:")
    for rel, count in sorted(results['relationship_breakdown'].items()):
        print(f"  RELSHIPP {rel}: {count} ({count/results['total_adults']*100:.1f}%)")
    
    # Show sample failures
    print(f"\nSample failures:")
    print(f"  Unmarried test failures: {sample_failures['unmarried_failures']}")
    print(f"  Qualifying person failures: {sample_failures['qualifying_person_failures']}")
    print(f"  Home cost test failures: {sample_failures['home_cost_failures']}")

if __name__ == "__main__":
    analyze_hoh_bottlenecks()
