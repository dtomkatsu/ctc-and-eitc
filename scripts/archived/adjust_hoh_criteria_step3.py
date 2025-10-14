#!/usr/bin/env python3
"""
Step 3: Adjust Head of Household criteria to reduce overcount.
Current: 14.1% vs 9.6% expected (4.5 percentage point overcount).
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.pums_loader import PUMSDataLoader
from tax.units.status.hoh import is_head_of_household, _has_qualifying_person, _paid_half_home_cost

def analyze_hoh_overcount():
    """Analyze current HoH filers to identify criteria for tightening."""
    
    print("Step 3: Analyzing Head of Household overcount...")
    
    # Load data
    pums_loader = PUMSDataLoader()
    person_df, hh_df = pums_loader.load_data()
    tax_units = pd.read_parquet("src/data/processed/tax_units_rule_based.parquet")
    
    # Filter out zero-weight households
    hh_weights = hh_df[hh_df['WGTP'] > 0]
    person_df = person_df[person_df['SERIALNO'].isin(hh_weights['SERIALNO'])]
    
    # Get current HoH filers
    hoh_units = tax_units[tax_units['filing_status'] == 'head_of_household']
    hoh_filer_ids = hoh_units['primary_filer_id'].tolist()
    
    print(f"Current HoH tax units: {len(hoh_units):,}")
    print(f"Current HoH filers: {len(hoh_filer_ids):,}")
    
    # Create person ID mapping
    person_df['person_id'] = person_df['SERIALNO'] + '_' + person_df['SPORDER'].astype(str)
    person_lookup = person_df.set_index('person_id')
    
    # Analyze characteristics of current HoH filers
    hoh_persons = person_lookup.loc[person_lookup.index.isin(hoh_filer_ids)]
    
    print(f"\nCurrent HoH Filer Characteristics:")
    print(f"Age distribution:")
    age_bins = [18, 25, 35, 45, 55, 65, 100]
    age_groups = pd.cut(hoh_persons['AGEP'], bins=age_bins, include_lowest=True)
    age_dist = age_groups.value_counts().sort_index()
    for age_range, count in age_dist.items():
        pct = count / len(hoh_persons) * 100
        print(f"  {age_range}: {count:,} ({pct:.1f}%)")
    
    print(f"\nMarital status:")
    mar_dist = hoh_persons['MAR'].value_counts()
    for mar_code, count in mar_dist.items():
        pct = count / len(hoh_persons) * 100
        mar_desc = {1: 'Married', 2: 'Widowed', 3: 'Divorced', 4: 'Separated', 5: 'Never married'}
        print(f"  {mar_desc.get(mar_code, f'Code {mar_code}')}: {count:,} ({pct:.1f}%)")
    
    print(f"\nRelationship to householder:")
    rel_dist = hoh_persons['RELSHIPP'].value_counts()
    for rel_code, count in rel_dist.items():
        pct = count / len(hoh_persons) * 100
        print(f"  RELSHIPP {rel_code}: {count:,} ({pct:.1f}%)")
    
    # Analyze dependents per HoH unit
    hoh_units_with_deps = hoh_units.copy()
    dep_dist = hoh_units_with_deps['num_dependents'].value_counts().sort_index()
    print(f"\nDependents per HoH unit:")
    for num_deps, count in dep_dist.items():
        pct = count / len(hoh_units) * 100
        print(f"  {num_deps} dependents: {count:,} ({pct:.1f}%)")
    
    # Sample detailed analysis of HoH filers
    print(f"\nDetailed Analysis of Sample HoH Filers:")
    sample_size = min(20, len(hoh_persons))
    sample_hoh = hoh_persons.head(sample_size)
    
    qualifying_person_issues = 0
    home_cost_issues = 0
    
    for person_id, person in sample_hoh.iterrows():
        hh_persons = person_df[person_df['SERIALNO'] == person['SERIALNO']]
        
        # Check qualifying person logic
        has_qualifying = _has_qualifying_person(person, hh_persons)
        
        # Check home cost logic  
        paid_half_cost = _paid_half_home_cost(person, hh_persons)
        
        # Get number of dependents for this filer
        unit_deps = hoh_units[hoh_units['primary_filer_id'] == person_id]['num_dependents'].iloc[0]
        
        print(f"  {person_id}: AGE={person['AGEP']}, MAR={person['MAR']}, REL={person['RELSHIPP']}")
        print(f"    Qualifying person: {has_qualifying}, Home cost: {paid_half_cost}, Deps: {unit_deps}")
        
        if not has_qualifying:
            qualifying_person_issues += 1
        if not paid_half_cost:
            home_cost_issues += 1
    
    print(f"\nSample Analysis Results:")
    print(f"  Qualifying person failures: {qualifying_person_issues}/{sample_size} ({qualifying_person_issues/sample_size*100:.1f}%)")
    print(f"  Home cost failures: {home_cost_issues}/{sample_size} ({home_cost_issues/sample_size*100:.1f}%)")
    
    # Calculate target reduction needed
    current_weighted_hoh = 115754  # From Step 1 results
    target_weighted_hoh = int(743109 * 0.096)  # 9.6% of total returns
    reduction_needed = current_weighted_hoh - target_weighted_hoh
    reduction_pct = reduction_needed / current_weighted_hoh * 100
    
    print(f"\n{'='*80}")
    print(f"REDUCTION TARGET ANALYSIS")
    print(f"{'='*80}")
    
    print(f"Current weighted HoH: {current_weighted_hoh:,}")
    print(f"Target weighted HoH: {target_weighted_hoh:,}")
    print(f"Reduction needed: {reduction_needed:,} ({reduction_pct:.1f}%)")
    
    # Suggest specific criteria adjustments
    print(f"\nSUGGESTED CRITERIA ADJUSTMENTS:")
    
    # Analyze by marital status
    married_hoh = (hoh_persons['MAR'] == 1).sum()
    if married_hoh > 0:
        married_pct = married_hoh / len(hoh_persons) * 100
        print(f"1. MARRIED HOH FILERS: {married_hoh:,} ({married_pct:.1f}%)")
        print(f"   - Consider stricter criteria for married individuals filing HoH")
        print(f"   - May indicate separation/special circumstances")
    
    # Analyze by age
    young_hoh = (hoh_persons['AGEP'] < 25).sum()
    if young_hoh > 0:
        young_pct = young_hoh / len(hoh_persons) * 100
        print(f"2. YOUNG HOH FILERS (<25): {young_hoh:,} ({young_pct:.1f}%)")
        print(f"   - Consider stricter qualifying person criteria for young filers")
    
    # Analyze by dependents
    no_deps_hoh = (hoh_units['num_dependents'] == 0).sum()
    if no_deps_hoh > 0:
        no_deps_pct = no_deps_hoh / len(hoh_units) * 100
        print(f"3. HOH WITH NO DEPENDENTS: {no_deps_hoh:,} ({no_deps_pct:.1f}%)")
        print(f"   - These should likely be single filers instead")
        print(f"   - Check qualifying person logic for non-dependent qualifying persons")
    
    # Analyze by relationship code
    non_householder_hoh = (hoh_persons['RELSHIPP'] != 20).sum()
    if non_householder_hoh > 0:
        non_hh_pct = non_householder_hoh / len(hoh_persons) * 100
        print(f"4. NON-HOUSEHOLDER HOH: {non_householder_hoh:,} ({non_hh_pct:.1f}%)")
        print(f"   - Consider restricting HoH to householders (RELSHIPP=20)")
    
    return {
        'current_hoh_count': len(hoh_units),
        'target_reduction': reduction_needed,
        'reduction_percentage': reduction_pct,
        'married_hoh': married_hoh,
        'young_hoh': young_hoh,
        'no_deps_hoh': no_deps_hoh,
        'non_householder_hoh': non_householder_hoh
    }

if __name__ == "__main__":
    analyze_hoh_overcount()
