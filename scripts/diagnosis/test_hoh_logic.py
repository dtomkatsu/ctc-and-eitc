#!/usr/bin/env python3
"""
Test HoH Qualification Logic

Test if the HoH logic is working correctly after ADJINC fixes.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
from src.tax.units.status.hoh import is_head_of_household

print("="*80)
print("HOH LOGIC TEST")
print("="*80)

# Load PUMS data
persons = pd.read_csv('data/raw/pums/psam_p15.csv')
persons = persons.set_index(persons['SERIALNO'].astype(str) + '_' + persons.index.astype(str))

# Find unmarried householders with children
unmarried_householders = persons[
    (persons['AGEP'] >= 18) & 
    (persons['MAR'].isin([3, 4, 5])) & 
    (persons['RELSHIPP'] == 20)
]

print(f"\nTesting {len(unmarried_householders):,} unmarried householders")

# Test a sample
qualified_count = 0
failed_unmarried = 0
failed_no_qualifying_person = 0
failed_no_dependents = 0
failed_home_cost = 0

sample_size = min(1000, len(unmarried_householders))
for i, (person_id, person) in enumerate(list(unmarried_householders.iterrows())[:sample_size]):
    # Get household members
    hh_id = person['SERIALNO']
    household = persons[persons['SERIALNO'] == hh_id]
    
    # Check if has children
    children = household[
        (household['AGEP'] < 18) &
        (household['RELSHIPP'].isin([22, 23, 24, 25]))
    ]
    
    has_deps = len(children) > 0
    
    if has_deps:
        # Test if qualifies
        if is_head_of_household(person, household, has_dependents=True):
            qualified_count += 1
        else:
            # Diagnose why failed
            from src.tax.units.status.hoh import _is_unmarried, _has_qualifying_person, _paid_half_home_cost
            
            if not _is_unmarried(person, household):
                failed_unmarried += 1
            elif not _has_qualifying_person(person, household):
                failed_no_qualifying_person += 1
            elif not _paid_half_home_cost(person, household):
                failed_home_cost += 1
            else:
                failed_no_dependents += 1

print(f"\nResults from {sample_size} unmarried householders with children:")
print(f"  Qualified as HoH: {qualified_count}")
print(f"  Failed - not unmarried: {failed_unmarried}")
print(f"  Failed - no qualifying person: {failed_no_qualifying_person}")  
print(f"  Failed - home cost test: {failed_home_cost}")
print(f"  Failed - no dependents param: {failed_no_dependents}")

# Extrapolate
if sample_size > 0:
    qualification_rate = qualified_count / sample_size
    print(f"\nQualification rate: {qualification_rate*100:.1f}%")
    
    # Estimate for all unmarried householders with children
    unmarried_hh_with_children = unmarried_householders[
        unmarried_householders['SERIALNO'].isin(
            persons[(persons['AGEP'] < 18) & (persons['RELSHIPP'].isin([22, 23, 24, 25]))]['SERIALNO'].unique()
        )
    ]
    
    estimated_qualified = len(unmarried_hh_with_children) * qualification_rate
    weighted_estimate = unmarried_hh_with_children['PWGTP'].sum() * qualification_rate
    
    print(f"\nEstimated HoH filers:")
    print(f"  Unweighted: {estimated_qualified:,.0f}")
    print(f"  Weighted: {weighted_estimate:,.0f}")
    print(f"  DOTAX target: 67,393")
    print(f"  Gap: {weighted_estimate - 67393:+,.0f}")
