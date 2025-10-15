#!/usr/bin/env python3
"""
Diagnose HoH Dependent Identification Issues

This script analyzes why we're not identifying enough dependents.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np

print("="*80)
print("HOH DEPENDENT IDENTIFICATION DIAGNOSTIC")
print("="*80)

# Load PUMS data
persons = pd.read_csv('data/raw/pums/psam_p15.csv')
print(f"\nTotal persons: {len(persons):,}")

# Check relationship codes in data
print("\n" + "="*80)
print("RELATIONSHIP CODES IN DATA")
print("="*80)

rel_counts = persons['RELSHIPP'].value_counts().sort_index()
print("\nRELSHIPP distribution:")
for rel, count in rel_counts.items():
    pct = count / len(persons) * 100
    print(f"  {rel}: {count:>6,} ({pct:>5.1f}%)")

# Check children in households
print("\n" + "="*80)
print("CHILDREN ANALYSIS")
print("="*80)

children = persons[persons['AGEP'] < 18]
print(f"\nChildren under 18: {len(children):,}")

print("\nChildren by RELSHIPP:")
for rel in sorted(children['RELSHIPP'].unique()):
    count = (children['RELSHIPP'] == rel).sum()
    print(f"  {rel}: {count:>5,}")

# Check households with children
households_with_children = children['SERIALNO'].nunique()
print(f"\nHouseholds with children: {households_with_children:,}")

# Check children with income
children_with_income = children[children['PINCP'].fillna(0) > 0]
print(f"\nChildren with ANY income: {len(children_with_income):,} ({len(children_with_income)/len(children)*100:.1f}%)")

income_brackets = [0, 1000, 5000, 10000, 50000]
for i in range(len(income_brackets) - 1):
    low, high = income_brackets[i], income_brackets[i+1]
    count = ((children['PINCP'].fillna(0) > low) & (children['PINCP'].fillna(0) <= high)).sum()
    print(f"  Income ${low:,}-${high:,}: {count:>5,}")

# Check students
print("\n" + "="*80)
print("STUDENT ANALYSIS")
print("="*80)

students_18_23 = persons[(persons['AGEP'] >= 18) & (persons['AGEP'] < 24)]
print(f"\nPersons aged 18-23: {len(students_18_23):,}")

if 'SCHL' in students_18_23.columns:
    # SCHL >= 16 is "Some college" or higher
    college_students = students_18_23[students_18_23['SCHL'] >= 16]
    print(f"In college or higher education: {len(college_students):,} ({len(college_students)/len(students_18_23)*100:.1f}%)")
    
    print(f"\nCollege students by RELSHIPP:")
    for rel in sorted(college_students['RELSHIPP'].unique()):
        count = (college_students['RELSHIPP'] == rel).sum()
        if count > 0:
            print(f"  {rel}: {count:>5,}")

# Check potential HoH filers (unmarried adults)
print("\n" + "="*80)
print("POTENTIAL HOH FILERS")
print("="*80)

# MAR = 5 is "Never married", 3 is "Divorced", 4 is "Widowed"
unmarried = persons[(persons['AGEP'] >= 18) & (persons['MAR'].isin([3, 4, 5]))]
print(f"\nUnmarried adults: {len(unmarried):,}")

# Check how many are householders
unmarried_householders = unmarried[unmarried['RELSHIPP'] == 20]
print(f"Unmarried householders (RELSHIPP=20): {len(unmarried_householders):,}")

# Check if they live with children
print(f"\nChecking if unmarried householders live with children...")
households_with_unmarried_hh = unmarried_householders['SERIALNO'].unique()
children_in_these_hh = persons[
    (persons['SERIALNO'].isin(households_with_unmarried_hh)) &
    (persons['AGEP'] < 18) &
    (persons['RELSHIPP'].isin([22, 23, 24, 25]))  # Child, adopted, step, grandchild
]
print(f"Children in unmarried householder households: {len(children_in_these_hh):,}")

unique_hh_with_children = children_in_these_hh['SERIALNO'].nunique()
print(f"Unmarried householder households with qualifying children: {unique_hh_with_children:,}")

# Estimate weighted counts
if 'PWGTP' in unmarried_householders.columns:
    weighted_count = unmarried_householders[
        unmarried_householders['SERIALNO'].isin(children_in_these_hh['SERIALNO'].unique())
    ]['PWGTP'].sum()
    print(f"Weighted estimate of potential HoH filers: {weighted_count:,.0f}")
    print(f"DOTAX target: 67,393")
    print(f"Gap: {weighted_count - 67393:+,.0f}")

# Check support test issue
print("\n" + "="*80)
print("SUPPORT TEST ANALYSIS")
print("="*80)

print("\nChildren (<18) with income by bracket:")
print("(These might be incorrectly disqualified by 'provides own support' test)")

for bracket in [500, 1000, 2000, 5000, 10000, 20000]:
    count = (children['PINCP'].fillna(0) < bracket).sum()
    pct = count / len(children) * 100
    print(f"  Income < ${bracket:>6,}: {count:>6,} ({pct:>5.1f}%)")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

print("\n1. **Fix 'provides_over_half_own_support' function**")
print("   Current: Returns True if person has ANY income")
print("   Should: Compare income to household costs/support")
print("   Impact: Would qualify most children with part-time jobs")

print("\n2. **Fix relationship code handling**")
print("   Issue: Uses code '3' which doesn't exist in PUMS")
print("   Should: Use codes 22, 23, 24, 25 for children")

print("\n3. **Relax student support test**")
print("   Current: Students with any income disqualified")
print("   Should: Allow students under 24 with reasonable income")

print("\n4. **Consider grandchildren**")
print("   Code 25 (grandchild) should be included more broadly")
