#!/usr/bin/env python3
"""
Diagnose Overcounting - Why 1.59 tax units per household?

This script analyzes the tax units to understand overcounting.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np

print("="*80)
print("OVERCOUNTING DIAGNOSTIC")
print("="*80)

# Load the regenerated tax units
tax_units = pd.read_parquet('data/processed/tax_units_regenerated_20251014_194028.parquet')

print(f"\nTotal tax units: {len(tax_units):,}")

# Analyze tax units per household
print("\n" + "="*80)
print("TAX UNITS PER HOUSEHOLD DISTRIBUTION")
print("="*80)

units_per_hh = tax_units.groupby('hh_id').size()

print(f"\nTotal households: {len(units_per_hh):,}")
print(f"Average units/household: {units_per_hh.mean():.2f}")
print(f"Median units/household: {units_per_hh.median():.0f}")
print(f"Max units/household: {units_per_hh.max():.0f}")

print("\nDistribution:")
for i in range(1, int(units_per_hh.max()) + 1):
    count = (units_per_hh == i).sum()
    pct = count / len(units_per_hh) * 100
    print(f"  {i} unit(s):  {count:>6,} households ({pct:>5.1f}%)")

# Analyze households with multiple units
print("\n" + "="*80)
print("HOUSEHOLDS WITH MULTIPLE TAX UNITS")
print("="*80)

multi_unit_hhs = units_per_hh[units_per_hh > 1].index

print(f"\nHouseholds with >1 unit: {len(multi_unit_hhs):,} ({len(multi_unit_hhs)/len(units_per_hh)*100:.1f}%)")

# Sample 10 households with multiple units
sample_hhs = list(multi_unit_hhs[:10])

for hh_id in sample_hhs:
    hh_units = tax_units[tax_units['hh_id'] == hh_id]
    print(f"\nHousehold {hh_id}: {len(hh_units)} tax units")
    for idx, unit in hh_units.iterrows():
        weight = unit.get('weight', unit.get('PWGTP', 1))
        income = unit.get('income', unit.get('total_income', 0))
        status = unit.get('filing_status', 'unknown')
        deps = unit.get('num_dependents', 0)
        print(f"  - {status:25} Income: ${income:>10,.0f}  Deps: {deps}  Weight: {weight:.0f}")

# Analyze by filing status
print("\n" + "="*80)
print("FILING STATUS IMPACT ON OVERCOUNTING")
print("="*80)

status_counts = tax_units['filing_status'].value_counts()
print("\nFiling status distribution (unweighted):")
for status, count in status_counts.items():
    print(f"  {status:25} {count:>6,} ({count/len(tax_units)*100:>5.1f}%)")

# Check if same adults appear in multiple units
print("\n" + "="*80)
print("CHECKING FOR DUPLICATE ADULTS")
print("="*80)

if 'primary_filer_id' in tax_units.columns:
    primary_counts = tax_units['primary_filer_id'].value_counts()
    duplicates = primary_counts[primary_counts > 1]
    
    if len(duplicates) > 0:
        print(f"\n⚠️  Found {len(duplicates):,} adults appearing in multiple tax units!")
        print(f"   Total duplicate occurrences: {duplicates.sum() - len(duplicates):,}")
        print("\nTop 10 most duplicated adults:")
        for adult_id, count in duplicates.head(10).items():
            print(f"  Adult {adult_id}: appears in {count} tax units")
    else:
        print("\n✅ No duplicate adults found")
else:
    print("\n⚠️  'primary_filer_id' column not found")

# Income distribution
print("\n" + "="*80)
print("INCOME CHARACTERISTICS")
print("="*80)

income_col = 'income' if 'income' in tax_units.columns else 'total_income'
if income_col in tax_units.columns:
    incomes = tax_units[income_col]
    
    print(f"\nIncome statistics:")
    print(f"  Mean: ${incomes.mean():,.0f}")
    print(f"  Median: ${incomes.median():,.0f}")
    print(f"  Min: ${incomes.min():,.0f}")
    print(f"  Max: ${incomes.max():,.0f}")
    
    print(f"\nIncome distribution:")
    print(f"  Zero or negative: {(incomes <= 0).sum():,} ({(incomes <= 0).sum()/len(incomes)*100:.1f}%)")
    print(f"  $1-$12,950: {((incomes > 0) & (incomes < 12950)).sum():,} ({((incomes > 0) & (incomes < 12950)).sum()/len(incomes)*100:.1f}%)")
    print(f"  $12,950+: {(incomes >= 12950).sum():,} ({(incomes >= 12950).sum()/len(incomes)*100:.1f}%)")
    
    # These are likely non-filers
    below_threshold = tax_units[incomes < 12950]
    print(f"\nBelow filing threshold ({len(below_threshold):,} units):")
    print(f"  Filing statuses:")
    for status, count in below_threshold['filing_status'].value_counts().items():
        print(f"    {status:25} {count:>6,}")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

print("\n1. **Lower MAX_TAX_UNITS_PER_HOUSEHOLD**:")
print(f"   Currently: 4")
print(f"   Recommended: 2 (allows married couple + 1 adult child)")
print(f"   Conservative: 3 (allows multi-generational)")

print("\n2. **Add Filing Threshold Filter**:")
below_threshold_count = (tax_units[income_col] < 12950).sum() if income_col in tax_units.columns else 0
print(f"   Remove {below_threshold_count:,} units with income < $12,950")

print("\n3. **Check for Duplicate Adults**:")
if 'primary_filer_id' in tax_units.columns and len(duplicates) > 0:
    print(f"   Fix {len(duplicates):,} adults appearing in multiple units")
else:
    print(f"   No duplicate adult issue detected")

print("\n4. **Expected Impact**:")
expected_removed = 0
if len(multi_unit_hhs) > 0:
    # Estimate: reduce 3+ unit households to 2 units
    excess_units = units_per_hh[units_per_hh > 2] - 2
    expected_removed = excess_units.sum()
    
expected_remaining = len(tax_units) - expected_removed - below_threshold_count
print(f"   Current: {len(tax_units):,} units")
print(f"   Remove multi-unit excess: -{expected_removed:,}")
print(f"   Remove below threshold: -{below_threshold_count:,}")
print(f"   Expected remaining: ~{expected_remaining:,} units")
print(f"   Target (DOTAX): 635,117")
print(f"   Coverage: {expected_remaining/635117*100:.1f}%")
