#!/usr/bin/env python3
"""
Diagnose MFS and Joint Filer Issues

Analyze why MFS is overcounted and Joint is slightly overcounted.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np

print("="*80)
print("MFS AND JOINT FILER DIAGNOSTIC")
print("="*80)

# Load tax units
tax_units = pd.read_parquet('data/processed/tax_units_regenerated_20251015_083731.parquet')

# Current vs target
current = {
    'married_filing_jointly': 228090,
    'married_filing_separately': 24623
}

target = {
    'married_filing_jointly': 216358,
    'married_filing_separately': 16007
}

print("\nCurrent situation:")
print(f"  Joint: {current['married_filing_jointly']:,} (target: {target['married_filing_jointly']:,})")
print(f"  MFS: {current['married_filing_separately']:,} (target: {target['married_filing_separately']:,})")

total_married = current['married_filing_jointly'] + current['married_filing_separately']
target_total_married = target['married_filing_jointly'] + target['married_filing_separately']

print(f"\nTotal married filers: {total_married:,} (target: {target_total_married:,})")
print(f"  Gap: {total_married - target_total_married:+,} ({(total_married/target_total_married - 1)*100:+.1f}%)")

# The issue is we have too many married filers overall
# AND the MFS/Joint ratio is wrong

print("\n" + "="*80)
print("MFS/JOINT RATIO ANALYSIS")
print("="*80)

current_mfs_rate = current['married_filing_separately'] / total_married * 100
target_mfs_rate = target['married_filing_separately'] / target_total_married * 100

print(f"\nMFS rate among married couples:")
print(f"  Current: {current_mfs_rate:.1f}%")
print(f"  Target: {target_mfs_rate:.1f}%")
print(f"  Need to reduce MFS rate from {current_mfs_rate:.1f}% to {target_mfs_rate:.1f}%")

# Check the tax units
joint_units = tax_units[tax_units['filing_status'] == 'married_filing_jointly']
mfs_units = tax_units[tax_units['filing_status'] == 'married_filing_separately']

print("\n" + "="*80)
print("TAX UNIT COUNTS")
print("="*80)

print(f"\nJoint filer tax units: {len(joint_units):,}")
print(f"  Average weight: {joint_units['weight'].mean():.2f}")
print(f"  Total weight: {joint_units['weight'].sum():,.0f}")

print(f"\nMFS tax units: {len(mfs_units):,}")
print(f"  Average weight: {mfs_units['weight'].mean():.2f}")
print(f"  Total weight: {mfs_units['weight'].sum():,.0f}")

# Load PUMS to check married couples
print("\n" + "="*80)
print("PUMS MARRIED COUPLE ANALYSIS")
print("="*80)

persons = pd.read_csv('data/raw/pums/psam_p15.csv')

# Count married couples (householder + spouse pairs)
householders = persons[(persons['MAR'] == 1) & (persons['RELSHIPP'] == 20)]
spouses = persons[(persons['MAR'] == 1) & (persons['RELSHIPP'] == 21)]

# Match by household
couples = []
for _, hh in householders.iterrows():
    hh_id = hh['SERIALNO']
    spouse = spouses[spouses['SERIALNO'] == hh_id]
    if len(spouse) > 0:
        couples.append({
            'SERIALNO': hh_id,
            'hh_weight': hh['PWGTP'],
            'spouse_weight': spouse.iloc[0]['PWGTP']
        })

couples_df = pd.DataFrame(couples)
print(f"\nMarried couples in PUMS: {len(couples_df):,}")
print(f"  Weighted: {couples_df['hh_weight'].sum():,.0f}")

# Our tax units
total_married_units = len(joint_units) + len(mfs_units)
print(f"\nTax units for married couples: {total_married_units:,}")
print(f"  Joint: {len(joint_units):,} ({len(joint_units)/total_married_units*100:.1f}%)")
print(f"  MFS: {len(mfs_units):,} ({len(mfs_units)/total_married_units*100:.1f}%)")

# Calculate what we need
print("\n" + "="*80)
print("REQUIRED ADJUSTMENTS")
print("="*80)

print("\n1. Total married filers adjustment:")
current_total = total_married
target_total = target_total_married
total_adjustment = target_total / current_total
print(f"   Current total: {current_total:,}")
print(f"   Target total: {target_total:,}")
print(f"   Need factor: {total_adjustment:.4f}")

print("\n2. MFS/Joint split adjustment:")
print(f"   Currently {len(mfs_units)} MFS + {len(joint_units)} Joint = {total_married_units} married units")
print(f"   MFS rate: {len(mfs_units)/total_married_units*100:.1f}%")
print(f"   Target MFS rate: {target_mfs_rate:.1f}%")

# To achieve target, we need to adjust the ratio
# Option 1: Adjust both calibration factors
joint_factor_needed = target['married_filing_jointly'] / (joint_units['weight'].sum() / 0.85)
mfs_factor_needed = target['married_filing_separately'] / (mfs_units['weight'].sum() / 0.27)

print(f"\n3. Individual calibration factors needed:")
print(f"   Joint: {joint_factor_needed:.4f} (current: 0.85)")
print(f"   MFS: {mfs_factor_needed:.4f} (current: 0.27)")

# Option 2: Change MFS scoring to create fewer MFS units
target_mfs_units = len(mfs_units) * (target_mfs_rate / current_mfs_rate)
print(f"\n4. Alternative - reduce MFS unit creation:")
print(f"   Current MFS units: {len(mfs_units):,}")
print(f"   Target MFS units: {target_mfs_units:.0f}")
print(f"   Need to reduce by: {len(mfs_units) - target_mfs_units:.0f} units ({(1 - target_mfs_units/len(mfs_units))*100:.1f}%)")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

print("\n**Option A: Adjust calibration factors only**")
print(f"  - Joint: 0.85 → {joint_factor_needed:.2f}")
print(f"  - MFS: 0.27 → {mfs_factor_needed:.2f}")
print("  - Pros: Simple, preserves MFS logic")
print("  - Cons: Large factor changes")

print("\n**Option B: Reduce MFS scoring + adjust factors**")
reduction_factor = target_mfs_rate / current_mfs_rate
print(f"  - Reduce MFS creation by {(1-reduction_factor)*100:.0f}% via scoring")
print(f"  - Then apply balanced calibration")
print("  - Pros: More realistic MFS behavior")
print("  - Cons: Requires tuning MFS scoring thresholds")

print("\n**Recommended: Option B**")
print(f"  1. Reduce MFS scoring to create ~{target_mfs_units:.0f} MFS units (vs current {len(mfs_units):,})")
print(f"  2. Apply balanced calibration factors")
