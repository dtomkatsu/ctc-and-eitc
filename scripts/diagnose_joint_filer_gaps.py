#!/usr/bin/env python3
"""
Diagnose why we're missing joint filers - analyze married couples in PUMS data
and understand why they're not being identified as joint filers.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

def main():
    # Load person data
    print("Loading data...")
    person_df = pd.read_parquet('data/processed/pums_person_processed.parquet')
    
    print(f"Total people: {len(person_df):,}")
    
    # Focus on married adults
    married_adults = person_df[(person_df['MAR'] == 1) & (person_df['AGEP'] >= 18)].copy()
    print(f"Married adults: {len(married_adults):,}")
    
    if len(married_adults) == 0:
        print("No married adults found in PUMS data!")
        return
    
    # Analyze relationship codes for married adults
    print(f"\nMarried adults by relationship code:")
    rel_counts = married_adults['RELSHIPP'].value_counts().sort_index()
    for rel_code, count in rel_counts.items():
        pct = count / len(married_adults) * 100
        print(f"  RELSHIPP {rel_code}: {count:,} ({pct:.1f}%)")
    
    # Look for householder/spouse pairs
    print(f"\n" + "="*80)
    print("ANALYZING HOUSEHOLDER/SPOUSE PAIRS")
    print("="*80)
    
    householder_spouse_pairs = 0
    potential_joint_filers = []
    
    # Group by household
    for hh_id, hh_group in married_adults.groupby('SERIALNO'):
        hh_married = hh_group.copy()
        
        if len(hh_married) < 2:
            continue  # Need at least 2 married adults
            
        # Look for householder (20) and spouse (21) pairs
        householders = hh_married[hh_married['RELSHIPP'] == 20]
        spouses = hh_married[hh_married['RELSHIPP'] == 21]
        
        if len(householders) >= 1 and len(spouses) >= 1:
            # Found potential householder/spouse pairs
            for _, householder in householders.iterrows():
                for _, spouse in spouses.iterrows():
                    householder_spouse_pairs += 1
                    
                    age_diff = abs(householder['AGEP'] - spouse['AGEP'])
                    hh_income = _calculate_income(householder)
                    sp_income = _calculate_income(spouse)
                    
                    # Generate proper person IDs
                    householder_person_id = f"{householder['SERIALNO']}_{householder['SPORDER']}"
                    spouse_person_id = f"{spouse['SERIALNO']}_{spouse['SPORDER']}"
                
                    potential_joint_filers.append({
                        'hh_id': hh_id,
                        'householder_id': householder_person_id,
                        'spouse_id': spouse_person_id,
                        'householder_age': householder['AGEP'],
                        'spouse_age': spouse['AGEP'],
                        'age_diff': age_diff,
                        'householder_income': hh_income,
                        'spouse_income': sp_income,
                        'total_income': hh_income + sp_income,
                        'householder_sex': householder.get('SEX', 0),
                        'spouse_sex': spouse.get('SEX', 0),
                        'opposite_sex': householder.get('SEX', 0) != spouse.get('SEX', 0)
                    })
    
    print(f"Found {householder_spouse_pairs:,} potential householder/spouse pairs")
    
    if len(potential_joint_filers) == 0:
        print("No householder/spouse pairs found!")
        
        # Analyze what relationship codes married adults have
        print(f"\nDebugging: All married adults by household size and relationship:")
        for hh_id, hh_group in married_adults.head(20).groupby('SERIALNO'):
            hh_married = hh_group.copy()
            print(f"  HH {hh_id}: {len(hh_married)} married adults")
            for _, person in hh_married.iterrows():
                print(f"    Person {person.name}: RELSHIPP={person['RELSHIPP']}, AGE={person['AGEP']}, SEX={person.get('SEX')}")
        return
    
    # Convert to DataFrame for analysis
    joint_df = pd.DataFrame(potential_joint_filers)
    
    print(f"\nAnalysis of potential joint filers:")
    print(f"  Average age difference: {joint_df['age_diff'].mean():.1f} years")
    print(f"  Average householder income: ${joint_df['householder_income'].mean():,.0f}")
    print(f"  Average spouse income: ${joint_df['spouse_income'].mean():,.0f}")
    print(f"  Average total income: ${joint_df['total_income'].mean():,.0f}")
    print(f"  Opposite sex couples: {joint_df['opposite_sex'].sum():,} ({joint_df['opposite_sex'].mean()*100:.1f}%)")
    
    # Now check what happens to these potential joint filers in the actual tax units
    print(f"\n" + "="*80)
    print("CHECKING ACTUAL TAX UNIT ASSIGNMENTS")
    print("="*80)
    
    # Load actual tax units
    tax_units = pd.read_parquet('src/data/processed/tax_units_rule_based.parquet')
    
    joint_tax_units = tax_units[tax_units['filing_status'] == 'married_filing_jointly']
    single_tax_units = tax_units[tax_units['filing_status'] == 'single']
    mfs_tax_units = tax_units[tax_units['filing_status'] == 'married_filing_separately']
    
    print(f"Actual tax units:")
    print(f"  Joint filers: {len(joint_tax_units):,}")
    print(f"  Single filers: {len(single_tax_units):,}")
    print(f"  MFS filers: {len(mfs_tax_units):,}")
    
    # Check how many of our potential joint filers actually became joint filers
    actual_joint_pairs = 0
    became_single = 0
    became_mfs = 0
    not_found = 0
    
    for _, potential in joint_df.iterrows():
        hh_id = potential['hh_id']
        householder_id = str(potential['householder_id'])
        spouse_id = str(potential['spouse_id'])
        
        # Check if they filed jointly (handle NaN secondary_filer_id)
        joint_match = joint_tax_units[
            ((joint_tax_units['primary_filer_id'] == householder_id) & 
             (joint_tax_units['secondary_filer_id'].fillna('') == spouse_id)) |
            ((joint_tax_units['primary_filer_id'] == spouse_id) & 
             (joint_tax_units['secondary_filer_id'].fillna('') == householder_id))
        ]
        
        if len(joint_match) > 0:
            actual_joint_pairs += 1
            continue
            
        # Check if they filed MFS
        mfs_match = mfs_tax_units[
            (mfs_tax_units['hh_id'] == hh_id) &
            ((mfs_tax_units['primary_filer_id'] == householder_id) |
             (mfs_tax_units['primary_filer_id'] == spouse_id))
        ]
        
        if len(mfs_match) >= 2:  # Both should be in MFS
            became_mfs += 1
            continue
            
        # Check if they became single filers
        single_householder = single_tax_units[single_tax_units['primary_filer_id'] == householder_id]
        single_spouse = single_tax_units[single_tax_units['primary_filer_id'] == spouse_id]
        
        if len(single_householder) > 0 and len(single_spouse) > 0:
            became_single += 1
            continue
            
        not_found += 1
    
    print(f"\nFate of {len(joint_df):,} potential joint filer pairs:")
    print(f"  Actually filed jointly: {actual_joint_pairs:,} ({actual_joint_pairs/len(joint_df)*100:.1f}%)")
    print(f"  Filed as MFS: {became_mfs:,} ({became_mfs/len(joint_df)*100:.1f}%)")
    print(f"  Filed as singles: {became_single:,} ({became_single/len(joint_df)*100:.1f}%)")
    print(f"  Not found in tax units: {not_found:,} ({not_found/len(joint_df)*100:.1f}%)")
    
    # Calculate the gap
    joint_capture_rate = actual_joint_pairs / len(joint_df) * 100
    print(f"\nJoint filer capture rate: {joint_capture_rate:.1f}%")
    print(f"Missing joint opportunities: {len(joint_df) - actual_joint_pairs:,}")
    
    # Show examples of missed opportunities
    if became_single > 0 or became_mfs > 0:
        print(f"\n" + "="*80)
        print("EXAMPLES OF MISSED JOINT FILING OPPORTUNITIES")
        print("="*80)
        
        examples_shown = 0
        for _, potential in joint_df.iterrows():
            if examples_shown >= 10:
                break
                
            hh_id = potential['hh_id']
            householder_id = str(potential['householder_id'])
            spouse_id = str(potential['spouse_id'])
            
            # Check if they filed jointly (handle NaN secondary_filer_id)
            joint_match = joint_tax_units[
                ((joint_tax_units['primary_filer_id'] == householder_id) & 
                 (joint_tax_units['secondary_filer_id'].fillna('') == spouse_id)) |
                ((joint_tax_units['primary_filer_id'] == spouse_id) & 
                 (joint_tax_units['secondary_filer_id'].fillna('') == householder_id))
            ]
            
            if len(joint_match) > 0:
                continue  # They did file jointly, skip
                
            # This is a missed opportunity
            print(f"  HH {hh_id}: Householder age {potential['householder_age']}, spouse age {potential['spouse_age']}")
            print(f"    Incomes: ${potential['householder_income']:,.0f} / ${potential['spouse_income']:,.0f}")
            
            # Check what actually happened
            single_householder = single_tax_units[single_tax_units['primary_filer_id'] == householder_id]
            single_spouse = single_tax_units[single_tax_units['primary_filer_id'] == spouse_id]
            mfs_match = mfs_tax_units[
                (mfs_tax_units['hh_id'] == hh_id) &
                ((mfs_tax_units['primary_filer_id'] == householder_id) |
                 (mfs_tax_units['primary_filer_id'] == spouse_id))
            ]
            
            if len(single_householder) > 0 and len(single_spouse) > 0:
                print(f"    → Both filed as single")
            elif len(mfs_match) >= 2:
                print(f"    → Filed as MFS")
            else:
                print(f"    → Status unknown")
                
            examples_shown += 1
    
    return joint_df

def _calculate_income(person: pd.Series) -> float:
    """Calculate total income for a person."""
    # Use PINCP (total person income) if available
    pincp = person.get('PINCP', 0)
    if pincp and pincp > 0:
        # Apply adjustment factor if it's a reasonable value
        adjinc = person.get('ADJINC', 1.0)
        if adjinc and adjinc > 0 and adjinc < 2.0:
            return float(pincp) * float(adjinc)
        return float(pincp)
    
    # Fallback to WAGP
    wagp = person.get('WAGP', 0)
    if wagp and wagp > 0:
        return float(wagp)
    
    return 0.0

if __name__ == "__main__":
    main()
