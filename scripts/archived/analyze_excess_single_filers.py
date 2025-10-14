#!/usr/bin/env python3
"""
Analyze excess single filers to understand why we're over-counting single filers
and under-counting joint filers.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

def main():
    # Load tax units and person data
    print("Loading data...")
    tax_units = pd.read_parquet('src/data/processed/tax_units_rule_based.parquet')
    person_df = pd.read_parquet('data/processed/pums_person_processed.parquet')
    
    # Focus on single filers
    single_units = tax_units[tax_units['filing_status'] == 'single'].copy()
    joint_units = tax_units[tax_units['filing_status'] == 'married_filing_jointly'].copy()
    
    print(f"Total single filers: {len(single_units):,}")
    print(f"Total joint filers: {len(joint_units):,}")
    print(f"Single/Joint ratio: {len(single_units)/len(joint_units):.2f}")
    
    # Expected from SOI benchmarks
    total_expected = 855541  # From SOI data
    expected_single = int(total_expected * 0.51)  # 51% single
    expected_joint = int(total_expected * 0.36)   # 36% joint
    
    print(f"\nExpected single filers: {expected_single:,}")
    print(f"Expected joint filers: {expected_joint:,}")
    print(f"Expected Single/Joint ratio: {expected_single/expected_joint:.2f}")
    
    print("\n" + "="*80)
    print("ANALYZING SINGLE FILERS FOR MISSED JOINT OPPORTUNITIES")
    print("="*80)
    
    # Look for married single filers who should potentially be joint filers
    married_single_filers = []
    
    for _, single_unit in single_units.iterrows():
        hh_id = single_unit['hh_id']
        primary_filer_id = single_unit['primary_filer_id']
        
        # Get all people in this household
        hh_members = person_df[person_df['SERIALNO'] == hh_id].copy()
        
        if len(hh_members) < 2:
            continue  # Single-person household
            
        # Get the single filer's info
        try:
            single_person = hh_members[hh_members.index.astype(str) == str(primary_filer_id)].iloc[0]
        except (IndexError, KeyError):
            continue
            
        # Check if this person is married
        if single_person.get('MAR', 0) != 1:
            continue  # Not married
            
        # Look for potential spouse in the household
        adults = hh_members[hh_members['AGEP'] >= 18]
        
        for _, other_adult in adults.iterrows():
            if str(other_adult.name) == str(primary_filer_id):
                continue  # Skip the single filer themselves
                
            # Check if this could be their spouse
            if other_adult.get('MAR', 0) != 1:
                continue  # Other adult not married
                
            # Check relationship codes
            single_rel = single_person.get('RELSHIPP', 0)
            other_rel = other_adult.get('RELSHIPP', 0)
            
            # Traditional householder/spouse pattern
            is_householder_spouse = ((single_rel == 20 and other_rel == 21) or
                                   (single_rel == 21 and other_rel == 20))
            
            # Check if the other adult is already in a tax unit
            other_adult_in_tax_unit = False
            for _, other_unit in tax_units.iterrows():
                if (str(other_unit['primary_filer_id']) == str(other_adult.name) or 
                    str(other_unit.get('secondary_filer_id', '')) == str(other_adult.name)):
                    other_adult_in_tax_unit = True
                    break
            
            if not other_adult_in_tax_unit and is_householder_spouse:
                # This looks like a missed joint filing opportunity
                age_diff = abs(single_person['AGEP'] - other_adult['AGEP'])
                single_income = _calculate_income(single_person)
                spouse_income = _calculate_income(other_adult)
                
                married_single_filers.append({
                    'hh_id': hh_id,
                    'single_filer_id': primary_filer_id,
                    'potential_spouse_id': other_adult.name,
                    'single_age': single_person['AGEP'],
                    'spouse_age': other_adult['AGEP'],
                    'age_diff': age_diff,
                    'single_income': single_income,
                    'spouse_income': spouse_income,
                    'single_rel': single_rel,
                    'spouse_rel': other_rel,
                    'both_married': True,
                    'is_householder_spouse': is_householder_spouse
                })
                break  # Found a potential spouse, no need to check others
    
    print(f"Found {len(married_single_filers):,} married single filers with potential spouses")
    
    if len(married_single_filers) == 0:
        print("No obvious missed joint filing opportunities found among single filers.")
        print("\nThis suggests the issue may be in:")
        print("1. Joint filer identification logic not finding married couples")
        print("2. MFS logic being too aggressive and splitting legitimate joint filers")
        print("3. Spouse identification criteria being too restrictive")
        return
    
    # Convert to DataFrame for analysis
    conversions_df = pd.DataFrame(married_single_filers)
    
    print(f"\nAnalysis of missed joint filing opportunities:")
    print(f"  Average age difference: {conversions_df['age_diff'].mean():.1f} years")
    print(f"  Average single filer income: ${conversions_df['single_income'].mean():,.0f}")
    print(f"  Average potential spouse income: ${conversions_df['spouse_income'].mean():,.0f}")
    print(f"  All are householder/spouse pairs: {conversions_df['is_householder_spouse'].all()}")
    
    # Show sample cases
    print(f"\nSample missed joint filing opportunities:")
    for i, case in conversions_df.head(10).iterrows():
        print(f"  HH {case['hh_id']}: Single filer age {case['single_age']}, spouse age {case['spouse_age']}, "
              f"incomes ${case['single_income']:,.0f}/${case['spouse_income']:,.0f}")
    
    # Estimate impact
    print(f"\n" + "="*80)
    print("ESTIMATED IMPACT OF FIXING MISSED JOINT OPPORTUNITIES")
    print("="*80)
    
    current_total = len(tax_units)
    current_single_count = len(single_units)
    current_joint_count = len(joint_units)
    
    # If we convert these to joint filers
    new_single_count = current_single_count - len(conversions_df) * 2  # Remove 2 single filers
    new_joint_count = current_joint_count + len(conversions_df)        # Add 1 joint filer
    new_total = current_total - len(conversions_df)                    # Net reduction in tax units
    
    current_single_pct = current_single_count / current_total * 100
    current_joint_pct = current_joint_count / current_total * 100
    
    new_single_pct = new_single_count / new_total * 100
    new_joint_pct = new_joint_count / new_total * 100
    
    print(f"Current distribution:")
    print(f"  Single: {current_single_pct:.1f}% ({current_single_count:,} filers)")
    print(f"  Joint: {current_joint_pct:.1f}% ({current_joint_count:,} filers)")
    
    print(f"\nAfter converting {len(conversions_df):,} missed opportunities:")
    print(f"  Single: {new_single_pct:.1f}% ({new_single_count:,} filers) - Change: {new_single_pct - current_single_pct:+.1f}pp")
    print(f"  Joint: {new_joint_pct:.1f}% ({new_joint_count:,} filers) - Change: {new_joint_pct - current_joint_pct:+.1f}pp")
    
    # Distance from targets
    target_single = 51.0
    target_joint = 36.0
    
    current_single_gap = current_single_pct - target_single
    current_joint_gap = current_joint_pct - target_joint
    
    new_single_gap = new_single_pct - target_single
    new_joint_gap = new_joint_pct - target_joint
    
    print(f"\nDistance from SOI targets:")
    print(f"  Single gap: {current_single_gap:+.1f}pp → {new_single_gap:+.1f}pp (improvement: {current_single_gap - new_single_gap:+.1f}pp)")
    print(f"  Joint gap: {current_joint_gap:+.1f}pp → {new_joint_gap:+.1f}pp (improvement: {new_joint_gap - current_joint_gap:+.1f}pp)")
    
    return conversions_df

def _calculate_income(person: pd.Series) -> float:
    """Calculate total income for a person."""
    # First check for WAGP (wages) as it's commonly used in test data
    wagp = person.get('WAGP', 0)
    if wagp and wagp > 0:
        return float(wagp)
    
    # Then check PINCP (total person income) if available
    pincp = person.get('PINCP', 0)
    if pincp and pincp > 0:
        # Apply adjustment factor if it's a reasonable value
        adjinc = person.get('ADJINC', 1.0)
        if adjinc and adjinc > 0 and adjinc < 2.0:  # Only apply if it's a reasonable adjustment
            return float(pincp) * float(adjinc)
        return float(pincp)
    
    # Fallback to summing individual components
    income = 0.0
    for col in ['WAGP', 'SEMP', 'INTP', 'RETP', 'SSP', 'SSIP', 'PAP', 'OIP']:
        value = person.get(col, 0)
        if value:
            income += float(value)
    
    # Only apply adjustment factor if it's a reasonable value
    adjinc = person.get('ADJINC', 1.0)
    if adjinc and adjinc > 0 and adjinc < 2.0:  # Only apply if it's a reasonable adjustment
        income *= float(adjinc)
    
    return income

if __name__ == "__main__":
    main()
