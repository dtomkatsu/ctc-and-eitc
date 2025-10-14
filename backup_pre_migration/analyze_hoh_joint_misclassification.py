#!/usr/bin/env python3
"""
Analyze whether some Head of Household filers should actually be classified as joint filers.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

def main():
    # Load tax units and person data
    print("Loading data...")
    tax_units = pd.read_parquet('src/data/processed/tax_units_rule_based.parquet')
    person_df = pd.read_parquet('data/processed/pums_person_processed.parquet')
    
    # Focus on HoH filers
    hoh_units = tax_units[tax_units['filing_status'] == 'head_of_household'].copy()
    print(f"Total HoH filers: {len(hoh_units):,}")
    
    # Debug: Check HoH characteristics
    print(f"\nHoH filer characteristics:")
    hoh_with_adults = 0
    hoh_married = 0
    
    for _, hoh_unit in hoh_units.iterrows():
        hh_id = hoh_unit['hh_id']
        hh_members = person_df[person_df['SERIALNO'] == hh_id]
        adults = hh_members[hh_members['AGEP'] >= 18]
        
        if len(adults) > 1:
            hoh_with_adults += 1
            
        primary_filer_id = hoh_unit['primary_filer_id']
        try:
            hoh_person = hh_members[hh_members.index.astype(str) == str(primary_filer_id)].iloc[0]
            if hoh_person.get('MAR', 0) == 1:  # Married
                hoh_married += 1
        except (IndexError, KeyError):
            continue
    
    print(f"  HoH filers in multi-adult households: {hoh_with_adults:,} ({hoh_with_adults/len(hoh_units)*100:.1f}%)")
    print(f"  HoH filers who are married: {hoh_married:,} ({hoh_married/len(hoh_units)*100:.1f}%)")
    
    print("\n" + "="*80)
    print("ANALYZING HOH FILERS FOR POTENTIAL JOINT FILING OPPORTUNITIES")
    print("="*80)
    
    potential_joint_conversions = []
    
    for _, hoh_unit in hoh_units.iterrows():
        hh_id = hoh_unit['hh_id']
        primary_filer_id = hoh_unit['primary_filer_id']
        
        # Get all people in this household
        hh_members = person_df[person_df['SERIALNO'] == hh_id].copy()
        
        if len(hh_members) < 2:
            continue  # Single-person household, can't be joint
            
        # Get the HoH filer's info
        try:
            hoh_person = hh_members[hh_members.index.astype(str) == str(primary_filer_id)].iloc[0]
        except (IndexError, KeyError):
            continue
            
        # Look for potential spouses in the household
        adults = hh_members[hh_members['AGEP'] >= 18]
        
        for _, other_adult in adults.iterrows():
            if str(other_adult.name) == str(primary_filer_id):
                continue  # Skip the HoH filer themselves
                
            # Check if this could be a spouse
            age_diff = abs(hoh_person['AGEP'] - other_adult['AGEP'])
            opposite_sex = hoh_person.get('SEX', 1) != other_adult.get('SEX', 1)
            both_married = (hoh_person.get('MAR', 0) == 1) and (other_adult.get('MAR', 0) == 1)
            
            # Traditional householder/spouse pattern
            is_householder_spouse = ((hoh_person.get('RELSHIPP') == 20 and other_adult.get('RELSHIPP') == 21) or
                                   (hoh_person.get('RELSHIPP') == 21 and other_adult.get('RELSHIPP') == 20))
            
            # Check if this adult is already in another tax unit
            other_adult_in_tax_unit = False
            for _, other_unit in tax_units.iterrows():
                if (str(other_unit['primary_filer_id']) == str(other_adult.name) or 
                    str(other_unit.get('secondary_filer_id', '')) == str(other_adult.name)):
                    other_adult_in_tax_unit = True
                    break
            
            # Potential joint filing scenarios
            if (age_diff <= 15 and opposite_sex and both_married and 
                not other_adult_in_tax_unit):
                
                potential_joint_conversions.append({
                    'hh_id': hh_id,
                    'hoh_filer_id': primary_filer_id,
                    'potential_spouse_id': other_adult.name,
                    'hoh_age': hoh_person['AGEP'],
                    'spouse_age': other_adult['AGEP'],
                    'age_diff': age_diff,
                    'hoh_income': hoh_person.get('PINCP', 0) or 0,
                    'spouse_income': other_adult.get('PINCP', 0) or 0,
                    'hoh_rel': hoh_person.get('RELSHIPP', 0),
                    'spouse_rel': other_adult.get('RELSHIPP', 0),
                    'is_householder_spouse': is_householder_spouse,
                    'hoh_dependents': hoh_unit['num_dependents'],
                    'hoh_unit_income': hoh_unit['income']
                })
                break  # Only consider first potential spouse
    
    conversions_df = pd.DataFrame(potential_joint_conversions)
    
    if len(conversions_df) == 0:
        print("No potential HoH->Joint conversions found.")
        return
        
    print(f"Found {len(conversions_df):,} potential HoH->Joint conversions")
    print(f"This represents {len(conversions_df)/len(hoh_units)*100:.1f}% of all HoH filers")
    
    # Analyze characteristics of potential conversions
    print("\nCharacteristics of potential conversions:")
    print(f"  Householder/spouse pairs: {conversions_df['is_householder_spouse'].sum():,} ({conversions_df['is_householder_spouse'].mean()*100:.1f}%)")
    print(f"  Average age difference: {conversions_df['age_diff'].mean():.1f} years")
    print(f"  Average HoH income: ${conversions_df['hoh_income'].mean():,.0f}")
    print(f"  Average spouse income: ${conversions_df['spouse_income'].mean():,.0f}")
    print(f"  Average dependents: {conversions_df['hoh_dependents'].mean():.1f}")
    
    # Prioritize conversions
    print("\n" + "="*80)
    print("PRIORITIZING CONVERSIONS")
    print("="*80)
    
    # Score potential conversions
    conversions_df['conversion_score'] = 0
    
    # Strong indicators for joint filing
    conversions_df.loc[conversions_df['is_householder_spouse'], 'conversion_score'] += 5  # Strongest indicator
    conversions_df.loc[conversions_df['age_diff'] <= 10, 'conversion_score'] += 2
    conversions_df.loc[conversions_df['age_diff'] <= 5, 'conversion_score'] += 1
    
    # Income factors
    conversions_df['total_income'] = conversions_df['hoh_income'] + conversions_df['spouse_income']
    conversions_df.loc[conversions_df['total_income'] > 50000, 'conversion_score'] += 2
    conversions_df.loc[conversions_df['spouse_income'] > 10000, 'conversion_score'] += 1  # Spouse has significant income
    
    # Both have income (dual earners more likely to file jointly)
    both_have_income = (conversions_df['hoh_income'] > 0) & (conversions_df['spouse_income'] > 0)
    conversions_df.loc[both_have_income, 'conversion_score'] += 1
    
    # Analyze score distribution
    print("Conversion score distribution:")
    score_dist = conversions_df['conversion_score'].value_counts().sort_index()
    for score, count in score_dist.items():
        pct = count / len(conversions_df) * 100
        print(f"  Score {score}: {count:,} cases ({pct:.1f}%)")
    
    # High-priority conversions (score >= 7)
    high_priority = conversions_df[conversions_df['conversion_score'] >= 7]
    print(f"\nHigh-priority conversions (score >= 7): {len(high_priority):,}")
    
    if len(high_priority) > 0:
        print("Sample high-priority cases:")
        for _, case in high_priority.head(5).iterrows():
            print(f"  HH {case['hh_id']}: HoH age {case['hoh_age']}, spouse age {case['spouse_age']}, "
                  f"incomes ${case['hoh_income']:,.0f}/${case['spouse_income']:,.0f}, "
                  f"{'householder/spouse' if case['is_householder_spouse'] else 'other'}, "
                  f"{case['hoh_dependents']} deps")
    
    # Estimate impact of conversions
    print(f"\n" + "="*80)
    print("ESTIMATED IMPACT OF CONVERSIONS")
    print("="*80)
    
    # Current filing status distribution
    current_total = len(tax_units)
    current_hoh_pct = len(hoh_units) / current_total * 100
    current_joint_pct = len(tax_units[tax_units['filing_status'] == 'married_filing_jointly']) / current_total * 100
    
    print(f"Current HoH rate: {current_hoh_pct:.1f}% ({len(hoh_units):,} units)")
    print(f"Current Joint rate: {current_joint_pct:.1f}%")
    
    # Different conversion scenarios
    scenarios = [
        ("High priority only (score >= 7)", len(high_priority)),
        ("Medium+ priority (score >= 6)", len(conversions_df[conversions_df['conversion_score'] >= 6])),
        ("All householder/spouse pairs", conversions_df['is_householder_spouse'].sum())
    ]
    
    for scenario_name, num_conversions in scenarios:
        if num_conversions > 0:
            new_hoh_count = len(hoh_units) - num_conversions
            new_joint_count = len(tax_units[tax_units['filing_status'] == 'married_filing_jointly']) + num_conversions
            
            new_hoh_pct = new_hoh_count / current_total * 100
            new_joint_pct = new_joint_count / current_total * 100
            
            hoh_change = new_hoh_pct - current_hoh_pct
            joint_change = new_joint_pct - current_joint_pct
            
            print(f"\n{scenario_name}: {num_conversions:,} conversions")
            print(f"  New HoH rate: {new_hoh_pct:.1f}% ({hoh_change:+.1f}%)")
            print(f"  New Joint rate: {new_joint_pct:.1f}% ({joint_change:+.1f}%)")
            print(f"  Distance from targets: HoH {abs(new_hoh_pct - 9.6):.1f}pp, Joint {abs(new_joint_pct - 36.0):.1f}pp")
    
    # Recommendation
    print(f"\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    # Find the scenario that gets closest to targets
    target_hoh = 9.6
    target_joint = 36.0
    
    best_scenario = None
    best_distance = float('inf')
    
    for scenario_name, num_conversions in scenarios:
        if num_conversions > 0:
            new_hoh_count = len(hoh_units) - num_conversions
            new_joint_count = len(tax_units[tax_units['filing_status'] == 'married_filing_jointly']) + num_conversions
            
            new_hoh_pct = new_hoh_count / current_total * 100
            new_joint_pct = new_joint_count / current_total * 100
            
            # Calculate combined distance from targets
            distance = abs(new_hoh_pct - target_hoh) + abs(new_joint_pct - target_joint)
            
            if distance < best_distance:
                best_distance = distance
                best_scenario = (scenario_name, num_conversions, new_hoh_pct, new_joint_pct)
    
    if best_scenario:
        scenario_name, num_conversions, new_hoh_pct, new_joint_pct = best_scenario
        print(f"Best scenario: {scenario_name}")
        print(f"  Converts {num_conversions:,} HoH filers to joint filers")
        print(f"  Results in HoH: {new_hoh_pct:.1f}% (target: 9.6%), Joint: {new_joint_pct:.1f}% (target: 36.0%)")
        print(f"  Combined distance from targets: {best_distance:.1f} percentage points")
    
    return conversions_df

if __name__ == "__main__":
    main()
