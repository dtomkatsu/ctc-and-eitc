#!/usr/bin/env python3
"""
Debug script to analyze the remaining joint filing gap (29.2% vs 36.0% expected)
and identify what types of married couples are still not being identified as joint filers.
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.pums_loader import PUMSDataLoader
from tax.units.status.mfj import _are_married

def debug_remaining_joint_gap():
    """Analyze the remaining joint filing gap in detail."""
    
    print("Debugging remaining joint filing gap...")
    
    # Load data
    tax_units = pd.read_parquet("src/data/processed/tax_units_rule_based.parquet")
    pums_loader = PUMSDataLoader()
    person_df, hh_df = pums_loader.load_data()
    
    print(f"Current filing status distribution:")
    filing_counts = tax_units['filing_status'].value_counts()
    total_units = len(tax_units)
    for status, count in filing_counts.items():
        pct = count / total_units * 100
        print(f"  {status}: {count:,} ({pct:.1f}%)")
    
    joint_units = len(tax_units[tax_units['filing_status'] == 'joint'])
    expected_joint_pct = 36.0
    expected_joint_count = int(total_units * expected_joint_pct / 100)
    missing_joint_count = expected_joint_count - joint_units
    
    print(f"\nJoint filing analysis:")
    print(f"Current joint filers: {joint_units:,} (29.2%)")
    print(f"Expected joint filers: {expected_joint_count:,} (36.0%)")
    print(f"Missing joint filers: {missing_joint_count:,}")
    
    # Analyze all households to find potential joint filers
    print(f"\nAnalyzing all households for potential joint filers...")
    
    potential_joint_households = []
    married_couple_patterns = {
        'householder_spouse': 0,      # Traditional householder (20) + spouse (21)
        'both_householders': 0,       # Both marked as householder (20)
        'other_married_pairs': 0,     # Other married combinations
        'single_married_adult': 0     # Only one married adult in household
    }
    
    # Sample more households for comprehensive analysis
    sample_size = 5000
    for hh_id in hh_df['SERIALNO'].unique()[:sample_size]:
        hh_persons = person_df[person_df['SERIALNO'] == hh_id]
        
        # Find all married adults (18+)
        married_adults = hh_persons[(hh_persons['AGEP'] >= 18) & (hh_persons['MAR'] == 1)]
        
        if len(married_adults) == 0:
            continue
        elif len(married_adults) == 1:
            married_couple_patterns['single_married_adult'] += 1
        elif len(married_adults) == 2:
            adult1, adult2 = married_adults.iloc[0], married_adults.iloc[1]
            rel1, rel2 = adult1['RELSHIPP'], adult2['RELSHIPP']
            
            # Classify the relationship pattern
            if (rel1 == 20 and rel2 == 21) or (rel1 == 21 and rel2 == 20):
                married_couple_patterns['householder_spouse'] += 1
                pattern = 'householder_spouse'
            elif rel1 == 20 and rel2 == 20:
                married_couple_patterns['both_householders'] += 1
                pattern = 'both_householders'
            else:
                married_couple_patterns['other_married_pairs'] += 1
                pattern = 'other_married_pairs'
            
            # Check if they would be identified as married by current logic
            are_married = _are_married(adult1, adult2)
            
            potential_joint_households.append({
                'hh_id': hh_id,
                'pattern': pattern,
                'are_married_result': are_married,
                'adult1_rel': rel1,
                'adult2_rel': rel2,
                'adult1_age': adult1['AGEP'],
                'adult2_age': adult2['AGEP'],
                'adult1_income': adult1.get('PINCP', 0) or 0,
                'adult2_income': adult2.get('PINCP', 0) or 0
            })
        else:
            # More than 2 married adults - complex household
            married_couple_patterns['other_married_pairs'] += 1
    
    print(f"\nMarried couple patterns in {sample_size} households:")
    for pattern, count in married_couple_patterns.items():
        print(f"  {pattern}: {count:,}")
    
    # Analyze which patterns are being missed by joint filer identification
    print(f"\nAnalyzing joint filer identification success by pattern:")
    
    pattern_analysis = {}
    for pattern in ['householder_spouse', 'both_householders', 'other_married_pairs']:
        pattern_couples = [h for h in potential_joint_households if h['pattern'] == pattern]
        if len(pattern_couples) == 0:
            continue
            
        identified_count = sum(1 for h in pattern_couples if h['are_married_result'])
        missed_count = len(pattern_couples) - identified_count
        
        pattern_analysis[pattern] = {
            'total': len(pattern_couples),
            'identified': identified_count,
            'missed': missed_count,
            'success_rate': identified_count / len(pattern_couples) * 100 if len(pattern_couples) > 0 else 0
        }
        
        print(f"  {pattern}:")
        print(f"    Total couples: {len(pattern_couples):,}")
        print(f"    Identified as married: {identified_count:,} ({identified_count/len(pattern_couples)*100:.1f}%)")
        print(f"    Missed: {missed_count:,} ({missed_count/len(pattern_couples)*100:.1f}%)")
    
    # Show examples of missed couples by pattern
    print(f"\nExamples of missed married couples:")
    
    for pattern in ['householder_spouse', 'both_householders', 'other_married_pairs']:
        missed_couples = [h for h in potential_joint_households 
                         if h['pattern'] == pattern and not h['are_married_result']]
        
        if len(missed_couples) > 0:
            print(f"\n{pattern.upper()} - Missed couples:")
            for i, couple in enumerate(missed_couples[:3]):  # Show first 3 examples
                print(f"  Example {i+1}: HH {couple['hh_id']}")
                print(f"    Adult 1: REL={couple['adult1_rel']}, AGE={couple['adult1_age']}, INCOME=${couple['adult1_income']:,.0f}")
                print(f"    Adult 2: REL={couple['adult2_rel']}, AGE={couple['adult2_age']}, INCOME=${couple['adult2_income']:,.0f}")
    
    # Check what filing status these missed couples actually got
    print(f"\nChecking actual filing status of missed couples:")
    
    missed_couples_sample = [h for h in potential_joint_households if not h['are_married_result']][:10]
    
    for couple in missed_couples_sample:
        hh_id = couple['hh_id']
        hh_tax_units = tax_units[tax_units['hh_id'] == hh_id]
        
        print(f"\nHousehold {hh_id} ({couple['pattern']}):")
        print(f"  Tax units created: {len(hh_tax_units)}")
        for _, unit in hh_tax_units.iterrows():
            print(f"    {unit['filing_status']}: {unit['primary_filer_id']}, deps={unit['num_dependents']}")
            if unit['secondary_filer_id']:
                print(f"      Secondary: {unit['secondary_filer_id']}")
    
    # Summary and recommendations
    print(f"\n{'='*80}")
    print(f"ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Missing joint filers: {missing_joint_count:,}")
    print(f"Most problematic patterns:")
    
    for pattern, analysis in pattern_analysis.items():
        if analysis['missed'] > 0:
            print(f"  {pattern}: {analysis['missed']:,} missed couples ({100-analysis['success_rate']:.1f}% miss rate)")
    
    print(f"\nRecommendations:")
    print(f"1. Investigate _are_married() function for edge cases")
    print(f"2. Check if relationship codes are being handled correctly")
    print(f"3. Consider if some married couples are being classified as MFS when they shouldn't be")
    print(f"4. Verify that all married couple patterns are being detected")

if __name__ == "__main__":
    debug_remaining_joint_gap()
