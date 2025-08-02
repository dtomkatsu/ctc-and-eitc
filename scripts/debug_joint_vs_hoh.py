#!/usr/bin/env python3
"""
Debug script to analyze the relationship between joint filing undercount 
and Head of Household overcount. Investigate if married couples are being
incorrectly classified as HoH instead of joint filers.
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.pums_loader import PUMSDataLoader
from tax.units.status.hoh import is_head_of_household
from tax.units.status.mfj import _are_married

def analyze_joint_vs_hoh():
    """Analyze the relationship between joint filing and HoH classification."""
    
    print("Analyzing joint vs Head of Household classification...")
    
    # Load tax units
    tax_units_file = Path("src/data/processed/tax_units_rule_based.parquet")
    if not tax_units_file.exists():
        print(f"Error: Tax units file not found: {tax_units_file}")
        return
    
    tax_units = pd.read_parquet(tax_units_file)
    print(f"Loaded {len(tax_units)} tax units")
    
    # Load PUMS data
    pums_loader = PUMSDataLoader()
    person_df, hh_df = pums_loader.load_data()
    print(f"Loaded {len(person_df)} persons from {len(hh_df)} households")
    
    # Analyze filing status distribution
    filing_status_counts = tax_units['filing_status'].value_counts()
    total_units = len(tax_units)
    
    print("\nCurrent Filing Status Distribution:")
    for status, count in filing_status_counts.items():
        pct = (count / total_units) * 100
        print(f"  {status}: {count:,} ({pct:.1f}%)")
    
    # Focus on HoH tax units
    hoh_units = tax_units[tax_units['filing_status'] == 'head_of_household'].copy()
    print(f"\nAnalyzing {len(hoh_units)} Head of Household tax units...")
    
    # Check if any HoH filers are actually married
    married_hoh_count = 0
    married_hoh_examples = []
    
    for idx, tax_unit in hoh_units.iterrows():
        primary_filer_id = tax_unit['primary_filer_id']
        
        # Extract household ID and person order from primary_filer_id
        if '_' not in primary_filer_id:
            continue
        hh_id, sporder = primary_filer_id.split('_')
        
        # Get the primary person's data
        primary_person = person_df[(person_df['SERIALNO'] == hh_id) & (person_df['SPORDER'] == int(sporder))]
        if len(primary_person) == 0:
            continue
            
        primary_person = primary_person.iloc[0]
        
        # Check if this person is married
        if primary_person['MAR'] == 1:  # 1 = married
            married_hoh_count += 1
            if len(married_hoh_examples) < 10:
                married_hoh_examples.append({
                    'filer_id': tax_unit['filer_id'],
                    'primary_filer_id': primary_filer_id,
                    'mar_status': primary_person['MAR'],
                    'relshipp': primary_person['RELSHIPP'],
                    'num_dependents': tax_unit['num_dependents'],
                    'household_id': hh_id
                })
    
    print(f"\nMarried people filing as Head of Household: {married_hoh_count:,} ({married_hoh_count/len(hoh_units)*100:.1f}% of HoH)")
    
    if married_hoh_examples:
        print("\nExamples of married HoH filers:")
        for example in married_hoh_examples:
            print(f"  Tax Unit {example['filer_id']}: MAR={example['mar_status']}, RELSHIPP={example['relshipp']}, "
                  f"Dependents={example['num_dependents']}, HH={example['household_id']}")
    
    # Now check joint filers to see if they have the characteristics we'd expect
    joint_units = tax_units[tax_units['filing_status'] == 'joint'].copy()
    print(f"\nAnalyzing {len(joint_units)} Joint filing tax units...")
    
    # Check if joint filers have dependents (which might make them eligible for HoH)
    joint_with_deps = 0
    joint_examples = []
    
    for idx, tax_unit in joint_units.iterrows():
        num_deps = tax_unit['num_dependents']
        if num_deps > 0:
            joint_with_deps += 1
            if len(joint_examples) < 5:
                joint_examples.append({
                    'filer_id': tax_unit['filer_id'],
                    'num_dependents': num_deps,
                    'household_id': tax_unit['hh_id']
                })
    
    print(f"Joint filers with dependents: {joint_with_deps:,} ({joint_with_deps/len(joint_units)*100:.1f}% of joint)")
    
    if joint_examples:
        print("\nExamples of joint filers with dependents:")
        for example in joint_examples:
            print(f"  Tax Unit {example['filer_id']}: Dependents={example['num_dependents']}, HH={example['household_id']}")
    
    # Analyze specific households with married couples
    print("\nAnalyzing households with married couples...")
    
    married_couple_households = []
    for hh_id in hh_df['SERIALNO'].unique()[:1000]:  # Sample first 1000 households
        hh_persons = person_df[person_df['SERIALNO'] == hh_id]
        
        # Check if household has married couple
        married_persons = hh_persons[hh_persons['MAR'] == 1]
        if len(married_persons) >= 2:
            # Check if they're married to each other (same household, different SPORDER)
            sporders = married_persons['SPORDER'].unique()
            if len(sporders) >= 2:
                married_couple_households.append(hh_id)
    
    print(f"Found {len(married_couple_households)} households with married couples (from sample)")
    
    # Check how these households are being processed
    if married_couple_households:
        sample_hh = married_couple_households[0]
        hh_tax_units = tax_units[tax_units['hh_id'] == sample_hh]
        
        print(f"\nExample household {sample_hh}:")
        print(f"  Tax units created: {len(hh_tax_units)}")
        for _, unit in hh_tax_units.iterrows():
            print(f"    {unit['filing_status']}: {unit['primary_filer_id']}, deps={unit['num_dependents']}")
    
    # Summary and recommendations
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print(f"Joint undercount: 26.6% vs 36.0% expected (-9.4 percentage points)")
    print(f"HoH overcount: 13.6% vs 9.6% expected (+4.0 percentage points)")
    print(f"Married people filing as HoH: {married_hoh_count:,} ({married_hoh_count/len(hoh_units)*100:.1f}% of HoH)")
    print(f"Joint filers with dependents: {joint_with_deps:,} ({joint_with_deps/len(joint_units)*100:.1f}% of joint)")
    
    if married_hoh_count > 0:
        print("\nPOTENTIAL ISSUE: Married people are being classified as Head of Household")
        print("This suggests the HoH qualification logic may not be properly checking marital status")
        print("or the joint filer identification logic may be missing some married couples")

if __name__ == "__main__":
    analyze_joint_vs_hoh()
