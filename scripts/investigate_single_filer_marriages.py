#!/usr/bin/env python3
"""
Investigate if any of our "single" filers are actually married people
who should be filing jointly but are being missed due to complex
household structures or edge cases.
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.pums_loader import PUMSDataLoader

def investigate_single_filer_marriages():
    """Check if single filers include married people who should file jointly."""
    
    print("Investigating single filers for missed married couples...")
    
    # Load data
    tax_units = pd.read_parquet("src/data/processed/tax_units_rule_based.parquet")
    pums_loader = PUMSDataLoader()
    person_df, hh_df = pums_loader.load_data()
    
    # Get single filers
    single_units = tax_units[tax_units['filing_status'] == 'single']
    print(f"Total single filers: {len(single_units):,}")
    
    # Check marital status of single filers
    single_filer_ids = single_units['primary_filer_id'].tolist()
    single_filer_persons = person_df[person_df.index.isin(single_filer_ids)]
    
    # Count married people filing as single
    married_single_filers = single_filer_persons[single_filer_persons['MAR'] == 1]
    print(f"Married people filing as single: {len(married_single_filers):,} ({len(married_single_filers)/len(single_units)*100:.1f}%)")
    
    if len(married_single_filers) > 0:
        print(f"\nAnalyzing married single filers...")
        
        # Sample some cases for detailed analysis
        sample_size = min(10, len(married_single_filers))
        sample_married_singles = married_single_filers.head(sample_size)
        
        for idx, person in sample_married_singles.iterrows():
            hh_id = person['SERIALNO']
            person_id = idx
            
            print(f"\n--- Case: {person_id} in household {hh_id} ---")
            print(f"Person: MAR={person['MAR']}, REL={person['RELSHIPP']}, AGE={person['AGEP']}")
            
            # Get all people in this household
            hh_persons = person_df[person_df['SERIALNO'] == hh_id]
            adults = hh_persons[hh_persons['AGEP'] >= 18]
            married_adults = adults[adults['MAR'] == 1]
            
            print(f"Household composition:")
            print(f"  Total people: {len(hh_persons)}")
            print(f"  Adults (18+): {len(adults)}")
            print(f"  Married adults: {len(married_adults)}")
            
            if len(married_adults) >= 2:
                print(f"  Other married adults in household:")
                for other_idx, other_person in married_adults.iterrows():
                    if other_idx != person_id:
                        print(f"    {other_idx}: MAR={other_person['MAR']}, REL={other_person['RELSHIPP']}, AGE={other_person['AGEP']}")
                
                # Check if any of these other married adults are also filing as single
                other_married_ids = [other_idx for other_idx in married_adults.index if other_idx != person_id]
                other_single_units = tax_units[tax_units['primary_filer_id'].isin(other_married_ids) & 
                                             (tax_units['filing_status'] == 'single')]
                
                if len(other_single_units) > 0:
                    print(f"  ⚠️  Found {len(other_single_units)} other married adults also filing as single!")
                    print(f"     This suggests a missed joint filing opportunity")
                
                # Check what tax units exist for this household
                hh_tax_units = tax_units[tax_units['hh_id'] == hh_id]
                print(f"  Tax units in household:")
                for _, unit in hh_tax_units.iterrows():
                    status = unit['filing_status']
                    primary = unit['primary_filer_id']
                    secondary = unit.get('secondary_filer_id', '')
                    deps = unit['num_dependents']
                    print(f"    {status}: {primary}" + (f" + {secondary}" if secondary else "") + f" (deps: {deps})")
    
    # Summary statistics
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    
    total_married_adults = len(person_df[(person_df['AGEP'] >= 18) & (person_df['MAR'] == 1)])
    married_in_joint_units = len(tax_units[tax_units['filing_status'] == 'joint']) * 2  # 2 people per joint unit
    married_in_mfs_units = len(tax_units[tax_units['filing_status'] == 'married_filing_separate'])
    married_filing_single = len(married_single_filers)
    
    print(f"Total married adults in PUMS: {total_married_adults:,}")
    print(f"Married adults in joint units: {married_in_joint_units:,} ({married_in_joint_units/total_married_adults*100:.1f}%)")
    print(f"Married adults in MFS units: {married_in_mfs_units:,} ({married_in_mfs_units/total_married_adults*100:.1f}%)")
    print(f"Married adults filing as single: {married_filing_single:,} ({married_filing_single/total_married_adults*100:.1f}%)")
    
    accounted_married = married_in_joint_units + married_in_mfs_units + married_filing_single
    unaccounted = total_married_adults - accounted_married
    
    print(f"Total accounted for: {accounted_married:,} ({accounted_married/total_married_adults*100:.1f}%)")
    if unaccounted != 0:
        print(f"⚠️  Unaccounted married adults: {unaccounted:,} ({unaccounted/total_married_adults*100:.1f}%)")
    
    # Potential for improvement
    if married_filing_single > 0:
        # Assume half of married single filers could potentially form joint units
        potential_new_joint_units = married_filing_single // 2
        current_joint_units = len(tax_units[tax_units['filing_status'] == 'joint'])
        total_units = len(tax_units)
        
        potential_joint_rate = (current_joint_units + potential_new_joint_units) / total_units * 100
        improvement = potential_joint_rate - (current_joint_units / total_units * 100)
        
        print(f"\nPOTENTIAL IMPROVEMENT:")
        print(f"If half of married single filers could form joint units:")
        print(f"  Potential new joint units: {potential_new_joint_units:,}")
        print(f"  New joint filing rate: {potential_joint_rate:.1f}% (improvement: +{improvement:.1f} pp)")
        print(f"  Remaining gap vs SOI: {36.0 - potential_joint_rate:.1f} pp")

if __name__ == "__main__":
    investigate_single_filer_marriages()
