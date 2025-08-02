#!/usr/bin/env python3
"""
Debug script to analyze the Head of Household overcount by examining
HoH filers who may not actually meet the qualification criteria.
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.pums_loader import PUMSDataLoader
from tax.units.status.hoh import is_head_of_household, _is_unmarried, _has_qualifying_person, _paid_half_home_cost

def debug_hoh_overcount():
    """Analyze HoH filers to find those who may not meet qualification criteria."""
    
    print("Debugging Head of Household overcount...")
    
    # Load data
    tax_units = pd.read_parquet("src/data/processed/tax_units_rule_based.parquet")
    pums_loader = PUMSDataLoader()
    person_df, hh_df = pums_loader.load_data()
    
    hoh_units = tax_units[tax_units['filing_status'] == 'head_of_household']
    print(f"Total HoH tax units: {len(hoh_units)}")
    print(f"Current HoH rate: {len(hoh_units)/len(tax_units)*100:.1f}% vs 9.6% expected")
    print(f"Overcount: {len(hoh_units)/len(tax_units)*100 - 9.6:.1f} percentage points")
    
    # Re-validate HoH filers against qualification criteria
    invalid_hoh_count = 0
    invalid_examples = []
    qualification_failures = {
        'unmarried': 0,
        'qualifying_person': 0,
        'home_cost': 0,
        'multiple_failures': 0
    }
    
    print("\nRe-validating HoH filers against qualification criteria...")
    
    for idx, tax_unit in hoh_units.iterrows():
        primary_filer_id = tax_unit['primary_filer_id']
        if '_' not in primary_filer_id:
            continue
        hh_id, sporder = primary_filer_id.split('_')
        
        # Get the primary person's data
        primary_person = person_df[(person_df['SERIALNO'] == hh_id) & (person_df['SPORDER'] == int(sporder))]
        if len(primary_person) == 0:
            continue
        primary_person = primary_person.iloc[0]
        
        # Get household data
        hh_persons = person_df[person_df['SERIALNO'] == hh_id]
        
        # Re-check HoH qualification
        should_be_hoh = is_head_of_household(primary_person, hh_persons)
        
        if not should_be_hoh:
            invalid_hoh_count += 1
            
            # Determine which criteria failed
            failures = []
            if not _is_unmarried(primary_person, hh_persons):
                failures.append('unmarried')
                qualification_failures['unmarried'] += 1
            if not _has_qualifying_person(primary_person, hh_persons):
                failures.append('qualifying_person')
                qualification_failures['qualifying_person'] += 1
            if not _paid_half_home_cost(primary_person, hh_persons):
                failures.append('home_cost')
                qualification_failures['home_cost'] += 1
            
            if len(failures) > 1:
                qualification_failures['multiple_failures'] += 1
            
            if len(invalid_examples) < 10:
                invalid_examples.append({
                    'filer_id': tax_unit['filer_id'],
                    'hh_id': hh_id,
                    'mar_status': primary_person['MAR'],
                    'relshipp': primary_person['RELSHIPP'],
                    'age': primary_person['AGEP'],
                    'num_dependents': tax_unit['num_dependents'],
                    'failures': failures,
                    'income': primary_person.get('PINCP', 0) or 0
                })
    
    print(f"\nValidation Results:")
    print(f"Invalid HoH filers: {invalid_hoh_count:,} ({invalid_hoh_count/len(hoh_units)*100:.1f}% of HoH)")
    print(f"Valid HoH filers: {len(hoh_units) - invalid_hoh_count:,}")
    
    if invalid_hoh_count > 0:
        print(f"\nQualification failure breakdown:")
        for criteria, count in qualification_failures.items():
            if count > 0:
                print(f"  {criteria}: {count:,} failures")
        
        print(f"\nExamples of invalid HoH filers:")
        for example in invalid_examples:
            failures_str = ', '.join(example['failures'])
            print(f"  {example['filer_id']}: MAR={example['mar_status']}, AGE={example['age']}, "
                  f"DEPS={example['num_dependents']}, INCOME=${example['income']:,.0f}")
            print(f"    Failed: {failures_str}")
    
    # Analyze HoH filers with no dependents (these are suspicious)
    hoh_no_deps = hoh_units[hoh_units['num_dependents'] == 0]
    print(f"\nHoH filers with 0 dependents: {len(hoh_no_deps):,} ({len(hoh_no_deps)/len(hoh_units)*100:.1f}% of HoH)")
    
    if len(hoh_no_deps) > 0:
        print("Examples of HoH filers with no dependents:")
        for idx, tax_unit in hoh_no_deps.head(5).iterrows():
            primary_filer_id = tax_unit['primary_filer_id']
            hh_id, sporder = primary_filer_id.split('_')
            primary_person = person_df[(person_df['SERIALNO'] == hh_id) & (person_df['SPORDER'] == int(sporder))]
            if len(primary_person) > 0:
                person = primary_person.iloc[0]
                print(f"  {tax_unit['filer_id']}: MAR={person['MAR']}, AGE={person['AGEP']}, "
                      f"RELSHIPP={person['RELSHIPP']}, INCOME=${person.get('PINCP', 0) or 0:,.0f}")
    
    # Calculate potential impact of fixing invalid HoH filers
    if invalid_hoh_count > 0:
        corrected_hoh_rate = (len(hoh_units) - invalid_hoh_count) / len(tax_units) * 100
        print(f"\n{'='*80}")
        print(f"IMPACT ANALYSIS")
        print(f"{'='*80}")
        print(f"Current HoH rate: {len(hoh_units)/len(tax_units)*100:.1f}%")
        print(f"If invalid HoH filers were reclassified as single:")
        print(f"  Corrected HoH rate: {corrected_hoh_rate:.1f}%")
        print(f"  Gap to SOI benchmark (9.6%): {corrected_hoh_rate - 9.6:.1f} percentage points")
        
        if corrected_hoh_rate < 9.6:
            print("  *** This would under-correct the HoH rate ***")
        elif abs(corrected_hoh_rate - 9.6) < 1.0:
            print("  *** This would bring HoH rate very close to benchmark! ***")
        else:
            print("  *** HoH rate would still be over-counted ***")

if __name__ == "__main__":
    debug_hoh_overcount()
