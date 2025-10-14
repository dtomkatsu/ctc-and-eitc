#!/usr/bin/env python3
"""
Step 4: Analyze what adjustments are needed to fine-tune HoH criteria
to reach the target rate of 9.6% (currently at 4.6%).
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.pums_loader import PUMSDataLoader

def analyze_hoh_fine_tuning():
    """Analyze what changes are needed to reach 9.6% HoH rate."""
    
    print("Step 4: Analyzing HoH fine-tuning requirements...")
    
    # Load data
    tax_units = pd.read_parquet("src/data/processed/tax_units_rule_based.parquet")
    pums_loader = PUMSDataLoader()
    person_df, hh_df = pums_loader.load_data()
    
    # Filter out zero-weight households
    hh_weights = hh_df[hh_df['WGTP'] > 0]
    person_df = person_df[person_df['SERIALNO'].isin(hh_weights['SERIALNO'])]
    
    # Current status
    current_hoh = len(tax_units[tax_units['filing_status'] == 'head_of_household'])
    current_single = len(tax_units[tax_units['filing_status'] == 'single'])
    total_units = len(tax_units)
    
    # Target calculations
    target_hoh_rate = 0.096  # 9.6%
    current_hoh_rate = current_hoh / total_units
    target_hoh_count = int(total_units * target_hoh_rate)
    additional_hoh_needed = target_hoh_count - current_hoh
    
    print(f"\nCurrent Status:")
    print(f"  Current HoH units: {current_hoh:,} ({current_hoh_rate*100:.1f}%)")
    print(f"  Target HoH units: {target_hoh_count:,} ({target_hoh_rate*100:.1f}%)")
    print(f"  Additional HoH needed: {additional_hoh_needed:,}")
    print(f"  Current single units: {current_single:,}")
    
    # Analyze current single filers to see who could become HoH
    single_units = tax_units[tax_units['filing_status'] == 'single']
    
    print(f"\nAnalyzing current single filers for HoH potential...")
    
    # Create person ID mapping
    person_df['person_id'] = person_df['SERIALNO'] + '_' + person_df['SPORDER'].astype(str)
    person_lookup = person_df.set_index('person_id')
    
    # Analyze single filers with dependents (these are prime HoH candidates)
    single_with_deps = single_units[single_units['num_dependents'] > 0]
    single_no_deps = single_units[single_units['num_dependents'] == 0]
    
    print(f"  Single filers with dependents: {len(single_with_deps):,}")
    print(f"  Single filers without dependents: {len(single_no_deps):,}")
    
    if len(single_with_deps) > 0:
        print(f"\nSingle filers with dependents (prime HoH candidates):")
        dep_dist = single_with_deps['num_dependents'].value_counts().sort_index()
        for num_deps, count in dep_dist.items():
            pct = count / len(single_with_deps) * 100
            print(f"  {num_deps} dependents: {count:,} ({pct:.1f}%)")
        
        # If we converted some single filers with dependents to HoH, would that be enough?
        if len(single_with_deps) >= additional_hoh_needed:
            conversion_rate = additional_hoh_needed / len(single_with_deps) * 100
            print(f"\n  ✅ Converting {conversion_rate:.1f}% of single filers with dependents would reach target")
        else:
            print(f"\n  ⚠️  Even converting ALL single filers with dependents ({len(single_with_deps):,}) ")
            print(f"      would only provide {len(single_with_deps):,} of {additional_hoh_needed:,} needed")
    
    # Analyze characteristics of current single filers with dependents
    if len(single_with_deps) > 0:
        print(f"\nCharacteristics of single filers with dependents:")
        
        # Sample analysis
        sample_size = min(20, len(single_with_deps))
        sample_single_with_deps = single_with_deps.head(sample_size)
        
        householder_count = 0
        young_count = 0
        unmarried_count = 0
        
        for _, unit in sample_single_with_deps.iterrows():
            filer_id = unit['primary_filer_id']
            if filer_id in person_lookup.index:
                person = person_lookup.loc[filer_id]
                age = person.get('AGEP', 0)
                rel = person.get('RELSHIPP', 0)
                mar = person.get('MAR', 0)
                
                if rel == 20:  # Householder
                    householder_count += 1
                if age < 25:
                    young_count += 1
                if mar in [2, 3, 4, 5]:  # Unmarried
                    unmarried_count += 1
        
        print(f"  Sample of {sample_size} single filers with dependents:")
        print(f"    Householders (REL=20): {householder_count}/{sample_size} ({householder_count/sample_size*100:.1f}%)")
        print(f"    Young (<25): {young_count}/{sample_size} ({young_count/sample_size*100:.1f}%)")
        print(f"    Unmarried: {unmarried_count}/{sample_size} ({unmarried_count/sample_size*100:.1f}%)")
    
    # Recommendations for fine-tuning
    print(f"\n{'='*80}")
    print(f"FINE-TUNING RECOMMENDATIONS")
    print(f"{'='*80}")
    
    print(f"Current HoH criteria are too restrictive. To reach 9.6% target:")
    
    print(f"\n1. RELAX DEPENDENT REQUIREMENT:")
    print(f"   - Current: Requires actual dependents in tax unit")
    print(f"   - Suggested: Allow HoH with qualifying persons even if not claimed as dependents")
    print(f"   - This would help capture single parents who can't claim children due to custody/support rules")
    
    print(f"\n2. RELAX HOUSEHOLDER PREFERENCE:")
    print(f"   - Current: Strongly prefers householders (REL=20)")
    print(f"   - Suggested: Allow non-householders with dependents to qualify more easily")
    print(f"   - Many legitimate HoH filers may not be the primary householder")
    
    print(f"\n3. RELAX AGE RESTRICTIONS:")
    print(f"   - Current: Stricter criteria for filers under 25")
    print(f"   - Suggested: Allow young parents with dependents to qualify")
    print(f"   - Young single parents are legitimate HoH filers")
    
    print(f"\n4. ADJUST QUALIFYING PERSON LOGIC:")
    print(f"   - Current: Very strict (only children/grandchildren under 19 or disabled)")
    print(f"   - Suggested: Allow some qualifying relatives and students under 24")
    print(f"   - This aligns better with actual tax rules")
    
    # Calculate optimal adjustment
    if len(single_with_deps) > 0:
        optimal_conversion_rate = min(100, additional_hoh_needed / len(single_with_deps) * 100)
        print(f"\n5. OPTIMAL CONVERSION TARGET:")
        print(f"   - Convert ~{optimal_conversion_rate:.1f}% of single filers with dependents to HoH")
        print(f"   - This would convert {int(len(single_with_deps) * optimal_conversion_rate / 100):,} units")
        print(f"   - Focus on: householders, parents with young children, unmarried filers")
    
    return {
        'current_hoh': current_hoh,
        'target_hoh': target_hoh_count,
        'additional_needed': additional_hoh_needed,
        'single_with_deps': len(single_with_deps) if len(single_with_deps) > 0 else 0
    }

if __name__ == "__main__":
    analyze_hoh_fine_tuning()
