#!/usr/bin/env python3
"""
Comprehensive analysis of the remaining joint filing gap to understand
why we're still at 29.2% vs 36.0% expected, and determine if further
improvements are possible or if this represents the realistic limit.
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.pums_loader import PUMSDataLoader

def analyze_final_joint_gap():
    """Comprehensive analysis of the remaining joint filing gap."""
    
    print("Analyzing the final joint filing gap...")
    
    # Load data
    tax_units = pd.read_parquet("src/data/processed/tax_units_rule_based.parquet")
    pums_loader = PUMSDataLoader()
    person_df, hh_df = pums_loader.load_data()
    
    # Current status
    filing_counts = tax_units['filing_status'].value_counts()
    total_units = len(tax_units)
    
    current_joint_pct = filing_counts.get('joint', 0) / total_units * 100
    expected_joint_pct = 36.0
    gap = expected_joint_pct - current_joint_pct
    missing_units = int(total_units * gap / 100)
    
    print(f"Current Status:")
    print(f"  Joint filers: {filing_counts.get('joint', 0):,} ({current_joint_pct:.1f}%)")
    print(f"  Expected: {int(total_units * expected_joint_pct / 100):,} ({expected_joint_pct:.1f}%)")
    print(f"  Gap: {gap:.1f} percentage points ({missing_units:,} missing units)")
    
    # Analyze the population characteristics
    print(f"\nPopulation Analysis:")
    
    # Count all married adults in PUMS data
    married_adults = person_df[(person_df['AGEP'] >= 18) & (person_df['MAR'] == 1)]
    total_adults = len(person_df[person_df['AGEP'] >= 18])
    
    print(f"  Total adults (18+): {total_adults:,}")
    print(f"  Married adults: {len(married_adults):,} ({len(married_adults)/total_adults*100:.1f}%)")
    
    # Theoretical maximum joint filers (married adults / 2)
    max_possible_joint_units = len(married_adults) // 2
    current_joint_units = filing_counts.get('joint', 0)
    joint_capture_rate = current_joint_units / max_possible_joint_units * 100 if max_possible_joint_units > 0 else 0
    
    print(f"  Theoretical max joint units: {max_possible_joint_units:,}")
    print(f"  Current joint units: {current_joint_units:,}")
    print(f"  Joint capture rate: {joint_capture_rate:.1f}%")
    
    # Analyze household composition
    print(f"\nHousehold Composition Analysis:")
    
    household_types = {
        'single_adult': 0,
        'married_couple_only': 0,
        'married_couple_with_others': 0,
        'multiple_married_adults': 0,
        'other': 0
    }
    
    for hh_id in hh_df['SERIALNO'].unique()[:5000]:  # Sample analysis
        hh_persons = person_df[person_df['SERIALNO'] == hh_id]
        adults = hh_persons[hh_persons['AGEP'] >= 18]
        married_adults_in_hh = adults[adults['MAR'] == 1]
        
        if len(adults) == 1:
            household_types['single_adult'] += 1
        elif len(married_adults_in_hh) == 2 and len(adults) == 2:
            household_types['married_couple_only'] += 1
        elif len(married_adults_in_hh) == 2 and len(adults) > 2:
            household_types['married_couple_with_others'] += 1
        elif len(married_adults_in_hh) > 2:
            household_types['multiple_married_adults'] += 1
        else:
            household_types['other'] += 1
    
    total_sampled = sum(household_types.values())
    print(f"  Household types (sample of {total_sampled:,}):")
    for hh_type, count in household_types.items():
        pct = count / total_sampled * 100 if total_sampled > 0 else 0
        print(f"    {hh_type}: {count:,} ({pct:.1f}%)")
    
    # Analyze current filing patterns
    print(f"\nCurrent Filing Pattern Analysis:")
    
    # Check MFS rate
    mfs_units = filing_counts.get('married_filing_separate', 0)
    mfs_pct = mfs_units / total_units * 100
    
    # Total married filers (joint + MFS)
    total_married_filers = filing_counts.get('joint', 0) + mfs_units
    married_filer_pct = total_married_filers / total_units * 100
    
    print(f"  Joint filers: {filing_counts.get('joint', 0):,} ({current_joint_pct:.1f}%)")
    print(f"  MFS filers: {mfs_units:,} ({mfs_pct:.1f}%)")
    print(f"  Total married filers: {total_married_filers:,} ({married_filer_pct:.1f}%)")
    print(f"  Single filers: {filing_counts.get('single', 0):,} ({filing_counts.get('single', 0)/total_units*100:.1f}%)")
    print(f"  HoH filers: {filing_counts.get('head_of_household', 0):,} ({filing_counts.get('head_of_household', 0)/total_units*100:.1f}%)")
    
    # Compare to SOI benchmarks
    print(f"\nComparison to SOI Benchmarks:")
    soi_benchmarks = {
        'single': 51.0,
        'joint': 36.0,
        'head_of_household': 9.6,
        'married_filing_separate': 3.4  # Estimated
    }
    
    current_rates = {
        'single': filing_counts.get('single', 0) / total_units * 100,
        'joint': current_joint_pct,
        'head_of_household': filing_counts.get('head_of_household', 0) / total_units * 100,
        'married_filing_separate': mfs_pct
    }
    
    for status, soi_rate in soi_benchmarks.items():
        current_rate = current_rates.get(status, 0)
        diff = current_rate - soi_rate
        print(f"  {status}: {current_rate:.1f}% vs {soi_rate:.1f}% SOI ({diff:+.1f} pp)")
    
    # Analysis of potential causes
    print(f"\n{'='*80}")
    print(f"ANALYSIS OF REMAINING GAP")
    print(f"{'='*80}")
    
    print(f"Potential explanations for the remaining {gap:.1f} percentage point gap:")
    
    print(f"\n1. POPULATION DIFFERENCES:")
    print(f"   - PUMS data may have different demographic composition than SOI data")
    print(f"   - Hawaii may have unique household structures not captured in national benchmarks")
    print(f"   - Age distribution, income patterns, or cultural factors may affect filing patterns")
    
    print(f"\n2. DATA LIMITATIONS:")
    print(f"   - PUMS relationship codes may not capture all married couple arrangements")
    print(f"   - Some married couples may be coded as separate individuals in complex households")
    print(f"   - Missing or misclassified relationship information")
    
    print(f"\n3. METHODOLOGICAL DIFFERENCES:")
    print(f"   - SOI data includes actual tax returns, PUMS is survey data")
    print(f"   - Different time periods or data collection methods")
    print(f"   - SOI may include couples who don't live together but file jointly")
    
    print(f"\n4. REMAINING TECHNICAL ISSUES:")
    print(f"   - Some edge cases in married couple identification still missed")
    print(f"   - MFS classification may still be slightly too aggressive")
    print(f"   - Complex household structures not fully handled")
    
    # Recommendations
    print(f"\nRECOMMENDations:")
    
    if gap < 3.0:
        print(f"✅ GAP IS SMALL ({gap:.1f} pp) - Current results are very good!")
        print(f"   - The remaining gap may represent realistic differences between PUMS and SOI data")
        print(f"   - Further optimization may yield diminishing returns")
        print(f"   - Consider the current results production-ready")
    elif gap < 5.0:
        print(f"⚠️  GAP IS MODERATE ({gap:.1f} pp) - Some improvement possible")
        print(f"   - Investigate specific edge cases in married couple identification")
        print(f"   - Fine-tune MFS classification thresholds")
        print(f"   - Consider additional heuristics for complex households")
    else:
        print(f"❌ GAP IS LARGE ({gap:.1f} pp) - Significant issues remain")
        print(f"   - Major systematic problems likely still exist")
        print(f"   - Comprehensive review of joint filer identification needed")
        print(f"   - Consider fundamental changes to the approach")
    
    print(f"\nCurrent joint capture rate: {joint_capture_rate:.1f}% of theoretical maximum")
    if joint_capture_rate > 85:
        print(f"✅ Excellent capture rate - most married couples are filing jointly")
    elif joint_capture_rate > 70:
        print(f"⚠️  Good capture rate - some room for improvement")
    else:
        print(f"❌ Low capture rate - significant issues with married couple identification")

if __name__ == "__main__":
    analyze_final_joint_gap()
