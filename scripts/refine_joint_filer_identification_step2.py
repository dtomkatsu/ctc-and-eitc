#!/usr/bin/env python3
"""
Step 2: Refine joint filer identification to capture more married couples.
Investigate additional heuristics for complex household structures.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.pums_loader import PUMSDataLoader
from tax.units.status.mfj import _are_married

def analyze_missed_joint_opportunities():
    """Analyze households to find missed joint filing opportunities."""
    
    print("Step 2: Analyzing missed joint filing opportunities...")
    
    # Load data
    pums_loader = PUMSDataLoader()
    person_df, hh_df = pums_loader.load_data()
    tax_units = pd.read_parquet("src/data/processed/tax_units_rule_based.parquet")
    
    # Filter out zero-weight households
    hh_weights = hh_df[hh_df['WGTP'] > 0]['SERIALNO'].tolist()
    person_df = person_df[person_df['SERIALNO'].isin(hh_weights)]
    
    print(f"Analyzing {len(person_df):,} people in {len(hh_weights):,} households...")
    
    # Find households with married adults that aren't filing jointly
    missed_opportunities = []
    
    # Sample analysis on first 5000 households for performance
    sample_households = hh_weights[:5000]
    
    for hh_id in sample_households:
        hh_persons = person_df[person_df['SERIALNO'] == hh_id]
        adults = hh_persons[hh_persons['AGEP'] >= 18]
        married_adults = adults[adults['MAR'] == 1]
        
        if len(married_adults) >= 2:
            # Check if this household has joint filers
            hh_tax_units = tax_units[tax_units['hh_id'] == hh_id]
            joint_units = hh_tax_units[hh_tax_units['filing_status'] == 'joint']
            
            if len(joint_units) == 0:
                # No joint filers - analyze why
                married_pairs = []
                married_list = married_adults.to_dict('records')
                
                # Check all possible pairs
                for i in range(len(married_list)):
                    for j in range(i+1, len(married_list)):
                        adult1 = married_list[i]
                        adult2 = married_list[j]
                        
                        # Convert to Series for _are_married function
                        adult1_series = pd.Series(adult1)
                        adult2_series = pd.Series(adult2)
                        
                        if _are_married(adult1_series, adult2_series):
                            married_pairs.append((adult1, adult2))
                
                if married_pairs:
                    # Found married pairs but no joint filing - this is a missed opportunity
                    missed_opportunities.append({
                        'hh_id': hh_id,
                        'num_married_adults': len(married_adults),
                        'num_married_pairs': len(married_pairs),
                        'married_pairs': married_pairs,
                        'tax_units': hh_tax_units[['filing_status', 'primary_filer_id']].to_dict('records')
                    })
    
    print(f"\nAnalysis Results:")
    print(f"Households analyzed: {len(sample_households):,}")
    print(f"Missed joint filing opportunities: {len(missed_opportunities):,}")
    
    if missed_opportunities:
        print(f"\nDetailed Analysis of Missed Opportunities:")
        
        # Analyze patterns in missed opportunities
        relationship_patterns = {}
        filing_patterns = {}
        
        for opp in missed_opportunities[:10]:  # Show first 10 cases
            print(f"\n--- Household {opp['hh_id']} ---")
            print(f"Married adults: {opp['num_married_adults']}")
            print(f"Married pairs found: {opp['num_married_pairs']}")
            
            for i, (adult1, adult2) in enumerate(opp['married_pairs']):
                rel1, rel2 = adult1['RELSHIPP'], adult2['RELSHIPP']
                age1, age2 = adult1['AGEP'], adult2['AGEP']
                
                print(f"  Pair {i+1}: REL={rel1}+{rel2}, AGE={age1}+{age2}")
                
                # Track relationship patterns
                rel_pattern = tuple(sorted([rel1, rel2]))
                relationship_patterns[rel_pattern] = relationship_patterns.get(rel_pattern, 0) + 1
            
            print(f"  Current filing statuses:")
            for unit in opp['tax_units']:
                print(f"    {unit['filing_status']}: {unit['primary_filer_id']}")
                filing_patterns[unit['filing_status']] = filing_patterns.get(unit['filing_status'], 0) + 1
        
        print(f"\nRelationship Patterns in Missed Opportunities:")
        for pattern, count in sorted(relationship_patterns.items(), key=lambda x: x[1], reverse=True):
            print(f"  REL {pattern[0]}+{pattern[1]}: {count:,} cases")
        
        print(f"\nCurrent Filing Patterns in Missed Opportunities:")
        for status, count in sorted(filing_patterns.items(), key=lambda x: x[1], reverse=True):
            print(f"  {status}: {count:,} cases")
    
    # Estimate potential improvement
    if missed_opportunities:
        # Extrapolate to full dataset
        sample_rate = len(sample_households) / len(hh_weights)
        estimated_total_missed = len(missed_opportunities) / sample_rate
        
        # Assume each missed opportunity could create 1 additional joint unit
        potential_new_joint_units = estimated_total_missed
        
        print(f"\n{'='*80}")
        print(f"POTENTIAL IMPROVEMENT ESTIMATE")
        print(f"{'='*80}")
        
        print(f"Missed opportunities in sample: {len(missed_opportunities):,}")
        print(f"Estimated total missed opportunities: {estimated_total_missed:,.0f}")
        print(f"Potential new joint units: {potential_new_joint_units:,.0f}")
        
        # Calculate impact on filing rates
        current_joint = 250160  # From Step 1 results
        new_joint_total = current_joint + potential_new_joint_units
        total_units = 823134  # From Step 1 results
        
        current_joint_rate = current_joint / total_units * 100
        new_joint_rate = new_joint_total / total_units * 100
        improvement = new_joint_rate - current_joint_rate
        
        print(f"Current joint rate: {current_joint_rate:.1f}%")
        print(f"Potential joint rate: {new_joint_rate:.1f}%")
        print(f"Improvement: +{improvement:.1f} percentage points")
        print(f"Remaining gap vs 36.0%: {36.0 - new_joint_rate:.1f} percentage points")
        
        if improvement > 1.0:
            print("✅ SIGNIFICANT IMPROVEMENT POSSIBLE")
        elif improvement > 0.5:
            print("⚠️  MODERATE IMPROVEMENT POSSIBLE")
        else:
            print("❌ LIMITED IMPROVEMENT EXPECTED")
    
    return missed_opportunities

if __name__ == "__main__":
    analyze_missed_joint_opportunities()
