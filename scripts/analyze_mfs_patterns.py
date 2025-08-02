#!/usr/bin/env python3
"""
Analyze married couples to understand patterns that should lead to MFS filing.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

def main():
    # Load person and household data
    print("Loading PUMS data...")
    person_df = pd.read_parquet('data/processed/pums_person_processed.parquet')
    hh_df = pd.read_parquet('data/processed/pums_household_processed.parquet')
    
    # Find all married adults
    married_adults = person_df[
        (person_df['MAR'] == 1) &  # Married
        (person_df['AGEP'] >= 18)   # Adults
    ].copy()
    
    print(f"Total married adults: {len(married_adults):,}")
    
    # Group by household to find married couples
    married_couples = []
    
    for serialno, hh_group in married_adults.groupby('SERIALNO'):
        if len(hh_group) >= 2:
            # Look for householder+spouse pairs first
            householder = hh_group[hh_group['RELSHIPP'] == 20]
            spouse = hh_group[hh_group['RELSHIPP'] == 21]
            
            if len(householder) == 1 and len(spouse) == 1:
                married_couples.append({
                    'serialno': serialno,
                    'person1_id': householder.iloc[0].name,
                    'person2_id': spouse.iloc[0].name,
                    'person1_age': householder.iloc[0]['AGEP'],
                    'person2_age': spouse.iloc[0]['AGEP'],
                    'person1_income': householder.iloc[0]['PINCP'] or 0,
                    'person2_income': spouse.iloc[0]['PINCP'] or 0,
                    'person1_rel': householder.iloc[0]['RELSHIPP'],
                    'person2_rel': spouse.iloc[0]['RELSHIPP'],
                    'couple_type': 'householder_spouse'
                })
            else:
                # Look for other married pairs (both MAR=1, similar age, opposite sex)
                adults_list = list(hh_group.iterrows())
                for i in range(len(adults_list)):
                    for j in range(i+1, len(adults_list)):
                        person1 = adults_list[i][1]
                        person2 = adults_list[j][1]
                        
                        age_diff = abs(person1['AGEP'] - person2['AGEP'])
                        opposite_sex = person1['SEX'] != person2['SEX']
                        
                        if age_diff <= 15 and opposite_sex:
                            married_couples.append({
                                'serialno': serialno,
                                'person1_id': person1.name,
                                'person2_id': person2.name,
                                'person1_age': person1['AGEP'],
                                'person2_age': person2['AGEP'],
                                'person1_income': person1['PINCP'] or 0,
                                'person2_income': person2['PINCP'] or 0,
                                'person1_rel': person1['RELSHIPP'],
                                'person2_rel': person2['RELSHIPP'],
                                'couple_type': 'inferred_couple'
                            })
                            break  # Only take first match per household
    
    couples_df = pd.DataFrame(married_couples)
    print(f"Identified {len(couples_df):,} married couples")
    
    # Analyze patterns that might indicate MFS filing
    couples_df['total_income'] = couples_df['person1_income'] + couples_df['person2_income']
    couples_df['income_ratio'] = np.where(
        (couples_df['person1_income'] > 0) & (couples_df['person2_income'] > 0),
        np.maximum(couples_df['person1_income'], couples_df['person2_income']) / 
        np.minimum(couples_df['person1_income'], couples_df['person2_income']),
        np.inf
    )
    couples_df['age_diff'] = abs(couples_df['person1_age'] - couples_df['person2_age'])
    couples_df['avg_age'] = (couples_df['person1_age'] + couples_df['person2_age']) / 2
    
    # Identify potential MFS scenarios based on tax literature
    print("\n" + "="*80)
    print("POTENTIAL MFS SCENARIOS ANALYSIS")
    print("="*80)
    
    # 1. Large income disparities (traditional reason for MFS)
    large_disparity = couples_df[couples_df['income_ratio'] > 5]
    print(f"1. Large income disparity (>5:1 ratio): {len(large_disparity):,} couples ({len(large_disparity)/len(couples_df)*100:.1f}%)")
    
    # 2. One spouse has very high income, other very low
    high_low_income = couples_df[
        ((couples_df['person1_income'] > 100000) & (couples_df['person2_income'] < 10000)) |
        ((couples_df['person2_income'] > 100000) & (couples_df['person1_income'] < 10000))
    ]
    print(f"2. High-low income pattern (>$100K vs <$10K): {len(high_low_income):,} couples ({len(high_low_income)/len(couples_df)*100:.1f}%)")
    
    # 3. Very young couples (more likely to file separately)
    young_couples = couples_df[couples_df['avg_age'] < 25]
    print(f"3. Very young couples (avg age <25): {len(young_couples):,} couples ({len(young_couples)/len(couples_df)*100:.1f}%)")
    
    # 4. Large age gaps (may indicate different life stages)
    large_age_gap = couples_df[couples_df['age_diff'] > 15]
    print(f"4. Large age gaps (>15 years): {len(large_age_gap):,} couples ({len(large_age_gap)/len(couples_df)*100:.1f}%)")
    
    # 5. Both spouses have significant income (dual high earners)
    dual_earners = couples_df[
        (couples_df['person1_income'] > 30000) & 
        (couples_df['person2_income'] > 30000)
    ]
    print(f"5. Dual earners (both >$30K): {len(dual_earners):,} couples ({len(dual_earners)/len(couples_df)*100:.1f}%)")
    
    # 6. Non-traditional couples (inferred vs householder/spouse)
    non_traditional = couples_df[couples_df['couple_type'] == 'inferred_couple']
    print(f"6. Non-traditional couples (not householder/spouse): {len(non_traditional):,} couples ({len(non_traditional)/len(couples_df)*100:.1f}%)")
    
    # 7. Complex household structures (multiple adults)
    household_sizes = person_df[person_df['AGEP'] >= 18].groupby('SERIALNO').size()
    complex_households = couples_df[
        couples_df['serialno'].isin(household_sizes[household_sizes > 2].index)
    ]
    print(f"7. Complex households (>2 adults): {len(complex_households):,} couples ({len(complex_households)/len(couples_df)*100:.1f}%)")
    
    # Combine multiple factors for MFS likelihood
    print("\n" + "="*80)
    print("COMBINED MFS LIKELIHOOD ANALYSIS")
    print("="*80)
    
    # Score each couple for MFS likelihood
    couples_df['mfs_score'] = 0
    
    # Income disparity factor
    couples_df.loc[couples_df['income_ratio'] > 10, 'mfs_score'] += 3
    couples_df.loc[(couples_df['income_ratio'] > 5) & (couples_df['income_ratio'] <= 10), 'mfs_score'] += 2
    couples_df.loc[(couples_df['income_ratio'] > 3) & (couples_df['income_ratio'] <= 5), 'mfs_score'] += 1
    
    # Age factors
    couples_df.loc[couples_df['avg_age'] < 25, 'mfs_score'] += 2
    couples_df.loc[couples_df['age_diff'] > 15, 'mfs_score'] += 1
    
    # Income level factors
    couples_df.loc[
        ((couples_df['person1_income'] > 100000) & (couples_df['person2_income'] < 10000)) |
        ((couples_df['person2_income'] > 100000) & (couples_df['person1_income'] < 10000)),
        'mfs_score'
    ] += 2
    
    # Dual high earners (sometimes file separately for tax optimization)
    couples_df.loc[
        (couples_df['person1_income'] > 50000) & (couples_df['person2_income'] > 50000),
        'mfs_score'
    ] += 1
    
    # Non-traditional couples
    couples_df.loc[couples_df['couple_type'] == 'inferred_couple', 'mfs_score'] += 1
    
    # Complex households
    couples_df.loc[couples_df['serialno'].isin(household_sizes[household_sizes > 2].index), 'mfs_score'] += 1
    
    # Analyze MFS score distribution
    print("MFS Score Distribution:")
    score_dist = couples_df['mfs_score'].value_counts().sort_index()
    for score, count in score_dist.items():
        pct = count / len(couples_df) * 100
        print(f"  Score {score}: {count:,} couples ({pct:.1f}%)")
    
    # Estimate MFS rates by score
    print(f"\nEstimated MFS candidates by score threshold:")
    for threshold in [1, 2, 3, 4, 5]:
        candidates = couples_df[couples_df['mfs_score'] >= threshold]
        pct = len(candidates) / len(couples_df) * 100
        print(f"  Score >={threshold}: {len(candidates):,} couples ({pct:.1f}% of all couples)")
    
    # To reach 3.1% MFS rate, we need about 3.1% of couples to file separately
    target_mfs_couples = int(len(couples_df) * 0.031)
    print(f"\nTo reach 3.1% MFS rate, need ~{target_mfs_couples:,} couples filing separately")
    
    # Find the score threshold that gets us closest to 3.1%
    for threshold in range(1, 8):
        candidates = couples_df[couples_df['mfs_score'] >= threshold]
        pct = len(candidates) / len(couples_df) * 100
        if pct <= 3.5:  # Close to target
            print(f"  Threshold {threshold} gives {len(candidates):,} couples ({pct:.1f}%) - closest to 3.1% target")
            
            # Show characteristics of these couples
            print(f"    Avg income ratio: {candidates['income_ratio'].replace([np.inf], np.nan).mean():.1f}")
            print(f"    Avg age: {candidates['avg_age'].mean():.1f}")
            print(f"    Non-traditional couples: {(candidates['couple_type'] == 'inferred_couple').sum():,}")
            break

if __name__ == "__main__":
    main()
