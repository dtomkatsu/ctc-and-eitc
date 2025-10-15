#!/usr/bin/env python3
"""
Diagnose MFS Scoring - Why are we getting 0% MFS filers?

This script analyzes married couples to understand their MFS scores
and determine why none are filing separately.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def calculate_mfs_score(adult1, adult2, num_adults):
    """Calculate MFS score for a married couple."""
    mfs_score = 0
    
    # Get basic info
    income1 = float(adult1.get('PINCP', 0) or 0)
    income2 = float(adult2.get('PINCP', 0) or 0)
    age1 = int(adult1.get('AGEP', 30))
    age2 = int(adult2.get('AGEP', 30))
    rel1 = adult1.get('RELSHIPP', 0)
    rel2 = adult2.get('RELSHIPP', 0)
    
    # Factor 1: Income disparity (strongest predictor)
    income_ratio = 0
    if income1 > 0 and income2 > 0:
        income_ratio = max(income1, income2) / min(income1, income2)
        if income_ratio > 20:
            mfs_score += 4  # Very large disparity
        elif income_ratio > 10:
            mfs_score += 3  # Large disparity
        elif income_ratio > 5:
            mfs_score += 2  # Moderate disparity
        elif income_ratio > 3:
            mfs_score += 1  # Small disparity
    
    # Factor 2: High-income/low-income pattern
    if ((income1 > 100000 and income2 < 10000) or 
        (income2 > 100000 and income1 < 10000)):
        mfs_score += 3
    
    # Factor 3: Age factors
    avg_age = (age1 + age2) / 2
    age_diff = abs(age1 - age2)
    
    if avg_age < 25:  # Very young couples
        mfs_score += 2
    elif avg_age < 30:  # Young couples
        mfs_score += 1
        
    if age_diff > 15:  # Large age gap
        mfs_score += 1
    
    # Factor 4: Dual high earners (tax optimization)
    if income1 > 50000 and income2 > 50000:
        mfs_score += 1
    
    # Factor 5: Non-traditional couple structure
    is_householder_spouse = ((rel1 == 20 and rel2 == 21) or 
                            (rel1 == 21 and rel2 == 20))
    if not is_householder_spouse:
        mfs_score += 1
    
    # Factor 6: Complex household (multiple adults)
    if num_adults > 2:
        mfs_score += 1
    
    # Factor 7: Very high total income (tax planning)
    total_income = income1 + income2
    if total_income > 200000:
        mfs_score += 1
    
    return {
        'mfs_score': mfs_score,
        'income1': income1,
        'income2': income2,
        'income_ratio': income_ratio,
        'age1': age1,
        'age2': age2,
        'avg_age': avg_age,
        'total_income': total_income,
        'is_householder_spouse': is_householder_spouse
    }


def main():
    """Main execution."""
    logger.info("="*80)
    logger.info("MFS SCORING DIAGNOSTIC")
    logger.info("="*80)
    
    # Load PUMS data
    logger.info("\nLoading PUMS data...")
    persons = pd.read_csv('data/raw/pums/psam_p15.csv')
    households = pd.read_csv('data/raw/pums/psam_h15.csv')
    
    logger.info(f"  Loaded {len(persons):,} persons")
    logger.info(f"  Loaded {len(households):,} households")
    
    # Find married couples (both currently married, MAR == 1)
    married_adults = persons[(persons['AGEP'] >= 18) & (persons['MAR'] == 1)]
    
    logger.info(f"\n  Found {len(married_adults):,} married adults")
    
    # Group by household and find married couples
    logger.info("\nAnalyzing married couples...")
    
    couples_data = []
    
    for serialno, hh_group in married_adults.groupby('SERIALNO'):
        hh_married = hh_group[hh_group['MAR'] == 1]
        
        if len(hh_married) >= 2:
            # Pair up married adults
            adults_list = hh_married.to_dict('records')
            
            # Simple pairing: first two married adults
            if len(adults_list) >= 2:
                adult1 = adults_list[0]
                adult2 = adults_list[1]
                
                # Get household info
                num_adults = len(hh_group)
                
                # Calculate MFS score
                score_data = calculate_mfs_score(adult1, adult2, num_adults)
                score_data['serialno'] = serialno
                score_data['weight'] = adult1.get('PWGTP', 1)
                
                couples_data.append(score_data)
    
    # Create DataFrame
    couples_df = pd.DataFrame(couples_data)
    
    logger.info(f"\nAnalyzed {len(couples_df):,} married couples")
    
    # Score distribution
    logger.info("\n" + "="*80)
    logger.info("MFS SCORE DISTRIBUTION")
    logger.info("="*80)
    
    score_dist = couples_df.groupby('mfs_score').agg({
        'serialno': 'count',
        'weight': 'sum'
    })
    score_dist.columns = ['Count', 'Weighted']
    score_dist['% of Couples'] = (score_dist['Weighted'] / score_dist['Weighted'].sum() * 100).round(2)
    
    print("\nScore | Unweighted | Weighted | % of Couples")
    print("------|------------|----------|-------------")
    for score in range(0, int(couples_df['mfs_score'].max()) + 1):
        if score in score_dist.index:
            row = score_dist.loc[score]
            print(f"  {score}   | {row['Count']:>10,} | {row['Weighted']:>8,.0f} | {row['% of Couples']:>7.2f}%")
        else:
            print(f"  {score}   | {0:>10,} | {0:>8,.0f} | {0:>7.2f}%")
    
    # Expected MFS filers by score
    logger.info("\n" + "="*80)
    logger.info("EXPECTED MFS FILERS BY SCORE")
    logger.info("="*80)
    
    print("\nScore | Weighted Couples | MFS Probability | Expected MFS Filers")
    print("------|------------------|-----------------|--------------------")
    
    mfs_probabilities = {
        0: 0.0,
        1: 0.0,
        2: 0.0,
        3: 0.0,
        4: 0.2,
        5: 0.5,
        6: 0.7,
        7: 1.0
    }
    
    total_expected_mfs = 0
    for score in range(0, int(couples_df['mfs_score'].max()) + 1):
        if score in score_dist.index:
            weighted = score_dist.loc[score, 'Weighted']
            prob = mfs_probabilities.get(score, 1.0 if score >= 7 else 0.0)
            expected = weighted * prob
            total_expected_mfs += expected
            print(f"  {score}   | {weighted:>16,.0f} | {prob:>15.0%} | {expected:>18,.0f}")
        else:
            print(f"  {score}   | {0:>16,.0f} | {0:>15.0%} | {0:>18,.0f}")
    
    print(f"\nTOTAL EXPECTED MFS FILERS: {total_expected_mfs:,.0f}")
    
    # Calculate total married filers (joint + MFS)
    total_married = score_dist['Weighted'].sum() * 2  # Each couple = 2 filers
    mfs_pct = (total_expected_mfs / total_married * 100) if total_married > 0 else 0
    
    print(f"Total married filers (joint + MFS): {total_married:,.0f}")
    print(f"Expected MFS percentage: {mfs_pct:.2f}%")
    print(f"DOTAX target: 2.5%")
    
    # Analyze score 4+ couples
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS OF SCORE 4+ COUPLES (MFS Candidates)")
    logger.info("="*80)
    
    mfs_candidates = couples_df[couples_df['mfs_score'] >= 4]
    
    if len(mfs_candidates) > 0:
        print(f"\nFound {len(mfs_candidates):,} couples with score >= 4")
        print(f"Weighted count: {mfs_candidates['weight'].sum():,.0f}")
        
        print("\nIncome characteristics:")
        print(f"  Avg income ratio: {mfs_candidates['income_ratio'].mean():.2f}")
        print(f"  Max income ratio: {mfs_candidates['income_ratio'].max():.2f}")
        print(f"  Avg total income: ${mfs_candidates['total_income'].mean():,.0f}")
        print(f"  Max total income: ${mfs_candidates['total_income'].max():,.0f}")
        
        print("\nAge characteristics:")
        print(f"  Avg age: {mfs_candidates['avg_age'].mean():.1f}")
        print(f"  Min avg age: {mfs_candidates['avg_age'].min():.1f}")
        
        print("\nTop 10 highest-scoring couples:")
        top_couples = mfs_candidates.nlargest(10, 'mfs_score')[
            ['mfs_score', 'income1', 'income2', 'income_ratio', 'age1', 'age2', 'total_income', 'weight']
        ]
        print(top_couples.to_string(index=False))
    else:
        print("\n⚠️ NO COUPLES WITH SCORE >= 4 FOUND!")
        print("This explains why we have 0% MFS filers.")
        
        print("\nHighest-scoring couples:")
        top_couples = couples_df.nlargest(10, 'mfs_score')[
            ['mfs_score', 'income1', 'income2', 'income_ratio', 'age1', 'age2', 'total_income', 'weight']
        ]
        print(top_couples.to_string(index=False))
    
    # Recommendations
    logger.info("\n" + "="*80)
    logger.info("RECOMMENDATIONS")
    logger.info("="*80)
    
    print("\nTo reach DOTAX target of 2.5% MFS filers:")
    target_mfs = total_married * 0.025
    print(f"  Target MFS filers: {target_mfs:,.0f}")
    print(f"  Current expected: {total_expected_mfs:,.0f}")
    print(f"  Gap: {target_mfs - total_expected_mfs:,.0f}")
    
    print("\nOptions:")
    print("  1. Lower MFS score threshold (currently 4)")
    print("  2. Increase MFS probabilities for existing scores")
    print("  3. Add additional scoring factors")
    print("  4. Use random sampling to reach exactly 2.5%")
    
    # Save detailed results
    output_file = 'analysis_results/calibration/mfs_scoring_analysis.csv'
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    couples_df.to_csv(output_file, index=False)
    logger.info(f"\n✅ Detailed results saved to: {output_file}")


if __name__ == '__main__':
    main()
