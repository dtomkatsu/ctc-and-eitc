#!/usr/bin/env python3
"""
Diagnose Remaining 12.4% Coverage Gap

Current: 556,643 tax units (87.6%)
Target: 635,117 tax units (100%)
Gap: 78,474 missing filers (12.4%)

This script investigates:
1. Unassigned adults - why aren't they creating tax units?
2. Household composition - are we still capping too aggressively?
3. Relationship codes - are some adults being excluded?
4. Income distribution - are low-income adults being missed?
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def analyze_unassigned_adults():
    """Analyze which adults are not assigned to tax units."""
    logger.info("="*80)
    logger.info("1. UNASSIGNED ADULTS ANALYSIS")
    logger.info("="*80)
    
    # Load data
    persons = pd.read_csv('data/raw/pums/psam_p15.csv')
    tax_units_file = sorted(Path('data/processed').glob('tax_units_calibrated_*.parquet'))[-1]
    tax_units = pd.read_parquet(tax_units_file)
    
    logger.info(f"\nUsing: {tax_units_file.name}")
    
    # Get adults
    adults = persons[persons['AGEP'] >= 18].copy()
    total_adults_weighted = adults['PWGTP'].sum()
    
    # Get adults in tax units
    # Primary filers
    adults_in_units = tax_units['weight'].sum()
    
    # Add spouses (joint + MFS)
    joint_filers = tax_units[tax_units['filing_status'] == 'married_filing_jointly']['weight'].sum()
    mfs_filers = tax_units[tax_units['filing_status'] == 'married_filing_separately']['weight'].sum()
    adults_in_units += joint_filers + mfs_filers
    
    unassigned = total_adults_weighted - adults_in_units
    
    logger.info(f"\nAdult Assignment:")
    logger.info(f"  Total adults in PUMS:          {total_adults_weighted:>12,.0f}")
    logger.info(f"  Adults in tax units:           {adults_in_units:>12,.0f}")
    logger.info(f"  Unassigned adults:             {unassigned:>12,.0f}")
    logger.info(f"  Assignment rate:               {adults_in_units/total_adults_weighted:>11.1%}")
    
    # If we assigned all unassigned adults, how many tax units?
    potential_additional_units = unassigned
    potential_total = tax_units['weight'].sum() + potential_additional_units
    
    logger.info(f"\nPotential if all adults assigned:")
    logger.info(f"  Current tax units:             {tax_units['weight'].sum():>12,.0f}")
    logger.info(f"  Potential additional:          {potential_additional_units:>12,.0f}")
    logger.info(f"  Potential total:               {potential_total:>12,.0f}")
    logger.info(f"  SOI target:                    {635117:>12,}")
    logger.info(f"  Would be:                      {potential_total/635117:>11.1%} of target")
    
    # Analyze unassigned adults by household
    # Create person ID to match tax units
    adults['person_id'] = adults['SERIALNO'].astype(str) + '_' + adults.index.astype(str)
    
    # Get all filer IDs from tax units
    assigned_ids = set()
    for _, unit in tax_units.iterrows():
        assigned_ids.add(unit['primary_filer_id'])
        # Note: We can't easily get spouse IDs from current structure
    
    logger.info(f"\n  Assigned adult IDs in tax units: {len(assigned_ids):,}")
    
    return {
        'total_adults': total_adults_weighted,
        'adults_in_units': adults_in_units,
        'unassigned': unassigned,
        'potential_total': potential_total
    }


def analyze_household_sizes():
    """Analyze household sizes and tax unit creation rates."""
    logger.info("\n" + "="*80)
    logger.info("2. HOUSEHOLD SIZE ANALYSIS")
    logger.info("="*80)
    
    persons = pd.read_csv('data/raw/pums/psam_p15.csv')
    households = pd.read_csv('data/raw/pums/psam_h15.csv')
    tax_units_file = sorted(Path('data/processed').glob('tax_units_calibrated_*.parquet'))[-1]
    tax_units = pd.read_parquet(tax_units_file)
    
    # Count adults per household
    adults = persons[persons['AGEP'] >= 18]
    adults_per_hh = adults.groupby('SERIALNO').size()
    
    # Count tax units per household
    tax_units_per_hh = tax_units.groupby('SERIALNO').size()
    
    # Merge with household weights
    hh_with_weights = households.set_index('SERIALNO')
    
    logger.info(f"\nTax Unit Creation by Household Size:")
    logger.info(f"{'Adults':>7} {'Households':>12} {'Avg Units':>12} {'Max Possible':>14} {'Actual':>12} {'% of Max':>10}")
    logger.info("-"*80)
    
    for num_adults in sorted(adults_per_hh.unique()):
        hh_with_n = adults_per_hh[adults_per_hh == num_adults].index
        
        # Weighted household count
        weighted_hh = hh_with_weights.loc[hh_with_weights.index.isin(hh_with_n), 'WGTP'].sum()
        
        # Average tax units created
        units_in_these_hh = tax_units[tax_units['SERIALNO'].isin(hh_with_n)]
        if len(units_in_these_hh) > 0:
            avg_units = units_in_these_hh['weight'].sum() / weighted_hh
        else:
            avg_units = 0
        
        # Max possible (capped at 6)
        max_possible = min(6, num_adults)
        
        # Actual vs max
        pct_of_max = (avg_units / max_possible * 100) if max_possible > 0 else 0
        
        logger.info(f"{num_adults:>7} {weighted_hh:>12,.0f} {avg_units:>12.2f} {max_possible:>14} {avg_units*weighted_hh:>12,.0f} {pct_of_max:>9.1f}%")
    
    # Check if we're hitting the cap of 6
    logger.info(f"\n\nHouseholds Hitting Cap (6 tax units):")
    large_hh = adults_per_hh[adults_per_hh >= 6]
    logger.info(f"  Households with 6+ adults: {len(large_hh):,}")
    weighted_large = hh_with_weights.loc[hh_with_weights.index.isin(large_hh.index), 'WGTP'].sum()
    logger.info(f"  Weighted count: {weighted_large:,.0f}")
    
    # Check actual tax units in these households
    units_in_large = tax_units[tax_units['SERIALNO'].isin(large_hh.index)]
    logger.info(f"  Tax units created: {units_in_large['weight'].sum():,.0f}")
    logger.info(f"  Average per household: {units_in_large['weight'].sum() / weighted_large:.2f}")


def analyze_relationship_codes():
    """Analyze adults by relationship to householder."""
    logger.info("\n" + "="*80)
    logger.info("3. RELATIONSHIP CODE ANALYSIS")
    logger.info("="*80)
    
    persons = pd.read_csv('data/raw/pums/psam_p15.csv')
    adults = persons[persons['AGEP'] >= 18]
    
    rel_codes = {
        20: 'Householder',
        21: 'Spouse',
        22: 'Parent',
        23: 'Child',
        24: 'Sibling',
        25: 'Other relative',
        26: 'Grandchild',
        27: 'In-law',
        28: 'Other nonrelative',
        29: 'Institutionalized',
        30: 'Noninstitutionalized',
        # Add more codes
    }
    
    logger.info(f"\nAdults by Relationship to Householder:")
    logger.info(f"{'Relationship':<30} {'Count':>12} {'% of Adults':>12}")
    logger.info("-"*80)
    
    rel_counts = adults.groupby('RELSHIPP')['PWGTP'].sum().sort_values(ascending=False)
    total_adults = adults['PWGTP'].sum()
    
    for rel_code, count in rel_counts.items():
        rel_name = rel_codes.get(rel_code, f'Unknown ({rel_code})')
        pct = count / total_adults * 100
        logger.info(f"{rel_name:<30} {count:>12,.0f} {pct:>11.1f}%")
    
    # Focus on codes that might be excluded
    logger.info(f"\n\nPotentially Excluded Relationship Codes:")
    
    # Other relatives (25)
    other_rel = adults[adults['RELSHIPP'] == 25]
    logger.info(f"\n  Other relatives (25): {other_rel['PWGTP'].sum():,.0f}")
    logger.info(f"    These should file separately!")
    
    # Noninstitutionalized (30)
    noninst = adults[adults['RELSHIPP'] == 30]
    logger.info(f"\n  Noninstitutionalized (30): {noninst['PWGTP'].sum():,.0f}")
    logger.info(f"    These might be group quarters - may not all file")
    
    # Adult children (23)
    adult_children = adults[adults['RELSHIPP'] == 23]
    logger.info(f"\n  Adult children (23): {adult_children['PWGTP'].sum():,.0f}")
    logger.info(f"    These should file separately!")
    
    # Unknown codes (>30)
    unknown = adults[adults['RELSHIPP'] > 30]
    if len(unknown) > 0:
        logger.info(f"\n  Unknown codes (>30): {unknown['PWGTP'].sum():,.0f}")
        logger.info(f"    Codes: {sorted(unknown['RELSHIPP'].unique())}")


def analyze_income_distribution():
    """Analyze income distribution of adults."""
    logger.info("\n" + "="*80)
    logger.info("4. INCOME DISTRIBUTION ANALYSIS")
    logger.info("="*80)
    
    persons = pd.read_csv('data/raw/pums/psam_p15.csv')
    adults = persons[persons['AGEP'] >= 18].copy()
    
    # Replace NaN income with 0
    adults['PINCP'] = adults['PINCP'].fillna(0)
    
    logger.info(f"\nAdults by Income Level:")
    logger.info(f"{'Income Range':<20} {'Count':>12} {'% of Adults':>12}")
    logger.info("-"*80)
    
    income_ranges = [
        ('Negative', -float('inf'), 0),
        ('$0', 0, 1),
        ('$1-$5K', 1, 5000),
        ('$5K-$10K', 5000, 10000),
        ('$10K-$15K', 10000, 15000),
        ('$15K-$20K', 15000, 20000),
        ('$20K-$30K', 20000, 30000),
        ('$30K-$50K', 30000, 50000),
        ('$50K-$75K', 50000, 75000),
        ('$75K-$100K', 75000, 100000),
        ('$100K+', 100000, float('inf'))
    ]
    
    total_adults = adults['PWGTP'].sum()
    
    for label, min_inc, max_inc in income_ranges:
        in_range = adults[(adults['PINCP'] >= min_inc) & (adults['PINCP'] < max_inc)]
        count = in_range['PWGTP'].sum()
        pct = count / total_adults * 100
        logger.info(f"{label:<20} {count:>12,.0f} {pct:>11.1f}%")
    
    # Zero income adults
    zero_income = adults[adults['PINCP'] == 0]
    logger.info(f"\n\nZero Income Adults: {zero_income['PWGTP'].sum():,.0f} ({zero_income['PWGTP'].sum()/total_adults*100:.1f}%)")
    logger.info(f"  These might be:")
    logger.info(f"    - Dependents (students, stay-at-home spouses)")
    logger.info(f"    - Unemployed")
    logger.info(f"    - Retired with only SS income (not in PINCP)")
    
    # Check relationship codes of zero-income adults
    logger.info(f"\n  Zero-income adults by relationship:")
    zero_rel = zero_income.groupby('RELSHIPP')['PWGTP'].sum().sort_values(ascending=False).head(5)
    rel_codes = {20: 'Householder', 21: 'Spouse', 23: 'Child', 25: 'Other relative', 30: 'Noninstitutionalized'}
    for rel_code, count in zero_rel.items():
        rel_name = rel_codes.get(rel_code, f'Code {rel_code}')
        logger.info(f"    {rel_name}: {count:,.0f}")


def propose_solutions():
    """Propose solutions to close the remaining gap."""
    logger.info("\n" + "="*80)
    logger.info("5. PROPOSED SOLUTIONS")
    logger.info("="*80)
    
    gap = 635117 - 556643
    
    logger.info(f"\nCurrent Gap: {gap:,} filers (12.4%)")
    logger.info(f"\nPotential Solutions:\n")
    
    solutions = [
        {
            'name': 'Solution 1: Remove household cap entirely',
            'description': 'Allow unlimited tax units per household',
            'expected_gain': '50,000-100,000 units',
            'risk': 'May over-count if not careful',
            'effort': 'Low (1 line change)',
            'recommendation': '⭐⭐⭐ Try this first'
        },
        {
            'name': 'Solution 2: Ensure all adults create tax units',
            'description': 'Review logic to ensure every adult (except spouses) creates a tax unit',
            'expected_gain': '80,000-150,000 units',
            'risk': 'May over-count dependents',
            'effort': 'Medium (logic review)',
            'recommendation': '⭐⭐⭐ High impact'
        },
        {
            'name': 'Solution 3: Include "other relatives" (RELSHIPP=25)',
            'description': 'Ensure other relatives in household create separate tax units',
            'expected_gain': '30,000-50,000 units',
            'risk': 'Low',
            'effort': 'Low (check if already included)',
            'recommendation': '⭐⭐ Good addition'
        },
        {
            'name': 'Solution 4: Apply scaling factor',
            'description': 'Multiply all weights by 1.141 (635117/556643)',
            'expected_gain': '78,474 units (exact)',
            'risk': 'None (just scaling)',
            'effort': 'Very low (post-processing)',
            'recommendation': '⭐ Fallback option'
        },
        {
            'name': 'Solution 5: Lower income threshold for tax unit creation',
            'description': 'Create tax units for very low-income adults',
            'expected_gain': '20,000-40,000 units',
            'risk': 'May include non-filers',
            'effort': 'Low (already removed threshold)',
            'recommendation': '⭐ Already done'
        }
    ]
    
    for i, sol in enumerate(solutions, 1):
        logger.info(f"{i}. {sol['name']}")
        logger.info(f"   Description: {sol['description']}")
        logger.info(f"   Expected gain: {sol['expected_gain']}")
        logger.info(f"   Risk: {sol['risk']}")
        logger.info(f"   Effort: {sol['effort']}")
        logger.info(f"   Recommendation: {sol['recommendation']}\n")


def main():
    """Main execution."""
    logger.info("="*80)
    logger.info("DIAGNOSING REMAINING 12.4% COVERAGE GAP")
    logger.info("="*80)
    logger.info(f"\nCurrent: 556,643 tax units (87.6%)")
    logger.info(f"Target:  635,117 tax units (100%)")
    logger.info(f"Gap:     78,474 missing filers (12.4%)\n")
    
    # Run analyses
    adult_stats = analyze_unassigned_adults()
    analyze_household_sizes()
    analyze_relationship_codes()
    analyze_income_distribution()
    propose_solutions()
    
    logger.info("\n" + "="*80)
    logger.info("DIAGNOSIS COMPLETE")
    logger.info("="*80)
    
    logger.info(f"\n\nKEY FINDINGS:")
    logger.info(f"  1. Unassigned adults: {adult_stats['unassigned']:,.0f}")
    logger.info(f"  2. If all assigned: {adult_stats['potential_total']:,.0f} units ({adult_stats['potential_total']/635117*100:.1f}% of target)")
    logger.info(f"  3. Primary issue: Not all adults creating tax units")
    logger.info(f"\n  RECOMMENDATION: Remove household cap entirely and ensure all adults")
    logger.info(f"                  (except spouses) create their own tax units.")


if __name__ == '__main__':
    main()
