#!/usr/bin/env python3
"""
Investigate Coverage Gap - Why are we missing 26.6% of expected filers?

This script analyzes:
1. PUMS population totals vs Census/SOI expectations
2. Household cap logic (max 2 tax units per household)
3. Dependent classification issues
4. Adult assignment and tax unit creation rates
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_pums_population():
    """Analyze PUMS population totals."""
    logger.info("="*80)
    logger.info("1. PUMS POPULATION ANALYSIS")
    logger.info("="*80)
    
    # Load PUMS data
    persons = pd.read_csv('data/raw/pums/psam_p15.csv')
    households = pd.read_csv('data/raw/pums/psam_h15.csv')
    
    logger.info(f"\nRaw PUMS Data:")
    logger.info(f"  Total persons (unweighted): {len(persons):,}")
    logger.info(f"  Total households (unweighted): {len(households):,}")
    
    # Weighted totals
    total_persons_weighted = persons['PWGTP'].sum()
    total_hh_weighted = households['WGTP'].sum()
    
    logger.info(f"\nWeighted PUMS Data:")
    logger.info(f"  Total persons (weighted): {total_persons_weighted:,.0f}")
    logger.info(f"  Total households (weighted): {total_hh_weighted:,.0f}")
    
    # Adults (age >= 18)
    adults = persons[persons['AGEP'] >= 18]
    total_adults_weighted = adults['PWGTP'].sum()
    
    logger.info(f"\nAdult Population:")
    logger.info(f"  Total adults (unweighted): {len(adults):,}")
    logger.info(f"  Total adults (weighted): {total_adults_weighted:,.0f}")
    
    # Expected vs actual
    soi_total = 635117  # DOTAX 2022 SOI total
    current_total = 466355  # Our current total
    
    logger.info(f"\nComparison to SOI:")
    logger.info(f"  SOI 2022 total filers:        {soi_total:>12,}")
    logger.info(f"  PUMS adults (weighted):       {total_adults_weighted:>12,.0f}")
    logger.info(f"  Current tax units:            {current_total:>12,}")
    logger.info(f"  Gap (SOI - current):          {soi_total - current_total:>12,}")
    logger.info(f"  Coverage rate:                {current_total / soi_total:>11.1%}")
    logger.info(f"  Adults per filer (expected):  {total_adults_weighted / soi_total:>11.2f}")
    logger.info(f"  Adults per tax unit (actual): {total_adults_weighted / current_total:>11.2f}")
    
    # Check if PUMS has enough adults
    if total_adults_weighted < soi_total:
        logger.warning(f"\n⚠️  PUMS has fewer adults ({total_adults_weighted:,.0f}) than SOI filers ({soi_total:,})!")
        logger.warning(f"   This is impossible - each filer must be at least one adult.")
        logger.warning(f"   Gap: {soi_total - total_adults_weighted:,.0f} adults")
    else:
        logger.info(f"\n✅ PUMS has enough adults ({total_adults_weighted:,.0f}) for SOI filers ({soi_total:,})")
        logger.info(f"   Surplus: {total_adults_weighted - soi_total:,.0f} adults")
    
    return {
        'total_persons': total_persons_weighted,
        'total_households': total_hh_weighted,
        'total_adults': total_adults_weighted,
        'soi_total': soi_total,
        'current_total': current_total
    }


def analyze_household_cap():
    """Analyze impact of household cap (max 2 tax units per household)."""
    logger.info("\n" + "="*80)
    logger.info("2. HOUSEHOLD CAP ANALYSIS")
    logger.info("="*80)
    
    # Load PUMS data
    persons = pd.read_csv('data/raw/pums/psam_p15.csv')
    households = pd.read_csv('data/raw/pums/psam_h15.csv')
    
    # Count adults per household
    adults = persons[persons['AGEP'] >= 18].copy()
    adults_per_hh = adults.groupby('SERIALNO').size()
    
    logger.info(f"\nAdults per Household Distribution:")
    logger.info(f"{'Adults':<10} {'Households':>12} {'% of HH':>10} {'Potential Tax Units':>20}")
    logger.info("-"*80)
    
    total_potential_units = 0
    capped_units = 0
    
    for num_adults in sorted(adults_per_hh.unique()):
        hh_count = (adults_per_hh == num_adults).sum()
        hh_pct = hh_count / len(adults_per_hh) * 100
        
        # Potential tax units (assuming each adult could file)
        potential = num_adults * hh_count
        
        # Capped at 2 per household
        capped = min(2, num_adults) * hh_count
        
        total_potential_units += potential
        capped_units += capped
        
        logger.info(f"{num_adults:<10} {hh_count:>12,} {hh_pct:>9.1f}% {potential:>20,} (capped: {capped:,})")
    
    logger.info("-"*80)
    logger.info(f"{'TOTAL':<10} {len(adults_per_hh):>12,} {'100.0%':>10} {total_potential_units:>20,} (capped: {capped_units:,})")
    
    # Weighted analysis
    hh_with_weights = households.set_index('SERIALNO')
    adults_with_hh = adults.merge(hh_with_weights[['WGTP']], left_on='SERIALNO', right_index=True, how='left')
    
    logger.info(f"\n\nWeighted Analysis:")
    logger.info(f"{'Adults':<10} {'Weighted HH':>15} {'Potential Units':>18} {'Capped Units':>15} {'Lost Units':>15}")
    logger.info("-"*80)
    
    total_weighted_potential = 0
    total_weighted_capped = 0
    total_weighted_lost = 0
    
    for num_adults in sorted(adults_per_hh.unique()):
        hh_with_n_adults = adults_per_hh[adults_per_hh == num_adults].index
        weighted_hh = hh_with_weights.loc[hh_with_weights.index.isin(hh_with_n_adults), 'WGTP'].sum()
        
        potential = num_adults * weighted_hh
        capped = min(2, num_adults) * weighted_hh
        lost = potential - capped
        
        total_weighted_potential += potential
        total_weighted_capped += capped
        total_weighted_lost += lost
        
        logger.info(f"{num_adults:<10} {weighted_hh:>15,.0f} {potential:>18,.0f} {capped:>15,.0f} {lost:>15,.0f}")
    
    logger.info("-"*80)
    logger.info(f"{'TOTAL':<10} {hh_with_weights['WGTP'].sum():>15,.0f} {total_weighted_potential:>18,.0f} {total_weighted_capped:>15,.0f} {total_weighted_lost:>15,.0f}")
    
    logger.info(f"\n\nHousehold Cap Impact:")
    logger.info(f"  Potential tax units (if no cap):  {total_weighted_potential:>12,.0f}")
    logger.info(f"  Capped tax units (max 2 per HH):  {total_weighted_capped:>12,.0f}")
    logger.info(f"  Lost due to cap:                  {total_weighted_lost:>12,.0f} ({total_weighted_lost/total_weighted_potential*100:.1f}%)")
    
    # Check households with >2 adults
    large_hh = adults_per_hh[adults_per_hh > 2]
    logger.info(f"\n\nHouseholds with >2 Adults:")
    logger.info(f"  Count (unweighted): {len(large_hh):,}")
    logger.info(f"  Count (weighted):   {hh_with_weights.loc[hh_with_weights.index.isin(large_hh.index), 'WGTP'].sum():,.0f}")
    logger.info(f"  % of all households: {len(large_hh) / len(adults_per_hh) * 100:.1f}%")
    
    return {
        'potential_units': total_weighted_potential,
        'capped_units': total_weighted_capped,
        'lost_units': total_weighted_lost
    }


def analyze_dependent_classification():
    """Analyze dependent classification and potential misclassification."""
    logger.info("\n" + "="*80)
    logger.info("3. DEPENDENT CLASSIFICATION ANALYSIS")
    logger.info("="*80)
    
    # Load tax units
    tax_units_file = sorted(Path('data/processed').glob('tax_units_calibrated_*.parquet'))[-1]
    logger.info(f"\nLoading: {tax_units_file.name}")
    tax_units = pd.read_parquet(tax_units_file)
    
    # Load PUMS data
    persons = pd.read_csv('data/raw/pums/psam_p15.csv')
    
    # Count dependents in tax units
    total_dependents = tax_units['num_dependents'].sum()
    weighted_dependents = (tax_units['num_dependents'] * tax_units['weight']).sum()
    
    logger.info(f"\nDependents in Tax Units:")
    logger.info(f"  Total dependents (unweighted): {total_dependents:,}")
    logger.info(f"  Total dependents (weighted):   {weighted_dependents:,.0f}")
    
    # Count children in PUMS
    children = persons[persons['AGEP'] < 18]
    total_children = len(children)
    weighted_children = children['PWGTP'].sum()
    
    logger.info(f"\nChildren in PUMS (<18):")
    logger.info(f"  Total children (unweighted): {total_children:,}")
    logger.info(f"  Total children (weighted):   {weighted_children:,.0f}")
    
    # Check if dependents match children
    logger.info(f"\nComparison:")
    logger.info(f"  Dependents in tax units: {weighted_dependents:>12,.0f}")
    logger.info(f"  Children in PUMS:        {weighted_children:>12,.0f}")
    logger.info(f"  Difference:              {weighted_children - weighted_dependents:>12,.0f}")
    logger.info(f"  Coverage rate:           {weighted_dependents / weighted_children:>11.1%}")
    
    # Analyze adults who might be misclassified
    adults = persons[persons['AGEP'] >= 18].copy()
    
    # Young adults (18-24) who might be dependents
    young_adults = adults[adults['AGEP'].between(18, 24)]
    weighted_young_adults = young_adults['PWGTP'].sum()
    
    logger.info(f"\nYoung Adults (18-24):")
    logger.info(f"  Count (unweighted): {len(young_adults):,}")
    logger.info(f"  Count (weighted):   {weighted_young_adults:,.0f}")
    logger.info(f"  % of adults:        {weighted_young_adults / adults['PWGTP'].sum() * 100:.1f}%")
    
    # Check relationship codes for young adults
    logger.info(f"\nYoung Adults by Relationship to Householder:")
    rel_counts = young_adults.groupby('RELSHIPP')['PWGTP'].sum().sort_values(ascending=False)
    
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
        30: 'Noninstitutionalized'
    }
    
    for rel_code, count in rel_counts.items():
        rel_name = rel_codes.get(rel_code, f'Unknown ({rel_code})')
        pct = count / weighted_young_adults * 100
        logger.info(f"  {rel_name:<30} {count:>12,.0f} ({pct:>5.1f}%)")
    
    # Adults who are children of householder (RELSHIPP=23)
    adult_children = adults[adults['RELSHIPP'] == 23]
    weighted_adult_children = adult_children['PWGTP'].sum()
    
    logger.info(f"\nAdult Children of Householder (RELSHIPP=23):")
    logger.info(f"  Count (unweighted): {len(adult_children):,}")
    logger.info(f"  Count (weighted):   {weighted_adult_children:,.0f}")
    logger.info(f"  % of adults:        {weighted_adult_children / adults['PWGTP'].sum() * 100:.1f}%")
    
    # Age distribution of adult children
    logger.info(f"\nAge Distribution of Adult Children:")
    for age_group in [(18, 21), (22, 24), (25, 29), (30, 39), (40, 100)]:
        age_min, age_max = age_group
        in_range = adult_children[adult_children['AGEP'].between(age_min, age_max)]
        weighted = in_range['PWGTP'].sum()
        pct = weighted / weighted_adult_children * 100 if weighted_adult_children > 0 else 0
        logger.info(f"  {age_min}-{age_max}: {weighted:>12,.0f} ({pct:>5.1f}%)")
    
    return {
        'weighted_dependents': weighted_dependents,
        'weighted_children': weighted_children,
        'weighted_young_adults': weighted_young_adults,
        'weighted_adult_children': weighted_adult_children
    }


def analyze_tax_unit_creation():
    """Analyze tax unit creation rates and adult assignment."""
    logger.info("\n" + "="*80)
    logger.info("4. TAX UNIT CREATION ANALYSIS")
    logger.info("="*80)
    
    # Load tax units
    tax_units_file = sorted(Path('data/processed').glob('tax_units_calibrated_*.parquet'))[-1]
    tax_units = pd.read_parquet(tax_units_file)
    
    # Load PUMS data
    persons = pd.read_csv('data/raw/pums/psam_p15.csv')
    adults = persons[persons['AGEP'] >= 18]
    
    total_adults = adults['PWGTP'].sum()
    
    logger.info(f"\nTax Unit Statistics:")
    logger.info(f"  Total tax units (unweighted): {len(tax_units):,}")
    logger.info(f"  Total tax units (weighted):   {tax_units['weight'].sum():,.0f}")
    
    # Count adults in tax units
    # Primary filers
    primary_filers = len(tax_units)
    weighted_primary = tax_units['weight'].sum()
    
    # Joint filers have 2 adults
    joint_filers = tax_units[tax_units['filing_status'] == 'married_filing_jointly']
    num_joint = len(joint_filers)
    weighted_joint = joint_filers['weight'].sum()
    
    # MFS filers have 2 adults (filing separately)
    mfs_filers = tax_units[tax_units['filing_status'] == 'married_filing_separately']
    num_mfs = len(mfs_filers)
    weighted_mfs = mfs_filers['weight'].sum()
    
    # Total adults in tax units
    adults_in_units = weighted_primary + weighted_joint + weighted_mfs
    
    logger.info(f"\nAdults in Tax Units:")
    logger.info(f"  Primary filers:                {weighted_primary:>12,.0f}")
    logger.info(f"  Joint filer spouses:           {weighted_joint:>12,.0f}")
    logger.info(f"  MFS filer spouses:             {weighted_mfs:>12,.0f}")
    logger.info(f"  Total adults in tax units:     {adults_in_units:>12,.0f}")
    logger.info(f"  Total adults in PUMS:          {total_adults:>12,.0f}")
    logger.info(f"  Unassigned adults:             {total_adults - adults_in_units:>12,.0f}")
    logger.info(f"  Adult assignment rate:         {adults_in_units / total_adults:>11.1%}")
    
    # Filing status breakdown
    logger.info(f"\nFiling Status Breakdown:")
    for status in ['single', 'married_filing_jointly', 'head_of_household', 'married_filing_separately']:
        count = len(tax_units[tax_units['filing_status'] == status])
        weighted = tax_units[tax_units['filing_status'] == status]['weight'].sum()
        pct = weighted / tax_units['weight'].sum() * 100
        
        # Adults per tax unit
        if status == 'married_filing_jointly' or status == 'married_filing_separately':
            adults_per_unit = 2
        else:
            adults_per_unit = 1
        
        total_adults_status = weighted * adults_per_unit
        
        status_name = status.replace('_', ' ').title()
        logger.info(f"  {status_name:<30} {weighted:>12,.0f} ({pct:>5.1f}%) - {total_adults_status:>12,.0f} adults")
    
    return {
        'adults_in_units': adults_in_units,
        'total_adults': total_adults,
        'unassigned_adults': total_adults - adults_in_units
    }


def generate_recommendations(pop_stats, cap_stats, dep_stats, unit_stats):
    """Generate recommendations based on analysis."""
    logger.info("\n" + "="*80)
    logger.info("5. RECOMMENDATIONS")
    logger.info("="*80)
    
    soi_total = pop_stats['soi_total']
    current_total = pop_stats['current_total']
    gap = soi_total - current_total
    
    logger.info(f"\nCoverage Gap Breakdown:")
    logger.info(f"  SOI target:           {soi_total:>12,}")
    logger.info(f"  Current tax units:    {current_total:>12,}")
    logger.info(f"  Gap:                  {gap:>12,} ({gap/soi_total*100:.1f}%)")
    
    # Analyze sources of gap
    logger.info(f"\n\nPotential Sources of Gap:")
    
    # 1. Household cap
    lost_to_cap = cap_stats['lost_units']
    logger.info(f"\n1. Household Cap (max 2 per HH):")
    logger.info(f"   Lost tax units: {lost_to_cap:>12,.0f} ({lost_to_cap/gap*100:.1f}% of gap)")
    if lost_to_cap > gap * 0.1:
        logger.info(f"   ⚠️  SIGNIFICANT - Consider removing or increasing cap")
    else:
        logger.info(f"   ✅ MINOR - Cap is not a major issue")
    
    # 2. Unassigned adults
    unassigned = unit_stats['unassigned_adults']
    logger.info(f"\n2. Unassigned Adults:")
    logger.info(f"   Unassigned adults: {unassigned:>12,.0f} ({unassigned/gap*100:.1f}% of gap)")
    if unassigned > gap * 0.5:
        logger.info(f"   ⚠️  CRITICAL - Many adults not assigned to tax units")
        logger.info(f"   → Review tax unit construction logic")
        logger.info(f"   → Check if adults are incorrectly classified as dependents")
    else:
        logger.info(f"   ✅ GOOD - Most adults assigned to tax units")
    
    # 3. PUMS coverage
    total_adults = pop_stats['total_adults']
    pums_gap = soi_total - total_adults
    if pums_gap > 0:
        logger.info(f"\n3. PUMS Population Coverage:")
        logger.info(f"   PUMS adults:       {total_adults:>12,.0f}")
        logger.info(f"   SOI filers:        {soi_total:>12,}")
        logger.info(f"   PUMS shortfall:    {pums_gap:>12,} ({pums_gap/gap*100:.1f}% of gap)")
        logger.info(f"   ⚠️  CRITICAL - PUMS has fewer adults than SOI filers!")
        logger.info(f"   → This is impossible - each filer must be at least one adult")
        logger.info(f"   → Likely data year mismatch (PUMS 2015 vs SOI 2022)")
    else:
        surplus = total_adults - soi_total
        logger.info(f"\n3. PUMS Population Coverage:")
        logger.info(f"   PUMS adults:       {total_adults:>12,.0f}")
        logger.info(f"   SOI filers:        {soi_total:>12,}")
        logger.info(f"   PUMS surplus:      {surplus:>12,.0f}")
        logger.info(f"   ✅ GOOD - PUMS has enough adults")
    
    # 4. Adult children
    adult_children = dep_stats['weighted_adult_children']
    logger.info(f"\n4. Adult Children (RELSHIPP=23):")
    logger.info(f"   Adult children:    {adult_children:>12,.0f} ({adult_children/gap*100:.1f}% of gap)")
    logger.info(f"   → These adults should file their own returns")
    logger.info(f"   → Check if they're being excluded from tax unit creation")
    
    # Summary recommendations
    logger.info(f"\n\n" + "="*80)
    logger.info("RECOMMENDED ACTIONS")
    logger.info("="*80)
    
    actions = []
    
    if lost_to_cap > gap * 0.1:
        actions.append({
            'priority': 1,
            'action': 'Remove or increase household cap',
            'impact': f'{lost_to_cap:,.0f} additional tax units',
            'effort': 'Low - simple code change'
        })
    
    if unassigned > gap * 0.5:
        actions.append({
            'priority': 1,
            'action': 'Fix adult assignment logic',
            'impact': f'{unassigned:,.0f} additional tax units',
            'effort': 'Medium - requires debugging'
        })
    
    if adult_children > gap * 0.2:
        actions.append({
            'priority': 2,
            'action': 'Ensure adult children create tax units',
            'impact': f'{adult_children:,.0f} potential tax units',
            'effort': 'Medium - review construction logic'
        })
    
    if pums_gap > 0:
        actions.append({
            'priority': 3,
            'action': 'Address data year mismatch',
            'impact': f'{pums_gap:,.0f} missing adults',
            'effort': 'High - requires newer PUMS data or scaling'
        })
    
    if not actions:
        actions.append({
            'priority': 1,
            'action': 'Apply scaling factor',
            'impact': f'Scale all weights by {soi_total/current_total:.3f}',
            'effort': 'Low - simple multiplication'
        })
    
    for i, action in enumerate(actions, 1):
        logger.info(f"\n{i}. {action['action']} (Priority {action['priority']})")
        logger.info(f"   Impact: {action['impact']}")
        logger.info(f"   Effort: {action['effort']}")


def main():
    """Main execution."""
    logger.info("="*80)
    logger.info("COVERAGE GAP INVESTIGATION")
    logger.info("="*80)
    logger.info("\nAnalyzing why we're missing 26.6% of expected filers...")
    
    # Run analyses
    pop_stats = analyze_pums_population()
    cap_stats = analyze_household_cap()
    dep_stats = analyze_dependent_classification()
    unit_stats = analyze_tax_unit_creation()
    
    # Generate recommendations
    generate_recommendations(pop_stats, cap_stats, dep_stats, unit_stats)
    
    logger.info("\n" + "="*80)
    logger.info("INVESTIGATION COMPLETE")
    logger.info("="*80)


if __name__ == '__main__':
    main()
