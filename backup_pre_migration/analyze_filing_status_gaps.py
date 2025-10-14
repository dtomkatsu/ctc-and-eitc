#!/usr/bin/env python3
"""
Comprehensive analysis of filing status gaps between PUMS and SOI benchmarks.

This script investigates the root causes of:
1. Too many single filers (61.2% vs 51.0% target)
2. Too few joint filers (25.3% vs 36.0% target)
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.pums_loader import PUMSDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_marital_status_distribution():
    """Analyze the distribution of marital status in PUMS data."""
    logger.info("Analyzing marital status distribution in PUMS data...")
    
    # Load PUMS data
    loader = PUMSDataLoader()
    person_df, hh_df = loader.load_data(state=15, year=2022)
    
    # Filter to adults only
    adults = person_df[person_df['AGEP'] >= 18].copy()
    
    print("\nMarital Status Distribution (Adults 18+)")
    print("=" * 50)
    
    # MAR codes: 1=Married, 2=Widowed, 3=Divorced, 4=Separated, 5=Never married
    mar_counts = adults['MAR'].value_counts().sort_index()
    total_adults = len(adults)
    
    mar_labels = {
        1: "Married",
        2: "Widowed", 
        3: "Divorced",
        4: "Separated",
        5: "Never married"
    }
    
    for mar_code, count in mar_counts.items():
        label = mar_labels.get(mar_code, f"Unknown ({mar_code})")
        pct = count / total_adults * 100
        print(f"  {label}: {count:,} ({pct:.1f}%)")
    
    print(f"\nTotal adults: {total_adults:,}")
    
    # Calculate expected filing patterns based on marital status
    married_adults = mar_counts.get(1, 0)
    unmarried_adults = total_adults - married_adults
    
    print(f"\nExpected Filing Patterns:")
    print(f"  Married adults: {married_adults:,} ({married_adults/total_adults*100:.1f}%)")
    print(f"  Unmarried adults: {unmarried_adults:,} ({unmarried_adults/total_adults*100:.1f}%)")
    
    # Estimate expected joint filers (assuming ~70-80% of married couples file jointly)
    estimated_couples = married_adults // 2  # Rough estimate
    estimated_joint_rate_range = (0.70, 0.80)
    
    print(f"\nEstimated married couples: ~{estimated_couples:,}")
    for rate in estimated_joint_rate_range:
        joint_units = int(estimated_couples * rate)
        joint_pct = joint_units / (total_adults * 0.6) * 100  # Rough tax unit estimate
        print(f"  If {rate*100:.0f}% file jointly: ~{joint_units:,} units (~{joint_pct:.1f}% of tax units)")
    
    return adults, hh_df

def analyze_household_composition():
    """Analyze household composition patterns that affect filing status."""
    logger.info("Analyzing household composition patterns...")
    
    # Load PUMS data
    loader = PUMSDataLoader()
    person_df, hh_df = loader.load_data(state=15, year=2022)
    
    # Merge person and household data
    merged = person_df.merge(hh_df, on='SERIALNO', how='left')
    adults = merged[merged['AGEP'] >= 18].copy()
    
    print("\nHousehold Composition Analysis")
    print("=" * 50)
    
    # Group by household and analyze composition
    hh_composition = []
    
    for serialno, hh_group in adults.groupby('SERIALNO'):
        hh_adults = len(hh_group)
        married_adults = len(hh_group[hh_group['MAR'] == 1])
        unmarried_adults = hh_adults - married_adults
        
        # Count relationship types
        relshipp_counts = hh_group['RELSHIPP'].value_counts()
        householders = relshipp_counts.get(20, 0)  # Householder
        spouses = relshipp_counts.get(21, 0)       # Spouse
        
        hh_composition.append({
            'serialno': serialno,
            'total_adults': hh_adults,
            'married_adults': married_adults,
            'unmarried_adults': unmarried_adults,
            'householders': householders,
            'spouses': spouses,
            'has_married_couple': (householders >= 1 and spouses >= 1),
            'multiple_married': married_adults > 2
        })
    
    comp_df = pd.DataFrame(hh_composition)
    
    print(f"Total households with adults: {len(comp_df):,}")
    print(f"Households with married couples (householder + spouse): {comp_df['has_married_couple'].sum():,}")
    print(f"Households with 3+ married adults: {comp_df['multiple_married'].sum():,}")
    
    # Analyze adult distribution
    adult_dist = comp_df['total_adults'].value_counts().sort_index()
    print(f"\nAdults per household:")
    for adults_count, hh_count in adult_dist.items():
        pct = hh_count / len(comp_df) * 100
        print(f"  {adults_count} adult(s): {hh_count:,} households ({pct:.1f}%)")
    
    # Analyze married adult distribution
    married_dist = comp_df['married_adults'].value_counts().sort_index()
    print(f"\nMarried adults per household:")
    for married_count, hh_count in married_dist.items():
        pct = hh_count / len(comp_df) * 100
        print(f"  {married_count} married adult(s): {hh_count:,} households ({pct:.1f}%)")
    
    return comp_df

def analyze_current_tax_units():
    """Analyze the current tax unit construction results."""
    logger.info("Analyzing current tax unit construction results...")
    
    # Load constructed tax units
    tax_units_path = project_root / "src/data/processed/tax_units_irs_based.parquet"
    if not tax_units_path.exists():
        logger.error(f"Tax units file not found: {tax_units_path}")
        return None
    
    tax_units = pd.read_parquet(tax_units_path)
    
    print("\nCurrent Tax Unit Analysis")
    print("=" * 50)
    
    # Filing status distribution
    status_counts = tax_units['filing_status'].value_counts()
    total_units = len(tax_units)
    
    print(f"Total tax units: {total_units:,}")
    print(f"\nFiling status distribution:")
    for status, count in status_counts.items():
        pct = count / total_units * 100
        print(f"  {status}: {count:,} ({pct:.1f}%)")
    
    # Analyze joint filers specifically
    joint_units = tax_units[tax_units['filing_status'] == 'married_filing_jointly']
    print(f"\nJoint filer analysis:")
    print(f"  Joint filer units: {len(joint_units):,}")
    print(f"  Adults in joint units: {len(joint_units) * 2:,}")  # Each joint unit has 2 adults
    
    # Analyze single filers
    single_units = tax_units[tax_units['filing_status'] == 'single']
    print(f"\nSingle filer analysis:")
    print(f"  Single filer units: {len(single_units):,}")
    
    # Check for potential misclassifications
    print(f"\nPotential misclassification indicators:")
    
    # Look at income patterns
    if 'income' in tax_units.columns:
        joint_median_income = joint_units['income'].median()
        single_median_income = single_units['income'].median()
        print(f"  Median income - Joint: ${joint_median_income:,.0f}, Single: ${single_median_income:,.0f}")
    
    # Look at household patterns
    if 'household_id' in tax_units.columns:
        # Count tax units per household
        units_per_hh = tax_units.groupby('household_id').size()
        multi_unit_hh = (units_per_hh > 1).sum()
        print(f"  Households with multiple tax units: {multi_unit_hh:,}")
        print(f"  Average tax units per household: {units_per_hh.mean():.2f}")
    
    return tax_units

def identify_gap_sources():
    """Identify specific sources of the filing status gaps."""
    logger.info("Identifying sources of filing status gaps...")
    
    print("\nGap Source Analysis")
    print("=" * 50)
    
    # Current vs target
    current_results = {
        'single': 61.2,
        'joint': 25.3,
        'hoh': 9.7,
        'mfs': 3.8
    }
    
    soi_targets = {
        'single': 51.0,
        'joint': 36.0,
        'hoh': 9.6,
        'mfs': 3.4
    }
    
    print("Filing Status Gaps:")
    for status in ['single', 'joint', 'hoh', 'mfs']:
        current = current_results[status]
        target = soi_targets[status]
        gap = current - target
        direction = "excess" if gap > 0 else "shortage"
        print(f"  {status.upper()}: {current:.1f}% vs {target:.1f}% target = {abs(gap):.1f}pp {direction}")
    
    # The main issue: need to convert ~5,000 single filers to joint filers
    total_units = 48262  # From previous run
    single_excess = (current_results['single'] - soi_targets['single']) / 100 * total_units
    joint_shortage = (soi_targets['joint'] - current_results['joint']) / 100 * total_units
    
    print(f"\nRequired adjustments (approximate):")
    print(f"  Reduce single filers by: ~{single_excess:,.0f} units")
    print(f"  Increase joint filers by: ~{joint_shortage:,.0f} units")
    print(f"  Net conversion needed: ~{(single_excess + joint_shortage) / 2:,.0f} units")
    
    # Potential sources of the gap
    print(f"\nPotential gap sources:")
    print(f"  1. Married couples incorrectly classified as single")
    print(f"  2. Single adults incorrectly not being paired with spouses")
    print(f"  3. MFS couples that should be filing jointly")
    print(f"  4. Complex households with misidentified relationships")
    print(f"  5. Data quality issues in PUMS relationship codes")

def main():
    """Run comprehensive gap analysis."""
    print("Hawaii PUMS Filing Status Gap Analysis")
    print("=" * 60)
    
    try:
        # 1. Analyze marital status distribution
        adults_df, hh_df = analyze_marital_status_distribution()
        
        # 2. Analyze household composition
        comp_df = analyze_household_composition()
        
        # 3. Analyze current tax units
        tax_units_df = analyze_current_tax_units()
        
        # 4. Identify gap sources
        identify_gap_sources()
        
        print(f"\nAnalysis complete! Key findings:")
        print(f"  - Need to convert ~5,000 single filers to joint filers")
        print(f"  - Focus on married couples incorrectly classified as single")
        print(f"  - Review spouse identification and joint filer creation logic")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()
