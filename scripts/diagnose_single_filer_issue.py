#!/usr/bin/env python3
"""
Diagnose the single filer classification issue in tax unit construction.
"""
import sys
import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.pums_loader import PUMSDataLoader
from src.tax.units.constructor import TaxUnitConstructor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_person_demographics():
    """Analyze the demographics of persons in PUMS data."""
    logger.info("Loading PUMS data for demographic analysis...")
    
    data_loader = PUMSDataLoader()
    person_df, hh_df = data_loader.load_data()
    
    print("PUMS Person Demographics Analysis")
    print("=" * 50)
    
    # Total persons
    total_persons = len(person_df)
    print(f"Total persons: {total_persons:,}")
    
    # Age distribution
    print(f"\nAge Distribution:")
    print(f"  Under 18: {len(person_df[person_df['AGEP'] < 18]):,} ({len(person_df[person_df['AGEP'] < 18])/total_persons*100:.1f}%)")
    print(f"  18-64: {len(person_df[(person_df['AGEP'] >= 18) & (person_df['AGEP'] < 65)]):,} ({len(person_df[(person_df['AGEP'] >= 18) & (person_df['AGEP'] < 65)])/total_persons*100:.1f}%)")
    print(f"  65+: {len(person_df[person_df['AGEP'] >= 65]):,} ({len(person_df[person_df['AGEP'] >= 65])/total_persons*100:.1f}%)")
    
    # Adults (potential tax filers)
    adults = person_df[person_df['AGEP'] >= 18]
    total_adults = len(adults)
    print(f"\nTotal adults (18+): {total_adults:,}")
    
    # Marital status of adults
    print(f"\nMarital Status of Adults:")
    mar_counts = adults['MAR'].value_counts().sort_index()
    for mar_code, count in mar_counts.items():
        pct = count / total_adults * 100
        mar_desc = {1: "Married", 2: "Widowed", 3: "Divorced", 4: "Separated", 5: "Never married"}
        print(f"  {mar_desc.get(mar_code, f'Code {mar_code}')}: {count:,} ({pct:.1f}%)")
    
    # Expected single filers (never married + widowed + divorced + separated)
    potential_single = adults[adults['MAR'].isin([2, 3, 4, 5])]
    print(f"\nPotential Single Filers (unmarried adults): {len(potential_single):,} ({len(potential_single)/total_adults*100:.1f}%)")
    
    # Married adults
    married_adults = adults[adults['MAR'] == 1]
    print(f"Married adults: {len(married_adults):,} ({len(married_adults)/total_adults*100:.1f}%)")
    print(f"Estimated married couples: {len(married_adults)/2:.0f}")
    
    return person_df, hh_df, adults

def analyze_tax_unit_assignments():
    """Analyze how adults are being assigned to tax units."""
    logger.info("Loading constructed tax units...")
    
    # Load the IRS-based tax units
    tax_units_file = Path("src/data/processed/tax_units_irs_based.parquet")
    if not tax_units_file.exists():
        logger.error("Tax units file not found. Please run tax unit construction first.")
        return None
    
    tax_units = pd.read_parquet(tax_units_file)
    
    print("\nTax Unit Assignment Analysis")
    print("=" * 50)
    
    # Filing status distribution
    status_counts = tax_units['filing_status'].value_counts()
    total_units = len(tax_units)
    
    print(f"Total tax units: {total_units:,}")
    print(f"\nFiling Status Distribution:")
    for status, count in status_counts.items():
        pct = count / total_units * 100
        print(f"  {status}: {count:,} ({pct:.1f}%)")
    
    # Count adults in each filing status
    print(f"\nAdults by Filing Status:")
    
    # Single filers: 1 adult each
    single_count = status_counts.get('single', 0)
    single_adults = single_count * 1
    
    # Joint filers: 2 adults each (married couples)
    joint_count = status_counts.get('joint', 0) + status_counts.get('married_filing_jointly', 0)
    joint_adults = joint_count * 2
    
    # Head of household: 1 adult each
    hoh_count = status_counts.get('head_of_household', 0)
    hoh_adults = hoh_count * 1
    
    # Married filing separately: 1 adult each (but they're married)
    mfs_count = status_counts.get('married_filing_separately', 0)
    mfs_adults = mfs_count * 1
    
    total_adults_in_units = single_adults + joint_adults + hoh_adults + mfs_adults
    
    print(f"  Single filers: {single_adults:,} adults")
    print(f"  Joint filers: {joint_adults:,} adults")
    print(f"  Head of household: {hoh_adults:,} adults")
    print(f"  Married filing separately: {mfs_adults:,} adults")
    print(f"  Total adults in tax units: {total_adults_in_units:,}")
    
    return tax_units

def analyze_household_structures():
    """Analyze household structures to understand single filer potential."""
    logger.info("Analyzing household structures...")
    
    data_loader = PUMSDataLoader()
    person_df, hh_df = data_loader.load_data()
    
    print("\nHousehold Structure Analysis")
    print("=" * 50)
    
    # Group by household
    households = person_df.groupby('SERIALNO')
    
    # Analyze household compositions
    household_types = []
    
    for hh_id, hh_persons in households:
        adults = hh_persons[hh_persons['AGEP'] >= 18]
        children = hh_persons[hh_persons['AGEP'] < 18]
        
        num_adults = len(adults)
        num_children = len(children)
        
        # Count married adults
        married_adults = adults[adults['MAR'] == 1]
        unmarried_adults = adults[adults['MAR'] != 1]
        
        household_types.append({
            'household_id': hh_id,
            'total_persons': len(hh_persons),
            'adults': num_adults,
            'children': num_children,
            'married_adults': len(married_adults),
            'unmarried_adults': len(unmarried_adults)
        })
    
    hh_analysis = pd.DataFrame(household_types)
    
    print(f"Total households: {len(hh_analysis):,}")
    
    # Household types that should produce single filers
    single_adult_hh = hh_analysis[hh_analysis['adults'] == 1]
    print(f"\nSingle adult households: {len(single_adult_hh):,}")
    
    single_adult_unmarried = single_adult_hh[single_adult_hh['unmarried_adults'] == 1]
    print(f"  - Unmarried single adults: {len(single_adult_unmarried):,}")
    
    single_adult_married = single_adult_hh[single_adult_hh['married_adults'] == 1]
    print(f"  - Married single adults (spouse not in household): {len(single_adult_married):,}")
    
    # Multi-adult households with unmarried adults
    multi_adult_hh = hh_analysis[hh_analysis['adults'] > 1]
    print(f"\nMulti-adult households: {len(multi_adult_hh):,}")
    
    # Count unmarried adults in multi-adult households
    multi_adult_with_unmarried = multi_adult_hh[multi_adult_hh['unmarried_adults'] > 0]
    total_unmarried_in_multi = multi_adult_with_unmarried['unmarried_adults'].sum()
    print(f"  - Unmarried adults in multi-adult households: {total_unmarried_in_multi:,}")
    
    # Expected single filers
    expected_single_filers = len(single_adult_unmarried) + total_unmarried_in_multi
    print(f"\nExpected single filers: {expected_single_filers:,}")
    
    return hh_analysis

def compare_expectations_vs_reality():
    """Compare expected vs actual single filer counts."""
    print("\nExpectation vs Reality Comparison")
    print("=" * 50)
    
    # Load data
    person_df, hh_df, adults = analyze_person_demographics()
    tax_units = analyze_tax_unit_assignments()
    hh_analysis = analyze_household_structures()
    
    if tax_units is None:
        return
    
    # Expected single filers from demographic analysis
    potential_single = adults[adults['MAR'].isin([2, 3, 4, 5])]
    expected_from_demographics = len(potential_single)
    
    # Expected single filers from household analysis
    single_adult_unmarried = hh_analysis[(hh_analysis['adults'] == 1) & (hh_analysis['unmarried_adults'] == 1)]
    multi_adult_with_unmarried = hh_analysis[hh_analysis['adults'] > 1]
    total_unmarried_in_multi = multi_adult_with_unmarried['unmarried_adults'].sum()
    expected_from_households = len(single_adult_unmarried) + total_unmarried_in_multi
    
    # Actual single filers from tax units
    actual_single = tax_units['filing_status'].value_counts().get('single', 0)
    
    print(f"Expected single filers (demographics): {expected_from_demographics:,}")
    print(f"Expected single filers (households): {expected_from_households:,}")
    print(f"Actual single filers (tax units): {actual_single:,}")
    
    print(f"\nDiscrepancy Analysis:")
    demo_gap = expected_from_demographics - actual_single
    hh_gap = expected_from_households - actual_single
    
    print(f"  Demographics gap: {demo_gap:,} ({demo_gap/expected_from_demographics*100:.1f}% missing)")
    print(f"  Household gap: {hh_gap:,} ({hh_gap/expected_from_households*100:.1f}% missing)")
    
    # Where did the missing single filers go?
    print(f"\nWhere did the missing single filers go?")
    status_counts = tax_units['filing_status'].value_counts()
    
    joint_count = status_counts.get('joint', 0) + status_counts.get('married_filing_jointly', 0)
    hoh_count = status_counts.get('head_of_household', 0)
    mfs_count = status_counts.get('married_filing_separately', 0)
    
    print(f"  Joint filers: {joint_count:,} units ({joint_count * 2:,} adults)")
    print(f"  Head of household: {hoh_count:,} units ({hoh_count:,} adults)")
    print(f"  Married filing separately: {mfs_count:,} units ({mfs_count:,} adults)")

def main():
    """Main analysis function."""
    logger.info("Starting single filer classification diagnostic...")
    
    compare_expectations_vs_reality()
    
    print(f"\nNext Steps:")
    print(f"1. Review tax unit construction logic for single filer assignment")
    print(f"2. Check if unmarried adults are being incorrectly classified as HoH")
    print(f"3. Verify marital status logic and spouse identification")
    print(f"4. Examine edge cases in household processing")

if __name__ == "__main__":
    main()
