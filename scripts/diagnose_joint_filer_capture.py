#!/usr/bin/env python3
"""
Diagnose the joint filer capture rate issue (106% of married couples).
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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_married_couples():
    """Analyze married couples in PUMS data."""
    logger.info("Loading PUMS data for married couple analysis...")
    
    data_loader = PUMSDataLoader()
    person_df, hh_df = data_loader.load_data()
    
    print("Married Couple Analysis")
    print("=" * 50)
    
    # Total married adults
    married_adults = person_df[person_df['MAR'] == 1]
    total_married_adults = len(married_adults)
    print(f"Total married adults: {total_married_adults:,}")
    
    # Group married adults by household
    married_by_household = married_adults.groupby('SERIALNO')
    
    household_married_counts = []
    for hh_id, hh_married in married_by_household:
        household_married_counts.append({
            'household_id': hh_id,
            'married_adults': len(hh_married),
            'adult_ids': list(hh_married.index)
        })
    
    hh_married_df = pd.DataFrame(household_married_counts)
    
    print(f"\nHouseholds with married adults: {len(hh_married_df):,}")
    
    # Distribution of married adults per household
    married_dist = hh_married_df['married_adults'].value_counts().sort_index()
    print(f"\nMarried adults per household:")
    for count, households in married_dist.items():
        print(f"  {count} married adult(s): {households:,} households")
    
    # Expected married couples (households with exactly 2 married adults)
    couples_hh = hh_married_df[hh_married_df['married_adults'] == 2]
    expected_couples = len(couples_hh)
    print(f"\nExpected married couples (2 married adults in same household): {expected_couples:,}")
    
    # Single married adults (spouse not in household)
    single_married_hh = hh_married_df[hh_married_df['married_adults'] == 1]
    single_married_adults = len(single_married_hh)
    print(f"Single married adults (spouse not in household): {single_married_adults:,}")
    
    # Unusual cases (3+ married adults in one household)
    unusual_hh = hh_married_df[hh_married_df['married_adults'] >= 3]
    if len(unusual_hh) > 0:
        print(f"Households with 3+ married adults: {len(unusual_hh):,}")
        total_unusual_adults = unusual_hh['married_adults'].sum()
        print(f"  Total adults in these households: {total_unusual_adults:,}")
    
    return person_df, hh_df, couples_hh, single_married_hh

def analyze_joint_filer_creation():
    """Analyze how joint filers are being created."""
    logger.info("Loading constructed tax units...")
    
    # Load the IRS-based tax units
    tax_units_file = Path("src/data/processed/tax_units_irs_based.parquet")
    if not tax_units_file.exists():
        logger.error("Tax units file not found. Please run tax unit construction first.")
        return None
    
    tax_units = pd.read_parquet(tax_units_file)
    
    print("\nJoint Filer Creation Analysis")
    print("=" * 50)
    
    # Count joint filers
    joint_units = tax_units[tax_units['filing_status'].isin(['joint', 'married_filing_jointly'])]
    total_joint_units = len(joint_units)
    total_joint_adults = total_joint_units * 2  # Each joint unit represents 2 adults
    
    print(f"Joint filer tax units: {total_joint_units:,}")
    print(f"Adults in joint filer units: {total_joint_adults:,}")
    
    # Breakdown by specific status
    joint_breakdown = joint_units['filing_status'].value_counts()
    print(f"\nJoint filer breakdown:")
    for status, count in joint_breakdown.items():
        print(f"  {status}: {count:,} units ({count * 2:,} adults)")
    
    return joint_units, total_joint_units

def calculate_capture_rates():
    """Calculate and explain the joint filer capture rate."""
    print("\nJoint Filer Capture Rate Analysis")
    print("=" * 50)
    
    # Get married couple data
    person_df, hh_df, couples_hh, single_married_hh = analyze_married_couples()
    
    # Get joint filer data
    joint_data = analyze_joint_filer_creation()
    if joint_data is None:
        return
    
    joint_units, total_joint_units = joint_data
    
    # Expected couples (2 married adults in same household)
    expected_couples = len(couples_hh)
    
    # Capture rate calculation
    capture_rate = (total_joint_units / expected_couples) * 100 if expected_couples > 0 else 0
    
    print(f"Expected married couples: {expected_couples:,}")
    print(f"Joint filer tax units: {total_joint_units:,}")
    print(f"Joint filer capture rate: {capture_rate:.1f}%")
    
    if capture_rate > 100:
        excess = total_joint_units - expected_couples
        print(f"\nEXCESS JOINT FILERS: {excess:,} more joint units than expected couples")
        print(f"This suggests:")
        print(f"  1. Some unmarried adults are being classified as joint filers")
        print(f"  2. Some single married adults (spouse not in household) are being paired")
        print(f"  3. Logic errors in spouse identification or joint filer creation")
    
    # Check for impossible joint filers
    print(f"\nPotential Issues:")
    
    # Single married adults who might be incorrectly paired
    single_married_adults = len(single_married_hh)
    if single_married_adults > 0:
        print(f"  - {single_married_adults:,} single married adults (spouse not in household)")
        print(f"    These should NOT be creating joint filers with non-spouses")
    
    # Check if we have more joint adults than total married adults
    total_married_adults = len(person_df[person_df['MAR'] == 1])
    joint_adults = total_joint_units * 2
    
    if joint_adults > total_married_adults:
        excess_adults = joint_adults - total_married_adults
        print(f"  - {excess_adults:,} more adults in joint units than total married adults!")
        print(f"    This means unmarried adults are being classified as joint filers")
    
    return expected_couples, total_joint_units, capture_rate

def analyze_filing_status_logic_issues():
    """Analyze potential issues in filing status assignment logic."""
    print("\nFiling Status Logic Issues Analysis")
    print("=" * 50)
    
    person_df, hh_df, couples_hh, single_married_hh = analyze_married_couples()
    
    # Load tax units
    tax_units_file = Path("src/data/processed/tax_units_irs_based.parquet")
    tax_units = pd.read_parquet(tax_units_file)
    
    # Count adults by marital status who should be single filers
    unmarried_adults = person_df[person_df['MAR'] != 1]
    total_unmarried = len(unmarried_adults)
    
    # Count actual single filers
    single_units = len(tax_units[tax_units['filing_status'] == 'single'])
    
    print(f"Unmarried adults (should mostly be single filers): {total_unmarried:,}")
    print(f"Actual single filer tax units: {single_units:,}")
    print(f"Missing single filers: {total_unmarried - single_units:,}")
    
    # Where did the missing single filers go?
    print(f"\nWhere did the missing single filers go?")
    
    # Joint filers (should be 0 unmarried adults)
    joint_count = len(tax_units[tax_units['filing_status'].isin(['joint', 'married_filing_jointly'])])
    joint_adults = joint_count * 2
    unmarried_in_joint = max(0, joint_adults - len(person_df[person_df['MAR'] == 1]))
    
    if unmarried_in_joint > 0:
        print(f"  - {unmarried_in_joint:,} unmarried adults incorrectly classified as joint filers")
    
    # Head of household (could include unmarried adults)
    hoh_count = len(tax_units[tax_units['filing_status'] == 'head_of_household'])
    print(f"  - {hoh_count:,} adults classified as head of household")
    
    # Married filing separately
    mfs_count = len(tax_units[tax_units['filing_status'] == 'married_filing_separately'])
    print(f"  - {mfs_count:,} adults classified as married filing separately")
    
    # Check the math
    accounted_adults = single_units + joint_adults + hoh_count + mfs_count
    total_adults = len(person_df[person_df['AGEP'] >= 18])
    
    print(f"\nAdult accounting:")
    print(f"  Total adults (18+): {total_adults:,}")
    print(f"  Adults in tax units: {accounted_adults:,}")
    print(f"  Difference: {accounted_adults - total_adults:,}")
    
    if accounted_adults > total_adults:
        print(f"  WARNING: More adults in tax units than exist in data!")

def main():
    """Main analysis function."""
    logger.info("Starting joint filer capture rate diagnostic...")
    
    expected_couples, total_joint_units, capture_rate = calculate_capture_rates()
    analyze_filing_status_logic_issues()
    
    print(f"\nKey Findings:")
    print(f"1. Joint filer capture rate: {capture_rate:.1f}% (should be ≤100%)")
    if capture_rate > 100:
        print(f"2. CRITICAL: {total_joint_units - expected_couples:,} excess joint filers detected")
        print(f"3. Root cause: Unmarried adults being incorrectly classified as joint filers")
    print(f"4. This explains the single filer shortage (unmarried adults misclassified)")
    
    print(f"\nNext Steps:")
    print(f"1. Review spouse identification logic (_are_married method)")
    print(f"2. Check joint filer creation logic (_create_joint_filer method)")
    print(f"3. Verify marital status checks in filing status assignment")
    print(f"4. Fix logic to ensure only married couples can file jointly")

if __name__ == "__main__":
    main()
