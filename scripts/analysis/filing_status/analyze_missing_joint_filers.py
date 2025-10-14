"""
Analyze why we're missing joint filers by comparing PUMS married couples to tax units.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import logging
from src.data.pums_loader import PUMSDataLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_missing_joint_filers():
    """Analyze the gap between expected and actual joint filers."""
    
    # Load PUMS data
    logger.info("Loading PUMS data...")
    loader = PUMSDataLoader()
    person_data, household_data = loader.load_data()
    
    # Load tax units
    logger.info("Loading tax units...")
    tax_units = pd.read_parquet('src/data/processed/tax_units_irs_based.parquet')
    
    logger.info(f"Total tax units: {len(tax_units):,}")
    
    # Count by filing status
    status_counts = tax_units['filing_status'].value_counts()
    
    logger.info("\n" + "="*80)
    logger.info("CURRENT FILING STATUS DISTRIBUTION")
    logger.info("="*80)
    
    for status, count in status_counts.items():
        pct = 100 * count / len(tax_units)
        logger.info(f"  {status}: {count:,} ({pct:.1f}%)")
    
    # Calculate gaps
    total_units = len(tax_units)
    current_joint = status_counts.get('married_filing_jointly', 0)
    current_mfs = status_counts.get('married_filing_separately', 0)
    current_single = status_counts.get('single', 0)
    current_hoh = status_counts.get('head_of_household', 0)
    
    # SOI targets
    target_joint_pct = 36.0
    target_single_pct = 51.0
    target_hoh_pct = 9.6
    target_mfs_pct = 3.4
    
    expected_joint = int(total_units * target_joint_pct / 100)
    expected_single = int(total_units * target_single_pct / 100)
    expected_hoh = int(total_units * target_hoh_pct / 100)
    expected_mfs = int(total_units * target_mfs_pct / 100)
    
    logger.info("\n" + "="*80)
    logger.info("GAP ANALYSIS")
    logger.info("="*80)
    
    logger.info(f"\nJoint Filers:")
    logger.info(f"  Current: {current_joint:,} ({100*current_joint/total_units:.1f}%)")
    logger.info(f"  Expected: {expected_joint:,} ({target_joint_pct:.1f}%)")
    logger.info(f"  Gap: {expected_joint - current_joint:,} ({target_joint_pct - 100*current_joint/total_units:.1f}pp)")
    
    logger.info(f"\nSingle Filers:")
    logger.info(f"  Current: {current_single:,} ({100*current_single/total_units:.1f}%)")
    logger.info(f"  Expected: {expected_single:,} ({target_single_pct:.1f}%)")
    logger.info(f"  Gap: {current_single - expected_single:,} ({100*current_single/total_units - target_single_pct:.1f}pp)")
    
    logger.info(f"\nHead of Household:")
    logger.info(f"  Current: {current_hoh:,} ({100*current_hoh/total_units:.1f}%)")
    logger.info(f"  Expected: {expected_hoh:,} ({target_hoh_pct:.1f}%)")
    logger.info(f"  Gap: {current_hoh - expected_hoh:,} ({100*current_hoh/total_units - target_hoh_pct:.1f}pp)")
    
    logger.info(f"\nMarried Filing Separately:")
    logger.info(f"  Current: {current_mfs:,} ({100*current_mfs/total_units:.1f}%)")
    logger.info(f"  Expected: {expected_mfs:,} ({target_mfs_pct:.1f}%)")
    logger.info(f"  Gap: {current_mfs - expected_mfs:,} ({100*current_mfs/total_units - target_mfs_pct:.1f}pp)")
    
    # Analyze PUMS married couples
    logger.info("\n" + "="*80)
    logger.info("PUMS MARRIED COUPLES ANALYSIS")
    logger.info("="*80)
    
    merged = person_data.merge(
        household_data[['SERIALNO', 'WGTP']],
        on='SERIALNO',
        how='left'
    )
    
    # Count married adults
    married_adults = merged[(merged['AGEP'] >= 18) & (merged['MAR'] == 1)]
    logger.info(f"\nTotal married adults in PUMS: {len(married_adults):,}")
    
    # Count by relationship
    rel_counts = married_adults['RELSHIPP'].value_counts()
    logger.info(f"\nMarried adults by relationship:")
    logger.info(f"  Householder (20): {rel_counts.get(20, 0):,}")
    logger.info(f"  Spouse (21): {rel_counts.get(21, 0):,}")
    logger.info(f"  Other: {len(married_adults) - rel_counts.get(20, 0) - rel_counts.get(21, 0):,}")
    
    # Estimate married couples (householder + spouse pairs)
    married_householders = rel_counts.get(20, 0)
    married_spouses = rel_counts.get(21, 0)
    estimated_couples = min(married_householders, married_spouses)
    
    logger.info(f"\nEstimated married couples (householder+spouse pairs): {estimated_couples:,}")
    
    # Compare to tax units
    total_married_units = current_joint + current_mfs
    logger.info(f"Married tax units (joint + MFS): {total_married_units:,}")
    logger.info(f"Difference: {total_married_units - estimated_couples:,}")
    
    if total_married_units > estimated_couples:
        logger.info(f"⚠️  We're creating {total_married_units - estimated_couples:,} MORE married units than couples!")
        logger.info("This suggests we're incorrectly pairing some married adults")
    elif total_married_units < estimated_couples:
        logger.info(f"⚠️  We're missing {estimated_couples - total_married_units:,} married couples!")
        logger.info("This suggests we're not identifying all married couples")
    
    # Recommendations
    logger.info("\n" + "="*80)
    logger.info("RECOMMENDATIONS")
    logger.info("="*80)
    
    joint_gap = expected_joint - current_joint
    single_excess = current_single - expected_single
    
    logger.info(f"\nTo reach SOI targets:")
    logger.info(f"  1. Need {joint_gap:,} more joint filers")
    logger.info(f"  2. Have {single_excess:,} excess single filers")
    logger.info(f"  3. MFS is {current_mfs - expected_mfs:,} below target")
    
    logger.info(f"\nPossible solutions:")
    if joint_gap > 0 and single_excess > 0:
        logger.info(f"  • Convert ~{min(joint_gap, single_excess):,} single filers to joint filers")
        logger.info(f"    (requires identifying more married couples)")
    
    if current_mfs < expected_mfs:
        logger.info(f"  • Increase MFS rate from {100*current_mfs/total_units:.1f}% to {target_mfs_pct:.1f}%")
        logger.info(f"    (requires converting {expected_mfs - current_mfs:,} joint filers to MFS)")
    
    # Check if the issue is identification or classification
    logger.info("\n" + "="*80)
    logger.info("ROOT CAUSE ANALYSIS")
    logger.info("="*80)
    
    if total_married_units < estimated_couples:
        missing_couples = estimated_couples - total_married_units
        logger.info(f"\n🔍 PRIMARY ISSUE: Missing {missing_couples:,} married couples")
        logger.info(f"   These couples exist in PUMS but aren't being identified")
        logger.info(f"   Likely causes:")
        logger.info(f"   - Second pass in _identify_joint_filers() is too restrictive")
        logger.info(f"   - Some householder+spouse pairs aren't being matched")
        logger.info(f"   - Age/sex criteria are filtering out valid couples")
    else:
        logger.info(f"\n✅ Married couple identification is working well")
        logger.info(f"   Issue is classification (joint vs MFS vs single)")
        logger.info(f"   Need to adjust MFS thresholds to balance joint/MFS rates")

if __name__ == '__main__':
    analyze_missing_joint_filers()
