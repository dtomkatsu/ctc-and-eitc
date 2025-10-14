"""
Diagnostic script to investigate why joint filers are 6.5pp below target.

This script will:
1. Analyze married couples in PUMS data
2. Check how many are filing jointly vs separately
3. Identify why some married couples aren't filing jointly
4. Compare against SOI benchmarks
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

def analyze_joint_filer_gap():
    """Analyze why joint filers are below target."""
    
    # Load PUMS data
    logger.info("Loading PUMS data...")
    loader = PUMSDataLoader()
    person_data, household_data = loader.load_data()
    
    logger.info(f"Loaded {len(person_data)} persons and {len(household_data)} households")
    
    # Merge person and household data
    merged = person_data.merge(
        household_data[['SERIALNO', 'WGTP', 'PUMA']],
        on='SERIALNO',
        how='left'
    )
    
    # Filter to adults only
    adults = merged[merged['AGEP'] >= 18].copy()
    logger.info(f"Found {len(adults)} adults")
    
    # Count married adults
    married_adults = adults[adults['MAR'] == 1]  # MAR=1 is married
    logger.info(f"Found {len(married_adults)} married adults")
    
    # Count married couples (should be roughly half of married adults)
    # Group by household and count married adults
    married_by_household = married_adults.groupby('SERIALNO').size()
    
    # Households with 2 married adults = married couples
    married_couple_households = married_by_household[married_by_household == 2]
    num_married_couples = len(married_couple_households)
    
    # Households with 1 married adult = spouse absent
    spouse_absent_households = married_by_household[married_by_household == 1]
    num_spouse_absent = len(spouse_absent_households)
    
    # Households with 3+ married adults = complex households
    complex_households = married_by_household[married_by_household >= 3]
    num_complex = len(complex_households)
    
    logger.info(f"\nMarried Household Analysis:")
    logger.info(f"  Married couple households (2 married adults): {num_married_couples:,}")
    logger.info(f"  Spouse absent households (1 married adult): {num_spouse_absent:,}")
    logger.info(f"  Complex households (3+ married adults): {num_complex:,}")
    
    # Analyze relationship codes for married adults
    rel_counts = married_adults['RELSHIPP'].value_counts()
    logger.info(f"\nMarried Adults by Relationship Code:")
    rel_labels = {
        20: 'Householder',
        21: 'Spouse',
        22: 'Biological child',
        23: 'Adopted child',
        24: 'Stepchild',
        25: 'Grandchild',
        26: 'Brother/sister',
        27: 'Parent',
        28: 'Parent-in-law',
        29: 'Son/daughter-in-law',
        30: 'Other relative',
        31: 'Roomer/boarder',
        32: 'Housemate/roommate',
        33: 'Unmarried partner',
        34: 'Foster child',
        35: 'Other nonrelative',
        36: 'Institutionalized GQ person',
        37: 'Noninstitutionalized GQ person'
    }
    
    for rel, count in sorted(rel_counts.items()):
        label = rel_labels.get(rel, f'Unknown ({rel})')
        pct = 100 * count / len(married_adults)
        logger.info(f"  {label} ({rel}): {count:,} ({pct:.1f}%)")
    
    # Load actual tax units to compare
    logger.info("\n" + "="*80)
    logger.info("ACTUAL TAX UNIT COMPARISON")
    logger.info("="*80)
    
    try:
        tax_units = pd.read_parquet('src/data/processed/tax_units_irs_based.parquet')
        
        total_units = len(tax_units)
        joint_units = len(tax_units[tax_units['filing_status'] == 'married_filing_jointly'])
        mfs_units = len(tax_units[tax_units['filing_status'] == 'married_filing_separately'])
        
        logger.info(f"\nTotal tax units: {total_units:,}")
        logger.info(f"Joint filers: {joint_units:,} ({100*joint_units/total_units:.1f}%)")
        logger.info(f"MFS filers: {mfs_units:,} ({100*mfs_units/total_units:.1f}%)")
        logger.info(f"Total married filers: {joint_units + mfs_units:,} ({100*(joint_units + mfs_units)/total_units:.1f}%)")
        
        logger.info(f"\nExpected joint (SOI): 36.0%")
        logger.info(f"Actual joint: {100*joint_units/total_units:.1f}%")
        logger.info(f"Gap: {36.0 - 100*joint_units/total_units:.1f} percentage points")
        
        # Calculate joint filing rate among married couples
        total_married_units = joint_units + mfs_units
        if total_married_units > 0:
            joint_rate = 100 * joint_units / total_married_units
            logger.info(f"\nJoint filing rate among married couples: {joint_rate:.1f}%")
            logger.info(f"MFS rate among married couples: {100 - joint_rate:.1f}%")
        
        # Compare married couples in PUMS vs tax units
        logger.info(f"\nMarried Couples Analysis:")
        logger.info(f"  Married couples in PUMS: {num_married_couples:,}")
        logger.info(f"  Married tax units (joint + MFS): {total_married_units:,}")
        logger.info(f"  Difference: {num_married_couples - total_married_units:,}")
        
        if num_married_couples > 0:
            capture_rate = 100 * total_married_units / num_married_couples
            logger.info(f"  Capture rate: {capture_rate:.1f}%")
        
        # Analyze why we might be missing married couples
        logger.info(f"\n" + "="*80)
        logger.info("POTENTIAL ISSUES")
        logger.info("="*80)
        
        # Check for married adults not in householder/spouse relationships
        married_non_primary = married_adults[~married_adults['RELSHIPP'].isin([20, 21])]
        logger.info(f"\nMarried adults NOT in householder/spouse roles: {len(married_non_primary):,}")
        logger.info(f"  These may be adult children, parents, or other relatives who are married")
        logger.info(f"  but whose spouse is not in the household")
        
        # Check households with only 1 married adult
        logger.info(f"\nHouseholds with spouse absent: {num_spouse_absent:,}")
        logger.info(f"  These married adults may file MFS or their spouse may be in a different household")
        
        # Estimate expected joint filers
        logger.info(f"\n" + "="*80)
        logger.info("EXPECTED VS ACTUAL")
        logger.info("="*80)
        
        # SOI benchmark: 36% of all returns are joint
        # If we have 45,550 tax units, we expect 16,398 joint filers
        expected_joint = int(total_units * 0.36)
        logger.info(f"\nExpected joint filers (36% of {total_units:,}): {expected_joint:,}")
        logger.info(f"Actual joint filers: {joint_units:,}")
        logger.info(f"Shortfall: {expected_joint - joint_units:,} joint filers")
        
        # This shortfall represents couples who should file jointly but aren't
        logger.info(f"\nTo reach 36% joint filers, we need to:")
        logger.info(f"  1. Convert {expected_joint - joint_units:,} MFS or single filers to joint")
        logger.info(f"  2. OR identify {expected_joint - joint_units:,} additional married couples")
        
    except Exception as e:
        logger.error(f"Error loading tax units: {e}")
        import traceback
        traceback.print_exc()
    
    return {
        'married_couples': num_married_couples,
        'spouse_absent': num_spouse_absent,
        'complex_households': num_complex
    }

if __name__ == '__main__':
    analyze_joint_filer_gap()
