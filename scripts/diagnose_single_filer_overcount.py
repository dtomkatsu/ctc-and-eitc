"""
Diagnose why single filers are being over-identified in the tax unit construction.

This script analyzes:
1. How many married adults are filing as single
2. How many unmarried adults are filing as single
3. The relationship codes of single filers
4. Comparison with SOI benchmarks
"""

import pandas as pd
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.pums_loader import PUMSDataLoader
from src.tax.units.constructor import TaxUnitConstructor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_single_filer_overcount():
    """Analyze why single filers are being over-identified."""
    
    # Load PUMS data
    logger.info("Loading PUMS data...")
    loader = PUMSDataLoader('data/raw/pums')
    person_df, hh_df = loader.load_data(state='15', year=2022)  # 15 is Hawaii's FIPS code
    
    logger.info(f"Loaded {len(person_df)} persons and {len(hh_df)} households")
    
    # Construct tax units
    logger.info("Constructing tax units...")
    constructor = TaxUnitConstructor(person_df, hh_df, progress_bar=False)
    tax_units_df = constructor.create_rule_based_units(parallel=False)
    
    logger.info(f"Created {len(tax_units_df)} tax units")
    
    # Analyze filing status distribution
    logger.info("\n" + "="*80)
    logger.info("FILING STATUS DISTRIBUTION")
    logger.info("="*80)
    
    status_counts = tax_units_df['filing_status'].value_counts()
    total_units = len(tax_units_df)
    
    for status, count in status_counts.items():
        pct = (count / total_units) * 100
        logger.info(f"{status:<30} {count:>6} ({pct:>5.1f}%)")
    
    # Load SOI benchmarks
    soi_file = Path('data/processed/soi_benchmarks_2022.csv')
    if soi_file.exists():
        soi_df = pd.read_csv(soi_file)
        
        logger.info("\n" + "="*80)
        logger.info("COMPARISON WITH SOI BENCHMARKS")
        logger.info("="*80)
        
        single_pct = (status_counts.get('single', 0) / total_units) * 100
        joint_pct = (status_counts.get('married_filing_jointly', 0) / total_units) * 100
        hoh_pct = (status_counts.get('head_of_household', 0) / total_units) * 100
        mfs_pct = (status_counts.get('married_filing_separate', 0) / total_units) * 100
        
        soi_single_pct = soi_df['single_pct'].iloc[0]
        soi_joint_pct = soi_df['joint_pct'].iloc[0]
        soi_hoh_pct = soi_df['hoh_pct'].iloc[0]
        soi_mfs_pct = soi_df['mfs_pct'].iloc[0]
        
        logger.info(f"{'Status':<25} {'Model %':>10} {'SOI %':>10} {'Gap (pp)':>12}")
        logger.info("-" * 60)
        logger.info(f"{'Single':<25} {single_pct:>9.1f}% {soi_single_pct:>9.1f}% {single_pct - soi_single_pct:>11.1f}pp")
        logger.info(f"{'Joint':<25} {joint_pct:>9.1f}% {soi_joint_pct:>9.1f}% {joint_pct - soi_joint_pct:>11.1f}pp")
        logger.info(f"{'Head of Household':<25} {hoh_pct:>9.1f}% {soi_hoh_pct:>9.1f}% {hoh_pct - soi_hoh_pct:>11.1f}pp")
        logger.info(f"{'Married Filing Sep':<25} {mfs_pct:>9.1f}% {soi_mfs_pct:>9.1f}% {mfs_pct - soi_mfs_pct:>11.1f}pp")
    
    # Analyze single filers in detail
    logger.info("\n" + "="*80)
    logger.info("SINGLE FILER ANALYSIS")
    logger.info("="*80)
    
    # Get all single filers
    single_filers = tax_units_df[tax_units_df['filing_status'] == 'single']
    logger.info(f"Total single filers: {len(single_filers)}")
    
    # Map back to person data
    single_filer_ids = single_filers['primary_filer_id'].tolist()
    
    # Create a mapping from person_id to person data
    person_df['person_id'] = person_df['SERIALNO'].astype(str) + '_' + person_df['SPORDER'].astype(str)
    person_df_indexed = person_df.set_index('person_id')
    
    # Get person data for single filers
    single_filer_persons = person_df_indexed.loc[single_filer_ids]
    
    # Analyze marital status of single filers
    logger.info("\nMarital Status of Single Filers:")
    mar_counts = single_filer_persons['MAR'].value_counts().sort_index()
    
    mar_labels = {
        1: "Married",
        2: "Widowed",
        3: "Divorced",
        4: "Separated",
        5: "Never married"
    }
    
    for mar_code, count in mar_counts.items():
        pct = (count / len(single_filer_persons)) * 100
        label = mar_labels.get(mar_code, f"Unknown ({mar_code})")
        logger.info(f"  {label:<20} {count:>6} ({pct:>5.1f}%)")
    
    # Analyze relationship codes of single filers
    logger.info("\nRelationship Codes of Single Filers:")
    rel_counts = single_filer_persons['RELSHIPP'].value_counts().sort_index()
    
    rel_labels = {
        20: "Householder",
        21: "Spouse",
        22: "Parent",
        23: "Child",
        24: "Sibling",
        25: "Other relative",
        26: "Grandchild",
        27: "In-law",
        28: "Other nonrelative",
        29: "Institutionalized GQ person",
        30: "Noninstitutionalized GQ person"
    }
    
    for rel_code, count in rel_counts.items():
        pct = (count / len(single_filer_persons)) * 100
        label = rel_labels.get(rel_code, f"Unknown ({rel_code})")
        logger.info(f"  {label:<30} {count:>6} ({pct:>5.1f}%)")
    
    # Analyze married people filing as single
    logger.info("\n" + "="*80)
    logger.info("MARRIED PEOPLE FILING AS SINGLE")
    logger.info("="*80)
    
    married_single_filers = single_filer_persons[single_filer_persons['MAR'] == 1]
    logger.info(f"Total married people filing as single: {len(married_single_filers)}")
    logger.info(f"Percentage of single filers: {len(married_single_filers) / len(single_filer_persons) * 100:.1f}%")
    
    if len(married_single_filers) > 0:
        logger.info("\nRelationship codes of married people filing as single:")
        married_rel_counts = married_single_filers['RELSHIPP'].value_counts().sort_index()
        
        for rel_code, count in married_rel_counts.items():
            pct = (count / len(married_single_filers)) * 100
            label = rel_labels.get(rel_code, f"Unknown ({rel_code})")
            logger.info(f"  {label:<30} {count:>6} ({pct:>5.1f}%)")
        
        # Check if these married people have spouses in the same household
        logger.info("\nChecking if married single filers have spouses in household:")
        
        married_with_spouse_in_hh = 0
        married_without_spouse_in_hh = 0
        
        for person_id in married_single_filers.index:
            serialno = person_id.split('_')[0]
            household_members = person_df[person_df['SERIALNO'] == serialno]
            
            # Check if there's a spouse (RELSHIPP=21) in the household
            has_spouse = (household_members['RELSHIPP'] == 21).any()
            
            if has_spouse:
                married_with_spouse_in_hh += 1
            else:
                married_without_spouse_in_hh += 1
        
        logger.info(f"  Married single filers WITH spouse in household: {married_with_spouse_in_hh}")
        logger.info(f"  Married single filers WITHOUT spouse in household: {married_without_spouse_in_hh}")
        
        if married_with_spouse_in_hh > 0:
            logger.info(f"\n⚠️  WARNING: {married_with_spouse_in_hh} married people are filing as single")
            logger.info(f"    even though their spouse is in the same household!")
            logger.info(f"    This indicates a problem with the joint filer identification logic.")
    
    # Analyze age distribution of single filers
    logger.info("\n" + "="*80)
    logger.info("AGE DISTRIBUTION OF SINGLE FILERS")
    logger.info("="*80)
    
    age_bins = [18, 25, 35, 45, 55, 65, 100]
    age_labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    single_filer_persons['age_group'] = pd.cut(single_filer_persons['AGEP'], bins=age_bins, labels=age_labels, right=False)
    
    age_counts = single_filer_persons['age_group'].value_counts().sort_index()
    
    for age_group, count in age_counts.items():
        pct = (count / len(single_filer_persons)) * 100
        logger.info(f"  {age_group:<10} {count:>6} ({pct:>5.1f}%)")
    
    # Summary and recommendations
    logger.info("\n" + "="*80)
    logger.info("SUMMARY AND RECOMMENDATIONS")
    logger.info("="*80)
    
    logger.info(f"\n1. Single filers are {single_pct - soi_single_pct:.1f} percentage points above SOI target")
    logger.info(f"2. {len(married_single_filers)} married people are filing as single ({len(married_single_filers) / len(single_filer_persons) * 100:.1f}% of single filers)")
    
    if married_with_spouse_in_hh > 0:
        logger.info(f"3. {married_with_spouse_in_hh} married people with spouses in household are filing as single")
        logger.info(f"   → This is the PRIMARY ISSUE causing single filer overcount")
        logger.info(f"   → Need to improve joint filer identification in _identify_joint_filers()")
    
    logger.info(f"\n4. Joint filers are {joint_pct - soi_joint_pct:.1f} percentage points below SOI target")
    logger.info(f"   → Converting {married_with_spouse_in_hh} married singles to joint would help close this gap")

if __name__ == '__main__':
    analyze_single_filer_overcount()
