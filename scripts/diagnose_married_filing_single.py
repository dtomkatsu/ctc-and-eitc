"""
Diagnostic script to investigate why some married people are filing as single.

This script will:
1. Load tax units and identify married adults filing as single
2. Check their household composition
3. Analyze why they weren't paired with a spouse
4. Identify patterns in missed married couples
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

def analyze_married_filing_single():
    """Analyze married adults filing as single."""
    
    # Load PUMS data
    logger.info("Loading PUMS data...")
    loader = PUMSDataLoader()
    person_data, household_data = loader.load_data()
    
    logger.info(f"Loaded {len(person_data)} persons and {len(household_data)} households")
    
    # Load tax units
    logger.info("Loading tax units...")
    tax_units = pd.read_parquet('src/data/processed/tax_units_irs_based.parquet')
    
    # Create a mapping of person ID to filing status
    person_to_filing_status = {}
    
    for _, unit in tax_units.iterrows():
        primary_id = unit['primary_filer_id']
        filing_status = unit['filing_status']
        person_to_filing_status[primary_id] = filing_status
        
        # For joint/MFS filers, also track secondary filer
        if 'secondary_filer_id' in unit and pd.notna(unit.get('secondary_filer_id')):
            secondary_id = unit['secondary_filer_id']
            person_to_filing_status[secondary_id] = filing_status
    
    # Merge person and household data
    merged = person_data.merge(
        household_data[['SERIALNO', 'WGTP', 'PUMA']],
        on='SERIALNO',
        how='left'
    )
    
    # Filter to married adults
    married_adults = merged[(merged['AGEP'] >= 18) & (merged['MAR'] == 1)].copy()
    logger.info(f"Found {len(married_adults)} married adults")
    
    # Add filing status to married adults
    married_adults['filing_status'] = married_adults.index.map(
        lambda x: person_to_filing_status.get(str(x), 'NOT_FOUND')
    )
    
    # Count filing statuses among married adults
    filing_status_counts = married_adults['filing_status'].value_counts()
    
    logger.info("\n" + "="*80)
    logger.info("FILING STATUS OF MARRIED ADULTS")
    logger.info("="*80)
    
    total_married = len(married_adults)
    for status, count in filing_status_counts.items():
        pct = 100 * count / total_married
        logger.info(f"  {status}: {count:,} ({pct:.1f}%)")
    
    # Focus on married adults filing as single
    married_filing_single = married_adults[married_adults['filing_status'] == 'single']
    logger.info(f"\n{len(married_filing_single)} married adults are filing as SINGLE")
    
    # Analyze their characteristics
    logger.info("\n" + "="*80)
    logger.info("CHARACTERISTICS OF MARRIED ADULTS FILING AS SINGLE")
    logger.info("="*80)
    
    # By relationship code
    rel_counts = married_filing_single['RELSHIPP'].value_counts()
    logger.info("\nBy Relationship Code:")
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
        pct = 100 * count / len(married_filing_single)
        logger.info(f"  {label} ({rel}): {count:,} ({pct:.1f}%)")
    
    # Check if their spouse is in the same household
    logger.info("\n" + "="*80)
    logger.info("SPOUSE PRESENCE ANALYSIS")
    logger.info("="*80)
    
    spouse_in_household = 0
    spouse_not_in_household = 0
    
    for idx, person in married_filing_single.iterrows():
        serialno = person['SERIALNO']
        rel = person['RELSHIPP']
        
        # Get all people in same household
        household = merged[merged['SERIALNO'] == serialno]
        
        # Check if spouse is present
        if rel == 20:  # Householder
            # Look for spouse (21)
            has_spouse = any(household['RELSHIPP'] == 21)
        elif rel == 21:  # Spouse
            # Look for householder (20)
            has_spouse = any(household['RELSHIPP'] == 20)
        else:
            # For others, check if there's another married adult with similar age and opposite sex
            other_married = household[
                (household['MAR'] == 1) & 
                (household.index != idx) &
                (household['AGEP'] >= 18)
            ]
            
            has_spouse = False
            for _, other in other_married.iterrows():
                age_diff = abs(person['AGEP'] - other['AGEP'])
                opposite_sex = person['SEX'] != other['SEX']
                
                if age_diff < 15 and opposite_sex:
                    has_spouse = True
                    break
        
        if has_spouse:
            spouse_in_household += 1
        else:
            spouse_not_in_household += 1
    
    logger.info(f"\nMarried adults filing single WITH spouse in household: {spouse_in_household:,}")
    logger.info(f"Married adults filing single WITHOUT spouse in household: {spouse_not_in_household:,}")
    
    if spouse_in_household > 0:
        pct_with_spouse = 100 * spouse_in_household / len(married_filing_single)
        logger.info(f"\n⚠️  {pct_with_spouse:.1f}% of married-filing-single have spouse in household!")
        logger.info("These are MISSED married couples that should be filing joint or MFS")
    
    # Analyze specific cases where spouse is in household
    logger.info("\n" + "="*80)
    logger.info("SAMPLE CASES: MARRIED FILING SINGLE WITH SPOUSE IN HOUSEHOLD")
    logger.info("="*80)
    
    sample_count = 0
    max_samples = 10
    
    for idx, person in married_filing_single.iterrows():
        if sample_count >= max_samples:
            break
            
        serialno = person['SERIALNO']
        rel = person['RELSHIPP']
        
        # Get all people in same household
        household = merged[merged['SERIALNO'] == serialno]
        
        # Check if spouse is present
        has_spouse = False
        spouse_info = None
        
        if rel == 20:  # Householder
            spouse = household[household['RELSHIPP'] == 21]
            if len(spouse) > 0:
                has_spouse = True
                spouse_info = spouse.iloc[0]
        elif rel == 21:  # Spouse
            householder = household[household['RELSHIPP'] == 20]
            if len(householder) > 0:
                has_spouse = True
                spouse_info = householder.iloc[0]
        else:
            # Check for potential spouse
            other_married = household[
                (household['MAR'] == 1) & 
                (household.index != idx) &
                (household['AGEP'] >= 18)
            ]
            
            for _, other in other_married.iterrows():
                age_diff = abs(person['AGEP'] - other['AGEP'])
                opposite_sex = person['SEX'] != other['SEX']
                
                if age_diff < 15 and opposite_sex:
                    has_spouse = True
                    spouse_info = other
                    break
        
        if has_spouse and spouse_info is not None:
            sample_count += 1
            
            logger.info(f"\nCase {sample_count}:")
            logger.info(f"  Person: ID={idx}, REL={rel} ({rel_labels.get(rel, 'Unknown')}), AGE={person['AGEP']}, SEX={person['SEX']}")
            logger.info(f"  Spouse: ID={spouse_info.name}, REL={spouse_info['RELSHIPP']} ({rel_labels.get(spouse_info['RELSHIPP'], 'Unknown')}), AGE={spouse_info['AGEP']}, SEX={spouse_info['SEX']}")
            logger.info(f"  Household: {serialno}, {len(household)} members, {len(household[household['AGEP'] >= 18])} adults")
            
            # Check spouse's filing status
            spouse_filing_status = person_to_filing_status.get(str(spouse_info.name), 'NOT_FOUND')
            logger.info(f"  Spouse filing status: {spouse_filing_status}")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    
    logger.info(f"\nTotal married adults: {total_married:,}")
    logger.info(f"Married filing single: {len(married_filing_single):,} ({100*len(married_filing_single)/total_married:.1f}%)")
    logger.info(f"  With spouse in household: {spouse_in_household:,}")
    logger.info(f"  Without spouse in household: {spouse_not_in_household:,}")
    
    logger.info(f"\nPotential additional joint/MFS filers: {spouse_in_household:,}")
    logger.info(f"This would increase married filers from {len(tax_units[tax_units['filing_status'].isin(['married_filing_jointly', 'married_filing_separately'])])} to {len(tax_units[tax_units['filing_status'].isin(['married_filing_jointly', 'married_filing_separately'])]) + spouse_in_household}")
    
    return {
        'married_filing_single': len(married_filing_single),
        'spouse_in_household': spouse_in_household,
        'spouse_not_in_household': spouse_not_in_household
    }

if __name__ == '__main__':
    analyze_married_filing_single()
