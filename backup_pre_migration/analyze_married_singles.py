"""
Analyze cases where married people are incorrectly filing as single.

This script identifies and analyzes cases where married individuals are being
incorrectly classified as single filers, despite having a spouse in the same household.
"""

import pandas as pd
import logging
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Set
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.pums_loader import PUMSDataLoader
from src.tax.units.constructor import TaxUnitConstructor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('married_singles_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def analyze_married_singles():
    """Analyze cases where married people are incorrectly filing as single."""
    
    # Load PUMS data
    logger.info("Loading PUMS data...")
    loader = PUMSDataLoader('data/raw/pums')
    person_df, hh_df = loader.load_data(state='15', year=2022)  # 15 is Hawaii's FIPS code
    
    # Add person_id for easier reference
    person_df['person_id'] = person_df['SERIALNO'].astype(str) + '_' + person_df['SPORDER'].astype(str)
    person_df = person_df.set_index('person_id')
    
    # Construct tax units
    logger.info("Constructing tax units...")
    constructor = TaxUnitConstructor(person_df, hh_df, progress_bar=False)
    tax_units_df = constructor.create_rule_based_units(parallel=False)
    
    # Get all single filers
    single_filers = tax_units_df[tax_units_df['filing_status'] == 'single']
    single_filer_ids = set(single_filers['primary_filer_id'])
    
    # Get all married people
    married_people = person_df[person_df['MAR'] == 1].copy()
    logger.info(f"Found {len(married_people)} married individuals in the data")
    
    # Find married people who are filing as single
    married_singles = married_people[married_people.index.isin(single_filer_ids)]
    logger.info(f"Found {len(married_singles)} married people filing as single")
    
    # For each married single, check if their spouse is in the household
    married_singles_with_spouse = []
    
    for person_id, person in married_singles.iterrows():
        household = person_df[person_df['SERIALNO'] == person['SERIALNO']]
        
        # Look for spouse in household
        spouse = household[
            (household['MAR'] == 1) &  # Also married
            (household.index != person_id) &  # Not the same person
            (
                # Either they're marked as spouse (21) or have same relationship code
                (household['RELSHIPP'] == 21) |
                (household['RELSHIPP'] == person['RELSHIPP'])
            )
        ]
        
        if not spouse.empty:
            spouse = spouse.iloc[0]  # Take the first match
            married_singles_with_spouse.append({
                'person_id': person_id,
                'age': person['AGEP'],
                'sex': person['SEX'],
                'relshipp': person['RELSHIPP'],
                'relshipp_desc': get_relshipp_desc(person['RELSHIPP']),
                'spouse_id': spouse.name,
                'spouse_age': spouse['AGEP'],
                'spouse_sex': spouse['SEX'],
                'spouse_relshipp': spouse['RELSHIPP'],
                'spouse_relshipp_desc': get_relshipp_desc(spouse['RELSHIPP']),
                'age_diff': abs(person['AGEP'] - spouse['AGEP']),
                'same_sex': person['SEX'] == spouse['SEX']
            })
    
    # Convert to DataFrame
    if married_singles_with_spouse:
        analysis_df = pd.DataFrame(married_singles_with_spouse)
        
        # Save to CSV for further analysis
        output_file = 'married_singles_analysis.csv'
        analysis_df.to_csv(output_file, index=False)
        logger.info(f"Saved analysis of {len(analysis_df)} married singles to {output_file}")
        
        # Print summary statistics
        logger.info("\n" + "="*80)
        logger.info("MARRIED SINGLES WITH SPOUSE IN HOUSEHOLD - SUMMARY")
        logger.info("="*80)
        
        # Relationship codes
        logger.info("\nRelationship Codes (Person):")
        rel_counts = analysis_df['relshipp_desc'].value_counts()
        for rel, count in rel_counts.items():
            logger.info(f"  {rel:<30} {count:>6} ({count/len(analysis_df)*100:.1f}%)")
        
        # Age differences
        logger.info("\nAge Differences:")
        logger.info(f"  Mean: {analysis_df['age_diff'].mean():.1f} years")
        logger.info(f"  Median: {analysis_df['age_diff'].median():.1f} years")
        logger.info(f"  Max: {analysis_df['age_diff'].max():.1f} years")
        logger.info(f"  Std Dev: {analysis_df['age_diff'].std():.1f} years")
        
        # Age difference distribution
        bins = [0, 5, 10, 15, 20, 30, 40, 50, 100]
        age_diff_bins = pd.cut(analysis_df['age_diff'], bins=bins)
        logger.info("\nAge Difference Distribution:")
        for bin_val, count in age_diff_bins.value_counts().sort_index().items():
            logger.info(f"  {str(bin_val):<15} {count:>6} ({count/len(analysis_df)*100:.1f}%)")
        
        # Same-sex couples
        same_sex_pct = analysis_df['same_sex'].mean() * 100
        logger.info(f"\nSame-sex couples: {same_sex_pct:.1f}%")
        
        # Spouse relationship codes
        logger.info("\nSpouse Relationship Codes:")
        spouse_rel_counts = analysis_df['spouse_relshipp_desc'].value_counts()
        for rel, count in spouse_rel_counts.items():
            logger.info(f"  {rel:<30} {count:>6} ({count/len(analysis_df)*100:.1f}%)")
        
        # Example cases
        logger.info("\nExample Cases:")
        for _, row in analysis_df.sample(min(5, len(analysis_df))).iterrows():
            logger.info(f"\n  Person: {row['person_id']} (age {row['age']}, {get_sex_desc(row['sex'])}, {row['relshipp_desc']})")
            logger.info(f"  Spouse: {row['spouse_id']} (age {row['spouse_age']}, {get_sex_desc(row['spouse_sex'])}, {row['spouse_relshipp_desc']})")
            logger.info(f"  Age difference: {row['age_diff']} years, Same sex: {row['same_sex']}")
    else:
        logger.info("No married singles with spouses in household found.")

def get_relshipp_desc(code: int) -> str:
    """Get description for relationship code."""
    relshipp_map = {
        0: "Reference person",
        1: "Husband/wife",
        2: "Biological son or daughter",
        3: "Adopted son or daughter",
        4: "Stepson or stepdaughter",
        5: "Brother or sister",
        6: "Father or mother",
        7: "Grandchild",
        8: "In-law",
        9: "Other relative",
        10: "Roomer, boarder, or housemate",
        11: "Housemate/roommate",
        12: "Unmarried partner",
        13: "Foster child",
        14: "Other non-relative",
        15: "Institutionalized group quarters person",
        16: "Noninstitutionalized group quarters person",
        17: "Spouse of householder",
        18: "Child of householder",
        19: "Other relative of householder",
        20: "Householder",
        21: "Spouse of householder",
        22: "Child of householder",
        23: "Other relative of householder",
        24: "Other non-relative of householder",
        25: "Group quarters person"
    }
    return relshipp_map.get(code, f"Unknown ({code})")

def get_sex_desc(code: int) -> str:
    """Get description for sex code."""
    return "Male" if code == 1 else "Female"

if __name__ == "__main__":
    analyze_married_singles()
