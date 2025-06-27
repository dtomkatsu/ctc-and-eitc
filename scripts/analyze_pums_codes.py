#!/usr/bin/env python3
"""
Analyze PUMS codes to understand the data structure.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_pums_codes():
    """Analyze PUMS codes to understand data structure."""
    
    # Load processed data
    person_df = pd.read_parquet('data/processed/pums_person_processed.parquet')
    
    logger.info(f"Loaded {len(person_df)} persons")
    
    # Check unique values for key fields
    logger.info("=== RELATIONSHIP CODES (RELSHIPP) ===")
    rel_counts = person_df['RELSHIPP'].value_counts().sort_index()
    for code, count in rel_counts.items():
        logger.info(f"Code {code}: {count} persons")
    
    logger.info("\n=== MARITAL STATUS CODES (MAR) ===")
    mar_counts = person_df['MAR'].value_counts().sort_index()
    for code, count in mar_counts.items():
        logger.info(f"Code {code}: {count} persons")
    
    logger.info("\n=== AGE DISTRIBUTION ===")
    age_groups = pd.cut(person_df['AGEP'], bins=[0, 18, 25, 35, 50, 65, 100], right=False)
    age_counts = age_groups.value_counts().sort_index()
    for group, count in age_counts.items():
        logger.info(f"{group}: {count} persons")
    
    # Look at household composition
    logger.info("\n=== HOUSEHOLD COMPOSITION ANALYSIS ===")
    
    # Sample some households to understand structure
    sample_households = person_df['SERIALNO'].unique()[:10]
    
    for serialno in sample_households:
        hh_members = person_df[person_df['SERIALNO'] == serialno]
        logger.info(f"\nHousehold {serialno}: {len(hh_members)} members")
        
        for _, person in hh_members.iterrows():
            logger.info(f"  Person {person.name}: Age {person['AGEP']}, "
                       f"Relationship {person['RELSHIPP']}, "
                       f"Marital {person['MAR']}")
    
    # Check for households with children
    logger.info("\n=== HOUSEHOLDS WITH CHILDREN ===")
    households_with_children = 0
    single_parent_households = 0
    
    for serialno, hh_group in person_df.groupby('SERIALNO'):
        adults = hh_group[hh_group['AGEP'] >= 18]
        children = hh_group[hh_group['AGEP'] < 18]
        
        if len(children) > 0:
            households_with_children += 1
            
            if len(adults) == 1:
                single_parent_households += 1
                adult = adults.iloc[0]
                logger.info(f"Single parent HH {serialno}: Adult age {adult['AGEP']}, "
                           f"marital {adult['MAR']}, rel {adult['RELSHIPP']}, "
                           f"{len(children)} children")
    
    logger.info(f"Total households with children: {households_with_children}")
    logger.info(f"Single parent households: {single_parent_households}")

if __name__ == "__main__":
    analyze_pums_codes()
