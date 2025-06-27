#!/usr/bin/env python3
"""
Debug filing status classification to understand why Head of Household is not being identified.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import logging
from src.data.pums_loader import PUMSDataLoader
from src.tax.units.status.hoh import is_head_of_household
from src.tax.units.dependencies import identify_dependents

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_hoh_classification():
    """Debug Head of Household classification."""
    
    # Load processed data
    person_df = pd.read_parquet('data/processed/pums_person_processed.parquet')
    hh_df = pd.read_parquet('data/processed/pums_household_processed.parquet')
    
    logger.info(f"Loaded {len(hh_df)} households and {len(person_df)} persons")
    
    # Take a sample for debugging - filter to only household records (not group quarters)
    # Household records have SERIALNO starting with year+HU, group quarters start with year+GQ
    household_persons = person_df[person_df['SERIALNO'].str.contains('HU', na=False)]
    sample_households = household_persons['SERIALNO'].unique()[:500]  # Larger sample
    person_sample = household_persons[household_persons['SERIALNO'].isin(sample_households)]
    
    logger.info(f"Filtered to {len(sample_households)} household records with {len(person_sample)} persons")
    
    # Look for potential HOH candidates
    hoh_candidates = 0
    hoh_qualified = 0
    
    # Group by household
    for serialno, hh_group in person_sample.groupby('SERIALNO'):
        adults = hh_group[hh_group['AGEP'] >= 18]
        
        if len(adults) == 0:
            continue
            
        # Check each adult for HOH qualification
        for _, adult in adults.iterrows():
            # Check if they have dependents
            all_dependents = identify_dependents(hh_group)
            dependents = all_dependents.get(adult.name, [])
            
            if len(dependents) > 0:
                hoh_candidates += 1
                logger.debug(f"HOH candidate: Person {adult.name} in household {serialno}")
                logger.debug(f"  Age: {adult.get('AGEP')}, Marital: {adult.get('MAR')}, Relationship: {adult.get('RELSHIPP')}")
                logger.debug(f"  Dependents: {len(dependents)}")
                
                # Test HOH qualification
                person_data = hh_group.copy()
                person_data['SERIALNO'] = serialno
                
                if is_head_of_household(adult, person_data):
                    hoh_qualified += 1
                    logger.info(f"✓ QUALIFIED HOH: Person {adult.name} in household {serialno}")
                    logger.info(f"  Age: {adult.get('AGEP')}, Marital: {adult.get('MAR')}, Income: {adult.get('PINCP', 0)}")
                else:
                    logger.debug(f"✗ Not qualified HOH: Person {adult.name}")
                    
                    # Debug why not qualified
                    from src.tax.units.status.hoh import _is_unmarried, _has_qualifying_person, _paid_half_home_cost
                    
                    unmarried = _is_unmarried(adult, person_data)
                    has_qualifying = _has_qualifying_person(adult, person_data)
                    paid_half = _paid_half_home_cost(adult, person_data)
                    
                    logger.debug(f"  Unmarried: {unmarried}, Has qualifying person: {has_qualifying}, Paid half: {paid_half}")
    
    logger.info(f"Summary: {hoh_candidates} HOH candidates, {hoh_qualified} qualified for HOH")
    
    # Also check marital status distribution
    marital_dist = person_sample[person_sample['AGEP'] >= 18]['MAR'].value_counts()
    logger.info(f"Marital status distribution: {marital_dist.to_dict()}")
    
    # Check relationship distribution
    rel_dist = person_sample['RELSHIPP'].value_counts()
    logger.info(f"Relationship distribution: {rel_dist.to_dict()}")
    
    # Check for single parents specifically
    single_parents = 0
    for serialno, hh_group in person_sample.groupby('SERIALNO'):
        adults = hh_group[hh_group['AGEP'] >= 18]
        children = hh_group[hh_group['AGEP'] < 18]
        
        if len(adults) == 1 and len(children) > 0:
            adult = adults.iloc[0]
            single_parents += 1
            logger.debug(f"Single parent household {serialno}: Adult {adult.name}, {len(children)} children")
            logger.debug(f"  Adult marital status: {adult.get('MAR')}, relationship: {adult.get('RELSHIPP')}")
    
    logger.info(f"Found {single_parents} single parent households")

if __name__ == "__main__":
    debug_hoh_classification()
