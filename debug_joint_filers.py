#!/usr/bin/env python3
"""
Debug script to investigate why joint filers are under-represented.
"""

import pandas as pd
import logging
from pathlib import Path
from src.tax.units import TaxUnitConstructor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_potential_joint_filers():
    """Analyze households with potential joint filers."""
    
    # Load processed data
    data_dir = Path("data/processed")
    person_file = data_dir / "pums_person_processed.parquet"
    hh_file = data_dir / "pums_household_processed.parquet"
    
    logger.info("Loading processed PUMS data...")
    person_df = pd.read_parquet(person_file)
    hh_df = pd.read_parquet(hh_file)
    
    # Calculate adults per household
    adults_per_hh = person_df[person_df['AGEP'] >= 18].groupby('SERIALNO').size()
    
    # Focus on households with exactly 2 adults (most likely to have joint filers)
    two_adult_households = adults_per_hh[adults_per_hh == 2].index.tolist()
    
    logger.info(f"Found {len(two_adult_households)} households with exactly 2 adults")
    
    # Create constructor
    constructor = TaxUnitConstructor(person_df, hh_df)
    
    # Sample a few households for detailed analysis
    sample_households = two_adult_households[:10]
    
    joint_filer_count = 0
    total_households_analyzed = 0
    
    for serialno in sample_households:
        logger.info(f"\n--- Analyzing household {serialno} ---")
        
        # Get household data
        hh_data = hh_df[hh_df['SERIALNO'] == serialno].iloc[0]
        hh_members = person_df[person_df['SERIALNO'] == serialno]
        adults = hh_members[hh_members['AGEP'] >= 18]
        
        logger.info(f"Household has {len(adults)} adults:")
        for idx, adult in adults.iterrows():
            logger.info(f"  Adult {idx}: Age {adult['AGEP']}, RELSHIPP {adult.get('RELSHIPP', 'N/A')}, MAR {adult.get('MAR', 'N/A')}")
        
        # Check if they are potential spouses
        if len(adults) == 2:
            adult1 = adults.iloc[0]
            adult2 = adults.iloc[1]
            
            is_spouse = constructor._is_potential_spouse(adult1, adult2)
            logger.info(f"Are they potential spouses? {is_spouse}")
            
            if is_spouse:
                joint_filer_count += 1
                logger.info("  -> This would create a joint filer")
            else:
                logger.info("  -> This would create 2 single filers")
                
                # Debug why they're not considered spouses
                logger.info(f"  Debug info:")
                logger.info(f"    Age difference: {abs(adult1['AGEP'] - adult2['AGEP'])}")
                logger.info(f"    Both adults in household: 2")
                logger.info(f"    Adult 1 MAR status: {adult1.get('MAR', 'N/A')}")
                logger.info(f"    Adult 2 MAR status: {adult2.get('MAR', 'N/A')}")
                
                # Check specific conditions from the spouse identification logic
                rel1 = str(adult1.get('RELSHIPP', ''))
                rel2 = str(adult2.get('RELSHIPP', ''))
                spouse_relationships = {'21', '22', '23', '24'}
                
                logger.info(f"    Direct spouse relationship: {(rel1 in spouse_relationships and rel2 == '20') or (rel2 in spouse_relationships and rel1 == '20')}")
                logger.info(f"    Both reference persons: {rel1 == '20' and rel2 == '20'}")
                
                if rel1 == '20' and rel2 == '20':
                    age_diff = abs(adult1['AGEP'] - adult2['AGEP'])
                    logger.info(f"    Age diff within limit (25): {age_diff <= 25}")
                    logger.info(f"    Only two adults: True")
                
                mar1 = adult1.get('MAR', 6)
                mar2 = adult2.get('MAR', 6)
                logger.info(f"    Both married (MAR=1): {mar1 == 1 and mar2 == 1}")
        
        total_households_analyzed += 1
    
    logger.info(f"\n--- Summary ---")
    logger.info(f"Analyzed {total_households_analyzed} two-adult households")
    logger.info(f"Found {joint_filer_count} potential joint filer households")
    logger.info(f"Joint filer rate in sample: {joint_filer_count/total_households_analyzed*100:.1f}%")
    
    # Now let's look at the overall marital status distribution
    logger.info(f"\n--- Overall Marital Status Distribution ---")
    mar_counts = person_df[person_df['AGEP'] >= 18]['MAR'].value_counts().sort_index()
    total_adults = len(person_df[person_df['AGEP'] >= 18])
    
    mar_labels = {
        1: "Married, spouse present",
        2: "Married, spouse absent", 
        3: "Separated",
        4: "Divorced",
        5: "Widowed",
        6: "Never married"
    }
    
    for mar_code, count in mar_counts.items():
        label = mar_labels.get(mar_code, f"Unknown ({mar_code})")
        pct = count / total_adults * 100
        logger.info(f"  {label}: {count:,} ({pct:.1f}%)")
    
    # Look at relationship codes
    logger.info(f"\n--- Relationship Code Distribution (Adults) ---")
    rel_counts = person_df[person_df['AGEP'] >= 18]['RELSHIPP'].value_counts().sort_index()
    
    rel_labels = {
        20: "Reference person",
        21: "Opposite-sex spouse",
        22: "Opposite-sex unmarried partner", 
        23: "Same-sex spouse",
        24: "Same-sex unmarried partner"
    }
    
    for rel_code, count in rel_counts.items():
        label = rel_labels.get(rel_code, f"Other ({rel_code})")
        pct = count / total_adults * 100
        logger.info(f"  {label}: {count:,} ({pct:.1f}%)")
    
    # Additional analysis: Look at households with married adults
    logger.info(f"\n--- Households with Married Adults ---")
    married_adults = person_df[(person_df['AGEP'] >= 18) & (person_df['MAR'] == 1)]
    married_households = married_adults.groupby('SERIALNO').size()
    
    logger.info(f"Total households with married adults: {len(married_households)}")
    logger.info(f"Households with 1 married adult: {sum(married_households == 1)}")
    logger.info(f"Households with 2 married adults: {sum(married_households == 2)}")
    logger.info(f"Households with 3+ married adults: {sum(married_households >= 3)}")

if __name__ == "__main__":
    analyze_potential_joint_filers()
