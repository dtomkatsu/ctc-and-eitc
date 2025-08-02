#!/usr/bin/env python3
"""
Debug spouse pairing logic to understand why joint filer capture rate is only 76.6%.

This script will trace through the exact spouse identification and pairing process
to find where married couples are being missed.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.pums_loader import PUMSDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_spouse_identification():
    """Debug the spouse identification process step by step."""
    logger.info("Starting spouse identification debug...")
    
    # Load PUMS data
    loader = PUMSDataLoader()
    person_df, hh_df = loader.load_data(state=15, year=2022)
    
    # Load constructed tax units
    tax_units_path = project_root / "src/data/processed/tax_units_irs_based.parquet"
    tax_units = pd.read_parquet(tax_units_path)
    
    # Filter to adults and add person IDs
    adults = person_df[person_df['AGEP'] >= 18].copy()
    adults['person_id'] = adults['SERIALNO'].astype(str) + '_' + adults['SPORDER'].astype(str)
    
    print("Spouse Identification Debug")
    print("=" * 50)
    
    # Step 1: Count total married adults
    married_adults = adults[adults['MAR'] == 1]
    print(f"Total married adults: {len(married_adults):,}")
    
    # Step 2: Count householder+spouse pairs
    householder_spouse_pairs = 0
    missing_spouse_households = []
    
    for serialno, hh_group in adults.groupby('SERIALNO'):
        hh_married = hh_group[hh_group['MAR'] == 1]
        if len(hh_married) < 2:
            continue  # Skip households with <2 married adults
            
        # Count relationship types
        relshipp_counts = hh_married['RELSHIPP'].value_counts()
        householders = relshipp_counts.get(20, 0)  # Householder
        spouses = relshipp_counts.get(21, 0)       # Spouse
        
        if householders >= 1 and spouses >= 1:
            householder_spouse_pairs += 1
        else:
            # This household has married adults but no householder+spouse pair
            missing_spouse_households.append({
                'serialno': serialno,
                'married_adults': len(hh_married),
                'householders': householders,
                'spouses': spouses,
                'relshipp_codes': list(hh_married['RELSHIPP'].values)
            })
    
    print(f"Households with householder+spouse pairs: {householder_spouse_pairs:,}")
    print(f"Households with married adults but no householder+spouse pair: {len(missing_spouse_households):,}")
    
    # Step 3: Count actual joint filers created
    joint_filers = tax_units[tax_units['filing_status'] == 'married_filing_jointly']
    print(f"Actual joint filer units created: {len(joint_filers):,}")
    
    # Step 4: Calculate capture rates
    theoretical_max = householder_spouse_pairs
    actual_created = len(joint_filers)
    capture_rate = (actual_created / theoretical_max) * 100 if theoretical_max > 0 else 0
    
    print(f"\nCapture Rate Analysis:")
    print(f"  Theoretical max joint filers: {theoretical_max:,}")
    print(f"  Actual joint filers created: {actual_created:,}")
    print(f"  Capture rate: {capture_rate:.1f}%")
    print(f"  Missing joint filers: {theoretical_max - actual_created:,}")
    
    # Step 5: Analyze missing spouse households
    if missing_spouse_households:
        print(f"\nAnalysis of {len(missing_spouse_households)} households with married adults but no householder+spouse pair:")
        
        # Group by relationship pattern
        pattern_counts = {}
        for hh in missing_spouse_households:
            pattern = tuple(sorted(hh['relshipp_codes']))
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        print("Relationship code patterns:")
        for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {pattern}: {count} households")
    
    return missing_spouse_households

def debug_joint_filer_creation():
    """Debug why identified spouse pairs aren't creating joint filers."""
    logger.info("Debugging joint filer creation process...")
    
    # Load PUMS data
    loader = PUMSDataLoader()
    person_df, hh_df = loader.load_data(state=15, year=2022)
    
    # Load constructed tax units
    tax_units_path = project_root / "src/data/processed/tax_units_irs_based.parquet"
    tax_units = pd.read_parquet(tax_units_path)
    
    # Filter to adults and add person IDs
    adults = person_df[person_df['AGEP'] >= 18].copy()
    adults['person_id'] = adults['SERIALNO'].astype(str) + '_' + adults['SPORDER'].astype(str)
    
    print("\nJoint Filer Creation Debug")
    print("=" * 50)
    
    # Get households that should have joint filers
    should_have_joint = []
    
    for serialno, hh_group in adults.groupby('SERIALNO'):
        hh_married = hh_group[hh_group['MAR'] == 1]
        if len(hh_married) < 2:
            continue
            
        # Count relationship types
        relshipp_counts = hh_married['RELSHIPP'].value_counts()
        householders = relshipp_counts.get(20, 0)
        spouses = relshipp_counts.get(21, 0)
        
        if householders >= 1 and spouses >= 1:
            should_have_joint.append(serialno)
    
    print(f"Households that should have joint filers: {len(should_have_joint):,}")
    
    # Check which ones actually have joint filers
    joint_filer_households = set(tax_units[tax_units['filing_status'] == 'married_filing_jointly']['hh_id'].dropna())
    
    # Convert to string for comparison (in case of type mismatch)
    joint_filer_households = {str(hh_id) for hh_id in joint_filer_households}
    should_have_joint_str = {str(serialno) for serialno in should_have_joint}
    
    print(f"Households that actually have joint filers: {len(joint_filer_households):,}")
    
    # Find missing joint filers
    missing_joint = should_have_joint_str - joint_filer_households
    print(f"Households missing joint filers: {len(missing_joint):,}")
    
    # Sample some missing households for analysis
    if missing_joint:
        print(f"\nSample of households missing joint filers:")
        sample_missing = list(missing_joint)[:5]  # First 5
        
        for serialno in sample_missing:
            hh_group = adults[adults['SERIALNO'] == serialno]
            hh_married = hh_group[hh_group['MAR'] == 1]
            
            print(f"\nHousehold {serialno}:")
            print(f"  Total adults: {len(hh_group)}")
            print(f"  Married adults: {len(hh_married)}")
            
            for _, person in hh_married.iterrows():
                print(f"    Person {person['person_id']}: RELSHIPP={person['RELSHIPP']}, MAR={person['MAR']}, AGE={person['AGEP']}")
            
            # Check what tax units were created for this household
            hh_tax_units = tax_units[tax_units['hh_id'] == serialno]
            if len(hh_tax_units) > 0:
                print(f"  Tax units created: {len(hh_tax_units)}")
                for _, unit in hh_tax_units.iterrows():
                    print(f"    {unit['filing_status']}: {unit.get('primary_filer_id', 'N/A')}")
            else:
                print(f"  No tax units found for this household!")
    
    return missing_joint

def main():
    """Run comprehensive spouse pairing debug."""
    print("Spouse Pairing Debug Analysis")
    print("=" * 60)
    
    try:
        # 1. Debug spouse identification
        missing_spouse_hh = debug_spouse_identification()
        
        # 2. Debug joint filer creation
        missing_joint_hh = debug_joint_filer_creation()
        
        print(f"\n" + "=" * 60)
        print("SUMMARY:")
        print(f"=" * 60)
        print(f"The joint filer capture rate discrepancy is likely due to:")
        print(f"1. Households with married adults but no householder+spouse relationship codes")
        print(f"2. Tax unit creation logic not properly handling identified spouse pairs")
        print(f"3. Data inconsistencies between person and tax unit datasets")
        
        print(f"\nNext steps:")
        print(f"- Fix relationship code handling for non-traditional married couples")
        print(f"- Ensure all identified spouse pairs create joint tax units")
        print(f"- Investigate data consistency issues")
        
    except Exception as e:
        logger.error(f"Debug analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()
