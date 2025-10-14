#!/usr/bin/env python3
"""
Analyze married adults who are being misclassified as single filers.

This script identifies specific cases where married adults should be filing jointly
but are instead being classified as single filers.
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

def analyze_married_single_filers():
    """Analyze married adults who are classified as single filers."""
    logger.info("Analyzing married adults classified as single filers...")
    
    # Load PUMS data
    loader = PUMSDataLoader()
    person_df, hh_df = loader.load_data(state=15, year=2022)
    
    # Load constructed tax units
    tax_units_path = project_root / "src/data/processed/tax_units_irs_based.parquet"
    if not tax_units_path.exists():
        logger.error(f"Tax units file not found: {tax_units_path}")
        return None
    
    tax_units = pd.read_parquet(tax_units_path)
    
    # Get single filers
    single_filers = tax_units[tax_units['filing_status'] == 'single']
    
    print("Married Adults Classified as Single Filers Analysis")
    print("=" * 60)
    
    # Extract filer IDs from single filer units
    single_filer_ids = []
    for _, unit in single_filers.iterrows():
        if pd.notna(unit.get('primary_filer_id')):
            single_filer_ids.append(unit['primary_filer_id'])
        if pd.notna(unit.get('secondary_filer_id')):
            single_filer_ids.append(unit['secondary_filer_id'])
    
    print(f"Total single filer units: {len(single_filers):,}")
    print(f"Total adults in single filer units: {len(single_filer_ids):,}")
    
    # Filter to adults only and add person IDs
    adults = person_df[person_df['AGEP'] >= 18].copy()
    adults['person_id'] = adults['SERIALNO'].astype(str) + '_' + adults['SPORDER'].astype(str)
    
    # Find married adults who are single filers
    married_single_filers = adults[
        (adults['person_id'].isin(single_filer_ids)) & 
        (adults['MAR'] == 1)  # Married
    ].copy()
    
    print(f"Married adults classified as single filers: {len(married_single_filers):,}")
    print(f"Percentage of single filers who are married: {len(married_single_filers)/len(single_filer_ids)*100:.1f}%")
    
    # Analyze their household situations
    print(f"\nHousehold Analysis of Married Single Filers:")
    
    # Group by household to see their living situations
    hh_analysis = []
    for serialno, hh_group in married_single_filers.groupby('SERIALNO'):
        hh_adults = adults[adults['SERIALNO'] == serialno]
        hh_married = hh_adults[hh_adults['MAR'] == 1]
        
        # Count relationship types
        relshipp_counts = hh_adults['RELSHIPP'].value_counts()
        householders = relshipp_counts.get(20, 0)
        spouses = relshipp_counts.get(21, 0)
        
        hh_analysis.append({
            'serialno': serialno,
            'married_single_filers': len(hh_group),
            'total_adults': len(hh_adults),
            'total_married': len(hh_married),
            'householders': householders,
            'spouses': spouses,
            'has_spouse_pair': (householders >= 1 and spouses >= 1),
            'multiple_married': len(hh_married) > 2
        })
    
    hh_df = pd.DataFrame(hh_analysis)
    
    print(f"Households with married single filers: {len(hh_df):,}")
    print(f"Households with householder+spouse pairs: {hh_df['has_spouse_pair'].sum():,}")
    print(f"Households with 3+ married adults: {hh_df['multiple_married'].sum():,}")
    
    # These are the key cases - married couples where one or both are filing single
    problematic_households = hh_df[hh_df['has_spouse_pair'] == True]
    print(f"\nPROBLEMATIC: Households with householder+spouse where someone files single: {len(problematic_households):,}")
    
    return married_single_filers, hh_df, problematic_households

def analyze_joint_filer_households():
    """Analyze households that should have joint filers but don't."""
    logger.info("Analyzing households that should have joint filers...")
    
    # Load PUMS data
    loader = PUMSDataLoader()
    person_df, hh_df = loader.load_data(state=15, year=2022)
    
    # Load constructed tax units
    tax_units_path = project_root / "src/data/processed/tax_units_irs_based.parquet"
    tax_units = pd.read_parquet(tax_units_path)
    
    # Get joint filer households
    joint_filers = tax_units[tax_units['filing_status'] == 'married_filing_jointly']
    joint_hh_ids = set(joint_filers['hh_id'].dropna())
    
    print("Households That Should Have Joint Filers")
    print("=" * 50)
    
    # Filter to adults and add person IDs
    adults = person_df[person_df['AGEP'] >= 18].copy()
    adults['person_id'] = adults['SERIALNO'].astype(str) + '_' + adults['SPORDER'].astype(str)
    
    # Find households with householder+spouse pairs
    spouse_households = []
    for serialno, hh_group in adults.groupby('SERIALNO'):
        relshipp_counts = hh_group['RELSHIPP'].value_counts()
        householders = relshipp_counts.get(20, 0)
        spouses = relshipp_counts.get(21, 0)
        
        if householders >= 1 and spouses >= 1:
            # This household has a householder+spouse pair
            married_adults = hh_group[hh_group['MAR'] == 1]
            spouse_households.append({
                'serialno': serialno,
                'total_adults': len(hh_group),
                'married_adults': len(married_adults),
                'householders': householders,
                'spouses': spouses,
                'has_joint_filer': serialno in joint_hh_ids
            })
    
    spouse_hh_df = pd.DataFrame(spouse_households)
    
    print(f"Total households with householder+spouse pairs: {len(spouse_hh_df):,}")
    print(f"Households with joint filers: {spouse_hh_df['has_joint_filer'].sum():,}")
    
    # Missing joint filers
    missing_joint = spouse_hh_df[spouse_hh_df['has_joint_filer'] == False]
    print(f"Households missing joint filers: {len(missing_joint):,}")
    print(f"Joint filer coverage: {spouse_hh_df['has_joint_filer'].mean()*100:.1f}%")
    
    return spouse_hh_df, missing_joint

def analyze_income_patterns():
    """Analyze income patterns of misclassified married adults."""
    logger.info("Analyzing income patterns...")
    
    # Load PUMS data
    loader = PUMSDataLoader()
    person_df, hh_df = loader.load_data(state=15, year=2022)
    
    # Load constructed tax units
    tax_units_path = project_root / "src/data/processed/tax_units_irs_based.parquet"
    tax_units = pd.read_parquet(tax_units_path)
    
    # Filter to adults and add person IDs
    adults = person_df[person_df['AGEP'] >= 18].copy()
    adults['person_id'] = adults['SERIALNO'].astype(str) + '_' + adults['SPORDER'].astype(str)
    
    print("Income Pattern Analysis")
    print("=" * 40)
    
    # Get filing status for each adult
    adult_filing_status = {}
    
    for _, unit in tax_units.iterrows():
        status = unit['filing_status']
        
        if pd.notna(unit.get('primary_filer_id')):
            adult_filing_status[unit['primary_filer_id']] = status
        if pd.notna(unit.get('secondary_filer_id')):
            adult_filing_status[unit['secondary_filer_id']] = status
    
    # Add filing status to adults dataframe
    adults['filing_status'] = adults['person_id'].map(adult_filing_status)
    
    # Analyze married adults by filing status
    married_adults = adults[adults['MAR'] == 1].copy()
    
    for status in ['single', 'married_filing_jointly', 'head_of_household', 'married_filing_separately']:
        status_group = married_adults[married_adults['filing_status'] == status]
        if len(status_group) > 0:
            median_income = status_group['PINCP'].median()
            mean_income = status_group['PINCP'].mean()
            print(f"{status}: {len(status_group):,} adults, median income: ${median_income:,.0f}, mean: ${mean_income:,.0f}")
    
    # High-income married single filers (potential misclassifications)
    married_single = married_adults[married_adults['filing_status'] == 'single']
    high_income_threshold = 50000  # $50k+
    high_income_married_single = married_single[married_single['PINCP'] >= high_income_threshold]
    
    print(f"\nHigh-income married adults filing single (${high_income_threshold:,}+): {len(high_income_married_single):,}")
    print(f"These are likely misclassifications - should be paired with spouses")
    
    return married_adults, high_income_married_single

def main():
    """Run comprehensive married misclassification analysis."""
    print("Married Adults Misclassification Analysis")
    print("=" * 60)
    
    try:
        # 1. Analyze married adults classified as single
        married_single, hh_analysis, problematic_hh = analyze_married_single_filers()
        
        # 2. Analyze households missing joint filers
        spouse_households, missing_joint = analyze_joint_filer_households()
        
        # 3. Analyze income patterns
        married_adults, high_income_single = analyze_income_patterns()
        
        print(f"\n" + "=" * 60)
        print("KEY FINDINGS:")
        print(f"=" * 60)
        print(f"1. Married adults filing single: {len(married_single):,}")
        print(f"2. Households with spouse pairs missing joint filers: {len(missing_joint):,}")
        print(f"3. High-income married adults filing single: {len(high_income_single):,}")
        print(f"4. These represent the ~5,000 units that need conversion")
        
        print(f"\nNEXT STEPS:")
        print(f"- Fix spouse identification logic to catch these cases")
        print(f"- Ensure householder+spouse pairs always create joint filers")
        print(f"- Review relationship code logic (RELSHIPP 20/21)")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()
