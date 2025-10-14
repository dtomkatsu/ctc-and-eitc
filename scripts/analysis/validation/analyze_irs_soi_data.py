#!/usr/bin/env python3
"""
Analyze IRS SOI (Statistics of Income) data by ZIP code for Hawaii.
This script examines the 2022 IRS data to understand filing patterns and tax characteristics
that can inform our tax unit construction model and CTC estimates.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_irs_soi_data():
    """Load and examine the IRS SOI data by ZIP code."""
    data_path = Path('data/irs_soi/22zp12hi.xlsx')
    
    if not data_path.exists():
        logger.error(f"IRS SOI data file not found: {data_path}")
        return None
    
    try:
        # Read the Excel file - it may have multiple sheets
        excel_file = pd.ExcelFile(data_path)
        logger.info(f"Excel file has {len(excel_file.sheet_names)} sheets: {excel_file.sheet_names}")
        
        # Read the main data sheet (usually the first one)
        df = pd.read_excel(data_path, sheet_name=0)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        
        # Display basic info about the dataset
        logger.info("Dataset shape: " + str(df.shape))
        logger.info("Column names:")
        for i, col in enumerate(df.columns):
            logger.info(f"  {i+1:2d}. {col}")
        
        return df, excel_file.sheet_names
        
    except Exception as e:
        logger.error(f"Error loading IRS SOI data: {e}")
        return None, None

def analyze_data_structure(df):
    """Analyze the structure and content of the IRS SOI data."""
    logger.info("\n=== DATA STRUCTURE ANALYSIS ===")
    
    # Basic statistics
    logger.info(f"Total records: {len(df):,}")
    logger.info(f"Total columns: {len(df.columns)}")
    
    # Check for ZIP code column
    zip_cols = [col for col in df.columns if 'zip' in col.lower() or 'postal' in col.lower()]
    logger.info(f"Potential ZIP code columns: {zip_cols}")
    
    # Look for filing status columns
    filing_cols = [col for col in df.columns if any(term in col.lower() 
                   for term in ['filing', 'status', 'single', 'joint', 'head', 'married'])]
    logger.info(f"Potential filing status columns: {filing_cols}")
    
    # Look for income columns
    income_cols = [col for col in df.columns if any(term in col.lower() 
                   for term in ['income', 'agi', 'wages', 'salary'])]
    logger.info(f"Potential income columns: {income_cols}")
    
    # Look for dependent/child columns
    dependent_cols = [col for col in df.columns if any(term in col.lower() 
                      for term in ['dependent', 'child', 'ctc', 'credit'])]
    logger.info(f"Potential dependent/child columns: {dependent_cols}")
    
    # Show first few rows
    logger.info("\nFirst 5 rows:")
    print(df.head().to_string())
    
    # Show data types
    logger.info("\nData types:")
    print(df.dtypes.to_string())
    
    return {
        'zip_cols': zip_cols,
        'filing_cols': filing_cols,
        'income_cols': income_cols,
        'dependent_cols': dependent_cols
    }

def examine_filing_patterns(df, column_info):
    """Examine filing status patterns in the IRS data."""
    logger.info("\n=== FILING PATTERN ANALYSIS ===")
    
    # Try to identify filing status columns and calculate distributions
    filing_cols = column_info['filing_cols']
    
    if filing_cols:
        for col in filing_cols:
            if df[col].dtype in ['int64', 'float64']:
                logger.info(f"\nFiling status distribution for {col}:")
                print(df[col].describe())
    
    # Look for return counts by filing status
    return_cols = [col for col in df.columns if 'return' in col.lower() or 'count' in col.lower()]
    logger.info(f"Potential return count columns: {return_cols}")
    
    return return_cols

def examine_income_patterns(df, column_info):
    """Examine income patterns in the IRS data."""
    logger.info("\n=== INCOME PATTERN ANALYSIS ===")
    
    income_cols = column_info['income_cols']
    
    for col in income_cols[:5]:  # Examine first 5 income columns
        if col in df.columns and df[col].dtype in ['int64', 'float64']:
            logger.info(f"\nIncome statistics for {col}:")
            print(df[col].describe())

def create_zip_to_puma_mapping():
    """Create a mapping from ZIP codes to PUMAs for Hawaii."""
    logger.info("\n=== CREATING ZIP-TO-PUMA MAPPING ===")
    
    # This would typically require a crosswalk file
    # For now, we'll note this as a requirement
    logger.info("ZIP-to-PUMA mapping would require:")
    logger.info("1. HUD USPS ZIP-to-County crosswalk")
    logger.info("2. Census tract-to-ZIP relationships")
    logger.info("3. Our existing tract-to-PUMA mappings")
    
    return None

def propose_integration_strategy(df, column_info):
    """Propose how to integrate IRS SOI data with tax unit construction."""
    logger.info("\n=== INTEGRATION STRATEGY PROPOSAL ===")
    
    proposals = []
    
    # 1. Filing Status Calibration
    if column_info['filing_cols']:
        proposals.append({
            'strategy': 'Filing Status Calibration',
            'description': 'Use IRS filing status distributions by ZIP/geography to calibrate tax unit constructor',
            'implementation': [
                'Map ZIP codes to PUMAs using crosswalk files',
                'Calculate filing status percentages by PUMA from IRS data',
                'Use as targets for weight raking or probability adjustment',
                'Replace current SOI state-level targets with geographic-specific targets'
            ],
            'benefit': 'More accurate geographic filing status distributions'
        })
    
    # 2. Income Distribution Validation
    if column_info['income_cols']:
        proposals.append({
            'strategy': 'Income Distribution Validation',
            'description': 'Validate PUMS income distributions against IRS actual income by geography',
            'implementation': [
                'Compare PUMS AGI distributions to IRS AGI by ZIP/PUMA',
                'Identify systematic biases in PUMS income reporting',
                'Apply income adjustment factors if needed',
                'Validate CTC eligibility income thresholds'
            ],
            'benefit': 'More accurate income-based tax calculations'
        })
    
    # 3. CTC Validation and Benchmarking
    if column_info['dependent_cols']:
        proposals.append({
            'strategy': 'CTC Validation and Benchmarking',
            'description': 'Use actual IRS CTC claims to validate our CTC estimates',
            'implementation': [
                'Extract actual CTC amounts and recipient counts by ZIP',
                'Map to PUMAs and compare with our estimates',
                'Identify discrepancies in dependent counting or eligibility',
                'Adjust dependent assignment logic if needed'
            ],
            'benefit': 'Validation of CTC calculation accuracy'
        })
    
    # 4. Geographic Targeting
    proposals.append({
        'strategy': 'Geographic Targeting Enhancement',
        'description': 'Use ZIP-level data for more precise geographic analysis',
        'implementation': [
            'Create ZIP-to-PUMA crosswalk for Hawaii',
            'Aggregate IRS data to PUMA level',
            'Use as additional validation layer for county estimates',
            'Enable ZIP-level CTC impact analysis if needed'
        ],
        'benefit': 'Higher geographic precision and validation'
    })
    
    return proposals

def main():
    """Main analysis function."""
    logger.info("Starting IRS SOI data analysis for Hawaii...")
    
    # Load the data
    df, sheet_names = load_irs_soi_data()
    if df is None:
        return
    
    # Analyze data structure
    column_info = analyze_data_structure(df)
    
    # Examine patterns
    return_cols = examine_filing_patterns(df, column_info)
    examine_income_patterns(df, column_info)
    
    # Create mapping strategy
    create_zip_to_puma_mapping()
    
    # Propose integration strategies
    proposals = propose_integration_strategy(df, column_info)
    
    # Output proposals
    logger.info("\n=== INTEGRATION PROPOSALS ===")
    for i, proposal in enumerate(proposals, 1):
        logger.info(f"\n{i}. {proposal['strategy']}")
        logger.info(f"   Description: {proposal['description']}")
        logger.info(f"   Implementation:")
        for step in proposal['implementation']:
            logger.info(f"     - {step}")
        logger.info(f"   Benefit: {proposal['benefit']}")
    
    # Save processed data
    output_path = Path('data/irs_soi/hawaii_irs_soi_processed.csv')
    df.to_csv(output_path, index=False)
    logger.info(f"\nProcessed data saved to: {output_path}")
    
    return df, proposals

if __name__ == "__main__":
    df, proposals = main()
