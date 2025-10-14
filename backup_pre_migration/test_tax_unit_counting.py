"""
Test script to verify tax unit counting in the tax unit constructor.

This script tests the updated tax unit counting logic to ensure all adults
are properly assigned to tax units and dependents are correctly allocated.
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('tax_unit_test.log')
    ]
)
logger = logging.getLogger(__name__)

def load_test_data():
    """Load test data for tax unit counting verification."""
    try:
        # Try to load the test data
        from src.data.pums_loader import PUMSDataLoader
        
        # Initialize the data loader
        data_loader = PUMSDataLoader()
        
        # Load a sample of the data (adjust sample size as needed)
        sample_size = 1000
        person_df, hh_df = data_loader.load_pums_data(sample_size=sample_size)
        
        logger.info(f"Loaded {len(person_df)} persons and {len(hh_df)} households")
        return person_df, hh_df
        
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        # Fall back to sample data if loading fails
        return create_sample_data()

def create_sample_data():
    """Create sample data for testing if real data is not available."""
    logger.info("Creating sample test data...")
    
    # Sample household data with all required columns
    hh_data = {
        'SERIALNO': ['HH1', 'HH2', 'HH3', 'HH4'],
        'ADJINC': [1.0, 1.0, 1.0, 1.0],
        'WGTP': [100, 100, 100, 100],
        'NP': [4, 2, 2, 3],  # Number of persons in household
        'TYPE': [1, 1, 1, 1],  # Housing unit type (1=Housing unit)
        'TEN': [1, 1, 2, 3],  # Tenure (1=Owned with mortgage, 2=Owned free and clear, 3=Rented)
        'HINCP': [95000, 60000, 35000, 105000],  # Household income
        'HUPAC': [1, 1, 1, 1],  # Housing unit presence and type of AC
        'ACCESSINET': [1, 1, 2, 1],  # Access to the internet
        'BROADBND': [1, 1, 2, 1],  # Broadband internet service
    }
    
    # Sample person data with different household structures and required columns
    person_data = {
        'SERIALNO': ['HH1', 'HH1', 'HH1', 'HH1', 'HH2', 'HH2', 'HH3', 'HH3', 'HH4', 'HH4', 'HH4'],
        'SPORDER': [1, 2, 3, 4, 1, 2, 1, 2, 1, 2, 3],
        'RELSHIPP': [1, 2, 3, 4, 1, 2, 1, 2, 1, 2, 3],  # 1=Householder, 2=Spouse, 3=Child, 4=Other relative
        'AGEP': [35, 34, 10, 5, 45, 43, 30, 8, 50, 48, 20],
        'MAR': [1, 1, 0, 0, 1, 1, 6, 0, 1, 1, 0],  # 1=Married, 6=Never married
        'SEX': [1, 2, 1, 2, 1, 2, 2, 1, 1, 2, 2],
        'PINCP': [50000, 45000, 0, 0, 60000, 0, 35000, 0, 55000, 50000, 15000],
        'WAGP': [48000, 43000, 0, 0, 58000, 0, 34000, 0, 53000, 48000, 14000],
        'SCHL': [21, 19, 4, 1, 20, 21, 19, 3, 21, 21, 18],  # Education level
        'DIS': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Disability status
        'MIL': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Military service
        'MILITARY': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Military service type
        'CIT': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Citizenship status
        'NATIVITY': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Nativity
        'MIG': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Mobility status
        'MIGSAME': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Same house 1 year ago
        'MIGPUMA': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Migration PUMA
        'MIGSP': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Migration state
        'POBP': [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15],  # Place of birth (state)
        'POWPUMA': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Place of work PUMA
        'POWSP': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Place of work state
        'RAC1P': [1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5],  # Race
        'RACASIAN': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Asian race
        'RACBLK': [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # Black race
        'RACNHPI': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Native Hawaiian/Pacific Islander
        'RACNUM': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Number of major race groups
        'RACPI': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Pacific Islander
        'RACSOR': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Some other race
        'RACWHT': [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # White race
        'SCIENGP': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Science and engineering field degree
        'SCIENGRLP': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Science and engineering related field degree
        'SFN': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Subfamily number
        'SFR': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Subfamily relationship
        'SOCP': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # SOC code
        'VPS': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Veteran period of service
        'WAOB': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # World area of birth
        'FOD1P': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Field of degree
        'FOD2P': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Field of degree 2
        'HICOV': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Health insurance coverage
        'HINS1': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Insurance through employer
        'HINS2': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Insurance purchased directly
        'HINS3': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Medicare
        'HINS4': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Medicaid
        'HINS5': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # TRICARE
        'HINS6': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # VA Health Care
        'HINS7': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Indian Health Service
        'INDP': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Industry code
        'JWTR': [1, 1, 0, 0, 1, 2, 1, 0, 1, 1, 1],  # Means of transportation to work
        'JWTRNS': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Public transportation to work
        'LANX': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Language other than English spoken at home
        'MARHD': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Divorced in the past year
        'MARHM': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Married in the past year
        'MARHT': [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],  # Times married
        'MSP': [1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0],  # Married, spouse present/absent
        'NAICSP': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # NAICS industry code
        'NATIVITY': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Nativity
        'NOP': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # N/A
        'OC': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Occupation code
        'OCCP': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Occupation code
        'PAOC': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Presence and age of own children
        'PERNP': [50000, 45000, 0, 0, 60000, 0, 35000, 0, 55000, 50000, 15000],  # Total person's earnings
        'PINCP': [50000, 45000, 0, 0, 60000, 0, 35000, 0, 55000, 50000, 15000],  # Total person's income
        'POBP': [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15],  # Place of birth (state)
        'POWPUMA': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Place of work PUMA
        'POWSP': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Place of work state
        'PRIVCOV': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Private health insurance coverage
        'PUBCOV': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Public health insurance coverage
        'QTRBIR': [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3],  # Quarter of birth
        'RAC1P': [1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5],  # Race
        'RAC2P': [1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5],  # Race (2+ races)
        'RAC3P': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Race (3+ races)
        'RACAIAN': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # American Indian/Alaska Native
        'RACASN': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],  # Asian
        'RACBLK': [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # Black/African American
        'RACNH': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Native Hawaiian
        'RACNUM': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Number of major race groups
        'RACPI': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Other Pacific Islander
        'RACSOR': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Some other race
        'RACWHT': [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # White
        'RC': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Relationship to householder
        'SCH': [1, 1, 3, 4, 1, 1, 1, 4, 1, 1, 1],  # School enrollment
        'SCHG': [0, 0, 2, 1, 0, 0, 0, 1, 0, 0, 0],  # Grade level attending
        'SCHL': [21, 19, 4, 1, 20, 21, 19, 3, 21, 21, 18],  # Educational attainment
        'SEMP': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Self-employment income
        'SEX': [1, 2, 1, 2, 1, 2, 2, 1, 1, 2, 2],  # Sex
        'SSIP': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Supplemental Security Income
        'SSP': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Social Security income
        'WAGP': [48000, 43000, 0, 0, 58000, 0, 34000, 0, 53000, 48000, 14000],  # Wages or salary income
        'WKHP': [40, 35, 0, 0, 45, 0, 30, 0, 40, 35, 20],  # Usual hours worked per week
        'WKL': [1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1],  # When last worked
        'WKW': [1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1],  # Weeks worked in past 12 months
        'WRK': [1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1],  # Worked last week
        'YOEP': [2000, 2002, 0, 0, 1995, 0, 2010, 0, 1990, 1992, 2015],  # Year of entry to the U.S.
    }
    
    person_df = pd.DataFrame(person_data)
    hh_df = pd.DataFrame(hh_data)
    
    # Set index for person_df and ensure SERIALNO is a column
    person_df.index = [f"P{i+1}" for i in range(len(person_df))]
    person_df['SERIALNO'] = person_data['SERIALNO']  # Ensure SERIALNO is a column
    
    # Set index for hh_df and ensure SERIALNO is a column
    hh_df = pd.DataFrame(hh_data)
    hh_df['SERIALNO'] = hh_data['SERIALNO']  # Ensure SERIALNO is a column
    hh_df = hh_df.set_index('SERIALNO')
    
    return person_df, hh_df

def test_tax_unit_counting():
    """Test the tax unit counting logic."""
    from src.tax.units.constructor import TaxUnitConstructor
    
    # Load test data
    person_df, hh_df = load_test_data()
    
    # Initialize the tax unit constructor
    logger.info("Initializing TaxUnitConstructor...")
    constructor = TaxUnitConstructor(person_df, hh_df)
    
    # Create tax units
    logger.info("Creating tax units...")
    tax_units = constructor.create_rule_based_units()
    
    # Analyze results
    logger.info("\n=== Test Results ===")
    logger.info(f"Total tax units created: {len(tax_units)}")
    
    # Count by filing status
    status_counts = {}
    for unit in tax_units:
        status = unit['filing_status']
        status_counts[status] = status_counts.get(status, 0) + 1
    
    logger.info("Tax units by filing status:")
    for status, count in status_counts.items():
        logger.info(f"  {status}: {count}")
    
    # Count total dependents
    total_dependents = sum(unit['num_dependents'] for unit in tax_units)
    logger.info(f"Total dependents assigned: {total_dependents}")
    
    # Verify all adults are in a tax unit
    adult_ids = set(person_df[person_df['AGEP'] >= 18].index)
    processed_adults = set()
    for unit in tax_units:
        processed_adults.add(unit['primary_filer_id'])
        if unit['secondary_filer_id']:
            processed_adults.add(unit['secondary_filer_id'])
    
    unassigned_adults = adult_ids - processed_adults
    if unassigned_adults:
        logger.warning(f"WARNING: {len(unassigned_adults)} adults not assigned to any tax unit")
    else:
        logger.info("SUCCESS: All adults assigned to tax units")
    
    return tax_units

if __name__ == "__main__":
    logger.info("Starting tax unit counting test...")
    tax_units = test_tax_unit_counting()
    logger.info("Test completed. Check tax_unit_test.log for detailed results.")
