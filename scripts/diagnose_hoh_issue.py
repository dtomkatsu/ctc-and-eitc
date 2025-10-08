"""
Diagnostic script to investigate why HoH qualification is only 3.4% instead of 9.6%.

This script will:
1. Load the PUMS data
2. Check how many people pass each HoH qualification test
3. Compare against the actual HoH filers in the tax units
4. Identify where the qualification is failing
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import logging
from src.data.pums_loader import PUMSDataLoader
from src.tax.units.status.hoh import is_head_of_household, _is_unmarried, _has_qualifying_person, _paid_half_home_cost

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_hoh_qualification():
    """Analyze HoH qualification at each step."""
    
    # Load PUMS data
    logger.info("Loading PUMS data...")
    loader = PUMSDataLoader()
    person_data, household_data = loader.load_data()
    
    logger.info(f"Loaded {len(person_data)} persons and {len(household_data)} households")
    
    # Merge person and household data
    merged = person_data.merge(
        household_data[['SERIALNO', 'WGTP', 'PUMA']],
        on='SERIALNO',
        how='left'
    )
    
    # Filter to adults only
    adults = merged[merged['AGEP'] >= 18].copy()
    logger.info(f"Found {len(adults)} adults")
    
    # Track qualification at each step
    results = {
        'total_adults': len(adults),
        'unmarried': 0,
        'has_qualifying_person': 0,
        'paid_half_home_cost': 0,
        'fully_qualified': 0,
        'with_dependents': 0,
        'without_dependents': 0
    }
    
    # Detailed breakdown by failure point
    failure_points = {
        'married': 0,
        'no_qualifying_person': 0,
        'no_home_cost': 0
    }
    
    # Track by marital status
    marital_status_counts = {}
    
    # Check each adult
    logger.info("Checking HoH qualification for each adult...")
    
    for idx, adult in adults.iterrows():
        mar = adult.get('MAR', 0)
        marital_status_counts[mar] = marital_status_counts.get(mar, 0) + 1
        
        # Test 1: Unmarried
        if not _is_unmarried(adult, merged):
            failure_points['married'] += 1
            continue
        results['unmarried'] += 1
        
        # Test 2: Has qualifying person
        if not _has_qualifying_person(adult, merged):
            failure_points['no_qualifying_person'] += 1
            continue
        results['has_qualifying_person'] += 1
        
        # Test 3: Paid half home cost
        if not _paid_half_home_cost(adult, merged):
            failure_points['no_home_cost'] += 1
            continue
        results['paid_half_home_cost'] += 1
        
        # Fully qualified
        results['fully_qualified'] += 1
        
        # Check if they have dependents in household
        household = merged[merged['SERIALNO'] == adult.get('SERIALNO')]
        children = household[
            (household['AGEP'] < 18) | 
            ((household['AGEP'] < 24) & (household['SCHL'] >= 16))
        ]
        
        if len(children) > 0:
            results['with_dependents'] += 1
        else:
            results['without_dependents'] += 1
    
    # Calculate percentages
    logger.info("\n" + "="*80)
    logger.info("HoH QUALIFICATION ANALYSIS")
    logger.info("="*80)
    
    logger.info(f"\nTotal adults: {results['total_adults']:,}")
    logger.info(f"\nMarital Status Distribution:")
    for mar, count in sorted(marital_status_counts.items()):
        mar_labels = {1: 'Married', 2: 'Widowed', 3: 'Divorced', 4: 'Separated', 5: 'Never married'}
        label = mar_labels.get(mar, f'Unknown ({mar})')
        pct = 100 * count / results['total_adults']
        logger.info(f"  {label}: {count:,} ({pct:.1f}%)")
    
    logger.info(f"\nQualification Steps:")
    logger.info(f"  1. Unmarried: {results['unmarried']:,} ({100*results['unmarried']/results['total_adults']:.1f}%)")
    logger.info(f"  2. Has qualifying person: {results['has_qualifying_person']:,} ({100*results['has_qualifying_person']/results['total_adults']:.1f}%)")
    logger.info(f"  3. Paid half home cost: {results['paid_half_home_cost']:,} ({100*results['paid_half_home_cost']/results['total_adults']:.1f}%)")
    logger.info(f"  FULLY QUALIFIED: {results['fully_qualified']:,} ({100*results['fully_qualified']/results['total_adults']:.1f}%)")
    
    logger.info(f"\nQualified HoH Breakdown:")
    logger.info(f"  With dependents: {results['with_dependents']:,} ({100*results['with_dependents']/results['fully_qualified']:.1f}% of qualified)")
    logger.info(f"  Without dependents: {results['without_dependents']:,} ({100*results['without_dependents']/results['fully_qualified']:.1f}% of qualified)")
    
    logger.info(f"\nFailure Points:")
    logger.info(f"  Failed at 'unmarried' test: {failure_points['married']:,} ({100*failure_points['married']/results['total_adults']:.1f}%)")
    logger.info(f"  Failed at 'qualifying person' test: {failure_points['no_qualifying_person']:,} ({100*failure_points['no_qualifying_person']/results['total_adults']:.1f}%)")
    logger.info(f"  Failed at 'home cost' test: {failure_points['no_home_cost']:,} ({100*failure_points['no_home_cost']/results['total_adults']:.1f}%)")
    
    # Load actual tax units to compare
    logger.info("\n" + "="*80)
    logger.info("ACTUAL TAX UNIT COMPARISON")
    logger.info("="*80)
    
    try:
        tax_units = pd.read_parquet('src/data/processed/tax_units_irs_based.parquet')
        
        total_units = len(tax_units)
        hoh_units = len(tax_units[tax_units['filing_status'] == 'head_of_household'])
        
        logger.info(f"\nTotal tax units: {total_units:,}")
        logger.info(f"HoH tax units: {hoh_units:,} ({100*hoh_units/total_units:.1f}%)")
        logger.info(f"\nExpected HoH (SOI): 9.6%")
        logger.info(f"Qualified adults: {100*results['fully_qualified']/results['total_adults']:.1f}%")
        logger.info(f"Actual HoH units: {100*hoh_units/total_units:.1f}%")
        
        logger.info(f"\nGap Analysis:")
        logger.info(f"  Qualified adults: {results['fully_qualified']:,}")
        logger.info(f"  Actual HoH units: {hoh_units:,}")
        logger.info(f"  Gap: {results['fully_qualified'] - hoh_units:,} ({100*(results['fully_qualified'] - hoh_units)/results['fully_qualified']:.1f}% loss rate)")
        
    except Exception as e:
        logger.error(f"Error loading tax units: {e}")
    
    return results

if __name__ == '__main__':
    analyze_hoh_qualification()
