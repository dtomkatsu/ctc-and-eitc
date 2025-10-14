#!/usr/bin/env python3
"""
Test script to validate that overcounting fixes are working correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import logging
from src.data.pums_loader import PUMSDataLoader
from src.tax.units.constructor import TaxUnitConstructor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_overcounting_fixes():
    """Test the comprehensive overcounting fixes."""
    logger.info("Testing overcounting fixes...")
    
    # Load data
    logger.info("Loading PUMS data...")
    pums_loader = PUMSDataLoader()
    person_df, hh_df = pums_loader.load_data()
    
    logger.info(f"Loaded {len(person_df):,} persons and {len(hh_df):,} households")
    
    # Create tax unit constructor with fixes
    logger.info("Creating tax units with overcounting fixes...")
    constructor = TaxUnitConstructor(
        person_df, 
        hh_df, 
        batch_size=1000, 
        num_processes=1,  # Use single process for easier debugging
        progress_bar=True
    )
    
    # Create tax units (this will automatically apply overcounting fixes)
    tax_units_df = constructor.create_rule_based_units(parallel=False)
    
    # Analyze results
    logger.info("Analyzing results...")
    
    # Basic statistics
    total_tax_units = len(tax_units_df)
    unique_households = len(person_df['SERIALNO'].unique())
    avg_units_per_household = total_tax_units / unique_households if unique_households > 0 else 0
    
    logger.info(f"Results:")
    logger.info(f"  Total tax units: {total_tax_units:,}")
    logger.info(f"  Unique households: {unique_households:,}")
    logger.info(f"  Avg units per household: {avg_units_per_household:.2f}")
    
    # Filing status distribution
    if not tax_units_df.empty:
        filing_status_counts = tax_units_df['filing_status'].value_counts()
        logger.info("Filing status distribution:")
        for status, count in filing_status_counts.items():
            pct = count / total_tax_units * 100
            logger.info(f"  {status}: {count:,} ({pct:.1f}%)")
    
    # Validation checks
    logger.info("Running validation checks...")
    
    # Check 1: Reasonable units per household ratio
    if avg_units_per_household > 2.0:
        logger.error(f"FAIL: Too many units per household ({avg_units_per_household:.2f} > 2.0)")
        return False
    elif avg_units_per_household > 1.5:
        logger.warning(f"WARNING: High units per household ({avg_units_per_household:.2f})")
    else:
        logger.info(f"PASS: Reasonable units per household ({avg_units_per_household:.2f})")
    
    # Check 2: No duplicate tax unit IDs
    if not tax_units_df.empty:
        duplicate_filer_ids = tax_units_df['filer_id'].duplicated().sum()
        if duplicate_filer_ids > 0:
            logger.error(f"FAIL: Found {duplicate_filer_ids} duplicate tax unit IDs")
            return False
        else:
            logger.info("PASS: No duplicate tax unit IDs")
    
    # Check 3: Valid filing status values
    if not tax_units_df.empty:
        valid_statuses = {'single', 'married_filing_jointly', 'married_filing_separately', 'head_of_household'}
        invalid_statuses = set(tax_units_df['filing_status'].unique()) - valid_statuses
        if invalid_statuses:
            logger.error(f"FAIL: Found invalid filing statuses: {invalid_statuses}")
            return False
        else:
            logger.info("PASS: All filing statuses are valid")
    
    # Check 4: Reasonable filing status distribution
    if not tax_units_df.empty:
        single_pct = (tax_units_df['filing_status'] == 'single').sum() / len(tax_units_df) * 100
        joint_pct = (tax_units_df['filing_status'] == 'married_filing_jointly').sum() / len(tax_units_df) * 100
        hoh_pct = (tax_units_df['filing_status'] == 'head_of_household').sum() / len(tax_units_df) * 100
        
        # Expected ranges based on Hawaii DOTAX data
        if not (40 <= single_pct <= 60):
            logger.warning(f"Single filers percentage ({single_pct:.1f}%) outside expected range (40-60%)")
        if not (25 <= joint_pct <= 45):
            logger.warning(f"Joint filers percentage ({joint_pct:.1f}%) outside expected range (25-45%)")
        if not (5 <= hoh_pct <= 15):
            logger.warning(f"Head of Household percentage ({hoh_pct:.1f}%) outside expected range (5-15%)")
    
    # Check validation summary from constructor
    if hasattr(constructor, 'validation_summary'):
        summary = constructor.validation_summary
        logger.info("Constructor validation summary:")
        logger.info(f"  Overcounting fixes applied: {summary.get('overcounting_fixes_applied', False)}")
        logger.info(f"  Final tax units: {summary.get('final_tax_units', 'N/A')}")
        
        if 'fixes_validation' in summary:
            fix_validation = summary['fixes_validation']
            logger.info(f"  Fix validation results:")
            logger.info(f"    Issues detected: {len(fix_validation.get('issues', []))}")
            for issue in fix_validation.get('issues', []):
                logger.info(f"      - {issue}")
    
    # Check 5: Coverage validation
    logger.info("Validating person coverage...")
    all_person_ids = set(person_df['SERIALNO'].astype(str) + '_' + person_df['SPORDER'].astype(str))
    covered_person_ids = set()
    
    if not tax_units_df.empty:
        for _, tax_unit in tax_units_df.iterrows():
            # Add primary filer
            if pd.notna(tax_unit['primary_filer_id']):
                covered_person_ids.add(str(tax_unit['primary_filer_id']))
            
            # Add secondary filer if exists
            if pd.notna(tax_unit.get('secondary_filer_id')):
                covered_person_ids.add(str(tax_unit['secondary_filer_id']))
            
            # Add dependents
            dependents = tax_unit['dependents']
            try:
                if pd.isna(dependents) or dependents is None:
                    continue
                    
                # Handle numpy arrays
                if hasattr(dependents, 'size'):
                    if dependents.size > 0:
                        covered_person_ids.update(str(dep_id) for dep_id in dependents.flat)
                # Handle lists/tuples
                elif isinstance(dependents, (list, tuple)):
                    if dependents:  # Check if non-empty
                        covered_person_ids.update(str(dep_id) for dep_id in dependents)
                # Handle strings
                elif isinstance(dependents, str) and dependents:
                    covered_person_ids.add(str(dependents))
                # Handle other iterables
                elif hasattr(dependents, '__iter__'):
                    try:
                        covered_person_ids.update(str(dep_id) for dep_id in dependents)
                    except (TypeError, ValueError):
                        logger.warning(f"Could not process dependents: {dependents}")
            except Exception as e:
                logger.warning(f"Error processing dependents: {e}")
                continue
    
    uncovered_persons = len(all_person_ids - covered_person_ids)
    coverage_pct = (len(covered_person_ids) / len(all_person_ids)) * 100 if all_person_ids else 0
    
    logger.info(f"Person coverage: {coverage_pct:.1f}% ({len(covered_person_ids):,} of {len(all_person_ids):,})")
    logger.info(f"Uncovered persons: {uncovered_persons:,}")
    
    if coverage_pct < 90:
        logger.warning(f"Low person coverage ({coverage_pct:.1f}% < 90%)")
    else:
        logger.info("PASS: Good person coverage")
    
    # Final assessment
    logger.info("="*60)
    if avg_units_per_household <= 2.0 and duplicate_filer_ids == 0:
        logger.info("OVERALL: Overcounting fixes appear to be working correctly!")
        return True
    else:
        logger.error("OVERALL: Overcounting issues may still exist")
        return False

def compare_before_after():
    """Compare results before and after fixes by temporarily disabling fixes."""
    logger.info("Comparing results before and after overcounting fixes...")
    
    # This would require modifying the constructor to optionally disable fixes
    # For now, just log that this comparison would be valuable
    logger.info("Note: To fully validate fixes, consider running with fixes disabled for comparison")

if __name__ == "__main__":
    success = test_overcounting_fixes()
    if success:
        logger.info("All tests passed!")
        sys.exit(0)
    else:
        logger.error("Some tests failed!")
        sys.exit(1)
