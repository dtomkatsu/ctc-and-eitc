"""
Test script for ML validator integration with tax unit construction.
"""
import os
import sys
import logging
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.tax.units.constructor import TaxUnitConstructor
from src.tax.units.validation.ml_validator import MLTaxUnitValidator
from src.tax.units.utils import setup_logging

# Configure logging
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_sample_data():
    """Load sample PUMS data for testing with diverse household structures."""
    data_dir = Path("data/raw/pums")
    
    # Load person and household data using correct file names
    person_df = pd.read_csv(data_dir / "psam_p15.csv")
    hh_df = pd.read_csv(data_dir / "psam_h15.csv")
    
    # Get household sizes to ensure diversity
    hh_sizes = person_df.groupby('SERIALNO').size()
    
    # Select diverse household types
    diverse_hh_ids = []
    
    # Single-person households (10)
    single_hh = hh_sizes[hh_sizes == 1].head(10).index.tolist()
    diverse_hh_ids.extend(single_hh)
    
    # Two-person households (15) - likely couples
    two_person_hh = hh_sizes[hh_sizes == 2].head(15).index.tolist()
    diverse_hh_ids.extend(two_person_hh)
    
    # Three-person households (10) - families with children
    three_person_hh = hh_sizes[hh_sizes == 3].head(10).index.tolist()
    diverse_hh_ids.extend(three_person_hh)
    
    # Four+ person households (10) - larger families
    large_hh = hh_sizes[hh_sizes >= 4].head(10).index.tolist()
    diverse_hh_ids.extend(large_hh)
    
    # Filter data to selected households
    hh_df = hh_df[hh_df['SERIALNO'].isin(diverse_hh_ids)]
    person_df = person_df[person_df['SERIALNO'].isin(diverse_hh_ids)]
    
    # Log household structure diversity
    final_sizes = person_df.groupby('SERIALNO').size()
    size_dist = final_sizes.value_counts().sort_index()
    
    logger.info(f"Sample data: {len(person_df)} persons in {len(hh_df)} households")
    logger.info("Household size distribution:")
    for size, count in size_dist.items():
        logger.info(f"  {size} persons: {count} households")
    
    return person_df, hh_df

def analyze_ml_validation_results(tax_units):
    """Analyze and display ML validation results."""
    # Count tax units with ML validation flags
    ml_flagged = [tu for tu in tax_units if 'ml_validation_flags' in tu and tu['ml_validation_flags']]
    
    # Count by filing status
    status_counts = {}
    for tu in tax_units:
        status = tu.get('filing_status', 'unknown')
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Count flags by type
    flag_types = {}
    for tu in ml_flagged:
        for flag in tu['ml_validation_flags']:
            flag_type = flag['issue']
            flag_types[flag_type] = flag_types.get(flag_type, 0) + 1
    
    # Calculate statistics
    total_units = len(tax_units)
    flagged_pct = (len(ml_flagged) / total_units * 100) if total_units > 0 else 0
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("ML VALIDATION RESULTS")
    logger.info("="*50)
    logger.info(f"Total tax units processed: {total_units}")
    logger.info(f"Tax units with ML validation flags: {len(ml_flagged)} ({flagged_pct:.1f}%)")
    
    logger.info("\nFiling Status Distribution:")
    for status, count in sorted(status_counts.items()):
        logger.info(f"- {status}: {count} units ({(count/total_units*100):.1f}%)")
    
    if flag_types:
        logger.info("\nValidation Flags by Type:")
        for flag_type, count in sorted(flag_types.items()):
            logger.info(f"- {flag_type}: {count} occurrences")
    
    # Show detailed examples of flagged tax units
    if ml_flagged:
        logger.info("\n" + "-"*50)
        logger.info("DETAILED EXAMPLES OF FLAGGED TAX UNITS")
        logger.info("-"*50)
        
        for i, tu in enumerate(ml_flagged[:min(3, len(ml_flagged))]):
            logger.info(f"\nExample {i+1}:")
            logger.info(f"- Filing Status: {tu.get('filing_status', 'unknown')}")
            logger.info(f"- Members: {tu.get('num_dependents', 0) + 1}")
            logger.info(f"- Total Income: ${tu.get('income', 0):,.2f}")
            logger.info("ML Validation Flags:")
            for flag in tu['ml_validation_flags']:
                logger.info(f"  - {flag['issue']} (Confidence: {flag.get('confidence', 0):.1%})")
                logger.info(f"    Suggested Status: {flag.get('suggested_status', 'N/A')}")
                if 'details' in flag:
                    logger.info(f"    Details: {flag['details']}")
    
    logger.info("\n" + "="*50)
    logger.info("ML VALIDATION COMPLETE")
    logger.info("="*50)

def main():
    """Test ML validator integration with tax unit construction."""
    # Path to trained ML model
    model_path = "models/tax_unit_validator.joblib"
    
    if not os.path.exists(model_path):
        logger.error(f"ML model not found at {model_path}. Please train the model first.")
        return
    
    logger.info("Loading sample data...")
    person_df, hh_df = load_sample_data()
    
    logger.info("Initializing tax unit constructor with ML validation...")
    constructor = TaxUnitConstructor(
        person_df=person_df,
        hh_df=hh_df,
        ml_model_path=model_path,
        batch_size=100,
        num_processes=os.cpu_count() - 1 if os.cpu_count() > 1 else 1
    )
    
    logger.info("Creating tax units with ML validation...")
    tax_units_df = constructor.create_rule_based_units()
    
    # Debug: Test ML validator directly on a few tax units
    logger.info("Testing ML validator directly...")
    if not tax_units_df.empty:
        # Get first few tax units as dictionaries
        sample_units = tax_units_df.head(5).to_dict('records')
        
        # Test ML validator directly
        ml_validator = MLTaxUnitValidator(model_path)
        validated_units = ml_validator.predict(sample_units)
        
        logger.info(f"Direct ML validation test on {len(sample_units)} units:")
        for i, (original, validated) in enumerate(zip(sample_units, validated_units)):
            logger.info(f"  Unit {i+1}: {original.get('filing_status')} -> {validated.get('filing_status')}")
            if 'validation_flags' in validated and validated['validation_flags']:
                logger.info(f"    Flags: {len(validated['validation_flags'])}")
            else:
                logger.info(f"    No validation flags")
    
    # Convert DataFrame to list of tax unit objects for analysis
    if not tax_units_df.empty:
        # Get the tax units from the 'tax_unit' column if it exists
        if 'tax_unit' in tax_units_df.columns:
            tax_units = tax_units_df['tax_unit'].tolist()
        else:
            tax_units = tax_units_df.to_dict('records')
        
        # Analyze and display results
        analyze_ml_validation_results(tax_units)
    else:
        logger.warning("No tax units were created")

if __name__ == "__main__":
    main()
