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
    """Load sample PUMS data for testing."""
    data_dir = Path("data/raw/pums")
    
    # Load person and household data using correct file names
    person_df = pd.read_csv(data_dir / "psam_p15.csv")
    hh_df = pd.read_csv(data_dir / "psam_h15.csv")
    
    # Take a small sample for testing - ensure matching household IDs
    sample_hh_ids = hh_df['SERIALNO'].head(50).tolist()
    hh_df = hh_df[hh_df['SERIALNO'].isin(sample_hh_ids)]
    person_df = person_df[person_df['SERIALNO'].isin(sample_hh_ids)]
    
    logger.info(f"Sample data: {len(person_df)} persons in {len(hh_df)} households")
    
    return person_df, hh_df

def main():
    """Test ML validator integration with tax unit construction."""
    # Path to trained ML model
    model_path = "models/tax_unit_validator.joblib"
    
    if not os.path.exists(model_path):
        logger.error(f"ML model not found at {model_path}. Please train the model first.")
        return
    
    # Load sample data
    logger.info("Loading sample data...")
    person_df, hh_df = load_sample_data()
    
    # Initialize tax unit constructor with ML validation
    logger.info("Initializing tax unit constructor with ML validation...")
    constructor = TaxUnitConstructor(
        person_df=person_df,
        hh_df=hh_df,
        batch_size=100,  # Smaller batch size for testing
        progress_bar=True,
        ml_model_path=model_path
    )
    
    # Create tax units
    logger.info("Creating tax units with ML validation...")
    tax_units = constructor.create_rule_based_units()
    
    # Analyze results
    if tax_units is not None and not tax_units.empty:
        logger.info(f"Created {len(tax_units)} tax units")
        
        # Count units with ML validation flags
        if 'ml_validation_flags' in tax_units.columns:
            flagged_count = tax_units['ml_validation_flags'].notna().sum()
            logger.info(f"Found {flagged_count} tax units with ML validation flags")
            
            # Print examples of flagged units
            if flagged_count > 0:
                flagged = tax_units[tax_units['ml_validation_flags'].notna()]
                logger.info("\nExamples of flagged tax units:")
                for _, row in flagged.head(3).iterrows():
                    print(f"\nFiler ID: {row['filer_id']}")
                    print(f"Filing Status: {row['filing_status']}")
                    print(f"Income: ${row.get('income', 0):,.2f}")
                    print(f"Dependents: {row.get('num_dependents', 0)}")
                    print("ML Validation Flags:")
                    for flag in row['ml_validation_flags']:
                        print(f"  - {flag['message']} (Confidence: {flag.get('confidence', 0.0):.2f})")
    else:
        logger.warning("No tax units were created")

if __name__ == "__main__":
    main()
