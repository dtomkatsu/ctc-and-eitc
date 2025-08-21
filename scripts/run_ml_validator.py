#!/usr/bin/env python3
"""
Run ML validator on constructed tax units and analyze the results.
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tax.units.constructor import TaxUnitConstructor
from src.tax.units.validation.ml_validator import MLTaxUnitValidator
from src.tax.units.validation import ValidationIssue, ValidationSeverity

# Configuration
MODEL_PATH = 'models/tax_unit_validator.joblib'
OUTPUT_DIR = 'output/validation'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_sample_data():
    """Load sample data for testing."""
    # Example: Load a small sample of PUMS data
    pums_dir = 'data/raw/pums'
    person_file = os.path.join(pums_dir, 'psam_p15.csv')
    hh_file = os.path.join(pums_dir, 'psam_h15.csv')
    
    print("Loading sample data...")
    persons = pd.read_csv(person_file, dtype={'SERIALNO': str}, nrows=1000)
    households = pd.read_csv(hh_file, dtype={'SERIALNO': str}, nrows=100)
    
    # Filter to only include persons in our sampled households
    persons = persons[persons['SERIALNO'].isin(households['SERIALNO'])]
    
    return persons, households

def main():
    print("Starting ML Validator on Tax Units")
    print("=" * 50)
    
    # 1. Load sample data
    print("\n1. Loading sample data...")
    persons, households = load_sample_data()
    print(f"   - Loaded {len(persons)} persons and {len(households)} households")
    
    # 2. Construct tax units
    print("\n2. Constructing tax units...")
    constructor = TaxUnitConstructor(
        persons, 
        households,
        batch_size=100,
        num_processes=2,
        progress_bar=True,
        ml_model_path=MODEL_PATH if os.path.exists(MODEL_PATH) else None
    )
    
    # Create tax units
    tax_units = constructor.create_rule_based_units(parallel=True)
    
    if isinstance(tax_units, list):
        tax_units = pd.DataFrame(tax_units)
    
    print(f"   - Created {len(tax_units)} tax units")
    
    # 3. Run ML validation if not already done during construction
    if 'ml_validation_flags' not in tax_units.columns:
        print("\n3. Running ML validation...")
        ml_validator = MLTaxUnitValidator(MODEL_PATH if os.path.exists(MODEL_PATH) else None)
        
        # Convert DataFrame to list of dicts for ML validation
        tax_units_list = tax_units.to_dict('records')
        
        # Use predict method to validate tax units
        validated_units = ml_validator.predict(tax_units_list)
        
        # Add ML validation flags back to DataFrame
        tax_units['ml_validation_flags'] = [unit.get('ml_validation_flags', []) for unit in validated_units]
    
    # 4. Analyze validation results
    print("\n4. Analyzing validation results...")
    
    # Count flagged units
    flagged_units = tax_units[tax_units['ml_validation_flags'].apply(lambda x: len(x) > 0)]
    print(f"   - {len(flagged_units)} out of {len(tax_units)} tax units flagged for review ({len(flagged_units)/len(tax_units)*100:.1f}%)")
    
    if not flagged_units.empty:
        # Get unique validation issues
        all_issues = []
        for _, row in flagged_units.iterrows():
            for issue in row['ml_validation_flags']:
                all_issues.append({
                    'filer_id': row.get('filer_id', 'unknown'),
                    'filing_status': row.get('filing_status', 'unknown'),
                    'issue': issue.get('message', 'No message'),
                    'confidence': issue.get('confidence', 0.0),
                    'suggested_status': issue.get('suggested_status', 'none')
                })
        
        issues_df = pd.DataFrame(all_issues)
        
        if not issues_df.empty:
            # Summary of issues
            print("\n   - Validation issues by type:")
            issue_counts = issues_df['issue'].value_counts()
            for issue, count in issue_counts.items():
                print(f"     - {issue}: {count} units")
            
            # Save detailed results
            output_file = os.path.join(OUTPUT_DIR, 'ml_validation_results.csv')
            flagged_units.to_csv(output_file, index=False)
            print(f"\nDetailed validation results saved to: {output_file}")
    
    print("\nML validation complete!")

if __name__ == "__main__":
    main()
