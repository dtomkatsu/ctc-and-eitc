"""
Test script for ML-based tax unit validation.

This script demonstrates how to use the ML validator to identify potential
misclassifications in tax unit filing statuses.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now import the module
from src.tax.units.validation.ml_validator import MLTaxUnitValidator, validate_tax_units

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_tax_units() -> list:
    """Create sample tax units for testing the ML validator."""
    return [
        {
            'id': 1,
            'filing_status': 'single',
            'income': 45000,
            'dependents': [
                {'age': 10, 'relationship': 'child'},
                {'age': 8, 'relationship': 'child'}
            ],
            'primary_filer': {'age': 35},
            'is_householder': 1,
            'housing_costs': 18000,
            'marital_status': 'single'
        },
        {
            'id': 2,
            'filing_status': 'hoh',
            'income': 250000,
            'dependents': [
                {'age': 15, 'relationship': 'child'}
            ],
            'primary_filer': {'age': 45},
            'is_householder': 1,
            'housing_costs': 30000,
            'marital_status': 'single'
        },
        {
            'id': 3,
            'filing_status': 'joint',
            'income': 120000,
            'dependents': [],
            'primary_filer': {'age': 40},
            'is_householder': 1,
            'housing_costs': 36000,
            'marital_status': 'married'
        },
        {
            'id': 4,
            'filing_status': 'single',
            'income': 30000,
            'dependents': [],
            'primary_filer': {'age': 28},
            'is_householder': 1,
            'housing_costs': 12000,
            'marital_status': 'single'
        }
    ]

def test_ml_validator():
    """Test the ML validator with sample tax units."""
    logger.info("Starting ML validator test...")
    
    # Create sample tax units
    tax_units = create_sample_tax_units()
    logger.info(f"Created {len(tax_units)} sample tax units")
    
    # Initialize validator
    validator = MLTaxUnitValidator()
    
    # Validate tax units
    validated_units = validator.predict(tax_units)
    
    # Print validation results
    print("\nValidation Results:")
    print("-" * 80)
    
    for unit in validated_units:
        print(f"\nTax Unit ID: {unit['id']}")
        print(f"Filing Status: {unit['filing_status'].upper()}")
        print(f"Income: ${unit['income']:,}")
        print(f"Dependents: {len(unit.get('dependents', []))}")
        
        flags = unit.get('validation_flags', [])
        if flags:
            print("\n  VALIDATION FLAGS:")
            for flag in flags:
                print(f"  - [{flag['code']}] {flag['message']} (Confidence: {flag['confidence']:.1f})")
        else:
            print("  No validation issues detected")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_ml_validator()
