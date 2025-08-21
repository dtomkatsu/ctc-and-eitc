#!/usr/bin/env python3
"""
Simple test script to check ML validator with lower thresholds.
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tax.units.validation.ml_validator import MLTaxUnitValidator
from src.tax.units.utils import setup_logging

# Configure logging
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_tax_units():
    """Create synthetic test tax units with potential issues."""
    test_units = [
        # Single filer with high income and dependents - might be HoH
        {
            'filer_id': 'test_1',
            'filing_status': 'single',
            'income': 75000,
            'num_dependents': 2,
            'dependents': ['child1', 'child2'],
            'primary_filer': {'age': 35, 'marital_status': 'single'},
            'is_householder': 1
        },
        # Head of household with very high income - might be incorrect
        {
            'filer_id': 'test_2', 
            'filing_status': 'head_of_household',
            'income': 250000,
            'num_dependents': 1,
            'dependents': ['child1'],
            'primary_filer': {'age': 55, 'marital_status': 'single'},
            'is_householder': 1
        },
        # Married person filing single - potential issue
        {
            'filer_id': 'test_3',
            'filing_status': 'single', 
            'income': 60000,
            'num_dependents': 1,
            'dependents': ['child1'],
            'primary_filer': {'age': 40, 'marital_status': 'married'},
            'is_householder': 1
        },
        # Normal joint filer - should be fine
        {
            'filer_id': 'test_4',
            'filing_status': 'joint',
            'income': 90000,
            'num_dependents': 2,
            'dependents': ['child1', 'child2'],
            'primary_filer': {'age': 32, 'marital_status': 'married'},
            'is_householder': 1
        }
    ]
    return test_units

def test_ml_validator_with_lower_threshold():
    """Test ML validator with modified confidence threshold."""
    model_path = "models/tax_unit_validator.joblib"
    
    if not os.path.exists(model_path):
        logger.error(f"ML model not found at {model_path}")
        return
    
    # Create test data
    test_units = create_test_tax_units()
    logger.info(f"Created {len(test_units)} test tax units")
    
    # Test ML validator
    ml_validator = MLTaxUnitValidator(model_path)
    
    # Temporarily lower confidence threshold for testing
    logger.info("Testing with original threshold (0.7)...")
    validated_units = ml_validator.predict(test_units)
    
    # Count flags
    flagged_count = sum(1 for unit in validated_units 
                       if 'validation_flags' in unit and unit['validation_flags'])
    
    logger.info(f"Original threshold results: {flagged_count}/{len(test_units)} units flagged")
    
    # Show detailed results
    for i, (original, validated) in enumerate(zip(test_units, validated_units)):
        logger.info(f"\nUnit {i+1}: {original['filing_status']} filer")
        logger.info(f"  Income: ${original['income']:,}, Dependents: {original['num_dependents']}")
        
        if 'validation_flags' in validated and validated['validation_flags']:
            logger.info(f"  Validation flags ({len(validated['validation_flags'])}):")
            for flag in validated['validation_flags']:
                logger.info(f"    - {flag.get('code', 'UNKNOWN')}: {flag.get('message', 'No message')}")
                logger.info(f"      Confidence: {flag.get('confidence', 0):.1%}")
        else:
            logger.info("  No validation flags")

if __name__ == "__main__":
    test_ml_validator_with_lower_threshold()
