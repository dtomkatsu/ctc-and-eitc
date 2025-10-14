#!/usr/bin/env python3
"""
Test ML validator with synthetic diverse tax units to see if it can flag misclassifications.
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tax.units.validation.ml_validator import MLTaxUnitValidator

# Configuration
MODEL_PATH = 'models/tax_unit_validator.joblib'

def create_synthetic_tax_units():
    """Create diverse synthetic tax units with known patterns."""
    tax_units = [
        # Clear single filer cases
        {
            'filer_id': 'single_1',
            'filing_status': 'single',
            'income': 45000,
            'num_dependents': 0,
            'primary_filer': {'age': 25, 'marital_status': 'never_married'},
            'housing_cost_ratio': 0.25,
            'is_householder': True
        },
        {
            'filer_id': 'single_2', 
            'filing_status': 'single',
            'income': 65000,
            'num_dependents': 0,
            'primary_filer': {'age': 35, 'marital_status': 'divorced'},
            'housing_cost_ratio': 0.30,
            'is_householder': True
        },
        
        # Clear joint filer cases
        {
            'filer_id': 'joint_1',
            'filing_status': 'married_filing_jointly',
            'income': 85000,
            'num_dependents': 2,
            'primary_filer': {'age': 32, 'marital_status': 'married'},
            'housing_cost_ratio': 0.20,
            'is_householder': True
        },
        {
            'filer_id': 'joint_2',
            'filing_status': 'married_filing_jointly', 
            'income': 120000,
            'num_dependents': 1,
            'primary_filer': {'age': 40, 'marital_status': 'married'},
            'housing_cost_ratio': 0.15,
            'is_householder': True
        },
        
        # Clear head of household cases
        {
            'filer_id': 'hoh_1',
            'filing_status': 'head_of_household',
            'income': 55000,
            'num_dependents': 2,
            'primary_filer': {'age': 28, 'marital_status': 'never_married'},
            'housing_cost_ratio': 0.35,
            'is_householder': True
        },
        {
            'filer_id': 'hoh_2',
            'filing_status': 'head_of_household',
            'income': 42000,
            'num_dependents': 1,
            'primary_filer': {'age': 35, 'marital_status': 'divorced'},
            'housing_cost_ratio': 0.40,
            'is_householder': True
        },
        
        # Potential misclassifications (should be flagged)
        {
            'filer_id': 'wrong_single',
            'filing_status': 'single',  # Wrong - should be HoH
            'income': 48000,
            'num_dependents': 2,  # Has dependents but filing single
            'primary_filer': {'age': 30, 'marital_status': 'never_married'},
            'housing_cost_ratio': 0.45,
            'is_householder': True
        },
        {
            'filer_id': 'wrong_joint',
            'filing_status': 'married_filing_jointly',  # Wrong - should be single
            'income': 35000,
            'num_dependents': 0,  # No dependents, not married
            'primary_filer': {'age': 26, 'marital_status': 'never_married'},
            'housing_cost_ratio': 0.25,
            'is_householder': True
        },
        {
            'filer_id': 'wrong_hoh',
            'filing_status': 'head_of_household',  # Wrong - should be single
            'income': 40000,
            'num_dependents': 0,  # No dependents
            'primary_filer': {'age': 45, 'marital_status': 'divorced'},
            'housing_cost_ratio': 0.30,
            'is_householder': True
        }
    ]
    
    return tax_units

def test_ml_validator_with_synthetic_data():
    """Test ML validator with synthetic diverse data."""
    print("Testing ML Validator with Synthetic Data")
    print("=" * 50)
    
    # 1. Create synthetic tax units
    print("\n1. Creating synthetic tax units...")
    tax_units = create_synthetic_tax_units()
    print(f"   - Created {len(tax_units)} synthetic tax units")
    
    # Show the test cases
    print("\n   - Test cases:")
    for unit in tax_units:
        age = unit['primary_filer']['age']
        marital = unit['primary_filer']['marital_status']
        print(f"     {unit['filer_id']}: {unit['filing_status']}, "
              f"income=${unit['income']:,}, deps={unit['num_dependents']}, "
              f"age={age}, marital={marital}")
    
    # 2. Initialize ML validator
    print("\n2. Initializing ML validator...")
    ml_validator = MLTaxUnitValidator(MODEL_PATH if os.path.exists(MODEL_PATH) else None)
    
    if ml_validator.model is None:
        print("   - ERROR: No ML model loaded!")
        return
    
    print(f"   - Model loaded successfully")
    
    # 3. Extract features
    print("\n3. Extracting features...")
    features = ml_validator.extract_features(tax_units)
    print(f"   - Feature shape: {features.shape}")
    
    # Show feature diversity
    print("\n   - Feature diversity:")
    for col in features.columns:
        unique_vals = features[col].nunique()
        print(f"     {col}: {unique_vals} unique values (range: {features[col].min():.2f} - {features[col].max():.2f})")
    
    # 4. Get predictions
    print("\n4. Getting ML predictions...")
    probas = ml_validator.model.predict_proba(features)
    predicted_classes = ml_validator.model.classes_
    
    # 5. Analyze each prediction
    print("\n5. Detailed prediction analysis:")
    print("   " + "="*80)
    
    flagged_count = 0
    for i, unit in enumerate(tax_units):
        current_status = str(unit.get('filing_status', '')).lower()
        max_idx = np.argmax(probas[i])
        predicted_status = str(predicted_classes[max_idx]).lower()
        confidence = float(probas[i][max_idx])
        
        # Normalize status names for comparison
        current_normalized = current_status.replace('married_filing_jointly', 'joint').replace('head_of_household', 'hoh')
        
        would_flag = predicted_status != current_normalized and confidence > 0.7
        if would_flag:
            flagged_count += 1
        
        print(f"   {unit['filer_id']}:")
        print(f"     Current: {current_status} -> Predicted: {predicted_status}")
        print(f"     Confidence: {confidence:.3f} | Would flag: {would_flag}")
        
        # Show all class probabilities
        for j, cls in enumerate(predicted_classes):
            prob = probas[i][j]
            marker = " <-- PREDICTED" if j == max_idx else ""
            print(f"     {cls}: {prob:.3f}{marker}")
        print()
    
    # 6. Test with different thresholds
    print(f"6. Flagging analysis with different thresholds:")
    for threshold in [0.4, 0.5, 0.6, 0.7, 0.8]:
        flags = 0
        for i, unit in enumerate(tax_units):
            current_status = str(unit.get('filing_status', '')).lower()
            max_idx = np.argmax(probas[i])
            predicted_status = str(predicted_classes[max_idx]).lower()
            confidence = float(probas[i][max_idx])
            
            current_normalized = current_status.replace('married_filing_jointly', 'joint').replace('head_of_household', 'hoh')
            
            if predicted_status != current_normalized and confidence > threshold:
                flags += 1
        
        print(f"   - Threshold {threshold}: {flags} units would be flagged")
    
    # 7. Run actual ML validator
    print(f"\n7. Running actual ML validator predict method...")
    validated_units = ml_validator.predict(tax_units.copy())
    
    actual_flags = sum(1 for unit in validated_units if unit.get('validation_flags'))
    print(f"   - Actual flags from predict method: {actual_flags}")
    
    if actual_flags > 0:
        print("   - Flagged units:")
        for unit in validated_units:
            if unit.get('validation_flags'):
                print(f"     {unit['filer_id']}: {len(unit['validation_flags'])} flags")
                for flag in unit['validation_flags']:
                    print(f"       - {flag.get('message', 'No message')}")

if __name__ == "__main__":
    test_ml_validator_with_synthetic_data()
