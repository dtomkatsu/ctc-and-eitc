#!/usr/bin/env python3
"""
Debug ML validator to understand why no units are being flagged.
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

# Configuration
MODEL_PATH = 'models/tax_unit_validator.joblib'

def debug_ml_predictions():
    """Debug ML validator predictions in detail."""
    print("ML Validator Debug Analysis")
    print("=" * 50)
    
    # 1. Load sample data
    print("\n1. Loading sample data...")
    pums_dir = 'data/raw/pums'
    person_file = os.path.join(pums_dir, 'psam_p15.csv')
    hh_file = os.path.join(pums_dir, 'psam_h15.csv')
    
    persons = pd.read_csv(person_file, dtype={'SERIALNO': str}, nrows=500)
    households = pd.read_csv(hh_file, dtype={'SERIALNO': str}, nrows=50)
    persons = persons[persons['SERIALNO'].isin(households['SERIALNO'])]
    
    print(f"   - Loaded {len(persons)} persons and {len(households)} households")
    
    # 2. Construct tax units
    print("\n2. Constructing tax units...")
    constructor = TaxUnitConstructor(persons, households, batch_size=50, num_processes=1)
    tax_units = constructor.create_rule_based_units(parallel=False)
    
    if isinstance(tax_units, list):
        tax_units_list = tax_units
    else:
        tax_units_list = tax_units.to_dict('records')
    
    print(f"   - Created {len(tax_units_list)} tax units")
    
    # 3. Initialize ML validator
    print("\n3. Initializing ML validator...")
    ml_validator = MLTaxUnitValidator(MODEL_PATH if os.path.exists(MODEL_PATH) else None)
    
    if ml_validator.model is None:
        print("   - ERROR: No ML model loaded!")
        return
    
    print(f"   - Model loaded successfully")
    print(f"   - Model classes: {ml_validator.model.classes_}")
    
    # 4. Extract features and analyze
    print("\n4. Analyzing features...")
    features = ml_validator.extract_features(tax_units_list)
    print(f"   - Feature columns: {list(features.columns)}")
    print(f"   - Feature shape: {features.shape}")
    
    # Show feature statistics
    print("\n   - Feature statistics:")
    for col in features.columns:
        print(f"     {col}: mean={features[col].mean():.2f}, std={features[col].std():.2f}, "
              f"min={features[col].min():.2f}, max={features[col].max():.2f}")
    
    # 5. Get model predictions
    print("\n5. Getting model predictions...")
    probas = ml_validator.model.predict_proba(features)
    predicted_classes = ml_validator.model.classes_
    
    print(f"   - Prediction shape: {probas.shape}")
    print(f"   - Classes: {predicted_classes}")
    
    # 6. Analyze predictions vs current status
    print("\n6. Analyzing predictions vs current filing status...")
    
    analysis_data = []
    for i, unit in enumerate(tax_units_list[:20]):  # Analyze first 20 units
        current_status = str(unit.get('filing_status', '')).lower()
        max_idx = np.argmax(probas[i])
        predicted_status = str(predicted_classes[max_idx]).lower()
        confidence = float(probas[i][max_idx])
        
        # Get all class probabilities
        class_probs = {str(cls).lower(): float(probas[i][j]) for j, cls in enumerate(predicted_classes)}
        
        analysis_data.append({
            'unit_id': i+1,
            'current_status': current_status,
            'predicted_status': predicted_status,
            'confidence': confidence,
            'income': unit.get('income', 0),
            'num_dependents': unit.get('num_dependents', 0),
            'would_flag': predicted_status != current_status and confidence > 0.7,
            **class_probs
        })
    
    analysis_df = pd.DataFrame(analysis_data)
    
    # Show detailed analysis
    print(f"\n   - Current status distribution:")
    status_counts = analysis_df['current_status'].value_counts()
    for status, count in status_counts.items():
        print(f"     {status}: {count}")
    
    print(f"\n   - Predicted status distribution:")
    pred_counts = analysis_df['predicted_status'].value_counts()
    for status, count in pred_counts.items():
        print(f"     {status}: {count}")
    
    print(f"\n   - Confidence statistics:")
    print(f"     Mean confidence: {analysis_df['confidence'].mean():.3f}")
    print(f"     Min confidence: {analysis_df['confidence'].min():.3f}")
    print(f"     Max confidence: {analysis_df['confidence'].max():.3f}")
    print(f"     Units with confidence > 0.7: {(analysis_df['confidence'] > 0.7).sum()}")
    
    print(f"\n   - Disagreement analysis:")
    disagreements = analysis_df[analysis_df['current_status'] != analysis_df['predicted_status']]
    print(f"     Units with different predictions: {len(disagreements)}")
    print(f"     High-confidence disagreements (>0.7): {disagreements['would_flag'].sum()}")
    
    if len(disagreements) > 0:
        print(f"\n   - Disagreement details:")
        for _, row in disagreements.head(10).iterrows():
            print(f"     Unit {row['unit_id']}: {row['current_status']} -> {row['predicted_status']} "
                  f"(conf: {row['confidence']:.3f}, income: ${row['income']:,.0f}, deps: {row['num_dependents']})")
    
    # 7. Test with lower confidence threshold
    print(f"\n7. Testing with lower confidence thresholds...")
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        flags = analysis_df[(analysis_df['current_status'] != analysis_df['predicted_status']) & 
                           (analysis_df['confidence'] > threshold)]
        print(f"   - Threshold {threshold}: {len(flags)} units would be flagged")
    
    # 8. Show sample predictions with all probabilities
    print(f"\n8. Sample predictions with all class probabilities:")
    for i in range(min(5, len(analysis_data))):
        row = analysis_data[i]
        print(f"   Unit {row['unit_id']}: Current={row['current_status']}, Predicted={row['predicted_status']}")
        for cls in predicted_classes:
            cls_lower = str(cls).lower()
            prob = row.get(cls_lower, 0)
            marker = " <-- PREDICTED" if cls_lower == row['predicted_status'] else ""
            print(f"     {cls_lower}: {prob:.3f}{marker}")
        print()

if __name__ == "__main__":
    debug_ml_predictions()
