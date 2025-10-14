"""
Train ML model for tax unit validation.

This script trains a machine learning model to identify potential misclassifications
in tax unit filing statuses using labeled training data.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tax.units.validation.ml_validator import MLTaxUnitValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_training_data(filepath: str) -> pd.DataFrame:
    """Load and preprocess training data.
    
    Args:
        filepath: Path to the training data file
        
    Returns:
        DataFrame containing the training data
    """
    logger.info(f"Loading training data from {filepath}")
    
    # For now, create synthetic training data
    # In a real scenario, this would load from a file
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'income': np.random.lognormal(10.5, 0.8, n_samples).astype(int),
        'num_dependents': np.random.poisson(1.5, n_samples).clip(0, 5),
        'age': np.random.normal(45, 12, n_samples).clip(18, 90).astype(int),
        'is_householder': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
        'marital_status': np.random.choice(['single', 'married', 'divorced', 'widowed'], 
                                         n_samples, p=[0.4, 0.4, 0.15, 0.05]),
        'housing_cost_ratio': np.random.beta(2, 5, n_samples) * 0.5,  # 0-50% of income
        'true_status': np.random.choice(['single', 'hoh', 'joint'], n_samples, 
                                      p=[0.5, 0.3, 0.2])
    }
    
    # Create some realistic patterns
    df = pd.DataFrame(data)
    
    # Adjust true_status based on patterns
    # Single with dependents more likely to be HoH
    hoh_mask = (df['num_dependents'] > 0) & (df['marital_status'] == 'single')
    df.loc[hoh_mask, 'true_status'] = np.random.choice(
        ['hoh', 'single'], 
        size=hoh_mask.sum(),
        p=[0.8, 0.2]
    )
    
    # High income more likely to be joint
    joint_mask = (df['income'] > 100000) & (df['marital_status'] == 'married')
    df.loc[joint_mask, 'true_status'] = 'joint'
    
    return df

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Train a random forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained RandomForestClassifier
    """
    logger.info("Training Random Forest classifier...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    """Evaluate the trained model.
    
    Args:
        model: Trained classifier
        X_test: Test features
        y_test: True test labels
    """
    logger.info("Evaluating model...")
    
    y_pred = model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nFeature Importances:")
    for name, importance in zip(X_test.columns, model.feature_importances_):
        print(f"{name}: {importance:.3f}")

def main():
    """Main training function."""
    # Load and prepare data
    data = load_training_data("data/processed/training_data.csv")
    
    # Prepare features and target using the same feature engineering as the validator
    validator = MLTaxUnitValidator()
    
    # Create test units with the same structure as real tax units
    test_units = data.to_dict('records')
    for i, unit in enumerate(test_units):
        unit['dependents'] = [{'age': 0}] * int(unit.get('num_dependents', 0))
        unit['primary_filer'] = {'age': unit.get('age', 0)}
        unit['housing_costs'] = unit.get('income', 0) * unit.get('housing_cost_ratio', 0)
    
    # Extract features using the validator's method
    X = validator.extract_features(test_units)
    y = data['true_status']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate
    evaluate_model(model, X_test, y_test)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/tax_unit_validator.joblib'
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Test the validator with the trained model
    test_validator(model_path)

def test_validator(model_path: str):
    """Test the ML validator with the trained model."""
    from src.tax.units.validation.ml_validator import MLTaxUnitValidator
    
    # Initialize validator with trained model
    validator = MLTaxUnitValidator(model_path=model_path)
    
    # Create test cases
    test_cases = [
        {
            'id': 1,
            'filing_status': 'single',
            'income': 45000,
            'dependents': [{'age': 10}, {'age': 8}],
            'primary_filer': {'age': 35},
            'is_householder': 1,
            'housing_costs': 18000,
            'marital_status': 'single'
        },
        {
            'id': 2,
            'filing_status': 'hoh',
            'income': 250000,
            'dependents': [{'age': 15}],
            'primary_filer': {'age': 45},
            'is_householder': 1,
            'housing_costs': 30000,
            'marital_status': 'single'
        }
    ]
    
    # Get predictions
    results = validator.predict(test_cases)
    
    # Print results
    print("\nValidator Test Results:")
    for unit in results:
        print(f"\nTax Unit {unit['id']}:")
        print(f"  Status: {unit['filing_status'].upper()}")
        if 'validation_flags' in unit:
            print("  Flags:")
            for flag in unit['validation_flags']:
                print(f"    - {flag['message']} (Confidence: {flag['confidence']:.2f})")
        else:
            print("  No issues detected")

if __name__ == "__main__":
    main()
