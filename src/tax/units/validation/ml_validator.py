"""
ML-based validation for tax unit classification.

This module provides machine learning-based validation to identify potential
misclassifications in tax unit filing statuses (Single, HoH, Joint).
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging
import joblib
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

# Constants for validation rules
MIN_INCOME_HOH = 10000  # Minimum income to qualify as HoH
MAX_INCOME_HOH = 200000  # Income above which HoH is less likely
MIN_AGE_HOH = 19  # Minimum age to file as HoH
MAX_AGE_HOH = 65  # Age after which HoH status might change
MIN_DEPENDENTS_HOH = 1  # Minimum dependents for HoH

class MLTaxUnitValidator:
    """Machine learning validator for tax unit classification.
    
    This class provides methods to validate tax unit classifications using
    machine learning models to identify potential misclassifications.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the ML validator.
        
        Args:
            model_path: Path to a pre-trained model. If None, uses rule-based validation.
        """
        self.model = None
        self.features = [
            'income',
            'num_dependents',
            'age',
            'has_children',
            'is_householder',
            'housing_cost_ratio',
            'marital_status_single',
            'marital_status_married',
            'marital_status_divorced',
            'marital_status_widowed'
        ]
        self.target = 'filing_status'
        
        # Load model if path is provided
        if model_path and Path(model_path).exists():
            try:
                self.model = joblib.load(model_path)
                logger.info(f"Loaded ML model from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model from {model_path}: {e}")
                self.model = None
        
    def extract_features(self, tax_units: List[Dict[str, Any]]) -> pd.DataFrame:
        """Extract features from tax units for ML validation.
        
        Args:
            tax_units: List of tax unit dictionaries
            
        Returns:
            DataFrame with extracted features
        """
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(tax_units)
        
        # Calculate derived features
        features = {}
        
        # Basic features
        features['income'] = df.get('income', 0).astype(float)
        features['num_dependents'] = df.get('dependents', []).apply(len).astype(int)
        features['age'] = df.get('primary_filer', {}).apply(
            lambda x: x.get('age', 0) if isinstance(x, dict) else 0
        ).astype(int)
        
        # Boolean features
        features['has_children'] = (features['num_dependents'] > 0).astype(int)
        features['is_householder'] = df.get('is_householder', 0).astype(int)
        
        # Financial ratios
        total_income = features['income'].clip(lower=1)  # Avoid division by zero
        housing_costs = df.get('housing_costs', 0).astype(float)
        features['housing_cost_ratio'] = (housing_costs / total_income).fillna(0)
        
        # Categorical features - one-hot encode marital status
        marital_status = df.get('marital_status', 'unknown').fillna('unknown')
        for status in ['single', 'married', 'divorced', 'widowed']:
            features[f'marital_status_{status}'] = (marital_status == status).astype(int)
        
        # Create final feature DataFrame with consistent column order
        feature_df = pd.DataFrame({k: features[k] for k in self.features if k in features})
        
        # Ensure all expected columns are present
        for col in self.features:
            if col not in feature_df.columns:
                feature_df[col] = 0
                
        return feature_df[self.features]  # Ensure consistent column order
    
    def predict(self, tax_units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict potential misclassifications.
        
        Args:
            tax_units: List of tax units to validate
            
        Returns:
            List of tax units with validation flags
        """
        if not tax_units:
            return []
            
        # Extract features
        features = self.extract_features(tax_units)
        
        # If no model is loaded, use rule-based validation
        if self.model is None:
            logger.warning("No ML model loaded. Using rule-based validation.")
            return self._rule_based_validation(tax_units, features)
            
        # Use ML model for prediction
        try:
            # Get predicted probabilities for each class
            probas = self.model.predict_proba(features)
            predicted_classes = self.model.classes_
            
            # For each tax unit, get the predicted class and confidence
            validated_units = []
            for i, unit in enumerate(tax_units):
                unit = unit.copy()
                current_status = str(unit.get('filing_status', '')).lower()
                
                # Get prediction with highest probability
                max_idx = np.argmax(probas[i])
                predicted_status = str(predicted_classes[max_idx]).lower()
                confidence = float(probas[i][max_idx])
                
                # Only flag if prediction is different from current status
                # and confidence is above threshold
                if (predicted_status != current_status and 
                    confidence > 0.7 and 
                    predicted_status in ['single', 'hoh', 'joint']):
                    
                    unit.setdefault('validation_flags', []).append({
                        'code': f'PREDICTED_{predicted_status.upper()}',
                        'message': (
                            f'Model predicts this should be {predicted_status.upper()} '
                            f'instead of {current_status.upper()}. '
                            f'(Confidence: {confidence:.1%})'
                        ),
                        'confidence': confidence,
                        'suggested_status': predicted_status
                    })
                
                # Add rule-based validations as well
                unit = self._apply_rule_based_validation(unit, features.iloc[i])
                validated_units.append(unit)
                
            return validated_units
            
        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            logger.warning("Falling back to rule-based validation")
            return self._rule_based_validation(tax_units, features)
    
    def _rule_based_validation(self, 
                             tax_units: List[Dict[str, Any]], 
                             features: pd.DataFrame) -> List[Dict[str, Any]]:
        """Apply rule-based validation to all tax units."""
        return [
            self._apply_rule_based_validation(unit, features.iloc[i])
            for i, unit in enumerate(tax_units)
        ]
    
    def _apply_rule_based_validation(self, 
                                   unit: Dict[str, Any], 
                                   features: pd.Series) -> Dict[str, Any]:
        """Apply rule-based validation to a single tax unit."""
        unit = unit.copy()
        validation_flags = []
        
        # Extract basic info
        filing_status = str(unit.get('filing_status', '')).lower()
        income = float(features.get('income', 0))
        num_dependents = int(features.get('num_dependents', 0))
        age = int(features.get('age', 0))
        is_householder = bool(features.get('is_householder', 0))
        housing_ratio = float(features.get('housing_cost_ratio', 0))
        
        # Rule 1: Single with dependents might qualify as HoH
        if (filing_status == 'single' and 
            num_dependents >= MIN_DEPENDENTS_HOH and
            income >= MIN_INCOME_HOH and
            MIN_AGE_HOH <= age <= MAX_AGE_HOH and
            is_householder):
            
            confidence = min(0.7 + (num_dependents * 0.1), 0.9)
            validation_flags.append({
                'code': 'POTENTIAL_HOH',
                'message': (
                    f'Single filer with {num_dependents} dependents might qualify as Head of Household. '
                    f'Verify if they meet all HoH requirements.'
                ),
                'confidence': confidence,
                'suggested_status': 'hoh'
            })
        
        # Rule 2: High income HoH might be incorrect
        if (filing_status == 'hoh' and 
            income > MAX_INCOME_HOH and
            num_dependents == 1 and
            age > 50):
            
            validation_flags.append({
                'code': 'HIGH_INCOME_HOH',
                'message': (
                    f'High income Head of Household (${income:,.2f}) with only one dependent. '
                    f'Verify if this filing status is correct.'
                ),
                'confidence': 0.8,
                'suggested_status': 'single' if num_dependents == 0 else 'joint'
            })
        
        # Rule 3: Married filing single might be incorrect
        if (filing_status == 'single' and
            features.get('marital_status_married', 0) == 1 and
            num_dependents > 0):
            
            validation_flags.append({
                'code': 'MARRIED_FILING_SINGLE',
                'message': 'Married taxpayer filing as single with dependents. Verify if should file jointly.',
                'confidence': 0.85,
                'suggested_status': 'joint'
            })
        
        # Rule 4: Low housing cost ratio for HoH
        if (filing_status == 'hoh' and
            housing_ratio < 0.1 and
            income > 50000):
            
            validation_flags.append({
                'code': 'LOW_HOUSING_RATIO',
                'message': (
                    f'Low housing cost ratio ({housing_ratio:.1%}) for HoH. '
                    f'Verify if they pay more than half the household costs.'
                ),
                'confidence': 0.7,
                'suggested_status': 'single'
            })
        
        # Add validation flags to unit if any were found
        if validation_flags:
            unit['validation_flags'] = validation_flags
            
        return unit
    
    def save_model(self, path: str):
        """Save the trained model to disk."""
        # TODO: Implement model saving
        pass
        
    def load_model(self, path: str):
        """Load a trained model from disk."""
        # TODO: Implement model loading
        pass


def validate_tax_units(tax_units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convenience function to validate tax units using the ML validator.
    
    Args:
        tax_units: List of tax units to validate
        
    Returns:
        List of tax units with validation flags
    """
    validator = MLTaxUnitValidator()
    return validator.predict(tax_units)
