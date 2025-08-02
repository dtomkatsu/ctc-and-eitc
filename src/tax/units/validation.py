"""
Tax Unit Validation Module

This module provides validation functions for tax units to ensure data consistency
and correctness throughout the tax unit construction process.
"""

from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
from dataclasses import dataclass
from enum import Enum, auto


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()


@dataclass
class ValidationIssue:
    """Represents a validation issue found during tax unit validation."""
    message: str
    severity: ValidationSeverity
    tax_unit_id: Optional[str] = None
    field: Optional[str] = None
    details: Optional[Dict] = None


class TaxUnitValidator:
    """
    Validates tax units for consistency and correctness.
    
    This class provides methods to validate individual tax units and collections
    of tax units, checking for various data quality issues and logical consistency.
    """
    
    @staticmethod
    def validate_tax_unit(tax_unit: Dict) -> List[ValidationIssue]:
        """
        Validate a single tax unit.
        
        Args:
            tax_unit: Dictionary representing a tax unit
            
        Returns:
            List of validation issues found
        """
        issues = []
        
        # Check required fields
        required_fields = [
            'filer_id', 'filing_status', 'income', 
            'num_dependents', 'dependents', 'hh_id'
        ]
        
        for field in required_fields:
            if field not in tax_unit:
                issues.append(ValidationIssue(
                    f"Missing required field: {field}",
                    ValidationSeverity.ERROR,
                    tax_unit.get('filer_id'),
                    field
                ))
        
        # Validate filing status
        valid_statuses = {
            'single', 'married_filing_jointly', 
            'married_filing_separately', 'head_of_household',
            'joint',  # Allow 'joint' as alias for 'married_filing_jointly'
            'married_filing_separate'  # Allow common variant
        }
        
        if 'filing_status' in tax_unit and tax_unit['filing_status'] not in valid_statuses:
            issues.append(ValidationIssue(
                f"Invalid filing status: {tax_unit['filing_status']}",
                ValidationSeverity.ERROR,
                tax_unit.get('filer_id'),
                'filing_status',
                {'valid_statuses': valid_statuses}
            ))
        
        # Validate dependents
        if 'dependents' in tax_unit and 'num_dependents' in tax_unit:
            if not isinstance(tax_unit['dependents'], list):
                issues.append(ValidationIssue(
                    "Dependents should be a list",
                    ValidationSeverity.ERROR,
                    tax_unit.get('filer_id'),
                    'dependents'
                ))
            elif len(tax_unit['dependents']) != tax_unit['num_dependents']:
                issues.append(ValidationIssue(
                    f"Number of dependents ({tax_unit['num_dependents']}) "
                    f"does not match length of dependents list ({len(tax_unit['dependents'])})",
                    ValidationSeverity.ERROR,
                    tax_unit.get('filer_id'),
                    'num_dependents'
                ))
        
        # Validate income
        if 'income' in tax_unit and not isinstance(tax_unit['income'], (int, float)):
            issues.append(ValidationIssue(
                f"Income must be a number, got {type(tax_unit['income']).__name__}",
                ValidationSeverity.ERROR,
                tax_unit.get('filer_id'),
                'income'
            ))
        
        return issues
    
    @staticmethod
    def validate_tax_units(tax_units: List[Dict]) -> List[ValidationIssue]:
        """
        Validate a collection of tax units.
        
        Args:
            tax_units: List of tax unit dictionaries
            
        Returns:
            List of validation issues found across all tax units
        """
        issues = []
        filer_ids = set()
        
        # Check for duplicate filer IDs
        for unit in tax_units:
            if 'filer_id' in unit:
                if unit['filer_id'] in filer_ids:
                    issues.append(ValidationIssue(
                        f"Duplicate filer_id: {unit['filer_id']}",
                        ValidationSeverity.ERROR,
                        unit['filer_id']
                    ))
                filer_ids.add(unit['filer_id'])
            
            # Validate individual tax unit
            issues.extend(TaxUnitValidator.validate_tax_unit(unit))
        
        return issues
    
    @staticmethod
    def validate_household_coverage(
        tax_units: List[Dict], 
        household_members: pd.DataFrame
    ) -> List[ValidationIssue]:
        """
        Validate that all household members are properly assigned to tax units.
        
        Args:
            tax_units: List of tax unit dictionaries
            household_members: DataFrame of all household members
            
        Returns:
            List of validation issues related to household coverage
        """
        issues = []
        
        if household_members.empty:
            issues.append(ValidationIssue(
                "Household members DataFrame is empty",
                ValidationSeverity.WARNING
            ))
            return issues
        
        # Track all person IDs in tax units
        assigned_person_ids = set()
        
        for unit in tax_units:
            # Add primary filer
            if 'primary_filer_id' in unit and unit['primary_filer_id']:
                assigned_person_ids.add(str(unit['primary_filer_id']))
            
            # Add secondary filer (for joint returns)
            if 'secondary_filer_id' in unit and unit['secondary_filer_id']:
                assigned_person_ids.add(str(unit['secondary_filer_id']))
            
            # Add dependents
            if 'dependents' in unit and unit['dependents']:
                for dep_id in unit['dependents']:
                    assigned_person_ids.add(str(dep_id))
        
        # Check for unassigned household members
        household_person_ids = set(household_members.index.astype(str))
        unassigned = household_person_ids - assigned_person_ids
        
        if unassigned:
            issues.append(ValidationIssue(
                f"Found {len(unassigned)} unassigned household members",
                ValidationSeverity.WARNING,
                details={
                    'unassigned_person_ids': list(unassigned)[:10],  # Limit to first 10 for brevity
                    'total_unassigned': len(unassigned)
                }
            ))
        
        return issues
