"""
Tax unit construction and analysis module.

This package provides functionality for constructing tax units from PUMS data
and analyzing their characteristics for tax policy modeling.
"""

"""
Tax unit construction and analysis module.

This package provides functionality for constructing tax units from PUMS data
and analyzing their characteristics for tax policy modeling.
"""

# Import key classes and functions to make them available at package level
from .base import TaxUnitConstructor, FILING_STATUS
from .dependencies import identify_dependents
from .income import calculate_tax_unit_income
from .relationships import identify_relationships
from .status import (
    is_head_of_household,
    is_married_filing_jointly,
    is_married_filing_separately
)
from .validation import (
    TaxUnitValidator,
    ValidationIssue,
    ValidationSeverity
)

# Define what gets imported with 'from tax.units import *'
__all__ = [
    'TaxUnitConstructor',
    'TaxUnitValidator',
    'ValidationIssue',
    'ValidationSeverity',
    'identify_dependents',
    'calculate_tax_unit_income',
    'identify_relationships',
    'is_head_of_household',
    'is_married_filing_jointly',
    'is_married_filing_separately',
    'FILING_STATUS'
]

# Version information
__version__ = '0.1.0'
