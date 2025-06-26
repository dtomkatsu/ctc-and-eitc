"""
Tax unit construction and analysis module.

This package provides functionality for constructing tax units from PUMS data
and analyzing their characteristics for tax policy modeling.
"""

# Import key classes and functions to make them available at package level
from .base import TaxUnitConstructor
from .filers import (
    BaseFiler,
    SingleFiler,
    JointFiler,
    SeparateFiler,
    HeadOfHouseholdFiler
)

# Define what gets imported with 'from tax.units import *'
__all__ = [
    'TaxUnitConstructor',
    'BaseFiler',
    'SingleFiler',
    'JointFiler',
    'SeparateFiler',
    'HeadOfHouseholdFiler',
]

# Version information
__version__ = '0.1.0'
