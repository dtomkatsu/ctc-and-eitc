"""
Filing status determination logic.

This package contains modules for determining tax filing status,
including rules for Single, Married Filing Jointly, Married Filing Separately,
and Head of Household statuses.
"""

from .mfj import is_married_filing_jointly
from .mfs import is_married_filing_separately
from .hoh import is_head_of_household

__all__ = [
    'is_married_filing_jointly',
    'is_married_filing_separately',
    'is_head_of_household',
]
