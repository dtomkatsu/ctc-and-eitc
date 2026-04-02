"""
Tax Credits Module

This module provides functionality for calculating various tax credits
including the Child Tax Credit (CTC) and Earned Income Tax Credit (EITC).
"""

from .eitc import calculate_eitc, calculate_eitc_for_tax_units

__all__ = [
    'calculate_eitc',
    'calculate_eitc_for_tax_units',
]
