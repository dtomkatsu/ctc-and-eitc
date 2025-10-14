"""
Configuration module for tax modeling.

This module contains configuration parameters for income growth,
inflation adjustments, and other modeling assumptions.
"""

from .income_growth import (
    GrowthFactors,
    RESIDENT_GROWTH,
    NONRESIDENT_GROWTH,
    get_growth_factors,
    apply_income_growth,
    log_growth_factors,
    validate_growth_factors
)

__all__ = [
    'GrowthFactors',
    'RESIDENT_GROWTH',
    'NONRESIDENT_GROWTH',
    'get_growth_factors',
    'apply_income_growth',
    'log_growth_factors',
    'validate_growth_factors'
]
