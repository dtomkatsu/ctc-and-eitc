"""
Income processing and separation module.

This module handles income-related calculations including:
- Capital gains separation
- Income source splitting
- Income growth adjustments
"""

from .capital_gains_separation import (
    apply_capital_gains_separation,
    get_capital_gains_summary,
    load_capital_gains_percentages,
    get_capital_gains_percentage
)

__all__ = [
    'apply_capital_gains_separation',
    'get_capital_gains_summary',
    'load_capital_gains_percentages',
    'get_capital_gains_percentage'
]
