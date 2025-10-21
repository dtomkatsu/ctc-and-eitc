"""
Hawaii Tax Deductions and Exemptions Module

Calculates taxable income from AGI by applying:
- Deductions (itemized vs standard)
- Personal exemptions

Supports policy modeling for revenue impact analysis.
"""

from .policy import DeductionPolicy
from .calculator import TaxableIncomeCalculator
from .parsers import parse_deduction_benchmarks, parse_exemption_benchmarks

__all__ = [
    'DeductionPolicy',
    'TaxableIncomeCalculator',
    'parse_deduction_benchmarks',
    'parse_exemption_benchmarks',
]
