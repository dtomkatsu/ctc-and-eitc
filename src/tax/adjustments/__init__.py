"""
Income adjustments and tax credits for Hawaii tax calculations
"""

from .agi_adjustments import estimate_agi_from_total_income
from .hawaii_credits import calculate_hawaii_credits
from .itemized_deductions import estimate_deduction

__all__ = ['estimate_agi_from_total_income', 'calculate_hawaii_credits', 'estimate_deduction']
