"""
Tax Credits Module

This module provides functionality for calculating various tax credits
including the Child Tax Credit (CTC) and Earned Income Tax Credit (EITC).
"""

from .ctc import calculate_ctc

__all__ = [
    'calculate_ctc'
]
