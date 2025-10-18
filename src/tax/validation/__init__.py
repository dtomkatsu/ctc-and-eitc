"""
Tax validation utilities for comparing model outputs against benchmarks.
"""

from .dotax_table_12a import DotaxTable12AValidator, validate_against_table_12a

__all__ = ['DotaxTable12AValidator', 'validate_against_table_12a']
