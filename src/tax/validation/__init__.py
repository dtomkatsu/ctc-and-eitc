"""
Tax validation utilities for comparing model outputs against benchmarks.
"""

from .dotax_table_12a import DotaxTable12AValidator, validate_against_table_12a
from .hybrid_tax_calibration import (
    apply_hybrid_tax_calibration,
    validate_hybrid_calibration,
    load_hybrid_benchmarks,
    get_hybrid_benchmark_summary
)

__all__ = [
    'DotaxTable12AValidator',
    'validate_against_table_12a',
    'apply_hybrid_tax_calibration',
    'validate_hybrid_calibration',
    'load_hybrid_benchmarks',
    'get_hybrid_benchmark_summary'
]
