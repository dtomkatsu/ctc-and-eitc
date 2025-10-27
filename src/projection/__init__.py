"""
Ensemble growth projection system for Hawaii tax revenue forecasting.

This module provides income projection capabilities that combine:
- ACS demographic and income trends
- BLS OES occupation-specific wage growth  
- Hierarchical matching strategies with confidence scoring
"""

from .ensemble import EnsembleProjector
from .occupation_matcher import OccupationMatcher

__all__ = ['EnsembleProjector', 'OccupationMatcher']
