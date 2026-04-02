"""
Source-Specific Income Growth Projector

Applies different growth rates to different income components (WAGP, SEMP, INTP, etc.)
rather than a single bracket-specific rate to total AGI.

Growth rate sources:
    - Wages (WAGP): BLS OES bracket-specific nominal rates (2022→2024 observed)
    - Self-employment (SEMP): BLS OES bracket-specific (proxy)
    - Interest (INTP): 5.0% — BEA HI div+int+rent CAGR 2022→2024 = 9.4%; moderated for
      Fed rate cuts 2024→2027 (data/raw/bea_hawaii_sainc5n.csv, line 46)
    - Dividends (DIV): 5.0% — equity market nominal; BEA line 46 upper bound
    - Retirement (RETP): 3.5% — inflation + growing retiree population
    - Social Security (SSP): 3.8% — known COLAs: 8.7% (2023) + 3.2% (2024) + 2.5%/yr
      projected (2025-2027) → compound CAGR = 3.8% (data/raw/bea_hawaii_sainc5n.csv, line 47)
    - Other income (OIP): 3.0% — general inflation proxy; BEA transfers 2022→2024 = 1.8%,
      blended with CPI
    - Capital gains: 5.0% nominal (long-run equity return)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# BLS OES bracket-specific nominal annual wage growth rates
# Derived from 2022→2024 observed data (WAGE_GROWTH_FINAL_SUMMARY.md)
_BLS_ANNUAL_RATES = {
    (0, 25_000):             0.070,
    (25_000, 50_000):        0.063,
    (50_000, 75_000):        0.056,
    (75_000, 100_000):       0.051,
    (100_000, 200_000):      0.046,
    (200_000, float('inf')): 0.042,
}

# Fixed nominal annual growth rates for non-wage income sources
# Derived from BEA SAINC5N Hawaii data (data/raw/bea_hawaii_sainc5n.csv)
# BEA line 46 (div+int+rent) 2022→2024 CAGR = 9.4% (interest-rate surge inflated)
# BEA line 47 (transfer receipts) 2022→2024 CAGR = 1.8% (post-COVID normalization)
# SS COLAs: 8.7% (2023), 3.2% (2024), 2.5%/yr projected → 5yr CAGR = 3.8%
_FIXED_RATES = {
    'intp': 0.050,   # Interest: BEA HI div+int+rent surge 2022→2024; moderated by Fed cuts
    'div':  0.050,   # Dividends: equity market nominal ~5%/yr
    'retp': 0.035,   # Retirement distributions: inflation + retiree population growth
    'ssp':  0.038,   # Social Security: 8.7%+3.2% COLA observed, 2.5%/yr projected → 3.8% CAGR
    'oip':  0.030,   # Other income: general inflation proxy (BEA transfers 1.8%, CPI ~3%)
}

# Capital gains
_CAP_GAINS_RATE = 0.050  # 5%/yr nominal


def _get_wage_bracket_rate(agi: float) -> float:
    """Get BLS bracket-specific rate for a given AGI level."""
    for (lo, hi), rate in _BLS_ANNUAL_RATES.items():
        if lo <= agi < hi:
            return rate
    return 0.042  # Default to highest-bracket rate


def _compute_growth_factor(annual_rate: float, years_observed: int,
                           years_projected: int, moderation: float) -> float:
    """Compute multi-year growth factor with moderation for projected years."""
    factor_obs = (1 + annual_rate) ** years_observed
    factor_proj = (1 + annual_rate * moderation) ** years_projected
    return factor_obs * factor_proj


class SourceSpecificGrowthProjector:
    """
    Project income forward using source-specific growth rates.

    Wages and self-employment use BLS OES bracket-specific rates (with moderation
    for projected years). All other sources use fixed nominal rates.
    """

    def __init__(
        self,
        years_observed: int = 2,
        years_projected: int = 3,
        moderation_factor: float = 0.70,
        cap_gains_rate: float = _CAP_GAINS_RATE,
        fixed_rates: Optional[Dict[str, float]] = None,
        bls_annual_rates: Optional[Dict[tuple, float]] = None,
        growth_scale_factor: float = 1.0,
    ):
        self.years_observed = years_observed
        self.years_projected = years_projected
        self.moderation = moderation_factor
        self.total_years = years_observed + years_projected
        self.growth_scale_factor = growth_scale_factor

        # Use externally-provided BLS rates (e.g. from 15-year OES CAGR) or fall back to hardcoded.
        # Merge: start with hardcoded rates as base (covers all brackets including $0-$25k),
        # then overlay BLS long-run rates where available. This ensures no bracket is left at 0%.
        # The $0-$25k bracket uses the hardcoded 7.0% (Hawaii min wage $10.10→$14 2022→2024),
        # which is more accurate than long-run OES CAGR since OES has no sub-$25k occupations.
        merged = dict(_BLS_ANNUAL_RATES)
        if bls_annual_rates:
            merged.update(bls_annual_rates)

        # Apply growth_scale_factor to all rates for CI scenario runs
        self._bls_rates = {k: v * growth_scale_factor for k, v in merged.items()}
        self.fixed_rates = {k: v * growth_scale_factor
                           for k, v in (fixed_rates or dict(_FIXED_RATES)).items()}
        self.cap_gains_rate = cap_gains_rate * growth_scale_factor

    def _grow_wage_component(self, values: np.ndarray, agi: np.ndarray) -> np.ndarray:
        """Grow wage/self-employment income using bracket-specific rates."""
        grown = values.copy()
        for (lo, hi), annual_rate in self._bls_rates.items():
            mask = (agi >= lo) & (agi < hi)
            factor = _compute_growth_factor(
                annual_rate, self.years_observed, self.years_projected, self.moderation
            )
            grown[mask] *= factor
        return grown

    def _grow_fixed_component(self, values: np.ndarray, source: str) -> np.ndarray:
        """Grow a component at a fixed annual rate (full rate for all years)."""
        rate = self.fixed_rates.get(source, 0.03)
        factor = (1 + rate) ** self.total_years
        return values * factor

    def project_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Project all income components forward and recompute AGI.

        Expects columns: primary_wagp, primary_semp, ..., secondary_wagp, ..., agi, capital_gains
        """
        result = df.copy()
        result['agi_2022'] = result['agi'].copy()

        # Check that income component columns exist
        if 'primary_wagp' not in result.columns:
            raise ValueError("Income component columns not found. Re-run regenerate_tax_units.py.")

        agi = result['agi'].values.astype(float)

        logger.info("Source-specific income growth (2022 → 2027, NOMINAL):")
        logger.info(f"  Wage/SE: BLS bracket-specific (obs {self.years_observed}yr, "
                     f"proj {self.years_projected}yr at {self.moderation:.0%} moderation)")
        for src, rate in self.fixed_rates.items():
            factor = (1 + rate) ** self.total_years
            logger.info(f"  {src.upper()}: {rate:.1%}/yr → x{factor:.3f} over {self.total_years}yr")

        # Grow each component
        wage_sources = ['wagp', 'semp']
        fixed_sources = list(self.fixed_rates.keys())
        all_sources = wage_sources + fixed_sources

        grown_total = np.zeros(len(result))

        for prefix in ['primary', 'secondary']:
            for src in all_sources:
                col = f'{prefix}_{src}'
                if col not in result.columns:
                    continue
                vals = result[col].values.astype(float)
                if src in wage_sources:
                    grown = self._grow_wage_component(vals, agi)
                else:
                    grown = self._grow_fixed_component(vals, src)
                result[f'{col}_2027'] = grown
                grown_total += grown

        # Capital gains grow at market rate — tracked separately from ordinary income.
        # Capital gains are NOT added to agi because they are taxed under a separate
        # tax policy (CG tax). Income tax bracket changes should only affect ordinary income.
        if 'capital_gains' in result.columns:
            cg_factor = (1 + self.cap_gains_rate) ** self.total_years
            result['capital_gains_2022'] = result['capital_gains'].copy()
            result['capital_gains'] = result['capital_gains'] * cg_factor
            logger.info(f"  Capital gains: {self.cap_gains_rate:.1%}/yr → x{cg_factor:.3f} (tracked separately)")

        # Add dependent income (difference between original income and component sum)
        comp_cols = [f'{p}_{s}' for p in ['primary', 'secondary'] for s in all_sources
                     if f'{p}_{s}' in result.columns]
        original_comp_sum = result[comp_cols].sum(axis=1).values
        dependent_income = np.maximum(0, result['income'].values - original_comp_sum)

        # Grow dependent income at a blended wage rate (most dependent income is from working teens)
        avg_wage_rate = 0.056  # ~$50-75k bracket rate as proxy
        dep_factor = _compute_growth_factor(
            avg_wage_rate, self.years_observed, self.years_projected, self.moderation
        )
        grown_dep_income = dependent_income * dep_factor
        grown_total += grown_dep_income

        # agi = ordinary income only (no capital gains). Consistent with 2022 baseline where
        # agi excluded capital gains. Use agi_with_capital_gains for surcharges that apply
        # to total income (e.g. millionaire's tax that covers all income sources).
        result['agi'] = grown_total
        if 'capital_gains' in result.columns:
            result['agi_with_capital_gains'] = grown_total + result['capital_gains']

        # Store growth factors for reporting
        result['income_growth_factor'] = np.where(
            result['agi_2022'] > 0,
            result['agi'] / result['agi_2022'],
            1.0
        )

        avg_factor = np.average(
            result['income_growth_factor'],
            weights=result['weight'].values
        )
        logger.info(f"\n  Weighted average income growth factor: x{avg_factor:.3f}")
        logger.info(f"  Weighted avg AGI: ${np.average(result['agi_2022'], weights=result['weight']):,.0f} "
                     f"(2022) → ${np.average(result['agi'], weights=result['weight']):,.0f} (2027)")

        return result
