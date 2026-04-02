"""
Simultaneous Two-Phase Calibrator

Replaces the fragile 6-stage sequential calibration with a single-pass approach:

Phase 1 — Entropy-balanced reweighting
    Find weights that simultaneously match ALL filer-count targets across every
    (filing_status x AGI_bracket) cell, while staying close to the original PUMS
    weights.  Uses iterative raking (multiplicative IPF) with bounded weight ratios.

Phase 2 — Direct bracket tax multipliers
    After weights are locked, compute one scalar multiplier per AGI bracket so
    that weighted tax matches the DOTAX Table A8 target exactly.

Design principles:
    - All constraints are satisfied in a single pass (no stages undoing each other)
    - No arbitrary ±60% caps, no 70% blending, no hardcoded marginal rates
    - Deductions must be folded into tax BEFORE this calibrator runs
    - Synthetic ultra-high filers must be present BEFORE this calibrator runs
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical DOTAX targets (Table A8, SOI 2022)
# ---------------------------------------------------------------------------

# Filer counts by AGI bracket (total 618,423)
DOTAX_FILER_TARGETS: Dict[Tuple[float, float], int] = {
    (0, 10_000):          115_285,
    (10_000, 20_000):      64_160,
    (20_000, 30_000):      57_835,
    (30_000, 40_000):      58_135,
    (40_000, 50_000):      53_555,
    (50_000, 75_000):      91_459,
    (75_000, 100_000):     54_976,
    (100_000, 150_000):    62_065,
    (150_000, 200_000):    27_976,
    (200_000, 300_000):    19_015,
    (300_000, 400_000):     5_729,
    (400_000, 500_000):     2_856,
    (500_000, 750_000):     2_549,
    (750_000, 1_000_000):   1_004,
    (1_000_000, np.inf):    1_824,
}

# Tax liability by AGI bracket in $M (total $3,029M)
DOTAX_TAX_TARGETS: Dict[Tuple[float, float], float] = {
    (0, 10_000):           3.0,
    (10_000, 20_000):     21.0,
    (20_000, 30_000):     51.0,
    (30_000, 40_000):     92.0,
    (40_000, 50_000):    116.0,
    (50_000, 75_000):    293.0,
    (75_000, 100_000):   261.0,
    (100_000, 150_000):  438.0,
    (150_000, 200_000):  294.0,
    (200_000, 300_000):  310.0,
    (300_000, 400_000):  153.0,
    (400_000, 500_000):  101.0,
    (500_000, 750_000):  149.0,
    (750_000, 1_000_000): 85.0,
    (1_000_000, np.inf): 663.0,
}

# Filing status targets (total 618,423)
DOTAX_STATUS_TARGETS: Dict[str, int] = {
    'single':                     326_470,
    'married_filing_jointly':     210_724,
    'head_of_household':           65_638,
    'married_filing_separately':   15_591,
}

AGI_BRACKETS = sorted(DOTAX_FILER_TARGETS.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bracket_index(agi: float) -> int:
    """Return the index into AGI_BRACKETS for a given AGI value."""
    for i, (lo, hi) in enumerate(AGI_BRACKETS):
        if lo <= agi < hi:
            return i
    return len(AGI_BRACKETS) - 1


def _bracket_label(lo: float, hi: float) -> str:
    if hi == np.inf:
        return f"${lo / 1000:.0f}k+"
    return f"${lo / 1000:.0f}k–${hi / 1000:.0f}k"


# ---------------------------------------------------------------------------
# Phase 1: Entropy-balanced reweighting via iterative raking
# ---------------------------------------------------------------------------

def phase1_reweight(
    df: pd.DataFrame,
    max_iterations: int = 100,
    tolerance: float = 0.005,
    weight_floor: float = 0.05,
    weight_ceiling_ratio: float = 15.0,
) -> pd.DataFrame:
    """
    Adjust weights to simultaneously match DOTAX filer-count targets across
    two margins: (a) AGI bracket and (b) filing status.

    Uses iterative raking (multiplicative IPF): alternate between the two
    margin constraints, scaling weights in each pass, until convergence.

    Args:
        df: Tax-unit DataFrame with columns 'agi', 'filing_status', 'weight'.
        max_iterations: Maximum raking iterations.
        tolerance: Convergence when every marginal is within this fraction of target.
        weight_floor: Minimum allowed weight (prevents zeros).
        weight_ceiling_ratio: Max ratio of calibrated-to-original weight.

    Returns:
        DataFrame with updated 'weight' column and 'weight_original' preserving
        the pre-calibration values.
    """
    result = df.copy()
    result['weight_original'] = result['weight'].copy()

    # Pre-compute bracket assignment (integer index for speed)
    result['_bracket_idx'] = result['agi'].apply(_bracket_index)

    # Map filing status to integer for fast groupby
    status_list = sorted(DOTAX_STATUS_TARGETS.keys())
    status_to_int = {s: i for i, s in enumerate(status_list)}
    result['_status_idx'] = result['filing_status'].map(status_to_int)

    # Drop rows that don't map to a known filing status
    unmapped = result['_status_idx'].isna()
    if unmapped.any():
        logger.warning(
            f"Dropping {unmapped.sum()} units with unrecognised filing_status "
            f"values: {result.loc[unmapped, 'filing_status'].unique().tolist()}"
        )
        result = result[~unmapped].copy()
        result['_status_idx'] = result['_status_idx'].astype(int)

    # Upper bound on weight per unit
    weight_caps = result['weight_original'] * weight_ceiling_ratio

    converged = False
    for iteration in range(1, max_iterations + 1):
        max_deviation = 0.0

        # --- Margin A: AGI bracket filer counts ---
        for b_idx, (lo, hi) in enumerate(AGI_BRACKETS):
            mask = result['_bracket_idx'] == b_idx
            current = result.loc[mask, 'weight'].sum()
            target = DOTAX_FILER_TARGETS[(lo, hi)]
            if current > 0 and target > 0:
                factor = target / current
                result.loc[mask, 'weight'] *= factor
                max_deviation = max(max_deviation, abs(factor - 1.0))

        # --- Margin B: Filing status totals ---
        for s_name in status_list:
            s_idx = status_to_int[s_name]
            mask = result['_status_idx'] == s_idx
            current = result.loc[mask, 'weight'].sum()
            target = DOTAX_STATUS_TARGETS[s_name]
            if current > 0 and target > 0:
                factor = target / current
                result.loc[mask, 'weight'] *= factor
                max_deviation = max(max_deviation, abs(factor - 1.0))

        # --- Enforce bounds ---
        result['weight'] = result['weight'].clip(lower=weight_floor, upper=weight_caps)

        if iteration <= 3 or iteration % 10 == 0:
            total_w = result['weight'].sum()
            logger.info(
                f"  Raking iteration {iteration:>3d}: max_deviation={max_deviation:.4f}  "
                f"total_filers={total_w:,.0f}"
            )

        if max_deviation < tolerance:
            converged = True
            break

    if converged:
        logger.info(f"  Phase 1 converged after {iteration} iterations (tol={tolerance})")
    else:
        logger.warning(
            f"  Phase 1 did NOT converge after {max_iterations} iterations "
            f"(max_deviation={max_deviation:.4f}, tol={tolerance})"
        )

    # --- Report final match quality ---
    logger.info("\n  Phase 1 — Filer count match by AGI bracket:")
    for b_idx, (lo, hi) in enumerate(AGI_BRACKETS):
        mask = result['_bracket_idx'] == b_idx
        actual = result.loc[mask, 'weight'].sum()
        target = DOTAX_FILER_TARGETS[(lo, hi)]
        pct = (actual / target - 1) * 100 if target > 0 else 0
        label = _bracket_label(lo, hi)
        logger.info(f"    {label:<18s}  {actual:>10,.0f} vs {target:>10,d}  ({pct:>+5.1f}%)")

    logger.info("\n  Phase 1 — Filer count match by filing status:")
    for s_name in status_list:
        s_idx = status_to_int[s_name]
        mask = result['_status_idx'] == s_idx
        actual = result.loc[mask, 'weight'].sum()
        target = DOTAX_STATUS_TARGETS[s_name]
        pct = (actual / target - 1) * 100 if target > 0 else 0
        logger.info(f"    {s_name:<30s}  {actual:>10,.0f} vs {target:>10,d}  ({pct:>+5.1f}%)")

    result.drop(columns=['_bracket_idx', '_status_idx'], inplace=True)
    return result


# ---------------------------------------------------------------------------
# Phase 2: Direct bracket tax multipliers
# ---------------------------------------------------------------------------

def phase2_tax_calibrate(
    df: pd.DataFrame,
    tax_col: str = 'hi_state_tax',
    weight_col: str = 'weight',
    multiplier_floor: float = 0.3,
    multiplier_ceiling: float = 5.0,
) -> pd.DataFrame:
    """
    Scale per-unit tax liabilities so that the weighted total in every AGI
    bracket matches the DOTAX Table A8 target.

    This is a single-pass, exact calibration: one scalar multiplier per
    bracket.  No iteration, no blending, no caps that prevent convergence.

    Multiplier bounds (floor/ceiling) prevent pathological results in
    brackets where the model tax is near-zero, but they are wide enough
    that well-populated brackets will hit their target exactly.

    Args:
        df: DataFrame with per-unit tax and weight columns.
        tax_col: Name of the tax liability column.
        weight_col: Name of the weight column.
        multiplier_floor: Minimum allowed tax multiplier.
        multiplier_ceiling: Maximum allowed tax multiplier.

    Returns:
        DataFrame with adjusted tax column and new 'tax_multiplier' column.
    """
    result = df.copy()
    result['tax_multiplier'] = 1.0
    result['_bracket_idx'] = result['agi'].apply(_bracket_index)

    total_model = 0.0
    total_target = 0.0

    logger.info("\n  Phase 2 — Tax calibration by AGI bracket:")

    for b_idx, (lo, hi) in enumerate(AGI_BRACKETS):
        mask = result['_bracket_idx'] == b_idx
        n_units = mask.sum()
        if n_units == 0:
            continue

        model_tax_m = (
            result.loc[mask, tax_col] * result.loc[mask, weight_col]
        ).sum() / 1_000_000
        target_tax_m = DOTAX_TAX_TARGETS[(lo, hi)]

        if model_tax_m > 0:
            raw_mult = target_tax_m / model_tax_m
            mult = np.clip(raw_mult, multiplier_floor, multiplier_ceiling)
        else:
            mult = 1.0

        result.loc[mask, tax_col] *= mult
        result.loc[mask, 'tax_multiplier'] = mult

        new_tax_m = (
            result.loc[mask, tax_col] * result.loc[mask, weight_col]
        ).sum() / 1_000_000

        total_model += new_tax_m
        total_target += target_tax_m

        label = _bracket_label(lo, hi)
        status = "ok" if abs(new_tax_m / target_tax_m - 1) < 0.02 else "CAPPED"
        logger.info(
            f"    {label:<18s}  model ${new_tax_m:>7.1f}M  target ${target_tax_m:>7.1f}M  "
            f"x{mult:.3f}  [{status}]"
        )

    gap_pct = (total_model / total_target - 1) * 100 if total_target > 0 else 0
    logger.info(
        f"\n    TOTAL            model ${total_model:>7.1f}M  target ${total_target:>7.1f}M  "
        f"({gap_pct:>+.1f}%)"
    )

    result.drop(columns=['_bracket_idx'], inplace=True)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SimultaneousCalibrator:
    """
    Two-phase simultaneous calibration.

    Usage::

        calibrator = SimultaneousCalibrator()
        tax_units = calibrator.calibrate(tax_units)

    Prerequisites (must be done BEFORE calling calibrate):
        1. Synthetic ultra-high filers already appended
        2. Tax calculated for all units (including deductions)
        3. Columns present: 'agi', 'filing_status', 'weight', 'hi_state_tax'
    """

    def __init__(
        self,
        raking_max_iter: int = 100,
        raking_tolerance: float = 0.005,
        weight_ceiling_ratio: float = 15.0,
        tax_multiplier_ceiling: float = 5.0,
    ):
        self.raking_max_iter = raking_max_iter
        self.raking_tolerance = raking_tolerance
        self.weight_ceiling_ratio = weight_ceiling_ratio
        self.tax_multiplier_ceiling = tax_multiplier_ceiling

    def calibrate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run both calibration phases and return the calibrated DataFrame."""
        required = {'agi', 'filing_status', 'weight', 'hi_state_tax'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        logger.info("=" * 72)
        logger.info("SIMULTANEOUS TWO-PHASE CALIBRATION")
        logger.info("=" * 72)

        pre_filers = df['weight'].sum()
        pre_tax = (df['hi_state_tax'] * df['weight']).sum() / 1_000_000
        logger.info(f"  Input: {len(df):,} units, {pre_filers:,.0f} weighted filers, ${pre_tax:,.1f}M tax")

        # Phase 1: reweight to match filer counts
        logger.info("\n--- Phase 1: Entropy-balanced reweighting (raking) ---")
        result = phase1_reweight(
            df,
            max_iterations=self.raking_max_iter,
            tolerance=self.raking_tolerance,
            weight_ceiling_ratio=self.weight_ceiling_ratio,
        )

        mid_filers = result['weight'].sum()
        mid_tax = (result['hi_state_tax'] * result['weight']).sum() / 1_000_000
        logger.info(f"\n  After Phase 1: {mid_filers:,.0f} weighted filers, ${mid_tax:,.1f}M tax")

        # Phase 2: bracket tax multipliers
        logger.info("\n--- Phase 2: Direct bracket tax multipliers ---")
        result = phase2_tax_calibrate(
            result,
            multiplier_ceiling=self.tax_multiplier_ceiling,
        )

        post_filers = result['weight'].sum()
        post_tax = (result['hi_state_tax'] * result['weight']).sum() / 1_000_000
        gap = (post_tax / 3029 - 1) * 100

        logger.info("\n" + "=" * 72)
        logger.info(f"  FINAL: {post_filers:,.0f} weighted filers, ${post_tax:,.1f}M tax  ({gap:>+.1f}% vs $3,029M)")
        logger.info("=" * 72)

        return result
