"""
Capital Gains Estimation for Hawaii

Calibrated directly to DOTAX SOI 2022 Table 21:
    Total capital gains: $2,995M from 43,443 resident filers

Approach:
    1. Assign capital gains participation using Table 21 filer counts
    2. Distribute initial amounts proportionally to AGI within each bracket
    3. Scale bracket totals to match Table 21 targets exactly
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# DOTAX SOI 2022 Table 21 — Residents
# Bracket: (filers_with_cap_gains, total_amount_millions)
TABLE_21_TARGETS = {
    (0, 10_000):          {'filers': 34,    'amount_m': 0.5},
    (10_000, 20_000):     {'filers': 13,    'amount_m': 0.0},
    (20_000, 30_000):     {'filers': 234,   'amount_m': 0.3},
    (30_000, 40_000):     {'filers': 1_616, 'amount_m': 3.6},
    (40_000, 50_000):     {'filers': 2_045, 'amount_m': 7.7},
    (50_000, 75_000):     {'filers': 6_044, 'amount_m': 30.1},
    (75_000, 100_000):    {'filers': 6_067, 'amount_m': 46.0},
    (100_000, 150_000):   {'filers': 9_146, 'amount_m': 112.0},
    (150_000, 200_000):   {'filers': 6_095, 'amount_m': 126.0},
    (200_000, 300_000):   {'filers': 5_577, 'amount_m': 242.0},
    (300_000, 400_000):   {'filers': 2_384, 'amount_m': 216.9},
    (400_000, float('inf')): {'filers': 4_188, 'amount_m': 2_210.3},
}

BRACKETS = sorted(TABLE_21_TARGETS.keys())


def _assign_bracket(agi: np.ndarray) -> np.ndarray:
    """Assign each AGI value to a Table 21 bracket index."""
    bounds = np.array([lo for lo, _ in BRACKETS] + [np.inf])
    return np.digitize(agi, bounds) - 1


def apply_capital_gains_to_dataframe(
    df: pd.DataFrame,
    agi_col: str = 'agi',
    taxable_income_col: Optional[str] = None,
    include_random: bool = True,
    assignment_method: str = 'random',
) -> pd.DataFrame:
    """
    Assign capital gains to tax units, calibrated to Table 21 bracket targets.

    Steps:
        1. Compute participation rate per bracket from Table 21 filer counts
        2. Select filers (top-AGI within bracket for deterministic, random otherwise)
        3. Assign initial cap gains proportional to AGI
        4. Scale per-bracket totals to match Table 21 amounts exactly

    Returns DataFrame with 'capital_gains' and 'agi_with_cap_gains' columns.
    """
    result = df.copy()
    n = len(result)
    agi = result[agi_col].values.astype(float)
    weights = result['weight'].values.astype(float)

    bracket_idx = _assign_bracket(agi)
    cap_gains = np.zeros(n)
    has_cg = np.zeros(n, dtype=bool)

    rng = np.random.default_rng(seed=42)

    for b_idx, (lo, hi) in enumerate(BRACKETS):
        mask = bracket_idx == b_idx
        if not mask.any():
            continue

        target = TABLE_21_TARGETS[(lo, hi)]
        target_filers = target['filers']
        target_amount_m = target['amount_m']

        if target_filers == 0 or target_amount_m <= 0:
            continue

        bracket_agi = agi[mask]
        bracket_weights = weights[mask]
        total_weighted = bracket_weights.sum()

        if total_weighted < 1:
            continue

        # Participation rate from Table 21
        participation_rate = min(target_filers / total_weighted, 1.0)

        # Select which filers get capital gains
        bracket_indices = np.where(mask)[0]

        if assignment_method == 'deterministic_top' or assignment_method == 'deterministic_income':
            # Top AGI filers get capital gains
            sorted_by_agi = bracket_indices[np.argsort(-bracket_agi)]
            cumulative_weight = np.cumsum(bracket_weights[np.argsort(-bracket_agi)])
            n_selected = np.searchsorted(cumulative_weight, target_filers, side='right')
            n_selected = max(1, min(n_selected, len(sorted_by_agi)))
            selected = sorted_by_agi[:n_selected]
        elif assignment_method == 'deterministic_hash':
            # Hash-based deterministic selection
            hash_vals = np.array([hash(f"{a:.2f}_{w:.2f}") % 10000 for a, w in zip(bracket_agi, bracket_weights)])
            sorted_by_hash = bracket_indices[np.argsort(-hash_vals)]
            cumulative_weight = np.cumsum(bracket_weights[np.argsort(-hash_vals)])
            n_selected = np.searchsorted(cumulative_weight, target_filers, side='right')
            n_selected = max(1, min(n_selected, len(sorted_by_hash)))
            selected = sorted_by_hash[:n_selected]
        else:
            # Random assignment
            probs = rng.random(len(bracket_indices))
            selected = bracket_indices[probs < participation_rate]
            if len(selected) == 0 and target_filers > 0:
                selected = bracket_indices[np.argsort(-bracket_agi)[:1]]

        has_cg[selected] = True

        # Assign initial cap gains proportional to AGI
        selected_agi = agi[selected]
        selected_weights = weights[selected]

        if selected_agi.sum() > 0:
            # Distribute proportionally to AGI
            agi_share = selected_agi / selected_agi.sum()
            # Initial assignment: spread target amount across selected filers
            # weighted by their AGI share, then adjust for weights
            weighted_total = (selected_weights * agi_share).sum()
            if weighted_total > 0:
                # Each filer's cap gains = target_total * (their_agi / sum_agi) * correction
                raw_cg = selected_agi * (target_amount_m * 1e6) / (selected_agi * selected_weights).sum()
                cap_gains[selected] = raw_cg
            else:
                cap_gains[selected] = target_amount_m * 1e6 / len(selected)
        else:
            cap_gains[selected] = target_amount_m * 1e6 / max(len(selected), 1)

        # Post-hoc bracket calibration: scale to match Table 21 exactly
        weighted_cg = (cap_gains[mask] * weights[mask]).sum()
        if weighted_cg > 0:
            scalar = (target_amount_m * 1e6) / weighted_cg
            cap_gains[mask] *= scalar

        # Cap individual cap gains at 3x AGI to prevent extreme values
        cg_mask = mask & (cap_gains > 0)
        if cg_mask.any():
            pre_cap_total = (cap_gains[cg_mask] * weights[cg_mask]).sum()
            max_cg = agi[cg_mask] * 3.0
            was_capped = cap_gains[cg_mask] > max_cg
            cap_gains[cg_mask] = np.minimum(cap_gains[cg_mask], max_cg)
            post_cap_total = (cap_gains[cg_mask] * weights[cg_mask]).sum()
            n_capped = weights[cg_mask][was_capped].sum()
            if pre_cap_total > post_cap_total + 1:
                logger.info(f"    3x AGI cap: {n_capped:,.0f} filers capped, "
                            f"${(pre_cap_total - post_cap_total)/1e6:.1f}M removed")

        # Post-cap re-calibration: scale remaining CG to still hit Table 21 target
        weighted_cg_post_cap = (cap_gains[mask] * weights[mask]).sum()
        if weighted_cg_post_cap > 0 and abs(weighted_cg_post_cap - target_amount_m * 1e6) > 1e3:
            rescale = (target_amount_m * 1e6) / weighted_cg_post_cap
            cap_gains[mask] *= rescale
            if abs(rescale - 1.0) > 0.001:
                logger.info(f"    Post-cap rescale: x{rescale:.3f} to match Table 21 target")

        # Final check after capping and re-calibration
        weighted_cg_final = (cap_gains[mask] * weights[mask]).sum() / 1e6
        label = f"${lo/1000:.0f}k\u2013${hi/1000:.0f}k" if hi != float('inf') else f"${lo/1000:.0f}k+"
        selected_weighted = weights[selected].sum()
        logger.info(
            f"  {label:<18s}  filers: {selected_weighted:>8,.0f} (target {target_filers:>6,})  "
            f"amount: ${weighted_cg_final:>8.1f}M (target ${target_amount_m:>8.1f}M)"
        )

    result['capital_gains'] = cap_gains
    result['has_capital_gains'] = has_cg
    result['agi_with_cap_gains'] = agi + cap_gains

    total_cg_m = (cap_gains * weights).sum() / 1e6
    total_filers_cg = weights[has_cg].sum()
    logger.info(f"\n  Total capital gains: ${total_cg_m:,.1f}M (target: $2,995M)")
    logger.info(f"  Filers with cap gains: {total_filers_cg:,.0f} (target: 43,443)")

    # CG/AGI ratio distribution for validation
    cg_filers_mask = cap_gains > 0
    if cg_filers_mask.any():
        ratios = cap_gains[cg_filers_mask] / np.maximum(agi[cg_filers_mask], 1)
        logger.info(f"  CG/AGI ratio: p50={np.median(ratios):.2f}, p90={np.percentile(ratios, 90):.2f}, "
                    f"p99={np.percentile(ratios, 99):.2f}, max={ratios.max():.2f}")

    return result


# Legacy class interface for backward compatibility
class CapitalGainsEstimator:
    """Kept for any code that imports the class directly."""
    def __init__(self):
        self.total_cap_gains_millions = 2995
        self.total_filers_with_cap_gains = 43443
