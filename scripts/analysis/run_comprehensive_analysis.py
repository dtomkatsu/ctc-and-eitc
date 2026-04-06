#!/usr/bin/env python3
"""
Comprehensive Tax Scenario Analysis: Act 46 vs HB 2306 HD1 vs SB 3125 SD1

Compares three scenarios:
  1. Act 46 (2025): current law
  2. HB 2306 HD1 (2027): top 3 rates +1pp, enhanced CDCC
  3. SB 3125 SD1 (2027): expanded brackets, formula CDCC
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from glob import glob

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tax.config import TaxSystemConfig, TaxSystemRegistry, TaxCalculator
from src.tax.adjustments.hawaii_credits import HawaiiTaxCredits
from src.tax.adjustments.itemized_deductions import ItemizedDeductionEstimator
from src.tax.adjustments.ultra_high_income_synthesizer_v2 import apply_ultra_high_income_synthesis_v2
from src.projection.source_specific_growth import (
    SourceSpecificGrowthProjector,
    _BLS_ANNUAL_RATES,
    _compute_growth_factor,
)
from src.projection.growth_rate_loader import load_all_rates

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_tax_units() -> pd.DataFrame:
    """Load the most recent calibrated tax units."""
    pattern = str(PROJECT_ROOT / "data/processed/tax_units_filing_status_calibrated_*.parquet")
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No calibrated tax unit files found matching {pattern}")

    path = files[-1]  # most recent
    df = pd.read_parquet(path)

    # Ensure num_exemptions exists
    if 'num_exemptions' not in df.columns:
        df['num_exemptions'] = df.apply(
            lambda r: (2 if r['filing_status'] == 'married_filing_jointly' else 1) + r.get('num_dependents', 0),
            axis=1,
        )

    logger.info(f"Loaded {len(df):,} tax units from {Path(path).name}")
    logger.info(f"  Weighted filers: {df['weight'].sum():,.0f}")
    logger.info(f"  Filers with dependents: {(df['num_dependents'] > 0).sum():,} "
                f"({df.loc[df['num_dependents']>0, 'weight'].sum():,.0f} weighted)")
    return df


def project_incomes_to_2027(tax_units: pd.DataFrame) -> pd.DataFrame:
    """
    Project income and filer weights from the 2022 calibrated baseline to TY 2027.

    Income growth:
      - If income components (primary_wagp, etc.) are present: use SourceSpecificGrowthProjector
        with bracket-differentiated BLS wage rates and source-specific fixed rates.
      - Otherwise: apply bracket-level AGI multipliers directly (fallback).
      Either path produces nominal 2027 AGI values, which is what we want — Hawaii's
      brackets are not inflation-indexed, so nominal growth drives bracket creep naturally.

    Population / weight adjustment:
      - Working-age (18-64): DBEDT -0.83%/yr
      - Seniors (65+):       DBEDT +5.0%/yr
    """
    logger.info("=" * 80)
    logger.info("STEP: PROJECT INCOMES AND WEIGHTS 2022 → 2027")
    logger.info("=" * 80)

    result = tax_units.copy()

    # --- 1. Income growth ---
    has_components = 'primary_wagp' in result.columns
    logger.info(f"  Income components available: {has_components}")

    agi_before = np.average(result['agi'], weights=result['weight'])
    total_agi_before = (result['agi'] * result['weight']).sum() / 1e9

    if has_components:
        projector = SourceSpecificGrowthProjector(
            years_observed=2,
            years_projected=3,
            moderation_factor=0.70,
        )
        result = projector.project_dataframe(result)
    else:
        logger.info("  Fallback: bracket-level AGI growth (no income component columns)")
        result['agi_2022'] = result['agi'].copy()
        agi = result['agi'].values.astype(float)
        grown_agi = agi.copy()
        for (lo, hi), annual_rate in _BLS_ANNUAL_RATES.items():
            mask = (agi >= lo) & (agi < hi)
            if not mask.any():
                continue
            factor = _compute_growth_factor(annual_rate, years_observed=2,
                                            years_projected=3, moderation=0.70)
            grown_agi[mask] *= factor
            n = int(mask.sum())
            logger.info(f"    ${lo/1e3:.0f}k–${hi/1e3 if hi < 1e9 else 'inf'}k  "
                        f"rate={annual_rate:.1%}/yr  factor={factor:.3f}x  n={n:,} units")
        result['agi'] = grown_agi

    agi_after = np.average(result['agi'], weights=result['weight'])
    total_agi_after = (result['agi'] * result['weight']).sum() / 1e9
    logger.info(f"\n  AGI: ${agi_before:,.0f} → ${agi_after:,.0f} (weighted avg)")
    logger.info(f"  Total AGI: ${total_agi_before:.2f}B → ${total_agi_after:.2f}B")

    # --- 2. Population / weight adjustment ---
    rates = load_all_rates(PROJECT_ROOT)
    pop_working = rates.get("pop_growth_working_age", -0.0083)
    pop_senior  = rates.get("pop_growth_senior",      0.050)
    years = 5  # 2022 → 2027

    if 'primary_agep' in result.columns:
        age = result['primary_agep'].values
        working_mask = (age >= 18) & (age < 65)
        senior_mask  = age >= 65

        working_factor = (1 + pop_working) ** years
        senior_factor  = (1 + pop_senior) ** years

        weights = result['weight'].values.copy()
        weights[working_mask] *= working_factor
        weights[senior_mask]  *= senior_factor
        result['weight'] = weights

        logger.info(f"\n  Population adjustment:")
        logger.info(f"    Working-age (18-64): {pop_working:.2%}/yr → ×{working_factor:.4f}  "
                    f"({working_mask.sum():,} units)")
        logger.info(f"    Seniors (65+):       {pop_senior:.2%}/yr → ×{senior_factor:.4f}  "
                    f"({senior_mask.sum():,} units)")
        logger.info(f"    Total weighted filers: {tax_units['weight'].sum():,.0f} → {result['weight'].sum():,.0f}")
    else:
        logger.info("  No primary_agep column — skipping age-stratified weight adjustment")

    # --- 3. Bracket-level reweighting to IRS Historic Table 2 targets ---
    try:
        from src.projection.bracket_reweighter import apply_bracket_reweighting
        result = apply_bracket_reweighting(
            result,
            target_year=2027,
            cache_dir=PROJECT_ROOT / "data/external/irs_hist2",
        )
    except Exception as _rw_exc:
        logger.warning(f"Bracket reweighting unavailable ({_rw_exc}) — skipping")

    logger.info("=" * 80)
    return result


def precompute_deductions(tax_units: pd.DataFrame, calculator: TaxCalculator,
                          standard_deduction_year: int = 2027) -> pd.DataFrame:
    """
    Compute per-unit effective deductions (max of standard vs itemized).

    Uses AGI, filing status, and filer age. Sets a fixed random seed for
    reproducibility. Applies Hawaii-specific Pease limitation (HRS § 235-2.4(f))
    using frozen 2009 AGI thresholds.

    Adds column 'effective_deduction' to the dataframe.
    """
    logger.info("Pre-computing effective deductions (standard vs itemized)...")
    estimator = ItemizedDeductionEstimator()
    np.random.seed(42)

    income_col = 'agi' if 'agi' in tax_units.columns else 'income'
    age_col = 'primary_agep' if 'primary_agep' in tax_units.columns else None

    deductions = np.zeros(len(tax_units))
    n_itemized = 0

    for i, (_, row) in enumerate(tax_units.iterrows()):
        agi = float(row[income_col])
        status = row['filing_status']
        age = int(row[age_col]) if age_col and pd.notna(row.get(age_col)) else None

        try:
            std_ded = calculator.get_standard_deduction(standard_deduction_year, status)
        except Exception:
            std_ded = 8_000

        result = estimator.get_deduction(agi, status, std_ded, age=age)
        deductions[i] = result['amount']
        if result['type'] == 'itemized':
            n_itemized += 1

    tax_units = tax_units.copy()
    tax_units['effective_deduction'] = deductions

    w = tax_units['weight'].values
    itemizer_mask = deductions > np.array([
        calculator.get_standard_deduction(standard_deduction_year, s)
        for s in tax_units['filing_status']
    ])
    logger.info(f"  Units itemizing: {n_itemized:,} ({n_itemized/len(tax_units)*100:.1f}%)")
    logger.info(f"  Weighted itemizers: {w[itemizer_mask].sum():,.0f} "
                f"({w[itemizer_mask].sum()/w.sum()*100:.1f}% of filers)")
    logger.info(f"  Avg effective deduction: ${np.average(deductions, weights=w):,.0f}")

    return tax_units


def compute_cg_tax(tax_units: pd.DataFrame) -> float:
    """
    Compute capital gains tax for synthetic $1M+ filers.

    Hawaii taxes CG at the lesser of the marginal rate or 7.25% (HRS § 235-7.5).
    For all $1M+ filers the marginal rate exceeds 7.25%, so the cap always applies.

    Uses synthetic_total_income × synthetic_cg_share × 0.0725 × weight for
    synthetic rows. Non-synthetic filers' CG is already captured in the bracket
    calculator (their agi includes ordinary income; CG is a separate line item
    handled by apply_capital_gains_to_dataframe if it was run, or ignored if not
    since PUMS top-coded $75.9M filers have negligible CG relative to synthetics).

    Returns total CG tax in millions.
    """
    is_synthetic = tax_units.get(
        'is_synthetic_ultra_high', pd.Series(False, index=tax_units.index)
    ).fillna(False)

    if not is_synthetic.any():
        logger.info("  No synthetic rows — CG tax: $0.0M")
        return 0.0

    synth = tax_units.loc[is_synthetic]
    if 'synthetic_total_income' not in synth.columns or 'synthetic_cg_share' not in synth.columns:
        logger.warning("  Synthetic rows missing total_income/cg_share — CG tax: $0.0M")
        return 0.0

    cg_income = synth['synthetic_total_income'] * synth['synthetic_cg_share']
    cg_tax_total = (cg_income * 0.0725 * synth['weight']).sum()
    cg_tax_m = cg_tax_total / 1e6

    logger.info(f"  CG tax (synthetic $1M+ filers @ 7.25% cap): ${cg_tax_m:.1f}M")
    logger.info(f"    Total synthetic CG income: ${(cg_income * synth['weight']).sum()/1e9:.2f}B")
    return cg_tax_m


def compute_scenario(tax_units: pd.DataFrame, config: TaxSystemConfig,
                    calculator: TaxCalculator, scenario_name: str) -> dict:
    """Compute revenue for a single scenario.

    Uses AGI as the income base and per-unit effective deductions
    (pre-computed via precompute_deductions — standard or itemized,
    whichever is larger, with Pease limitation applied).
    """
    logger.info(f"Computing {scenario_name}...")

    income_col = 'agi' if 'agi' in tax_units.columns else 'income'
    deduction_col = 'effective_deduction' if 'effective_deduction' in tax_units.columns else None

    result = calculator.calculate_revenue(
        tax_units,
        config,
        filing_status_col='filing_status',
        income_col=income_col,
        weight_col='weight',
        num_exemptions_col='num_exemptions',
        num_dependents_col='num_dependents',
        deduction_col=deduction_col,
    )

    logger.info(f"  {scenario_name} revenue: ${result['total_revenue_millions']:.1f}M")
    return result


def apply_synthesis(tax_units: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Pareto-based ultra-high-income synthesis ($1M+ filers).

    Logs before/after summary for the $1M+ bracket, then delegates to
    apply_ultra_high_income_synthesis_v2.  After synthesis the synthetic rows
    have num_dependents but no num_exemptions, so we recompute that column for
    the new rows here.
    """
    income_col = 'agi' if 'agi' in tax_units.columns else 'income'
    mask_1m = tax_units[income_col] >= 1_000_000

    # --- Before summary ---
    before_count  = tax_units.loc[mask_1m, 'weight'].sum()
    before_avg_agi = (
        (tax_units.loc[mask_1m, income_col] * tax_units.loc[mask_1m, 'weight']).sum()
        / tax_units.loc[mask_1m, 'weight'].sum()
        if before_count > 0 else 0.0
    )
    before_total_agi = (
        (tax_units.loc[mask_1m, income_col] * tax_units.loc[mask_1m, 'weight']).sum() / 1e9
    )

    logger.info("=" * 80)
    logger.info("PARETO SYNTHESIS — BEFORE")
    logger.info(f"  $1M+ weighted filers : {before_count:,.0f}")
    logger.info(f"  $1M+ avg AGI         : ${before_avg_agi:,.0f}")
    logger.info(f"  $1M+ total AGI       : ${before_total_agi:.2f}B")
    logger.info("=" * 80)

    # --- Clean up any leftover synthetic rows from a previous run ---
    if 'is_synthetic_ultra_high' in tax_units.columns:
        prev_synth = tax_units['is_synthetic_ultra_high'] == True  # noqa: E712
        if prev_synth.any():
            n_old = int(prev_synth.sum())
            logger.info(f"  Stripping {n_old} residual synthetic rows from previous run")
            tax_units = tax_units.loc[~prev_synth].copy()
        tax_units['is_synthetic_ultra_high'] = False

    # --- Use the natural $1M+ count after income growth ---
    # Income growth has already been applied before apply_synthesis() is called,
    # so the $1M+ count already reflects 2027 bracket creep (filers who were
    # sub-$1M in 2022 and crossed the threshold after nominal income growth).
    # No manual estimation is needed here.
    projected_1m_count = int(round(tax_units.loc[mask_1m, 'weight'].sum()))
    logger.info(f"  $1M+ count after income growth: {projected_1m_count:,} (natural, no manual projection)")

    # --- Apply synthesis ---
    tax_units = apply_ultra_high_income_synthesis_v2(
        tax_units, target_tax_m=663.0, pareto_alpha=1.5,
        total_1m_filers=projected_1m_count,
    )

    # --- Recompute num_exemptions for all rows (synthetic rows lack it) ---
    tax_units['num_exemptions'] = tax_units.apply(
        lambda r: (2 if r['filing_status'] == 'married_filing_jointly' else 1)
                  + int(r.get('num_dependents') or 0),
        axis=1,
    )

    # --- After summary (use synthetic_total_income for $1M+ gate on synthetics) ---
    # Synthetic rows have agi = ordinary income only; use synthetic_total_income if present
    total_income_col = (
        'synthetic_total_income'
        if 'synthetic_total_income' in tax_units.columns
        else income_col
    )
    is_synthetic = tax_units.get(
        'is_synthetic_ultra_high', pd.Series(False, index=tax_units.index)
    ).fillna(False)

    # For non-synthetic rows keep agi; for synthetic rows use synthetic_total_income
    representative_income = np.where(
        is_synthetic,
        tax_units[total_income_col].fillna(tax_units[income_col]),
        tax_units[income_col],
    )
    after_mask_1m = representative_income >= 1_000_000
    after_weights  = tax_units['weight'].values
    after_count    = after_weights[after_mask_1m].sum()
    after_avg_agi  = (
        np.average(representative_income[after_mask_1m], weights=after_weights[after_mask_1m])
        if after_count > 0 else 0.0
    )
    after_total_agi = (
        (representative_income[after_mask_1m] * after_weights[after_mask_1m]).sum() / 1e9
    )

    logger.info("=" * 80)
    logger.info("PARETO SYNTHESIS — AFTER")
    logger.info(f"  $1M+ weighted filers : {after_count:,.0f}")
    logger.info(f"  $1M+ avg AGI         : ${after_avg_agi:,.0f}")
    logger.info(f"  $1M+ total AGI       : ${after_total_agi:.2f}B")
    logger.info("=" * 80)

    return tax_units


def main():
    print("\n" + "=" * 100)
    print("COMPREHENSIVE TAX SCENARIO ANALYSIS")
    print("=" * 100)

    # Load data
    tax_units = load_tax_units()
    calculator = TaxCalculator(PROJECT_ROOT)

    # Project incomes and weights from 2022 calibrated baseline to TY 2027.
    # Must run BEFORE synthesis so that bracket creep is captured in the
    # natural $1M+ count rather than estimated via a separate formula.
    tax_units = project_incomes_to_2027(tax_units)

    # Apply Pareto synthesis AFTER income growth (uses natural post-growth $1M+ count)
    # and BEFORE precompute_deductions so synthetic rows are covered by the deduction pass.
    tax_units = apply_synthesis(tax_units)

    # Pre-compute effective deductions once (same for all scenarios — all use 2027 std ded)
    # Must run after synthesis so synthetic rows receive an effective_deduction.
    tax_units = precompute_deductions(tax_units, calculator, standard_deduction_year=2027)

    # Define scenarios (all modeled at TY 2027 for apples-to-apples comparison)
    scenarios = {
        'Act 46 (2027 baseline)': TaxSystemRegistry.get_act46_2027_system(),
        'HB 2306 HD1 (2027)': TaxSystemRegistry.get_hb2306_hd1_2027_system(),
        'SB 3125 SD1 (2027)': TaxSystemRegistry.get_sb3125_sd1_2027_system(),
    }

    # Compute capital gains tax once (same for all scenarios — CG taxed at capped rate)
    logger.info("Computing capital gains tax for synthetic $1M+ filers...")
    cg_tax_m = compute_cg_tax(tax_units)

    # Compute all scenarios
    results = {}
    for name, config in scenarios.items():
        try:
            results[name] = compute_scenario(tax_units, config, calculator, name)
        except Exception as e:
            logger.error(f"Error computing {name}: {e}")
            return 1

    # Print summary — bracket revenue only (CDCC excluded pending eligibility model refinement)
    print("\n" + "=" * 100)
    print("SUMMARY: Pre-Credit Revenue Comparison (millions)")
    print("Note: CDCC excluded from net — eligibility model under refinement")
    print("Note: CG tax applies to synthetic $1M+ filers only (all at 7.25% cap)")
    print("=" * 100)

    act46_pre = results['Act 46 (2027 baseline)']['total_revenue_before_credits_millions']
    hb_pre    = results['HB 2306 HD1 (2027)']['total_revenue_before_credits_millions']
    sb_pre    = results['SB 3125 SD1 (2027)']['total_revenue_before_credits_millions']

    act46_total = act46_pre + cg_tax_m
    hb_total    = hb_pre    + cg_tax_m
    sb_total    = sb_pre    + cg_tax_m

    print(f"\n{'Scenario':<40} {'Bracket':<12} {'CG Tax':<10} {'Total':<12} {'vs Act 46'}")
    print("-" * 95)
    print(f"{'Act 46 (2027 baseline)':<40} ${act46_pre:>9,.1f}M  ${cg_tax_m:>7,.1f}M  ${act46_total:>9,.1f}M")
    print(f"{'HB 2306 HD1 (2027)':<40} ${hb_pre:>9,.1f}M  ${cg_tax_m:>7,.1f}M  ${hb_total:>9,.1f}M  {hb_total-act46_total:+>8,.1f}M")
    print(f"{'SB 3125 SD1 (2027)':<40} ${sb_pre:>9,.1f}M  ${cg_tax_m:>7,.1f}M  ${sb_total:>9,.1f}M  {sb_total-act46_total:+>8,.1f}M")

    # Bracket impact breakdown
    print("\n" + "=" * 100)
    print("BRACKET IMPACT DETAIL (bracket revenue only, CG identical across scenarios)")
    print("=" * 100)

    for name in ['HB 2306 HD1 (2027)', 'SB 3125 SD1 (2027)']:
        r = results[name]
        base = results['Act 46 (2027 baseline)']
        bracket_diff = r['total_revenue_before_credits_millions'] - base['total_revenue_before_credits_millions']
        print(f"\n{name}:")
        print(f"  Bracket revenue impact:          ${bracket_diff:>10,.1f}M")

    print("\n" + "=" * 100)
    print("Analysis complete!")
    print("=" * 100 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
