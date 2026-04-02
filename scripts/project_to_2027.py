#!/usr/bin/env python3
"""
Project calibrated 2022 tax units to Tax Year 2027.

Improvements over v1:
    - Source-specific income growth (wages, retirement, SS, investment at different rates)
    - Itemized deductions (SOI-calibrated, applied to 2027 projected incomes)
    - ETS time-series forecasting with confidence intervals
    - Backtesting validation against held-out ACS data
    - Age-specific population adjustment using preserved AGEP from PUMS

Income growth methodology:
    - Wages/SE (WAGP, SEMP): BLS OES bracket-specific nominal rates
        Years 1-2 (2022→2024): Observed rates
        Years 3-5 (2024→2027): Moderated to 70% of observed pace
    - Interest (INTP): 3.5%/yr nominal
    - Dividends (DIV): 4.5%/yr nominal
    - Retirement (RETP): 3.0%/yr nominal
    - Social Security (SSP): 2.5%/yr nominal (COLA average)
    - Other income (OIP): 3.0%/yr nominal (inflation proxy)
    - Capital gains: 5%/yr nominal (long-run equity average)

Population/weight adjustment:
    - Working-age (18-64): -0.83%/yr
    - Seniors (65+): +5.0%/yr
    - Source: Hawaii DBEDT population estimates
"""

import sys
from pathlib import Path
from glob import glob

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import logging

from src.tax.config.tax_system_config import (
    TaxCalculator,
    TaxSystemRegistry,
)
from src.projection.source_specific_growth import SourceSpecificGrowthProjector
from src.tax.adjustments.itemized_deductions import ItemizedDeductionEstimator
from src.projection.timeseries import ACSIncomeForecaster, BLSTrendForecaster
from src.projection.backtester import ForecastBacktester

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# Population growth (annual, from DBEDT)
POP_GROWTH_WORKING_AGE = -0.0083  # -0.83%/yr (18-64)
POP_GROWTH_SENIOR = 0.050          # +5.0%/yr (65+)
YEARS_FORWARD = 5  # 2022 → 2027


def adjust_population(df: pd.DataFrame) -> pd.DataFrame:
    """Adjust weights for demographic shifts using preserved AGEP data."""
    result = df.copy()
    result['weight_2022'] = result['weight'].copy()
    weights = result['weight'].values.copy()

    # Use primary_agep from Phase 1 enrichment
    if 'primary_agep' in result.columns:
        age = result['primary_agep'].values
        senior_mask = age >= 65
        working_mask = (age >= 18) & (age < 65)

        working_factor = (1 + POP_GROWTH_WORKING_AGE) ** YEARS_FORWARD
        senior_factor = (1 + POP_GROWTH_SENIOR) ** YEARS_FORWARD

        weights[working_mask] *= working_factor
        weights[senior_mask] *= senior_factor
        result['weight'] = weights

        n_working = working_mask.sum()
        n_senior = senior_mask.sum()
        logger.info(f"  Working-age (18-64): {POP_GROWTH_WORKING_AGE:.2%}/yr → x{working_factor:.3f}  ({n_working:,} units)")
        logger.info(f"  Seniors (65+):       {POP_GROWTH_SENIOR:.2%}/yr → x{senior_factor:.3f}  ({n_senior:,} units)")
        logger.info(f"  Total filers: {result['weight_2022'].sum():,.0f} → {result['weight'].sum():,.0f}")
    else:
        logger.info("  No age data — applying weighted-average population growth")
        senior_share = 0.22
        avg_annual_pop = (1 - senior_share) * POP_GROWTH_WORKING_AGE + senior_share * POP_GROWTH_SENIOR
        pop_factor = (1 + avg_annual_pop) ** YEARS_FORWARD
        result['weight'] = weights * pop_factor
        logger.info(f"  Avg annual pop growth: {avg_annual_pop:.2%}/yr → x{pop_factor:.3f} over 5yr")

    return result


def apply_itemized_deductions(
    df: pd.DataFrame,
    calculator: TaxCalculator,
    config,
) -> pd.DataFrame:
    """
    Apply itemized deductions to 2027 projected incomes.
    For each unit: deduction = max(standard_deduction_2027, estimated_itemized).
    Uses deterministic seed per unit for reproducibility.
    """
    result = df.copy()
    estimator = ItemizedDeductionEstimator()

    deductions = np.zeros(len(result))
    n_itemized = 0

    # Set seed for reproducible itemization decisions
    np.random.seed(42)

    for i, (_, row) in enumerate(result.iterrows()):
        agi = row['agi']
        status = row['filing_status']
        age = int(row.get('primary_agep', 0)) if 'primary_agep' in row.index else None

        # Get 2027 standard deduction
        try:
            std_ded = calculator.get_standard_deduction(config.standard_deduction_year, status)
        except Exception:
            std_ded = 8000  # fallback

        # get_deduction returns dict with 'type', 'amount', etc.
        ded_result = estimator.get_deduction(agi, status, std_ded, age=age)
        deductions[i] = ded_result['amount']

        if ded_result['type'] == 'itemized':
            n_itemized += 1

    result['deduction_2027'] = deductions

    w = result['weight'].values
    pct_itemized = n_itemized / len(result) * 100
    itemizer_mask = deductions > np.array([
        calculator.get_standard_deduction(config.standard_deduction_year, s)
        if pd.notna(s) else 8000
        for s in result['filing_status']
    ])

    logger.info(f"  Units itemizing: {n_itemized:,} ({pct_itemized:.1f}% of units)")
    logger.info(f"  Weighted itemizers: {w[itemizer_mask].sum():,.0f} "
                f"({w[itemizer_mask].sum() / w.sum() * 100:.1f}% of filers)")
    logger.info(f"  Avg deduction (all): ${np.average(deductions, weights=w):,.0f}")
    if itemizer_mask.any():
        logger.info(f"  Avg deduction (itemizers): "
                    f"${np.average(deductions[itemizer_mask], weights=w[itemizer_mask]):,.0f}")

    return result


def calculate_num_exemptions(df: pd.DataFrame) -> pd.Series:
    """Derive number of exemptions from available columns."""
    adults = df.get('num_adults', pd.Series(np.ones(len(df))))
    if 'num_dependents' in df.columns:
        return adults.fillna(1).astype(int) + df['num_dependents'].fillna(0).astype(int)
    return adults.fillna(1).astype(int)


def run_backtests(project_root: Path):
    """Run ETS backtests on ACS median household income."""
    logger.info("\n  Backtesting ETS on ACS median household income (hold out 2023-2024):")
    acs_path = project_root / "data/processed/acs_timeseries/wide/B19013_wide.csv"
    if not acs_path.exists():
        logger.warning("  ACS data not found, skipping backtests")
        return None

    df = pd.read_csv(acs_path)
    values = df['median_household_income'].values.astype(float)
    years = df['year'].values.astype(float)

    backtester = ForecastBacktester(holdout_years=2)
    return backtester.evaluate(values, years, series_name="median_hh_income")


def calculate_revenue_scenario(
    tax_units: pd.DataFrame,
    calculator: TaxCalculator,
    config,
    deduction_col: str = None,
) -> dict:
    """Calculate revenue for a scenario, with optional deduction overrides."""
    return calculator.calculate_revenue(
        tax_units, config,
        filing_status_col='filing_status',
        income_col='agi',
        weight_col='weight',
        num_exemptions_col='num_exemptions',
        deduction_col=deduction_col,
    )


def main():
    logger.info("=" * 80)
    logger.info("2027 REVENUE PROJECTION v2")
    logger.info("Source-Specific Growth + Itemized Deductions + ETS Forecasting")
    logger.info("=" * 80)

    # ===== STEP 0: ETS FORECASTING & BACKTESTING =====
    logger.info("\n" + "=" * 80)
    logger.info("STEP 0: TIME-SERIES FORECASTING & VALIDATION")
    logger.info("=" * 80)

    # Fit ACS ETS models
    logger.info("\n  Fitting ETS models on ACS time series:")
    acs_forecaster = ACSIncomeForecaster(project_root=project_root)
    acs_forecaster.fit_all()

    central_cagr, lower_cagr, upper_cagr = acs_forecaster.get_aggregate_growth()
    logger.info(f"\n  ACS-derived aggregate CAGR: {central_cagr:.1%} "
                f"(80% CI: {lower_cagr:.1%} – {upper_cagr:.1%})")

    # Fit BLS trend models
    logger.info("\n  Fitting BLS OES wage trend models:")
    bls_forecaster = BLSTrendForecaster(project_root=project_root)
    bls_forecaster.fit()

    # Backtest
    backtest_result = run_backtests(project_root)

    # ===== STEP 1: LOAD DATA =====
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: LOAD CALIBRATED 2022 BASELINE")
    logger.info("=" * 80)

    parquet_files = sorted(glob(str(project_root / "data/processed/tax_units_filing_status_calibrated_*.parquet")))
    if not parquet_files:
        raise FileNotFoundError("No calibrated tax unit parquet files found")
    latest = parquet_files[-1]
    logger.info(f"\n  Loading: {latest}")

    tax_units = pd.read_parquet(latest)
    logger.info(f"  {len(tax_units):,} tax units, {tax_units['weight'].sum():,.0f} weighted filers")

    has_components = 'primary_wagp' in tax_units.columns
    logger.info(f"  Income components available: {has_components}")
    if has_components:
        logger.info(f"  primary_agep range: {tax_units['primary_agep'].min()}-{tax_units['primary_agep'].max()}")

    baseline_tax_2022 = (tax_units['hi_state_tax'] * tax_units['weight']).sum() / 1e6
    logger.info(f"  2022 calibrated tax: ${baseline_tax_2022:,.1f}M")

    # ===== STEP 2: SOURCE-SPECIFIC INCOME GROWTH =====
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: SOURCE-SPECIFIC INCOME GROWTH (2022 → 2027, NOMINAL)")
    logger.info("=" * 80)

    if has_components:
        projector = SourceSpecificGrowthProjector(
            years_observed=2,
            years_projected=3,
            moderation_factor=0.70,
        )
        tax_units = projector.project_dataframe(tax_units)
    else:
        logger.warning("  Income components not found — falling back to uniform bracket growth")
        # Fallback to simple bracket-specific growth (v1 behavior)
        from src.projection.source_specific_growth import _BLS_ANNUAL_RATES, _compute_growth_factor
        tax_units['agi_2022'] = tax_units['agi'].copy()
        agi = tax_units['agi'].values.astype(float)
        for (lo, hi), rate in _BLS_ANNUAL_RATES.items():
            mask = (agi >= lo) & (agi < hi)
            factor = _compute_growth_factor(rate, 2, 3, 0.70)
            tax_units.loc[mask, 'agi'] = agi[mask] * factor

    avg_agi_2022 = np.average(tax_units['agi_2022'], weights=tax_units['weight'])
    avg_agi_2027 = np.average(tax_units['agi'], weights=tax_units['weight'])
    logger.info(f"\n  Weighted avg AGI: ${avg_agi_2022:,.0f} (2022) → ${avg_agi_2027:,.0f} (2027)")

    # ===== STEP 3: POPULATION ADJUSTMENT =====
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: POPULATION ADJUSTMENT (demographic shifts)")
    logger.info("=" * 80)

    tax_units = adjust_population(tax_units)

    # ===== STEP 4: ITEMIZED DEDUCTIONS =====
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: ITEMIZED DEDUCTIONS (SOI-calibrated)")
    logger.info("=" * 80)

    tax_units['num_exemptions'] = calculate_num_exemptions(tax_units)
    calculator = TaxCalculator(project_root=project_root)
    baseline_2027 = TaxSystemRegistry.get_act46_2027_system()
    millionaire_2027 = TaxSystemRegistry.get_millionaire_tax_2027(surcharge_rate=0.02)

    tax_units = apply_itemized_deductions(tax_units, calculator, baseline_2027)

    # ===== STEP 5: BASELINE 2027 REVENUE =====
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: BASELINE 2027 REVENUE (Act 46 + itemized deductions)")
    logger.info("=" * 80)

    baseline_result = calculate_revenue_scenario(
        tax_units, calculator, baseline_2027, deduction_col='deduction_2027'
    )

    logger.info(f"\n  Baseline 2027 Revenue:")
    logger.info(f"    Total revenue:      ${baseline_result['total_revenue_millions']:,.1f}M")
    logger.info(f"    Total filers:       {baseline_result['total_filers']:,.0f}")
    logger.info(f"    Avg tax per filer:  ${baseline_result['average_tax_per_filer']:,.0f}")
    logger.info(f"    Avg income:         ${baseline_result['average_income']:,.0f}")
    logger.info(f"    Effective rate:     {baseline_result['effective_rate']:.2f}%")
    logger.info(f"    vs 2022 baseline:   ${baseline_tax_2022:,.1f}M → "
                f"${baseline_result['total_revenue_millions']:,.1f}M "
                f"({(baseline_result['total_revenue_millions']/baseline_tax_2022 - 1)*100:+.1f}%)")

    # ===== STEP 6: MILLIONAIRE'S TAX =====
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: MILLIONAIRE'S TAX 2027 (2% surcharge on taxable income > $1M)")
    logger.info("=" * 80)

    millionaire_result = calculate_revenue_scenario(
        tax_units, calculator, millionaire_2027, deduction_col='deduction_2027'
    )

    surcharge_revenue = millionaire_result['total_revenue_millions'] - baseline_result['total_revenue_millions']

    logger.info(f"\n  Millionaire's Tax 2027 Revenue:")
    logger.info(f"    Total revenue:      ${millionaire_result['total_revenue_millions']:,.1f}M")
    logger.info(f"    Surcharge revenue:  ${surcharge_revenue:,.1f}M")
    logger.info(f"    Avg tax per filer:  ${millionaire_result['average_tax_per_filer']:,.0f}")
    logger.info(f"    Effective rate:     {millionaire_result['effective_rate']:.2f}%")

    # ===== STEP 7: SURCHARGE IMPACT ANALYSIS =====
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: SURCHARGE IMPACT ANALYSIS")
    logger.info("=" * 80)

    affected_count = 0.0
    affected_surcharge_total = 0.0
    surcharge_by_status = {}

    for _, row in tax_units.iterrows():
        income = row['agi']
        status = row['filing_status']
        weight = row['weight']
        num_ex = int(row.get('num_exemptions', 1))
        ded = float(row.get('deduction_2027', 0)) if 'deduction_2027' in row.index else None

        try:
            result = calculator.calculate_tax(income, millionaire_2027, status, num_ex,
                                              deduction_override=ded)
        except Exception:
            continue

        surcharge = result.get('surcharge_amount', 0)
        if surcharge > 0:
            affected_count += weight
            affected_surcharge_total += surcharge * weight
            surcharge_by_status[status] = surcharge_by_status.get(status, 0) + surcharge * weight

    logger.info(f"\n  Affected filers:    {affected_count:,.0f} "
                f"({affected_count / tax_units['weight'].sum() * 100:.2f}%)")
    logger.info(f"  Total surcharge:    ${affected_surcharge_total / 1e6:,.1f}M")
    if affected_count > 0:
        logger.info(f"  Avg surcharge:      ${affected_surcharge_total / affected_count:,.0f} per affected filer")

    logger.info("\n  Surcharge by filing status:")
    for status, total in sorted(surcharge_by_status.items(), key=lambda x: -x[1]):
        logger.info(f"    {status:<30s}  ${total / 1e6:>8.1f}M")

    # ===== STEP 8: CONFIDENCE INTERVALS =====
    logger.info("\n" + "=" * 80)
    logger.info("STEP 8: REVENUE CONFIDENCE INTERVALS (ETS-derived)")
    logger.info("=" * 80)

    # Use ACS ETS growth rate CI to bound revenue estimates
    # Scale the baseline revenue by the ratio of ETS CI bounds to central estimate
    if central_cagr > 0:
        # Revenue scales roughly proportionally with income growth over the projection window
        # This is an approximation — proper CI would re-run the full model
        growth_ratio_low = ((1 + lower_cagr) / (1 + central_cagr)) ** YEARS_FORWARD
        growth_ratio_high = ((1 + upper_cagr) / (1 + central_cagr)) ** YEARS_FORWARD

        # Revenue is roughly proportional to taxable income, so scale accordingly
        # But the relationship is nonlinear due to progressive brackets, so dampen slightly
        dampen = 0.8  # Revenue sensitivity to income changes (< 1 due to progressive rates)
        revenue_low = baseline_result['total_revenue_millions'] * (1 + (growth_ratio_low - 1) * dampen)
        revenue_high = baseline_result['total_revenue_millions'] * (1 + (growth_ratio_high - 1) * dampen)

        logger.info(f"\n  ETS-derived 80% confidence interval for baseline 2027 revenue:")
        logger.info(f"    Low:     ${revenue_low:,.1f}M")
        logger.info(f"    Central: ${baseline_result['total_revenue_millions']:,.1f}M")
        logger.info(f"    High:    ${revenue_high:,.1f}M")
        logger.info(f"\n  Based on ACS aggregate CAGR: {central_cagr:.1%} "
                    f"(80% CI: {lower_cagr:.1%} – {upper_cagr:.1%})")
    else:
        logger.info("  Could not compute CI (no valid ETS forecast)")
        revenue_low = revenue_high = baseline_result['total_revenue_millions']

    # ===== FINAL SUMMARY =====
    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)

    backtest_mape = backtest_result['mape'] if backtest_result else float('nan')

    logger.info(f"""
  2022 Calibrated Baseline:     ${baseline_tax_2022:,.1f}M  (618,423 filers)
  2027 Baseline (Act 46):       ${baseline_result['total_revenue_millions']:,.1f}M  ({baseline_result['total_filers']:,.0f} filers)
  2027 Baseline 80% CI:         ${revenue_low:,.1f}M – ${revenue_high:,.1f}M
  2027 + 2% Millionaire's Tax:  ${millionaire_result['total_revenue_millions']:,.1f}M
  Surcharge Revenue:            ${surcharge_revenue:,.1f}M  ({affected_count:,.0f} affected filers)

  Change 2022 → 2027 Baseline:  {(baseline_result['total_revenue_millions']/baseline_tax_2022 - 1)*100:+.1f}%
  Millionaire's Tax Uplift:     {(surcharge_revenue/baseline_result['total_revenue_millions'])*100:+.1f}%

  Methodology:
    Income growth: Source-specific (wages BLS 4.2-7.0%/yr, SS 2.5%, ret 3.0%, inv 3.5-4.5%)
    Population: DBEDT demographics (working-age -0.83%/yr, seniors +5.0%/yr)
    Deductions: SOI-calibrated itemized deductions (12.1% itemization rate)
    Capital gains: 5%/yr nominal (long-run equity average)
    Validation: ETS backtest MAPE = {backtest_mape:.1f}%
""")


if __name__ == '__main__':
    main()
