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
    - Interest (INTP): 5.0%/yr nominal (BEA Hawaii, rate-cycle adjusted)
    - Dividends (DIV): 5.0%/yr nominal (BEA Hawaii)
    - Retirement (RETP): 3.5%/yr nominal
    - Social Security (SSP): 3.8%/yr nominal (COLA compound 2022→2027)
    - Other income (OIP): 3.0%/yr nominal (inflation proxy)
    - Capital gains: 5%/yr nominal — tracked separately, NOT in regular income tax base
      (income tax bracket changes and capital gains tax are separate policies)

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
from src.projection.growth_rate_loader import load_all_rates
from src.tax.adjustments.itemized_deductions import ItemizedDeductionEstimator
from src.projection.timeseries import ACSIncomeForecaster, BLSTrendForecaster, DOTAXCollectionsForecaster
from src.projection.backtester import ForecastBacktester

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# Load data-driven growth rates (falls back to hardcoded defaults if data missing)
PROJECT_ROOT = Path(__file__).parent.parent if Path(__file__).parent.name == "scripts" else Path(".")
_RATES = load_all_rates(PROJECT_ROOT)

# Population growth (annual, from data or DBEDT fallback)
POP_GROWTH_WORKING_AGE = _RATES["pop_growth_working_age"]  # default: -0.83%/yr (18-64)
POP_GROWTH_SENIOR = _RATES["pop_growth_senior"]              # default: +5.0%/yr (65+)
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


HAWAII_CG_MAX_RATE = 0.0725


def compute_cg_tax(tax_units, calculator, config):
    """Compute capital gains tax: min(marginal_rate, 7.25%) × capital_gains."""
    cg_tax_total = 0.0
    cg_filers = 0.0
    if 'capital_gains' not in tax_units.columns:
        return 0.0, 0.0
    for _, row in tax_units.iterrows():
        cg = float(row.get('capital_gains', 0) or 0)
        if cg <= 0:
            continue
        try:
            ded = float(row.get('deduction_2027', 0)) if 'deduction_2027' in row.index else None
            tax_result = calculator.calculate_tax(
                row['agi'], config, row['filing_status'],
                num_exemptions=int(row.get('num_exemptions', 1)),
                deduction_override=ded,
            )
            marginal_rate = tax_result['marginal_rate'] / 100.0
            cg_rate = min(marginal_rate, HAWAII_CG_MAX_RATE)
            cg_tax_total += cg * cg_rate * row['weight']
            cg_filers += row['weight']
        except Exception:
            continue
    return cg_tax_total / 1e6, cg_filers


def compute_surcharge(tax_units, calculator, config, surcharge_rate=0.02, threshold=1_000_000):
    """Compute 2% millionaire's surcharge on total income (agi + capital gains)."""
    surcharge_total = 0.0
    affected = 0.0
    has_cg = 'agi_with_capital_gains' in tax_units.columns
    for _, row in tax_units.iterrows():
        total_income = row['agi_with_capital_gains'] if has_cg else row['agi']
        try:
            ded = float(row.get('deduction_2027', 0)) if 'deduction_2027' in row.index else None
            std_ded = calculator.get_standard_deduction(config.standard_deduction_year, row['filing_status'])
            effective_ded = ded if ded is not None else std_ded
            exemptions = int(row.get('num_exemptions', 1)) * config.personal_exemption
            taxable = max(0.0, total_income - effective_ded - exemptions)
        except Exception:
            continue
        if taxable > threshold:
            surcharge_total += (taxable - threshold) * surcharge_rate * row['weight']
            affected += row['weight']
    return surcharge_total / 1e6, affected


def run_projection_scenario(
    tax_units_base: pd.DataFrame,
    calculator: TaxCalculator,
    baseline_config,
    bls_bracket_rates: dict,
    growth_scale_factor: float = 1.0,
    label: str = "central",
    verbose: bool = True,
) -> dict:
    """
    Run full projection pipeline (Steps 2–6) and return summary dict.

    Args:
        tax_units_base: Original 2022 tax units (will be copied)
        growth_scale_factor: Multiplier on all growth rates (1.0 = central, <1 = low, >1 = high)
        label: Scenario label for logging
        verbose: If True, log detailed output
    """
    tax_units = tax_units_base.copy()
    has_components = 'primary_wagp' in tax_units.columns

    if verbose:
        logger.info(f"\n  --- Scenario: {label} (growth scale: {growth_scale_factor:.3f}) ---")

    # Step 2: Income growth
    if has_components:
        wage_rates = bls_bracket_rates if bls_bracket_rates else None
        projector = SourceSpecificGrowthProjector(
            years_observed=2,
            years_projected=3,
            moderation_factor=0.70,
            bls_annual_rates=wage_rates,
            growth_scale_factor=growth_scale_factor,
            fixed_rates=_RATES["fixed_rates"],
            cap_gains_rate=_RATES["cap_gains_rate"],
        )
        tax_units = projector.project_dataframe(tax_units)
    else:
        from src.projection.source_specific_growth import _BLS_ANNUAL_RATES, _compute_growth_factor
        tax_units['agi_2022'] = tax_units['agi'].copy()
        agi = tax_units['agi'].values.astype(float)
        for (lo, hi), rate in _BLS_ANNUAL_RATES.items():
            mask = (agi >= lo) & (agi < hi)
            factor = _compute_growth_factor(rate * growth_scale_factor, 2, 3, 0.70)
            tax_units.loc[mask, 'agi'] = agi[mask] * factor

    # Step 3: Population adjustment
    tax_units = adjust_population(tax_units)

    # Step 4: Itemized deductions
    tax_units['num_exemptions'] = calculate_num_exemptions(tax_units)
    tax_units = apply_itemized_deductions(tax_units, calculator, baseline_config)

    # Step 5: Baseline revenue
    baseline_result = calculate_revenue_scenario(
        tax_units, calculator, baseline_config, deduction_col='deduction_2027'
    )

    # Step 5b: CG tax
    cg_tax_m, cg_filers = compute_cg_tax(tax_units, calculator, baseline_config)

    # Step 6: Millionaire's surcharge
    surcharge_m, affected_filers = compute_surcharge(tax_units, calculator, baseline_config)

    total_m = baseline_result['total_revenue_millions'] + cg_tax_m

    if verbose:
        logger.info(f"    Ordinary income tax: ${baseline_result['total_revenue_millions']:,.1f}M")
        logger.info(f"    Capital gains tax:   ${cg_tax_m:,.1f}M")
        logger.info(f"    Total (ord + CG):    ${total_m:,.1f}M")
        logger.info(f"    Surcharge:           ${surcharge_m:,.1f}M ({affected_filers:,.0f} filers)")

    return {
        'ordinary_income_tax_m': baseline_result['total_revenue_millions'],
        'cg_tax_m': cg_tax_m,
        'total_m': total_m,
        'surcharge_m': surcharge_m,
        'total_filers': baseline_result['total_filers'],
        'affected_filers': affected_filers,
        'avg_tax_per_filer': baseline_result['average_tax_per_filer'],
        'avg_income': baseline_result['average_income'],
        'effective_rate': baseline_result['effective_rate'],
        'label': label,
    }


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

    # Fit BLS trend models (15-year OES series 2009-2023)
    logger.info("\n  Fitting BLS OES wage trend models (2009-2023):")
    bls_forecaster = BLSTrendForecaster(project_root=project_root)
    bls_forecaster.fit()
    bls_bracket_rates = bls_forecaster.get_all_bracket_rates()
    if bls_bracket_rates:
        logger.info(f"  BLS-derived bracket rates available for {len(bls_bracket_rates)} brackets")
    else:
        logger.info("  No BLS bracket rates available — will use hardcoded _BLS_ANNUAL_RATES")

    # Backtest ACS ETS
    backtest_result = run_backtests(project_root)

    # DOTAX collections time series
    logger.info("\n  DOTAX Individual Income Tax Collections (2016-2024):")
    dotax_forecaster = DOTAXCollectionsForecaster(project_root=project_root)
    dotax_forecaster.load_and_aggregate()
    dotax_collections_backtest = dotax_forecaster.backtest(holdout_years=2)
    dotax_central, dotax_lower, dotax_upper = dotax_forecaster.forecast_ty_liability(target_ty=2027)

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


    # ===== STEPS 2-6: RUN THREE SCENARIOS (low / central / high) =====
    logger.info("\n" + "=" * 80)
    logger.info("STEPS 2-6: PROJECTION SCENARIOS (low / central / high growth)")
    logger.info("=" * 80)

    calculator = TaxCalculator(project_root=project_root)
    baseline_2027 = TaxSystemRegistry.get_act46_2027_system()

    # Compute growth scale factors from ACS ETS confidence interval
    if central_cagr > 0.001:
        scale_low = lower_cagr / central_cagr
        scale_high = upper_cagr / central_cagr
    else:
        scale_low = scale_high = 1.0

    logger.info(f"\n  ACS ETS CAGR: {central_cagr:.1%} (80% CI: {lower_cagr:.1%} - {upper_cagr:.1%})")
    logger.info(f"  Growth scale factors: low={scale_low:.3f}, central=1.000, high={scale_high:.3f}")

    # Run all three scenarios through the full pipeline
    central = run_projection_scenario(
        tax_units, calculator, baseline_2027, bls_bracket_rates,
        growth_scale_factor=1.0, label="central", verbose=True,
    )
    low = run_projection_scenario(
        tax_units, calculator, baseline_2027, bls_bracket_rates,
        growth_scale_factor=scale_low, label="low (80% CI)", verbose=True,
    )
    high = run_projection_scenario(
        tax_units, calculator, baseline_2027, bls_bracket_rates,
        growth_scale_factor=scale_high, label="high (80% CI)", verbose=True,
    )

    # ===== CONFIDENCE INTERVAL SUMMARY =====
    logger.info("\n" + "=" * 80)
    logger.info("CONFIDENCE INTERVALS (full model re-run, not heuristic)")
    logger.info("=" * 80)

    logger.info(f"\n  {'Metric':<30s}  {'Low':>10s}  {'Central':>10s}  {'High':>10s}")
    logger.info(f"  {'-'*30}  {'-'*10}  {'-'*10}  {'-'*10}")
    logger.info(f"  {'Ordinary income tax':<30s}  ${low['ordinary_income_tax_m']:>8,.1f}M  ${central['ordinary_income_tax_m']:>8,.1f}M  ${high['ordinary_income_tax_m']:>8,.1f}M")
    logger.info(f"  {'Capital gains tax':<30s}  ${low['cg_tax_m']:>8,.1f}M  ${central['cg_tax_m']:>8,.1f}M  ${high['cg_tax_m']:>8,.1f}M")
    logger.info(f"  {'Total (ord + CG)':<30s}  ${low['total_m']:>8,.1f}M  ${central['total_m']:>8,.1f}M  ${high['total_m']:>8,.1f}M")
    logger.info(f"  {'Surcharge (2% millionaire)':<30s}  ${low['surcharge_m']:>8,.1f}M  ${central['surcharge_m']:>8,.1f}M  ${high['surcharge_m']:>8,.1f}M")

    # ===== FINAL SUMMARY =====
    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)

    backtest_mape = backtest_result['mape'] if backtest_result else float('nan')

    dotax_mape = dotax_collections_backtest.get('mape', 9.4)
    dotax_corrected = dotax_central * (1 - dotax_mape / 100)

    total_with_cg = central['total_m']
    surcharge_revenue = central['surcharge_m']
    affected_count = central['affected_filers']
    millionaire_total_m = central['ordinary_income_tax_m'] + surcharge_revenue

    logger.info(f"""
  +-----------------------------------------------------------------+
  |  REVENUE ESTIMATES: TY2027                                      |
  +-----------------------------------------------------------------+
  |  2022 Calibrated Baseline:      ${baseline_tax_2022:>8,.1f}M  (618,423 filers) |
  |                                                                 |
  |  -- Microsimulation: Act 46 Law (primary estimate) --           |
  |  2027 Ordinary income tax:      ${central['ordinary_income_tax_m']:>8,.1f}M  ({central['total_filers']:>7,.0f} filers) |
  |  2027 Capital gains tax:        ${central['cg_tax_m']:>8,.1f}M  (max 7.25% rate)      |
  |  2027 Total (ord + CG):         ${total_with_cg:>8,.1f}M                        |
  |  2027 Total 80% CI:             ${low['total_m']:>8,.1f}M - ${high['total_m']:,.1f}M         |
  |  2027 + 2% Millionaire Tax:     ${millionaire_total_m:>8,.1f}M  (ordinary income)  |
  |  Surcharge Revenue:             ${surcharge_revenue:>8,.1f}M  ({affected_count:>6,.0f} filers)  |
  |  Surcharge 80% CI:              ${low['surcharge_m']:>8,.1f}M - ${high['surcharge_m']:,.1f}M         |
  |                                                                 |
  |  -- DOTAX Collections ETS (trend extrapolation, no law change) -|
  |  TY2027 raw ETS estimate:       ${dotax_central:>8,.1f}M  (MAPE {dotax_mape:.1f}%)       |
  |  TY2027 bias-corrected:         ${dotax_corrected:>8,.1f}M                        |
  |  DOTAX estimate 80% CI:         ${dotax_lower:>8,.1f}M - ${dotax_upper:,.1f}M         |
  |                                                                 |
  |  -- Interpretation --                                           |
  |  Microsim vs DOTAX:             ${(total_with_cg - dotax_corrected):>+8.1f}M ({(total_with_cg/dotax_corrected - 1)*100:>+.1f}%)          |
  |  Implied Act 46 revenue cost:   ${dotax_corrected - total_with_cg:>8,.1f}M vs trend          |
  +-----------------------------------------------------------------+

  Change 2022 -> 2027 (ord + CG):  {(total_with_cg/baseline_tax_2022 - 1)*100:+.1f}%
  Millionaire's Tax Uplift:       {(surcharge_revenue/central['ordinary_income_tax_m'])*100:+.1f}%

  Methodology:
    Income growth: Source-specific (wages BLS bracket-specific, SS 3.8%, ret 3.5%, inv 5.0%)
    Population: DBEDT demographics (working-age -0.83%/yr, seniors +5.0%/yr)
    Deductions: SOI-calibrated itemized deductions
    Capital gains: 5%/yr nominal, taxed at min(marginal_rate, 7.25%)
    Confidence intervals: Full model re-run at ACS ETS 80% CI growth bounds
    ACS ETS backtest MAPE:           {backtest_mape:.1f}%
    DOTAX collections backtest MAPE: {dotax_mape:.1f}% (COVID outlier corrected)
""")


if __name__ == '__main__':
    main()
