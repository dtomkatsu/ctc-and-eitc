#!/usr/bin/env python3
"""
Project calibrated 2022 tax units to Tax Year 2027.

Steps:
    1. Load calibrated 2022 baseline (618,423 filers, $3,029M tax)
    2. Grow incomes by bracket-specific real growth rates (5 years)
    3. Calculate 2027 baseline revenue (Act 46 brackets + expanded deductions)
    4. Calculate 2027 millionaire's tax revenue (+ 2% surcharge on taxable income > $1M)
    5. Output comparison
"""

import sys
from pathlib import Path
from glob import glob

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
import logging

from src.tax.config.tax_system_config import (
    TaxCalculator,
    TaxSystemRegistry,
    compare_systems,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bracket-specific real annual growth rates (BLS OES derived, 2022→2027)
# ---------------------------------------------------------------------------
ANNUAL_GROWTH_RATES = {
    (0, 25_000):          0.028,  # 2.8%
    (25_000, 50_000):     0.025,  # 2.5%
    (50_000, 75_000):     0.022,  # 2.2%
    (75_000, 100_000):    0.020,  # 2.0%
    (100_000, 200_000):   0.018,  # 1.8%
    (200_000, float('inf')): 0.017,  # 1.7%
}

YEARS_FORWARD = 5  # 2022 → 2027


def grow_incomes(df: pd.DataFrame, agi_col: str = 'agi') -> pd.DataFrame:
    """Apply bracket-specific income growth over 5 years."""
    result = df.copy()
    result['agi_2022'] = result[agi_col].copy()

    agi = result[agi_col].values.astype(float)
    growth_factors = np.ones(len(agi))

    for (lo, hi), annual_rate in ANNUAL_GROWTH_RATES.items():
        mask = (agi >= lo) & (agi < hi)
        growth_factors[mask] = (1 + annual_rate) ** YEARS_FORWARD

    result[agi_col] = agi * growth_factors
    result['income_growth_factor'] = growth_factors

    # Also grow capital gains proportionally if present
    if 'capital_gains' in result.columns:
        result['capital_gains'] = result['capital_gains'] * growth_factors

    logger.info("Income growth applied (2022 → 2027):")
    for (lo, hi), annual_rate in ANNUAL_GROWTH_RATES.items():
        factor = (1 + annual_rate) ** YEARS_FORWARD
        label = f"${lo/1000:.0f}k–${hi/1000:.0f}k" if hi != float('inf') else f"${lo/1000:.0f}k+"
        n = ((agi >= lo) & (agi < hi)).sum()
        logger.info(f"  {label:<15s}  {annual_rate:.1%}/yr  x{factor:.3f} over 5yr  ({n:,} units)")

    avg_factor = np.average(growth_factors, weights=result['weight'].values)
    logger.info(f"  Weighted average growth factor: x{avg_factor:.3f}")

    return result


def calculate_num_exemptions(df: pd.DataFrame) -> pd.Series:
    """Derive number of exemptions from available columns."""
    adults = df.get('num_adults', pd.Series(np.ones(len(df))))
    if 'num_dependents' in df.columns:
        return adults.fillna(1).astype(int) + df['num_dependents'].fillna(0).astype(int)
    return adults.fillna(1).astype(int)


def main():
    logger.info("=" * 80)
    logger.info("2027 REVENUE PROJECTION WITH MILLIONAIRE'S TAX SCENARIO")
    logger.info("=" * 80)

    # --- Load calibrated 2022 baseline ---
    parquet_files = sorted(glob(str(project_root / "data/processed/tax_units_filing_status_calibrated_*.parquet")))
    if not parquet_files:
        raise FileNotFoundError("No calibrated tax unit parquet files found")
    latest = parquet_files[-1]
    logger.info(f"\nLoading baseline: {latest}")

    tax_units = pd.read_parquet(latest)
    logger.info(f"  {len(tax_units):,} tax units, {tax_units['weight'].sum():,.0f} weighted filers")

    baseline_tax_2022 = (tax_units['hi_state_tax'] * tax_units['weight']).sum() / 1e6
    logger.info(f"  2022 calibrated tax: ${baseline_tax_2022:,.1f}M")

    # --- Grow incomes to 2027 ---
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: INCOME GROWTH (2022 → 2027)")
    logger.info("=" * 80)

    tax_units = grow_incomes(tax_units, agi_col='agi')

    avg_agi_2022 = np.average(tax_units['agi_2022'], weights=tax_units['weight'])
    avg_agi_2027 = np.average(tax_units['agi'], weights=tax_units['weight'])
    logger.info(f"\n  Weighted avg AGI: ${avg_agi_2022:,.0f} (2022) → ${avg_agi_2027:,.0f} (2027)")

    # --- Prepare exemptions ---
    tax_units['num_exemptions'] = calculate_num_exemptions(tax_units)

    # --- Initialize calculator ---
    calculator = TaxCalculator(project_root=project_root)

    # --- Get configs ---
    baseline_2027 = TaxSystemRegistry.get_act46_2027_system()
    millionaire_2027 = TaxSystemRegistry.get_millionaire_tax_2027(surcharge_rate=0.02)

    # --- Calculate baseline 2027 revenue ---
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: BASELINE 2027 REVENUE (Act 46 brackets + expanded deductions)")
    logger.info("=" * 80)

    baseline_result = calculator.calculate_revenue(
        tax_units, baseline_2027,
        filing_status_col='filing_status',
        income_col='agi',
        weight_col='weight',
        num_exemptions_col='num_exemptions',
    )

    logger.info(f"\n  Baseline 2027 Revenue:")
    logger.info(f"    Total revenue:      ${baseline_result['total_revenue_millions']:,.1f}M")
    logger.info(f"    Total filers:       {baseline_result['total_filers']:,.0f}")
    logger.info(f"    Avg tax per filer:  ${baseline_result['average_tax_per_filer']:,.0f}")
    logger.info(f"    Avg income:         ${baseline_result['average_income']:,.0f}")
    logger.info(f"    Effective rate:     {baseline_result['effective_rate']:.2f}%")
    logger.info(f"    vs 2022 baseline:   ${baseline_tax_2022:,.1f}M → ${baseline_result['total_revenue_millions']:,.1f}M "
                f"({(baseline_result['total_revenue_millions']/baseline_tax_2022 - 1)*100:+.1f}%)")

    # --- Calculate millionaire's tax 2027 revenue ---
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: MILLIONAIRE'S TAX 2027 (2% surcharge on taxable income > $1M)")
    logger.info("=" * 80)

    millionaire_result = calculator.calculate_revenue(
        tax_units, millionaire_2027,
        filing_status_col='filing_status',
        income_col='agi',
        weight_col='weight',
        num_exemptions_col='num_exemptions',
    )

    surcharge_revenue = millionaire_result['total_revenue_millions'] - baseline_result['total_revenue_millions']

    logger.info(f"\n  Millionaire's Tax 2027 Revenue:")
    logger.info(f"    Total revenue:      ${millionaire_result['total_revenue_millions']:,.1f}M")
    logger.info(f"    Surcharge revenue:  ${surcharge_revenue:,.1f}M")
    logger.info(f"    Avg tax per filer:  ${millionaire_result['average_tax_per_filer']:,.0f}")
    logger.info(f"    Effective rate:     {millionaire_result['effective_rate']:.2f}%")

    # --- Detailed surcharge analysis ---
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: SURCHARGE IMPACT ANALYSIS")
    logger.info("=" * 80)

    affected_count = 0.0
    affected_surcharge_total = 0.0
    surcharge_by_status = {}

    for _, row in tax_units.iterrows():
        income = row['agi']
        status = row['filing_status']
        weight = row['weight']
        num_ex = int(row.get('num_exemptions', 1))

        try:
            result = calculator.calculate_tax(income, millionaire_2027, status, num_ex)
        except Exception:
            continue

        surcharge = result.get('surcharge_amount', 0)
        if surcharge > 0:
            affected_count += weight
            affected_surcharge_total += surcharge * weight
            surcharge_by_status[status] = surcharge_by_status.get(status, 0) + surcharge * weight

    logger.info(f"\n  Affected filers:    {affected_count:,.0f} ({affected_count / tax_units['weight'].sum() * 100:.2f}%)")
    logger.info(f"  Total surcharge:    ${affected_surcharge_total / 1e6:,.1f}M")
    if affected_count > 0:
        logger.info(f"  Avg surcharge:      ${affected_surcharge_total / affected_count:,.0f} per affected filer")

    logger.info("\n  Surcharge by filing status:")
    for status, total in sorted(surcharge_by_status.items(), key=lambda x: -x[1]):
        logger.info(f"    {status:<30s}  ${total / 1e6:>8.1f}M")

    # --- Revenue by AGI bracket comparison ---
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: REVENUE BY AGI BRACKET")
    logger.info("=" * 80)

    agi_brackets = [
        (0, 25_000), (25_000, 50_000), (50_000, 75_000), (75_000, 100_000),
        (100_000, 150_000), (150_000, 200_000), (200_000, 300_000),
        (300_000, 500_000), (500_000, 1_000_000), (1_000_000, float('inf')),
    ]

    logger.info(f"\n  {'Bracket':<20s}  {'Filers':>10s}  {'Baseline':>12s}  {'Millionaire':>12s}  {'Surcharge':>12s}")
    logger.info(f"  {'-'*20}  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*12}")

    for lo, hi in agi_brackets:
        mask = (tax_units['agi'] >= lo) & (tax_units['agi'] < hi)
        if not mask.any():
            continue

        bracket_units = tax_units[mask]
        w = bracket_units['weight'].values
        filers = w.sum()

        # Calculate baseline tax for this bracket
        base_taxes = []
        mill_taxes = []
        for _, row in bracket_units.iterrows():
            try:
                b = calculator.calculate_tax(row['agi'], baseline_2027, row['filing_status'],
                                             int(row.get('num_exemptions', 1)))
                m = calculator.calculate_tax(row['agi'], millionaire_2027, row['filing_status'],
                                             int(row.get('num_exemptions', 1)))
                base_taxes.append(b['tax_liability'])
                mill_taxes.append(m['tax_liability'])
            except Exception:
                base_taxes.append(0)
                mill_taxes.append(0)

        base_rev = np.sum(np.array(base_taxes) * w) / 1e6
        mill_rev = np.sum(np.array(mill_taxes) * w) / 1e6
        sur_rev = mill_rev - base_rev

        label = f"${lo/1000:.0f}k–${hi/1000:.0f}k" if hi != float('inf') else f"${lo/1000:.0f}k+"
        logger.info(f"  {label:<20s}  {filers:>10,.0f}  ${base_rev:>10.1f}M  ${mill_rev:>10.1f}M  ${sur_rev:>10.1f}M")

    # --- Final summary ---
    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info(f"""
  2022 Calibrated Baseline:     ${baseline_tax_2022:,.1f}M  (618,423 filers)
  2027 Baseline (Act 46):       ${baseline_result['total_revenue_millions']:,.1f}M  ({baseline_result['total_filers']:,.0f} filers)
  2027 + 2% Millionaire's Tax:  ${millionaire_result['total_revenue_millions']:,.1f}M
  Surcharge Revenue:            ${surcharge_revenue:,.1f}M  ({affected_count:,.0f} affected filers)

  Change 2022 → 2027 Baseline:  {(baseline_result['total_revenue_millions']/baseline_tax_2022 - 1)*100:+.1f}%
  Millionaire's Tax Uplift:     {(surcharge_revenue/baseline_result['total_revenue_millions'])*100:+.1f}%
""")


if __name__ == '__main__':
    main()
