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


def main():
    print("\n" + "=" * 100)
    print("COMPREHENSIVE TAX SCENARIO ANALYSIS")
    print("=" * 100)

    # Load data
    tax_units = load_tax_units()
    calculator = TaxCalculator(PROJECT_ROOT)

    # Pre-compute effective deductions once (same for all scenarios — all use 2027 std ded)
    tax_units = precompute_deductions(tax_units, calculator, standard_deduction_year=2027)

    # Define scenarios (all modeled at TY 2027 for apples-to-apples comparison)
    scenarios = {
        'Act 46 (2027 baseline)': TaxSystemRegistry.get_act46_2027_system(),
        'HB 2306 HD1 (2027)': TaxSystemRegistry.get_hb2306_hd1_2027_system(),
        'SB 3125 SD1 (2027)': TaxSystemRegistry.get_sb3125_sd1_2027_system(),
    }

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
    print("=" * 100)

    act46_pre = results['Act 46 (2027 baseline)']['total_revenue_before_credits_millions']
    hb_pre    = results['HB 2306 HD1 (2027)']['total_revenue_before_credits_millions']
    sb_pre    = results['SB 3125 SD1 (2027)']['total_revenue_before_credits_millions']

    print(f"\nAct 46 (2027 baseline):              ${act46_pre:>10,.1f}M  (pre-credit)")
    print(f"HB 2306 HD1 (2027):                  ${hb_pre:>10,.1f}M  {hb_pre-act46_pre:+>8,.1f}M vs Act 46")
    print(f"SB 3125 SD1 (2027):                  ${sb_pre:>10,.1f}M  {sb_pre-act46_pre:+>8,.1f}M vs Act 46")

    # Bracket impact breakdown
    print("\n" + "=" * 100)
    print("BRACKET IMPACT DETAIL")
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
