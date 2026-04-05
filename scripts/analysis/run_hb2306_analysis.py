#!/usr/bin/env python3
"""
HB 2306 HD1 Impact Analysis

Compares Act 46 baseline (2025/2027) vs HB 2306 HD1:
  - Bracket changes: top 3 rates +1pp (9->10%, 10->11%, 11->12%)
  - Enhanced CDCC: refundable, $10k/$20k expense caps, 50%-5% rate schedule
  - Sunset extension: Act 163 credits extended to 2032

Output:
  1. Revenue comparison table (bracket impact + credit impact)
  2. Distributional analysis by AGI bracket
  3. CDCC impact breakdown
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

from src.tax.config import TaxSystemConfig, TaxSystemRegistry, TaxCalculator, compare_systems
from src.tax.adjustments.hawaii_credits import HawaiiTaxCredits

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
        extra = 1 if 'married_filing_jointly' not in df['filing_status'].values else 0
        df['num_exemptions'] = df.apply(
            lambda r: (2 if r['filing_status'] == 'married_filing_jointly' else 1) + r.get('num_dependents', 0),
            axis=1,
        )

    logger.info(f"Loaded {len(df):,} tax units from {Path(path).name}")
    logger.info(f"  Weighted filers: {df['weight'].sum():,.0f}")
    logger.info(f"  Filers with dependents: {(df['num_dependents'] > 0).sum():,} "
                f"({df.loc[df['num_dependents']>0, 'weight'].sum():,.0f} weighted)")
    return df


def distributional_analysis(
    tax_units: pd.DataFrame,
    baseline_config: TaxSystemConfig,
    scenario_config: TaxSystemConfig,
    calculator: TaxCalculator,
) -> pd.DataFrame:
    """Per-AGI-bracket comparison of baseline vs HB 2306."""

    brackets = [
        (0, 25_000, "Under $25k"),
        (25_000, 50_000, "$25k-$50k"),
        (50_000, 75_000, "$50k-$75k"),
        (75_000, 100_000, "$75k-$100k"),
        (100_000, 200_000, "$100k-$200k"),
        (200_000, 500_000, "$200k-$500k"),
        (500_000, 1_000_000, "$500k-$1M"),
        (1_000_000, float('inf'), "$1M+"),
    ]

    rows = []
    for lo, hi, label in brackets:
        mask = (tax_units['income'] >= lo) & (tax_units['income'] < hi)
        subset = tax_units[mask]
        if subset.empty:
            continue

        base_rev = calculator.calculate_revenue(subset, baseline_config)
        scen_rev = calculator.calculate_revenue(subset, scenario_config)

        rows.append({
            'AGI Bracket': label,
            'Filers': f"{subset['weight'].sum():,.0f}",
            'Baseline $M': f"{base_rev['total_revenue_millions']:,.1f}",
            'HB2306 $M': f"{scen_rev['total_revenue_millions']:,.1f}",
            'Diff $M': f"{scen_rev['total_revenue_millions'] - base_rev['total_revenue_millions']:+,.1f}",
            'Baseline CDCC $M': f"{base_rev['total_cdcc_millions']:,.1f}",
            'HB2306 CDCC $M': f"{scen_rev['total_cdcc_millions']:,.1f}",
            'CDCC Diff $M': f"{scen_rev['total_cdcc_millions'] - base_rev['total_cdcc_millions']:+,.1f}",
        })

    return pd.DataFrame(rows)


def main():
    print("\n" + "=" * 100)
    print("HB 2306 HD1 IMPACT ANALYSIS")
    print("=" * 100)

    # Load data
    tax_units = load_tax_units()
    calculator = TaxCalculator(PROJECT_ROOT)

    # Define systems
    baseline = TaxSystemRegistry.get_act46_2025_system()
    hb2306 = TaxSystemRegistry.get_hb2306_hd1_2027_system()

    # ── Overall comparison ──
    print("\n" + "=" * 100)
    print("1. OVERALL REVENUE COMPARISON: Act 46 (2025) vs HB 2306 HD1 (2027)")
    print("=" * 100)

    comparison = compare_systems(tax_units, baseline, hb2306, calculator)
    print("\n" + comparison.to_string(index=False))

    # Extract key numbers
    base_row = comparison.iloc[0]
    hb_row = comparison.iloc[1]
    diff_row = comparison.iloc[2]

    print(f"\n  Bracket impact (pre-credit): "
          f"${diff_row['revenue_before_credits_millions']:+,.1f}M")
    print(f"  Credit impact:               "
          f"${-diff_row['total_credits_millions']:+,.1f}M")
    print(f"  CDCC impact:                 "
          f"${-diff_row['total_cdcc_millions']:+,.1f}M")
    print(f"  NET revenue impact:          "
          f"${diff_row['revenue_millions']:+,.1f}M")

    # ── Bracket-only comparison (ignore credits) ──
    print("\n" + "=" * 100)
    print("2. BRACKET-ONLY IMPACT (top 3 rates +1pp)")
    print("=" * 100)

    # Create a version of HB2306 without enhanced credits for isolation
    hb2306_brackets_only = TaxSystemConfig(
        name="hb2306_brackets_only",
        year=2027,
        bracket_year=2027,
        standard_deduction_year=2025,
        personal_exemption=1200,
        description="HB2306 brackets only (no enhanced CDCC)",
        bracket_scenario='hb2306_hd1',
        credit_scenario=None,  # Use current-law credits
    )

    bracket_comparison = compare_systems(tax_units, baseline, hb2306_brackets_only, calculator)
    bracket_diff = bracket_comparison.iloc[2]
    print(f"\n  Revenue from +1pp on top 3 rates: ${bracket_diff['revenue_millions']:+,.1f}M")

    # ── CDCC-only comparison ──
    print("\n" + "=" * 100)
    print("3. ENHANCED CDCC IMPACT (refundable, expanded thresholds)")
    print("=" * 100)

    # Use baseline brackets but enhanced credits
    cdcc_only = TaxSystemConfig(
        name="cdcc_enhanced_only",
        year=2027,
        bracket_year=2025,
        standard_deduction_year=2025,
        personal_exemption=1200,
        description="Current brackets + HB2306 enhanced CDCC",
        credit_scenario='hb2306_hd1',
    )

    cdcc_comparison = compare_systems(tax_units, baseline, cdcc_only, calculator)
    cdcc_diff = cdcc_comparison.iloc[2]
    print(f"\n  Revenue impact from enhanced CDCC: ${cdcc_diff['revenue_millions']:+,.1f}M")
    print(f"  Additional CDCC cost: ${cdcc_diff['total_cdcc_millions']:+,.1f}M")

    # ── Distributional analysis ──
    print("\n" + "=" * 100)
    print("4. DISTRIBUTIONAL ANALYSIS BY AGI BRACKET")
    print("=" * 100)

    dist = distributional_analysis(tax_units, baseline, hb2306, calculator)
    print("\n" + dist.to_string(index=False))

    # ── Spot checks ──
    print("\n" + "=" * 100)
    print("5. SPOT CHECKS")
    print("=" * 100)

    spot_cases = [
        (700_000, 'married_filing_jointly', 0, "Joint filer $700k, no kids"),
        (50_000, 'head_of_household', 2, "HoH $50k, 2 kids"),
        (100_000, 'married_filing_jointly', 2, "Joint $100k, 2 kids"),
        (200_000, 'single', 0, "Single $200k, no kids"),
    ]

    credit_calc_base = HawaiiTaxCredits(year=2025)
    credit_calc_hb = HawaiiTaxCredits(year=2027)

    for income, status, deps, desc in spot_cases:
        base_tax = calculator.calculate_tax(income, baseline, status, 1 + deps)
        hb_tax = calculator.calculate_tax(income, hb2306, status, 1 + deps)

        base_cr = credit_calc_base.calculate_total_credits(
            income, status, deps, base_tax['tax_liability'], credit_scenario=None)
        hb_cr = credit_calc_hb.calculate_total_credits(
            income, status, deps, hb_tax['tax_liability'], credit_scenario='hb2306_hd1')

        base_net = base_tax['tax_liability'] - base_cr['total']
        hb_net = hb_tax['tax_liability'] - hb_cr['total']

        print(f"\n  {desc}:")
        print(f"    Baseline: tax=${base_tax['tax_liability']:,.0f}, credits=${base_cr['total']:,.0f}, "
              f"CDCC=${base_cr['child_care']:,.0f}, net=${base_net:,.0f}")
        print(f"    HB 2306:  tax=${hb_tax['tax_liability']:,.0f}, credits=${hb_cr['total']:,.0f}, "
              f"CDCC=${hb_cr['child_care']:,.0f}, net=${hb_net:,.0f}")
        print(f"    Diff:     ${hb_net - base_net:+,.0f}")

    print("\n" + "=" * 100)
    print("Analysis complete!")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
