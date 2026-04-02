"""
Hawaii State Income Tax Liability Module

Calculates Hawaii state income tax for tax year 2023.

Hawaii has 12 marginal tax brackets with a top rate of 11%, one of the
highest in the nation. Unlike the federal system, Hawaii still has a
personal exemption ($1,144 per exemption in 2023).

Sources:
  - Hawaii Form N-11 Instructions (2023)
  - Hawaii Tax Table, Schedule Z
  - HRS § 235-51

Limitations / simplifications:
  - AGI proxy: uses the tax unit's `income` field (PINCP * ADJINC), which
    omits above-the-line deductions (IRA, HSA, student loan interest, etc.)
  - Itemized deductions are not modeled; standard deduction is always used.
  - Hawaii tax credits beyond the low-income refundable credit are not modeled.
  - MFS tax units each pay tax on their own income (no income splitting).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class HawaiiTaxParameters:
    """Hawaii income tax parameters for tax year 2023."""

    # Standard deduction amounts by filing status
    standard_deduction: Dict[str, float] = field(default_factory=lambda: {
        'single':                    2_200,
        'married_filing_jointly':    4_400,
        'married_filing_separately': 2_200,
        'head_of_household':         3_212,
    })

    # Personal exemption per exemption claimed (taxpayer + spouse + each dependent)
    personal_exemption_per_unit: float = 1_144

    # Tax brackets: list of (upper_bound, marginal_rate) pairs, from lowest to highest.
    # The last bracket has upper_bound = infinity.
    # Brackets are keyed by filing status.
    brackets: Dict[str, List[Tuple[float, float]]] = field(default_factory=lambda: {
        'single': [
            (2_400,   0.014),
            (4_800,   0.032),
            (9_600,   0.055),
            (14_400,  0.064),
            (19_200,  0.068),
            (24_000,  0.072),
            (36_000,  0.076),
            (48_000,  0.079),
            (150_000, 0.0825),
            (175_000, 0.09),
            (200_000, 0.10),
            (float('inf'), 0.11),
        ],
        'married_filing_jointly': [
            (4_800,   0.014),
            (9_600,   0.032),
            (19_200,  0.055),
            (28_800,  0.064),
            (38_400,  0.068),
            (48_000,  0.072),
            (72_000,  0.076),
            (96_000,  0.079),
            (300_000, 0.0825),
            (350_000, 0.09),
            (400_000, 0.10),
            (float('inf'), 0.11),
        ],
        'head_of_household': [
            (3_600,   0.014),
            (7_200,   0.032),
            (14_400,  0.055),
            (21_600,  0.064),
            (28_800,  0.068),
            (36_000,  0.072),
            (54_000,  0.076),
            (72_000,  0.079),
            (225_000, 0.0825),
            (262_500, 0.09),
            (300_000, 0.10),
            (float('inf'), 0.11),
        ],
        # MFS uses same brackets as single
        'married_filing_separately': [
            (2_400,   0.014),
            (4_800,   0.032),
            (9_600,   0.055),
            (14_400,  0.064),
            (19_200,  0.068),
            (24_000,  0.072),
            (36_000,  0.076),
            (48_000,  0.079),
            (150_000, 0.0825),
            (175_000, 0.09),
            (200_000, 0.10),
            (float('inf'), 0.11),
        ],
    })

    # Low-income tax refund / credit (Hawaii Form N-11, Schedule X)
    # Refundable credit for filers below income thresholds.
    # Simplified: $110 per exemption if Hawaii AGI <= $20,000 (single) / $30,000 (joint/HoH)
    low_income_credit_per_exemption: float = 110
    low_income_threshold: Dict[str, float] = field(default_factory=lambda: {
        'single':                    20_000,
        'married_filing_jointly':    30_000,
        'married_filing_separately': 20_000,
        'head_of_household':         30_000,
    })


HAWAII_2023 = HawaiiTaxParameters()


# ---------------------------------------------------------------------------
# Core calculation
# ---------------------------------------------------------------------------

def calculate_hawaii_tax(tax_unit: Dict, params: HawaiiTaxParameters = HAWAII_2023) -> Dict[str, float]:
    """
    Calculate Hawaii state income tax liability for a single tax unit.

    Args:
        tax_unit: Dictionary with:
            - filing_status: str
            - income: float  (Hawaii AGI proxy — PINCP * ADJINC, total income)
            - num_dependents: int
        params: HawaiiTaxParameters (defaults to 2023)

    Returns:
        Dictionary with:
            - hi_agi: Hawaii Adjusted Gross Income (proxy)
            - hi_standard_deduction: Standard deduction applied
            - hi_personal_exemptions: Total personal exemption amount
            - hi_taxable_income: Taxable income after deductions/exemptions
            - hi_tax_before_credits: Tax from bracket calculation
            - hi_low_income_credit: Low-income refundable credit
            - hi_tax_liability: Net Hawaii tax (may be negative if credit exceeds liability)
    """
    result = {
        'hi_agi': 0.0,
        'hi_standard_deduction': 0.0,
        'hi_personal_exemptions': 0.0,
        'hi_taxable_income': 0.0,
        'hi_tax_before_credits': 0.0,
        'hi_low_income_credit': 0.0,
        'hi_tax_liability': 0.0,
    }

    filing_status = tax_unit.get('filing_status', 'single')
    agi = float(tax_unit.get('income', 0) or 0)
    num_dependents = int(tax_unit.get('num_dependents', 0))

    result['hi_agi'] = agi

    # Standard deduction
    std_ded = params.standard_deduction.get(filing_status, params.standard_deduction['single'])
    result['hi_standard_deduction'] = std_ded

    # Personal exemptions: taxpayer + spouse (if joint) + dependents
    num_exemptions = _count_exemptions(filing_status, num_dependents)
    exemption_amount = num_exemptions * params.personal_exemption_per_unit
    result['hi_personal_exemptions'] = exemption_amount

    # Taxable income (cannot be negative)
    taxable_income = max(0.0, agi - std_ded - exemption_amount)
    result['hi_taxable_income'] = taxable_income

    # Bracket tax
    brackets = params.brackets.get(filing_status, params.brackets['single'])
    tax_before_credits = _apply_brackets(taxable_income, brackets)
    result['hi_tax_before_credits'] = tax_before_credits

    # Low-income refundable credit
    threshold = params.low_income_threshold.get(filing_status, 20_000)
    if agi <= threshold:
        low_income_credit = num_exemptions * params.low_income_credit_per_exemption
    else:
        low_income_credit = 0.0
    result['hi_low_income_credit'] = low_income_credit

    # Net liability (can be negative = refund due to low-income credit)
    result['hi_tax_liability'] = tax_before_credits - low_income_credit

    return result


def _count_exemptions(filing_status: str, num_dependents: int) -> int:
    """
    Count the number of Hawaii personal exemptions for a tax unit.

    Exemptions:
    - 1 for the taxpayer
    - 1 for the spouse (joint filers only)
    - 1 per dependent
    """
    taxpayer_exemptions = 2 if filing_status == 'married_filing_jointly' else 1
    return taxpayer_exemptions + num_dependents


def _apply_brackets(taxable_income: float, brackets: List[Tuple[float, float]]) -> float:
    """
    Apply progressive tax brackets to taxable income.

    Args:
        taxable_income: Amount to tax
        brackets: List of (upper_bound, marginal_rate) tuples, ordered low to high.
                  The last tuple should have upper_bound = float('inf').

    Returns:
        Total tax owed
    """
    if taxable_income <= 0:
        return 0.0

    total_tax = 0.0
    prev_bound = 0.0

    for upper_bound, rate in brackets:
        if taxable_income <= prev_bound:
            break
        income_in_bracket = min(taxable_income, upper_bound) - prev_bound
        total_tax += income_in_bracket * rate
        prev_bound = upper_bound

    return round(total_tax, 2)


def calculate_hawaii_tax_for_units(tax_units_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Hawaii income tax for all tax units in a DataFrame.

    Adds columns: hi_agi, hi_standard_deduction, hi_personal_exemptions,
    hi_taxable_income, hi_tax_before_credits, hi_low_income_credit, hi_tax_liability.

    Args:
        tax_units_df: DataFrame of tax units.

    Returns:
        DataFrame with Hawaii tax columns added.
    """
    results = []
    for _, row in tax_units_df.iterrows():
        unit = row.to_dict()
        hi_tax = calculate_hawaii_tax(unit)
        unit.update(hi_tax)
        results.append(unit)
    return pd.DataFrame(results)
