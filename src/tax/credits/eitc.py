"""
Earned Income Tax Credit (EITC) Calculation Module

Implements the federal EITC based on 2023 tax law parameters.
The EITC is a refundable credit for low-to-moderate income workers.

Key rules:
- Must have earned income (wages or self-employment)
- Investment income must not exceed $11,000
- Married Filing Separately filers are ineligible
- Credit amount depends on filing status, number of qualifying children,
  and earned income / AGI (the lesser determines the credit)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class EITCChildParameters:
    """EITC parameters for a specific number of qualifying children."""
    max_credit: float
    phase_in_rate: float          # Rate applied to earned income in phase-in region
    phase_in_ends: float          # Earned income where max credit is first reached
    phaseout_start_single: float  # AGI/EI where phaseout begins (single/HoH/MFS)
    phaseout_start_joint: float   # AGI/EI where phaseout begins (MFJ)
    phaseout_rate: float          # Rate at which credit is reduced in phaseout region


@dataclass
class EITCParameters:
    """
    2023 EITC parameters by number of qualifying children.

    Source: IRS Revenue Procedure 2022-38
    """
    investment_income_limit: float = 11_000  # No EITC if investment income exceeds this

    by_children: Dict[int, EITCChildParameters] = field(default_factory=lambda: {
        0: EITCChildParameters(
            max_credit=600,
            phase_in_rate=0.0765,
            phase_in_ends=7_840,
            phaseout_start_single=9_800,
            phaseout_start_joint=17_640,
            phaseout_rate=0.0765,
        ),
        1: EITCChildParameters(
            max_credit=3_995,
            phase_in_rate=0.34,
            phase_in_ends=11_750,
            phaseout_start_single=21_560,
            phaseout_start_joint=28_120,
            phaseout_rate=0.1598,
        ),
        2: EITCChildParameters(
            max_credit=6_604,
            phase_in_rate=0.40,
            phase_in_ends=16_510,
            phaseout_start_single=21_560,
            phaseout_start_joint=28_120,
            phaseout_rate=0.2106,
        ),
        3: EITCChildParameters(  # 3 or more children
            max_credit=7_430,
            phase_in_rate=0.45,
            phase_in_ends=16_510,
            phaseout_start_single=21_560,
            phaseout_start_joint=28_120,
            phaseout_rate=0.2106,
        ),
    })


def calculate_eitc(tax_unit: Dict, tax_year: int = 2023) -> Dict[str, float]:
    """
    Calculate the EITC for a single tax unit.

    Args:
        tax_unit: Dictionary with keys:
            - filing_status: str
            - income: float  (AGI proxy — PINCP * ADJINC)
            - earned_income: float  (WAGP + SEMP, ADJINC-adjusted)
            - investment_income: float  (INTP, ADJINC-adjusted)
            - dependents_details: list of dicts (age, relationship, citizenship)
            - num_dependents: int
        tax_year: Tax year (default 2023)

    Returns:
        Dictionary with:
            - eitc_amount: Total EITC credit
            - eitc_qualifying_children: Number of qualifying children for EITC
            - eitc_eligible: Whether the unit is eligible for EITC at all
    """
    params = EITCParameters()

    result = {
        'eitc_amount': 0.0,
        'eitc_qualifying_children': 0,
        'eitc_eligible': False,
    }

    filing_status = tax_unit.get('filing_status', 'single')

    # MFS filers are categorically ineligible
    if filing_status == 'married_filing_separately':
        return result

    earned_income = float(tax_unit.get('earned_income', 0) or 0)
    investment_income = float(tax_unit.get('investment_income', 0) or 0)
    agi = float(tax_unit.get('income', 0) or 0)

    # Must have positive earned income
    if earned_income <= 0:
        return result

    # Investment income test — exceeding the limit disqualifies the whole unit
    if investment_income > params.investment_income_limit:
        return result

    # Count qualifying children for EITC (age < 19, or < 24 if student, or disabled)
    dependents = tax_unit.get('dependents_details') or []
    num_qualifying = _count_qualifying_children_eitc(dependents)
    result['eitc_qualifying_children'] = num_qualifying

    # Clamp to the highest-bracket key (3 = "3 or more")
    bracket_key = min(num_qualifying, 3)
    child_params = params.by_children[bracket_key]

    is_joint = filing_status == 'married_filing_jointly'
    phaseout_start = child_params.phaseout_start_joint if is_joint else child_params.phaseout_start_single

    # EITC is the lesser of:
    #   (a) credit based on earned income alone
    #   (b) credit based on AGI (for phaseout purposes, IRS uses the higher of EI or AGI)
    # The IRS rule: use earned income to determine phase-in, and the GREATER of
    # earned income or AGI for the phaseout comparison.
    phaseout_income = max(earned_income, agi)

    credit = _compute_credit(earned_income, phaseout_income, phaseout_start, child_params)

    if credit > 0:
        result['eitc_eligible'] = True
        result['eitc_amount'] = round(credit, 2)

    return result


def _count_qualifying_children_eitc(dependents: List[Dict]) -> int:
    """
    Count dependents who qualify as EITC qualifying children.

    EITC qualifying child rules (differ from CTC):
    - Age: under 19; or under 24 if a full-time student (SCHL >= 16); or any age if disabled (DIS=1)
    - Relationship: biological child (22), adopted child (23), stepchild (24),
                    grandchild (25), brother/sister (26), foster child (34)
    - Must be a US citizen/national/resident alien (CIT 1-4)
    - Cannot be married filing jointly (simplified: check MAR != 1)

    Note: EITC does NOT have the under-17 age cap that CTC uses.
    """
    QUALIFYING_RELATIONSHIPS = {22, 23, 24, 25, 26, 34}
    count = 0

    for dep in dependents:
        age = int(dep.get('age', 0))
        relationship = dep.get('relationship', 0)
        citizenship = dep.get('citizenship', 1)
        school_level = int(dep.get('school_level', 0))  # SCHL field if stored
        disabled = dep.get('disabled', False)           # DIS == 1 if stored

        # Relationship test
        try:
            rel_int = int(relationship)
        except (ValueError, TypeError):
            continue
        if rel_int not in QUALIFYING_RELATIONSHIPS:
            continue

        # Age test
        age_qualifies = (
            age < 19 or
            (age < 24 and school_level >= 16) or
            disabled
        )
        if not age_qualifies:
            continue

        # Citizenship test (CIT 1-4 = US citizen or national; 5 = not a citizen)
        try:
            cit = int(citizenship)
        except (ValueError, TypeError):
            cit = 1
        if cit > 4:
            continue

        count += 1

    return count


def _compute_credit(
    earned_income: float,
    phaseout_income: float,
    phaseout_start: float,
    params: EITCChildParameters,
) -> float:
    """
    Compute the EITC amount given earned income, phaseout income, and parameters.

    Three regions:
    1. Phase-in:   EI < phase_in_ends       → credit = EI * phase_in_rate
    2. Flat:       phase_in_ends ≤ EI ≤ phaseout_start  → credit = max_credit
    3. Phase-out:  EI > phaseout_start      → credit = max_credit - excess * phaseout_rate
    """
    # Phase-in credit based on earned income
    phase_in_credit = min(earned_income * params.phase_in_rate, params.max_credit)

    # Phase-out reduction based on phaseout_income (greater of EI or AGI)
    if phaseout_income <= phaseout_start:
        reduction = 0.0
    else:
        excess = phaseout_income - phaseout_start
        reduction = excess * params.phaseout_rate

    credit = max(0.0, phase_in_credit - reduction)
    return credit


def calculate_eitc_for_tax_units(tax_units_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate EITC for all tax units in a DataFrame.

    Adds columns: eitc_amount, eitc_qualifying_children, eitc_eligible.

    Args:
        tax_units_df: DataFrame of tax units (must include income, earned_income,
                      investment_income, filing_status, dependents_details).

    Returns:
        DataFrame with EITC columns added.
    """
    results = []
    for _, row in tax_units_df.iterrows():
        unit = row.to_dict()
        eitc = calculate_eitc(unit)
        unit.update(eitc)
        results.append(unit)
    return pd.DataFrame(results)


def get_eitc_summary_stats(tax_units_df: pd.DataFrame) -> Dict[str, float]:
    """Summary statistics for EITC across tax units (weighted by 'weight' column)."""
    if 'eitc_amount' not in tax_units_df.columns:
        tax_units_df = calculate_eitc_for_tax_units(tax_units_df)

    weight = tax_units_df.get('weight', pd.Series(1.0, index=tax_units_df.index))
    eligible = tax_units_df['eitc_eligible']

    return {
        'total_tax_units': int((weight).sum()),
        'units_receiving_eitc': int((weight[eligible]).sum()),
        'total_eitc_amount': float((tax_units_df['eitc_amount'] * weight).sum()),
        'average_eitc_per_recipient': float(
            (tax_units_df.loc[eligible, 'eitc_amount'] * weight[eligible]).sum()
            / weight[eligible].sum()
        ) if weight[eligible].sum() > 0 else 0.0,
        'total_qualifying_children': int(
            (tax_units_df['eitc_qualifying_children'] * weight).sum()
        ),
    }
