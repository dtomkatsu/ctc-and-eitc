#!/usr/bin/env python3
"""
Calculate 2026 revenue using correct Act 46 tax system:
- 2024 tax brackets and rates
- 2025 standard deductions (interpolated)
- Personal exemptions ($1,200 per person)
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 2024 standard deductions (Act 46 for taxable years beginning after December 31, 2024)
STANDARD_DEDUCTIONS_2024 = {
    'single': 4400,
    'married_filing_jointly': 8800,
    'married_filing_separately': 4400,
    'head_of_household': 6424,
    'qualifying_widow': 8800
}

# Personal exemption for 2025 (indexed from 2022's $1,144)
PERSONAL_EXEMPTION_2025 = 1200

# Act 46 tax brackets for taxable years beginning after December 31, 2024 (i.e., 2025 tax year)
TAX_BRACKETS_2024 = {
    'married_filing_jointly': [
        (0, 19200, 0.014),
        (19200, 28800, 0.032),
        (28800, 38400, 0.055),
        (38400, 48000, 0.064),
        (48000, 72000, 0.068),
        (72000, 96000, 0.072),
        (96000, 250000, 0.076),
        (250000, 350000, 0.079),
        (350000, 450000, 0.0825),
        (450000, 550000, 0.090),
        (550000, 650000, 0.100),
        (650000, float('inf'), 0.110),  # Correct Act 46 top bracket
    ],
    'head_of_household': [
        (0, 14400, 0.014),
        (14400, 21600, 0.032),
        (21600, 28800, 0.055),
        (28800, 36000, 0.064),
        (36000, 54000, 0.068),
        (54000, 72000, 0.072),
        (72000, 187500, 0.076),
        (187500, 262500, 0.079),
        (262500, 337500, 0.0825),
        (337500, 412500, 0.090),
        (412500, 487500, 0.100),
        (487500, float('inf'), 0.110),
    ],
    'single': [
        (0, 9600, 0.014),
        (9600, 14400, 0.032),
        (14400, 19200, 0.055),
        (19200, 24000, 0.064),
        (24000, 36000, 0.068),
        (36000, 48000, 0.072),
        (48000, 125000, 0.076),
        (125000, 175000, 0.079),
        (175000, 225000, 0.0825),
        (225000, 275000, 0.090),
        (275000, 325000, 0.100),
        (325000, float('inf'), 0.110),
    ],
    'married_filing_separately': [
        (0, 9600, 0.014),
        (9600, 14400, 0.032),
        (14400, 19200, 0.055),
        (19200, 24000, 0.064),
        (24000, 36000, 0.068),
        (36000, 48000, 0.072),
        (48000, 125000, 0.076),
        (125000, 175000, 0.079),
        (175000, 225000, 0.0825),
        (225000, 275000, 0.090),
        (275000, 325000, 0.100),
        (325000, float('inf'), 0.110),
    ],
}


def calculate_bracket_tax(taxable_income: float, brackets: list) -> float:
    """Calculate tax using progressive bracket structure."""
    if taxable_income <= 0:
        return 0
    
    tax = 0
    for i, (min_inc, max_inc, rate) in enumerate(brackets):
        if taxable_income <= min_inc:
            break
        
        bracket_income = min(taxable_income, max_inc) - min_inc
        tax += bracket_income * rate
    
    return tax


def calculate_tax_act46(income: float, filing_status: str, num_exemptions: int = 1) -> dict:
    """
    Calculate Hawaii tax using Act 46 (2024) system with 2025 deductions and exemptions.
    
    Args:
        income: Gross income (AGI)
        filing_status: Filing status
        num_exemptions: Number of personal exemptions (typically 1 + num_dependents)
    
    Returns:
        Dict with tax calculation details
    """
    # Get standard deduction for this filing status
    std_deduction = STANDARD_DEDUCTIONS_2024.get(filing_status, 0)
    
    # Calculate personal exemptions
    personal_exemptions = num_exemptions * PERSONAL_EXEMPTION_2025
    
    # Calculate taxable income
    taxable_income = max(0, income - std_deduction - personal_exemptions)
    
    # Get brackets for this filing status
    brackets = TAX_BRACKETS_2024.get(filing_status, TAX_BRACKETS_2024['single'])
    
    # Calculate tax
    tax = calculate_bracket_tax(taxable_income, brackets)
    
    return {
        'gross_income': income,
        'standard_deduction': std_deduction,
        'personal_exemptions': personal_exemptions,
        'taxable_income': taxable_income,
        'tax_liability': tax,
        'effective_rate': (tax / income * 100) if income > 0 else 0
    }


def main():
    """Calculate 2026 revenue using correct Act 46 tax system."""
    
    logger.info("Act 46 Revenue Calculation (Corrected)")
    logger.info("="*80)
    logger.info("Tax System: Act 46 (2024 brackets/rates, 2024 standard deductions, 2025 personal exemptions)")
    logger.info("="*80)
    
    # Load 2026 tax units
    tax_units_path = Path("data/processed/projections/tax_units_2026_baseline.parquet")
    df = pd.read_parquet(tax_units_path)
    
    logger.info(f"\nLoaded {len(df):,} tax units, {df['weight'].sum():,.0f} weighted filers")
    
    # Calculate taxes with Act 46 system
    liabilities = []
    for idx, row in df.iterrows():
        income = row['income']
        status = row['filing_status']
        # Assume 1 exemption per filer + num_dependents
        num_exemptions = 1 + row.get('num_dependents', 0)
        
        tax_result = calculate_tax_act46(income, status, num_exemptions)
        liabilities.append(tax_result['tax_liability'])
    
    df['tax_liability_act46'] = liabilities
    
    # Calculate total revenue
    total_revenue = (df['tax_liability_act46'] * df['weight']).sum() / 1e6
    
    logger.info(f"\n2026 Revenue with Act 46 (corrected):")
    logger.info(f"  Total revenue: ${total_revenue:,.1f}M")
    logger.info(f"  Average tax per filer: ${total_revenue * 1e6 / df['weight'].sum():,.0f}")
    logger.info(f"  Effective rate: {(total_revenue * 1e6 / (df['income'] * df['weight']).sum()) * 100:.2f}%")
    
    # Save updated projection
    output_path = Path("data/processed/projections/tax_units_2026_act46_corrected.parquet")
    df.to_parquet(output_path)
    logger.info(f"\nSaved to: {output_path}")
    
    return total_revenue


if __name__ == "__main__":
    main()
