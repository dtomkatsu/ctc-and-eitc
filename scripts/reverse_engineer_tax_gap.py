#!/usr/bin/env python3
"""
Reverse-engineer what's missing in our tax calculation by comparing
our bracket-based rates vs DOTax empirical effective rates.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import glob

# Load latest file
files = glob.glob('data/processed/tax_units_calibrated_*.parquet')
if not files:
    print("No tax units file found")
    sys.exit(1)

latest_file = max(files)
df = pd.read_parquet(latest_file)

print('='*100)
print('REVERSE ENGINEERING: What are we missing in our tax calculation?')
print('='*100)

# DOTax effective tax rates (BEFORE Credits)
dotax_rates_before = {
    'single': {
        (0, 2400): 0.0145,
        (2400, 4800): 0.0202,
        (4800, 9600): 0.0335,
        (9600, 14400): 0.0439,
        (14400, 19200): 0.0503,
        (19200, 24000): 0.0547,
        (24000, 36000): 0.0603,
        (36000, 48000): 0.0651,
        (48000, 150000): 0.0723,
        (150000, 175000): 0.0776,
        (175000, 200000): 0.0791,
        (200000, float('inf')): 0.0987,
    },
    'married_filing_jointly': {
        (0, 4800): 0.0154,
        (4800, 9600): 0.0200,
        (9600, 19200): 0.0337,
        (19200, 28800): 0.0440,
        (28800, 38400): 0.0503,
        (38400, 48000): 0.0547,
        (48000, 72000): 0.0603,
        (72000, 96000): 0.0651,
        (96000, 300000): 0.0724,
        (300000, 350000): 0.0775,
        (350000, 400000): 0.0791,
        (400000, float('inf')): 0.0909,
    },
    'head_of_household': {
        (0, 3600): 0.0140,
        (3600, 7200): 0.0200,
        (7200, 14400): 0.0337,
        (14400, 21600): 0.0441,
        (21600, 28800): 0.0503,
        (28800, 36000): 0.0547,
        (36000, 54000): 0.0600,
        (54000, 72000): 0.0650,
        (72000, 225000): 0.0714,
        (225000, 262500): 0.0776,
        (262500, 300000): 0.0797,
        (300000, float('inf')): 0.0927,
    },
}

def get_dotax_rate(taxable_income, filing_status):
    """Get DOTax effective tax rate."""
    rates = dotax_rates_before.get(filing_status, dotax_rates_before['single'])
    for (min_ti, max_ti), rate in rates.items():
        if min_ti <= taxable_income < max_ti:
            return rate
    return list(rates.values())[-1]

# Calculate what DOTax rate implies vs what our brackets produce
df['dotax_eff_rate'] = df.apply(
    lambda row: get_dotax_rate(row['hi_taxable_income'], row['filing_status']),
    axis=1
)

df['model_eff_rate'] = df['hi_state_tax'] / df['hi_taxable_income']
df['model_eff_rate'] = df['model_eff_rate'].replace([float('inf'), -float('inf')], 0).fillna(0)

# Calculate the "missing" tax
df['rate_gap'] = df['dotax_eff_rate'] - df['model_eff_rate']
df['missing_tax'] = df['hi_taxable_income'] * df['rate_gap']

print(f'\n📊 OVERALL GAP ANALYSIS:')
print(f'   ')
total_model_tax = (df['hi_state_tax'] * df['weight']).sum() / 1_000_000
total_dotax_tax = (df['hi_taxable_income'] * df['dotax_eff_rate'] * df['weight']).sum() / 1_000_000
total_missing = (df['missing_tax'] * df['weight']).sum() / 1_000_000

print(f'   Model tax (bracket-based):     ${total_model_tax:,.1f}M')
print(f'   DOTax tax (empirical rates):   ${total_dotax_tax:,.1f}M')
print(f'   Missing tax:                   ${total_missing:,.1f}M ({total_missing/total_model_tax*100:.1f}%)')

# Analyze the gap by income level
print(f'\n🔍 WHERE IS THE MISSING TAX?')
print(f'   ')

income_brackets = [
    (0, 25000, '$0-$25k'),
    (25000, 50000, '$25k-$50k'),
    (50000, 75000, '$50k-$75k'),
    (75000, 100000, '$75k-$100k'),
    (100000, 150000, '$100k-$150k'),
    (150000, 200000, '$150k-$200k'),
    (200000, 300000, '$200k-$300k'),
    (300000, 500000, '$300k-$500k'),
    (500000, float('inf'), '$500k+'),
]

print(f'{"Income Bracket":<20} {"Filers":>10} {"Model Rate":>12} {"DOTax Rate":>12} {"Gap":>8} {"Missing $M":>12}')
print('-'*80)

for min_inc, max_inc, label in income_brackets:
    mask = (df['hi_taxable_income'] >= min_inc) & (df['hi_taxable_income'] < max_inc)
    bracket_df = df[mask]
    
    if len(bracket_df) == 0:
        continue
    
    filers = bracket_df['weight'].sum()
    
    # Weighted average rates
    total_taxable = (bracket_df['hi_taxable_income'] * bracket_df['weight']).sum()
    model_tax = (bracket_df['hi_state_tax'] * bracket_df['weight']).sum()
    dotax_tax = (bracket_df['hi_taxable_income'] * bracket_df['dotax_eff_rate'] * bracket_df['weight']).sum()
    
    model_rate = model_tax / total_taxable * 100 if total_taxable > 0 else 0
    dotax_rate = dotax_tax / total_taxable * 100 if total_taxable > 0 else 0
    gap = dotax_rate - model_rate
    
    missing = (bracket_df['missing_tax'] * bracket_df['weight']).sum() / 1_000_000
    
    print(f'{label:<20} {filers:>10,.0f} {model_rate:>11.2f}% {dotax_rate:>11.2f}% {gap:>7.2f}pp ${missing:>11,.1f}')

# Analyze by filing status
print(f'\n📋 GAP BY FILING STATUS:')
print(f'{"Filing Status":<30} {"Model Rate":>12} {"DOTax Rate":>12} {"Gap":>8} {"Missing $M":>12}')
print('-'*80)

for status in ['single', 'married_filing_jointly', 'head_of_household', 'married_filing_separately']:
    status_df = df[df['filing_status'] == status]
    if len(status_df) == 0:
        continue
    
    total_taxable = (status_df['hi_taxable_income'] * status_df['weight']).sum()
    model_tax = (status_df['hi_state_tax'] * status_df['weight']).sum()
    dotax_tax = (status_df['hi_taxable_income'] * status_df['dotax_eff_rate'] * status_df['weight']).sum()
    
    model_rate = model_tax / total_taxable * 100 if total_taxable > 0 else 0
    dotax_rate = dotax_tax / total_taxable * 100 if total_taxable > 0 else 0
    gap = dotax_rate - model_rate
    
    missing = (status_df['missing_tax'] * status_df['weight']).sum() / 1_000_000
    
    print(f'{status.replace("_", " ").title():<30} {model_rate:>11.2f}% {dotax_rate:>11.2f}% {gap:>7.2f}pp ${missing:>11,.1f}')

# Key insight: What could cause systematic underestimation?
print(f'\n💡 POTENTIAL ROOT CAUSES:')
print(f'   ')
print(f'   The model systematically underestimates tax by {total_missing/total_model_tax*100:.1f}%.')
print(f'   ')
print(f'   Possible explanations:')
print(f'   ')
print(f'   1. **Tax Brackets Issue**:')
print(f'      - Are we using the correct 2022 Hawaii tax brackets?')
print(f'      - Could there be a bracket calculation bug?')
print(f'   ')
print(f'   2. **Missing Taxes/Surcharges**:')
print(f'      - Hawaii temporary income tax surcharge (expired 2021)?')
print(f'      - Alternative Minimum Tax (AMT)?')
print(f'      - Other state-level taxes?')
print(f'   ')
print(f'   3. **Deduction/Exemption Overestimation**:')
print(f'      - Are we giving too much in deductions?')
print(f'      - Are exemptions calculated correctly?')
print(f'   ')
print(f'   4. **Taxable Income Definition**:')
print(f'      - Does DOTax "taxable income" include something we\'re missing?')
print(f'      - Are we correctly calculating taxable income?')

# Check if the gap is consistent across brackets or concentrated
print(f'\n🎯 PATTERN ANALYSIS:')
print(f'   ')

# Calculate correlation between income and rate gap
df_with_income = df[df['hi_taxable_income'] > 0].copy()
rate_gap_by_income = df_with_income.groupby(pd.cut(df_with_income['hi_taxable_income'], 
                                                     bins=[0, 50000, 100000, 200000, float('inf')],
                                                     labels=['<$50k', '$50k-$100k', '$100k-$200k', '$200k+'])).apply(
    lambda x: (x['rate_gap'] * x['hi_taxable_income'] * x['weight']).sum() / (x['hi_taxable_income'] * x['weight']).sum() * 100
)

print(f'   Average rate gap by income level:')
for bracket, gap in rate_gap_by_income.items():
    print(f'      {bracket:<15}: {gap:>6.2f} percentage points')

print(f'\n🔬 DIAGNOSTIC RECOMMENDATION:')
print(f'   ')
print(f'   1. Verify Hawaii 2022 tax brackets are correct')
print(f'   2. Check if marginal tax calculation has a bug')
print(f'   3. Investigate if deductions/exemptions are too high')
print(f'   4. Look for missing state taxes or surcharges')
print(f'   5. Compare a few sample calculations manually')
