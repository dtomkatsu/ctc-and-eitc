#!/usr/bin/env python3
"""
Diagnose why taxable income is on target but tax liability is 14.8% below target.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import glob
import numpy as np

# Load latest file
files = glob.glob('data/processed/tax_units_calibrated_*.parquet')
if not files:
    print("No tax units file found")
    sys.exit(1)

latest_file = max(files)
df = pd.read_parquet(latest_file)

print('='*80)
print('TAX LIABILITY GAP DIAGNOSIS')
print('='*80)

# Calculate totals
num_filers = df['weight'].sum()
total_taxable = (df['hi_taxable_income'] * df['weight']).sum() / 1_000_000
total_tax = (df['hi_state_tax'] * df['weight']).sum() / 1_000_000

# DOTax benchmarks
dotax_total_taxable = 40517  # millions
dotax_total_tax = 4770  # millions (Table 21, adjusted for 2022)

print(f'\n📊 OVERALL COMPARISON:')
print(f'   ')
print(f'   Taxable Income:')
print(f'     Our model:                 ${total_taxable:,.1f}M')
print(f'     DOTax target:              ${dotax_total_taxable:,.1f}M')
print(f'     Difference:                ${total_taxable - dotax_total_taxable:+,.1f}M ({(total_taxable/dotax_total_taxable-1)*100:+.1f}%)')
print(f'   ')
print(f'   Tax Liability:')
print(f'     Our model:                 ${total_tax:,.1f}M')
print(f'     DOTax target:              ${dotax_total_tax:,.1f}M')
print(f'     Difference:                ${total_tax - dotax_total_tax:+,.1f}M ({(total_tax/dotax_total_tax-1)*100:+.1f}%)')
print(f'   ')
print(f'   Effective Tax Rate:')
print(f'     Our model:                 {total_tax/total_taxable*100:.2f}%')
print(f'     DOTax implied:             {dotax_total_tax/dotax_total_taxable*100:.2f}%')
print(f'     Difference:                {(total_tax/total_taxable - dotax_total_tax/dotax_total_taxable)*100:+.2f} pp')

# Calculate effective tax rates by income bracket
print(f'\n🔍 EFFECTIVE TAX RATES BY TAXABLE INCOME BRACKET:')
print(f'{"Bracket":<20} {"Filers":>10} {"Taxable ($M)":>15} {"Tax ($M)":>12} {"Eff Rate":>10}')
print('-'*80)

brackets = [
    (0, 10000, '$0-$10k'),
    (10000, 20000, '$10k-$20k'),
    (20000, 30000, '$20k-$30k'),
    (30000, 40000, '$30k-$40k'),
    (40000, 50000, '$40k-$50k'),
    (50000, 75000, '$50k-$75k'),
    (75000, 100000, '$75k-$100k'),
    (100000, 150000, '$100k-$150k'),
    (150000, 200000, '$150k-$200k'),
    (200000, 300000, '$200k-$300k'),
    (300000, 400000, '$300k-$400k'),
    (400000, float('inf'), '$400k+'),
]

for min_inc, max_inc, label in brackets:
    mask = (df['hi_taxable_income'] >= min_inc) & (df['hi_taxable_income'] < max_inc)
    bracket_df = df[mask]
    
    if len(bracket_df) == 0:
        continue
    
    bracket_filers = bracket_df['weight'].sum()
    bracket_taxable = (bracket_df['hi_taxable_income'] * bracket_df['weight']).sum() / 1_000_000
    bracket_tax = (bracket_df['hi_state_tax'] * bracket_df['weight']).sum() / 1_000_000
    
    if bracket_taxable > 0:
        eff_rate = bracket_tax / bracket_taxable * 100
    else:
        eff_rate = 0
    
    print(f'{label:<20} {bracket_filers:>10,.0f} {bracket_taxable:>15,.1f} {bracket_tax:>12,.1f} {eff_rate:>9.2f}%')

# Check distribution of taxable income vs DOTax
print(f'\n📊 DISTRIBUTION COMPARISON:')
print(f'   ')

# DOTax 2022 Table 12A income distribution (AGI brackets, approximate)
dotax_brackets_agi = {
    (0, 10000): {'filers': 136395, 'tax': 3.0},
    (10000, 20000): {'filers': 63036, 'tax': 21.0},
    (20000, 30000): {'filers': 57382, 'tax': 51.0},
    (30000, 40000): {'filers': 60023, 'tax': 92.0},
    (40000, 50000): {'filers': 53551, 'tax': 116.0},
    (50000, 75000): {'filers': 91603, 'tax': 293.0},
    (75000, 100000): {'filers': 54955, 'tax': 261.0},
    (100000, 150000): {'filers': 62112, 'tax': 438.0},
    (150000, 200000): {'filers': 28013, 'tax': 294.0},
    (200000, 300000): {'filers': 18908, 'tax': 310.0},
    (300000, 400000): {'filers': 6076, 'tax': 153.0},
    (400000, float('inf')): {'filers': 8875, 'tax': 998.0},
}

print(f'{"AGI Bracket":<20} {"DOTax Filers":>15} {"Our Filers":>15} {"Diff":>10} {"DOTax Tax":>12} {"Our Tax":>12} {"Diff":>10}')
print('-'*110)

for (min_agi, max_agi), dotax_data in dotax_brackets_agi.items():
    mask = (df['agi'] >= min_agi) & (df['agi'] < max_agi)
    our_filers = df[mask]['weight'].sum()
    our_tax = (df[mask]['hi_state_tax'] * df[mask]['weight']).sum() / 1_000_000
    
    if max_agi == float('inf'):
        label = f'${min_agi//1000:,}k+'
    else:
        label = f'${min_agi//1000:,}k-${max_agi//1000:,}k'
    
    filer_diff = (our_filers / dotax_data['filers'] - 1) * 100 if dotax_data['filers'] > 0 else 0
    tax_diff = (our_tax / dotax_data['tax'] - 1) * 100 if dotax_data['tax'] > 0 else 0
    
    print(f'{label:<20} {dotax_data["filers"]:>15,} {our_filers:>15,.0f} {filer_diff:>9.1f}% {dotax_data["tax"]:>11,.1f} {our_tax:>11,.1f} {tax_diff:>9.1f}%')

# Check for potential issues
print(f'\n🎯 POTENTIAL ROOT CAUSES:')
print(f'   ')

# 1. Check if high-income tax is too low
high_income_mask = df['hi_taxable_income'] >= 200000
high_income_taxable = (df[high_income_mask]['hi_taxable_income'] * df[high_income_mask]['weight']).sum() / 1_000_000
high_income_tax = (df[high_income_mask]['hi_state_tax'] * df[high_income_mask]['weight']).sum() / 1_000_000
high_income_eff_rate = high_income_tax / high_income_taxable * 100 if high_income_taxable > 0 else 0

# DOTax high income ($200k+)
dotax_high_income_tax = 153.0 + 998.0 + 310.0  # $200k-$300k, $300k-$400k, $400k+
dotax_high_income_filers = 6076 + 8875 + 18908

print(f'   1. HIGH-INCOME BRACKETS ($200k+):')
print(f'      Our tax from high-income:  ${high_income_tax:,.1f}M')
print(f'      DOTax tax from high-income: ${dotax_high_income_tax:,.1f}M')
print(f'      Shortfall:                  ${high_income_tax - dotax_high_income_tax:+,.1f}M ({(high_income_tax/dotax_high_income_tax-1)*100:+.1f}%)')
print(f'      Our eff rate (high income): {high_income_eff_rate:.2f}%')

# 2. Check marginal tax rates
print(f'\n   2. TAX BRACKET APPLICATION:')
print(f'      Check if Hawaii tax brackets are correct')
print(f'      Check if marginal rates are applied correctly')

# 3. Check if credits are reducing tax too much
if 'tax_before_credits' in df.columns:
    total_tax_before_credits = (df['tax_before_credits'] * df['weight']).sum() / 1_000_000
    total_credits = total_tax_before_credits - total_tax
    print(f'\n   3. TAX CREDITS:')
    print(f'      Tax before credits:         ${total_tax_before_credits:,.1f}M')
    print(f'      Total credits applied:      ${total_credits:,.1f}M')
    print(f'      Tax after credits:          ${total_tax:,.1f}M')
    print(f'      Credits as % of pre-credit: {total_credits/total_tax_before_credits*100:.1f}%')

# 4. Check average tax by filing status
print(f'\n   4. TAX BY FILING STATUS:')
print(f'      {"Status":<25} {"Filers":>10} {"Avg Taxable":>15} {"Avg Tax":>12} {"Eff Rate":>10}')
print(f'      ' + '-'*80)
for status in ['single', 'married_filing_jointly', 'head_of_household', 'married_filing_separately']:
    status_df = df[df['filing_status'] == status]
    if len(status_df) == 0:
        continue
    
    status_filers = status_df['weight'].sum()
    avg_taxable = (status_df['hi_taxable_income'] * status_df['weight']).sum() / status_filers
    avg_tax = (status_df['hi_state_tax'] * status_df['weight']).sum() / status_filers
    eff_rate = avg_tax / avg_taxable * 100 if avg_taxable > 0 else 0
    
    print(f'      {status:<25} {status_filers:>10,.0f} ${avg_taxable:>14,.0f} ${avg_tax:>11,.0f} {eff_rate:>9.2f}%')

print(f'\n💡 RECOMMENDED ACTIONS:')
print(f'   ')
print(f'   The 14.8% tax shortfall with correct taxable income suggests:')
print(f'   ')
print(f'   A. High-income brackets severely underestimate tax (${high_income_tax - dotax_high_income_tax:+,.1f}M shortfall)')
print(f'   B. Possible issues:')
print(f'      - Hawaii tax brackets/rates may be incorrect or outdated')
print(f'      - Marginal tax calculation error')
print(f'      - Missing alternative minimum tax (AMT)')
print(f'      - Excessive tax credits being applied')
print(f'   ')
print(f'   Next steps:')
print(f'   1. Verify Hawaii 2022 tax brackets and rates in hawaii_calculator.py')
print(f'   2. Check if marginal tax calculation is correct')
print(f'   3. Review tax credit application (if any)')
print(f'   4. Consider if high earners use different strategies not modeled')
