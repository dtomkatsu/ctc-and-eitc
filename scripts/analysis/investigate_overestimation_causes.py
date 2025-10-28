#!/usr/bin/env python3
"""
Deep Dive into Overestimation Causes

Two main issues identified:
1. More filers in certain brackets: +$253M
2. Higher average tax per filer: +$333M

This script investigates both.
"""

import pandas as pd
import numpy as np
from pathlib import Path

print('DEEP DIVE: ROOT CAUSES OF OVERESTIMATION')
print('=' * 80)
print()

# Load validation data
validation = pd.read_csv('data/processed/validation_table_12a_20251016_093454.csv')

# ISSUE 1: WHY DO WE HAVE MORE FILERS?
print('ISSUE 1: WHY DO WE HAVE MORE FILERS IN $50K-$150K RANGE?')
print('=' * 80)
print()

print('Population distribution comparison:')
print('-' * 80)

total_dotax = validation[validation['agi_bracket'] == 'TOTAL']['dotax_returns'].values[0]
total_model = validation[validation['agi_bracket'] == 'TOTAL']['model_returns'].values[0]

print(f'Total filers - DOTax: {total_dotax:,.0f}')
print(f'Total filers - Model: {total_model:,.0f}')
print(f'Difference: {total_model - total_dotax:,.0f} ({(total_model/total_dotax - 1)*100:+.2f}%)')
print()

print('Distribution by bracket:')
for _, row in validation.iterrows():
    if row['agi_bracket'] != 'TOTAL':
        dotax_pct = (row['dotax_returns'] / total_dotax) * 100
        model_pct = (row['model_returns'] / total_model) * 100
        pct_point_diff = model_pct - dotax_pct
        
        if abs(pct_point_diff) > 1:  # Show significant differences
            status = '⚠️' if abs(pct_point_diff) > 2 else '→'
            bracket = row['agi_bracket']
            print(f'{status} {bracket:12} | DOTax: {dotax_pct:5.1f}% | Model: {model_pct:5.1f}% | Diff: {pct_point_diff:+5.2f}pp')

print()
print('HYPOTHESIS 1A: Income Distribution Shift')
print('  - PUMS income levels may be systematically higher than actual AGI')
print('  - People earning $45k in PUMS might report $40k AGI on taxes')
print('  - This would shift distribution to higher brackets')
print()

print('HYPOTHESIS 1B: Population Scaling Error')
print('  - We scale PUMS sample by ~20x factor')
print('  - If scaling is uneven across income levels, creates distribution issues')
print('  - PUMS weights may not match actual tax filer distribution')
print()

print('HYPOTHESIS 1C: Non-Filer Inclusion')
print('  - PUMS includes ALL individuals with income')
print('  - Some PUMS individuals may not actually file taxes (students, retirees, etc.)')
print('  - We may be overrepresenting certain demographics')
print()

# ISSUE 2: WHY HIGHER TAX PER FILER?
print()
print('ISSUE 2: WHY IS AVERAGE TAX 14-17% HIGHER PER FILER?')
print('=' * 80)
print()

print('Average tax comparison (middle-income brackets):')
print('-' * 80)

for _, row in validation.iterrows():
    bracket = row['agi_bracket']
    if row['min_agi'] >= 20000 and row['max_agi'] < 200000:
        dotax_avg = row['dotax_avg_tax']
        model_avg = row['model_avg_tax']
        diff_dollars = model_avg - dotax_avg
        diff_pct = (diff_dollars / dotax_avg) * 100
        
        print(f'{bracket:12} | DOTax: ${dotax_avg:7,.0f} | Model: ${model_avg:7,.0f} | +${diff_dollars:5,.0f} (+{diff_pct:4.1f}%)')

print()
print('HYPOTHESIS 2A: Insufficient Deductions')
print('  - Our model applies: itemized (-1.2%) + AGI adj (-1.1%) = -2.3%')
print('  - Plus Hawaii credits (-2.1%) = -4.4% total')
print('  - Real taxpayers may take more aggressive deductions')
print('  - Standard deductions in 2017 were MUCH lower than 2024+')
print()

print('HYPOTHESIS 2B: Bracket Application Error')  
print('  - 2017 had lower standard deductions')
print('  - 2017 Joint: $4,400 vs 2024: $8,800')
print('  - Are we applying 2017 standard deductions correctly?')
print('  - Check if we\'re using 2022 deductions instead of 2017')
print()

print('HYPOTHESIS 2C: Filing Status Mix')
print('  - Different filing statuses have different tax rates')
print('  - If we have wrong mix of Single vs Joint vs HoH, affects avg tax')
print('  - Joint filers have wider brackets = lower tax on same income')
print()

# Let's check the 2017 standard deductions
print('VERIFICATION: 2017 Standard Deductions')
print('-' * 80)

deductions_file = Path('data/raw/hawaii_standard_deductions_by_year.csv')
if deductions_file.exists():
    deductions = pd.read_csv(deductions_file)
    print(deductions[deductions['Year'] == 2017].to_string(index=False))
    print()
    print('✓ 2017 deductions confirmed in data file')
else:
    print('⚠️  Deductions file not found')

print()

# Calculate what effective tax rate SHOULD be with 2017 policy
print('THEORETICAL CHECK: Expected Tax with 2017 Policy')
print('-' * 80)
print()

print('Example: $75k income, Joint filer')
print('  2017 standard deduction: $4,400')
print('  Taxable income: $70,600')
print()
print('  2017 Hawaii brackets (Joint):')
print('  $0-$4,800 @ 1.4%: $67')
print('  $4,800-$9,600 @ 3.2%: $154')
print('  $9,600-$19,200 @ 5.5%: $528')  
print('  $19,200-$28,800 @ 6.4%: $614')
print('  $28,800-$38,400 @ 6.8%: $653')
print('  $38,400-$48,000 @ 7.2%: $691')
print('  $48,000-$72,000 @ 7.6%: $1,718')
print('  Total estimated: ~$4,425')
print('  Effective rate: ~5.9%')
print()
print('  From our model avg for $75k-$100k: ~$5,551 (6.3% effective)')
print('  From DOTax benchmark: ~$4,751 (5.4% effective)')
print()
print('  → Our model is 16.8% higher than DOTax benchmark')
print()

print('=' * 80)
print('NEXT INVESTIGATION STEPS:')
print('=' * 80)
print('1. Load actual PUMS data and compare income vs AGI distributions')
print('2. Manually calculate tax for sample cases with 2017 brackets')
print('3. Check if deductions/exemptions are being applied correctly')
print('4. Verify population scaling factors and weights')
print('5. Review filing status distribution in model vs DOTax')
