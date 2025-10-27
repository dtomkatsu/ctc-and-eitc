#!/usr/bin/env python3
"""
Investigate why Scenario A (2017 brackets/deductions) produces such high revenue.

The issue: Applying 2017 brackets (designed for 2017 incomes) to 2026 incomes
creates artificial bracket creep that exaggerates the revenue difference.
"""

import pandas as pd
import numpy as np

print('INVESTIGATING SCENARIO A HIGH REVENUE')
print('=' * 70)

print('\nPROBLEM ANALYSIS:')
print('-' * 70)
print('We are applying 2017 brackets to 2026 income levels')
print('2017 brackets were designed for 2017-era incomes')
print('2026 incomes are ~10-15% higher due to inflation and economic growth')
print('This pushes everyone into higher brackets artificially')

print('\nBRACKET WIDTH COMPARISON (Joint Filers):')
print('-' * 70)
print('2017 Brackets (designed for 2017 incomes):')
print('  $0-$4,800: 1.4%       (width: $4,800)')
print('  $4,800-$9,600: 3.2%   (width: $4,800)')
print('  $9,600-$19,200: 5.5%  (width: $9,600)')
print('  $19,200-$28,800: 6.4% (width: $9,600)')

print('\n2026 Brackets (designed for 2026 incomes):')
print('  $0-$28,800: 1.4%      (width: $28,800 - 6x wider!)')
print('  $28,800-$38,400: 3.2% (width: $9,600)')
print('  $38,400-$48,000: 5.5% (width: $9,600)')
print('  $48,000-$72,000: 6.4% (width: $24,000)')

print('\nSTANDARD DEDUCTION COMPARISON:')
print('-' * 70)
print('2017: Joint $4,400, HoH $3,212, Single $2,200')
print('2026: Joint $16,000, HoH $12,000, Single $8,000')
print('Increase: 3.6x for joint, 3.7x for HoH, 3.6x for single')

print('\nEXAMPLE: $100,000 JOINT FILER IN 2026')
print('-' * 70)

income = 100000

# 2017 policy
std_ded_2017 = 4400
taxable_2017 = income - std_ded_2017
print(f'\nWith 2017 Policy (on 2026 income):')
print(f'  Gross income: ${income:,}')
print(f'  Standard deduction: ${std_ded_2017:,}')
print(f'  Taxable income: ${taxable_2017:,}')

# Approximate tax calculation for 2017
# Using 2017 Joint brackets up to $96,000
tax_2017 = 0
tax_2017 += 4800 * 0.014  # First bracket
tax_2017 += 4800 * 0.032  # Second bracket
tax_2017 += 9600 * 0.055  # Third bracket
tax_2017 += 9600 * 0.064  # Fourth bracket
tax_2017 += 9600 * 0.068  # Fifth bracket
tax_2017 += 9600 * 0.072  # Sixth bracket
tax_2017 += 24000 * 0.076  # Seventh bracket
tax_2017 += 24000 * 0.079  # Eighth bracket
# Remainder taxed at 8.25%
tax_2017 += (taxable_2017 - 96000) * 0.0825

print(f'  Approximate tax: ${tax_2017:,.0f}')
print(f'  Effective rate: {tax_2017/income*100:.2f}%')

# 2026 policy
std_ded_2026 = 16000
taxable_2026 = income - std_ded_2026
print(f'\nWith 2026 Policy (on 2026 income):')
print(f'  Gross income: ${income:,}')
print(f'  Standard deduction: ${std_ded_2026:,}')
print(f'  Taxable income: ${taxable_2026:,}')

# Approximate tax calculation for 2026
tax_2026 = 0
tax_2026 += 28800 * 0.014  # First bracket
tax_2026 += 9600 * 0.032   # Second bracket
tax_2026 += 9600 * 0.055   # Third bracket
tax_2026 += 24000 * 0.064  # Fourth bracket
tax_2026 += (taxable_2026 - 72000) * 0.068  # Fifth bracket

print(f'  Approximate tax: ${tax_2026:,.0f}')
print(f'  Effective rate: {tax_2026/income*100:.2f}%')

print(f'\nDIFFERENCE:')
print(f'  Tax difference: ${tax_2017 - tax_2026:,.0f}')
print(f'  Percentage higher with 2017 policy: {(tax_2017/tax_2026 - 1)*100:.1f}%')

print('\n\nINFLATION/GROWTH IMPACT:')
print('-' * 70)
# Estimated growth from 2017 to 2026
years = 2026 - 2017
annual_growth = 0.036  # 3.6% from ensemble model
total_growth = (1 + annual_growth) ** years
print(f'Years between 2017 and 2026: {years}')
print(f'Estimated annual income growth: {annual_growth*100:.1f}%')
print(f'Total income growth 2017->2026: {(total_growth-1)*100:.1f}%')

# What would 2017 income be?
income_2017_level = income / total_growth
print(f'\n2026 income of ${income:,} would be ${income_2017_level:,.0f} in 2017')
print(f'That is ${income - income_2017_level:,.0f} lower')

print('\n\nSOLUTIONS TO FIX THE COMPARISON:')
print('=' * 70)

print('\n✅ OPTION 1: Deflate 2026 incomes to 2017 levels')
print('-' * 70)
print('Approach: Divide all 2026 incomes by growth factor before applying 2017 brackets')
print('Formula: income_2017_level = income_2026 / 1.36')
print('Pros: Fair comparison - same real income under both policies')
print('Cons: More complex to implement')

print('\n✅ OPTION 2: Inflate 2017 brackets to 2026 levels')
print('-' * 70)
print('Approach: Multiply all 2017 bracket thresholds by growth factor')
print('Formula: threshold_2026 = threshold_2017 × 1.36')
print('Pros: Easier to implement, preserves 2026 income projections')
print('Cons: Creates hypothetical "adjusted 2017" brackets')

print('\nExample adjusted 2017 brackets for joint filers:')
inflation_factor = total_growth
brackets_2017 = [
    (0, 4800, 1.4),
    (4800, 9600, 3.2),
    (9600, 19200, 5.5),
    (19200, 28800, 6.4),
    (28800, 38400, 6.8),
    (38400, 48000, 7.2),
    (48000, 72000, 7.6),
    (72000, 96000, 7.9),
    (96000, 300000, 8.25),
]

print(f'  Using inflation factor: {inflation_factor:.3f}')
for min_inc, max_inc, rate in brackets_2017[:5]:
    adj_min = min_inc * inflation_factor
    adj_max = max_inc * inflation_factor
    print(f'  ${adj_min:,.0f}-${adj_max:,.0f}: {rate}%')

print('\n⚠️  OPTION 3: Accept current methodology')
print('-' * 70)
print('Current approach combines:')
print('  1. Pure policy impact (brackets + deductions)')
print('  2. Bracket creep from inflation (unindexed 2017 brackets)')
print('Interpretation: Shows total cost of NOT updating tax code for inflation')
print('Limitation: Not a clean policy-only comparison')

print('\n\nRECOMMENDATION:')
print('=' * 70)
print('Use OPTION 2: Inflate 2017 brackets to 2026 purchasing power')
print('This provides a fair comparison by removing inflation effects')
print('Shows pure policy impact of wider brackets + higher standard deductions')

print('\nExpected result with inflation-adjusted 2017 brackets:')
print('  Revenue difference should drop from ~30% to ~10-15%')
print('  Representing true policy change impact, not bracket creep')
