#!/usr/bin/env python3
"""
Comprehensive diagnostic to understand the taxable income gap.
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

print('='*80)
print('COMPREHENSIVE TAXABLE INCOME DIAGNOSTIC')
print('='*80)

# Calculate all the values
num_filers = df['weight'].sum()
total_income = (df['income'] * df['weight']).sum() / 1_000_000
total_agi_with_cg = (df['agi'] * df['weight']).sum() / 1_000_000
total_agi_without_cg = (df['agi_without_cap_gains'] * df['weight']).sum() / 1_000_000
total_cap_gains = (df['capital_gains'] * df['weight']).sum() / 1_000_000
total_agi_adj = (df['agi_adjustments'] * df['weight']).sum() / 1_000_000

# Deductions
correct_std_ded = {
    'single': 2200,
    'married_filing_jointly': 4400,
    'married_filing_separately': 2200,
    'head_of_household': 3212,
    'qualifying_widow': 4400
}
df['correct_std_ded'] = df['filing_status'].map(correct_std_ded)
total_std_ded = (df['correct_std_ded'] * df['weight']).sum() / 1_000_000

if 'total_deductions' in df.columns:
    total_actual_ded = (df['total_deductions'] * df['weight']).sum() / 1_000_000
else:
    total_actual_ded = total_std_ded

# Exemptions
personal_exemption = 1144
total_exemptions = ((df['num_adults'] + df['num_dependents']) * personal_exemption * df['weight']).sum() / 1_000_000

# Taxable income
total_taxable = (df['hi_taxable_income'] * df['weight']).sum() / 1_000_000

print(f'\n📊 OUR MODEL (in millions):')
print(f'   Number of filers:              {num_filers:,.0f}')
print(f'   ')
print(f'   1. PUMS Total Income:          ${total_income:,.1f}M')
print(f'   2. - AGI Adjustments:          ${total_agi_adj:,.1f}M ({total_agi_adj/total_income*100:.2f}%)')
print(f'   3. = AGI (without cap gains):  ${total_agi_without_cg:,.1f}M')
print(f'   4. + Capital Gains:            ${total_cap_gains:,.1f}M')
print(f'   5. = AGI (with cap gains):     ${total_agi_with_cg:,.1f}M')
print(f'   6. - Deductions:               ${total_actual_ded:,.1f}M')
print(f'   7. - Personal Exemptions:      ${total_exemptions:,.1f}M')
print(f'   8. = HI Taxable Income:        ${total_taxable:,.1f}M')

# DOTax benchmarks
dotax_filers = 635000  # approximate
dotax_total_income_soi = 57070  # From SOI Table (whole state)
dotax_total_agi_soi = 56500  # From SOI Table (whole state)
dotax_taxable_income = 40517  # From Table 21
dotax_cap_gains = 2995  # From Table 21
dotax_base_taxable = dotax_taxable_income - dotax_cap_gains

print(f'\n📊 DOTAX BENCHMARKS (in millions):')
print(f'   Number of filers:              {dotax_filers:,.0f} (SOI)')
print(f'   ')
print(f'   Total Income (SOI):            ${dotax_total_income_soi:,.1f}M')
print(f'   Total AGI (SOI):               ${dotax_total_agi_soi:,.1f}M')
print(f'   ')
print(f'   Capital Gains (Table 21):      ${dotax_cap_gains:,.1f}M')
print(f'   Total Taxable Income (Table 21): ${dotax_taxable_income:,.1f}M')
print(f'   Base Taxable (implied):        ${dotax_base_taxable:,.1f}M')

print(f'\n🔍 COMPARISON:')
print(f'   ')
print(f'   Filers:')
print(f'     Our model:                   {num_filers:,.0f}')
print(f'     DOTax:                       {dotax_filers:,.0f}')
print(f'     Ratio:                       {num_filers/dotax_filers:.1%}')
print(f'   ')
print(f'   Total Income:')
print(f'     Our model:                   ${total_income:,.1f}M')
print(f'     DOTax SOI:                   ${dotax_total_income_soi:,.1f}M')
print(f'     Ratio:                       {total_income/dotax_total_income_soi:.1%}')
print(f'   ')
print(f'   Total AGI:')
print(f'     Our model:                   ${total_agi_with_cg:,.1f}M')
print(f'     DOTax SOI:                   ${dotax_total_agi_soi:,.1f}M')
print(f'     Ratio:                       {total_agi_with_cg/dotax_total_agi_soi:.1%}')
print(f'   ')
print(f'   Capital Gains:')
print(f'     Our model:                   ${total_cap_gains:,.1f}M')
print(f'     DOTax Table 21:              ${dotax_cap_gains:,.1f}M')
print(f'     Difference:                  ${total_cap_gains - dotax_cap_gains:+,.1f}M ({(total_cap_gains/dotax_cap_gains-1)*100:+.1f}%)')
print(f'   ')
print(f'   Total Taxable Income:')
print(f'     Our model:                   ${total_taxable:,.1f}M')
print(f'     DOTax Table 21:              ${dotax_taxable_income:,.1f}M')
print(f'     Difference:                  ${total_taxable - dotax_taxable_income:+,.1f}M ({(total_taxable/dotax_taxable_income-1)*100:+.1f}%)')

# Calculate implied values from DOTax
# If we have the right number of filers and the right taxable income target,
# work backwards to see what AGI should be
required_agi_without_cg = dotax_base_taxable + total_actual_ded + total_exemptions

print(f'\n💡 WORKING BACKWARDS FROM TARGET:')
print(f'   ')
print(f'   Target base taxable:           ${dotax_base_taxable:,.1f}M')
print(f'   + Our deductions:              ${total_actual_ded:,.1f}M')
print(f'   + Our exemptions:              ${total_exemptions:,.1f}M')
print(f'   = Required AGI (no CG):        ${required_agi_without_cg:,.1f}M')
print(f'   ')
print(f'   Our AGI (no CG):               ${total_agi_without_cg:,.1f}M')
print(f'   Required AGI (no CG):          ${required_agi_without_cg:,.1f}M')
print(f'   Excess AGI:                    ${total_agi_without_cg - required_agi_without_cg:+,.1f}M')
print(f'   ')
print(f'   This excess AGI needs to be eliminated through:')
print(f'   Option A: Reduce PUMS income by {(total_agi_without_cg - required_agi_without_cg)/total_income*100:.1f}%')
print(f'   Option B: Increase AGI adjustments by ${(total_agi_without_cg - required_agi_without_cg):,.1f}M')
print(f'   Option C: Increase deductions by ${(total_agi_without_cg - required_agi_without_cg):,.1f}M')

# Check if the issue is that we're comparing to the wrong benchmark
print(f'\n🎯 KEY INSIGHT:')
print(f'   ')
print(f'   Our total income (${total_income:,.1f}M) is only {total_income/dotax_total_income_soi:.1%} of DOTax SOI')
print(f'   Our total AGI (${total_agi_with_cg:,.1f}M) is only {total_agi_with_cg/dotax_total_agi_soi:.1%} of DOTax SOI')
print(f'   ')
print(f'   BUT our taxable income (${total_taxable:,.1f}M) is {total_taxable/dotax_taxable_income:.1%} of Table 21')
print(f'   ')
print(f'   This suggests:')
if total_income/dotax_total_income_soi < 0.85:
    print(f'   ⚠️  We may be MISSING filers or income (only {total_income/dotax_total_income_soi:.1%} of SOI)')
    print(f'   ⚠️  But weight calibration says we have the right number of filers')
    print(f'   ⚠️  So the issue is likely that SOI includes ALL income, but Table 21 is taxable income only')
else:
    print(f'   ✅ Income levels seem reasonable relative to SOI')
    
if total_taxable > dotax_taxable_income:
    print(f'   ⚠️  Taxable income is {(total_taxable/dotax_taxable_income-1)*100:.1f}% too high')
    print(f'   ⚠️  Need to reduce by ${total_taxable - dotax_taxable_income:,.1f}M')
