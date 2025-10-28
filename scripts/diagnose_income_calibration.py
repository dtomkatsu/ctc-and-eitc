#!/usr/bin/env python3
"""
Diagnose if PUMS income needs calibration to match actual tax returns.
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
print('INCOME CALIBRATION DIAGNOSIS')
print('='*80)

# Our data
total_income = (df['income'] * df['weight']).sum() / 1_000_000
total_agi_without_cg = (df['agi_without_cap_gains'] * df['weight']).sum() / 1_000_000
num_filers = df['weight'].sum()

avg_income = total_income * 1_000_000 / num_filers
avg_agi = total_agi_without_cg * 1_000_000 / num_filers

print(f'\n📊 Our Model (from PUMS):')
print(f'   Total filers:              {num_filers:,.0f}')
print(f'   Total income:              ${total_income:,.1f}M')
print(f'   Avg income per filer:      ${avg_income:,.0f}')
print(f'   Total AGI (no CG):         ${total_agi_without_cg:,.1f}M')
print(f'   Avg AGI per filer:         ${avg_agi:,.0f}')

# DOTax SOI 2022 benchmarks (approximations)
dotax_filers = 635000  # approximate
dotax_total_income_approx = 57070  # millions (from SOI)
dotax_total_agi_approx = 56500  # millions (from SOI)

# But we need to match taxable income, so work backwards
dotax_base_taxable = 37522  # millions

# Standard deductions and exemptions
correct_std_ded = {
    'single': 2200,
    'married_filing_jointly': 4400,
    'married_filing_separately': 2200,
    'head_of_household': 3212,
    'qualifying_widow': 4400
}
df['correct_std_ded'] = df['filing_status'].map(correct_std_ded)
total_std_ded = (df['correct_std_ded'] * df['weight']).sum() / 1_000_000

personal_exemption = 1144
total_exemptions = ((df['num_adults'] + df['num_dependents']) * personal_exemption * df['weight']).sum() / 1_000_000

required_agi = dotax_base_taxable + total_std_ded + total_exemptions
required_avg_agi = required_agi * 1_000_000 / num_filers

print(f'\n📊 DOTax Benchmarks:')
print(f'   Total filers:              {dotax_filers:,.0f} (approx)')
print(f'   Total income:              ${dotax_total_income_approx:,.1f}M (SOI, whole state)')
print(f'   Total AGI:                 ${dotax_total_agi_approx:,.1f}M (SOI, whole state)')
print(f'   Required AGI (from TI):    ${required_agi:,.1f}M (calculated from taxable income)')
print(f'   Avg AGI per filer needed:  ${required_avg_agi:,.0f}')

print(f'\n🔍 Income Comparison:')
print(f'   Our avg income:            ${avg_income:,.0f}')
print(f'   Our avg AGI:               ${avg_agi:,.0f}')
print(f'   Required avg AGI:          ${required_avg_agi:,.0f}')
print(f'   Gap in avg AGI:            ${avg_agi - required_avg_agi:+,.0f} ({(avg_agi/required_avg_agi-1)*100:+.1f}%)')

# Calculate required income calibration
income_scale = required_avg_agi / avg_agi
print(f'\n💡 Income Calibration Factor Needed:')
print(f'   Current AGI per filer:     ${avg_agi:,.0f}')
print(f'   Target AGI per filer:      ${required_avg_agi:,.0f}')
print(f'   Required scaling:          {income_scale:.4f} ({(1-income_scale)*100:.1f}% reduction)')

print(f'\n🎯 If we apply {income_scale:.4f} scaling to income:')
scaled_income = total_income * income_scale
scaled_agi = total_agi_without_cg * income_scale
print(f'   Scaled total income:       ${scaled_income:,.1f}M')
print(f'   Scaled total AGI:          ${scaled_agi:,.1f}M')
print(f'   Target AGI:                ${required_agi:,.1f}M')
print(f'   Remaining gap:             ${scaled_agi - required_agi:+,.1f}M')

# Alternative: Check if this is a PUMS vs actual tax return issue
print(f'\n📋 Root Cause Analysis:')
print(f'   PUMS data represents survey responses, not actual tax returns')
print(f'   People tend to:')
print(f'   - Over-report income in surveys')
print(f'   - Include non-taxable income')
print(f'   - Not account for all deductions')
print(f'   ')
print(f'   Recommended approach:')
print(f'   1. Apply income calibration factor of {income_scale:.4f} to PUMS income')
print(f'   2. This brings AGI from ${avg_agi:,.0f} to ${avg_agi * income_scale:,.0f} per filer')
print(f'   3. Results in total AGI of ${scaled_agi:,.1f}M (target: ${required_agi:,.1f}M)')
