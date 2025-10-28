#!/usr/bin/env python3
"""
Diagnose AGI adjustments to identify what needs to be increased.
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
print('AGI ADJUSTMENTS DIAGNOSIS')
print('='*80)

# Calculate current values
total_income = (df['income'] * df['weight']).sum() / 1_000_000
total_agi = (df['agi'] * df['weight']).sum() / 1_000_000
total_agi_adj = (df['agi_adjustments'] * df['weight']).sum() / 1_000_000

# DOTax benchmark (from SOI 2022 Hawaii)
# We need to work backwards from the taxable income target
dotax_base_taxable = 37522  # millions (from Table 21: 40517 - 2995)

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

# Work backwards to find required AGI (without cap gains)
# Base taxable = AGI - std_ded - exemptions
# Therefore: AGI = Base taxable + std_ded + exemptions
required_agi = dotax_base_taxable + total_std_ded + total_exemptions

print(f'\n💰 Current AGI Flow (in millions):')
print(f'   Total Income:              ${total_income:,.1f}M')
print(f'   - AGI Adjustments:         ${total_agi_adj:,.1f}M ({total_agi_adj/total_income*100:.2f}%)')
print(f'   = Current AGI (no CG):     ${total_agi:,.1f}M')

print(f'\n🎯 Required Values to Match DOTax:')
print(f'   Target base taxable:       ${dotax_base_taxable:,.1f}M')
print(f'   + Standard deductions:     ${total_std_ded:,.1f}M')
print(f'   + Personal exemptions:     ${total_exemptions:,.1f}M')
print(f'   = Required AGI (no CG):    ${required_agi:,.1f}M')

print(f'\n📊 Gap Analysis:')
current_base_taxable = total_agi - total_std_ded - total_exemptions
gap_in_agi = total_agi - required_agi
gap_in_base_taxable = current_base_taxable - dotax_base_taxable

print(f'   Current AGI (no CG):       ${total_agi:,.1f}M')
print(f'   Required AGI (no CG):      ${required_agi:,.1f}M')
print(f'   AGI excess:                ${gap_in_agi:+,.1f}M ({gap_in_agi/required_agi*100:+.2f}%)')

print(f'\n   Current base taxable:      ${current_base_taxable:,.1f}M')
print(f'   Target base taxable:       ${dotax_base_taxable:,.1f}M')
print(f'   Base taxable excess:       ${gap_in_base_taxable:+,.1f}M ({gap_in_base_taxable/dotax_base_taxable*100:+.2f}%)')

print(f'\n🔧 Required Action:')
print(f'   Need to INCREASE AGI adjustments by: ${gap_in_agi:,.1f}M')
print(f'   This is {gap_in_agi/total_income*100:.2f}% of total income')

print(f'\n   Current AGI adjustment rate: {total_agi_adj/total_income*100:.2f}%')
required_agi_adj_rate = (total_agi_adj + gap_in_agi) / total_income * 100
print(f'   Required AGI adjustment rate: {required_agi_adj_rate:.2f}%')
print(f'   Increase needed: {required_agi_adj_rate - (total_agi_adj/total_income*100):+.2f} percentage points')

# SOI comparison
print(f'\n📋 SOI 2022 Hawaii Benchmark:')
print(f'   Total Income: ~$57.07B')
print(f'   AGI: ~$56.50B')
print(f'   AGI Adjustments: ~$563M (0.99% of income)')
print(f'   Note: SOI shows 0.99%, but we need {required_agi_adj_rate:.2f}% to match taxable income')

# Breakdown by adjustment type (if available)
print(f'\n🔍 Potential Areas to Increase:')
print(f'   1. IRA contributions (currently ~0.5-2.0% of income for savers)')
print(f'   2. Self-employed health insurance (~0.35% base, 3.5x for SE)')
print(f'   3. Self-employed retirement (~0.12% base, 5.5x for SE)')
print(f'   4. Student loan interest (~0.04-0.08% depending on age)')
print(f'   5. Health Savings Account (HSA) deductions (NOT CURRENTLY MODELED)')
print(f'   6. Alimony paid (NOT CURRENTLY MODELED - pre-2019)')
print(f'   7. Moving expenses (military only, NOT CURRENTLY MODELED)')
print(f'   8. Other deductions (self-employment tax, etc.)')

print(f'\n💡 Recommendation:')
multiplier = required_agi_adj_rate / (total_agi_adj/total_income*100)
print(f'   Multiply all current AGI adjustment rates by {multiplier:.3f}')
print(f'   This will increase adjustments from ${total_agi_adj:,.1f}M to ${total_agi_adj * multiplier:,.1f}M')
print(f'   Expected result: AGI = ${total_income - (total_agi_adj * multiplier):,.1f}M (target: ${required_agi:,.1f}M)')
