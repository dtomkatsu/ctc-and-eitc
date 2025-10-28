#!/usr/bin/env python3
"""
Analyze why total taxable income is 10.7% higher than DOTax benchmark.
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
print('TAXABLE INCOME GAP ANALYSIS')
print('='*80)

# DOTax Table 21 benchmark
dotax_total_taxable = 40517  # millions
dotax_cap_gains = 2995  # millions
dotax_taxable_without_cg = dotax_total_taxable - dotax_cap_gains  # Implied

print(f'\n📊 DOTax Table 21 Benchmarks (in millions):')
print(f'   Total taxable income:        ${dotax_total_taxable:,.1f}M')
print(f'   Capital gains:               ${dotax_cap_gains:,.1f}M')
print(f'   Implied non-CG taxable:      ${dotax_taxable_without_cg:,.1f}M')
print(f'   Cap gains as % of total TI:  {dotax_cap_gains/dotax_total_taxable*100:.2f}%')

# Our model
total_taxable = (df['hi_taxable_income'] * df['weight']).sum() / 1_000_000
total_cap_gains = (df['capital_gains'] * df['weight']).sum() / 1_000_000
total_agi = (df['agi'] * df['weight']).sum() / 1_000_000
total_agi_without_cg = (df['agi_without_cap_gains'] * df['weight']).sum() / 1_000_000

# Standard deductions
correct_std_ded = {
    'single': 2200,
    'married_filing_jointly': 4400,
    'married_filing_separately': 2200,
    'head_of_household': 3212,
    'qualifying_widow': 4400
}
df['correct_std_ded'] = df['filing_status'].map(correct_std_ded)
total_std_ded = (df['correct_std_ded'] * df['weight']).sum() / 1_000_000

# Exemptions
personal_exemption = 1144
total_exemptions = ((df['num_adults'] + df['num_dependents']) * personal_exemption * df['weight']).sum() / 1_000_000

print(f'\n📊 Our Model (in millions):')
print(f'   Total taxable income:        ${total_taxable:,.1f}M')
print(f'   Capital gains:               ${total_cap_gains:,.1f}M')
print(f'   AGI without cap gains:       ${total_agi_without_cg:,.1f}M')
print(f'   Standard deductions:         ${total_std_ded:,.1f}M')
print(f'   Personal exemptions:         ${total_exemptions:,.1f}M')

# Calculate implied non-CG taxable income
implied_non_cg_taxable = total_agi_without_cg - total_std_ded - total_exemptions

print(f'\n🧮 Breakdown:')
print(f'   AGI without CG:              ${total_agi_without_cg:,.1f}M')
print(f'   - Standard deductions:       ${total_std_ded:,.1f}M')
print(f'   - Personal exemptions:       ${total_exemptions:,.1f}M')
print(f'   = Non-CG taxable income:     ${implied_non_cg_taxable:,.1f}M')
print(f'   + Capital gains:             ${total_cap_gains:,.1f}M')
print(f'   = Total taxable income:      ${implied_non_cg_taxable + total_cap_gains:,.1f}M')

print(f'\n🔍 Comparison to DOTax:')
non_cg_pct = (implied_non_cg_taxable/dotax_taxable_without_cg-1)*100
cg_pct = (total_cap_gains/dotax_cap_gains-1)*100
total_pct = (total_taxable/dotax_total_taxable-1)*100

print(f'   Our non-CG taxable:          ${implied_non_cg_taxable:,.1f}M')
print(f'   DOTax non-CG taxable:        ${dotax_taxable_without_cg:,.1f}M')
print(f'   Non-CG difference:           ${implied_non_cg_taxable - dotax_taxable_without_cg:+,.1f}M ({non_cg_pct:+.1f}%)')

print(f'\n   Our capital gains:           ${total_cap_gains:,.1f}M')
print(f'   DOTax capital gains:         ${dotax_cap_gains:,.1f}M')
print(f'   Cap gains difference:        ${total_cap_gains - dotax_cap_gains:+,.1f}M ({cg_pct:+.1f}%)')

print(f'\n   Our total taxable:           ${total_taxable:,.1f}M')
print(f'   DOTax total taxable:         ${dotax_total_taxable:,.1f}M')
print(f'   Total difference:            ${total_taxable - dotax_total_taxable:+,.1f}M ({total_pct:+.1f}%)')

# Identify the main driver
print(f'\n🎯 Main Drivers of {total_pct:.1f}% Gap:')
total_gap = total_taxable - dotax_total_taxable
non_cg_gap = implied_non_cg_taxable - dotax_taxable_without_cg
cg_gap = total_cap_gains - dotax_cap_gains

print(f'   Total gap to close:          ${total_gap:,.1f}M (100%)')
print(f'   From excess base income:     ${non_cg_gap:,.1f}M ({non_cg_gap/total_gap*100:.1f}%)')
print(f'   From excess cap gains:       ${cg_gap:,.1f}M ({cg_gap/total_gap*100:.1f}%)')

# Proposed scaling factors
print(f'\n💡 Calibration Strategy:')
print(f'\nOption 1: Scale everything proportionally')
overall_scale = dotax_total_taxable / total_taxable
print(f'   Apply scaling factor: {overall_scale:.4f} ({(1-overall_scale)*100:.1f}% reduction)')
print(f'   Result: ${total_taxable * overall_scale:,.1f}M (target: ${dotax_total_taxable:,.1f}M)')

print(f'\nOption 2: Scale capital gains only')
cg_scale = dotax_cap_gains / total_cap_gains
new_total_with_cg_scale = implied_non_cg_taxable + (total_cap_gains * cg_scale)
print(f'   Capital gains scale: {cg_scale:.4f} ({(1-cg_scale)*100:.1f}% reduction)')
print(f'   Result: ${new_total_with_cg_scale:,.1f}M')
print(f'   Remaining gap: ${new_total_with_cg_scale - dotax_total_taxable:+,.1f}M ({(new_total_with_cg_scale/dotax_total_taxable-1)*100:+.1f}%)')

print(f'\nOption 3: Scale both components separately')
base_scale = dotax_taxable_without_cg / implied_non_cg_taxable
cg_scale = dotax_cap_gains / total_cap_gains
new_total_both_scale = (implied_non_cg_taxable * base_scale) + (total_cap_gains * cg_scale)
print(f'   Base income scale: {base_scale:.4f} ({(1-base_scale)*100:.1f}% reduction)')
print(f'   Capital gains scale: {cg_scale:.4f} ({(1-cg_scale)*100:.1f}% reduction)')
print(f'   Result: ${new_total_both_scale:,.1f}M (target: ${dotax_total_taxable:,.1f}M)')
print(f'   Final gap: ${new_total_both_scale - dotax_total_taxable:+,.1f}M ({(new_total_both_scale/dotax_total_taxable-1)*100:+.1f}%)')

print(f'\n📋 Recommendation:')
print(f'   Use Option 3: Scale both components separately')
print(f'   - Reduces capital gains from ${total_cap_gains:.1f}M to ${total_cap_gains * cg_scale:.1f}M')
print(f'   - Reduces base income from ${implied_non_cg_taxable:.1f}M to ${implied_non_cg_taxable * base_scale:.1f}M')
print(f'   - Achieves total taxable income of ${dotax_total_taxable:.1f}M (±0% of target)')

# Check current AGI adjustments
total_agi_adj = (df['agi_adjustments'] * df['weight']).sum() / 1_000_000
print(f'\n💵 Current AGI Adjustments:')
print(f'   Total AGI adjustments: ${total_agi_adj:,.1f}M ({total_agi_adj/total_agi_without_cg*100:.2f}% of AGI)')

# What additional adjustment would we need?
needed_base_reduction = implied_non_cg_taxable - dotax_taxable_without_cg
print(f'\n🔧 Required Additional Adjustments:')
print(f'   Need to reduce base income by: ${needed_base_reduction:,.1f}M')
print(f'   This is {needed_base_reduction/total_agi_without_cg*100:.2f}% of AGI without cap gains')
print(f'   Could be achieved through:')
print(f'   - Increased itemized deductions')
print(f'   - Additional AGI adjustments')
print(f'   - Direct scaling factor')
