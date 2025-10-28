#!/usr/bin/env python3
"""
Diagnose why base income (AGI without capital gains) is 8.8% too high.
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
print('BASE INCOME GAP DIAGNOSIS')
print('='*80)

# Current values
total_agi_without_cg = (df['agi_without_cap_gains'] * df['weight']).sum() / 1_000_000
total_income = (df['income'] * df['weight']).sum() / 1_000_000
total_agi_adj = (df['agi_adjustments'] * df['weight']).sum() / 1_000_000

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

# Calculate base taxable income
base_taxable = total_agi_without_cg - total_std_ded - total_exemptions

# DOTax benchmark
dotax_base_taxable = 37522  # millions (from Table 21: 40517 - 2995)

print(f'\n💰 Income Flow (in millions):')
print(f'   1. PUMS Total Income:          ${total_income:,.1f}M')
print(f'   2. - AGI Adjustments:          ${total_agi_adj:,.1f}M ({total_agi_adj/total_income*100:.2f}%)')
print(f'   3. = AGI (without cap gains):  ${total_agi_without_cg:,.1f}M')
print(f'   4. - Standard Deductions:      ${total_std_ded:,.1f}M')
print(f'   5. - Personal Exemptions:      ${total_exemptions:,.1f}M')
print(f'   6. = Base Taxable Income:      ${base_taxable:,.1f}M')

print(f'\n🎯 Comparison to DOTax:')
print(f'   Our base taxable:              ${base_taxable:,.1f}M')
print(f'   DOTax base taxable (implied):  ${dotax_base_taxable:,.1f}M')
print(f'   Gap:                           ${base_taxable - dotax_base_taxable:+,.1f}M ({(base_taxable/dotax_base_taxable-1)*100:+.1f}%)')

# Check if itemized deductions are being used
if 'total_deductions' in df.columns:
    total_itemized = (df['total_deductions'] * df['weight']).sum() / 1_000_000
    print(f'\n🏠 Deductions Analysis:')
    print(f'   Total deductions used:         ${total_itemized:,.1f}M')
    print(f'   Standard deductions:           ${total_std_ded:,.1f}M')
    print(f'   Excess from itemizing:         ${total_itemized - total_std_ded:,.1f}M')
    
    # Count itemizers
    num_itemizers = ((df['total_deductions'] > df['correct_std_ded']) * df['weight']).sum()
    total_filers = df['weight'].sum()
    print(f'   Itemizers:                     {num_itemizers:,.0f} ({num_itemizers/total_filers*100:.1f}%)')

# Investigate AGI adjustments
print(f'\n🔍 AGI Adjustments Detail:')
print(f'   Current total:                 ${total_agi_adj:,.1f}M ({total_agi_adj/total_income*100:.2f}% of income)')

# Load DOTax SOI data if available
try:
    soi_file = 'data/processed/soi_benchmarks_2022.csv'
    soi = pd.read_csv(soi_file)
    
    if 'total_agi_adjustments' in soi.columns:
        dotax_agi_adj = soi['total_agi_adjustments'].iloc[0] / 1_000_000
        print(f'   DOTax AGI adjustments:         ${dotax_agi_adj:,.1f}M')
        print(f'   Our adjustments vs DOTax:      {(total_agi_adj/dotax_agi_adj-1)*100:+.1f}%')
except:
    print(f'   (DOTax comparison not available)')

# Check number of filers
total_filers = df['weight'].sum()
print(f'\n👥 Filer Count:')
print(f'   Our model:                     {total_filers:,.0f} filers')
print(f'   DOTax 2022:                    ~635,000 filers (estimated)')
print(f'   Ratio:                         {total_filers/635000:.2f}x')

# Average income per filer
avg_income = total_income * 1_000_000 / total_filers
avg_agi = total_agi_without_cg * 1_000_000 / total_filers
avg_base_taxable = base_taxable * 1_000_000 / total_filers

print(f'\n📊 Per-Filer Averages:')
print(f'   Avg total income:              ${avg_income:,.0f}')
print(f'   Avg AGI (no cap gains):        ${avg_agi:,.0f}')
print(f'   Avg base taxable income:       ${avg_base_taxable:,.0f}')

dotax_avg_base_taxable = dotax_base_taxable * 1_000_000 / 635000
print(f'   DOTax avg base taxable (est):  ${dotax_avg_base_taxable:,.0f}')
print(f'   Difference:                    ${avg_base_taxable - dotax_avg_base_taxable:+,.0f} ({(avg_base_taxable/dotax_avg_base_taxable-1)*100:+.1f}%)')

# Potential issues
print(f'\n🔎 Potential Root Causes:')

# 1. Too many filers
filer_gap_pct = (total_filers/635000-1)*100
if abs(filer_gap_pct) > 5:
    print(f'\n   ⚠️  Issue 1: Filer count is {filer_gap_pct:+.1f}% off')
    print(f'      Impact: More filers → more total income')
    print(f'      Estimated contribution: {filer_gap_pct:.1f}% of the gap')

# 2. Income per filer too high
income_per_filer_gap_pct = (avg_base_taxable/dotax_avg_base_taxable-1)*100
if abs(income_per_filer_gap_pct) > 5:
    print(f'\n   ⚠️  Issue 2: Income per filer is {income_per_filer_gap_pct:+.1f}% too high')
    print(f'      Possible causes:')
    print(f'      - PUMS income values are higher than actual tax returns')
    print(f'      - Missing AGI adjustments (should be ~{(total_income - total_agi_without_cg)/total_income*100:.1f}% but could be higher)')
    print(f'      - Itemized deductions being double-counted')

# 3. AGI adjustments too low
expected_agi_adj_pct = 5.0  # Rough estimate
if (total_agi_adj / total_income * 100) < expected_agi_adj_pct:
    print(f'\n   ⚠️  Issue 3: AGI adjustments may be too low')
    print(f'      Current: {total_agi_adj/total_income*100:.2f}% of income')
    print(f'      Typical: ~{expected_agi_adj_pct}% of income')
    print(f'      Missing: ${(total_income * expected_agi_adj_pct/100) - total_agi_adj:,.1f}M')

# 4. Check if deductions are properly reducing taxable income
if 'total_deductions' in df.columns:
    # The itemized deductions should be REPLACING standard deductions, not adding to them
    # Check if this is happening correctly
    actual_deduction_reduction = total_itemized
    expected_deduction_reduction = total_std_ded  # At minimum
    
    if actual_deduction_reduction > expected_deduction_reduction * 1.5:
        print(f'\n   ⚠️  Issue 4: Deductions may be inflated')
        print(f'      Total deductions: ${actual_deduction_reduction:,.1f}M')
        print(f'      Standard deductions: ${expected_deduction_reduction:,.1f}M')
        print(f'      Excess: ${actual_deduction_reduction - expected_deduction_reduction:,.1f}M')
        print(f'      But this REDUCES taxable income, so not the cause of excess')

print(f'\n💡 Recommended Actions:')
print(f'\n   1. Verify filer count matches DOTax (currently {filer_gap_pct:+.1f}% off)')
print(f'   2. Increase AGI adjustments to reduce AGI')
print(f'   3. Review PUMS income inflation/adjustment factors')
print(f'   4. Check if itemized deductions are correctly calculated')
