#!/usr/bin/env python3
"""
Diagnose why our bracket-based calculation systematically underestimates tax by ~14%
when the brackets themselves are correct.
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
print('SYSTEMATIC UNDERESTIMATION DIAGNOSIS')
print('='*100)
print()
print('Our brackets are correct for 2022, but we systematically underestimate tax by ~14%.')
print('Let\'s investigate what could cause this:')
print()

# Hypothesis 1: Are we calculating taxable income correctly?
print('='*100)
print('HYPOTHESIS 1: Taxable Income Calculation Error')
print('='*100)
print()
print('If our taxable income is HIGHER than what DOTax uses, we would apply')
print('higher bracket rates but still get lower effective rates if the base is wrong.')
print()

# Check a sample
sample = df[df['hi_taxable_income'].between(95000, 105000) & (df['filing_status'] == 'married_filing_jointly')].head(5)

print('Sample tax units with ~$100k taxable income (MFJ):')
print(f'{"AGI":>12} {"Deductions":>12} {"Exemptions":>12} {"Taxable":>12} {"Tax":>10} {"Rate":>8}')
print('-'*80)
for _, row in sample.iterrows():
    print(f'${row["agi"]:>11,.0f} ${row.get("deduction_amount", 0):>11,.0f} ${row.get("exemption_amount", 0):>11,.0f} ${row["hi_taxable_income"]:>11,.0f} ${row["hi_state_tax"]:>9,.0f} {row["hi_state_tax"]/row["hi_taxable_income"]*100:>7.2f}%')

print()
print('DOTax Table 13B says MFJ with $96k-$300k taxable should have 7.24% effective rate.')
print()

# Hypothesis 2: Are deductions/exemptions too high?
print('='*100)
print('HYPOTHESIS 2: Deductions/Exemptions Too High')
print('='*100)
print()
print('If we\'re giving too much in deductions/exemptions, taxable income would be')
print('artificially LOW, causing us to underestimate tax.')
print()

# Compare our deductions vs what would be implied
total_agi = (df['agi'] * df['weight']).sum() / 1_000_000
total_taxable = (df['hi_taxable_income'] * df['weight']).sum() / 1_000_000
total_deductions = (df.get('deduction_amount', df['agi'] * 0) * df['weight']).sum() / 1_000_000
total_exemptions = (df.get('exemption_amount', df['agi'] * 0) * df['weight']).sum() / 1_000_000

print(f'Our model:')
print(f'  Total AGI:                    ${total_agi:,.1f}M')
print(f'  Total deductions:             ${total_deductions:,.1f}M')
print(f'  Total exemptions:             ${total_exemptions:,.1f}M')
print(f'  Total taxable income:         ${total_taxable:,.1f}M')
print(f'  Implied: AGI - Ded - Exempt = ${total_agi - total_deductions - total_exemptions:,.1f}M')
print()

# Check if this matches
if abs((total_agi - total_deductions - total_exemptions) - total_taxable) > 100:
    print('⚠️  WARNING: AGI - Deductions - Exemptions ≠ Taxable Income!')
    print(f'   Difference: ${(total_agi - total_deductions - total_exemptions) - total_taxable:+,.1f}M')
    print()

# Hypothesis 3: Does DOTax include something in "tax" that we don't?
print('='*100)
print('HYPOTHESIS 3: DOTax "Tax" Includes Additional Components')
print('='*100)
print()
print('DOTax Table 13A/B/C shows "Average Effective Tax Rate Before Credits".')
print('This might include:')
print('  - Regular income tax (what we calculate)')
print('  - Use tax')
print('  - Other state taxes')
print('  - Mandatory contributions')
print()

# Calculate what multiplier would be needed
our_total_tax = (df['hi_state_tax'] * df['weight']).sum() / 1_000_000
dotax_target = 3029  # From Table 12A

multiplier_needed = dotax_target / our_total_tax

print(f'Our total tax:        ${our_total_tax:,.1f}M')
print(f'DOTax target:         ${dotax_target:,.1f}M')
print(f'Multiplier needed:    {multiplier_needed:.4f} ({(multiplier_needed-1)*100:.1f}% increase)')
print()

# Hypothesis 4: Are we missing high-income filers?
print('='*100)
print('HYPOTHESIS 4: Missing High-Income Filers')
print('='*100)
print()
print('If we\'re missing high-income filers who pay higher effective rates,')
print('our overall effective rate would be too low.')
print()

# Check filer distribution
income_brackets = [
    (0, 50000, '$0-$50k'),
    (50000, 100000, '$50k-$100k'),
    (100000, 200000, '$100k-$200k'),
    (200000, 500000, '$200k-$500k'),
    (500000, float('inf'), '$500k+'),
]

print(f'{"Income Bracket":<20} {"Filers":>12} {"Avg Tax":>12} {"Eff Rate":>10} {"Total Tax $M":>15}')
print('-'*80)

for min_inc, max_inc, label in income_brackets:
    mask = (df['hi_taxable_income'] >= min_inc) & (df['hi_taxable_income'] < max_inc)
    bracket_df = df[mask]
    
    if len(bracket_df) == 0:
        continue
    
    filers = bracket_df['weight'].sum()
    total_tax = (bracket_df['hi_state_tax'] * bracket_df['weight']).sum()
    total_taxable = (bracket_df['hi_taxable_income'] * bracket_df['weight']).sum()
    
    avg_tax = total_tax / filers if filers > 0 else 0
    eff_rate = total_tax / total_taxable * 100 if total_taxable > 0 else 0
    
    print(f'{label:<20} {filers:>12,.0f} ${avg_tax:>11,.0f} {eff_rate:>9.2f}% ${total_tax/1_000_000:>14,.1f}')

print()

# Hypothesis 5: Taxable income definition mismatch
print('='*100)
print('HYPOTHESIS 5: Taxable Income Definition Mismatch')
print('='*100)
print()
print('DOTax "taxable income" in Tables 13A/B/C might be defined differently than')
print('what we calculate. For example:')
print('  - We calculate: AGI - Deductions - Exemptions')
print('  - DOTax might use: AGI - Deductions only (exemptions handled differently)')
print('  - Or DOTax might include adjustments we\'re missing')
print()

# Test: What if we recalculate tax using AGI - Deductions only?
df_test = df.copy()
df_test['alt_taxable'] = df_test['agi'] - df_test.get('deduction_amount', 0)
df_test['alt_taxable'] = df_test['alt_taxable'].clip(lower=0)

# Recalculate tax on this alternative taxable income
from src.tax.hawaii_calculator import HawaiiTaxCalculator
calc = HawaiiTaxCalculator()

def calc_tax_alt(row):
    return calc.calculate_tax(row['alt_taxable'], row['filing_status'])

df_test['alt_tax'] = df_test.apply(calc_tax_alt, axis=1)

alt_total_tax = (df_test['alt_tax'] * df_test['weight']).sum() / 1_000_000
alt_total_taxable = (df_test['alt_taxable'] * df_test['weight']).sum() / 1_000_000

print(f'If we exclude exemptions from taxable income:')
print(f'  Alternative taxable income:   ${alt_total_taxable:,.1f}M')
print(f'  Alternative tax:              ${alt_total_tax:,.1f}M')
print(f'  Alternative effective rate:   {alt_total_tax/alt_total_taxable*100:.2f}%')
print(f'  vs DOTax target:              ${dotax_target:,.1f}M')
print(f'  Difference:                   ${alt_total_tax - dotax_target:+,.1f}M ({(alt_total_tax/dotax_target-1)*100:+.1f}%)')
print()

# Hypothesis 6: Credits are already subtracted in our calculation
print('='*100)
print('HYPOTHESIS 6: Are We Already Subtracting Credits?')
print('='*100)
print()
print('DOTax Table 13A/B/C shows rates "Before Credits" and "After Credits".')
print('If our calculation already includes some credits, we would underestimate.')
print()

# Check if we have any credit columns
credit_cols = [col for col in df.columns if 'credit' in col.lower()]
if credit_cols:
    print(f'Found credit columns: {credit_cols}')
    for col in credit_cols:
        total_credits = (df[col] * df['weight']).sum() / 1_000_000
        print(f'  {col}: ${total_credits:,.1f}M')
else:
    print('No credit columns found in data.')

print()
print('='*100)
print('SUMMARY & RECOMMENDATION')
print('='*100)
print()
print(f'We systematically underestimate tax by {(multiplier_needed-1)*100:.1f}%.')
print()
print('Most likely causes:')
print('  1. **Exemptions handling**: DOTax might not subtract exemptions from taxable income')
print(f'     (Test: Excluding exemptions gives ${alt_total_tax:,.1f}M vs ${dotax_target:,.1f}M target)')
print('  2. **Additional taxes**: DOTax "tax" might include use tax or other components')
print('  3. **Definition mismatch**: "Taxable income" defined differently in DOTax data')
print()
print('Next steps:')
print('  1. Verify how Hawaii defines "taxable income" for tax year 2022')
print('  2. Check if exemptions are subtracted before or after tax calculation')
print('  3. Look for Hawaii tax form instructions to clarify')
print('  4. Consider if DOTax Table 13 uses a different base than Table 21')
