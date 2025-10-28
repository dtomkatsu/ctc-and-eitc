#!/usr/bin/env python3
"""
Diagnose if deductions are being applied correctly and whether they're too high.
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
print('DEDUCTION DIAGNOSIS')
print('='*100)
print()

# Calculate actual deductions being applied
df['implied_deductions'] = df['agi'] - df['hi_taxable_income']

print('📊 DEDUCTION ANALYSIS:')
print()

# Overall statistics
total_agi = (df['agi'] * df['weight']).sum() / 1_000_000
total_taxable = (df['hi_taxable_income'] * df['weight']).sum() / 1_000_000
total_implied_deductions = (df['implied_deductions'] * df['weight']).sum() / 1_000_000

print(f'Total AGI:                    ${total_agi:,.1f}M')
print(f'Total taxable income:         ${total_taxable:,.1f}M')
print(f'Implied total deductions:     ${total_implied_deductions:,.1f}M')
print(f'Deductions as % of AGI:       {total_implied_deductions/total_agi*100:.1f}%')
print()

# Check standard deduction amounts
print('🔍 STANDARD DEDUCTION AMOUNTS:')
print()

# 2022 Hawaii standard deductions (should be):
correct_std_deductions = {
    'single': 2200,
    'married_filing_jointly': 4400,
    'married_filing_separately': 2200,
    'head_of_household': 3212,
}

print(f'{"Filing Status":<30} {"Correct 2022":>15} {"Avg Applied":>15} {"Difference":>15}')
print('-'*80)

for status in ['single', 'married_filing_jointly', 'head_of_household', 'married_filing_separately']:
    status_df = df[df['filing_status'] == status]
    if len(status_df) == 0:
        continue
    
    correct_amount = correct_std_deductions.get(status, 0)
    
    # Calculate average deduction per filer
    avg_deduction = (status_df['implied_deductions'] * status_df['weight']).sum() / status_df['weight'].sum()
    
    diff = avg_deduction - correct_amount
    
    print(f'{status.replace("_", " ").title():<30} ${correct_amount:>14,.0f} ${avg_deduction:>14,.0f} ${diff:>14,.0f}')

print()

# Check exemptions
print('🔍 EXEMPTION ANALYSIS:')
print()

# 2022 Hawaii personal exemption: $1,144 per person
personal_exemption = 1144

# Calculate implied exemptions
# For each filing status, calculate average number of people
print(f'{"Filing Status":<30} {"Avg People":>12} {"Avg Exemption":>15} {"Expected":>15}')
print('-'*80)

for status in ['single', 'married_filing_jointly', 'head_of_household', 'married_filing_separately']:
    status_df = df[df['filing_status'] == status]
    if len(status_df) == 0:
        continue
    
    # Expected number of adults
    if status == 'married_filing_jointly':
        expected_adults = 2
    else:
        expected_adults = 1
    
    # Average total people (adults + dependents)
    avg_dependents = (status_df['num_dependents'] * status_df['weight']).sum() / status_df['weight'].sum()
    avg_people = expected_adults + avg_dependents
    
    # Expected exemption
    expected_exemption = avg_people * personal_exemption
    
    # Get standard deduction for this status
    std_deduction = correct_std_deductions.get(status, 0)
    
    # Implied exemption = total deduction - standard deduction
    avg_total_deduction = (status_df['implied_deductions'] * status_df['weight']).sum() / status_df['weight'].sum()
    implied_exemption = avg_total_deduction - std_deduction
    
    print(f'{status.replace("_", " ").title():<30} {avg_people:>12.2f} ${implied_exemption:>14,.0f} ${expected_exemption:>14,.0f}')

print()

# Detailed breakdown by income level
print('📊 DEDUCTION BREAKDOWN BY INCOME LEVEL:')
print()

income_brackets = [
    (0, 25000, '$0-$25k'),
    (25000, 50000, '$25k-$50k'),
    (50000, 75000, '$50k-$75k'),
    (75000, 100000, '$75k-$100k'),
    (100000, 150000, '$100k-$150k'),
    (150000, 200000, '$150k-$200k'),
    (200000, float('inf'), '$200k+'),
]

print(f'{"Income Bracket":<20} {"Avg AGI":>12} {"Avg Taxable":>12} {"Avg Deduction":>15} {"Deduction %":>12}')
print('-'*80)

for min_inc, max_inc, label in income_brackets:
    mask = (df['agi'] >= min_inc) & (df['agi'] < max_inc)
    bracket_df = df[mask]
    
    if len(bracket_df) == 0:
        continue
    
    avg_agi = (bracket_df['agi'] * bracket_df['weight']).sum() / bracket_df['weight'].sum()
    avg_taxable = (bracket_df['hi_taxable_income'] * bracket_df['weight']).sum() / bracket_df['weight'].sum()
    avg_deduction = (bracket_df['implied_deductions'] * bracket_df['weight']).sum() / bracket_df['weight'].sum()
    deduction_pct = avg_deduction / avg_agi * 100 if avg_agi > 0 else 0
    
    print(f'{label:<20} ${avg_agi:>11,.0f} ${avg_taxable:>11,.0f} ${avg_deduction:>14,.0f} {deduction_pct:>11.1f}%')

print()

# Check if itemized deductions are being used
if 'total_deductions' in df.columns:
    print('🔍 ITEMIZED DEDUCTIONS:')
    print()
    
    itemizers = df[df['total_deductions'] > 0]
    num_itemizers = itemizers['weight'].sum()
    total_filers = df['weight'].sum()
    
    print(f'Filers who itemize:           {num_itemizers:,.0f} ({num_itemizers/total_filers*100:.1f}%)')
    
    if len(itemizers) > 0:
        avg_itemized = (itemizers['total_deductions'] * itemizers['weight']).sum() / itemizers['weight'].sum()
        print(f'Average itemized deduction:   ${avg_itemized:,.0f}')
        print()
        
        # Compare itemizers vs standard deduction users
        print(f'{"Filing Status":<30} {"% Itemize":>12} {"Avg Itemized":>15}')
        print('-'*60)
        
        for status in ['single', 'married_filing_jointly', 'head_of_household']:
            status_itemizers = itemizers[itemizers['filing_status'] == status]
            status_all = df[df['filing_status'] == status]
            
            if len(status_all) == 0:
                continue
            
            pct_itemize = status_itemizers['weight'].sum() / status_all['weight'].sum() * 100
            
            if len(status_itemizers) > 0:
                avg_itemized = (status_itemizers['total_deductions'] * status_itemizers['weight']).sum() / status_itemizers['weight'].sum()
            else:
                avg_itemized = 0
            
            print(f'{status.replace("_", " ").title():<30} {pct_itemize:>11.1f}% ${avg_itemized:>14,.0f}')

print()

# Test: What if deductions are too high?
print('💡 HYPOTHESIS: What if deductions are 10% too high?')
print()

# Recalculate tax with 10% lower deductions
from src.tax.hawaii_calculator import HawaiiTaxCalculator
calc = HawaiiTaxCalculator()

df_test = df.copy()
df_test['reduced_deductions'] = df_test['implied_deductions'] * 0.9
df_test['alt_taxable'] = df_test['agi'] - df_test['reduced_deductions']
df_test['alt_taxable'] = df_test['alt_taxable'].clip(lower=0)

def calc_tax_alt(row):
    return calc.calculate_tax(row['alt_taxable'], row['filing_status'])

df_test['alt_tax'] = df_test.apply(calc_tax_alt, axis=1)

alt_total_tax = (df_test['alt_tax'] * df_test['weight']).sum() / 1_000_000
current_total_tax = (df['hi_state_tax'] * df['weight']).sum() / 1_000_000
dotax_target = 3029

print(f'Current tax (with current deductions):  ${current_total_tax:,.1f}M')
print(f'Alt tax (with 10% lower deductions):    ${alt_total_tax:,.1f}M')
print(f'DOTax target:                            ${dotax_target:,.1f}M')
print(f'')
print(f'Improvement: ${alt_total_tax - current_total_tax:+,.1f}M ({(alt_total_tax/current_total_tax-1)*100:+.1f}%)')
print(f'Remaining gap: ${alt_total_tax - dotax_target:+,.1f}M ({(alt_total_tax/dotax_target-1)*100:+.1f}%)')

print()
print('='*100)
print('SUMMARY')
print('='*100)
print()
print(f'Total deductions applied: ${total_implied_deductions:,.1f}M ({total_implied_deductions/total_agi*100:.1f}% of AGI)')
print()
print('If deductions are too high by 10%, we would recover:')
print(f'  ${alt_total_tax - current_total_tax:,.1f}M in additional tax')
print(f'  This would close {(alt_total_tax - current_total_tax)/(dotax_target - current_total_tax)*100:.1f}% of the gap to DOTax')
