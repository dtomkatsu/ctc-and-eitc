#!/usr/bin/env python3
"""
Compare our model's effective tax rates against DOTax Table A8.
Table A8 shows effective rates based on AGI (not taxable income).
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
print('COMPARISON: Our Model vs DOTax Table A8')
print('='*100)
print()
print('Table A8 shows "Effective Tax Rates Based on TAXABLE INCOME" (Before Credits)')
print()

# DOTax Table A8 data (Before Credits, based on TAXABLE INCOME)
dotax_a8 = {
    (0, 10000): {'filers': 115285, 'tax_m': 3.0, 'eff_rate_taxable': 2.3},
    (10000, 20000): {'filers': 64160, 'tax_m': 21.0, 'eff_rate_taxable': 3.9},
    (20000, 30000): {'filers': 57835, 'tax_m': 51.0, 'eff_rate_taxable': 5.0},
    (30000, 40000): {'filers': 59827, 'tax_m': 92.0, 'eff_rate_taxable': 5.6},
    (40000, 50000): {'filers': 53555, 'tax_m': 116.0, 'eff_rate_taxable': 6.0},
    (50000, 75000): {'filers': 91459, 'tax_m': 293.0, 'eff_rate_taxable': 6.4},
    (75000, 100000): {'filers': 54976, 'tax_m': 261.0, 'eff_rate_taxable': 6.7},
    (100000, 150000): {'filers': 62065, 'tax_m': 438.0, 'eff_rate_taxable': 7.0},
    (150000, 200000): {'filers': 27976, 'tax_m': 294.0, 'eff_rate_taxable': 7.3},
    (200000, 300000): {'filers': 18937, 'tax_m': 310.0, 'eff_rate_taxable': 7.6},
    (300000, 400000): {'filers': 6076, 'tax_m': 153.0, 'eff_rate_taxable': 7.9},
    (400000, 500000): {'filers': 2926, 'tax_m': 101.0, 'eff_rate_taxable': 8.3},
    (500000, 750000): {'filers': 2991, 'tax_m': 149.0, 'eff_rate_taxable': 8.7},
    (750000, 1000000): {'filers': 1134, 'tax_m': 85.0, 'eff_rate_taxable': 9.1},
    (1000000, float('inf')): {'filers': 1824, 'tax_m': 663.0, 'eff_rate_taxable': 9.9},
}

print(f'{"AGI Bracket":<20} {"DOTax Rate":>12} {"Our Rate":>12} {"Gap":>10} {"DOTax Tax":>12} {"Our Tax":>12} {"Tax Gap":>12}')
print('-'*100)

total_dotax_tax = 0
total_our_tax = 0
total_dotax_filers = 0
total_our_filers = 0

for (min_agi, max_agi), dotax_data in dotax_a8.items():
    # Get our data for this bracket
    mask = (df['agi'] >= min_agi) & (df['agi'] < max_agi)
    bracket_df = df[mask]
    
    if len(bracket_df) == 0:
        our_filers = 0
        our_tax_m = 0
        our_rate = 0
    else:
        our_filers = bracket_df['weight'].sum()
        our_tax_m = (bracket_df['hi_state_tax'] * bracket_df['weight']).sum() / 1_000_000
        our_taxable_m = (bracket_df['hi_taxable_income'] * bracket_df['weight']).sum() / 1_000_000
        our_rate = (our_tax_m / our_taxable_m * 100) if our_taxable_m > 0 else 0
    
    dotax_rate = dotax_data['eff_rate_taxable']
    dotax_tax_m = dotax_data['tax_m']
    
    rate_gap = our_rate - dotax_rate
    tax_gap = our_tax_m - dotax_tax_m
    
    total_dotax_tax += dotax_tax_m
    total_our_tax += our_tax_m
    total_dotax_filers += dotax_data['filers']
    total_our_filers += our_filers
    
    if max_agi == float('inf'):
        label = f'${min_agi//1000:,}k+'
    else:
        label = f'${min_agi//1000:,}k-${max_agi//1000:,}k'
    
    status = '✅' if abs(rate_gap) < 0.5 else ('⚠️' if abs(rate_gap) < 1.0 else '❌')
    
    print(f'{label:<20} {dotax_rate:>11.1f}% {our_rate:>11.1f}% {rate_gap:>9.1f}pp ${dotax_tax_m:>10,.1f}M ${our_tax_m:>10,.1f}M ${tax_gap:>10,.1f}M {status}')

print('-'*100)
print(f'{"TOTAL":<20} {total_dotax_tax/total_dotax_filers*100:>11.1f}% {total_our_tax/total_our_filers*100:>11.1f}% {(total_our_tax/total_our_filers - total_dotax_tax/total_dotax_filers)*100:>9.1f}pp ${total_dotax_tax:>10,.1f}M ${total_our_tax:>10,.1f}M ${total_our_tax - total_dotax_tax:>10,.1f}M')

print()
print('='*100)
print('ANALYSIS')
print('='*100)
print()

# Count brackets within tolerance
within_05pp = sum(1 for (min_agi, max_agi), dotax_data in dotax_a8.items() 
                  if abs((df[(df['agi'] >= min_agi) & (df['agi'] < max_agi)]['hi_state_tax'] * 
                         df[(df['agi'] >= min_agi) & (df['agi'] < max_agi)]['weight']).sum() / 
                        (df[(df['agi'] >= min_agi) & (df['agi'] < max_agi)]['hi_taxable_income'] * 
                         df[(df['agi'] >= min_agi) & (df['agi'] < max_agi)]['weight']).sum() * 100 
                        - dotax_data['eff_rate_taxable']) < 0.5 
                  if len(df[(df['agi'] >= min_agi) & (df['agi'] < max_agi)]) > 0)

within_10pp = sum(1 for (min_agi, max_agi), dotax_data in dotax_a8.items() 
                  if abs((df[(df['agi'] >= min_agi) & (df['agi'] < max_agi)]['hi_state_tax'] * 
                         df[(df['agi'] >= min_agi) & (df['agi'] < max_agi)]['weight']).sum() / 
                        (df[(df['agi'] >= min_agi) & (df['agi'] < max_agi)]['hi_taxable_income'] * 
                         df[(df['agi'] >= min_agi) & (df['agi'] < max_agi)]['weight']).sum() * 100 
                        - dotax_data['eff_rate_taxable']) < 1.0 
                  if len(df[(df['agi'] >= min_agi) & (df['agi'] < max_agi)]) > 0)

print(f'Brackets within ±0.5pp: {within_05pp}/{len(dotax_a8)} ({within_05pp/len(dotax_a8)*100:.1f}%)')
print(f'Brackets within ±1.0pp: {within_10pp}/{len(dotax_a8)} ({within_10pp/len(dotax_a8)*100:.1f}%)')
print()
print(f'Total tax: ${total_our_tax:,.1f}M vs ${total_dotax_tax:,.1f}M ({(total_our_tax/total_dotax_tax-1)*100:+.1f}%)')
print()

if abs(total_our_tax / total_dotax_tax - 1) < 0.10:
    print('✅ Overall tax is within 10% of DOTax Table A8!')
    print('   The model is performing well when using AGI-based brackets.')
else:
    print('⚠️  Overall tax differs by more than 10% from DOTax Table A8.')
    print('   Further investigation needed.')
