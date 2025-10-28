#!/usr/bin/env python3
"""
Compare model's effective tax rates vs DOTax Tables 13A, 13B, 13C.
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
print('EFFECTIVE TAX RATE COMPARISON: MODEL VS DOTAX TABLES 13A/13B/13C')
print('='*100)

# DOTax effective tax rates by bracket (BEFORE Credits column - to match Table 12A)
dotax_rates = {
    'single': {
        # (min_taxable, max_taxable): eff_rate_before_credits
        (0, 2400): 0.0145,
        (2400, 4800): 0.0202,
        (4800, 9600): 0.0335,
        (9600, 14400): 0.0439,
        (14400, 19200): 0.0503,
        (19200, 24000): 0.0547,
        (24000, 36000): 0.0603,
        (36000, 48000): 0.0651,
        (48000, 150000): 0.0723,
        (150000, 175000): 0.0776,
        (175000, 200000): 0.0791,
        (200000, float('inf')): 0.0987,
    },
    'married_filing_jointly': {
        (0, 4800): 0.0154,
        (4800, 9600): 0.0200,
        (9600, 19200): 0.0337,
        (19200, 28800): 0.0440,
        (28800, 38400): 0.0503,
        (38400, 48000): 0.0547,
        (48000, 72000): 0.0603,
        (72000, 96000): 0.0651,
        (96000, 300000): 0.0724,
        (300000, 350000): 0.0775,
        (350000, 400000): 0.0791,
        (400000, float('inf')): 0.0909,
    },
    'head_of_household': {
        (0, 3600): 0.0140,
        (3600, 7200): 0.0200,
        (7200, 14400): 0.0337,
        (14400, 21600): 0.0441,
        (21600, 28800): 0.0503,
        (28800, 36000): 0.0547,
        (36000, 54000): 0.0600,
        (54000, 72000): 0.0650,
        (72000, 225000): 0.0714,
        (225000, 262500): 0.0776,
        (262500, 300000): 0.0797,
        (300000, float('inf')): 0.0927,
    },
    'married_filing_separately': {
        # Use same as single for now (not in tables)
        (0, 2400): 0.0145,
        (2400, 4800): 0.0202,
        (4800, 9600): 0.0335,
        (9600, 14400): 0.0439,
        (14400, 19200): 0.0503,
        (19200, 24000): 0.0547,
        (24000, 36000): 0.0603,
        (36000, 48000): 0.0651,
        (48000, 150000): 0.0723,
        (150000, 175000): 0.0776,
        (175000, 200000): 0.0791,
        (200000, float('inf')): 0.0987,
    }
}

def get_dotax_rate(taxable_income, filing_status):
    """Get DOTax effective tax rate for given taxable income and status."""
    rates = dotax_rates.get(filing_status, dotax_rates['single'])
    for (min_ti, max_ti), rate in rates.items():
        if min_ti <= taxable_income < max_ti:
            return rate
    # If above all brackets, use highest bracket rate
    return list(rates.values())[-1]

# Calculate model's effective tax rate for each tax unit
df['dotax_eff_rate'] = df.apply(
    lambda row: get_dotax_rate(row['hi_taxable_income'], row['filing_status']),
    axis=1
)

df['model_eff_rate'] = df['hi_state_tax'] / df['hi_taxable_income']
df['model_eff_rate'] = df['model_eff_rate'].replace([float('inf'), -float('inf')], 0).fillna(0)

# Calculate implied tax based on DOTax rates
df['dotax_implied_tax'] = df['hi_taxable_income'] * df['dotax_eff_rate']

# Summary comparison
print(f'\n📊 OVERALL COMPARISON:')
print(f'   ')
total_model_tax = (df['hi_state_tax'] * df['weight']).sum() / 1_000_000
total_dotax_implied_tax = (df['dotax_implied_tax'] * df['weight']).sum() / 1_000_000
total_taxable = (df['hi_taxable_income'] * df['weight']).sum() / 1_000_000

print(f'   Model total tax:               ${total_model_tax:,.1f}M')
print(f'   DOTax implied tax (13A/B/C):   ${total_dotax_implied_tax:,.1f}M')
print(f'   Difference:                    ${total_model_tax - total_dotax_implied_tax:+,.1f}M ({(total_model_tax/total_dotax_implied_tax-1)*100:+.1f}%)')
print(f'   ')
print(f'   Model effective rate:          {total_model_tax/total_taxable*100:.2f}%')
print(f'   DOTax implied effective rate:  {total_dotax_implied_tax/total_taxable*100:.2f}%')

# Detailed comparison by filing status and bracket
for filing_status in ['single', 'married_filing_jointly', 'head_of_household', 'married_filing_separately']:
    status_df = df[df['filing_status'] == filing_status]
    
    if len(status_df) == 0:
        continue
    
    print(f'\n{"="*100}')
    print(f'FILING STATUS: {filing_status.upper().replace("_", " ")}')
    print(f'{"="*100}')
    
    brackets = list(dotax_rates[filing_status].keys())
    
    print(f'\n{"Taxable Income":<25} {"Filers":>10} {"Model Tax":>12} {"DOTax Tax":>12} {"Diff":>10} {"Model %":>10} {"DOTax %":>10} {"Gap":>8}')
    print('-'*100)
    
    for min_ti, max_ti in brackets:
        mask = (status_df['hi_taxable_income'] >= min_ti) & (status_df['hi_taxable_income'] < max_ti)
        bracket_df = status_df[mask]
        
        if len(bracket_df) == 0:
            continue
        
        filers = bracket_df['weight'].sum()
        model_tax_total = (bracket_df['hi_state_tax'] * bracket_df['weight']).sum() / 1_000_000
        dotax_tax_total = (bracket_df['dotax_implied_tax'] * bracket_df['weight']).sum() / 1_000_000
        
        bracket_taxable = (bracket_df['hi_taxable_income'] * bracket_df['weight']).sum() / 1_000_000
        
        if bracket_taxable > 0:
            model_eff = model_tax_total / bracket_taxable * 100
            dotax_eff = dotax_tax_total / bracket_taxable * 100
        else:
            model_eff = 0
            dotax_eff = 0
        
        diff_pct = (model_tax_total / dotax_tax_total - 1) * 100 if dotax_tax_total != 0 else 0
        gap = model_eff - dotax_eff
        
        if max_ti == float('inf'):
            bracket_label = f'${min_ti//1000:,}k+'
        else:
            bracket_label = f'${min_ti//1000:,}k-${max_ti//1000:,}k'
        
        print(f'{bracket_label:<25} {filers:>10,.0f} ${model_tax_total:>11,.1f} ${dotax_tax_total:>11,.1f} {diff_pct:>9.1f}% {model_eff:>9.2f}% {dotax_eff:>9.2f}% {gap:>7.2f}pp')

# Identify main gaps
print(f'\n{"="*100}')
print(f'KEY FINDINGS')
print(f'{"="*100}')

# Gap by filing status
print(f'\n📊 Tax Gap by Filing Status:')
print(f'{"Filing Status":<30} {"Model Tax":>12} {"DOTax Tax":>12} {"Gap ($M)":>12} {"Gap %":>10}')
print('-'*80)
for status in ['single', 'married_filing_jointly', 'head_of_household', 'married_filing_separately']:
    status_df = df[df['filing_status'] == status]
    if len(status_df) == 0:
        continue
    
    model_tax = (status_df['hi_state_tax'] * status_df['weight']).sum() / 1_000_000
    dotax_tax = (status_df['dotax_implied_tax'] * status_df['weight']).sum() / 1_000_000
    gap = model_tax - dotax_tax
    gap_pct = (model_tax / dotax_tax - 1) * 100 if dotax_tax != 0 else 0
    
    print(f'{status.replace("_", " ").title():<30} ${model_tax:>11,.1f} ${dotax_tax:>11,.1f} ${gap:>11,.1f} {gap_pct:>9.1f}%')

# Most problematic brackets
print(f'\n🎯 Largest Tax Gaps (Top 10 Brackets):')
gap_analysis = []

for filing_status in ['single', 'married_filing_jointly', 'head_of_household', 'married_filing_separately']:
    status_df = df[df['filing_status'] == filing_status]
    if len(status_df) == 0:
        continue
    
    brackets = list(dotax_rates[filing_status].keys())
    for min_ti, max_ti in brackets:
        mask = (status_df['hi_taxable_income'] >= min_ti) & (status_df['hi_taxable_income'] < max_ti)
        bracket_df = status_df[mask]
        
        if len(bracket_df) == 0:
            continue
        
        model_tax = (bracket_df['hi_state_tax'] * bracket_df['weight']).sum() / 1_000_000
        dotax_tax = (bracket_df['dotax_implied_tax'] * bracket_df['weight']).sum() / 1_000_000
        gap = model_tax - dotax_tax
        
        if max_ti == float('inf'):
            bracket_label = f'${min_ti//1000:,}k+'
        else:
            bracket_label = f'${min_ti//1000:,}k-${max_ti//1000:,}k'
        
        gap_analysis.append({
            'filing_status': filing_status,
            'bracket': bracket_label,
            'gap': gap
        })

gap_df = pd.DataFrame(gap_analysis).sort_values('gap', key=abs, ascending=False).head(10)
print(f'{"Filing Status":<30} {"Bracket":<20} {"Gap ($M)":>12}')
print('-'*70)
for _, row in gap_df.iterrows():
    print(f'{row["filing_status"].replace("_", " ").title():<30} {row["bracket"]:<20} ${row["gap"]:>11,.1f}')

print(f'\n💡 SOLUTION:')
print(f'   Apply DOTax effective tax rates from Tables 13A/B/C instead of calculating from brackets')
print(f'   This will achieve ${total_dotax_implied_tax:,.1f}M tax (target ~$4,770M from Table 12A)')
