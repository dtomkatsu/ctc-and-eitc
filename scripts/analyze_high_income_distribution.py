#!/usr/bin/env python3
"""
Analyze high-income distribution and propose Pareto distribution calibration.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
import glob
from scipy import stats

# Load latest file
files = glob.glob('data/processed/tax_units_calibrated_*.parquet')
if not files:
    print("No tax units file found")
    sys.exit(1)

latest_file = max(files)
df = pd.read_parquet(latest_file)

print('='*100)
print('HIGH-INCOME DISTRIBUTION ANALYSIS')
print('='*100)
print()

# DOTax Table A8 high-income brackets
dotax_high_income = {
    (200000, 300000): {'filers': 18937, 'tax_m': 310.0},
    (300000, 400000): {'filers': 6076, 'tax_m': 153.0},
    (400000, 500000): {'filers': 2926, 'tax_m': 101.0},
    (500000, 750000): {'filers': 2991, 'tax_m': 149.0},
    (750000, 1000000): {'filers': 1134, 'tax_m': 85.0},
    (1000000, float('inf')): {'filers': 1824, 'tax_m': 663.0},
}

print('📊 CURRENT HIGH-INCOME DISTRIBUTION (AGI >= $200k):')
print()
print(f'{"Bracket":<20} {"DOTax Filers":>15} {"Our Filers":>15} {"Gap":>12} {"DOTax Tax":>12} {"Our Tax":>12}')
print('-'*100)

total_dotax_filers = 0
total_our_filers = 0
total_dotax_tax = 0
total_our_tax = 0

for (min_agi, max_agi), dotax_data in dotax_high_income.items():
    mask = (df['agi'] >= min_agi) & (df['agi'] < max_agi)
    bracket_df = df[mask]
    
    our_filers = bracket_df['weight'].sum()
    our_tax = (bracket_df['hi_state_tax'] * bracket_df['weight']).sum() / 1_000_000
    
    dotax_filers = dotax_data['filers']
    dotax_tax = dotax_data['tax_m']
    
    filer_gap = our_filers - dotax_filers
    
    if max_agi == float('inf'):
        label = f'${min_agi//1000:,}k+'
    else:
        label = f'${min_agi//1000:,}k-${max_agi//1000:,}k'
    
    print(f'{label:<20} {dotax_filers:>15,} {our_filers:>15,.0f} {filer_gap:>11,.0f} ${dotax_tax:>10,.1f}M ${our_tax:>10,.1f}M')
    
    total_dotax_filers += dotax_filers
    total_our_filers += our_filers
    total_dotax_tax += dotax_tax
    total_our_tax += our_tax

print('-'*100)
print(f'{"TOTAL":<20} {total_dotax_filers:>15,} {total_our_filers:>15,.0f} {total_our_filers - total_dotax_filers:>11,.0f} ${total_dotax_tax:>10,.1f}M ${total_our_tax:>10,.1f}M')

print()
print(f'High-income filer gap: {(total_our_filers - total_dotax_filers) / total_dotax_filers * 100:+.1f}%')
print(f'High-income tax gap: {(total_our_tax - total_dotax_tax) / total_dotax_tax * 100:+.1f}%')

# Analyze Pareto distribution
print()
print('='*100)
print('PARETO DISTRIBUTION ANALYSIS')
print('='*100)
print()

# Get high-income data
high_income_df = df[df['agi'] >= 200000].copy()

# Fit Pareto distribution to current data (weighted)
# Pareto distribution: P(X > x) = (x_min / x)^alpha
# where x_min is the minimum value and alpha is the shape parameter

# For income distribution, typical Pareto alpha is 1.5-2.5
print('Analyzing current income distribution for AGI >= $200k:')
print()

# Calculate empirical distribution
income_values = high_income_df['agi'].values
weights = high_income_df['weight'].values

# Weighted mean and percentiles
weighted_mean = np.average(income_values, weights=weights)
weighted_median = np.percentile(income_values, 50)
weighted_p90 = np.percentile(income_values, 90)
weighted_p99 = np.percentile(income_values, 99)

print(f'Current distribution (weighted):')
print(f'  Mean AGI:     ${weighted_mean:,.0f}')
print(f'  Median AGI:   ${weighted_median:,.0f}')
print(f'  90th pct:     ${weighted_p90:,.0f}')
print(f'  99th pct:     ${weighted_p99:,.0f}')
print()

# Fit Pareto distribution
# Use method of moments: alpha = mean / (mean - x_min)
x_min = 200000
alpha_estimate = weighted_mean / (weighted_mean - x_min)
print(f'Estimated Pareto parameters:')
print(f'  x_min (threshold): ${x_min:,}')
print(f'  alpha (shape):     {alpha_estimate:.3f}')
print()

# For income distributions, alpha typically ranges from 1.5 (very unequal) to 2.5 (less unequal)
# Alpha = 1.5 means top 10% earn 40% of income above threshold
# Alpha = 2.0 means top 10% earn 26% of income above threshold
# Alpha = 2.5 means top 10% earn 18% of income above threshold

print('Typical Pareto alpha values for income:')
print('  1.5: Very high inequality (top 1% much richer)')
print('  2.0: Moderate inequality')
print('  2.5: Lower inequality')
print(f'  {alpha_estimate:.2f}: Our current distribution')
print()

# Calculate what Pareto distribution would imply
print('='*100)
print('PROPOSED SOLUTION: Pareto-Based High-Income Calibration')
print('='*100)
print()

print('Strategy:')
print('  1. Keep PUMS-based filers for AGI < $200k (good coverage)')
print('  2. For AGI >= $200k, use Pareto distribution to:')
print('     a) Adjust filer counts to match DOTax totals')
print('     b) Reallocate income within high brackets to match patterns')
print('     c) Ensure filing status distribution remains realistic')
print()

# Calculate target Pareto alpha to match DOTax distribution
# We want to match the ratio of filers in different brackets

# Ratio of $1M+ to $200k-$300k filers
dotax_ratio = dotax_high_income[(1000000, float('inf'))]['filers'] / dotax_high_income[(200000, 300000)]['filers']
our_ratio = (df[df['agi'] >= 1000000]['weight'].sum() / 
             df[(df['agi'] >= 200000) & (df['agi'] < 300000)]['weight'].sum()) if len(df[(df['agi'] >= 200000) & (df['agi'] < 300000)]) > 0 else 0

print(f'Filer ratio ($1M+ / $200k-$300k):')
print(f'  DOTax target: {dotax_ratio:.4f} ({dotax_ratio:.2%})')
print(f'  Our current:  {our_ratio:.4f} ({our_ratio:.2%})')
print()

# Calculate what alpha would give us the right ratio
# For Pareto: P(X > x) = (x_min / x)^alpha
# Ratio of P(X > 1M) / P(X > 200k) = (200k / 1M)^alpha = (0.2)^alpha
# Target ratio = 1824 / 18937 = 0.0963
# So (0.2)^alpha = 0.0963
# alpha = log(0.0963) / log(0.2) = 1.46

target_alpha = np.log(dotax_ratio) / np.log(200000 / 1000000)
print(f'Target Pareto alpha to match DOTax: {target_alpha:.3f}')
print()

# Implementation approach
print('='*100)
print('IMPLEMENTATION APPROACH')
print('='*100)
print()

print('Option 1: Reweight Existing High-Income Filers')
print('  • Scale weights of existing AGI >= $200k filers')
print('  • Use Pareto-based probabilities to adjust weights')
print('  • Preserve filing status and dependent counts')
print('  • Pros: Simple, preserves real tax unit characteristics')
print('  • Cons: May not reach highest income levels')
print()

print('Option 2: Synthetic High-Income Filers')
print('  • Generate synthetic filers for AGI > $500k using Pareto')
print('  • Sample filing status from observed patterns')
print('  • Calculate taxes using actual brackets')
print('  • Pros: Can reach any income level needed')
print('  • Cons: More complex, synthetic data')
print()

print('Option 3: Hybrid Approach (RECOMMENDED)')
print('  • Use PUMS for AGI < $500k (good coverage)')
print('  • Reweight $500k-$1M filers using Pareto probabilities')
print('  • Add synthetic filers for AGI > $1M if needed')
print('  • Calibrate to match DOTax totals by bracket')
print()

# Calculate expected improvement
print('='*100)
print('EXPECTED IMPROVEMENT')
print('='*100)
print()

# If we match filer counts in high brackets, what tax would we get?
expected_tax_improvement = 0
for (min_agi, max_agi), dotax_data in dotax_high_income.items():
    mask = (df['agi'] >= min_agi) & (df['agi'] < max_agi)
    bracket_df = df[mask]
    
    if len(bracket_df) > 0:
        our_filers = bracket_df['weight'].sum()
        dotax_filers = dotax_data['filers']
        
        # If we scale weights to match filer count
        scale_factor = dotax_filers / our_filers if our_filers > 0 else 1
        
        current_tax = (bracket_df['hi_state_tax'] * bracket_df['weight']).sum() / 1_000_000
        adjusted_tax = current_tax * scale_factor
        dotax_tax = dotax_data['tax_m']
        
        tax_improvement = adjusted_tax - current_tax
        expected_tax_improvement += tax_improvement

print(f'Current high-income tax (AGI >= $200k):  ${total_our_tax:,.1f}M')
print(f'DOTax target:                            ${total_dotax_tax:,.1f}M')
print(f'Current gap:                             ${total_our_tax - total_dotax_tax:+,.1f}M ({(total_our_tax / total_dotax_tax - 1) * 100:+.1f}%)')
print()
print(f'After reweighting to match filer counts: ${total_our_tax + expected_tax_improvement:,.1f}M')
print(f'Expected improvement:                    ${expected_tax_improvement:+,.1f}M')
print(f'Remaining gap:                           ${total_our_tax + expected_tax_improvement - total_dotax_tax:+,.1f}M')
print()

if abs((total_our_tax + expected_tax_improvement) / total_dotax_tax - 1) < 0.05:
    print('✅ Reweighting high-income filers would bring us within 5% of target!')
else:
    print('⚠️  Additional adjustments beyond reweighting may be needed.')
