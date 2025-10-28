#!/usr/bin/env python3
"""
Diagnose Middle-Income Overestimation

Investigates why the model overestimates tax revenue by +$578M in middle-income brackets.
"""

import pandas as pd
import numpy as np
from pathlib import Path

print('MIDDLE-INCOME OVERESTIMATION DIAGNOSTIC')
print('=' * 80)
print()

# Load validation data
validation_file = Path('data/processed/validation_table_12a_20251016_093454.csv')
validation = pd.read_csv(validation_file)

# Load comprehensive benchmarks
benchmarks_file = Path('data/processed/detailed_tax_liability_benchmarks.csv')
benchmarks = pd.read_csv(benchmarks_file)

# 1. INCOME DISTRIBUTION COMPARISON
print('1. INCOME DISTRIBUTION ANALYSIS')
print('-' * 80)
print('Comparing number of returns by income bracket:')
print()

for _, row in validation.iterrows():
    if row['agi_bracket'] != 'TOTAL':
        dotax_returns = row['dotax_returns']
        model_returns = row['model_returns']
        returns_diff = model_returns - dotax_returns
        pct_diff = (returns_diff / dotax_returns * 100) if dotax_returns > 0 else 0
        
        status = '✓' if abs(pct_diff) < 20 else '⚠️' if abs(pct_diff) < 40 else '❌'
        print(f'{status} {row["agi_bracket"]:12} | DOTax: {dotax_returns:8,.0f} | Model: {model_returns:8,.0f} | Diff: {returns_diff:+8,.0f} ({pct_diff:+5.1f}%)')

print()
print('KEY OBSERVATION: Check if population distribution matches or if we have too many filers in certain brackets')
print()

# 2. AVERAGE TAX PER FILER COMPARISON
print('2. AVERAGE TAX PER FILER COMPARISON')
print('-' * 80)
print('This shows if we\'re calculating too much tax per person:')
print()

for _, row in validation.iterrows():
    if row['agi_bracket'] != 'TOTAL':
        dotax_avg = row['dotax_avg_tax']
        model_avg = row['model_avg_tax']
        avg_diff = model_avg - dotax_avg
        pct_diff = (avg_diff / dotax_avg * 100) if dotax_avg > 0 else 0
        
        status = '✓' if abs(pct_diff) < 20 else '⚠️' if abs(pct_diff) < 40 else '❌'
        print(f'{status} {row["agi_bracket"]:12} | DOTax: ${dotax_avg:8,.0f} | Model: ${model_avg:8,.0f} | Diff: ${avg_diff:+8,.0f} ({pct_diff:+5.1f}%)')

print()
print('KEY OBSERVATION: Check if we\'re applying brackets correctly or missing deductions')
print()

# 3. DECOMPOSE THE OVERESTIMATION
print('3. OVERESTIMATION DECOMPOSITION')
print('-' * 80)
print('Breaking down where the +$578M comes from:')
print()

total_overestimate_from_more_filers = 0
total_overestimate_from_higher_taxes = 0

for _, row in validation.iterrows():
    if row['agi_bracket'] != 'TOTAL' and row['agi_bracket'] != '$400k+':
        dotax_returns = row['dotax_returns']
        model_returns = row['model_returns']
        dotax_avg = row['dotax_avg_tax']
        model_avg = row['model_avg_tax']
        
        dotax_revenue = row['dotax_tax_millions']
        model_revenue = row['model_tax_millions']
        total_diff = model_revenue - dotax_revenue
        
        if total_diff > 0:  # Only look at overestimates
            # Decompose into: (1) more filers, (2) higher tax per filer
            # If we had same number of filers but DOTax avg tax:
            revenue_with_dotax_avg = (model_returns * dotax_avg) / 1_000_000
            
            # Contribution from having more filers
            filer_contribution = revenue_with_dotax_avg - dotax_revenue
            
            # Contribution from higher avg tax per filer  
            tax_contribution = model_revenue - revenue_with_dotax_avg
            
            total_overestimate_from_more_filers += max(0, filer_contribution)
            total_overestimate_from_higher_taxes += max(0, tax_contribution)
            
            if total_diff > 20:  # Only show significant brackets
                print(f'{row["agi_bracket"]:12} | Total: ${total_diff:+6.0f}M')
                print(f'             | - From more filers: ${filer_contribution:+6.0f}M')
                print(f'             | - From higher taxes: ${tax_contribution:+6.0f}M')
                print()

print(f'TOTAL OVERESTIMATION FROM MORE FILERS: ${total_overestimate_from_more_filers:.0f}M')
print(f'TOTAL OVERESTIMATION FROM HIGHER TAXES: ${total_overestimate_from_higher_taxes:.0f}M')
print()

# 4. EFFECTIVE TAX RATE ANALYSIS
print('4. EFFECTIVE TAX RATE COMPARISON')
print('-' * 80)
print('Checking if our tax calculation is too aggressive:')
print()

# Calculate effective rates from benchmarks
for _, row in validation.iterrows():
    if row['agi_bracket'] != 'TOTAL':
        # Calculate effective rate for DOTax
        dotax_agi = (row['min_agi'] + row['max_agi']) / 2 if row['max_agi'] != float('inf') else row['min_agi'] * 2
        dotax_effective_rate = (row['dotax_avg_tax'] / dotax_agi * 100) if dotax_agi > 0 else 0
        
        # Calculate effective rate for Model
        model_effective_rate = (row['model_avg_tax'] / dotax_agi * 100) if dotax_agi > 0 else 0
        
        rate_diff = model_effective_rate - dotax_effective_rate
        
        if row['min_agi'] >= 20000 and row['min_agi'] < 200000:  # Focus on middle income
            status = '✓' if abs(rate_diff) < 1 else '⚠️' if abs(rate_diff) < 2 else '❌'
            print(f'{status} {row["agi_bracket"]:12} | DOTax: {dotax_effective_rate:5.2f}% | Model: {model_effective_rate:5.2f}% | Diff: {rate_diff:+5.2f}pp')

print()

# 5. DEDUCTION/ADJUSTMENT IMPACT CHECK
print('5. DEDUCTION & ADJUSTMENT IMPACT')
print('-' * 80)
print('Checking if our -4.4% adjustment protocol is sufficient:')
print()

# Load the benchmark data to check deduction patterns
benchmark_df = pd.read_csv('data/processed/comprehensive_tax_benchmarks_enhanced.csv')

# Calculate average deduction rates
total_agi = benchmark_df['total_agi_millions'].sum()
total_tax_before = benchmark_df['tax_before_credits_millions'].sum()
total_tax_after = benchmark_df['tax_after_credits_millions'].sum()

if total_tax_before > 0:
    credit_reduction_rate = ((total_tax_before - total_tax_after) / total_tax_before) * 100
    print(f'Benchmark data shows:')
    print(f'  Total AGI: ${total_agi:.0f}M')
    print(f'  Tax before credits: ${total_tax_before:.0f}M')
    print(f'  Tax after credits: ${total_tax_after:.0f}M')
    print(f'  Credit reduction rate: {credit_reduction_rate:.1f}%')
    print()
    print(f'Our model applies:')
    print(f'  Itemized deductions: -1.2%')
    print(f'  AGI adjustments: -1.1%')
    print(f'  Hawaii credits: -2.1%')
    print(f'  TOTAL: -4.4%')
    print()
    if credit_reduction_rate > 10:
        print(f'⚠️  WARNING: Benchmark shows {credit_reduction_rate:.1f}% reduction, we only apply 4.4%!')
        print(f'   We may need to increase our adjustment factors significantly.')
    else:
        print(f'✓ Our adjustment factors seem reasonable')

print()

# 6. SUMMARY OF FINDINGS
print('6. SUMMARY & RECOMMENDATIONS')
print('=' * 80)
print()
print('Based on the analysis above, the likely causes are:')
print()
print('A. POPULATION/FILER COUNT ISSUES:')
print('   - Check if scaling factor is too high')
print('   - Verify PUMS sample represents actual tax filers')
print('   - Some PUMS individuals may not file taxes')
print()
print('B. TAX CALCULATION ISSUES:')
print('   - Average tax per filer is consistently higher than DOTax')
print('   - May be applying 2017 brackets incorrectly')
print('   - Standard deductions may not be applied properly')
print()
print('C. INSUFFICIENT ADJUSTMENTS:')
print('   - Our -4.4% total adjustments may be too low')
print('   - Need to verify against actual benchmark reduction rates')
print('   - May need more aggressive deduction modeling')
print()
print('NEXT STEPS:')
print('1. Compare PUMS income distribution vs actual DOTax AGI distribution')
print('2. Validate 2017 tax bracket application with sample calculations')
print('3. Review and potentially increase adjustment factors')
print('4. Check population scaling methodology')
