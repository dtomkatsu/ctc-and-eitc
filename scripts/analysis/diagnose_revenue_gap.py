#!/usr/bin/env python3
"""Quick diagnostic to understand the 29.9% revenue gap."""

import pandas as pd
import numpy as np

print('REVENUE GAP DIAGNOSTIC')
print('=' * 70)

# Load the detailed results
results_2017 = pd.read_parquet('data/processed/policy_analysis/enhanced_2017_deductions_2026_brackets.parquet')
results_2026 = pd.read_parquet('data/processed/policy_analysis/enhanced_2026_deductions_2026_brackets.parquet')

print(f'Loaded {len(results_2017):,} tax units')
print()

# Check if files loaded correctly
print('Sample data check:')
print(f"2017 scenario columns: {results_2017.columns.tolist()[:5]}...")
print(f"2026 scenario columns: {results_2026.columns.tolist()[:5]}...")
print()

# Income distribution
def income_group(income):
    if income < 50000: return 'Under $50K'
    elif income < 100000: return '$50K-$100K'  
    elif income < 200000: return '$100K-$200K'
    else: return 'Over $200K'

results_2017['income_group'] = results_2017['income'].apply(income_group)
results_2026['income_group'] = results_2026['income'].apply(income_group)

print('BY INCOME GROUP:')
print('-' * 70)

comparison = pd.DataFrame({
    'count': results_2017.groupby('income_group').size(),
    'avg_income': results_2017.groupby('income_group')['income'].mean(),
    'tax_2017': results_2017.groupby('income_group')['scenario_2017_final_tax'].mean(),
    'tax_2026': results_2026.groupby('income_group')['scenario_2026_final_tax'].mean(),
})

comparison['reduction_pct'] = (1 - comparison['tax_2026'] / comparison['tax_2017']) * 100
comparison = comparison.round(0)

print(comparison.to_string())
print()

# Overall statistics
print('OVERALL STATISTICS:')
print('-' * 70)
print(f"Total revenue 2017 policy: ${results_2017['scenario_2017_final_tax'].sum():,.0f}")
print(f"Total revenue 2026 policy: ${results_2026['scenario_2026_final_tax'].sum():,.0f}")
print(f"Average effective rate 2017: {(results_2017['scenario_2017_final_tax'] / results_2017['income']).mean() * 100:.2f}%")
print(f"Average effective rate 2026: {(results_2026['scenario_2026_final_tax'] / results_2026['income']).mean() * 100:.2f}%")
print()

# Check a few specific examples
print('SAMPLE CALCULATIONS (first 5 tax units):')
print('-' * 70)

sample = pd.DataFrame({
    'income': results_2017['income'].head(5),
    'tax_2017': results_2017['scenario_2017_final_tax'].head(5),
    'tax_2026': results_2026['scenario_2026_final_tax'].head(5),
    'savings': (results_2017['scenario_2017_final_tax'] - results_2026['scenario_2026_final_tax']).head(5),
    'pct_reduction': ((1 - results_2026['scenario_2026_final_tax'].head(5) / results_2017['scenario_2017_final_tax'].head(5)) * 100)
})

print(sample.round(0).to_string(index=False))
print()

# Check for outliers
print('OUTLIER CHECK:')
print('-' * 70)

pct_reduction = (1 - results_2026['scenario_2026_final_tax'] / results_2017['scenario_2017_final_tax']) * 100
outliers = pct_reduction[(pct_reduction > 50) | (pct_reduction < 0)]

if len(outliers) > 0:
    print(f'⚠️  Found {len(outliers)} tax units with unusual reductions:')
    print(f'   Min reduction: {pct_reduction.min():.1f}%')
    print(f'   Max reduction: {pct_reduction.max():.1f}%')
    print(f'   Median reduction: {pct_reduction.median():.1f}%')
else:
    print('✓ No outliers found')
    print(f'  Reduction range: {pct_reduction.min():.1f}% to {pct_reduction.max():.1f}%')
    print(f'  Median reduction: {pct_reduction.median():.1f}%')

print()
print('CONCLUSION:')
print('-' * 70)

avg_reduction = pct_reduction.mean()
if avg_reduction > 35:
    print(f'⚠️  Average reduction of {avg_reduction:.1f}% seems HIGH')
    print('   This suggests possible calculation issue or truly massive policy change')
elif avg_reduction > 25:
    print(f'⚠️  Average reduction of {avg_reduction:.1f}% is LARGE but could be legitimate')
    print('   Given 3.6× standard deduction increase + 6× bracket expansion')
else:
    print(f'✓ Average reduction of {avg_reduction:.1f}% seems reasonable')
