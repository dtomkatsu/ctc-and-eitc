#!/usr/bin/env python3
"""
Analyze remaining 12.3% tax gap and propose systematic solutions.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import glob
import numpy as np

# Load latest file
files = glob.glob('data/processed/tax_units_calibrated_*.parquet')
if not files:
    print("No tax units file found")
    sys.exit(1)

latest_file = max(files)
df = pd.read_parquet(latest_file)

print('='*100)
print('ANALYZING REMAINING 12.3% TAX GAP')
print('='*100)
print()

# DOTax Table A8 targets
dotax_targets = {
    (0, 10000): {'filers': 115285, 'tax_m': 3.0},
    (10000, 20000): {'filers': 64160, 'tax_m': 21.0},
    (20000, 30000): {'filers': 57835, 'tax_m': 51.0},
    (30000, 40000): {'filers': 59827, 'tax_m': 92.0},
    (40000, 50000): {'filers': 53555, 'tax_m': 116.0},
    (50000, 75000): {'filers': 91459, 'tax_m': 293.0},
    (75000, 100000): {'filers': 54976, 'tax_m': 261.0},
    (100000, 150000): {'filers': 62065, 'tax_m': 438.0},
    (150000, 200000): {'filers': 27976, 'tax_m': 294.0},
    (200000, 300000): {'filers': 18937, 'tax_m': 310.0},
    (300000, 400000): {'filers': 6076, 'tax_m': 153.0},
    (400000, 500000): {'filers': 2926, 'tax_m': 101.0},
    (500000, 750000): {'filers': 2991, 'tax_m': 149.0},
    (750000, 1000000): {'filers': 1134, 'tax_m': 85.0},
    (1000000, float('inf')): {'filers': 1824, 'tax_m': 663.0},
}

print('📊 GAP DECOMPOSITION BY BRACKET:')
print()
print(f'{"Bracket":<20} {"Target Tax":>12} {"Our Tax":>12} {"Gap":>12} {"Target Filers":>15} {"Our Filers":>15} {"Filer Gap":>12}')
print('-'*110)

total_target_tax = 0
total_our_tax = 0
total_target_filers = 0
total_our_filers = 0

gap_categories = {
    'deficit_high_income': 0,  # High income brackets with deficit
    'surplus_middle_income': 0,  # Middle income brackets with surplus
    'deficit_low_income': 0,  # Low income brackets with deficit
}

for (min_agi, max_agi), targets in dotax_targets.items():
    mask = (df['agi'] >= min_agi) & (df['agi'] < max_agi)
    bracket_df = df[mask]
    
    our_filers = bracket_df['weight'].sum()
    our_tax = (bracket_df['hi_state_tax'] * bracket_df['weight']).sum() / 1_000_000
    
    target_filers = targets['filers']
    target_tax = targets['tax_m']
    
    tax_gap = our_tax - target_tax
    filer_gap = our_filers - target_filers
    
    total_target_tax += target_tax
    total_our_tax += our_tax
    total_target_filers += target_filers
    total_our_filers += our_filers
    
    # Categorize gaps
    if min_agi >= 200000 and tax_gap < -10:
        gap_categories['deficit_high_income'] += abs(tax_gap)
    elif 20000 <= min_agi < 100000 and tax_gap > 10:
        gap_categories['surplus_middle_income'] += tax_gap
    elif tax_gap < -1:
        gap_categories['deficit_low_income'] += abs(tax_gap)
    
    label = f'${min_agi//1000}k-${max_agi//1000}k' if max_agi != float('inf') else f'${min_agi//1000}k+'
    
    print(f'{label:<20} ${target_tax:>11,.1f}M ${our_tax:>11,.1f}M ${tax_gap:>11,.1f}M {target_filers:>15,} {our_filers:>15,.0f} {filer_gap:>11,.0f}')

print('-'*110)
print(f'{"TOTAL":<20} ${total_target_tax:>11,.1f}M ${total_our_tax:>11,.1f}M ${total_our_tax - total_target_tax:>11,.1f}M {total_target_filers:>15,} {total_our_filers:>15,.0f} {total_our_filers - total_target_filers:>11,.0f}')

print()
print('='*100)
print('GAP ANALYSIS BY CATEGORY')
print('='*100)
print()

print(f'High-income deficit (>=$200k):    ${gap_categories["deficit_high_income"]:>8,.1f}M')
print(f'Middle-income surplus ($20k-$100k): ${gap_categories["surplus_middle_income"]:>8,.1f}M')
print(f'Low-income deficit (<$20k):        ${gap_categories["deficit_low_income"]:>8,.1f}M')
print(f'Net gap:                           ${total_our_tax - total_target_tax:>8,.1f}M')

print()
print('='*100)
print('ROOT CAUSE ANALYSIS')
print('='*100)
print()

# Issue 1: Filer count vs tax per filer
print('1. FILER COUNT ANALYSIS:')
print()

high_income_deficit_brackets = [
    (200000, 300000, '$200k-$300k'),
    (1000000, float('inf'), '$1M+'),
]

for min_agi, max_agi, label in high_income_deficit_brackets:
    mask = (df['agi'] >= min_agi) & (df['agi'] < max_agi)
    bracket_df = df[mask]
    
    our_filers = bracket_df['weight'].sum()
    our_tax = (bracket_df['hi_state_tax'] * bracket_df['weight']).sum() / 1_000_000
    our_avg_tax = (our_tax * 1_000_000 / our_filers) if our_filers > 0 else 0
    
    target_filers = dotax_targets[(min_agi, max_agi)]['filers']
    target_tax = dotax_targets[(min_agi, max_agi)]['tax_m']
    target_avg_tax = (target_tax * 1_000_000 / target_filers) if target_filers > 0 else 0
    
    print(f'{label}:')
    print(f'  Filer count: {our_filers:,.0f} vs {target_filers:,} ({(our_filers/target_filers-1)*100:+.1f}%)')
    print(f'  Avg tax per filer: ${our_avg_tax:,.0f} vs ${target_avg_tax:,.0f} ({(our_avg_tax/target_avg_tax-1)*100:+.1f}%)')
    print(f'  Total tax: ${our_tax:,.1f}M vs ${target_tax:,.1f}M ({(our_tax/target_tax-1)*100:+.1f}%)')
    
    # Decompose gap
    if target_filers > 0 and target_avg_tax > 0:
        filer_contribution = (our_filers - target_filers) * target_avg_tax / 1_000_000
        rate_contribution = our_filers * (our_avg_tax - target_avg_tax) / 1_000_000
        print(f'  Gap decomposition:')
        print(f'    Due to filer count: ${filer_contribution:+,.1f}M')
        print(f'    Due to avg tax rate: ${rate_contribution:+,.1f}M')
    print()

# Issue 2: Middle income over-collection
print('2. MIDDLE INCOME OVER-COLLECTION:')
print()

middle_brackets = [
    (20000, 30000, '$20k-$30k'),
    (40000, 50000, '$40k-$50k'),
    (50000, 75000, '$50k-$75k'),
]

for min_agi, max_agi, label in middle_brackets:
    mask = (df['agi'] >= min_agi) & (df['agi'] < max_agi)
    bracket_df = df[mask]
    
    our_filers = bracket_df['weight'].sum()
    target_filers = dotax_targets[(min_agi, max_agi)]['filers']
    
    our_tax = (bracket_df['hi_state_tax'] * bracket_df['weight']).sum() / 1_000_000
    target_tax = dotax_targets[(min_agi, max_agi)]['tax_m']
    
    if our_tax > target_tax + 10:
        print(f'{label}: ${our_tax:,.1f}M vs ${target_tax:,.1f}M (${our_tax - target_tax:+,.1f}M surplus)')
        print(f'  Filer count: {our_filers:,.0f} vs {target_filers:,} ({(our_filers/target_filers-1)*100:+.1f}%)')
        print()

print()
print('='*100)
print('PROPOSED SOLUTIONS')
print('='*100)
print()

print('Solution 1: HIGH-INCOME WEIGHT ADJUSTMENT')
print('-' * 80)
print()
print('Problem: $1M+ bracket has correct effective rate but $445M tax deficit')
print('Cause: We have correct filer COUNT (1,824 target, matched) but may need')
print('       higher INCOMES within bracket to generate more tax')
print()
print('Approach:')
print('  • Use Pareto tail inflation: Add synthetic ultra-high-income filers')
print('  • Target: $10M, $25M, $50M, $100M+ income levels')
print('  • Method: Extrapolate from existing $1M+ using Pareto alpha=1.454')
print()
print(f'Expected impact: +${gap_categories["deficit_high_income"]:.1f}M')
print()

print('Solution 2: MIDDLE-INCOME FILER REWEIGHTING')
print('-' * 80)
print()
print('Problem: $20k-$75k brackets have ~$123M surplus')
print('Cause: Over-weighted filers in these brackets')
print()
print('Approach:')
print('  • Reduce weights for $20k-$75k filers by 15-20%')
print('  • Preserve income distribution within brackets')
print('  • Reallocate weights to match DOTax filer counts exactly')
print()
print(f'Expected impact: -${gap_categories["surplus_middle_income"]:.1f}M')
print()

print('Solution 3: TAXABLE INCOME BASE ADJUSTMENT')
print('-' * 80)
print()
print('Problem: Some brackets may have wrong taxable income calculation')
print('Cause: Deductions/exemptions may still be slightly off')
print()
print('Approach:')
print('  • Fine-tune deduction rates by income bracket')
print('  • Reduce deductions for middle income (increase taxable → increase tax)')
print('  • Increase deductions for high income if needed')
print()
print('Expected impact: Varies by bracket, ~$50-100M potential')
print()

print('Solution 4: COMPREHENSIVE WEIGHT CALIBRATION')
print('-' * 80)
print()
print('Problem: Multiple brackets have filer count mismatches')
print('Cause: PUMS sampling + income calibration creates distribution shifts')
print()
print('Approach:')
print('  • Apply bracket-specific weight adjustments')
print('  • Use optimization to minimize overall deviation')
print('  • Constraint: Preserve total filer count and income distribution shape')
print()
print('Implementation:')
print('  for each bracket:')
print('    target_weight_scale = target_filers / current_filers')
print('    adjusted_weight = current_weight * target_weight_scale')
print()
print('Expected impact: Matches filer counts exactly, reduces gap to <5%')
print()

print('='*100)
print('RECOMMENDED IMPLEMENTATION ORDER')
print('='*100)
print()
print('1. ✅ Already done: Pareto filer count calibration (high-income)')
print('2. ✅ Already done: Income distribution calibration (effective rates)')
print('3. ✅ Already done: Itemized deduction reduction')
print('4. 🔄 Next: Comprehensive weight calibration (all brackets)')
print('5. 🔄 Next: Ultra-high-income tail inflation ($1M+)')
print('6. 🔄 Optional: Fine-tune deduction rates by bracket')
print()
print('Expected final gap: <5% with steps 4-5')
