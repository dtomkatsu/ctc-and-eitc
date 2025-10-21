#!/usr/bin/env python3
"""
Analyze Deduction Patterns from Hawaii SOI Data

Quick analysis to understand:
1. Itemization propensity by income
2. Average deduction amounts by bracket
3. Exemption patterns by income and family size
"""

import pandas as pd
import numpy as np
from pathlib import Path


def analyze_deductions():
    """Analyze Table A-4-2 deduction patterns."""
    print("="*80)
    print("DEDUCTION PATTERNS ANALYSIS")
    print("="*80)
    
    # Manual data entry from A4-2 (taxable returns)
    data = {
        'agi_min': [0, 10000, 20000, 30000, 40000, 50000, 75000, 100000, 150000, 200000, 300000, 400000],
        'agi_max': [10000, 20000, 30000, 40000, 50000, 75000, 100000, 150000, 200000, 300000, 400000, 999999999],
        'itemized_returns': [2588, 10690, 15222, 29552, 39198, 77709, 51126, 54341, 25446, 14723, 4459, 5972],
        'itemized_amount_m': [9, 70, 129, 212, 285, 747, 701, 1064, 663, 363, 126, 552],
        'standard_returns': [34248, 45359, 39744, 29046, 13785, 13168, 3630, 7572, 2480, 4188, 1606, 2895],
        'standard_amount_m': [77, 117, 106, 84, 46, 48, 12, 19, 7, 16, 6, 11],
    }
    
    df = pd.DataFrame(data)
    
    # Calculate metrics
    df['total_returns'] = df['itemized_returns'] + df['standard_returns']
    df['itemize_pct'] = (df['itemized_returns'] / df['total_returns'] * 100).round(1)
    df['avg_itemized'] = (df['itemized_amount_m'] * 1e6 / df['itemized_returns']).round(0)
    df['avg_standard'] = (df['standard_amount_m'] * 1e6 / df['standard_returns']).round(0)
    df['agi_midpoint'] = ((df['agi_min'] + df['agi_max']) / 2).apply(
        lambda x: min(x, 500000)  # Cap at 500k for display
    )
    
    print("\n" + "="*80)
    print("ITEMIZATION PROPENSITY BY INCOME")
    print("="*80)
    print(f"\n{'AGI Range':<25} {'Total':>10} {'Itemized':>10} {'Standard':>10} {'Itemize %':>12}")
    print("-"*80)
    
    for _, row in df.iterrows():
        if row['agi_max'] < 999999999:
            agi_range = f"${row['agi_min']//1000}k-${row['agi_max']//1000}k"
        else:
            agi_range = f"${row['agi_min']//1000}k+"
        
        print(f"{agi_range:<25} {row['total_returns']:>10,.0f} {row['itemized_returns']:>10,.0f} "
              f"{row['standard_returns']:>10,.0f} {row['itemize_pct']:>11.1f}%")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print(f"""
1. **Itemization increases with income**:
   - Under $40k: {df[df['agi_max'] <= 40000]['itemize_pct'].mean():.1f}% itemize
   - $40k-$100k: {df[(df['agi_min'] >= 40000) & (df['agi_max'] <= 100000)]['itemize_pct'].mean():.1f}% itemize
   - Over $100k: {df[df['agi_min'] >= 100000]['itemize_pct'].mean():.1f}% itemize

2. **Average deduction amounts**:
   - Avg itemized: ${df['avg_itemized'].mean():,.0f} (ranges ${df['avg_itemized'].min():,.0f} - ${df['avg_itemized'].max():,.0f})
   - Avg standard: ${df['avg_standard'].mean():,.0f} (fairly consistent)

3. **Total deductions claimed**:
   - Itemized: ${df['itemized_amount_m'].sum():.0f}M
   - Standard: ${df['standard_amount_m'].sum():.0f}M
   - Total: ${(df['itemized_amount_m'].sum() + df['standard_amount_m'].sum()):.0f}M

4. **Critical income thresholds**:
   - ~$50k: Itemization becomes more common (>74%)
   - ~$75k: Strong preference for itemizing (>93%)
   - ~$100k+: Almost universal itemization (>85%)
""")
    
    return df


def analyze_exemptions():
    """Analyze Table A-5 exemption patterns."""
    print("\n" + "="*80)
    print("PERSONAL EXEMPTION PATTERNS")
    print("="*80)
    
    # Manual data entry from A-5 (taxable returns)
    data = {
        'agi_min': [0, 10000, 20000, 30000, 40000, 50000, 75000, 100000, 150000, 200000, 300000, 400000],
        'agi_max': [10000, 20000, 30000, 40000, 50000, 75000, 100000, 150000, 200000, 300000, 400000, 999999999],
        'total_returns': [36836, 56049, 54966, 58598, 52983, 90877, 54756, 61913, 27926, 18911, 6065, 8867],
        'regular_exemptions': [28574, 77138, 85324, 96168, 89025, 161501, 114032, 151239, 76627, 52203, 15883, 22571],
        'dependent_exemptions': [4043, 19975, 22741, 27328, 25333, 44732, 34246, 49265, 26260, 17747, 5030, 6855],
        'age_exemptions': [5193, 15209, 15470, 13933, 11748, 20478, 12177, 12824, 6277, 5165, 2351, 4184],
        'exemption_amount_m': [39, 107, 117, 128, 117, 211, 146, 189, 95, 66, 21, 31],
    }
    
    df = pd.DataFrame(data)
    
    # Calculate metrics
    df['total_exemptions'] = df['regular_exemptions'] + df['dependent_exemptions'] + df['age_exemptions']
    df['avg_exemptions_per_return'] = (df['total_exemptions'] / df['total_returns']).round(1)
    df['avg_dependents_per_return'] = (df['dependent_exemptions'] / df['total_returns']).round(2)
    df['pct_with_age_exemption'] = (df['age_exemptions'] / df['total_returns'] * 100).round(1)
    df['exemption_value'] = (df['exemption_amount_m'] * 1e6 / df['total_exemptions']).round(0)
    
    print(f"\n{'AGI Range':<25} {'Returns':>10} {'Avg Exemptions':>15} {'Avg Dependents':>15} {'Age %':>10}")
    print("-"*80)
    
    for _, row in df.iterrows():
        if row['agi_max'] < 999999999:
            agi_range = f"${row['agi_min']//1000}k-${row['agi_max']//1000}k"
        else:
            agi_range = f"${row['agi_min']//1000}k+"
        
        print(f"{agi_range:<25} {row['total_returns']:>10,.0f} {row['avg_exemptions_per_return']:>15.1f} "
              f"{row['avg_dependents_per_return']:>15.2f} {row['pct_with_age_exemption']:>9.1f}%")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print(f"""
1. **Family size patterns**:
   - Low income (<$20k): {df[df['agi_max'] <= 20000]['avg_exemptions_per_return'].mean():.1f} exemptions/return
   - Middle income ($40k-$100k): {df[(df['agi_min'] >= 40000) & (df['agi_max'] <= 100000)]['avg_exemptions_per_return'].mean():.1f} exemptions/return
   - High income (>$100k): {df[df['agi_min'] >= 100000]['avg_exemptions_per_return'].mean():.1f} exemptions/return

2. **Dependent patterns**:
   - Peak dependent rate at ${df.loc[df['avg_dependents_per_return'].idxmax(), 'agi_min']//1000}k-${df.loc[df['avg_dependents_per_return'].idxmax(), 'agi_max']//1000}k AGI
   - Average: {df['avg_dependents_per_return'].mean():.2f} dependents per return

3. **Age exemptions**:
   - Low income: Higher rate ({df[df['agi_max'] <= 20000]['pct_with_age_exemption'].mean():.1f}% with age exemptions)
   - High income: Lower rate ({df[df['agi_min'] >= 100000]['pct_with_age_exemption'].mean():.1f}% with age exemptions)

4. **Exemption value**:
   - Estimated per-exemption value: ${df['exemption_value'].mean():,.0f}
   - Total exemption amount: ${df['exemption_amount_m'].sum():.0f}M
   - Total exemptions claimed: {df['total_exemptions'].sum():,.0f}
   
5. **Policy implication**:
   - Eliminating exemptions would increase taxable income by ${df['exemption_amount_m'].sum():.0f}M
   - Average tax increase per return: ${df['exemption_amount_m'].sum() * 1e6 / df['total_returns'].sum():,.0f}
""")
    
    return df


def calculate_policy_scenario():
    """Example policy scenario calculation."""
    print("\n" + "="*80)
    print("POLICY SCENARIO: Increase Standard Deduction by 20%")
    print("="*80)
    
    print("""
Current (2022):
- Single: $2,200
- Joint: $4,400

Proposed (20% increase):
- Single: $2,640 (+$440)
- Joint: $5,280 (+$880)

Expected Impact:
1. More taxpayers will choose standard deduction (reduce itemization)
2. Lower taxable income → lower tax revenue
3. Simplification benefit (fewer itemizers)

To model this:
1. Calculate current taxable income with baseline deductions
2. Recalculate with increased standard deduction
3. Apply Hawaii tax calculator to both scenarios
4. Compare total revenue

Example code:
```python
# Baseline
baseline_policy = DeductionPolicy(year=2022)
baseline_tax = calculate_total_tax(tax_units, baseline_policy)

# Scenario
scenario_policy = DeductionPolicy(year=2022)
scenario_policy.set_policy_change('standard_deduction.single', 2640)
scenario_policy.set_policy_change('standard_deduction.joint', 5280)
scenario_tax = calculate_total_tax(tax_units, scenario_policy)

revenue_loss = baseline_tax - scenario_tax
print(f'Estimated revenue loss: ${revenue_loss/1e6:.1f}M')
```
""")


def main():
    """Run all analyses."""
    deduction_df = analyze_deductions()
    exemption_df = analyze_exemptions()
    calculate_policy_scenario()
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
1. ✅ Files copied to data/raw/Selected Resident Return Data/
2. ✅ Design document created: DEDUCTIONS_MODULE_DESIGN.md
3. 📋 Review design and confirm approach
4. 🚀 Start implementing:
   - Parse deduction benchmarks (A4-2)
   - Parse exemption benchmarks (A5)
   - Create DeductionPolicy class
   - Implement TaxableIncomeCalculator
   - Integrate with Hawaii tax calculator

Ready to start building the module!
""")


if __name__ == '__main__':
    main()
