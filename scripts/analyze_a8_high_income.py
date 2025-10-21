#!/usr/bin/env python3
"""
Analyze A8 data to reverse-engineer AGI for high-income brackets.

Table A-8 provides:
- Number of returns by AGI bracket
- Tax liability before/after credits
- Effective tax rates (before credits)

We can reverse-engineer: Taxable Income ≈ Tax / Effective Rate
Then estimate: AGI ≈ Taxable Income + Standard Deduction
"""

import pandas as pd
import numpy as np


def parse_a8_data():
    """Parse A8 table to extract high-income bracket data."""
    df = pd.read_csv('data/raw/Selected Resident Return Data/Dotax Soi 2022 - A8.csv', header=None)
    
    # Data starts at row 7
    high_income_brackets = []
    
    # Extract high-income brackets (rows 19-23: $400k+)
    for i in range(19, 24):
        bracket_min = str(df.iloc[i, 0]).strip()
        bracket_mid = str(df.iloc[i, 1]).strip()
        bracket_max = str(df.iloc[i, 2]).strip()
        
        returns = float(str(df.iloc[i, 4]).replace(',', ''))
        tax_before_str = str(df.iloc[i, 6]).replace('$', '').replace(',', '').strip()
        tax_before_millions = float(tax_before_str) if tax_before_str != 'nan' else 0
        
        # Column 14: Effective rate based on Taxable Income
        # Column 16: Effective rate based on Hawaii AGI (what we want!)
        effective_rate_taxable_str = str(df.iloc[i, 14]).replace('%', '').strip()
        effective_rate_agi_str = str(df.iloc[i, 16]).replace('%', '').strip()
        
        try:
            effective_rate_taxable = float(effective_rate_taxable_str) / 100 if effective_rate_taxable_str != 'nan' else None
            effective_rate_agi = float(effective_rate_agi_str) / 100 if effective_rate_agi_str != 'nan' else None
        except:
            effective_rate_taxable = None
            effective_rate_agi = None
        
        high_income_brackets.append({
            'bracket_label': f"{bracket_min} {bracket_mid} {bracket_max}",
            'bracket_min': bracket_min,
            'bracket_max': bracket_max,
            'returns': returns,
            'tax_before_millions': tax_before_millions,
            'effective_rate_taxable': effective_rate_taxable,
            'effective_rate_agi': effective_rate_agi
        })
    
    return high_income_brackets


def reverse_engineer_agi(brackets):
    """Reverse engineer AGI from tax liability and AGI-based effective rates."""
    print('='*80)
    print('REVERSE-ENGINEERING AGI FROM A8 DATA')
    print('='*80)
    print('\nFormula: Hawaii AGI = Tax (Before Credits) / Effective Rate (Hawaii AGI basis)')
    print('No standard deduction adjustment needed - using AGI-based rate directly!')
    print()
    
    results = []
    
    for data in brackets:
        returns = data['returns']
        tax_m = data['tax_before_millions']
        eff_rate_taxable = data['effective_rate_taxable']
        eff_rate_agi = data['effective_rate_agi']
        
        if eff_rate_agi and eff_rate_agi > 0:
            # Direct calculation: AGI = Tax / Effective Rate (AGI basis)
            est_agi_m = tax_m / eff_rate_agi
            avg_agi = (est_agi_m * 1e6) / returns if returns > 0 else 0
            
            # Also calculate taxable income for comparison
            est_taxable_income_m = tax_m / eff_rate_taxable if eff_rate_taxable else None
            
            results.append({
                **data,
                'est_taxable_income_millions': est_taxable_income_m,
                'est_agi_millions': est_agi_m,
                'avg_agi': avg_agi
            })
            
            print(f"{data['bracket_label']:<30}")
            print(f"  Returns: {returns:>15,.0f}")
            print(f"  Tax before credits: {tax_m:>10,.0f}M")
            print(f"  Effective rate (Taxable): {eff_rate_taxable*100:>8.1f}%")
            print(f"  Effective rate (AGI): {eff_rate_agi*100:>12.1f}% ✅")
            print(f"  Est Hawaii AGI: {est_agi_m:>16,.0f}M")
            print(f"  Avg AGI per return: ${avg_agi:>12,.0f}")
            print()
    
    return results


def create_enhanced_benchmarks(a8_results):
    """Create enhanced benchmarks with A8 data."""
    print('='*80)
    print('ENHANCED HIGH-INCOME BENCHMARKS')
    print('='*80)
    print()
    
    total_returns = sum(r['returns'] for r in a8_results)
    total_agi = sum(r['est_agi_millions'] for r in a8_results)
    
    print(f"Total $400k+ returns: {total_returns:,.0f}")
    print(f"Total $400k+ AGI: ${total_agi:,.0f}M (${total_agi*1e6:,.0f})")
    print()
    
    print("Breakdown by sub-bracket:")
    print(f"{'Bracket':<30} {'Returns':>12} {'AGI (M)':>15} {'% of $400k+':<15}")
    print('-'*80)
    
    for r in a8_results:
        pct_returns = (r['returns'] / total_returns * 100) if total_returns > 0 else 0
        print(f"{r['bracket_label']:<30} {r['returns']:>12,.0f} "
              f"{r['est_agi_millions']:>15,.0f} {pct_returns:>14.1f}%")
    
    print('-'*80)
    print(f"{'TOTAL':<30} {total_returns:>12,.0f} {total_agi:>15,.0f}")
    print()
    
    return a8_results


def main():
    print("\nAnalyzing A8 data for high-income brackets...\n")
    
    # Parse A8
    brackets = parse_a8_data()
    
    # Reverse engineer AGI
    results = reverse_engineer_agi(brackets)
    
    # Create enhanced benchmarks
    enhanced = create_enhanced_benchmarks(results)
    
    print('='*80)
    print('NEXT STEPS')
    print('='*80)
    print("1. Use A8 return counts to split $400k+ bracket in comprehensive parser")
    print("2. Use reverse-engineered AGI estimates for calibration targets")
    print("3. Update hybrid_tax_calibration.py to use finer brackets")
    print()


if __name__ == '__main__':
    main()
