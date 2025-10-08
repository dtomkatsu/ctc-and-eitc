"""
Calculate Hawaii State Taxes with AGI Adjustments and Credits

This script implements the full tax calculation pipeline:
1. Adjust PUMS income to AGI
2. Calculate tax liability
3. Apply Hawaii state tax credits
4. Compare to actual revenue
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tax.brackets import load_tax_data
from src.tax.adjustments import estimate_agi_from_total_income, calculate_hawaii_credits


def load_tax_units():
    """Load tax units"""
    possible_files = [
        Path(__file__).parent.parent / 'data' / 'processed' / 'tax_units_rule_based.parquet',
        Path(__file__).parent.parent / 'data' / 'processed' / 'hawaii_ctc_full_population_20250722_161935.parquet',
    ]
    
    for file_path in possible_files:
        if file_path.exists():
            return pd.read_parquet(file_path)
    
    raise FileNotFoundError("No tax units file found")


def calculate_with_full_adjustments(tax_units, calculator, year, weight_col='PWGTP'):
    """Calculate taxes with AGI adjustments and credits"""
    
    print(f"\nCalculating taxes for {year} with full adjustments...")
    print("="*80)
    
    results = []
    
    for idx, row in tax_units.iterrows():
        if idx % 5000 == 0:
            print(f"  Processing {idx:,} / {len(tax_units):,}...")
        
        # Step 1: Adjust income to AGI
        total_income = row['income']
        age = row.get('age', None)  # If available
        filing_status = row['filing_status']
        
        # Estimate AGI
        agi = estimate_agi_from_total_income(
            total_income,
            age=age,
            filing_status=filing_status,
            is_self_employed=(total_income > 50000 and np.random.random() < 0.10)
        )
        
        # Step 2: Calculate tax on AGI
        tax_info = calculator.calculate_tax(agi, year, filing_status, use_taxable_income=False)
        tax_before_credits = tax_info['tax_liability']
        
        # Step 3: Apply Hawaii state tax credits
        num_dependents = row.get('num_dependents', 0)
        credits = calculate_hawaii_credits(
            agi=agi,
            filing_status=filing_status,
            num_dependents=num_dependents,
            tax_before_credits=tax_before_credits,
            year=year
        )
        
        # Step 4: Calculate net tax
        net_tax = max(0, tax_before_credits - credits['total_nonrefundable'])
        # Refundable credits can create negative tax (refund)
        net_tax_with_refundable = net_tax - credits['total_refundable']
        
        results.append({
            'total_income': total_income,
            'agi': agi,
            'agi_adjustment': total_income - agi,
            'taxable_income': tax_info['taxable_income'],
            'tax_before_credits': tax_before_credits,
            'credits_nonrefundable': credits['total_nonrefundable'],
            'credits_refundable': credits['total_refundable'],
            'credits_total': credits['total'],
            'net_tax': max(0, net_tax_with_refundable),  # Don't count refunds as negative revenue
            'weight': row[weight_col]
        })
    
    return pd.DataFrame(results)


def compare_scenarios(tax_units, calculator, year, weight_col='PWGTP'):
    """Compare different calculation scenarios"""
    
    print("\n" + "="*80)
    print(f"SCENARIO COMPARISON: {year} TAX BRACKETS")
    print("="*80)
    
    # Scenario 1: No adjustments (baseline)
    print("\n## Scenario 1: No Adjustments (Baseline)")
    print("-"*80)
    results_base = []
    for _, row in tax_units.iterrows():
        tax_info = calculator.calculate_tax(row['income'], year, row['filing_status'])
        results_base.append({
            'tax': tax_info['tax_liability'],
            'weight': row[weight_col]
        })
    
    base_df = pd.DataFrame(results_base)
    base_revenue = (base_df['tax'] * base_df['weight']).sum()
    print(f"Total Revenue: ${base_revenue:,.2f}")
    
    # Scenario 2: With AGI adjustments only
    print("\n## Scenario 2: With AGI Adjustments")
    print("-"*80)
    results_agi = []
    for _, row in tax_units.iterrows():
        agi = estimate_agi_from_total_income(
            row['income'],
            filing_status=row['filing_status'],
            is_self_employed=(row['income'] > 50000 and np.random.random() < 0.10)
        )
        tax_info = calculator.calculate_tax(agi, year, row['filing_status'])
        results_agi.append({
            'tax': tax_info['tax_liability'],
            'weight': row[weight_col]
        })
    
    agi_df = pd.DataFrame(results_agi)
    agi_revenue = (agi_df['tax'] * agi_df['weight']).sum()
    print(f"Total Revenue: ${agi_revenue:,.2f}")
    print(f"Change from baseline: ${agi_revenue - base_revenue:,.2f} ({(agi_revenue/base_revenue - 1)*100:+.1f}%)")
    
    # Scenario 3: With AGI adjustments AND credits
    print("\n## Scenario 3: With AGI Adjustments AND Credits")
    print("-"*80)
    results_full = calculate_with_full_adjustments(tax_units, calculator, year, weight_col)
    
    full_revenue = (results_full['net_tax'] * results_full['weight']).sum()
    total_credits = (results_full['credits_total'] * results_full['weight']).sum()
    
    print(f"Total Revenue: ${full_revenue:,.2f}")
    print(f"Total Credits Applied: ${total_credits:,.2f}")
    print(f"Change from baseline: ${full_revenue - base_revenue:,.2f} ({(full_revenue/base_revenue - 1)*100:+.1f}%)")
    print(f"Change from AGI-only: ${full_revenue - agi_revenue:,.2f} ({(full_revenue/agi_revenue - 1)*100:+.1f}%)")
    
    return {
        'base_revenue': base_revenue,
        'agi_revenue': agi_revenue,
        'full_revenue': full_revenue,
        'total_credits': total_credits,
        'results_df': results_full
    }


def main():
    """Main analysis function"""
    
    print("\n" + "="*80)
    print("HAWAII TAX CALCULATION WITH AGI ADJUSTMENTS AND CREDITS")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    tax_units = load_tax_units()
    calculator = load_tax_data()
    
    weight_col = 'PWGTP' if 'PWGTP' in tax_units.columns else 'weight'
    
    print(f"Loaded {len(tax_units):,} tax units")
    print(f"Weighted total: {tax_units[weight_col].sum():,.0f}")
    
    # Calculate for 2017 brackets
    results_2017 = compare_scenarios(tax_units, calculator, 2017, weight_col)
    
    # Calculate for 2024 brackets
    results_2024 = compare_scenarios(tax_units, calculator, 2024, weight_col)
    
    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY: 2017 vs 2024 BRACKETS")
    print("="*80)
    
    actual_2023_revenue = 3_100_465_000
    
    print("\n## 2017 BRACKETS")
    print("-"*80)
    print(f"No adjustments:       ${results_2017['base_revenue']:>15,.2f}")
    print(f"With AGI adj:         ${results_2017['agi_revenue']:>15,.2f}")
    print(f"With AGI + credits:   ${results_2017['full_revenue']:>15,.2f}")
    print(f"Actual 2023 revenue:  ${actual_2023_revenue:>15,.2f}")
    print(f"Accuracy:             {results_2017['full_revenue']/actual_2023_revenue*100:>15.1f}%")
    
    print("\n## 2024 BRACKETS")
    print("-"*80)
    print(f"No adjustments:       ${results_2024['base_revenue']:>15,.2f}")
    print(f"With AGI adj:         ${results_2024['agi_revenue']:>15,.2f}")
    print(f"With AGI + credits:   ${results_2024['full_revenue']:>15,.2f}")
    
    print("\n## BRACKET SHIFT IMPACT (2017 → 2024)")
    print("-"*80)
    
    for scenario, label in [
        ('base_revenue', 'No adjustments'),
        ('agi_revenue', 'With AGI adj'),
        ('full_revenue', 'With AGI + credits')
    ]:
        revenue_2017 = results_2017[scenario]
        revenue_2024 = results_2024[scenario]
        change = revenue_2024 - revenue_2017
        pct_change = (revenue_2024 / revenue_2017 - 1) * 100
        
        print(f"\n{label}:")
        print(f"  2017: ${revenue_2017:>12,.0f}")
        print(f"  2024: ${revenue_2024:>12,.0f}")
        print(f"  Change: ${change:>12,.0f} ({pct_change:>+6.1f}%)")
    
    # Save detailed results
    output_file = Path(__file__).parent.parent / 'data' / 'processed' / 'tax_calculations_with_adjustments.parquet'
    results_2017['results_df']['year'] = 2017
    results_2024['results_df']['year'] = 2024
    
    combined_results = pd.concat([results_2017['results_df'], results_2024['results_df']])
    combined_results.to_parquet(output_file)
    
    print(f"\n\nDetailed results saved to: {output_file}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
