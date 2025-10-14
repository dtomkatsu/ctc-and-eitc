"""
Calculate Hawaii State Taxes - FULL MODEL

Complete implementation with:
1. AGI adjustments
2. Itemized vs standard deductions
3. Hawaii state tax credits
4. Comparison to actual revenue
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tax.brackets import load_tax_data
from src.tax.adjustments import estimate_agi_from_total_income, calculate_hawaii_credits, estimate_deduction


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


def calculate_full_model(tax_units, calculator, year, weight_col='PWGTP'):
    """Calculate taxes with full model: AGI + itemized deductions + credits"""
    
    print(f"\nCalculating taxes for {year} with FULL MODEL...")
    print("="*80)
    
    results = []
    itemized_count = 0
    
    for idx, row in tax_units.iterrows():
        if idx % 5000 == 0:
            print(f"  Processing {idx:,} / {len(tax_units):,}...")
        
        # Step 1: Adjust income to AGI
        total_income = row['income']
        age = row.get('age', None)
        filing_status = row['filing_status']
        
        agi = estimate_agi_from_total_income(
            total_income,
            age=age,
            filing_status=filing_status,
            is_self_employed=(total_income > 50000 and np.random.random() < 0.10)
        )
        
        # Step 2: Get standard deduction for this year/status
        # Use calculator to get standard deduction
        temp_calc = calculator.calculate_tax(agi, year, filing_status, use_taxable_income=False)
        standard_deduction = temp_calc['standard_deduction']
        
        # Step 3: Determine deduction (standard vs itemized)
        deduction_info = estimate_deduction(agi, filing_status, standard_deduction, age)
        
        if deduction_info['type'] == 'itemized':
            itemized_count += 1
        
        # Step 4: Calculate tax on taxable income
        taxable_income = max(0, agi - deduction_info['amount'])
        tax_info = calculator.calculate_tax(taxable_income, year, filing_status, use_taxable_income=True)
        tax_before_credits = tax_info['tax_liability']
        
        # Step 5: Apply Hawaii state tax credits
        num_dependents = row.get('num_dependents', 0)
        credits = calculate_hawaii_credits(
            agi=agi,
            filing_status=filing_status,
            num_dependents=num_dependents,
            tax_before_credits=tax_before_credits,
            year=year
        )
        
        # Step 6: Calculate net tax
        net_tax = max(0, tax_before_credits - credits['total_nonrefundable'])
        net_tax_with_refundable = net_tax - credits['total_refundable']
        
        results.append({
            'total_income': total_income,
            'agi': agi,
            'agi_adjustment': total_income - agi,
            'deduction_type': deduction_info['type'],
            'deduction_amount': deduction_info['amount'],
            'benefit_over_standard': deduction_info['benefit_over_standard'],
            'taxable_income': taxable_income,
            'tax_before_credits': tax_before_credits,
            'credits_nonrefundable': credits['total_nonrefundable'],
            'credits_refundable': credits['total_refundable'],
            'credits_total': credits['total'],
            'net_tax': max(0, net_tax_with_refundable),
            'weight': row[weight_col]
        })
    
    results_df = pd.DataFrame(results)
    
    # Calculate itemization rate
    total_weighted = results_df['weight'].sum()
    itemized_weighted = results_df[results_df['deduction_type'] == 'itemized']['weight'].sum()
    itemization_rate = itemized_weighted / total_weighted * 100
    
    print(f"\n  Itemization Rate: {itemization_rate:.1f}% (SOI target: 12.1%)")
    
    return results_df


def main():
    """Main analysis function"""
    
    print("\n" + "="*80)
    print("HAWAII TAX CALCULATION - FULL MODEL")
    print("(AGI Adjustments + Itemized Deductions + Tax Credits)")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    tax_units = load_tax_units()
    calculator = load_tax_data()
    
    weight_col = 'PWGTP' if 'PWGTP' in tax_units.columns else 'weight'
    
    print(f"Loaded {len(tax_units):,} tax units")
    print(f"Weighted total: {tax_units[weight_col].sum():,.0f}")
    
    # Calculate for 2017 brackets
    print("\n" + "="*80)
    print("2017 TAX BRACKETS")
    print("="*80)
    results_2017 = calculate_full_model(tax_units, calculator, 2017, weight_col)
    
    revenue_2017 = (results_2017['net_tax'] * results_2017['weight']).sum()
    total_agi_adj_2017 = (results_2017['agi_adjustment'] * results_2017['weight']).sum()
    total_itemized_benefit_2017 = (results_2017['benefit_over_standard'] * results_2017['weight']).sum()
    total_credits_2017 = (results_2017['credits_total'] * results_2017['weight']).sum()
    
    print(f"\nTotal Revenue: ${revenue_2017:,.2f}")
    print(f"Total AGI Adjustments: ${total_agi_adj_2017:,.2f}")
    print(f"Total Itemized Benefit: ${total_itemized_benefit_2017:,.2f}")
    print(f"Total Credits: ${total_credits_2017:,.2f}")
    
    # Calculate for 2024 brackets
    print("\n" + "="*80)
    print("2024 TAX BRACKETS")
    print("="*80)
    results_2024 = calculate_full_model(tax_units, calculator, 2024, weight_col)
    
    revenue_2024 = (results_2024['net_tax'] * results_2024['weight']).sum()
    total_agi_adj_2024 = (results_2024['agi_adjustment'] * results_2024['weight']).sum()
    total_itemized_benefit_2024 = (results_2024['benefit_over_standard'] * results_2024['weight']).sum()
    total_credits_2024 = (results_2024['credits_total'] * results_2024['weight']).sum()
    
    print(f"\nTotal Revenue: ${revenue_2024:,.2f}")
    print(f"Total AGI Adjustments: ${total_agi_adj_2024:,.2f}")
    print(f"Total Itemized Benefit: ${total_itemized_benefit_2024:,.2f}")
    print(f"Total Credits: ${total_credits_2024:,.2f}")
    
    # Summary comparison
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    actual_2023_revenue = 3_100_465_000
    
    print("\n## 2017 BRACKETS (Full Model)")
    print("-"*80)
    print(f"Estimated Revenue:        ${revenue_2017:>15,.2f}")
    print(f"Actual 2023 Revenue:      ${actual_2023_revenue:>15,.2f}")
    print(f"Accuracy:                 {revenue_2017/actual_2023_revenue*100:>15.1f}%")
    print(f"Difference:               ${revenue_2017 - actual_2023_revenue:>15,.2f}")
    
    print("\n## 2024 BRACKETS (Full Model)")
    print("-"*80)
    print(f"Estimated Revenue:        ${revenue_2024:>15,.2f}")
    
    print("\n## BRACKET SHIFT IMPACT (2017 → 2024)")
    print("-"*80)
    change = revenue_2024 - revenue_2017
    pct_change = (revenue_2024 / revenue_2017 - 1) * 100
    
    print(f"2017 Revenue:             ${revenue_2017:>15,.2f}")
    print(f"2024 Revenue:             ${revenue_2024:>15,.2f}")
    print(f"Change:                   ${change:>15,.2f}")
    print(f"% Change:                 {pct_change:>15.1f}%")
    
    print("\n## MODEL COMPONENTS IMPACT")
    print("-"*80)
    print(f"AGI Adjustments:          ${total_agi_adj_2017:>15,.2f}")
    print(f"Itemized Deduction Benefit: ${total_itemized_benefit_2017:>15,.2f}")
    print(f"Tax Credits:              ${total_credits_2017:>15,.2f}")
    print(f"Total Reductions:         ${total_agi_adj_2017 + total_itemized_benefit_2017 + total_credits_2017:>15,.2f}")
    
    # Save results
    output_file = Path(__file__).parent.parent / 'data' / 'processed' / 'tax_calculations_full_model.parquet'
    results_2017['year'] = 2017
    results_2024['year'] = 2024
    
    combined_results = pd.concat([results_2017, results_2024], ignore_index=True)
    combined_results.to_parquet(output_file)
    
    print(f"\n\nDetailed results saved to: {output_file}")
    print("="*80 + "\n")
if __name__ == '__main__':
    main()
