"""
Analyze Total Revenue Impact: 2017 vs 2024 Tax Brackets and Standard Deductions

This script calculates the total state tax revenue difference between 2017 and 2024
tax brackets and standard deductions using the constructed tax units.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tax.brackets import load_tax_data


def load_tax_units():
    """Load tax units from processed data"""
    # Try multiple possible file locations
    possible_files = [
        Path(__file__).parent.parent / 'data' / 'processed' / 'tax_units_rule_based.parquet',
        Path(__file__).parent.parent / 'data' / 'processed' / 'hawaii_ctc_full_population_20250722_161935.parquet',
        Path(__file__).parent.parent / 'data' / 'processed' / 'tax_units_with_geography.parquet',
    ]
    
    tax_units_file = None
    for file_path in possible_files:
        if file_path.exists():
            tax_units_file = file_path
            break
    
    if tax_units_file is None:
        print("Error: No tax units file found. Tried:")
        for f in possible_files:
            print(f"  - {f}")
        print("\nPlease run construct_tax_units.py first to generate tax units.")
        sys.exit(1)
    
    print(f"Loading tax units from: {tax_units_file}")
    tax_units = pd.read_parquet(tax_units_file)
    print(f"Loaded {len(tax_units):,} tax units\n")
    
    # Ensure required columns exist
    required_cols = ['income', 'filing_status']
    missing_cols = [col for col in required_cols if col not in tax_units.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(tax_units.columns)}")
        sys.exit(1)
    
    return tax_units


def calculate_revenue_by_year(tax_units, calculator, year, weight_col='PWGTP'):
    """Calculate total revenue for a given year"""
    print(f"Calculating taxes for {year}...")
    
    # Calculate taxes for each tax unit
    results = []
    for _, row in tax_units.iterrows():
        income = row['income']
        filing_status = row['filing_status']
        weight = row.get(weight_col, 1)
        
        try:
            tax_info = calculator.calculate_tax(income, year, filing_status)
            results.append({
                'tax_liability': tax_info['tax_liability'],
                'taxable_income': tax_info['taxable_income'],
                'standard_deduction': tax_info['standard_deduction'],
                'effective_rate': tax_info['effective_rate'],
                'weight': weight
            })
        except Exception as e:
            print(f"Warning: Error calculating tax for income ${income:,.2f}, status {filing_status}: {e}")
            results.append({
                'tax_liability': 0,
                'taxable_income': 0,
                'standard_deduction': 0,
                'effective_rate': 0,
                'weight': weight
            })
    
    results_df = pd.DataFrame(results)
    
    # Calculate weighted totals
    total_revenue = (results_df['tax_liability'] * results_df['weight']).sum()
    total_taxable_income = (results_df['taxable_income'] * results_df['weight']).sum()
    total_deductions = (results_df['standard_deduction'] * results_df['weight']).sum()
    avg_effective_rate = (results_df['effective_rate'] * results_df['weight']).sum() / results_df['weight'].sum()
    
    return {
        'year': year,
        'total_revenue': total_revenue,
        'total_taxable_income': total_taxable_income,
        'total_deductions': total_deductions,
        'avg_effective_rate': avg_effective_rate,
        'results_df': results_df
    }


def analyze_by_filing_status(tax_units, calculator, years, weight_col='PWGTP'):
    """Analyze revenue impact by filing status"""
    print("\n" + "="*80)
    print("REVENUE ANALYSIS BY FILING STATUS")
    print("="*80)
    
    filing_statuses = tax_units['filing_status'].unique()
    
    for status in filing_statuses:
        status_units = tax_units[tax_units['filing_status'] == status].copy()
        
        print(f"\n{status.upper()}")
        print("-"*80)
        print(f"Number of tax units: {len(status_units):,} "
              f"(weighted: {status_units[weight_col].sum():,.0f})")
        
        status_results = {}
        for year in years:
            results = []
            for _, row in status_units.iterrows():
                tax_info = calculator.calculate_tax(row['income'], year, row['filing_status'])
                results.append({
                    'tax_liability': tax_info['tax_liability'],
                    'weight': row[weight_col]
                })
            
            results_df = pd.DataFrame(results)
            total_revenue = (results_df['tax_liability'] * results_df['weight']).sum()
            status_results[year] = total_revenue
        
        # Display results
        print(f"\n  2017 Revenue: ${status_results[2017]:>15,.2f}")
        print(f"  2024 Revenue: ${status_results[2024]:>15,.2f}")
        print(f"  Difference:   ${status_results[2024] - status_results[2017]:>15,.2f}")
        print(f"  % Change:     {((status_results[2024] - status_results[2017]) / status_results[2017] * 100):>15.2f}%")


def analyze_by_income_quintile(tax_units, calculator, years, weight_col='PWGTP'):
    """Analyze revenue impact by income quintile"""
    print("\n" + "="*80)
    print("REVENUE ANALYSIS BY INCOME QUINTILE")
    print("="*80)
    
    # Create income quintiles
    tax_units['income_quintile'] = pd.qcut(
        tax_units['income'], 
        5, 
        labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (Highest)']
    )
    
    for quintile in ['Q1 (Lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (Highest)']:
        quintile_units = tax_units[tax_units['income_quintile'] == quintile].copy()
        
        print(f"\n{quintile}")
        print("-"*80)
        print(f"Income range: ${quintile_units['income'].min():,.2f} - ${quintile_units['income'].max():,.2f}")
        print(f"Number of tax units: {len(quintile_units):,} "
              f"(weighted: {quintile_units[weight_col].sum():,.0f})")
        
        quintile_results = {}
        for year in years:
            results = []
            for _, row in quintile_units.iterrows():
                tax_info = calculator.calculate_tax(row['income'], year, row['filing_status'])
                results.append({
                    'tax_liability': tax_info['tax_liability'],
                    'weight': row[weight_col]
                })
            
            results_df = pd.DataFrame(results)
            total_revenue = (results_df['tax_liability'] * results_df['weight']).sum()
            quintile_results[year] = total_revenue
        
        # Display results
        print(f"\n  2017 Revenue: ${quintile_results[2017]:>15,.2f}")
        print(f"  2024 Revenue: ${quintile_results[2024]:>15,.2f}")
        print(f"  Difference:   ${quintile_results[2024] - quintile_results[2017]:>15,.2f}")
        if quintile_results[2017] > 0:
            print(f"  % Change:     {((quintile_results[2024] - quintile_results[2017]) / quintile_results[2017] * 100):>15.2f}%")


def main():
    """Main analysis function"""
    print("\n" + "="*80)
    print("HAWAII STATE TAX REVENUE ANALYSIS: 2017 vs 2024")
    print("="*80)
    print("\nThis analysis compares total state tax revenue under:")
    print("  - 2017 tax brackets and standard deductions")
    print("  - 2024 tax brackets and standard deductions")
    print("\nUsing the same population of tax units from PUMS data.\n")
    
    # Load data
    tax_units = load_tax_units()
    calculator = load_tax_data()
    
    # Determine weight column
    if 'PWGTP' in tax_units.columns:
        weight_col = 'PWGTP'
    elif 'weight' in tax_units.columns:
        weight_col = 'weight'
    else:
        print("Warning: No weight column found. Using unweighted analysis.")
        tax_units['weight'] = 1
        weight_col = 'weight'
    
    print(f"Using weight column: {weight_col}")
    print(f"Total weighted tax units: {tax_units[weight_col].sum():,.0f}\n")
    
    # Calculate revenue for both years
    print("="*80)
    print("CALCULATING TOTAL REVENUE")
    print("="*80 + "\n")
    
    results_2017 = calculate_revenue_by_year(tax_units, calculator, 2017, weight_col)
    results_2024 = calculate_revenue_by_year(tax_units, calculator, 2024, weight_col)
    
    # Display summary
    print("\n" + "="*80)
    print("SUMMARY: TOTAL STATE TAX REVENUE")
    print("="*80)
    
    print(f"\n2017 TAX SYSTEM:")
    print(f"  Total Revenue:           ${results_2017['total_revenue']:>18,.2f}")
    print(f"  Total Taxable Income:    ${results_2017['total_taxable_income']:>18,.2f}")
    print(f"  Total Deductions:        ${results_2017['total_deductions']:>18,.2f}")
    print(f"  Avg Effective Rate:      {results_2017['avg_effective_rate']:>18.2f}%")
    
    print(f"\n2024 TAX SYSTEM:")
    print(f"  Total Revenue:           ${results_2024['total_revenue']:>18,.2f}")
    print(f"  Total Taxable Income:    ${results_2024['total_taxable_income']:>18,.2f}")
    print(f"  Total Deductions:        ${results_2024['total_deductions']:>18,.2f}")
    print(f"  Avg Effective Rate:      {results_2024['avg_effective_rate']:>18.2f}%")
    
    # Calculate differences
    revenue_diff = results_2024['total_revenue'] - results_2017['total_revenue']
    revenue_pct_change = (revenue_diff / results_2017['total_revenue']) * 100
    deduction_diff = results_2024['total_deductions'] - results_2017['total_deductions']
    deduction_pct_change = (deduction_diff / results_2017['total_deductions']) * 100
    
    print(f"\nCHANGE (2024 vs 2017):")
    print(f"  Revenue Difference:      ${revenue_diff:>18,.2f}")
    print(f"  Revenue % Change:        {revenue_pct_change:>18.2f}%")
    print(f"  Deduction Difference:    ${deduction_diff:>18,.2f}")
    print(f"  Deduction % Change:      {deduction_pct_change:>18.2f}%")
    
    if revenue_diff < 0:
        print(f"\n  ⚠️  Revenue DECREASED by ${abs(revenue_diff):,.2f} ({abs(revenue_pct_change):.2f}%)")
        print(f"      This represents a revenue loss to the state.")
    else:
        print(f"\n  ✓  Revenue INCREASED by ${revenue_diff:,.2f} ({revenue_pct_change:.2f}%)")
    
    # Detailed analysis
    analyze_by_filing_status(tax_units, calculator, [2017, 2024], weight_col)
    analyze_by_income_quintile(tax_units, calculator, [2017, 2024], weight_col)
    
    # Save detailed results
    print("\n" + "="*80)
    print("SAVING DETAILED RESULTS")
    print("="*80)
    
    # Create detailed comparison DataFrame
    tax_units_comparison = tax_units.copy()
    
    # Add 2017 calculations
    results_2017_list = []
    for _, row in tax_units.iterrows():
        tax_info = calculator.calculate_tax(row['income'], 2017, row['filing_status'])
        results_2017_list.append({
            'tax_2017': tax_info['tax_liability'],
            'taxable_income_2017': tax_info['taxable_income'],
            'deduction_2017': tax_info['standard_deduction'],
            'effective_rate_2017': tax_info['effective_rate']
        })
    
    results_2017_df = pd.DataFrame(results_2017_list)
    for col in results_2017_df.columns:
        tax_units_comparison[col] = results_2017_df[col]
    
    # Add 2024 calculations
    results_2024_list = []
    for _, row in tax_units.iterrows():
        tax_info = calculator.calculate_tax(row['income'], 2024, row['filing_status'])
        results_2024_list.append({
            'tax_2024': tax_info['tax_liability'],
            'taxable_income_2024': tax_info['taxable_income'],
            'deduction_2024': tax_info['standard_deduction'],
            'effective_rate_2024': tax_info['effective_rate']
        })
    
    results_2024_df = pd.DataFrame(results_2024_list)
    for col in results_2024_df.columns:
        tax_units_comparison[col] = results_2024_df[col]
    
    # Add difference columns
    tax_units_comparison['tax_change'] = tax_units_comparison['tax_2024'] - tax_units_comparison['tax_2017']
    tax_units_comparison['tax_change_pct'] = (
        (tax_units_comparison['tax_2024'] - tax_units_comparison['tax_2017']) / 
        tax_units_comparison['tax_2017'] * 100
    )
    
    # Save to file
    output_file = Path(__file__).parent.parent / 'data' / 'processed' / 'revenue_comparison_2017_vs_2024.parquet'
    tax_units_comparison.to_parquet(output_file)
    print(f"\nDetailed comparison saved to: {output_file}")
    
    # Also save summary CSV
    summary_file = Path(__file__).parent.parent / 'data' / 'processed' / 'revenue_summary_2017_vs_2024.csv'
    summary_data = {
        'Metric': [
            'Total Revenue 2017',
            'Total Revenue 2024',
            'Revenue Change',
            'Revenue % Change',
            'Total Deductions 2017',
            'Total Deductions 2024',
            'Deduction Change',
            'Deduction % Change',
            'Avg Effective Rate 2017',
            'Avg Effective Rate 2024'
        ],
        'Value': [
            results_2017['total_revenue'],
            results_2024['total_revenue'],
            revenue_diff,
            revenue_pct_change,
            results_2017['total_deductions'],
            results_2024['total_deductions'],
            deduction_diff,
            deduction_pct_change,
            results_2017['avg_effective_rate'],
            results_2024['avg_effective_rate']
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary saved to: {summary_file}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
