"""
Calculate Hawaii State Income Taxes for Tax Units

This script demonstrates how to:
1. Load tax brackets and standard deductions
2. Calculate taxes for individual scenarios
3. Apply tax calculations to a full dataset of tax units
4. Compare tax scenarios across different years
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tax.brackets import HawaiiTaxCalculator, load_tax_data


def example_individual_calculations():
    """Demonstrate individual tax calculations"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Individual Tax Calculations")
    print("="*70)
    
    calculator = load_tax_data()
    
    # Test cases for different filing statuses and income levels
    test_cases = [
        (30000, 2024, 'single', "Single filer, $30K income"),
        (75000, 2024, 'married_filing_jointly', "Joint filers, $75K income"),
        (50000, 2024, 'head_of_household', "Head of Household, $50K income"),
        (100000, 2024, 'single', "Single filer, $100K income"),
        (200000, 2024, 'married_filing_jointly', "Joint filers, $200K income"),
    ]
    
    for income, year, status, description in test_cases:
        result = calculator.calculate_tax(income, year, status)
        
        print(f"\n{description}")
        print("-" * 70)
        print(f"  Gross Income:        ${result['gross_income']:>12,.2f}")
        print(f"  Standard Deduction:  ${result['standard_deduction']:>12,.2f}")
        print(f"  Taxable Income:      ${result['taxable_income']:>12,.2f}")
        print(f"  Tax Liability:       ${result['tax_liability']:>12,.2f}")
        print(f"  Effective Rate:      {result['effective_rate']:>12.2f}%")
        print(f"  Marginal Rate:       {result['marginal_rate']:>12.2f}%")
        print(f"  Tax Bracket:         {result['bracket']}")


def example_year_comparison():
    """Compare tax liability across different years"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Tax Bracket Shift Analysis")
    print("="*70)
    
    calculator = load_tax_data()
    
    # Compare a $75,000 joint filer across different years
    income = 75000
    filing_status = 'married_filing_jointly'
    years = [2017, 2024, 2026, 2028]
    
    print(f"\nScenario: Joint filers with ${income:,} income")
    print("-" * 70)
    
    comparison = calculator.compare_scenarios(income, filing_status, years)
    
    # Display results
    print(f"\n{'Year':<8} {'Tax Liability':<15} {'Effective Rate':<15} {'Marginal Rate':<15}")
    print("-" * 70)
    
    for _, row in comparison.iterrows():
        print(f"{int(row['year']):<8} "
              f"${row['tax_liability']:>10,.2f}    "
              f"{row['effective_rate']:>8.2f}%       "
              f"{row['marginal_rate']:>8.2f}%")
    
    # Calculate changes
    baseline = comparison[comparison['year'] == 2024].iloc[0]
    print(f"\nChanges from 2024 baseline:")
    print("-" * 70)
    
    for _, row in comparison.iterrows():
        if row['year'] != 2024:
            tax_change = row['tax_liability'] - baseline['tax_liability']
            pct_change = (tax_change / baseline['tax_liability']) * 100
            print(f"{int(row['year'])}: ${tax_change:+,.2f} ({pct_change:+.1f}%)")


def example_income_distribution_analysis():
    """Analyze tax liability across income distribution"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Tax Liability by Income Level")
    print("="*70)
    
    calculator = load_tax_data()
    
    # Create income distribution
    incomes = [10000, 25000, 50000, 75000, 100000, 150000, 200000, 300000, 500000]
    filing_status = 'single'
    year = 2024
    
    print(f"\nFiling Status: Single")
    print(f"Year: {year}")
    print("-" * 70)
    print(f"{'Income':<15} {'Tax':<15} {'Effective Rate':<15} {'Marginal Rate':<15}")
    print("-" * 70)
    
    for income in incomes:
        result = calculator.calculate_tax(income, year, filing_status)
        print(f"${income:>10,}    "
              f"${result['tax_liability']:>10,.2f}    "
              f"{result['effective_rate']:>8.2f}%       "
              f"{result['marginal_rate']:>8.2f}%")


def apply_to_tax_units():
    """Apply tax calculations to actual tax units dataset"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Applying to Tax Units Dataset")
    print("="*70)
    
    # Check if tax units file exists
    tax_units_file = Path(__file__).parent.parent / 'data' / 'processed' / 'tax_units_rule_based.parquet'
    
    if not tax_units_file.exists():
        print(f"\nTax units file not found: {tax_units_file}")
        print("Run construct_tax_units.py first to generate tax units.")
        return
    
    print(f"\nLoading tax units from: {tax_units_file}")
    tax_units = pd.read_parquet(tax_units_file)
    
    print(f"Loaded {len(tax_units):,} tax units")
    
    # Load calculator
    calculator = load_tax_data()
    
    # Calculate taxes for 2024
    print("\nCalculating Hawaii state income taxes for 2024...")
    tax_units_with_tax = calculator.calculate_tax_for_dataframe(
        tax_units,
        income_col='income',
        filing_status_col='filing_status',
        year=2024
    )
    
    # Summary statistics
    print("\n" + "="*70)
    print("Tax Calculation Summary")
    print("="*70)
    
    # Calculate weighted totals if weight column exists
    if 'PWGTP' in tax_units_with_tax.columns:
        weight_col = 'PWGTP'
    elif 'weight' in tax_units_with_tax.columns:
        weight_col = 'weight'
    else:
        weight_col = None
    
    if weight_col:
        total_tax = (tax_units_with_tax['hi_tax_tax_liability'] * 
                    tax_units_with_tax[weight_col]).sum()
        total_income = (tax_units_with_tax['hi_tax_gross_income'] * 
                       tax_units_with_tax[weight_col]).sum()
        avg_effective_rate = (tax_units_with_tax['hi_tax_effective_rate'] * 
                             tax_units_with_tax[weight_col]).sum() / tax_units_with_tax[weight_col].sum()
        
        print(f"\nWeighted Totals:")
        print(f"  Total Income:        ${total_income:>15,.2f}")
        print(f"  Total Tax Liability: ${total_tax:>15,.2f}")
        print(f"  Avg Effective Rate:  {avg_effective_rate:>15.2f}%")
    
    # By filing status
    print("\n\nBy Filing Status:")
    print("-" * 70)
    
    for status in tax_units_with_tax['filing_status'].unique():
        status_data = tax_units_with_tax[tax_units_with_tax['filing_status'] == status]
        
        if weight_col:
            count = (status_data[weight_col].sum())
            avg_income = (status_data['hi_tax_gross_income'] * status_data[weight_col]).sum() / count
            avg_tax = (status_data['hi_tax_tax_liability'] * status_data[weight_col]).sum() / count
            avg_rate = (status_data['hi_tax_effective_rate'] * status_data[weight_col]).sum() / count
        else:
            count = len(status_data)
            avg_income = status_data['hi_tax_gross_income'].mean()
            avg_tax = status_data['hi_tax_tax_liability'].mean()
            avg_rate = status_data['hi_tax_effective_rate'].mean()
        
        print(f"\n{status}:")
        print(f"  Count:           {count:>15,.0f}")
        print(f"  Avg Income:      ${avg_income:>15,.2f}")
        print(f"  Avg Tax:         ${avg_tax:>15,.2f}")
        print(f"  Avg Eff. Rate:   {avg_rate:>15.2f}%")
    
    # Save results
    output_file = Path(__file__).parent.parent / 'data' / 'processed' / 'tax_units_with_hi_tax.parquet'
    tax_units_with_tax.to_parquet(output_file)
    print(f"\n\nResults saved to: {output_file}")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("HAWAII STATE INCOME TAX CALCULATOR")
    print("="*70)
    
    try:
        # Run examples
        example_individual_calculations()
        example_year_comparison()
        example_income_distribution_analysis()
        apply_to_tax_units()
        
        print("\n" + "="*70)
        print("All examples completed successfully!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
