"""
Diagnose Model Disparities and Create Calibration

This script:
1. Compares our model to IRS SOI benchmarks
2. Identifies sources of disparity
3. Creates calibration factors to match actual revenue
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tax.brackets import load_tax_data


def load_soi_benchmarks():
    """Load SOI benchmarks"""
    soi_file = Path(__file__).parent.parent / 'data' / 'processed' / 'soi_benchmarks_2022.csv'
    return pd.read_csv(soi_file).iloc[0].to_dict()


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


def compare_to_soi(tax_units, soi, weight_col='PWGTP'):
    """Compare our model to SOI benchmarks"""
    
    print("\n" + "="*80)
    print("COMPARISON TO IRS SOI BENCHMARKS (2022)")
    print("="*80)
    
    # Calculate our totals
    our_total_units = tax_units[weight_col].sum()
    our_total_income = (tax_units['income'] * tax_units[weight_col]).sum()
    our_avg_income = our_total_income / our_total_units
    
    # Filing status distribution
    filing_status_counts = {}
    for status in tax_units['filing_status'].unique():
        count = tax_units[tax_units['filing_status'] == status][weight_col].sum()
        filing_status_counts[status] = count
    
    # Map to SOI categories
    our_single = filing_status_counts.get('single', 0)
    our_joint = filing_status_counts.get('married_filing_jointly', 0)
    our_hoh = filing_status_counts.get('head_of_household', 0)
    our_mfs = filing_status_counts.get('married_filing_separate', 0)
    
    print("\n## TAX UNITS / RETURNS")
    print("-"*80)
    print(f"{'Metric':<30} {'Our Model':>15} {'SOI 2022':>15} {'Difference':>15}")
    print("-"*80)
    print(f"{'Total Units/Returns':<30} {our_total_units:>15,.0f} {soi['total_returns']:>15,.0f} {our_total_units - soi['total_returns']:>15,.0f}")
    print(f"{'Coverage Rate':<30} {'':<15} {'':<15} {our_total_units / soi['total_returns'] * 100:>14.1f}%")
    
    print("\n## FILING STATUS DISTRIBUTION")
    print("-"*80)
    print(f"{'Status':<30} {'Our Model':>15} {'SOI 2022':>15} {'Difference':>15}")
    print("-"*80)
    print(f"{'Single':<30} {our_single:>15,.0f} {soi['single_returns']:>15,.0f} {our_single - soi['single_returns']:>15,.0f}")
    print(f"{'  % of total':<30} {our_single/our_total_units*100:>14.1f}% {soi['single_pct']:>14.1f}% {(our_single/our_total_units - soi['single_pct']/100)*100:>14.1f}pp")
    print(f"{'Joint':<30} {our_joint:>15,.0f} {soi['joint_returns']:>15,.0f} {our_joint - soi['joint_returns']:>15,.0f}")
    print(f"{'  % of total':<30} {our_joint/our_total_units*100:>14.1f}% {soi['joint_pct']:>14.1f}% {(our_joint/our_total_units - soi['joint_pct']/100)*100:>14.1f}pp")
    print(f"{'Head of Household':<30} {our_hoh:>15,.0f} {soi['hoh_returns']:>15,.0f} {our_hoh - soi['hoh_returns']:>15,.0f}")
    print(f"{'  % of total':<30} {our_hoh/our_total_units*100:>14.1f}% {soi['hoh_pct']:>14.1f}% {(our_hoh/our_total_units - soi['hoh_pct']/100)*100:>14.1f}pp")
    print(f"{'Married Filing Separate':<30} {our_mfs:>15,.0f} {soi['mfs_returns']:>15,.0f} {our_mfs - soi['mfs_returns']:>15,.0f}")
    print(f"{'  % of total':<30} {our_mfs/our_total_units*100:>14.1f}% {soi['mfs_pct']:>14.1f}% {(our_mfs/our_total_units - soi['mfs_pct']/100)*100:>14.1f}pp")
    
    print("\n## INCOME COMPARISON")
    print("-"*80)
    print(f"{'Metric':<30} {'Our Model':>15} {'SOI 2022':>15} {'Ratio':>15}")
    print("-"*80)
    print(f"{'Total Income':<30} ${our_total_income:>14,.0f} ${soi['total_income']:>14,.0f} {our_total_income/soi['total_income']:>14.2f}x")
    print(f"{'Average Income':<30} ${our_avg_income:>14,.2f} ${soi['avg_agi']:>14,.2f} {our_avg_income/soi['avg_agi']:>14.2f}x")
    
    return {
        'unit_coverage': our_total_units / soi['total_returns'],
        'income_ratio': our_total_income / soi['total_income'],
        'avg_income_ratio': our_avg_income / soi['avg_agi']
    }


def calculate_revenue_with_adjustments(tax_units, calculator, year, weight_col='PWGTP'):
    """Calculate revenue with different adjustment scenarios"""
    
    print("\n" + "="*80)
    print(f"REVENUE CALCULATION SCENARIOS ({year})")
    print("="*80)
    
    # Scenario 1: No adjustments (current model)
    print("\n## Scenario 1: No Adjustments (Current Model)")
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
    
    # Scenario 2: Reduce income by 10% (rough AGI adjustment)
    print("\n## Scenario 2: Reduce Income by 10% (AGI Adjustment)")
    print("-"*80)
    results_adj10 = []
    for _, row in tax_units.iterrows():
        adjusted_income = row['income'] * 0.90
        tax_info = calculator.calculate_tax(adjusted_income, year, row['filing_status'])
        results_adj10.append({
            'tax': tax_info['tax_liability'],
            'weight': row[weight_col]
        })
    
    adj10_df = pd.DataFrame(results_adj10)
    adj10_revenue = (adj10_df['tax'] * adj10_df['weight']).sum()
    print(f"Total Revenue: ${adj10_revenue:,.2f}")
    print(f"Change: ${adj10_revenue - base_revenue:,.2f} ({(adj10_revenue/base_revenue - 1)*100:+.1f}%)")
    
    # Scenario 3: Reduce income by 20%
    print("\n## Scenario 3: Reduce Income by 20%")
    print("-"*80)
    results_adj20 = []
    for _, row in tax_units.iterrows():
        adjusted_income = row['income'] * 0.80
        tax_info = calculator.calculate_tax(adjusted_income, year, row['filing_status'])
        results_adj20.append({
            'tax': tax_info['tax_liability'],
            'weight': row[weight_col]
        })
    
    adj20_df = pd.DataFrame(results_adj20)
    adj20_revenue = (adj20_df['tax'] * adj20_df['weight']).sum()
    print(f"Total Revenue: ${adj20_revenue:,.2f}")
    print(f"Change: ${adj20_revenue - base_revenue:,.2f} ({(adj20_revenue/base_revenue - 1)*100:+.1f}%)")
    
    # Scenario 4: Calibration factor to match actual 2023 revenue
    actual_2023_revenue = 3_100_465_000
    calibration_factor = actual_2023_revenue / base_revenue
    
    print("\n## Scenario 4: Calibrated to Actual 2023 Revenue")
    print("-"*80)
    print(f"Actual 2023 Revenue: ${actual_2023_revenue:,.2f}")
    print(f"Calibration Factor: {calibration_factor:.4f}")
    print(f"Implied Income Reduction: {(1 - calibration_factor)*100:.1f}%")
    
    return {
        'base_revenue': base_revenue,
        'adj10_revenue': adj10_revenue,
        'adj20_revenue': adj20_revenue,
        'calibration_factor': calibration_factor,
        'actual_2023_revenue': actual_2023_revenue
    }


def main():
    """Main diagnostic function"""
    
    print("\n" + "="*80)
    print("HAWAII TAX MODEL DIAGNOSTIC & CALIBRATION")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    soi = load_soi_benchmarks()
    tax_units = load_tax_units()
    calculator = load_tax_data()
    
    weight_col = 'PWGTP' if 'PWGTP' in tax_units.columns else 'weight'
    
    # Compare to SOI
    comparison = compare_to_soi(tax_units, soi, weight_col)
    
    # Calculate revenue scenarios
    revenue_scenarios = calculate_revenue_with_adjustments(tax_units, calculator, 2017, weight_col)
    
    # Summary and recommendations
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
    print("="*80)
    
    print("\n## Key Findings:")
    print("-"*80)
    print(f"1. Tax Unit Coverage: {comparison['unit_coverage']*100:.1f}% of SOI returns")
    print(f"   - We have {comparison['unit_coverage']*100:.1f}% of expected tax units")
    print(f"   - Missing ~{(1-comparison['unit_coverage'])*100:.1f}% of filers")
    
    print(f"\n2. Income Levels: {comparison['income_ratio']:.2f}x SOI total income")
    print(f"   - Our total income is {comparison['income_ratio']:.2f}x SOI")
    print(f"   - Average income is {comparison['avg_income_ratio']:.2f}x SOI average")
    
    print(f"\n3. Revenue Overestimate: {revenue_scenarios['base_revenue']/revenue_scenarios['actual_2023_revenue']:.2f}x actual")
    print(f"   - Model estimates: ${revenue_scenarios['base_revenue']:,.0f}")
    print(f"   - Actual 2023:     ${revenue_scenarios['actual_2023_revenue']:,.0f}")
    print(f"   - Overestimate:    ${revenue_scenarios['base_revenue'] - revenue_scenarios['actual_2023_revenue']:,.0f}")
    
    print("\n## Likely Causes of Disparity:")
    print("-"*80)
    print("1. **Income Measurement**")
    print("   - PUMS captures total income, not AGI")
    print("   - Missing: retirement contributions, HSA, self-employment deductions")
    print("   - Estimated impact: -10% to -15% of income")
    
    print("\n2. **Tax Credits Not Modeled**")
    print("   - Food/excise tax credit")
    print("   - Renewable energy credits")
    print("   - Other Hawaii state credits")
    print("   - Estimated impact: -5% to -10% of revenue")
    
    print("\n3. **Itemized Deductions**")
    print("   - We only model standard deduction")
    print("   - High-income filers often itemize")
    print("   - Estimated impact: -5% to -8% of revenue")
    
    print("\n4. **Tax Unit Coverage**")
    print(f"   - We have {comparison['unit_coverage']*100:.1f}% of expected filers")
    print("   - Some non-filers in PUMS data")
    print(f"   - Estimated impact: -{(1-comparison['unit_coverage'])*100:.1f}% of revenue")
    
    print("\n## Recommended Calibration:")
    print("-"*80)
    print(f"**Calibration Factor: {revenue_scenarios['calibration_factor']:.4f}**")
    print(f"\nTo match actual 2023 revenue of ${revenue_scenarios['actual_2023_revenue']:,.0f}:")
    print(f"  - Multiply all tax liabilities by {revenue_scenarios['calibration_factor']:.4f}")
    print(f"  - OR reduce all incomes by {(1-revenue_scenarios['calibration_factor'])*100:.1f}%")
    print(f"  - OR implement detailed adjustments (AGI, credits, itemization)")
    
    print("\n## Next Steps:")
    print("-"*80)
    print("1. **Quick Fix**: Apply calibration factor for immediate accuracy")
    print("2. **Medium-term**: Implement AGI adjustments (retirement, etc.)")
    print("3. **Long-term**: Add Hawaii tax credits and itemized deductions")
    print("4. **Validation**: Compare bracket-by-bracket to SOI data")
    
    # Save calibration factor
    calibration_file = Path(__file__).parent.parent / 'data' / 'processed' / 'calibration_factor.txt'
    with open(calibration_file, 'w') as f:
        f.write(f"{revenue_scenarios['calibration_factor']:.6f}\n")
    
    print(f"\n\nCalibration factor saved to: {calibration_file}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
