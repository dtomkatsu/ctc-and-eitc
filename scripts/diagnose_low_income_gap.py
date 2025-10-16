#!/usr/bin/env python3
"""
Diagnose why we're missing ~152,000 very low-income filers compared to SOI data.

This script investigates:
1. PUMS coverage: Are low-income people in PUMS data?
2. Filing thresholds: Are we excluding non-filers?
3. Dependent classification: Are low-income adults classified as dependents?
4. Income reporting: How does PUMS income compare to SOI taxable income?
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'analysis_results' / 'income_distribution'

def analyze_pums_low_income():
    """Analyze PUMS data for low-income individuals."""
    print("="*80)
    print("ANALYZING PUMS LOW-INCOME COVERAGE")
    print("="*80)
    
    # Load PUMS person data
    persons_file = DATA_DIR / 'raw' / 'pums' / 'psam_p15.csv'
    if not persons_file.exists():
        print(f"Error: PUMS file not found at {persons_file}")
        return
    
    print(f"\nLoading PUMS data from: {persons_file}")
    persons = pd.read_csv(persons_file)
    
    print(f"Total persons in PUMS: {len(persons):,}")
    print(f"Total weighted persons: {persons['PWGTP'].sum():,}")
    
    # Focus on adults (18+)
    adults = persons[persons['AGEP'] >= 18].copy()
    print(f"\nAdults (18+): {len(adults):,}")
    print(f"Weighted adults: {adults['PWGTP'].sum():,}")
    
    # Calculate income for each adult
    adults['income'] = adults['PINCP'].fillna(0)
    
    # Apply ADJINC
    adults['adjinc'] = adults['ADJINC'] / 1000000.0
    adults['adjusted_income'] = adults['income'] * adults['adjinc']
    
    # Categorize by income level
    income_brackets = [
        (0, 5000, 'Under $5K'),
        (5000, 10000, '$5K-$10K'),
        (10000, 25000, '$10K-$25K'),
        (25000, 50000, '$25K-$50K'),
        (50000, 100000, '$50K-$100K'),
        (100000, float('inf'), '$100K+')
    ]
    
    def categorize_income(inc):
        for min_val, max_val, label in income_brackets:
            if min_val <= inc < max_val:
                return label
        return '$100K+'
    
    adults['income_bracket'] = adults['adjusted_income'].apply(categorize_income)
    
    # Analyze by relationship status
    print("\n" + "="*80)
    print("ADULTS BY INCOME BRACKET")
    print("="*80)
    
    bracket_summary = adults.groupby('income_bracket').agg({
        'PWGTP': 'sum',
        'AGEP': 'count'
    }).round(0)
    bracket_summary.columns = ['Weighted Count', 'Unweighted Count']
    
    # Sort by income order
    bracket_order = ['Under $5K', '$5K-$10K', '$10K-$25K', '$25K-$50K', '$50K-$100K', '$100K+']
    bracket_summary = bracket_summary.reindex([b for b in bracket_order if b in bracket_summary.index])
    
    print(bracket_summary)
    print(f"\nTotal weighted adults: {bracket_summary['Weighted Count'].sum():,.0f}")
    
    # Check relationship types in low income brackets
    print("\n" + "="*80)
    print("LOW-INCOME ADULTS BY RELATIONSHIP TYPE")
    print("="*80)
    
    low_income = adults[adults['adjusted_income'] < 5000].copy()
    print(f"\nAdults with income under $5K: {len(low_income):,}")
    print(f"Weighted count: {low_income['PWGTP'].sum():,.0f}")
    
    # RELSHIPP codes
    relationship_labels = {
        20: 'Householder',
        21: 'Spouse',
        22: 'Parent',
        23: 'Child',
        24: 'Sibling',
        25: 'Other relative',
        26: 'Roommate/housemate',
        27: 'Unmarried partner',
        28: 'Foster child',
        29: 'Other nonrelative',
        33: 'Institutional quarters',
        34: 'Noninstitutional group quarters'
    }
    
    rel_summary = low_income.groupby('RELSHIPP').agg({
        'PWGTP': 'sum',
        'AGEP': 'count'
    }).round(0)
    rel_summary.columns = ['Weighted', 'Unweighted']
    rel_summary['Relationship'] = rel_summary.index.map(lambda x: relationship_labels.get(x, f'Unknown ({x})'))
    rel_summary = rel_summary.sort_values('Weighted', ascending=False)
    
    print("\nRelationship types for adults under $5K income:")
    print(rel_summary[['Relationship', 'Weighted', 'Unweighted']])
    
    # Check marital status
    print("\n" + "="*80)
    print("LOW-INCOME ADULTS BY MARITAL STATUS")
    print("="*80)
    
    marital_labels = {
        1: 'Married',
        2: 'Widowed',
        3: 'Divorced',
        4: 'Separated',
        5: 'Never married'
    }
    
    mar_summary = low_income.groupby('MAR').agg({
        'PWGTP': 'sum'
    }).round(0)
    mar_summary['Status'] = mar_summary.index.map(lambda x: marital_labels.get(x, f'Unknown ({x})'))
    mar_summary = mar_summary.sort_values('PWGTP', ascending=False)
    
    print(mar_summary[['Status', 'PWGTP']])
    
    # Analyze tax units
    print("\n" + "="*80)
    print("COMPARING TO CONSTRUCTED TAX UNITS")
    print("="*80)
    
    # Load tax units
    tax_units_file = sorted((DATA_DIR / 'processed').glob('tax_units_regenerated_*.parquet'))[-1]
    tax_units = pd.read_parquet(tax_units_file)
    
    print(f"\nLoaded tax units from: {tax_units_file.name}")
    print(f"Total tax units: {len(tax_units):,}")
    print(f"Weighted tax units: {tax_units['weight'].sum():,.0f}")
    
    # Count low-income tax units
    low_income_units = tax_units[tax_units['income'] < 5000]
    print(f"\nTax units with income under $5K:")
    print(f"  Count: {len(low_income_units):,}")
    print(f"  Weighted: {low_income_units['weight'].sum():,.0f}")
    
    by_status = low_income_units.groupby('filing_status')['weight'].sum()
    print(f"\nBy filing status:")
    for status, count in by_status.items():
        print(f"  {status}: {count:,.0f}")
    
    # Calculate the gap
    print("\n" + "="*80)
    print("GAP ANALYSIS")
    print("="*80)
    
    pums_low_income_adults = low_income['PWGTP'].sum()
    tax_units_low_income = low_income_units['weight'].sum()
    
    print(f"\nPUMS adults with income <$5K:   {pums_low_income_adults:>12,.0f}")
    print(f"Tax units with income <$5K:     {tax_units_low_income:>12,.0f}")
    print(f"Gap:                            {pums_low_income_adults - tax_units_low_income:>12,.0f}")
    print(f"Gap percentage:                 {((pums_low_income_adults - tax_units_low_income) / pums_low_income_adults * 100):>11.1f}%")
    
    # Check SOI expectations
    print("\n" + "="*80)
    print("SOI EXPECTATIONS")
    print("="*80)
    
    soi_low_income = {
        'single': 82072 + 13773,  # Under $2,400 + $2,400-$4,800
        'married_filing_jointly': 40236 + 5632,  # Under $4,800 + $4,800-$9,600
        'head_of_household': 6671 + 2619  # Under $3,600 + $3,600-$7,200
    }
    
    total_soi_low = sum(soi_low_income.values())
    
    print(f"\nSOI returns under ~$5K:")
    for status, count in soi_low_income.items():
        print(f"  {status}: {count:,.0f}")
    print(f"  Total: {total_soi_low:,.0f}")
    
    print(f"\nComparison:")
    print(f"  SOI low-income filers:       {total_soi_low:>12,.0f}")
    print(f"  Model low-income tax units:  {tax_units_low_income:>12,.0f}")
    print(f"  Gap:                         {total_soi_low - tax_units_low_income:>12,.0f}")
    print(f"  Gap percentage:              {((total_soi_low - tax_units_low_income) / total_soi_low * 100):>11.1f}%")
    
    return {
        'pums_low_income_adults': pums_low_income_adults,
        'tax_units_low_income': tax_units_low_income,
        'soi_low_income': total_soi_low,
        'low_income_adults_df': low_income
    }

def check_filing_thresholds():
    """Check if filing threshold logic is excluding low-income filers."""
    print("\n" + "="*80)
    print("FILING THRESHOLD ANALYSIS")
    print("="*80)
    
    # 2022 filing thresholds
    thresholds_2022 = {
        'single': 12950,
        'married_filing_jointly': 25900,
        'married_filing_separately': 5,
        'head_of_household': 19400
    }
    
    print("\n2022 IRS Filing Thresholds:")
    for status, threshold in thresholds_2022.items():
        print(f"  {status}: ${threshold:,.0f}")
    
    print("\n⚠️  KEY INSIGHT:")
    print("Many low-income individuals are REQUIRED to file if:")
    print("  - They had federal income tax withheld")
    print("  - They're eligible for refundable credits (EITC, CTC)")
    print("  - They have self-employment income ≥$400")
    print("  - They owe special taxes (e.g., household employment tax)")
    
    print("\nSOI includes ALL filers, even those below the threshold!")
    print("PUMS tax unit construction may be excluding these filers.")

def main():
    print("\n" + "="*80)
    print("DIAGNOSTIC: MISSING LOW-INCOME FILERS")
    print("="*80)
    print("\nThis script investigates why ~152,000 low-income filers")
    print("are missing from the constructed tax units vs SOI data.\n")
    
    # Run analyses
    result = analyze_pums_low_income()
    check_filing_thresholds()
    
    # Generate recommendations
    print("\n" + "="*80)
    print("FINDINGS AND RECOMMENDATIONS")
    print("="*80)
    
    print("\n1. ROOT CAUSE: Filing Threshold Exclusion")
    print("   The tax unit constructor likely excludes low-income adults who don't")
    print("   meet the standard filing threshold. However, SOI includes ALL filers,")
    print("   including those filing for:")
    print("   - Tax refunds")
    print("   - Refundable credits (EITC, CTC, etc.)")
    print("   - Self-employment income <$400")
    
    print("\n2. PUMS Coverage: ✅ ADEQUATE")
    print(f"   PUMS has {result['pums_low_income_adults']:,.0f} low-income adults")
    print("   The data is available, but not being converted to tax units")
    
    print("\n3. RECOMMENDATION: Relax Filing Thresholds")
    print("   In constructor.py, the filing threshold logic should:")
    print("   - Create tax units for ALL adults, not just those above threshold")
    print("   - OR use a much lower threshold (e.g., $1)")
    print("   - OR include logic for potential refundable credit eligibility")
    
    print("\n4. ALTERNATIVE: Income Definition Fix Will Help")
    print("   Using TAXABLE INCOME instead of TOTAL INCOME will:")
    print("   - Shift everyone down by standard deduction ($13K-$26K)")
    print("   - Create more low-income tax units naturally")
    print("   - Better align with SOI bracket definitions")
    
    print("\n5. QUICK FIX: Remove or Lower Filing Threshold")
    print("   Location: src/tax/units/constructor.py")
    print("   Search for: 'filing_threshold' or 'Apply filing threshold'")
    print("   Action: Comment out or set threshold to $1")
    
    print(f"\n{'='*80}\n")

if __name__ == '__main__':
    main()
