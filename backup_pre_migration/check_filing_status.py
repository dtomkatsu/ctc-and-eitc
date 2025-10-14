#!/usr/bin/env python3
"""
Quick diagnostic script to check the filing status distribution in our tax units.
"""
import pandas as pd
import numpy as np

def main():
    # Load the tax units
    print("Loading tax units...")
    df = pd.read_parquet('src/data/processed/tax_units_rule_based.parquet')
    
    print(f"Total tax units: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    
    # Check filing status distribution
    print("\nFiling Status Distribution (Unweighted):")
    print(df['filing_status'].value_counts())
    
    print("\nFiling Status Distribution (Percentages):")
    print(df['filing_status'].value_counts(normalize=True) * 100)
    
    # Check if weights exist
    if 'weight' in df.columns:
        print(f"\nWeight column exists. Missing weights: {df['weight'].isna().sum()}")
        print(f"Weight range: {df['weight'].min():.2f} to {df['weight'].max():.2f}")
        
        # Fill missing weights with 1
        df['weight'] = df['weight'].fillna(1.0)
        
        # Calculate weighted distribution
        print("\nFiling Status Distribution (Weighted):")
        weighted_counts = df.groupby('filing_status')['weight'].sum()
        print(weighted_counts)
        
        print("\nFiling Status Distribution (Weighted Percentages):")
        weighted_pct = (weighted_counts / weighted_counts.sum()) * 100
        print(weighted_pct)
        
        # Compare to SOI benchmarks
        print("\n" + "="*80)
        print("COMPARISON TO SOI BENCHMARKS (2022 Hawaii DOTAX)")
        print("="*80)
        
        soi_benchmarks = {
            'single': 51.0,
            'married_filing_jointly': 36.0,
            'married_filing_separately': 3.1,
            'head_of_household': 9.6
        }
        
        print(f"{'Filing Status':<25} {'PUMS %':<10} {'SOI %':<10} {'Gap':<10}")
        print("-" * 55)
        
        for status, soi_pct in soi_benchmarks.items():
            pums_pct = weighted_pct.get(status, 0.0)
            gap = pums_pct - soi_pct
            print(f"{status:<25} {pums_pct:>7.1f}% {soi_pct:>7.1f}% {gap:>+7.1f}%")
    else:
        print("\nNo weight column found!")

if __name__ == "__main__":
    main()
