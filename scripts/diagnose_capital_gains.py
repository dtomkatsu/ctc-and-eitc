#!/usr/bin/env python3
"""
Diagnose Capital Gains Distribution

Compare model's capital gains distribution against DOTax Table 21 benchmarks.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np

# Load the latest tax units
import glob
import os

parquet_files = glob.glob('data/processed/tax_units_calibrated_*.parquet')
if not parquet_files:
    print("ERROR: No tax units file found!")
    sys.exit(1)

latest_file = max(parquet_files, key=os.path.getmtime)
print(f"Loading: {latest_file}")
df = pd.read_parquet(latest_file)

# Table 21 benchmarks by bracket
table_21_benchmarks = {
    (0, 10000): {'filers': 34, 'amount_millions': 0.5},
    (10000, 20000): {'filers': 13, 'amount_millions': 0.0},
    (20000, 30000): {'filers': 234, 'amount_millions': 0.3},
    (30000, 40000): {'filers': 1616, 'amount_millions': 3.6},
    (40000, 50000): {'filers': 2045, 'amount_millions': 7.7},
    (50000, 75000): {'filers': 6044, 'amount_millions': 30.1},
    (75000, 100000): {'filers': 6067, 'amount_millions': 46.0},
    (100000, 150000): {'filers': 9146, 'amount_millions': 112.0},
    (150000, 200000): {'filers': 6095, 'amount_millions': 126.0},
    (200000, 300000): {'filers': 5577, 'amount_millions': 242.0},
    (300000, 400000): {'filers': 2384, 'amount_millions': 216.9},
    (400000, float('inf')): {'filers': 4188, 'amount_millions': 2210.3},
}

print("="*100)
print("CAPITAL GAINS DISTRIBUTION DIAGNOSIS")
print("="*100)
print("\n📋 NOTE: This compares AMOUNTS of capital gains (in dollars), not tax liability.")
print("   Capital gains are added to AGI, which then affects taxable income and tax liability.")
print("   Table 21 shows capital gains amounts by bracket, not the tax on those gains.\n")
print()

# Analyze by bracket
results = []

for (min_agi, max_agi), benchmark in table_21_benchmarks.items():
    # Filter to bracket
    if max_agi == float('inf'):
        mask = df['agi_without_cap_gains'] >= min_agi
        bracket_label = f"${min_agi//1000}k+"
    else:
        mask = (df['agi_without_cap_gains'] >= min_agi) & (df['agi_without_cap_gains'] < max_agi)
        bracket_label = f"${min_agi//1000}k-${max_agi//1000}k"
    
    bracket_df = df[mask]
    
    # Calculate model metrics
    has_cap_gains = bracket_df['capital_gains'] > 0
    model_filers = (has_cap_gains * bracket_df['weight']).sum()
    model_amount_millions = (bracket_df['capital_gains'] * bracket_df['weight']).sum() / 1_000_000
    model_avg_per_filer = model_amount_millions * 1_000_000 / model_filers if model_filers > 0 else 0
    
    # Benchmark metrics
    bench_filers = benchmark['filers']
    bench_amount_millions = benchmark['amount_millions']
    bench_avg_per_filer = bench_amount_millions * 1_000_000 / bench_filers if bench_filers > 0 else 0
    
    # Calculate gaps
    filer_gap_pct = ((model_filers - bench_filers) / bench_filers * 100) if bench_filers > 0 else 0
    amount_gap_pct = ((model_amount_millions - bench_amount_millions) / bench_amount_millions * 100) if bench_amount_millions > 0 else 0
    avg_gap_pct = ((model_avg_per_filer - bench_avg_per_filer) / bench_avg_per_filer * 100) if bench_avg_per_filer > 0 else 0
    
    results.append({
        'bracket': bracket_label,
        'model_filers': model_filers,
        'bench_filers': bench_filers,
        'filer_gap_pct': filer_gap_pct,
        'model_amount_m': model_amount_millions,
        'bench_amount_m': bench_amount_millions,
        'amount_gap_pct': amount_gap_pct,
        'model_avg': model_avg_per_filer,
        'bench_avg': bench_avg_per_filer,
        'avg_gap_pct': avg_gap_pct,
    })

# Create DataFrame for easy display
results_df = pd.DataFrame(results)

print("\n📊 FILERS WITH CAPITAL GAINS BY BRACKET")
print("-" * 100)
print(f"{'Bracket':<15} {'Model':>10} {'Benchmark':>10} {'Gap':>12} {'% Gap':>10}")
print("-" * 100)
for _, row in results_df.iterrows():
    gap_icon = "✅" if abs(row['filer_gap_pct']) <= 10 else "⚠️" if abs(row['filer_gap_pct']) <= 25 else "❌"
    print(f"{row['bracket']:<15} {row['model_filers']:>10,.0f} {row['bench_filers']:>10,.0f} "
          f"{row['model_filers']-row['bench_filers']:>12,.0f} {row['filer_gap_pct']:>9.1f}% {gap_icon}")

print("-" * 100)
print(f"{'TOTAL':<15} {results_df['model_filers'].sum():>10,.0f} {results_df['bench_filers'].sum():>10,.0f} "
      f"{results_df['model_filers'].sum()-results_df['bench_filers'].sum():>12,.0f} "
      f"{(results_df['model_filers'].sum()-results_df['bench_filers'].sum())/results_df['bench_filers'].sum()*100:>9.1f}%")

print("\n\n💰 CAPITAL GAINS AMOUNTS BY BRACKET (Millions)")
print("-" * 100)
print(f"{'Bracket':<15} {'Model':>10} {'Benchmark':>10} {'Gap':>12} {'% Gap':>10}")
print("-" * 100)
for _, row in results_df.iterrows():
    gap_icon = "✅" if abs(row['amount_gap_pct']) <= 10 else "⚠️" if abs(row['amount_gap_pct']) <= 25 else "❌"
    print(f"{row['bracket']:<15} ${row['model_amount_m']:>9,.1f} ${row['bench_amount_m']:>9,.1f} "
          f"${row['model_amount_m']-row['bench_amount_m']:>11,.1f} {row['amount_gap_pct']:>9.1f}% {gap_icon}")

print("-" * 100)
print(f"{'TOTAL':<15} ${results_df['model_amount_m'].sum():>9,.1f} ${results_df['bench_amount_m'].sum():>9,.1f} "
      f"${results_df['model_amount_m'].sum()-results_df['bench_amount_m'].sum():>11,.1f} "
      f"{(results_df['model_amount_m'].sum()-results_df['bench_amount_m'].sum())/results_df['bench_amount_m'].sum()*100:>9.1f}%")

print("\n\n📈 AVERAGE CAPITAL GAINS PER FILER BY BRACKET")
print("-" * 100)
print(f"{'Bracket':<15} {'Model':>12} {'Benchmark':>12} {'Gap':>12} {'% Gap':>10}")
print("-" * 100)
for _, row in results_df.iterrows():
    gap_icon = "✅" if abs(row['avg_gap_pct']) <= 10 else "⚠️" if abs(row['avg_gap_pct']) <= 25 else "❌"
    print(f"{row['bracket']:<15} ${row['model_avg']:>11,.0f} ${row['bench_avg']:>11,.0f} "
          f"${row['model_avg']-row['bench_avg']:>11,.0f} {row['avg_gap_pct']:>9.1f}% {gap_icon}")

print("-" * 100)
total_model_avg = results_df['model_amount_m'].sum() * 1_000_000 / results_df['model_filers'].sum()
total_bench_avg = results_df['bench_amount_m'].sum() * 1_000_000 / results_df['bench_filers'].sum()
print(f"{'TOTAL':<15} ${total_model_avg:>11,.0f} ${total_bench_avg:>11,.0f} "
      f"${total_model_avg-total_bench_avg:>11,.0f} "
      f"{(total_model_avg-total_bench_avg)/total_bench_avg*100:>9.1f}%")

print("\n\n🎯 KEY FINDINGS:")
print("-" * 100)
print(f"1. Total filers: {results_df['model_filers'].sum():,.0f} vs {results_df['bench_filers'].sum():,.0f} benchmark "
      f"({(results_df['model_filers'].sum()/results_df['bench_filers'].sum()-1)*100:+.1f}%)")
print(f"2. Total amount: ${results_df['model_amount_m'].sum():,.1f}M vs ${results_df['bench_amount_m'].sum():,.1f}M benchmark "
      f"({(results_df['model_amount_m'].sum()/results_df['bench_amount_m'].sum()-1)*100:+.1f}%)")
print(f"3. Avg per filer: ${total_model_avg:,.0f} vs ${total_bench_avg:,.0f} benchmark "
      f"({(total_model_avg/total_bench_avg-1)*100:+.1f}%)")

# Identify specific issues
print("\n📋 SPECIFIC ISSUES:")
over_filers = results_df[results_df['filer_gap_pct'] > 25]
if len(over_filers) > 0:
    print(f"\n⚠️  Brackets with TOO MANY filers (>25% over):")
    for _, row in over_filers.iterrows():
        print(f"   - {row['bracket']}: {row['filer_gap_pct']:+.1f}%")

under_amounts = results_df[results_df['amount_gap_pct'] < -25]
if len(under_amounts) > 0:
    print(f"\n⚠️  Brackets with TOO LOW amounts (>25% under):")
    for _, row in under_amounts.iterrows():
        print(f"   - {row['bracket']}: {row['amount_gap_pct']:+.1f}%")

print("\n" + "="*100)
