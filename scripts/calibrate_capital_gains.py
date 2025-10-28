#!/usr/bin/env python3
"""
Calculate exact participation rates and scaling factors to match Table 21.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import glob
import os

# Load the latest tax units
parquet_files = glob.glob('data/processed/tax_units_calibrated_*.parquet')
latest_file = max(parquet_files, key=os.path.getmtime)
print(f"Loading: {latest_file}\n")
df = pd.read_parquet(latest_file)

# Table 21 benchmarks
table_21 = {
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
print("CALCULATING EXACT PARTICIPATION RATES AND SCALING FACTORS")
print("="*100)
print()

results = []

for (min_agi, max_agi), benchmark in table_21.items():
    # Filter to bracket (using agi_without_cap_gains to avoid circular logic)
    if max_agi == float('inf'):
        mask = df['agi_without_cap_gains'] >= min_agi
        bracket_label = f"${min_agi//1000}k+"
    else:
        mask = (df['agi_without_cap_gains'] >= min_agi) & (df['agi_without_cap_gains'] < max_agi)
        bracket_label = f"${min_agi//1000}k-${max_agi//1000}k"
    
    bracket_df = df[mask]
    
    # Total weighted filers in this bracket
    total_weighted_filers = bracket_df['weight'].sum()
    
    # Target filers with cap gains from Table 21
    target_filers = benchmark['filers']
    target_amount_millions = benchmark['amount_millions']
    
    # Calculate required participation rate
    if total_weighted_filers > 0:
        required_participation_rate = target_filers / total_weighted_filers
    else:
        required_participation_rate = 0
    
    # Calculate current amounts IF all filers participated
    # Use AGI without cap gains and apply the cap gains rate
    bracket_df_copy = bracket_df.copy()
    bracket_df_copy['estimated_taxable'] = bracket_df_copy['agi_without_cap_gains'] * 0.85
    
    # Get the cap gains rate for this bracket from the estimator
    from src.tax.adjustments.capital_gains import CapitalGainsEstimator
    estimator = CapitalGainsEstimator()
    
    sample_agi = (min_agi + (max_agi if max_agi != float('inf') else min_agi + 100000)) / 2
    cap_gains_rate = estimator.get_capital_gains_rate(sample_agi)
    
    # Calculate base capital gains (before scaling) for all filers
    bracket_df_copy['base_cap_gains'] = bracket_df_copy['estimated_taxable'] * cap_gains_rate
    
    # Total base capital gains if all target filers participated
    total_base_millions = (bracket_df_copy.nlargest(int(target_filers), 'weight')['base_cap_gains'] * 
                           bracket_df_copy.nlargest(int(target_filers), 'weight')['weight']).sum() / 1_000_000
    
    # Calculate required scaling factor
    if total_base_millions > 0:
        required_scaling = target_amount_millions / total_base_millions
    else:
        required_scaling = 1.0
    
    results.append({
        'bracket': bracket_label,
        'min_agi': min_agi,
        'max_agi': max_agi,
        'total_filers': total_weighted_filers,
        'target_filers': target_filers,
        'required_participation': required_participation_rate,
        'target_amount_m': target_amount_millions,
        'required_scaling': required_scaling,
    })

results_df = pd.DataFrame(results)

print("\n📊 REQUIRED PARTICIPATION RATES")
print("-" * 100)
print(f"{'Bracket':<15} {'Total Filers':>15} {'Target Filers':>15} {'Participation Rate':>20}")
print("-" * 100)
for _, row in results_df.iterrows():
    print(f"{row['bracket']:<15} {row['total_filers']:>15,.0f} {row['target_filers']:>15,.0f} {row['required_participation']:>19.4f}")

print("\n\n💰 REQUIRED SCALING FACTORS")
print("-" * 100)
print(f"{'Bracket':<15} {'Target Amount (M)':>20} {'Required Scaling':>20}")
print("-" * 100)
for _, row in results_df.iterrows():
    print(f"{row['bracket']:<15} ${row['target_amount_m']:>18,.1f} {row['required_scaling']:>19.2f}")

print("\n\n" + "="*100)
print("PYTHON CODE FOR capital_gains.py")
print("="*100)
print("\nPARTICIPATION_RATES = {")
for _, row in results_df.iterrows():
    if row['max_agi'] == float('inf'):
        print(f"    ({int(row['min_agi'])}, float('inf')): {row['required_participation']:.6f},  # {row['target_filers']:.0f} filers")
    else:
        print(f"    ({int(row['min_agi'])}, {int(row['max_agi'])}): {row['required_participation']:.6f},  # {row['target_filers']:.0f} filers")
print("}")

print("\nself.bracket_scaling = {")
for _, row in results_df.iterrows():
    if row['max_agi'] == float('inf'):
        print(f"    ({int(row['min_agi'])}, float('inf')): {row['required_scaling']:.2f},  # ${row['target_amount_m']:.1f}M")
    else:
        print(f"    ({int(row['min_agi'])}, {int(row['max_agi'])}): {row['required_scaling']:.2f},  # ${row['target_amount_m']:.1f}M")
print("}")
