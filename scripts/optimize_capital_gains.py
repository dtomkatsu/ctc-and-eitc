#!/usr/bin/env python3
"""
Systematic optimization of capital gains scaling factors to match Table 21 exactly.
Uses scipy optimization to find scaling factors that minimize error across all brackets.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
import glob
import os
from scipy.optimize import minimize
from src.tax.adjustments.capital_gains import CapitalGainsEstimator

# Table 21 targets
TABLE_21_TARGETS = {
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

# Load the latest tax units
parquet_files = glob.glob('data/processed/tax_units_calibrated_*.parquet')
latest_file = max(parquet_files, key=os.path.getmtime)
print(f"Loading: {latest_file}")
df = pd.read_parquet(latest_file)

# Prepare bracket data for optimization
bracket_data = {}
for (min_agi, max_agi), target in TABLE_21_TARGETS.items():
    # Filter to bracket
    if max_agi == float('inf'):
        mask = df['agi_without_cap_gains'] >= min_agi
    else:
        mask = (df['agi_without_cap_gains'] >= min_agi) & (df['agi_without_cap_gains'] < max_agi)
    
    bracket_df = df[mask]
    
    # Calculate base capital gains amounts (before scaling)
    estimator = CapitalGainsEstimator()
    sample_agi = (min_agi + (max_agi if max_agi != float('inf') else min_agi + 100000)) / 2
    cap_gains_rate = estimator.get_capital_gains_rate(sample_agi)
    
    # Get participation rate
    participation_rate = estimator.get_participation_rate(sample_agi)
    
    # Calculate expected filers and base amounts
    total_filers = bracket_df['weight'].sum()
    expected_filers_with_cg = total_filers * participation_rate
    
    # Calculate base capital gains for expected filers
    bracket_df_copy = bracket_df.copy()
    bracket_df_copy['estimated_taxable'] = bracket_df_copy['agi_without_cap_gains'] * 0.85
    bracket_df_copy['base_cap_gains'] = bracket_df_copy['estimated_taxable'] * cap_gains_rate
    
    # Get the top filers who would have capital gains
    top_filers = bracket_df_copy.nlargest(int(expected_filers_with_cg), 'weight')
    base_amount_millions = (top_filers['base_cap_gains'] * top_filers['weight']).sum() / 1_000_000
    
    bracket_data[(min_agi, max_agi)] = {
        'target_amount': target['amount_millions'],
        'base_amount': base_amount_millions,
        'weight': target['amount_millions']  # Weight by target amount for optimization
    }

print("\nBracket Base Amounts vs Targets:")
print("-" * 80)
print(f"{'Bracket':<15} {'Base Amount':<15} {'Target':<15} {'Required Scaling':<20}")
print("-" * 80)
for (min_agi, max_agi), data in bracket_data.items():
    bracket_label = f"${min_agi//1000}k+" if max_agi == float('inf') else f"${min_agi//1000}k-${max_agi//1000}k"
    required_scaling = data['target_amount'] / data['base_amount'] if data['base_amount'] > 0 else 1.0
    print(f"{bracket_label:<15} ${data['base_amount']:<14.2f} ${data['target_amount']:<14.2f} {required_scaling:<19.2f}")

def objective_function(scaling_factors):
    """
    Objective function to minimize: weighted sum of squared percentage errors.
    """
    total_error = 0.0
    total_weight = 0.0
    
    for i, ((min_agi, max_agi), data) in enumerate(bracket_data.items()):
        scaling_factor = scaling_factors[i]
        predicted_amount = data['base_amount'] * scaling_factor
        target_amount = data['target_amount']
        
        if target_amount > 0:
            # Percentage error
            error = abs(predicted_amount - target_amount) / target_amount
            weight = data['weight']
            total_error += error * error * weight
            total_weight += weight
        else:
            # For zero targets, penalize any positive prediction
            if predicted_amount > 0.1:  # Allow small rounding errors
                total_error += (predicted_amount * 10) ** 2  # Heavy penalty
                total_weight += 1
    
    return total_error / max(total_weight, 1)

def constraint_total_amount(scaling_factors):
    """
    Constraint: total amount should equal Table 21 total (2995.4M)
    """
    total_predicted = 0.0
    for i, ((min_agi, max_agi), data) in enumerate(bracket_data.items()):
        scaling_factor = scaling_factors[i]
        total_predicted += data['base_amount'] * scaling_factor
    
    target_total = sum(target['amount_millions'] for target in TABLE_21_TARGETS.values())
    return abs(total_predicted - target_total) - 1.0  # Allow ±1M error

# Initial guess: current scaling factors
initial_scaling = []
for (min_agi, max_agi), data in bracket_data.items():
    if data['base_amount'] > 0:
        initial_scaling.append(data['target_amount'] / data['base_amount'])
    else:
        initial_scaling.append(1.0)

print(f"\nInitial scaling factors: {initial_scaling}")

# Bounds: scaling factors should be positive and reasonable
bounds = [(0.1, 100.0) for _ in range(len(bracket_data))]

# Constraints
constraints = [
    {'type': 'ineq', 'fun': lambda x: -constraint_total_amount(x)},  # Total within ±1M
]

print("\nOptimizing scaling factors...")
print("Target: Each bracket within ±1% and total within ±1M")

# Run optimization
result = minimize(
    objective_function,
    initial_scaling,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'maxiter': 1000, 'ftol': 1e-9}
)

if result.success:
    optimal_scaling = result.x
    print(f"\n✅ Optimization successful!")
    print(f"Final objective value: {result.fun:.6f}")
    
    print("\n" + "="*100)
    print("OPTIMAL SCALING FACTORS")
    print("="*100)
    
    # Validate results
    total_predicted = 0.0
    all_within_1pct = True
    
    print(f"{'Bracket':<15} {'Scaling':<12} {'Predicted':<12} {'Target':<12} {'Error %':<12} {'Status':<10}")
    print("-" * 100)
    
    for i, ((min_agi, max_agi), data) in enumerate(bracket_data.items()):
        scaling_factor = optimal_scaling[i]
        predicted_amount = data['base_amount'] * scaling_factor
        target_amount = data['target_amount']
        
        if target_amount > 0:
            error_pct = abs(predicted_amount - target_amount) / target_amount * 100
        else:
            error_pct = 0 if predicted_amount < 0.1 else 999
        
        status = "✅" if error_pct <= 1.0 else "❌"
        if error_pct > 1.0:
            all_within_1pct = False
        
        bracket_label = f"${min_agi//1000}k+" if max_agi == float('inf') else f"${min_agi//1000}k-${max_agi//1000}k"
        print(f"{bracket_label:<15} {scaling_factor:<11.2f} ${predicted_amount:<11.1f} ${target_amount:<11.1f} {error_pct:<11.1f}% {status:<10}")
        
        total_predicted += predicted_amount
    
    target_total = sum(target['amount_millions'] for target in TABLE_21_TARGETS.values())
    total_error_pct = abs(total_predicted - target_total) / target_total * 100
    total_status = "✅" if total_error_pct <= 0.1 else "❌"
    
    print("-" * 100)
    print(f"{'TOTAL':<15} {'':<12} ${total_predicted:<11.1f} ${target_total:<11.1f} {total_error_pct:<11.1f}% {total_status:<10}")
    
    print("\n" + "="*100)
    print("PYTHON CODE FOR capital_gains.py")
    print("="*100)
    print("\nself.bracket_scaling = {")
    for i, (min_agi, max_agi) in enumerate(bracket_data.keys()):
        if max_agi == float('inf'):
            print(f"    ({int(min_agi)}, float('inf')): {optimal_scaling[i]:.3f},")
        else:
            print(f"    ({int(min_agi)}, {int(max_agi)}): {optimal_scaling[i]:.3f},")
    print("}")
    
    if all_within_1pct and total_error_pct <= 0.1:
        print(f"\n🎯 SUCCESS: All brackets within ±1% and total within ±0.1%!")
    else:
        print(f"\n⚠️  Some brackets still outside ±1% target. Consider adjusting constraints.")

else:
    print(f"\n❌ Optimization failed: {result.message}")
    print("Try adjusting initial conditions or constraints.")
