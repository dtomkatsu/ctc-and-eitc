#!/usr/bin/env python3
"""
Improved optimization that tests actual capital gains implementation.
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
import tempfile
import importlib.util

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

def test_scaling_factors(scaling_factors):
    """
    Test scaling factors by temporarily modifying the estimator and running it.
    Returns actual results by bracket.
    """
    # Create temporary modified capital_gains.py
    original_file = 'src/tax/adjustments/capital_gains.py'
    
    with open(original_file, 'r') as f:
        content = f.read()
    
    # Replace the bracket_scaling dictionary
    scaling_dict = "{\n"
    bracket_keys = list(TABLE_21_TARGETS.keys())
    for i, (min_agi, max_agi) in enumerate(bracket_keys):
        if max_agi == float('inf'):
            scaling_dict += f"            ({int(min_agi)}, float('inf')): {scaling_factors[i]:.6f},\n"
        else:
            scaling_dict += f"            ({int(min_agi)}, {int(max_agi)}): {scaling_factors[i]:.6f},\n"
    scaling_dict += "        }"
    
    # Find and replace the bracket_scaling section
    start_marker = "self.bracket_scaling = {"
    end_marker = "        }"
    
    start_idx = content.find(start_marker)
    if start_idx == -1:
        raise ValueError("Could not find bracket_scaling in capital_gains.py")
    
    # Find the end of the bracket_scaling dict
    brace_count = 0
    end_idx = start_idx + len(start_marker)
    while end_idx < len(content):
        if content[end_idx] == '{':
            brace_count += 1
        elif content[end_idx] == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx += 1
                break
        end_idx += 1
    
    # Replace the content
    new_content = content[:start_idx] + "self.bracket_scaling = " + scaling_dict + content[end_idx:]
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
        temp_file.write(new_content)
        temp_file_path = temp_file.name
    
    try:
        # Load the temporary module
        spec = importlib.util.spec_from_file_location("temp_capital_gains", temp_file_path)
        temp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(temp_module)
        
        # Apply capital gains to a sample of the data (for speed)
        sample_df = df.sample(n=min(10000, len(df)), random_state=42)
        
        result_df = temp_module.apply_capital_gains_to_dataframe(
            sample_df,
            agi_col='agi_without_cap_gains',
            include_random=True
        )
        
        # Calculate results by bracket
        results = {}
        for (min_agi, max_agi) in TABLE_21_TARGETS.keys():
            if max_agi == float('inf'):
                mask = result_df['agi_without_cap_gains'] >= min_agi
            else:
                mask = (result_df['agi_without_cap_gains'] >= min_agi) & (result_df['agi_without_cap_gains'] < max_agi)
            
            bracket_df = result_df[mask]
            
            # Scale up from sample to full population
            scale_factor = len(df) / len(sample_df)
            
            has_cap_gains = bracket_df['capital_gains'] > 0
            filers_with_cg = (has_cap_gains * bracket_df['weight']).sum() * scale_factor
            amount_millions = (bracket_df['capital_gains'] * bracket_df['weight']).sum() * scale_factor / 1_000_000
            
            results[(min_agi, max_agi)] = {
                'filers': filers_with_cg,
                'amount_millions': amount_millions
            }
        
        return results
        
    finally:
        # Clean up temporary file
        os.unlink(temp_file_path)

def objective_function(scaling_factors):
    """
    Objective function: minimize weighted sum of squared percentage errors.
    """
    try:
        results = test_scaling_factors(scaling_factors)
        
        total_error = 0.0
        total_weight = 0.0
        
        for (min_agi, max_agi), target in TABLE_21_TARGETS.items():
            predicted = results[(min_agi, max_agi)]
            
            # Amount error (weighted by target amount)
            if target['amount_millions'] > 0:
                amount_error = abs(predicted['amount_millions'] - target['amount_millions']) / target['amount_millions']
                weight = target['amount_millions']
                total_error += amount_error * amount_error * weight
                total_weight += weight
            else:
                # For zero targets, penalize any positive prediction
                if predicted['amount_millions'] > 0.1:
                    total_error += (predicted['amount_millions'] * 10) ** 2
                    total_weight += 1
        
        return total_error / max(total_weight, 1)
        
    except Exception as e:
        print(f"Error in objective function: {e}")
        return 1e6  # Large penalty for errors

# Start with current scaling factors as initial guess
current_scaling = [7.988, 1.000, 2.289, 1.327, 0.881, 0.744, 0.827, 0.895, 1.041, 1.072, 1.093, 2.762]

print("Testing current scaling factors...")
current_results = test_scaling_factors(current_scaling)

print("\nCurrent Results vs Targets:")
print("-" * 80)
print(f"{'Bracket':<15} {'Predicted':<12} {'Target':<12} {'Error %':<12}")
print("-" * 80)

total_predicted = 0
total_target = 0

for (min_agi, max_agi), target in TABLE_21_TARGETS.items():
    predicted = current_results[(min_agi, max_agi)]
    
    if target['amount_millions'] > 0:
        error_pct = abs(predicted['amount_millions'] - target['amount_millions']) / target['amount_millions'] * 100
    else:
        error_pct = 0 if predicted['amount_millions'] < 0.1 else 999
    
    bracket_label = f"${min_agi//1000}k+" if max_agi == float('inf') else f"${min_agi//1000}k-${max_agi//1000}k"
    print(f"{bracket_label:<15} ${predicted['amount_millions']:<11.1f} ${target['amount_millions']:<11.1f} {error_pct:<11.1f}%")
    
    total_predicted += predicted['amount_millions']
    total_target += target['amount_millions']

total_error_pct = abs(total_predicted - total_target) / total_target * 100
print("-" * 80)
print(f"{'TOTAL':<15} ${total_predicted:<11.1f} ${total_target:<11.1f} {total_error_pct:<11.1f}%")

print(f"\nCurrent objective value: {objective_function(current_scaling):.6f}")

# Check if current results are already good enough
max_bracket_error = 0
for (min_agi, max_agi), target in TABLE_21_TARGETS.items():
    predicted = current_results[(min_agi, max_agi)]
    if target['amount_millions'] > 0:
        error_pct = abs(predicted['amount_millions'] - target['amount_millions']) / target['amount_millions'] * 100
        max_bracket_error = max(max_bracket_error, error_pct)

if max_bracket_error <= 1.0 and total_error_pct <= 1.0:
    print(f"\n✅ Current scaling factors already achieve ±1% target!")
    print(f"Max bracket error: {max_bracket_error:.1f}%")
    print(f"Total error: {total_error_pct:.1f}%")
else:
    print(f"\n🔧 Optimizing to achieve ±1% target...")
    print(f"Current max bracket error: {max_bracket_error:.1f}%")
    print(f"Current total error: {total_error_pct:.1f}%")
    
    # Bounds: scaling factors should be positive and reasonable
    bounds = [(0.1, 100.0) for _ in range(len(current_scaling))]
    
    # Run optimization
    result = minimize(
        objective_function,
        current_scaling,
        method='Nelder-Mead',  # More robust for noisy functions
        bounds=bounds,
        options={'maxiter': 50, 'xatol': 1e-3}  # Reduced iterations for speed
    )
    
    if result.success:
        optimal_scaling = result.x
        print(f"\n✅ Optimization completed!")
        print(f"Final objective value: {result.fun:.6f}")
        
        # Test final results
        final_results = test_scaling_factors(optimal_scaling)
        
        print("\n" + "="*80)
        print("OPTIMAL SCALING FACTORS")
        print("="*80)
        print(f"{'Bracket':<15} {'Scaling':<12} {'Predicted':<12} {'Target':<12} {'Error %':<12}")
        print("-" * 80)
        
        total_predicted = 0
        total_target = 0
        
        for i, ((min_agi, max_agi), target) in enumerate(TABLE_21_TARGETS.items()):
            predicted = final_results[(min_agi, max_agi)]
            
            if target['amount_millions'] > 0:
                error_pct = abs(predicted['amount_millions'] - target['amount_millions']) / target['amount_millions'] * 100
            else:
                error_pct = 0 if predicted['amount_millions'] < 0.1 else 999
            
            bracket_label = f"${min_agi//1000}k+" if max_agi == float('inf') else f"${min_agi//1000}k-${max_agi//1000}k"
            print(f"{bracket_label:<15} {optimal_scaling[i]:<11.3f} ${predicted['amount_millions']:<11.1f} ${target['amount_millions']:<11.1f} {error_pct:<11.1f}%")
            
            total_predicted += predicted['amount_millions']
            total_target += target['amount_millions']
        
        total_error_pct = abs(total_predicted - total_target) / total_target * 100
        print("-" * 80)
        print(f"{'TOTAL':<15} {'':<12} ${total_predicted:<11.1f} ${total_target:<11.1f} {total_error_pct:<11.1f}%")
        
        print("\nOptimal scaling factors:")
        print(optimal_scaling)
        
    else:
        print(f"\n❌ Optimization failed: {result.message}")
