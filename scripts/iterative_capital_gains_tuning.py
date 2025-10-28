#!/usr/bin/env python3
"""
Iterative tuning of capital gains scaling factors to achieve ±1% accuracy.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
import glob
import os

# Table 21 targets
TABLE_21_TARGETS = {
    (0, 10000): 0.5,
    (10000, 20000): 0.0,
    (20000, 30000): 0.3,
    (30000, 40000): 3.6,
    (40000, 50000): 7.7,
    (50000, 75000): 30.1,
    (75000, 100000): 46.0,
    (100000, 150000): 112.0,
    (150000, 200000): 126.0,
    (200000, 300000): 242.0,
    (300000, 400000): 216.9,
    (400000, float('inf')): 2210.3,
}

def get_current_results():
    """Get current capital gains results by bracket."""
    # Load the latest tax units
    parquet_files = glob.glob('data/processed/tax_units_calibrated_*.parquet')
    latest_file = max(parquet_files, key=os.path.getmtime)
    df = pd.read_parquet(latest_file)
    
    results = {}
    for (min_agi, max_agi), target in TABLE_21_TARGETS.items():
        if max_agi == float('inf'):
            mask = df['agi_without_cap_gains'] >= min_agi
        else:
            mask = (df['agi_without_cap_gains'] >= min_agi) & (df['agi_without_cap_gains'] < max_agi)
        
        bracket_df = df[mask]
        amount_millions = (bracket_df['capital_gains'] * bracket_df['weight']).sum() / 1_000_000
        results[(min_agi, max_agi)] = amount_millions
    
    return results

def update_scaling_factor(bracket, new_scaling):
    """Update a specific bracket's scaling factor in the capital_gains.py file."""
    file_path = 'src/tax/adjustments/capital_gains.py'
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the specific line for this bracket
    if bracket[1] == float('inf'):
        search_pattern = f"({int(bracket[0])}, float('inf')):"
    else:
        search_pattern = f"({int(bracket[0])}, {int(bracket[1])}):"
    
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if search_pattern in line:
            # Extract the comment if it exists
            if '#' in line:
                comment = line[line.index('#'):]
            else:
                comment = ""
            
            # Replace the line
            indent = len(line) - len(line.lstrip())
            lines[i] = ' ' * indent + f"{search_pattern} {new_scaling:.3f},  {comment}"
            break
    
    # Write back to file
    with open(file_path, 'w') as f:
        f.write('\n'.join(lines))

def regenerate_and_get_results():
    """Regenerate tax units and return new results."""
    os.system('python scripts/regenerate_tax_units.py > /dev/null 2>&1')
    return get_current_results()

# Get initial results
print("Getting current results...")
current_results = get_current_results()

print("\nCurrent Results vs Targets:")
print("-" * 80)
print(f"{'Bracket':<15} {'Current':<12} {'Target':<12} {'Error %':<12} {'Adjustment':<12}")
print("-" * 80)

total_current = 0
total_target = 0
adjustments_needed = []

for (min_agi, max_agi), target in TABLE_21_TARGETS.items():
    current = current_results[(min_agi, max_agi)]
    
    if target > 0:
        error_pct = abs(current - target) / target * 100
        # Calculate adjustment factor needed
        adjustment_factor = target / current if current > 0 else 1.0
    else:
        error_pct = 0 if current < 0.1 else 999
        adjustment_factor = 0.1  # Reduce to near zero
    
    bracket_label = f"${min_agi//1000}k+" if max_agi == float('inf') else f"${min_agi//1000}k-${max_agi//1000}k"
    status = "✅" if error_pct <= 1.0 else "❌"
    
    print(f"{bracket_label:<15} ${current:<11.1f} ${target:<11.1f} {error_pct:<11.1f}% {adjustment_factor:<11.3f} {status}")
    
    total_current += current
    total_target += target
    
    if error_pct > 1.0:
        adjustments_needed.append(((min_agi, max_agi), adjustment_factor))

total_error_pct = abs(total_current - total_target) / total_target * 100
print("-" * 80)
print(f"{'TOTAL':<15} ${total_current:<11.1f} ${total_target:<11.1f} {total_error_pct:<11.1f}%")

if not adjustments_needed:
    print(f"\n✅ All brackets already within ±1% target!")
else:
    print(f"\n🔧 Need to adjust {len(adjustments_needed)} brackets")
    
    # Read current scaling factors
    file_path = 'src/tax/adjustments/capital_gains.py'
    with open(file_path, 'r') as f:
        content = f.read()
    
    current_scaling = {}
    lines = content.split('\n')
    for line in lines:
        if ': ' in line and '# $' in line:
            for (min_agi, max_agi) in TABLE_21_TARGETS.keys():
                if max_agi == float('inf'):
                    search_pattern = f"({int(min_agi)}, float('inf')):"
                else:
                    search_pattern = f"({int(min_agi)}, {int(max_agi)}):"
                
                if search_pattern in line:
                    # Extract current scaling factor
                    parts = line.split(':')[1].split(',')[0].strip()
                    # Remove any comments
                    if '#' in parts:
                        parts = parts.split('#')[0].strip()
                    current_scaling[(min_agi, max_agi)] = float(parts)
                    break
    
    print("\nApplying adjustments...")
    for (bracket, adjustment_factor) in adjustments_needed:
        current_factor = current_scaling.get(bracket, 1.0)
        new_factor = current_factor * adjustment_factor
        
        # Apply some damping to avoid overshooting
        damping = 0.7  # Only apply 70% of the adjustment
        new_factor = current_factor + (new_factor - current_factor) * damping
        
        bracket_label = f"${bracket[0]//1000}k+" if bracket[1] == float('inf') else f"${bracket[0]//1000}k-${bracket[1]//1000}k"
        print(f"  {bracket_label}: {current_factor:.3f} → {new_factor:.3f} (adjustment: {adjustment_factor:.3f})")
        
        update_scaling_factor(bracket, new_factor)
    
    print("\nRegenerating tax units with new scaling factors...")
    new_results = regenerate_and_get_results()
    
    print("\nNew Results vs Targets:")
    print("-" * 80)
    print(f"{'Bracket':<15} {'New':<12} {'Target':<12} {'Error %':<12} {'Status':<10}")
    print("-" * 80)
    
    total_new = 0
    total_target = 0
    all_within_1pct = True
    
    for (min_agi, max_agi), target in TABLE_21_TARGETS.items():
        new_amount = new_results[(min_agi, max_agi)]
        
        if target > 0:
            error_pct = abs(new_amount - target) / target * 100
        else:
            error_pct = 0 if new_amount < 0.1 else 999
        
        status = "✅" if error_pct <= 1.0 else "❌"
        if error_pct > 1.0:
            all_within_1pct = False
        
        bracket_label = f"${min_agi//1000}k+" if max_agi == float('inf') else f"${min_agi//1000}k-${max_agi//1000}k"
        print(f"{bracket_label:<15} ${new_amount:<11.1f} ${target:<11.1f} {error_pct:<11.1f}% {status:<10}")
        
        total_new += new_amount
        total_target += target
    
    total_error_pct = abs(total_new - total_target) / total_target * 100
    total_status = "✅" if total_error_pct <= 1.0 else "❌"
    print("-" * 80)
    print(f"{'TOTAL':<15} ${total_new:<11.1f} ${total_target:<11.1f} {total_error_pct:<11.1f}% {total_status:<10}")
    
    if all_within_1pct and total_error_pct <= 1.0:
        print(f"\n🎯 SUCCESS: All brackets within ±1% and total within ±1%!")
    else:
        print(f"\n⚠️  Some brackets still outside ±1%. Run script again for further refinement.")
        print(f"   (This is an iterative process - may need 2-3 runs to converge)")
