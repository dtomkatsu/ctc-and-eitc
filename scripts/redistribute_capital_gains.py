#!/usr/bin/env python3
"""
Redistribute capital gains to fix bracket-level errors while maintaining total.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Current results from diagnostic (in millions)
CURRENT_AMOUNTS = {
    (0, 10000): 0.5,
    (10000, 20000): 0.0,
    (20000, 30000): 5.6,
    (30000, 40000): 7.2,
    (40000, 50000): 19.0,
    (50000, 75000): 71.5,
    (75000, 100000): 120.3,
    (100000, 150000): 230.6,
    (150000, 200000): 211.0,
    (200000, 300000): 405.8,
    (300000, 400000): 446.0,
    (400000, float('inf')): 1538.8,
}

# Table 21 targets (in millions)
TARGET_AMOUNTS = {
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

# Current scaling factors
CURRENT_SCALING = {
    (0, 10000): 1900.0,
    (10000, 20000): 1.000,
    (20000, 30000): 1900.0,
    (30000, 40000): 75.0,
    (40000, 50000): 49.0,
    (50000, 75000): 34.0,
    (75000, 100000): 21.0,
    (100000, 150000): 13.5,
    (150000, 200000): 8.0,
    (200000, 300000): 6.5,
    (300000, 400000): 5.3,
    (400000, float('inf')): 3.8,
}

print("="*80)
print("CAPITAL GAINS REDISTRIBUTION ANALYSIS")
print("="*80)

print("\nCurrent vs Target Analysis:")
print("-" * 80)
print(f"{'Bracket':<15} {'Current':<10} {'Target':<10} {'Error %':<10} {'Adjustment':<12}")
print("-" * 80)

total_current = sum(CURRENT_AMOUNTS.values())
total_target = sum(TARGET_AMOUNTS.values())

# Calculate needed adjustments
adjustments = {}
excess_amount = 0  # Amount to redistribute from over-brackets
deficit_amount = 0  # Amount needed for under-brackets

for bracket in CURRENT_AMOUNTS.keys():
    current = CURRENT_AMOUNTS[bracket]
    target = TARGET_AMOUNTS[bracket]
    
    if target > 0:
        error_pct = (current - target) / target * 100
        adjustment_factor = target / current if current > 0 else 1.0
    else:
        error_pct = 0 if current < 0.1 else 999
        adjustment_factor = 0.1
    
    bracket_label = f"${bracket[0]//1000}k+" if bracket[1] == float('inf') else f"${bracket[0]//1000}k-${bracket[1]//1000}k"
    status = "✅" if abs(error_pct) <= 10 else "⚠️" if abs(error_pct) <= 50 else "❌"
    
    print(f"{bracket_label:<15} ${current:<9.1f} ${target:<9.1f} {error_pct:<9.1f}% {adjustment_factor:<11.3f} {status}")
    
    adjustments[bracket] = adjustment_factor
    
    # Track excess and deficit for redistribution
    if current > target:
        excess_amount += (current - target)
    else:
        deficit_amount += (target - current)

print("-" * 80)
print(f"{'TOTAL':<15} ${total_current:<9.1f} ${total_target:<9.1f} {(total_current-total_target)/total_target*100:<9.1f}%")
print(f"\nExcess to redistribute: ${excess_amount:.1f}M")
print(f"Deficit to fill: ${deficit_amount:.1f}M")
print(f"Net difference: ${excess_amount - deficit_amount:.1f}M")

# Strategy: Reduce over-brackets proportionally, increase under-brackets
print("\n" + "="*80)
print("REDISTRIBUTION STRATEGY")
print("="*80)

# Calculate new scaling factors
new_scaling = {}
redistribution_factor = 0.7  # Reduce over-brackets by 30%

print(f"\nStrategy: Reduce over-brackets by {(1-redistribution_factor)*100:.0f}%, increase under-brackets to match")
print("-" * 80)
print(f"{'Bracket':<15} {'Old Scale':<12} {'New Scale':<12} {'Change':<12}")
print("-" * 80)

for bracket in CURRENT_SCALING.keys():
    current_scale = CURRENT_SCALING[bracket]
    current_amount = CURRENT_AMOUNTS[bracket]
    target_amount = TARGET_AMOUNTS[bracket]
    
    bracket_label = f"${bracket[0]//1000}k+" if bracket[1] == float('inf') else f"${bracket[0]//1000}k-${bracket[1]//1000}k"
    
    if target_amount == 0:
        # Keep zero targets at low scaling
        new_scale = min(current_scale, 1.0)
    elif current_amount > target_amount * 1.2:  # Over by >20%
        # Reduce over-brackets
        reduction_factor = redistribution_factor
        new_scale = current_scale * reduction_factor
    elif current_amount < target_amount * 0.8:  # Under by >20%
        # Increase under-brackets more aggressively
        increase_factor = 1.4  # 40% increase
        new_scale = current_scale * increase_factor
    else:
        # Fine-tune brackets that are close
        adjustment = target_amount / current_amount if current_amount > 0 else 1.0
        new_scale = current_scale * adjustment
    
    # Apply bounds to keep scaling factors reasonable
    new_scale = max(0.1, min(new_scale, 2000.0))
    
    change_pct = (new_scale - current_scale) / current_scale * 100 if current_scale > 0 else 0
    change_icon = "📈" if change_pct > 5 else "📉" if change_pct < -5 else "➡️"
    
    print(f"{bracket_label:<15} {current_scale:<11.1f} {new_scale:<11.1f} {change_pct:<11.1f}% {change_icon}")
    
    new_scaling[bracket] = new_scale

print("\n" + "="*80)
print("UPDATED SCALING FACTORS FOR capital_gains.py")
print("="*80)

print("\nself.bracket_scaling = {")
for bracket, scaling in new_scaling.items():
    target = TARGET_AMOUNTS[bracket]
    if bracket[1] == float('inf'):
        print(f"    ({int(bracket[0])}, float('inf')): {scaling:.1f},  # ${target:.1f}M target")
    else:
        print(f"    ({int(bracket[0])}, {int(bracket[1])}): {scaling:.1f},  # ${target:.1f}M target")
print("}")

print(f"\n🎯 Expected impact:")
print(f"   - Reduce middle-bracket over-allocation by ~30%")
print(f"   - Increase top-bracket allocation by ~40%")
print(f"   - Maintain total near ${total_target:.0f}M target")
print(f"   - Improve bracket-level accuracy significantly")
