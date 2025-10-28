#!/usr/bin/env python3
"""
Ultra fine-tune capital gains to get even closer to Table 21 targets.
"""

# Latest results from diagnostic
CURRENT_AMOUNTS = {
    (0, 10000): 0.0,
    (10000, 20000): 0.0,
    (20000, 30000): 1.0,
    (30000, 40000): 3.9,
    (40000, 50000): 5.1,
    (50000, 75000): 29.8,
    (75000, 100000): 46.9,
    (100000, 150000): 100.4,
    (150000, 200000): 120.3,
    (200000, 300000): 228.9,
    (300000, 400000): 248.3,
    (400000, float('inf')): 2169.7,
}

# Table 21 targets
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
    (0, 10000): 186.3,
    (10000, 20000): 1.0,
    (20000, 30000): 665.0,
    (30000, 40000): 33.8,
    (40000, 50000): 20.0,
    (50000, 75000): 13.3,
    (75000, 100000): 8.6,
    (100000, 150000): 6.6,
    (150000, 200000): 4.6,
    (200000, 300000): 4.0,
    (300000, 400000): 3.0,
    (400000, float('inf')): 5.8,
}

print("="*80)
print("ULTRA FINE-TUNING FOR MAXIMUM ACCURACY")
print("="*80)

print("\nCurrent vs Target Analysis:")
print("-" * 80)
print(f"{'Bracket':<15} {'Current':<10} {'Target':<10} {'Error %':<10} {'Adj Factor':<12}")
print("-" * 80)

total_current = sum(CURRENT_AMOUNTS.values())
total_target = sum(TARGET_AMOUNTS.values())

# Calculate ultra-precise adjustment factors
new_scaling = {}
for bracket in CURRENT_AMOUNTS.keys():
    current = CURRENT_AMOUNTS[bracket]
    target = TARGET_AMOUNTS[bracket]
    current_scale = CURRENT_SCALING[bracket]
    
    if target > 0:
        error_pct = (current - target) / target * 100
        # Calculate exact adjustment needed
        adjustment_factor = target / current if current > 0 else 1.0
        
        # For very small errors, apply smaller adjustments to avoid overshooting
        if abs(error_pct) < 20:
            adjustment_factor = 1.0 + (adjustment_factor - 1.0) * 0.8  # 80% of adjustment
        
    else:
        error_pct = 0 if current < 0.1 else 999
        # For zero targets, set to very small value if current is positive
        adjustment_factor = 0.01 if current > 0.1 else 1.0
    
    # Special handling for problematic brackets
    if bracket == (0, 10000) and current < 0.1:
        # If current is near zero but target is 0.5, increase significantly
        adjustment_factor = 10.0
    elif bracket == (20000, 30000) and current > target:
        # Reduce the over-allocation more aggressively
        adjustment_factor = target / current * 0.5  # More aggressive reduction
    
    # Apply the adjustment to current scaling
    new_scale = current_scale * adjustment_factor
    
    # Apply bounds to keep reasonable
    new_scale = max(0.01, min(new_scale, 10000.0))
    
    bracket_label = f"${bracket[0]//1000}k+" if bracket[1] == float('inf') else f"${bracket[0]//1000}k-${bracket[1]//1000}k"
    status = "✅" if abs(error_pct) <= 5 else "⚠️" if abs(error_pct) <= 15 else "❌"
    
    print(f"{bracket_label:<15} ${current:<9.1f} ${target:<9.1f} {error_pct:<9.1f}% {adjustment_factor:<11.3f} {status}")
    
    new_scaling[bracket] = new_scale

print("-" * 80)
print(f"{'TOTAL':<15} ${total_current:<9.1f} ${total_target:<9.1f} {(total_current-total_target)/total_target*100:<9.1f}%")

print("\n" + "="*80)
print("ULTRA-PRECISE SCALING FACTOR ADJUSTMENTS")
print("="*80)
print(f"{'Bracket':<15} {'Old Scale':<12} {'New Scale':<12} {'Change %':<12}")
print("-" * 80)

for bracket in CURRENT_SCALING.keys():
    old_scale = CURRENT_SCALING[bracket]
    new_scale = new_scaling[bracket]
    change_pct = (new_scale - old_scale) / old_scale * 100 if old_scale > 0 else 0
    
    bracket_label = f"${bracket[0]//1000}k+" if bracket[1] == float('inf') else f"${bracket[0]//1000}k-${bracket[1]//1000}k"
    change_icon = "📈" if change_pct > 5 else "📉" if change_pct < -5 else "➡️"
    
    print(f"{bracket_label:<15} {old_scale:<11.1f} {new_scale:<11.1f} {change_pct:<11.1f}% {change_icon}")

print("\n" + "="*80)
print("ULTRA-OPTIMIZED SCALING FACTORS FOR capital_gains.py")
print("="*80)

print("\nself.bracket_scaling = {")
for bracket, scaling in new_scaling.items():
    target = TARGET_AMOUNTS[bracket]
    if bracket[1] == float('inf'):
        print(f"    ({int(bracket[0])}, float('inf')): {scaling:.1f},  # ${target:.1f}M target")
    else:
        print(f"    ({int(bracket[0])}, {int(bracket[1])}): {scaling:.1f},  # ${target:.1f}M target")
print("}")

# Calculate expected total after adjustments
expected_total = 0
for bracket in CURRENT_AMOUNTS.keys():
    current = CURRENT_AMOUNTS[bracket]
    target = TARGET_AMOUNTS[bracket]
    adjustment = new_scaling[bracket] / CURRENT_SCALING[bracket] if CURRENT_SCALING[bracket] > 0 else 1
    expected_amount = current * adjustment
    expected_total += expected_amount

print(f"\n🎯 Expected Results:")
print(f"   - Total amount: ~${expected_total:.0f}M (target: ${total_target:.0f}M)")
print(f"   - Expected error: ~{(expected_total-total_target)/total_target*100:.1f}%")
print(f"   - Should achieve <±1% total accuracy")
print(f"   - Most brackets should be within ±5% of targets")
