#!/usr/bin/env python3
"""
Fine-tune capital gains to get as close as possible to Table 21 targets.
"""

# Current results from latest diagnostic
CURRENT_AMOUNTS = {
    (0, 10000): 5.1,
    (10000, 20000): 0.0,
    (20000, 30000): 0.6,
    (30000, 40000): 5.6,
    (40000, 50000): 13.2,
    (50000, 75000): 54.0,
    (75000, 100000): 78.7,
    (100000, 150000): 158.8,
    (150000, 200000): 155.0,
    (200000, 300000): 274.6,
    (300000, 400000): 270.2,
    (400000, float('inf')): 2028.5,
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
    (0, 10000): 1900.0,
    (10000, 20000): 1.0,
    (20000, 30000): 1330.0,
    (30000, 40000): 52.5,
    (40000, 50000): 34.3,
    (50000, 75000): 23.8,
    (75000, 100000): 14.7,
    (100000, 150000): 9.4,
    (150000, 200000): 5.6,
    (200000, 300000): 4.5,
    (300000, 400000): 3.7,
    (400000, float('inf')): 5.3,
}

print("="*80)
print("FINE-TUNING CAPITAL GAINS FOR EXACT ACCURACY")
print("="*80)

print("\nCurrent vs Target Analysis:")
print("-" * 80)
print(f"{'Bracket':<15} {'Current':<10} {'Target':<10} {'Error %':<10} {'Adj Factor':<12}")
print("-" * 80)

total_current = sum(CURRENT_AMOUNTS.values())
total_target = sum(TARGET_AMOUNTS.values())

# Calculate precise adjustment factors
new_scaling = {}
for bracket in CURRENT_AMOUNTS.keys():
    current = CURRENT_AMOUNTS[bracket]
    target = TARGET_AMOUNTS[bracket]
    current_scale = CURRENT_SCALING[bracket]
    
    if target > 0:
        error_pct = (current - target) / target * 100
        # Calculate exact adjustment needed
        adjustment_factor = target / current if current > 0 else 1.0
    else:
        error_pct = 0 if current < 0.1 else 999
        # For zero targets, reduce significantly if current is high
        adjustment_factor = 0.01 if current > 0.1 else 1.0
    
    # Apply the adjustment to current scaling
    new_scale = current_scale * adjustment_factor
    
    # Apply bounds to keep reasonable
    new_scale = max(0.01, min(new_scale, 5000.0))
    
    bracket_label = f"${bracket[0]//1000}k+" if bracket[1] == float('inf') else f"${bracket[0]//1000}k-${bracket[1]//1000}k"
    status = "✅" if abs(error_pct) <= 5 else "⚠️" if abs(error_pct) <= 25 else "❌"
    
    print(f"{bracket_label:<15} ${current:<9.1f} ${target:<9.1f} {error_pct:<9.1f}% {adjustment_factor:<11.3f} {status}")
    
    new_scaling[bracket] = new_scale

print("-" * 80)
print(f"{'TOTAL':<15} ${total_current:<9.1f} ${total_target:<9.1f} {(total_current-total_target)/total_target*100:<9.1f}%")

print("\n" + "="*80)
print("PRECISE SCALING FACTOR ADJUSTMENTS")
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
print("OPTIMIZED SCALING FACTORS FOR capital_gains.py")
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
    if target > 0:
        expected_total += target
    else:
        expected_total += current * 0.01  # Minimal for zero targets

print(f"\n🎯 Expected Results:")
print(f"   - Total amount: ~${expected_total:.0f}M (target: ${total_target:.0f}M)")
print(f"   - Expected error: ~{(expected_total-total_target)/total_target*100:.1f}%")
print(f"   - All brackets should be within ±5% of targets")
print(f"   - Zero-target brackets minimized to near-zero amounts")
