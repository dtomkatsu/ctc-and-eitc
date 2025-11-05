#!/usr/bin/env python3
"""Display Act 46 (2024) tax bracket structure."""

import pandas as pd

# 2024 Act 46 tax brackets
brackets_data = []

# Joint/Surviving Spouse
joint_brackets = [
    (0, 19200, 1.4),
    (19200, 28800, 3.2),
    (28800, 38400, 5.5),
    (38400, 48000, 6.4),
    (48000, 72000, 6.8),
    (72000, 96000, 7.2),
    (96000, 250000, 7.6),
    (250000, 350000, 7.9),
    (350000, 450000, 8.25),
    (450000, 550000, 9.0),
    (550000, 650000, 10.0),
    (650000, float('inf'), 11.0),
]

for min_inc, max_inc, rate in joint_brackets:
    max_str = f"${max_inc:,.0f}" if max_inc != float('inf') else "No limit"
    brackets_data.append({
        'Filing Status': 'Joint/Surviving Spouse',
        'Income Min': f"${min_inc:,.0f}",
        'Income Max': max_str,
        'Rate (%)': rate
    })

# Head of Household
hoh_brackets = [
    (0, 14400, 1.4),
    (14400, 21600, 3.2),
    (21600, 28800, 5.5),
    (28800, 36000, 6.4),
    (36000, 54000, 6.8),
    (54000, 72000, 7.2),
    (72000, 187500, 7.6),
    (187500, 262500, 7.9),
    (262500, 337500, 8.25),
    (337500, 412500, 9.0),
    (412500, 487500, 10.0),
    (487500, float('inf'), 11.0),
]

for min_inc, max_inc, rate in hoh_brackets:
    max_str = f"${max_inc:,.0f}" if max_inc != float('inf') else "No limit"
    brackets_data.append({
        'Filing Status': 'Head of Household',
        'Income Min': f"${min_inc:,.0f}",
        'Income Max': max_str,
        'Rate (%)': rate
    })

# Single/MFS
single_brackets = [
    (0, 9600, 1.4),
    (9600, 14400, 3.2),
    (14400, 19200, 5.5),
    (19200, 24000, 6.4),
    (24000, 36000, 6.8),
    (36000, 48000, 7.2),
    (48000, 125000, 7.6),
    (125000, 175000, 7.9),
    (175000, 225000, 8.25),
    (225000, 275000, 9.0),
    (275000, 325000, 10.0),
    (325000, float('inf'), 11.0),
]

for min_inc, max_inc, rate in single_brackets:
    max_str = f"${max_inc:,.0f}" if max_inc != float('inf') else "No limit"
    brackets_data.append({
        'Filing Status': 'Single/MFS',
        'Income Min': f"${min_inc:,.0f}",
        'Income Max': max_str,
        'Rate (%)': rate
    })

df = pd.DataFrame(brackets_data)

print("\n" + "="*90)
print("ACT 46 (2024) TAX BRACKETS AND RATES")
print("Applied to 2026 tax year with 2025 standard deductions and personal exemptions")
print("="*90)

print("\n2025 STANDARD DEDUCTIONS:")
print("  Joint/Surviving Spouse: $12,400")
print("  Head of Household: $9,212")
print("  Single/MFS: $6,200")

print("\n2025 PERSONAL EXEMPTIONS:")
print("  $1,200 per person (filer + dependents)")

print("\n" + "="*90)
print("TAX RATE SCHEDULE")
print("="*90)

# Group by filing status
for status in ['Joint/Surviving Spouse', 'Head of Household', 'Single/MFS']:
    status_df = df[df['Filing Status'] == status]
    print(f"\n{status}:")
    print("-" * 90)
    for _, row in status_df.iterrows():
        print(f"  {row['Income Min']:>15} to {row['Income Max']:>15}  →  {row['Rate (%)']:>5}%")

print("\n" + "="*90)
print("KEY FEATURES:")
print("  • Top marginal rate: 11.0% (starts at $650k Joint, $487.5k HoH, $325k Single)")
print("  • 12 brackets for each filing status")
print("  • Progressive rate structure from 1.4% to 11.0%")
print("="*90 + "\n")

# Save to CSV
output_df = pd.DataFrame(brackets_data)
output_df.to_csv('data/processed/projections/act46_2024_brackets_complete.csv', index=False)
print("Saved complete bracket table to: data/processed/projections/act46_2024_brackets_complete.csv\n")
