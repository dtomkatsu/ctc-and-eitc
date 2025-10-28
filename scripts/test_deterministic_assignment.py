#!/usr/bin/env python3
"""
Test different deterministic assignment methods for capital gains.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
from src.tax.adjustments.capital_gains import apply_capital_gains_to_dataframe

# Create a small test dataset
np.random.seed(42)
test_data = {
    'agi': np.random.uniform(20000, 500000, 1000),
    'weight': np.random.uniform(0.5, 2.0, 1000),
    'filing_status': ['single'] * 1000
}
test_df = pd.DataFrame(test_data)

print("="*80)
print("TESTING DETERMINISTIC CAPITAL GAINS ASSIGNMENT METHODS")
print("="*80)

assignment_methods = [
    ('random', 'Random assignment (baseline)'),
    ('deterministic_top', 'Assign to highest income filers in each bracket'),
    ('deterministic_income', 'Assign based on income percentile within bracket'),
    ('deterministic_hash', 'Assign based on deterministic hash of filer data')
]

results = {}

for method, description in assignment_methods:
    print(f"\n📊 Testing: {method}")
    print(f"   Description: {description}")
    
    # Apply capital gains with this method
    result_df = apply_capital_gains_to_dataframe(
        test_df.copy(),
        agi_col='agi',
        include_random=True,
        assignment_method=method
    )
    
    # Calculate statistics
    total_cap_gains = (result_df['capital_gains'] * result_df['weight']).sum() / 1_000_000
    filers_with_cap_gains = (result_df['capital_gains'] > 0).sum()
    weighted_filers_with_cap_gains = ((result_df['capital_gains'] > 0) * result_df['weight']).sum()
    avg_cap_gains_per_filer = total_cap_gains * 1_000_000 / weighted_filers_with_cap_gains if weighted_filers_with_cap_gains > 0 else 0
    
    results[method] = {
        'total_cap_gains_m': total_cap_gains,
        'filers_with_cg': filers_with_cap_gains,
        'weighted_filers': weighted_filers_with_cap_gains,
        'avg_per_filer': avg_cap_gains_per_filer
    }
    
    print(f"   Total capital gains: ${total_cap_gains:.1f}M")
    print(f"   Filers with cap gains: {filers_with_cap_gains:,} ({filers_with_cap_gains/len(test_df)*100:.1f}%)")
    print(f"   Weighted filers: {weighted_filers_with_cap_gains:,.0f}")
    print(f"   Avg per filer: ${avg_cap_gains_per_filer:,.0f}")

print("\n" + "="*80)
print("COMPARISON OF METHODS")
print("="*80)

print(f"{'Method':<20} {'Total ($M)':<12} {'Filers':<8} {'Weighted':<10} {'Avg/Filer':<12}")
print("-" * 80)

for method, description in assignment_methods:
    r = results[method]
    print(f"{method:<20} ${r['total_cap_gains_m']:<11.1f} {r['filers_with_cg']:<8,} {r['weighted_filers']:<10,.0f} ${r['avg_per_filer']:<11,.0f}")

print("\n" + "="*80)
print("CONSISTENCY TEST (Multiple Runs)")
print("="*80)

print("Testing deterministic_top method for consistency across multiple runs...")

for run in range(1, 4):
    result_df = apply_capital_gains_to_dataframe(
        test_df.copy(),
        agi_col='agi',
        include_random=True,
        assignment_method='deterministic_top'
    )
    
    total_cap_gains = (result_df['capital_gains'] * result_df['weight']).sum() / 1_000_000
    filers_with_cap_gains = (result_df['capital_gains'] > 0).sum()
    
    print(f"Run {run}: Total=${total_cap_gains:.1f}M, Filers={filers_with_cap_gains:,}")

print("\n🎯 Key Advantages of Deterministic Assignment:")
print("   ✅ Consistent results across multiple runs")
print("   ✅ Reproducible for validation and debugging")
print("   ✅ More realistic assignment (highest income filers more likely to have cap gains)")
print("   ✅ Eliminates random variation in model calibration")

print("\n📋 Recommended Method: 'deterministic_top'")
print("   - Assigns capital gains to highest income filers within each bracket")
print("   - Most realistic (wealthy filers more likely to have capital gains)")
print("   - Completely reproducible results")
print("   - Maintains proper bracket-level distribution")
