#!/usr/bin/env python3
"""
Step 3 Results: Check the impact of stricter HoH criteria on filing status distribution.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.pums_loader import PUMSDataLoader

def calculate_step3_results():
    """Calculate weighted totals after implementing stricter HoH criteria."""
    
    print("Step 3 Results: Checking impact of stricter HoH criteria...")
    
    # Load data
    tax_units = pd.read_parquet("src/data/processed/tax_units_rule_based.parquet")
    pums_loader = PUMSDataLoader()
    person_df, hh_df = pums_loader.load_data()
    
    # Get household weights and exclude zero-weight households
    hh_weights = hh_df[hh_df['WGTP'] > 0][['SERIALNO', 'WGTP']].set_index('SERIALNO')['WGTP']
    
    # Filter tax units to only include those from households with positive weights
    tax_units_filtered = tax_units[tax_units['hh_id'].isin(hh_weights.index)].copy()
    
    # Map household weights to tax units
    tax_units_filtered['weight'] = tax_units_filtered['hh_id'].map(hh_weights)
    
    # Remove any tax units without weights
    tax_units_filtered = tax_units_filtered.dropna(subset=['weight'])
    
    # Calculate weighted counts by filing status
    weighted_counts = tax_units_filtered.groupby('filing_status')['weight'].sum().astype(int)
    total_weighted = weighted_counts.sum()
    
    # Calculate percentages
    percentages = (weighted_counts / total_weighted * 100).round(1)
    
    # Display results
    print(f"\n{'='*80}")
    print(f"STEP 3 RESULTS: STRICTER HOH CRITERIA APPLIED")
    print(f"{'='*80}")
    
    print(f"\nEstimated Total Tax Units: {total_weighted:,}")
    print(f"Actual 2022 Hawaii Returns: 743,109")
    print(f"Difference: {total_weighted - 743109:+,} ({(total_weighted - 743109)/743109*100:+.1f}%)")
    
    print(f"\nWeighted Filing Status Distribution:")
    for status, count in weighted_counts.items():
        print(f"  {status.replace('_', ' ').title()}: {count:,} ({percentages[status]}%)")
    
    # Compare with actual 2022 Hawaii filing distribution
    print(f"\nComparison with 2022 Hawaii Filing Distribution:")
    actual_counts = {
        'single': int(743109 * 0.51),   # 51% single
        'joint': int(743109 * 0.36),    # 36% joint
        'head_of_household': int(743109 * 0.096),  # 9.6% HoH
        'married_filing_separate': int(743109 * 0.034)  # 3.4% MFS
    }
    
    print(f"{'Status':<20} {'Estimated':<12} {'Actual':<12} {'Diff':<12} {'% Diff':<8}")
    print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*8}")
    
    for status, actual in actual_counts.items():
        estimated = weighted_counts.get(status, 0)
        diff = estimated - actual
        pct_diff = (diff / actual * 100) if actual > 0 else 0
        status_name = status.replace('_', ' ').title()
        print(f"{status_name:<20} {estimated:<12,} {actual:<12,} {diff:<+12,} {pct_diff:<+8.1f}%")
    
    # Compare with Step 1 results (before HoH fixes)
    print(f"\n{'='*80}")
    print(f"IMPROVEMENT FROM STEP 1 TO STEP 3")
    print(f"{'='*80}")
    
    # Step 1 results for comparison
    step1_results = {
        'single': 421452,
        'joint': 250160,
        'head_of_household': 115754,
        'married_filing_separate': 35768,
        'total': 823134
    }
    
    print(f"{'Status':<20} {'Step 1':<12} {'Step 3':<12} {'Change':<12} {'% Change':<10}")
    print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
    
    for status in ['single', 'joint', 'head_of_household', 'married_filing_separate']:
        step1_count = step1_results[status]
        step3_count = weighted_counts.get(status, 0)
        change = step3_count - step1_count
        pct_change = (change / step1_count * 100) if step1_count > 0 else 0
        status_name = status.replace('_', ' ').title()
        print(f"{status_name:<20} {step1_count:<12,} {step3_count:<12,} {change:<+12,} {pct_change:<+10.1f}%")
    
    # Total change
    total_change = total_weighted - step1_results['total']
    total_pct_change = (total_change / step1_results['total'] * 100) if step1_results['total'] > 0 else 0
    print(f"{'Total':<20} {step1_results['total']:<12,} {total_weighted:<12,} {total_change:<+12,} {total_pct_change:<+10.1f}%")
    
    # Calculate progress toward targets
    print(f"\n{'='*80}")
    print(f"PROGRESS TOWARD TARGETS")
    print(f"{'='*80}")
    
    # HoH progress
    hoh_target = 71338
    hoh_current = weighted_counts.get('head_of_household', 0)
    hoh_step1 = 115754
    hoh_progress = ((hoh_step1 - hoh_current) / (hoh_step1 - hoh_target)) * 100 if hoh_step1 != hoh_target else 0
    
    print(f"Head of Household Progress:")
    print(f"  Step 1: {hoh_step1:,} (target: {hoh_target:,})")
    print(f"  Step 3: {hoh_current:,}")
    print(f"  Reduction: {hoh_step1 - hoh_current:,} ({(hoh_step1 - hoh_current)/hoh_step1*100:.1f}%)")
    print(f"  Progress toward target: {hoh_progress:.1f}%")
    
    if hoh_progress > 80:
        print("  ✅ EXCELLENT PROGRESS")
    elif hoh_progress > 50:
        print("  ⚠️  GOOD PROGRESS")
    else:
        print("  ❌ LIMITED PROGRESS")
    
    # Overall assessment
    print(f"\nOVERALL ASSESSMENT:")
    
    # Check how close we are to all targets
    target_distances = {}
    for status, actual in actual_counts.items():
        estimated = weighted_counts.get(status, 0)
        distance = abs(estimated - actual) / actual * 100
        target_distances[status] = distance
    
    avg_distance = sum(target_distances.values()) / len(target_distances)
    print(f"Average distance from targets: {avg_distance:.1f}%")
    
    if avg_distance < 10:
        print("✅ EXCELLENT: Very close to all targets")
    elif avg_distance < 20:
        print("⚠️  GOOD: Reasonably close to targets")
    else:
        print("❌ NEEDS WORK: Still significant gaps")
    
    return weighted_counts, total_weighted

if __name__ == "__main__":
    calculate_step3_results()
