#!/usr/bin/env python3
"""
Step 1: Fix zero-weight household handling in tax unit weighting.
Excludes zero-weight households (group quarters) per Census PUMS documentation.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.pums_loader import PUMSDataLoader

def calculate_weighted_totals_step1():
    """Calculate weighted tax unit totals with proper zero-weight handling."""
    
    print("Step 1: Fixing zero-weight household handling...")
    
    # Load data
    tax_units = pd.read_parquet("src/data/processed/tax_units_rule_based.parquet")
    pums_loader = PUMSDataLoader()
    person_df, hh_df = pums_loader.load_data()
    
    # Get household weights and exclude zero-weight households
    print(f"Total households before filtering: {len(hh_df):,}")
    hh_weights_all = hh_df[['SERIALNO', 'WGTP']].set_index('SERIALNO')['WGTP']
    
    # Filter out zero-weight households (group quarters)
    hh_weights = hh_weights_all[hh_weights_all > 0]
    zero_weight_count = (hh_weights_all == 0).sum()
    
    print(f"Zero-weight households excluded: {zero_weight_count:,} ({zero_weight_count/len(hh_df)*100:.1f}%)")
    print(f"Households with positive weights: {len(hh_weights):,}")
    
    # Filter tax units to only include those from households with positive weights
    tax_units_filtered = tax_units[tax_units['hh_id'].isin(hh_weights.index)].copy()
    
    print(f"Tax units before filtering: {len(tax_units):,}")
    print(f"Tax units after filtering: {len(tax_units_filtered):,}")
    print(f"Tax units excluded: {len(tax_units) - len(tax_units_filtered):,}")
    
    # Map household weights to tax units
    tax_units_filtered['weight'] = tax_units_filtered['hh_id'].map(hh_weights)
    
    # Check for any tax units without weights (shouldn't happen now)
    missing_weights = tax_units_filtered['weight'].isna().sum()
    if missing_weights > 0:
        print(f"Warning: {missing_weights:,} tax units still missing weights")
        # Remove these rather than filling
        tax_units_filtered = tax_units_filtered.dropna(subset=['weight'])
    
    # Calculate weighted counts by filing status
    weighted_counts = tax_units_filtered.groupby('filing_status')['weight'].sum().astype(int)
    total_weighted = weighted_counts.sum()
    
    # Calculate percentages
    percentages = (weighted_counts / total_weighted * 100).round(1)
    
    # Display results
    print(f"\n{'='*80}")
    print(f"STEP 1 RESULTS: ZERO-WEIGHT HOUSEHOLDS EXCLUDED")
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
    
    # Calculate scaling factor to match actual returns
    scaling_factor = 743109 / total_weighted if total_weighted > 0 else 0
    print(f"\nScaling factor to match actual returns: {scaling_factor:.4f}")
    
    # Summary of improvements
    print(f"\n{'='*80}")
    print(f"STEP 1 IMPACT SUMMARY")
    print(f"{'='*80}")
    
    # Compare with previous results (from the corrected script)
    previous_total = 823134  # From previous run
    improvement = previous_total - total_weighted
    
    print(f"Previous total estimate: {previous_total:,}")
    print(f"New total estimate: {total_weighted:,}")
    print(f"Reduction: {improvement:,} ({improvement/previous_total*100:.1f}%)")
    print(f"Distance from actual: {abs(total_weighted - 743109):,} (was {abs(previous_total - 743109):,})")
    
    if abs(total_weighted - 743109) < abs(previous_total - 743109):
        print("✅ IMPROVEMENT: Closer to actual tax return count!")
    else:
        print("⚠️  Still need further refinement")
    
    return tax_units_filtered, weighted_counts, total_weighted

if __name__ == "__main__":
    calculate_weighted_totals_step1()
