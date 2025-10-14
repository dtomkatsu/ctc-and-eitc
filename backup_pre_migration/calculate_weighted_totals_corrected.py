#!/usr/bin/env python3
"""
Calculate properly weighted tax unit totals using Census PUMS weights.
Applies household weights (WGTP) to scale tax unit counts to population estimates.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.pums_loader import PUMSDataLoader

def calculate_weighted_totals():
    """Calculate weighted tax unit totals using proper PUMS weighting methodology."""
    
    print("Calculating weighted tax unit totals with proper PUMS weighting...")
    
    # Load data
    tax_units = pd.read_parquet("src/data/processed/tax_units_rule_based.parquet")
    pums_loader = PUMSDataLoader()
    person_df, hh_df = pums_loader.load_data()
    
    # Get household weights
    hh_weights = hh_df[['SERIALNO', 'WGTP']].set_index('SERIALNO')['WGTP']
    
    # Map household weights to tax units using hh_id (which should match SERIALNO)
    tax_units['weight'] = tax_units['hh_id'].map(hh_weights)
    
    # Check for any tax units without weights (shouldn't happen if hh_id matches SERIALNO)
    missing_weights = tax_units['weight'].isna().sum()
    if missing_weights > 0:
        print(f"Warning: {missing_weights:,} tax units could not be matched to household weights")
        # For now, fill missing weights with the mean weight
        mean_weight = hh_weights.mean()
        tax_units['weight'] = tax_units['weight'].fillna(mean_weight)
    
    # Calculate weighted counts by filing status
    weighted_counts = tax_units.groupby('filing_status')['weight'].sum().astype(int)
    total_weighted = weighted_counts.sum()
    
    # Calculate percentages
    percentages = (weighted_counts / total_weighted * 100).round(1)
    
    # Display results
    print(f"\n{'='*80}")
    print(f"PROPERLY WEIGHTED TAX UNIT TOTALS")
    print(f"{'='*80}")
    
    print(f"\nEstimated Total Tax Units: {total_weighted:,}")
    print(f"Actual 2022 Hawaii Returns: 743,109")
    
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
    
    for status, actual in actual_counts.items():
        estimated = weighted_counts.get(status, 0)
        diff = estimated - actual
        pct_diff = (diff / actual * 100) if actual > 0 else 0
        print(f"  {status.replace('_', ' ').title()}:")
        print(f"    Estimated: {estimated:,}")
        print(f"    Actual:    {actual:,}")
        print(f"    Difference: {diff:+,} ({pct_diff:+.1f}%)")
    
    # Calculate scaling factor to match actual returns
    scaling_factor = 743109 / total_weighted if total_weighted > 0 else 0
    print(f"\nScaling factor to match actual returns: {scaling_factor:.4f}")
    
    # Calculate scaled counts
    print(f"\nScaled to Match 743,109 Returns:")
    for status, count in weighted_counts.items():
        scaled = int(count * scaling_factor)
        print(f"  {status.replace('_', ' ').title()}: {scaled:,} "
              f"({percentages[status]}% of total)")
    
    # Check household weights distribution
    print(f"\nHousehold Weight Statistics:")
    print(f"  Min weight: {hh_weights.min()}")
    print(f"  Max weight: {hh_weights.max()}")
    print(f"  Mean weight: {hh_weights.mean():.2f}")
    print(f"  Median weight: {hh_weights.median()}")
    print(f"  Total weighted households: {hh_weights.sum():,}")
    
    return weighted_counts, total_weighted

if __name__ == "__main__":
    calculate_weighted_totals()
