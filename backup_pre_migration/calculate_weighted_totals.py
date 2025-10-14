#!/usr/bin/env python3
"""
Calculate weighted tax unit totals to compare with Hawaii's actual 743,109 tax returns.
Applies PUMS person weights (PWGTP) to scale up to full population estimates.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.pums_loader import PUMSDataLoader

def calculate_weighted_totals():
    """Calculate weighted tax unit totals and compare with actual filing counts."""
    
    print("Calculating weighted tax unit totals...")
    
    # Load data
    tax_units = pd.read_parquet("src/data/processed/tax_units_rule_based.parquet")
    pums_loader = PUMSDataLoader()
    person_df, hh_df = pums_loader.load_data()
    
    # Get person weights (PWGTP) and create a mapping from person ID to weight
    # Need to create the same ID format used in tax_units
    person_df['person_id'] = person_df['SERIALNO'] + '_' + person_df['SPORDER'].astype(str)
    person_weights = person_df.set_index('person_id')['PWGTP']
    
    # Map weights to tax units
    tax_units['weight'] = tax_units['primary_filer_id'].map(person_weights)
    
    # For joint filers, average the weights of both filers
    joint_units = tax_units[tax_units['filing_status'] == 'joint']
    if not joint_units.empty:
        joint_weights = joint_units['secondary_filer_id'].map(person_weights)
        tax_units.loc[joint_units.index, 'weight'] = (
            tax_units.loc[joint_units.index, 'weight'] + joint_weights
        ) / 2
    
    # Calculate weighted counts by filing status
    weighted_counts = tax_units.groupby('filing_status')['weight'].sum().astype(int)
    total_weighted = weighted_counts.sum()
    
    # Calculate percentages
    percentages = (weighted_counts / total_weighted * 100).round(1)
    
    # Display results
    print(f"\n{'='*60}")
    print(f"WEIGHTED TAX UNIT TOTALS")
    print(f"{'='*60}")
    
    print(f"\nEstimated Total Tax Units: {total_weighted:,}")
    print(f"Actual 2022 Hawaii Returns: 743,109")
    
    print(f"\nWeighted Filing Status Distribution:")
    for status, count in weighted_counts.items():
        print(f"  {status.replace('_', ' ').title()}: {count:,} ({percentages[status]}%)")
    
    # Compare with actual 2022 Hawaii filing distribution
    print(f"\nComparison with 2022 Hawaii Filing Distribution:")
    actual_counts = {
        'single': int(743109 * 0.51),  # 51% single
        'joint': int(743109 * 0.36),   # 36% joint
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
    print(f"\nScaling factor to match actual returns: {scaling_factor:.2f}")
    
    # Calculate scaled counts
    print(f"\nScaled to Match 743,109 Returns:")
    for status, count in weighted_counts.items():
        scaled = int(count * scaling_factor)
        print(f"  {status.replace('_', ' ').title()}: {scaled:,} "
              f"({percentages[status]}% of total)")
    
    return weighted_counts, total_weighted

if __name__ == "__main__":
    calculate_weighted_totals()
