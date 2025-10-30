"""
Calibrate filing status weights to match DOTAX SOI benchmarks.

This script applies post-hoc weight adjustments to ensure filing status
distributions match the targets as closely as possible.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Target distributions from DOTAX SOI 2022
TARGETS = {
    'single': 0.5100,  # 51.0%
    'married_filing_jointly': 0.3600,  # 36.0%
    'head_of_household': 0.0960,  # 9.6%
    'married_filing_separately': 0.0340,  # 3.4%
}

def standardize_filing_status(df):
    """Standardize filing status names."""
    status_map = {
        'single': 'single',
        'Single': 'single',
        'married_filing_jointly': 'married_filing_jointly',
        'Married Filing Jointly': 'married_filing_jointly',
        'married_filing_separately': 'married_filing_separately',
        'Married Filing Separately': 'married_filing_separately',
        'head_of_household': 'head_of_household',
        'Head of Household': 'head_of_household',
        'Head Of Household': 'head_of_household',
        'qualifying_widow': 'married_filing_jointly',  # Combine with joint
    }
    
    return df['filing_status'].map(status_map).fillna('single')

def calculate_calibration_factors(df):
    """
    Calculate calibration factors to match target distributions.
    
    Uses iterative proportional fitting (raking) to adjust weights while
    preserving the total weight sum.
    """
    # Standardize filing status
    df = df.copy()
    df['filing_status_std'] = standardize_filing_status(df)
    
    # Calculate current distribution
    total_weight = df['weight'].sum()
    current_dist = df.groupby('filing_status_std')['weight'].sum() / total_weight
    
    print("Current distribution:")
    for status, share in current_dist.items():
        target = TARGETS.get(status, 0)
        gap = share - target
        print(f"  {status}: {share:.4f} (target: {target:.4f}, gap: {gap:+.4f})")
    
    # Calculate calibration factors
    factors = {}
    for status in TARGETS.keys():
        current = current_dist.get(status, 0)
        target = TARGETS[status]
        if current > 0:
            factors[status] = target / current
        else:
            factors[status] = 1.0
    
    print("\nCalibration factors:")
    for status, factor in factors.items():
        print(f"  {status}: {factor:.4f}")
    
    return factors

def apply_calibration(df, factors):
    """Apply calibration factors to weights."""
    df = df.copy()
    df['filing_status_std'] = standardize_filing_status(df)
    
    # Apply factors
    df['calibration_factor'] = df['filing_status_std'].map(factors)
    df['weight_calibrated'] = df['weight'] * df['calibration_factor']
    
    # Verify results
    total_weight = df['weight_calibrated'].sum()
    final_dist = df.groupby('filing_status_std')['weight_calibrated'].sum() / total_weight
    
    print("\nFinal distribution after calibration:")
    for status, share in final_dist.items():
        target = TARGETS.get(status, 0)
        gap = share - target
        status_emoji = "✅" if abs(gap) < 0.001 else "⚠️"
        print(f"  {status}: {share:.4f} (target: {target:.4f}, gap: {gap:+.4f}) {status_emoji}")
    
    # Replace original weight with calibrated weight
    df['weight'] = df['weight_calibrated']
    df = df.drop(columns=['filing_status_std', 'calibration_factor', 'weight_calibrated'])
    
    return df

def main():
    """Main calibration workflow."""
    print("=" * 100)
    print("FILING STATUS WEIGHT CALIBRATION")
    print("=" * 100)
    print()
    
    # Find most recent tax units file
    processed_dir = Path('data/processed')
    tax_unit_files = sorted(processed_dir.glob('tax_units_calibrated_*.parquet'))
    
    if not tax_unit_files:
        print("Error: No tax_units_calibrated_*.parquet files found")
        return
    
    input_file = tax_unit_files[-1]
    print(f"Input file: {input_file}")
    print()
    
    # Load data
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df):,} tax units")
    print(f"Total weight: {df['weight'].sum():,.0f}")
    print()
    
    # Calculate calibration factors
    factors = calculate_calibration_factors(df)
    print()
    
    # Apply calibration
    df_calibrated = apply_calibration(df, factors)
    
    # Save calibrated data
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    output_file = processed_dir / f'tax_units_filing_status_calibrated_{timestamp}.parquet'
    df_calibrated.to_parquet(output_file, index=False)
    
    print()
    print(f"✅ Calibrated tax units saved to: {output_file}")
    print()
    print("=" * 100)
    print("CALIBRATION COMPLETE")
    print("=" * 100)

if __name__ == '__main__':
    main()
