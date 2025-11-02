"""
Post-processing fix for MFS income distribution.

Converts low-income MFS filers to joint filers to match DOTAX average AGI target of $196,726.
This is necessary because the income threshold check in _should_file_separately() uses raw PUMS
income, which differs from final calibrated AGI.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# DOTAX targets for MFS
DOTAX_MFS_RETURNS = 16007
DOTAX_MFS_AVG_AGI = 196726
DOTAX_MFS_TOTAL_TAX = 289  # Million

# Minimum AGI threshold for MFS eligibility (post-processing)
MIN_MFS_AGI = 150000

def standardize_filing_status(df):
    """Standardize filing status names."""
    status_map = {
        'single': 'Single',
        'Single': 'Single',
        'married_filing_jointly': 'Joint',
        'Married Filing Jointly': 'Joint',
        'married_filing_separately': 'MFS',
        'Married Filing Separately': 'MFS',
        'head_of_household': 'Head of Household',
        'Head of Household': 'Head of Household',
        'Head Of Household': 'Head of Household',
        'qualifying_widow': 'Joint',
    }
    
    return df['filing_status'].map(status_map).fillna('Other')

def main():
    print("=" * 100)
    print("MFS INCOME DISTRIBUTION FIX (POST-PROCESSING)")
    print("=" * 100)
    print()
    
    # Find most recent calibrated file
    processed_dir = Path('data/processed')
    calibrated_files = sorted(processed_dir.glob('tax_units_filing_status_calibrated_*.parquet'))
    
    if not calibrated_files:
        print("Error: No calibrated tax units file found")
        return
    
    input_file = calibrated_files[-1]
    print(f"Input file: {input_file}")
    print()
    
    # Load data
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df):,} tax units")
    print()
    
    # Standardize filing status
    df['filing_status_clean'] = standardize_filing_status(df)
    
    # Analyze current MFS distribution
    mfs_df = df[df['filing_status_clean'] == 'MFS'].copy()
    
    print("BEFORE FIX:")
    print(f"  Total MFS units: {len(mfs_df):,}")
    print(f"  Weighted MFS returns: {mfs_df['weight'].sum():,.0f}")
    print(f"  Average AGI: ${(mfs_df['agi'] * mfs_df['weight']).sum() / mfs_df['weight'].sum():,.0f}")
    print(f"  Target AGI: ${DOTAX_MFS_AVG_AGI:,.0f}")
    print()
    
    # Identify low-income MFS filers to convert to joint
    low_income_mfs_mask = (df['filing_status_clean'] == 'MFS') & (df['agi'] < MIN_MFS_AGI)
    low_income_mfs_count = low_income_mfs_mask.sum()
    low_income_mfs_weighted = df[low_income_mfs_mask]['weight'].sum()
    
    print(f"Low-income MFS filers (AGI < ${MIN_MFS_AGI:,}):")
    print(f"  Units to convert: {low_income_mfs_count:,}")
    print(f"  Weighted returns: {low_income_mfs_weighted:,.0f}")
    print(f"  Percentage of MFS: {low_income_mfs_weighted / mfs_df['weight'].sum() * 100:.1f}%")
    print()
    
    # Convert low-income MFS to joint filers
    print("Converting low-income MFS filers to joint filers...")
    df.loc[low_income_mfs_mask, 'filing_status'] = 'married_filing_jointly'
    df.loc[low_income_mfs_mask, 'filing_status_clean'] = 'Joint'
    
    # Get remaining high-income MFS units
    mfs_df_after = df[df['filing_status_clean'] == 'MFS'].copy()
    
    print()
    print("AFTER LOW-INCOME CONVERSION:")
    print(f"  Total MFS units: {len(mfs_df_after):,}")
    print(f"  Weighted MFS returns: {mfs_df_after['weight'].sum():,.0f}")
    
    if len(mfs_df_after) == 0:
        print("  ⚠️  No MFS units remaining after conversion!")
        return
    
    avg_agi_after = (mfs_df_after['agi'] * mfs_df_after['weight']).sum() / mfs_df_after['weight'].sum()
    print(f"  Average AGI: ${avg_agi_after:,.0f}")
    print(f"  Target AGI: ${DOTAX_MFS_AVG_AGI:,.0f}")
    print(f"  Gap: {(avg_agi_after - DOTAX_MFS_AVG_AGI) / DOTAX_MFS_AVG_AGI * 100:+.1f}%")
    print()
    
    # Step 2: Calibrate MFS weights to match DOTAX targets exactly
    print("=" * 100)
    print("STEP 2: CALIBRATE MFS WEIGHTS TO MATCH DOTAX TARGETS")
    print("=" * 100)
    print()
    
    # Strategy: Adjust individual MFS weights to achieve target average AGI and return count
    # We'll use a scoring approach to select/weight units that bring us closest to target
    
    mfs_indices = df[df['filing_status_clean'] == 'MFS'].index
    current_weighted_agi = (df.loc[mfs_indices, 'agi'] * df.loc[mfs_indices, 'weight']).sum()
    current_total_weight = df.loc[mfs_indices, 'weight'].sum()
    current_avg_agi = current_weighted_agi / current_total_weight
    
    # Calculate distance from target AGI for each MFS unit
    df.loc[mfs_indices, '_agi_distance'] = np.abs(df.loc[mfs_indices, 'agi'] - DOTAX_MFS_AVG_AGI)
    
    # Sort MFS units by AGI distance (closest to target first)
    mfs_sorted = df.loc[mfs_indices].copy().sort_values('_agi_distance')
    
    # Two-phase calibration:
    # Phase 1: Select units closest to target AGI until we hit target return count
    # Phase 2: Fine-tune weights to match exact targets
    
    print(f"Current MFS state:")
    print(f"  Returns: {current_total_weight:,.0f} (target: {DOTAX_MFS_RETURNS:,})")
    print(f"  Avg AGI: ${current_avg_agi:,.0f} (target: ${DOTAX_MFS_AVG_AGI:,})")
    print()
    
    # Calculate global scaling factor to match target return count
    return_scaling_factor = DOTAX_MFS_RETURNS / current_total_weight
    
    print(f"Applying return count scaling: {return_scaling_factor:.4f}")
    df.loc[mfs_indices, 'weight'] = df.loc[mfs_indices, 'weight'] * return_scaling_factor
    
    # Check if average AGI is now close to target
    new_weighted_agi = (df.loc[mfs_indices, 'agi'] * df.loc[mfs_indices, 'weight']).sum()
    new_total_weight = df.loc[mfs_indices, 'weight'].sum()
    new_avg_agi = new_weighted_agi / new_total_weight
    
    print(f"After return count scaling:")
    print(f"  Returns: {new_total_weight:,.0f} (target: {DOTAX_MFS_RETURNS:,})")
    print(f"  Avg AGI: ${new_avg_agi:,.0f} (target: ${DOTAX_MFS_AVG_AGI:,})")
    print()
    
    # If average AGI is still off, apply iterative AGI-based weight adjustment
    agi_gap = new_avg_agi - DOTAX_MFS_AVG_AGI
    
    if abs(agi_gap) > 1000:  # More than $1K gap
        print("Applying proportional weight adjustment...")
        
        # Simple proportional adjustment:
        # If avg AGI is too high, reduce weights on high-AGI units proportionally
        # If avg AGI is too low, increase weights on high-AGI units proportionally
        
        for idx in mfs_indices:
            unit_agi = df.loc[idx, 'agi']
            current_weight = df.loc[idx, 'weight']
            
            # Calculate adjustment based on distance from average (not target)
            agi_ratio = unit_agi / new_avg_agi if new_avg_agi > 0 else 1.0
            
            if agi_gap > 0:  # Avg AGI too high - reduce high-AGI units
                if agi_ratio > 1.0:  # Unit above average
                    # Reduce weight proportional to how far above average
                    adjustment = 1.0 / (1.0 + 0.12 * (agi_ratio - 1.0))
                else:  # Unit below average
                    # Increase weight proportional to how far below average
                    adjustment = 1.0 + 0.12 * (1.0 - agi_ratio)
            else:  # Avg AGI too low - increase high-AGI units
                if agi_ratio > 1.0:  # Unit above average
                    # Increase weight proportional to how far above average
                    adjustment = 1.0 + 0.12 * (agi_ratio - 1.0)
                else:  # Unit below average
                    # Reduce weight proportional to how far below average
                    adjustment = 1.0 / (1.0 + 0.12 * (1.0 - agi_ratio))
            
            df.loc[idx, 'weight'] = current_weight * adjustment
        
        # Renormalize to maintain target return count
        final_total_weight = df.loc[mfs_indices, 'weight'].sum()
        renorm_factor = DOTAX_MFS_RETURNS / final_total_weight
        df.loc[mfs_indices, 'weight'] = df.loc[mfs_indices, 'weight'] * renorm_factor
        
        # Verify result
        check_weighted_agi = (df.loc[mfs_indices, 'agi'] * df.loc[mfs_indices, 'weight']).sum()
        check_total_weight = df.loc[mfs_indices, 'weight'].sum()
        check_avg_agi = check_weighted_agi / check_total_weight
        
        print(f"  After proportional adjustment: Avg AGI = ${check_avg_agi:,.0f} (target: ${DOTAX_MFS_AVG_AGI:,}, gap: {(check_avg_agi - DOTAX_MFS_AVG_AGI) / DOTAX_MFS_AVG_AGI * 100:+.1f}%)")
    
    # Final statistics
    final_weighted_agi = (df.loc[mfs_indices, 'agi'] * df.loc[mfs_indices, 'weight']).sum()
    final_total_weight = df.loc[mfs_indices, 'weight'].sum()
    final_avg_agi = final_weighted_agi / final_total_weight
    final_total_tax = (df.loc[mfs_indices, 'hi_state_tax'] * df.loc[mfs_indices, 'weight']).sum() / 1_000_000
    
    print()
    print("FINAL MFS CALIBRATION RESULTS:")
    print(f"  Returns: {final_total_weight:,.0f} (target: {DOTAX_MFS_RETURNS:,}, gap: {(final_total_weight - DOTAX_MFS_RETURNS) / DOTAX_MFS_RETURNS * 100:+.1f}%)")
    print(f"  Avg AGI: ${final_avg_agi:,.0f} (target: ${DOTAX_MFS_AVG_AGI:,}, gap: {(final_avg_agi - DOTAX_MFS_AVG_AGI) / DOTAX_MFS_AVG_AGI * 100:+.1f}%)")
    print(f"  Total Tax: ${final_total_tax:,.1f}M (target: ${DOTAX_MFS_TOTAL_TAX:,.0f}M, gap: {(final_total_tax - DOTAX_MFS_TOTAL_TAX) / DOTAX_MFS_TOTAL_TAX * 100:+.1f}%)")
    print()
    
    # Clean up temporary column
    df = df.drop(columns=['_agi_distance'], errors='ignore')
    print()
    
    # Check filing status distribution
    print("=" * 100)
    print("FILING STATUS DISTRIBUTION AFTER FIX")
    print("=" * 100)
    print()
    
    total_weight = df['weight'].sum()
    status_counts = df.groupby('filing_status_clean')['weight'].sum()
    
    targets = {
        'Single': 0.5100,
        'Joint': 0.3600,
        'Head of Household': 0.0960,
        'MFS': 0.0340,
    }
    
    print(f"{'Status':<20} {'Count':>15} {'Share %':>10} {'Target %':>10} {'Gap':>10}")
    print("-" * 100)
    
    for status in ['Single', 'Joint', 'Head of Household', 'MFS']:
        count = status_counts.get(status, 0)
        share = count / total_weight * 100
        target = targets[status] * 100
        gap = share - target
        status_emoji = "✅" if abs(gap) < 1.0 else "⚠️" if abs(gap) < 3.0 else "❌"
        print(f"{status:<20} {count:>15,.0f} {share:>9.2f}% {target:>9.2f}% {gap:>+9.2f}pp {status_emoji}")
    
    print()
    print(f"Total: {total_weight:,.0f}")
    print()
    
    # Save fixed data
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = processed_dir / f'tax_units_mfs_fixed_{timestamp}.parquet'
    
    # Drop temporary column
    df = df.drop(columns=['filing_status_clean'])
    
    # Fix parquet serialization
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)
    
    df.to_parquet(output_file, index=False)
    
    print(f"✅ Fixed tax units saved to: {output_file}")
    print()
    print("=" * 100)
    print("MFS FIX COMPLETE")
    print("=" * 100)

if __name__ == '__main__':
    main()
