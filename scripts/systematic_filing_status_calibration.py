"""
Systematic Filing Status Calibration Pipeline

Precisely aligns ALL filing statuses (Single, Joint, HoH, MFS) with DOTAX benchmarks
for both average AGI and total tax revenue. This is a reproducible pipeline component
that can be applied to any new data.

Author: Cascade AI
Date: November 2, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# DOTAX SOI 2022 Benchmarks (Table 5A)
DOTAX_BENCHMARKS = {
    'Single': {
        'returns': 335198,
        'agi_millions': 14297,
        'tax_millions': 864,
        'avg_agi': 42652,  # Calculated: 14297M / 335198
        'avg_tax': 2578,   # Calculated: 864M / 335198
    },
    'Joint': {
        'returns': 216358,
        'agi_millions': 26551,
        'tax_millions': 1674,
        'avg_agi': 122718,  # Calculated: 26551M / 216358
        'avg_tax': 7737,    # Calculated: 1674M / 216358
    },
    'Head of Household': {
        'returns': 67393,
        'agi_millions': 3744,
        'tax_millions': 202,
        'avg_agi': 55555,   # Calculated: 3744M / 67393
        'avg_tax': 2997,    # Calculated: 202M / 67393
    },
    'MFS': {
        'returns': 16007,
        'agi_millions': 3149,
        'tax_millions': 289,
        'avg_agi': 196726,  # Calculated: 3149M / 16007
        'avg_tax': 18055,   # Calculated: 289M / 16007
    },
}

# Filing status share targets (must sum to 1.0)
FILING_STATUS_SHARES = {
    'Single': 0.5100,
    'Joint': 0.3600,
    'Head of Household': 0.0960,
    'MFS': 0.0340,
}


def standardize_filing_status(df):
    """Standardize filing status names to canonical form."""
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
        'qualifying_widow': 'Joint',  # Combine with joint
    }
    
    return df['filing_status'].map(status_map).fillna('Other')


def calibrate_agi_distribution(df, filing_status, target_avg_agi, tolerance=0.02):
    """
    Calibrate AGI distribution for a specific filing status to match target average.
    
    Args:
        df: DataFrame with tax units for this filing status
        filing_status: Name of filing status being calibrated
        target_avg_agi: Target average AGI from DOTAX
        tolerance: Acceptable percentage deviation from target (default 2%)
    
    Returns:
        Calibrated DataFrame
    """
    if len(df) == 0:
        return df
    
    current_weighted_agi = (df['agi'] * df['weight']).sum()
    current_total_weight = df['weight'].sum()
    current_avg_agi = current_weighted_agi / current_total_weight if current_total_weight > 0 else 0
    
    gap_pct = (current_avg_agi - target_avg_agi) / target_avg_agi if target_avg_agi > 0 else 0
    
    logger.info(f"  {filing_status} AGI calibration:")
    logger.info(f"    Current avg AGI: ${current_avg_agi:,.0f}")
    logger.info(f"    Target avg AGI: ${target_avg_agi:,.0f}")
    logger.info(f"    Gap: {gap_pct*100:+.1f}%")
    
    if abs(gap_pct) <= tolerance:
        logger.info(f"    ✅ Already within tolerance ({tolerance*100:.0f}%)")
        return df
    
    # Apply iterative weight adjustment to converge on target
    df = df.copy()
    max_iterations = 20
    
    for iteration in range(max_iterations):
        # Calculate current state
        current_weighted_agi = (df['agi'] * df['weight']).sum()
        current_total_weight = df['weight'].sum()
        current_avg_agi = current_weighted_agi / current_total_weight
        
        gap = current_avg_agi - target_avg_agi
        gap_pct = gap / target_avg_agi if target_avg_agi > 0 else 0
        
        if abs(gap_pct) <= tolerance:
            logger.info(f"    ✅ Converged after {iteration + 1} iterations")
            break
        
        # Adjust weights based on AGI distance from target
        # Units closer to target get higher weights
        for idx in df.index:
            unit_agi = df.loc[idx, 'agi']
            current_weight = df.loc[idx, 'weight']
            
            # Calculate how far this unit is from the target
            distance_ratio = abs(unit_agi - target_avg_agi) / target_avg_agi if target_avg_agi > 0 else 1.0
            
            # Weight adjustment based on distance
            # Units closer to target get boosted, farther ones get reduced
            if gap > 0:  # Current avg too high - favor lower AGI units
                if unit_agi < target_avg_agi:
                    adjustment = 1.0 + 0.05 * (1.0 - distance_ratio)  # Boost lower units
                else:
                    adjustment = 1.0 - 0.05 * distance_ratio  # Reduce higher units
            else:  # Current avg too low - favor higher AGI units
                if unit_agi > target_avg_agi:
                    adjustment = 1.0 + 0.05 * (1.0 - distance_ratio)  # Boost higher units
                else:
                    adjustment = 1.0 - 0.05 * distance_ratio  # Reduce lower units
            
            # Apply adjustment with bounds
            adjustment = max(0.5, min(1.5, adjustment))
            df.loc[idx, 'weight'] = current_weight * adjustment
        
        # Renormalize weights to maintain total
        total_weight = df['weight'].sum()
        if total_weight > 0:
            df['weight'] = df['weight'] * (current_total_weight / total_weight)
    
    # Final verification
    final_weighted_agi = (df['agi'] * df['weight']).sum()
    final_total_weight = df['weight'].sum()
    final_avg_agi = final_weighted_agi / final_total_weight if final_total_weight > 0 else 0
    final_gap_pct = (final_avg_agi - target_avg_agi) / target_avg_agi if target_avg_agi > 0 else 0
    
    logger.info(f"    Final avg AGI: ${final_avg_agi:,.0f} (gap: {final_gap_pct*100:+.1f}%)")
    
    return df


def calibrate_tax_revenue(df, filing_status, target_tax_millions, tolerance=0.05):
    """
    Calibrate tax revenue for a specific filing status to match target.
    
    Args:
        df: DataFrame with tax units for this filing status
        filing_status: Name of filing status being calibrated
        target_tax_millions: Target total tax revenue in millions from DOTAX
        tolerance: Acceptable percentage deviation from target (default 5%)
    
    Returns:
        Calibrated DataFrame
    """
    if len(df) == 0:
        return df
    
    current_total_tax = (df['hi_state_tax'] * df['weight']).sum() / 1_000_000
    gap_pct = (current_total_tax - target_tax_millions) / target_tax_millions if target_tax_millions > 0 else 0
    
    logger.info(f"  {filing_status} tax revenue calibration:")
    logger.info(f"    Current total tax: ${current_total_tax:.1f}M")
    logger.info(f"    Target total tax: ${target_tax_millions:.0f}M")
    logger.info(f"    Gap: {gap_pct*100:+.1f}%")
    
    if abs(gap_pct) <= tolerance:
        logger.info(f"    ✅ Already within tolerance ({tolerance*100:.0f}%)")
        return df
    
    # Apply uniform scaling to weights to match tax revenue target
    # This preserves the AGI distribution while adjusting total revenue
    df = df.copy()
    
    # Calculate scaling factor
    if current_total_tax > 0:
        scaling_factor = target_tax_millions / current_total_tax
    else:
        scaling_factor = 1.0
    
    # Apply scaling with bounds to prevent extreme adjustments
    scaling_factor = max(0.5, min(2.0, scaling_factor))
    
    df['weight'] = df['weight'] * scaling_factor
    
    # Final verification
    final_total_tax = (df['hi_state_tax'] * df['weight']).sum() / 1_000_000
    final_gap_pct = (final_total_tax - target_tax_millions) / target_tax_millions if target_tax_millions > 0 else 0
    
    logger.info(f"    Scaling factor applied: {scaling_factor:.4f}")
    logger.info(f"    Final total tax: ${final_total_tax:.1f}M (gap: {final_gap_pct*100:+.1f}%)")
    
    return df


def apply_filing_status_shares(df):
    """
    Apply final calibration to ensure filing status shares match DOTAX targets exactly.
    
    Args:
        df: DataFrame with all tax units
    
    Returns:
        DataFrame with calibrated filing status shares
    """
    df = df.copy()
    df['filing_status_clean'] = standardize_filing_status(df)
    
    total_weight = df['weight'].sum()
    current_dist = df.groupby('filing_status_clean')['weight'].sum() / total_weight
    
    logger.info("  Filing status share calibration:")
    
    # Calculate adjustment factors
    factors = {}
    for status, target_share in FILING_STATUS_SHARES.items():
        current_share = current_dist.get(status, 0)
        if current_share > 0:
            factors[status] = target_share / current_share
        else:
            factors[status] = 1.0
        
        logger.info(f"    {status}: {current_share:.4f} → {target_share:.4f} (factor: {factors[status]:.4f})")
    
    # Apply factors
    df['__calibration_factor'] = df['filing_status_clean'].map(factors).fillna(1.0)
    df['weight'] = df['weight'] * df['__calibration_factor']
    
    # Clean up temporary column only (keep filing_status_clean for next steps)
    df = df.drop(columns=['__calibration_factor'])
    
    return df


def validate_results(df):
    """
    Validate final results against DOTAX benchmarks.
    
    Args:
        df: Calibrated DataFrame
    
    Returns:
        Dictionary with validation metrics
    """
    df['filing_status_clean'] = standardize_filing_status(df)
    
    results = {}
    
    logger.info("\n" + "="*100)
    logger.info("VALIDATION RESULTS")
    logger.info("="*100)
    
    for status in ['Single', 'Joint', 'Head of Household', 'MFS']:
        status_df = df[df['filing_status_clean'] == status]
        
        if len(status_df) == 0:
            continue
        
        # Calculate metrics
        model_returns = status_df['weight'].sum()
        model_agi_total = (status_df['agi'] * status_df['weight']).sum() / 1_000_000
        model_tax_total = (status_df['hi_state_tax'] * status_df['weight']).sum() / 1_000_000
        model_avg_agi = model_agi_total * 1_000_000 / model_returns if model_returns > 0 else 0
        
        # Get benchmarks
        dotax = DOTAX_BENCHMARKS[status]
        
        # Calculate gaps
        agi_gap_pct = (model_avg_agi - dotax['avg_agi']) / dotax['avg_agi'] * 100 if dotax['avg_agi'] > 0 else 0
        tax_gap_pct = (model_tax_total - dotax['tax_millions']) / dotax['tax_millions'] * 100 if dotax['tax_millions'] > 0 else 0
        
        # Store results
        results[status] = {
            'returns': model_returns,
            'avg_agi': model_avg_agi,
            'total_tax': model_tax_total,
            'agi_gap_pct': agi_gap_pct,
            'tax_gap_pct': tax_gap_pct,
        }
        
        # Log results
        agi_emoji = "✅" if abs(agi_gap_pct) < 5 else "⚠️" if abs(agi_gap_pct) < 10 else "❌"
        tax_emoji = "✅" if abs(tax_gap_pct) < 5 else "⚠️" if abs(tax_gap_pct) < 10 else "❌"
        
        logger.info(f"\n{status}:")
        logger.info(f"  Returns: {model_returns:,.0f} (target: {dotax['returns']:,})")
        logger.info(f"  Avg AGI: ${model_avg_agi:,.0f} vs ${dotax['avg_agi']:,.0f} ({agi_gap_pct:+.1f}%) {agi_emoji}")
        logger.info(f"  Total Tax: ${model_tax_total:.1f}M vs ${dotax['tax_millions']:.0f}M ({tax_gap_pct:+.1f}%) {tax_emoji}")
    
    # Total revenue
    total_tax = (df['hi_state_tax'] * df['weight']).sum() / 1_000_000
    total_dotax = sum(b['tax_millions'] for b in DOTAX_BENCHMARKS.values())
    total_gap_pct = (total_tax - total_dotax) / total_dotax * 100
    
    total_emoji = "✅" if abs(total_gap_pct) < 3 else "⚠️" if abs(total_gap_pct) < 5 else "❌"
    
    logger.info("\n" + "="*100)
    logger.info(f"TOTAL REVENUE: ${total_tax:.1f}M vs ${total_dotax:.0f}M ({total_gap_pct:+.1f}%) {total_emoji}")
    logger.info("="*100)
    
    results['total'] = {
        'revenue': total_tax,
        'gap_pct': total_gap_pct,
    }
    
    return results


def main(input_file=None):
    """
    Main calibration pipeline.
    
    Args:
        input_file: Path to input parquet file (optional, will find latest if not provided)
    
    Returns:
        Path to calibrated output file
    """
    logger.info("="*100)
    logger.info("SYSTEMATIC FILING STATUS CALIBRATION PIPELINE")
    logger.info("="*100)
    logger.info("")
    
    # Find input file
    if input_file is None:
        processed_dir = Path('data/processed')
        # Look for most recent MFS-fixed file or calibrated file
        candidates = sorted(processed_dir.glob('tax_units_*calibrated_*.parquet'))
        if not candidates:
            candidates = sorted(processed_dir.glob('tax_units_*.parquet'))
        
        if not candidates:
            raise FileNotFoundError("No tax units file found")
        
        input_file = candidates[-1]
    
    logger.info(f"Input file: {input_file}")
    
    # Load data
    df = pd.read_parquet(input_file)
    logger.info(f"Loaded {len(df):,} tax units")
    logger.info(f"Total weighted returns: {df['weight'].sum():,.0f}")
    
    # Standardize filing status
    df['filing_status_clean'] = standardize_filing_status(df)
    
    # Step 1: Calibrate AGI distributions for each filing status
    logger.info("\n" + "="*80)
    logger.info("STEP 1: CALIBRATE AGI DISTRIBUTIONS")
    logger.info("="*80)
    
    calibrated_dfs = []
    
    for status in ['Single', 'Joint', 'Head of Household', 'MFS']:
        logger.info(f"\nCalibrating {status}...")
        
        status_df = df[df['filing_status_clean'] == status].copy()
        
        if len(status_df) == 0:
            logger.warning(f"  ⚠️  No units found for {status}")
            continue
        
        # Calibrate AGI distribution
        target_avg_agi = DOTAX_BENCHMARKS[status]['avg_agi']
        status_df = calibrate_agi_distribution(status_df, status, target_avg_agi)
        
        calibrated_dfs.append(status_df)
    
    # Combine calibrated dataframes
    df_calibrated = pd.concat(calibrated_dfs, ignore_index=True)
    
    # Step 2: Apply filing status shares FIRST (before tax revenue calibration)
    # This ensures we have the correct number of returns for each status
    logger.info("\n" + "="*80)
    logger.info("STEP 2: APPLY FILING STATUS SHARES (Before Tax Calibration)")
    logger.info("="*80)
    
    df_calibrated = apply_filing_status_shares(df_calibrated)
    
    # Step 3: Calibrate tax revenue for each filing status
    # Now that return counts are correct, we can accurately scale tax revenue
    logger.info("\n" + "="*80)
    logger.info("STEP 3: CALIBRATE TAX REVENUE (With Correct Return Counts)")
    logger.info("="*80)
    
    calibrated_dfs = []
    
    for status in ['Single', 'Joint', 'Head of Household', 'MFS']:
        logger.info(f"\nCalibrating {status}...")
        
        status_df = df_calibrated[df_calibrated['filing_status_clean'] == status].copy()
        
        if len(status_df) == 0:
            continue
        
        # Calibrate tax revenue
        target_tax = DOTAX_BENCHMARKS[status]['tax_millions']
        status_df = calibrate_tax_revenue(status_df, status, target_tax)
        
        calibrated_dfs.append(status_df)
    
    # Combine calibrated dataframes
    df_final = pd.concat(calibrated_dfs, ignore_index=True)
    
    # Step 4: Validate results
    logger.info("\n" + "="*80)
    logger.info("STEP 4: VALIDATE RESULTS")
    logger.info("="*80)
    
    validation_results = validate_results(df_final)
    
    # Save output
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = Path('data/processed') / f'tax_units_systematically_calibrated_{timestamp}.parquet'
    
    # Clean up temporary columns
    df_final = df_final.drop(columns=['filing_status_clean'], errors='ignore')
    
    # Fix parquet serialization
    for col in df_final.columns:
        if df_final[col].dtype == 'object':
            df_final[col] = df_final[col].astype(str)
    
    df_final.to_parquet(output_file, index=False)
    
    logger.info(f"\n✅ Calibrated tax units saved to: {output_file}")
    
    # Save validation report
    report_file = output_file.with_suffix('.json')
    import json
    with open(report_file, 'w') as f:
        json.dump(validation_results, f, indent=2, default=float)
    
    logger.info(f"📊 Validation report saved to: {report_file}")
    
    logger.info("\n" + "="*100)
    logger.info("CALIBRATION COMPLETE")
    logger.info("="*100)
    
    return output_file


if __name__ == '__main__':
    import sys
    
    # Allow passing input file as command line argument
    input_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    output_file = main(input_file)
    
    print(f"\nFinal output: {output_file}")
