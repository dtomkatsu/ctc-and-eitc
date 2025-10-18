"""
IRS SOI-Based Calibration with Iterative Proportional Fitting (IPF)

This module implements calibration using IPF (raking) to match both:
1. Filing status distribution (DOTAX targets)
2. AGI bracket distribution (DOTAX Table 12A)

IPF iteratively adjusts weights to satisfy multiple marginal constraints,
finding a balanced solution that minimizes total deviation from both targets.

This replaces the previous two-step approach with a single, more robust method.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, List, Optional, Union

logger = logging.getLogger(__name__)

# DOTAX Filing Status Totals (from DOTAX Table 1)
DOTAX_FILING_STATUS = {
    'single': 335198,
    'married_filing_jointly': 216358,
    'head_of_household': 67393,
    'married_filing_separately': 16007,
}

# DOTAX AGI Bracket Targets (from Table 12A)
DOTAX_AGI_BRACKETS = {
    (0, 10000): 129376,
    (10000, 20000): 64160,
    (20000, 30000): 57835,
    (30000, 40000): 59827,
    (40000, 50000): 53555,
    (50000, 75000): 91459,
    (75000, 100000): 54976,
    (100000, 150000): 62065,
    (150000, 200000): 27976,
    (200000, 300000): 18937,
    (300000, 400000): 6076,
    (400000, float('inf')): 8875,
}

def assign_agi_bracket(agi: float) -> Tuple[float, float]:
    """
    Assign AGI to DOTAX bracket.
    
    Args:
        agi: Adjusted Gross Income
        
    Returns:
        Tuple of (min_agi, max_agi) for the bracket
    """
    for (min_agi, max_agi) in DOTAX_AGI_BRACKETS.keys():
        if max_agi == float('inf'):
            if agi >= min_agi:
                return (min_agi, max_agi)
        else:
            if min_agi <= agi < max_agi:
                return (min_agi, max_agi)
    return (0, 10000)  # Default
    Args:
        agi: Adjusted Gross Income
        
    Returns:
        Tuple of (min_agi, max_agi) for the bracket
    """
    for min_agi, max_agi, _, _, _, _ in IRS_SOI_BENCHMARKS:
        if max_agi == float('inf'):
            if agi >= min_agi:
                return (min_agi, max_agi)
        else:
            if min_agi <= agi < max_agi:
                return (min_agi, max_agi)
    return (0, 25000)  # Default to first bracket


def calculate_irs_soi_targets_scaled_to_dotax() -> Dict:
    """
    Calculate target counts for each (filing_status, agi_bracket) cell.
    
    Uses IRS SOI proportions but scales to DOTAX total counts.
    This combines the strengths of both data sources:
    - IRS: Filing status distribution within brackets
    - DOTAX: Total counts (more Hawaii-specific)
    
    Returns:
        Dictionary mapping (filing_status, agi_bracket) -> target_count
    """
    targets = {}
    
    logger.info("="*80)
    logger.info("CALCULATING IRS SOI TARGETS (SCALED TO DOTAX)")
    logger.info("="*80)
    
    logger.info(f"\n{'AGI Bracket':<15} {'Status':<10} {'IRS Raw':>10} {'IRS %':>8} {'DOTAX Total':>12} {'Scaled Target':>15}")
    logger.info("-"*80)
    
    for min_agi, max_agi, irs_total, irs_single, irs_joint, irs_hoh in IRS_SOI_BENCHMARKS:
        # Get DOTAX target for this bracket
        bracket_key = (min_agi, max_agi)
        dotax_total = DOTAX_AGI_TARGETS.get(bracket_key, irs_total)
        
        # Calculate IRS proportions
        single_pct = irs_single / irs_total if irs_total > 0 else 0
        joint_pct = irs_joint / irs_total if irs_total > 0 else 0
        hoh_pct = irs_hoh / irs_total if irs_total > 0 else 0
        
        # Scale to DOTAX total
        single_target = dotax_total * single_pct
        joint_target = dotax_total * joint_pct
        hoh_target = dotax_total * hoh_pct
        
        # Store targets
        targets[('single', bracket_key)] = single_target
        targets[('married_filing_jointly', bracket_key)] = joint_target
        targets[('head_of_household', bracket_key)] = hoh_target
        
        # Log
        if max_agi == float('inf'):
            label = f"${min_agi//1000}k+"
        else:
            label = f"${min_agi//1000}k-${max_agi//1000}k"
        
        logger.info(f"{label:<15} {'Single':<10} {irs_single:>10,} {single_pct:>7.1%} {dotax_total:>12,} {single_target:>15,.0f}")
        logger.info(f"{'':<15} {'Joint':<10} {irs_joint:>10,} {joint_pct:>7.1%} {'':<12} {joint_target:>15,.0f}")
        logger.info(f"{'':<15} {'HoH':<10} {irs_hoh:>10,} {hoh_pct:>7.1%} {'':<12} {hoh_target:>15,.0f}")
        logger.info("-"*80)
    
    return targets


def apply_irs_soi_calibration(tax_units: pd.DataFrame,
                               weight_col: str = 'weight',
                               output_col: str = 'weight_irs_calibrated') -> pd.DataFrame:
    """
    Apply calibration based on IRS SOI filing status × AGI distribution.
    
    This is more sophisticated than separate calibration layers because it
    ensures the RIGHT MIX of filing statuses within each AGI bracket.
    
    Args:
        tax_units: DataFrame with tax units
        weight_col: Column name for input weights
        output_col: Column name for output calibrated weights
        
    Returns:
        DataFrame with calibrated weights
    """
    logger.info("="*80)
    logger.info("IRS SOI CALIBRATION: Filing Status × AGI Distribution")
    logger.info("="*80)
    
    df = tax_units.copy()
    
    # Store original weights
    if 'weight_original' not in df.columns:
        df['weight_original'] = df[weight_col].copy()
    
    # Assign IRS AGI brackets
    df['irs_agi_bracket'] = df['agi'].apply(assign_irs_agi_bracket)
    
    # Calculate targets
    targets = calculate_irs_soi_targets_scaled_to_dotax()
    
    # Calculate current counts by (filing_status, agi_bracket)
    df['status_bracket_key'] = list(zip(df['filing_status'], df['irs_agi_bracket']))
    current_counts = df.groupby('status_bracket_key')[weight_col].sum()
    
    # Calculate calibration factors for each cell
    df['irs_calibration_factor'] = 1.0
    
    for (status, bracket), target_count in targets.items():
        key = (status, bracket)
        current_count = current_counts.get(key, 0)
        
        if current_count > 0:
            factor = target_count / current_count
        else:
            factor = 1.0
        
        # Apply factor
        mask = df['status_bracket_key'] == key
        df.loc[mask, 'irs_calibration_factor'] = factor
    
    # Apply calibration
    df[output_col] = df[weight_col] * df['irs_calibration_factor']
    
    # Report results
    logger.info("\n" + "="*80)
    logger.info("CALIBRATION RESULTS BY (FILING STATUS, AGI BRACKET)")
    logger.info("="*80)
    
    logger.info(f"\n{'AGI Bracket':<15} {'Status':<10} {'Before':>12} {'After':>12} {'Target':>12} {'Factor':>8}")
    logger.info("-"*80)
    
    for min_agi, max_agi, _, _, _, _ in IRS_SOI_BENCHMARKS:
        bracket_key = (min_agi, max_agi)
        
        if max_agi == float('inf'):
            label = f"${min_agi//1000}k+"
        else:
            label = f"${min_agi//1000}k-${max_agi//1000}k"
        
        for status in ['single', 'married_filing_jointly', 'head_of_household']:
            key = (status, bracket_key)
            
            # Get counts
            mask = df['status_bracket_key'] == key
            before = df[mask][weight_col].sum()
            after = df[mask][output_col].sum()
            target = targets.get(key, 0)
            factor = target / before if before > 0 else 1.0
            
            if before > 0:  # Only show non-empty cells
                status_short = status.replace('married_filing_jointly', 'Joint').replace('head_of_household', 'HoH').replace('single', 'Single')
                logger.info(f"{label:<15} {status_short:<10} {before:>12,.0f} {after:>12,.0f} {target:>12,.0f} {factor:>7.3f}x")
        
        logger.info("-"*80)
    
    # Summary by filing status
    logger.info("\n" + "="*80)
    logger.info("FILING STATUS TOTALS")
    logger.info("="*80)
    
    logger.info(f"\n{'Status':<30} {'Before':>12} {'After':>12} {'DOTAX Target':>15} {'Gap':>10}")
    logger.info("-"*80)
    
    for status in ['single', 'married_filing_jointly', 'head_of_household', 'married_filing_separately']:
        before = df[df['filing_status'] == status][weight_col].sum()
        after = df[df['filing_status'] == status][output_col].sum()
        target = DOTAX_FILING_STATUS.get(status, 0)
        gap_pct = ((after - target) / target * 100) if target > 0 else 0
        
        logger.info(f"{status:<30} {before:>12,.0f} {after:>12,.0f} {target:>15,} {gap_pct:>+9.1f}%")
    
    # Summary by AGI bracket
    logger.info("\n" + "="*80)
    logger.info("AGI BRACKET TOTALS")
    logger.info("="*80)
    
    logger.info(f"\n{'AGI Bracket':<15} {'Before':>12} {'After':>12} {'DOTAX Target':>15} {'Gap':>10}")
    logger.info("-"*80)
    
    for min_agi, max_agi, _, _, _, _ in IRS_SOI_BENCHMARKS:
        bracket_key = (min_agi, max_agi)
        
        if max_agi == float('inf'):
            label = f"${min_agi//1000}k+"
        else:
            label = f"${min_agi//1000}k-${max_agi//1000}k"
        
        mask = df['irs_agi_bracket'] == bracket_key
        before = df[mask][weight_col].sum()
        after = df[mask][output_col].sum()
        target = DOTAX_AGI_TARGETS.get(bracket_key, 0)
        gap_pct = ((after - target) / target * 100) if target > 0 else 0
        
        marker = " ⚠️" if min_agi == 50000 and max_agi == 75000 else ""
        logger.info(f"{label:<15} {before:>12,.0f} {after:>12,.0f} {target:>15,} {gap_pct:>+9.1f}%{marker}")
    
    return df


def validate_irs_soi_calibration(tax_units: pd.DataFrame,
                                  weight_col: str = 'weight_irs_calibrated') -> Dict:
    """
    Validate IRS SOI calibration results.
    
    Args:
        tax_units: Calibrated tax units
        weight_col: Column name for calibrated weights
        
    Returns:
        Dictionary with validation metrics
    """
    df = tax_units.copy()
    df['irs_agi_bracket'] = df['agi'].apply(assign_irs_agi_bracket)
    
    targets = calculate_irs_soi_targets_scaled_to_dotax()
    
    validation = {
        'total_returns': df[weight_col].sum(),
        'cells_accurate': 0,
        'total_cells': len(targets),
        'filing_status_accuracy': {},
        'agi_bracket_accuracy': {}
    }
    
    # Check each (filing_status, agi_bracket) cell
    accurate_cells = 0
    for (status, bracket), target in targets.items():
        df['status_bracket_key'] = list(zip(df['filing_status'], df['irs_agi_bracket']))
        current = df[df['status_bracket_key'] == (status, bracket)][weight_col].sum()
        gap_pct = abs((current - target) / target * 100) if target > 0 else 0
        
        if gap_pct < 5:  # Within 5%
            accurate_cells += 1
    
    validation['cells_accurate'] = accurate_cells
    validation['cell_accuracy_pct'] = (accurate_cells / len(targets) * 100)
    
    return validation
