"""
Iterative Proportional Fitting (IPF) Calibration

This module implements IPF/raking to simultaneously calibrate tax unit weights
to match both:
1. Filing status distribution (DOTAX targets)
2. AGI bracket distribution (DOTAX Table 12A)

IPF iteratively adjusts weights to satisfy multiple marginal constraints,
finding a balanced solution that minimizes total deviation from both targets.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, List

logger = logging.getLogger(__name__)


# DOTAX Filing Status Totals
DOTAX_FILING_STATUS = {
    'single': 335198,
    'married_filing_jointly': 216358,
    'head_of_household': 67393,
    'married_filing_separately': 16007,
}

# DOTAX AGI Brackets (from Table 12A)
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
    """Assign AGI to DOTAX bracket."""
    for (min_agi, max_agi), _ in DOTAX_AGI_BRACKETS.items():
        if max_agi == float('inf'):
            if agi >= min_agi:
                return (min_agi, max_agi)
        else:
            if min_agi <= agi < max_agi:
                return (min_agi, max_agi)
    return (0, 10000)  # Default


def apply_ipf_calibration(
    tax_units: pd.DataFrame,
    weight_col: str = 'weight',
    output_col: str = 'weight_ipf_calibrated',
    max_iterations: int = 50,
    convergence_threshold: float = 0.01
) -> pd.DataFrame:
    """
    Apply Iterative Proportional Fitting (IPF) calibration.
    
    IPF iteratively adjusts weights to satisfy both filing status and AGI
    bracket constraints simultaneously.
    
    Args:
        tax_units: DataFrame with tax units
        weight_col: Column name for input weights
        output_col: Column name for output calibrated weights
        max_iterations: Maximum number of IPF iterations
        convergence_threshold: Convergence criterion (max % deviation)
        
    Returns:
        DataFrame with IPF-calibrated weights
    """
    logger.info("="*80)
    logger.info("ITERATIVE PROPORTIONAL FITTING (IPF) CALIBRATION")
    logger.info("="*80)
    logger.info(f"\nMax iterations: {max_iterations}")
    logger.info(f"Convergence threshold: {convergence_threshold}%")
    
    df = tax_units.copy()
    
    # Store original weights
    if 'weight_original' not in df.columns:
        df['weight_original'] = df[weight_col].copy()
    
    # Initialize working weight
    df[output_col] = df[weight_col].copy()
    
    # Assign AGI brackets
    df['agi_bracket'] = df['agi'].apply(assign_agi_bracket)
    
    # IPF iteration
    iteration_history = []
    
    for iteration in range(max_iterations):
        logger.info(f"\n--- Iteration {iteration + 1} ---")
        
        # STEP 1: Adjust for filing status constraints
        for status, target in DOTAX_FILING_STATUS.items():
            mask = df['filing_status'] == status
            current = df.loc[mask, output_col].sum()
            
            if current > 0:
                factor = target / current
                df.loc[mask, output_col] *= factor
                logger.debug(f"  Filing status '{status}': factor={factor:.4f}")
        
        # STEP 2: Adjust for AGI bracket constraints
        for bracket_key, target in DOTAX_AGI_BRACKETS.items():
            mask = df['agi_bracket'] == bracket_key
            current = df.loc[mask, output_col].sum()
            
            if current > 0:
                factor = target / current
                df.loc[mask, output_col] *= factor
                
                if max_agi := bracket_key[1]:
                    if max_agi == float('inf'):
                        label = f"${bracket_key[0]//1000}k+"
                    else:
                        label = f"${bracket_key[0]//1000}k-${max_agi//1000}k"
                    logger.debug(f"  AGI bracket {label}: factor={factor:.4f}")
        
        # Check convergence
        filing_status_gaps = []
        for status, target in DOTAX_FILING_STATUS.items():
            current = df[df['filing_status'] == status][output_col].sum()
            gap_pct = abs((current - target) / target * 100)
            filing_status_gaps.append(gap_pct)
        
        agi_bracket_gaps = []
        for bracket_key, target in DOTAX_AGI_BRACKETS.items():
            current = df[df['agi_bracket'] == bracket_key][output_col].sum()
            gap_pct = abs((current - target) / target * 100)
            agi_bracket_gaps.append(gap_pct)
        
        max_filing_gap = max(filing_status_gaps)
        max_agi_gap = max(agi_bracket_gaps)
        avg_filing_gap = np.mean(filing_status_gaps)
        avg_agi_gap = np.mean(agi_bracket_gaps)
        
        iteration_history.append({
            'iteration': iteration + 1,
            'max_filing_gap': max_filing_gap,
            'max_agi_gap': max_agi_gap,
            'avg_filing_gap': avg_filing_gap,
            'avg_agi_gap': avg_agi_gap
        })
        
        logger.info(f"  Filing status: max gap={max_filing_gap:.3f}%, avg gap={avg_filing_gap:.3f}%")
        logger.info(f"  AGI brackets:  max gap={max_agi_gap:.3f}%, avg gap={avg_agi_gap:.3f}%")
        
        # Check convergence
        if max_filing_gap < convergence_threshold and max_agi_gap < convergence_threshold:
            logger.info(f"\n✅ Converged after {iteration + 1} iterations!")
            logger.info(f"   Max filing status gap: {max_filing_gap:.3f}%")
            logger.info(f"   Max AGI bracket gap: {max_agi_gap:.3f}%")
            break
    else:
        logger.warning(f"\n⚠️  Did not fully converge after {max_iterations} iterations")
        logger.warning(f"   Max filing status gap: {max_filing_gap:.3f}%")
        logger.warning(f"   Max AGI bracket gap: {max_agi_gap:.3f}%")
    
    # Report final results
    logger.info("\n" + "="*80)
    logger.info("FINAL IPF CALIBRATION RESULTS")
    logger.info("="*80)
    
    # Filing status validation
    logger.info("\nFiling Status Distribution:")
    logger.info(f"{'Status':<30} {'Before':>12} {'After':>12} {'Target':>12} {'Gap':>10}")
    logger.info("-"*80)
    
    for status, target in DOTAX_FILING_STATUS.items():
        before = df[df['filing_status'] == status][weight_col].sum()
        after = df[df['filing_status'] == status][output_col].sum()
        gap_pct = ((after - target) / target * 100) if target > 0 else 0
        
        marker = " ✅" if abs(gap_pct) < 1.0 else " ⚠️" if abs(gap_pct) < 5.0 else " ❌"
        logger.info(f"{status:<30} {before:>12,.0f} {after:>12,.0f} {target:>12,} {gap_pct:>+9.2f}%{marker}")
    
    # AGI bracket validation
    logger.info("\nAGI Bracket Distribution:")
    logger.info(f"{'Bracket':<20} {'Before':>12} {'After':>12} {'Target':>12} {'Gap':>10}")
    logger.info("-"*80)
    
    for bracket_key, target in DOTAX_AGI_BRACKETS.items():
        min_agi, max_agi = bracket_key
        
        if max_agi == float('inf'):
            label = f"${min_agi//1000}k+"
        else:
            label = f"${min_agi//1000}k-${max_agi//1000}k"
        
        before = df[df['agi_bracket'] == bracket_key][weight_col].sum()
        after = df[df['agi_bracket'] == bracket_key][output_col].sum()
        gap_pct = ((after - target) / target * 100) if target > 0 else 0
        
        bracket_marker = " 🎯" if min_agi == 50000 and max_agi == 75000 else ""
        marker = " ✅" if abs(gap_pct) < 1.0 else " ⚠️" if abs(gap_pct) < 5.0 else " ❌"
        logger.info(f"{label:<20} {before:>12,.0f} {after:>12,.0f} {target:>12,} {gap_pct:>+9.1f}%{bracket_marker}{marker}")
    
    # Summary statistics
    filing_accurate = sum(1 for status, target in DOTAX_FILING_STATUS.items()
                         if abs((df[df['filing_status'] == status][output_col].sum() - target) / target * 100) < 1.0)
    agi_accurate = sum(1 for bracket_key, target in DOTAX_AGI_BRACKETS.items()
                      if abs((df[df['agi_bracket'] == bracket_key][output_col].sum() - target) / target * 100) < 1.0)
    
    logger.info("\n" + "="*80)
    logger.info("ACCURACY SUMMARY")
    logger.info("="*80)
    logger.info(f"\nFiling statuses within ±1%: {filing_accurate}/{len(DOTAX_FILING_STATUS)} ({filing_accurate/len(DOTAX_FILING_STATUS)*100:.1f}%)")
    logger.info(f"AGI brackets within ±1%:    {agi_accurate}/{len(DOTAX_AGI_BRACKETS)} ({agi_accurate/len(DOTAX_AGI_BRACKETS)*100:.1f}%)")
    logger.info(f"Total iterations: {len(iteration_history)}")
    
    return df


def validate_ipf_calibration(
    tax_units: pd.DataFrame,
    weight_col: str = 'weight_ipf_calibrated'
) -> Dict:
    """
    Validate IPF calibration results.
    
    Args:
        tax_units: Calibrated tax units
        weight_col: Column name for calibrated weights
        
    Returns:
        Dictionary with validation metrics
    """
    df = tax_units.copy()
    df['agi_bracket'] = df['agi'].apply(assign_agi_bracket)
    
    validation = {
        'total_returns': df[weight_col].sum(),
        'filing_status': {},
        'agi_brackets': {},
        'summary': {}
    }
    
    # Filing status validation
    for status, target in DOTAX_FILING_STATUS.items():
        current = df[df['filing_status'] == status][weight_col].sum()
        gap_pct = abs((current - target) / target * 100) if target > 0 else 0
        
        validation['filing_status'][status] = {
            'current': current,
            'target': target,
            'gap_pct': gap_pct,
            'accurate_1pct': gap_pct < 1.0,
            'accurate_5pct': gap_pct < 5.0
        }
    
    # AGI bracket validation
    for bracket_key, target in DOTAX_AGI_BRACKETS.items():
        current = df[df['agi_bracket'] == bracket_key][weight_col].sum()
        gap_pct = abs((current - target) / target * 100) if target > 0 else 0
        
        validation['agi_brackets'][bracket_key] = {
            'current': current,
            'target': target,
            'gap_pct': gap_pct,
            'accurate_1pct': gap_pct < 1.0,
            'accurate_5pct': gap_pct < 5.0
        }
    
    # Summary statistics
    filing_accurate_1pct = sum(1 for v in validation['filing_status'].values() if v['accurate_1pct'])
    agi_accurate_1pct = sum(1 for v in validation['agi_brackets'].values() if v['accurate_1pct'])
    
    validation['summary'] = {
        'filing_status_1pct_accuracy': filing_accurate_1pct / len(DOTAX_FILING_STATUS) * 100,
        'agi_bracket_1pct_accuracy': agi_accurate_1pct / len(DOTAX_AGI_BRACKETS) * 100,
        'overall_accuracy': (filing_accurate_1pct + agi_accurate_1pct) / (len(DOTAX_FILING_STATUS) + len(DOTAX_AGI_BRACKETS)) * 100
    }
    
    return validation
