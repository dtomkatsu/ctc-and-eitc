"""
Gentle SOI Calibration Options

This module provides several alternative calibration approaches that make smaller,
more targeted adjustments to filing status distributions rather than massive changes.
"""

import random
import hashlib
import logging
from typing import List, Dict, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)


def option_1_partial_calibration(tax_units: List[Dict[str, Any]], 
                                target_joint_pct: float = 0.36,
                                max_adjustment_pct: float = 0.02) -> List[Dict[str, Any]]:
    """
    Option 1: Partial Calibration - Only adjust if gap is large, and cap the adjustment.
    
    This approach only makes adjustments if the gap is significant (>2%) and limits
    the maximum adjustment to avoid massive changes.
    
    Args:
        tax_units: List of tax unit dictionaries
        target_joint_pct: Target percentage of joint filers
        max_adjustment_pct: Maximum percentage point adjustment allowed
        
    Returns:
        List[Dict[str, Any]]: Gently adjusted tax units
    """
    if not tax_units:
        return tax_units
        
    current_joint = sum(1 for u in tax_units if u.get('filing_status') == 'joint')
    current_total = len(tax_units)
    current_pct = current_joint / current_total if current_total > 0 else 0
    
    gap = target_joint_pct - current_pct
    
    logger.info(f"Option 1 - Partial Calibration: Current={current_pct:.1%}, Target={target_joint_pct:.1%}, Gap={gap:.1%}")
    
    # Only adjust if gap is significant
    if abs(gap) < 0.02:  # Less than 2% gap
        logger.info("Gap is small (<2%), no adjustment needed")
        return tax_units
    
    # Cap the adjustment to max_adjustment_pct
    actual_adjustment = min(abs(gap), max_adjustment_pct)
    if gap < 0:
        actual_adjustment = -actual_adjustment
    
    target_adjusted_pct = current_pct + actual_adjustment
    num_to_convert = int(abs(actual_adjustment) * current_total)
    
    logger.info(f"Making partial adjustment: {actual_adjustment:.1%} ({num_to_convert} units)")
    
    tax_units_copy = [unit.copy() for unit in tax_units]
    
    if actual_adjustment > 0:
        # Convert some single to joint
        single_units = [u for u in tax_units_copy if u.get('filing_status') == 'single']
        single_units.sort(key=lambda x: x.get('income', 0), reverse=True)
        
        for unit in single_units[:num_to_convert]:
            unit['filing_status'] = 'joint'
            unit['calibrated'] = True
            unit['calibration_method'] = 'partial'
    else:
        # Convert some joint to single
        joint_units = [u for u in tax_units_copy if u.get('filing_status') == 'joint']
        joint_units.sort(key=lambda x: x.get('income', 0))
        
        for unit in joint_units[:num_to_convert]:
            unit['filing_status'] = 'single'
            unit['calibrated'] = True
            unit['calibration_method'] = 'partial'
    
    final_joint = sum(1 for u in tax_units_copy if u.get('filing_status') == 'joint')
    final_pct = final_joint / current_total
    calibrated_count = sum(1 for u in tax_units_copy if u.get('calibrated', False))
    
    logger.info(f"Partial calibration: {current_pct:.1%} -> {final_pct:.1%} ({calibrated_count} units adjusted)")
    
    return tax_units_copy


def option_2_probability_nudging(tax_units: List[Dict[str, Any]], 
                                target_joint_pct: float = 0.36,
                                nudge_strength: float = 0.1) -> List[Dict[str, Any]]:
    """
    Option 2: Probability Nudging - Adjust filing decisions for borderline cases only.
    
    This approach identifies tax units that were close to the filing status boundary
    and re-evaluates their filing status with a small probability nudge.
    
    Args:
        tax_units: List of tax unit dictionaries
        target_joint_pct: Target percentage of joint filers
        nudge_strength: How much to nudge probabilities (0.0-1.0)
        
    Returns:
        List[Dict[str, Any]]: Nudged tax units
    """
    if not tax_units:
        return tax_units
        
    current_joint = sum(1 for u in tax_units if u.get('filing_status') == 'joint')
    current_total = len(tax_units)
    current_pct = current_joint / current_total if current_total > 0 else 0
    
    gap = target_joint_pct - current_pct
    
    logger.info(f"Option 2 - Probability Nudging: Current={current_pct:.1%}, Target={target_joint_pct:.1%}, Gap={gap:.1%}")
    
    if abs(gap) < 0.01:  # Within 1%
        logger.info("Already close to target, no nudging needed")
        return tax_units
    
    tax_units_copy = [unit.copy() for unit in tax_units]
    
    # Identify borderline cases (those with income between $30K-$80K)
    borderline_units = [
        u for u in tax_units_copy 
        if 30000 <= u.get('income', 0) <= 80000
    ]
    
    # Calculate how many borderline units to adjust
    max_borderline_to_adjust = min(len(borderline_units), int(abs(gap) * current_total * 1.5))
    
    if gap > 0:
        # Need more joint filers - nudge single filers toward joint
        single_borderline = [u for u in borderline_units if u.get('filing_status') == 'single']
        single_borderline.sort(key=lambda x: x.get('income', 0), reverse=True)
        
        for i, unit in enumerate(single_borderline[:max_borderline_to_adjust]):
            # Use deterministic randomness based on unit ID
            seed = hash(str(unit.get('filer_id', i))) % 1000000
            random.seed(seed)
            
            # Apply nudge - increase probability of joint filing
            if random.random() < nudge_strength:
                unit['filing_status'] = 'joint'
                unit['calibrated'] = True
                unit['calibration_method'] = 'probability_nudge'
    else:
        # Need fewer joint filers - nudge joint filers toward single
        joint_borderline = [u for u in borderline_units if u.get('filing_status') == 'joint']
        joint_borderline.sort(key=lambda x: x.get('income', 0))
        
        for i, unit in enumerate(joint_borderline[:max_borderline_to_adjust]):
            seed = hash(str(unit.get('filer_id', i))) % 1000000
            random.seed(seed)
            
            if random.random() < nudge_strength:
                unit['filing_status'] = 'single'
                unit['calibrated'] = True
                unit['calibration_method'] = 'probability_nudge'
    
    final_joint = sum(1 for u in tax_units_copy if u.get('filing_status') == 'joint')
    final_pct = final_joint / current_total
    calibrated_count = sum(1 for u in tax_units_copy if u.get('calibrated', False))
    
    logger.info(f"Probability nudging: {current_pct:.1%} -> {final_pct:.1%} ({calibrated_count} units adjusted)")
    
    return tax_units_copy


def option_3_income_weighted_adjustment(tax_units: List[Dict[str, Any]], 
                                      target_joint_pct: float = 0.36,
                                      confidence_threshold: float = 0.7) -> List[Dict[str, Any]]:
    """
    Option 3: Income-Weighted Adjustment - Only adjust units where we have low confidence.
    
    This approach calculates a "confidence score" for each filing status decision
    and only adjusts those with low confidence scores.
    
    Args:
        tax_units: List of tax unit dictionaries
        target_joint_pct: Target percentage of joint filers
        confidence_threshold: Only adjust units with confidence below this threshold
        
    Returns:
        List[Dict[str, Any]]: Adjusted tax units
    """
    if not tax_units:
        return tax_units
        
    current_joint = sum(1 for u in tax_units if u.get('filing_status') == 'joint')
    current_total = len(tax_units)
    current_pct = current_joint / current_total if current_total > 0 else 0
    
    gap = target_joint_pct - current_pct
    
    logger.info(f"Option 3 - Income-Weighted: Current={current_pct:.1%}, Target={target_joint_pct:.1%}, Gap={gap:.1%}")
    
    if abs(gap) < 0.01:
        logger.info("Already close to target")
        return tax_units
    
    tax_units_copy = [unit.copy() for unit in tax_units]
    
    # Calculate confidence scores for each unit
    for unit in tax_units_copy:
        income = unit.get('income', 0)
        filing_status = unit.get('filing_status')
        
        # High income -> high confidence in joint filing
        # Low income -> high confidence in single filing
        # Middle income -> lower confidence
        if filing_status == 'joint':
            if income > 75000:
                confidence = 0.9  # High confidence
            elif income > 40000:
                confidence = 0.6  # Medium confidence
            else:
                confidence = 0.3  # Low confidence
        else:  # single
            if income < 25000:
                confidence = 0.9  # High confidence
            elif income < 60000:
                confidence = 0.6  # Medium confidence
            else:
                confidence = 0.3  # Low confidence
        
        unit['confidence_score'] = confidence
    
    # Find low-confidence units that could be adjusted
    low_confidence_units = [u for u in tax_units_copy if u.get('confidence_score', 1.0) < confidence_threshold]
    
    num_to_convert = min(len(low_confidence_units), int(abs(gap) * current_total))
    
    if gap > 0:
        # Convert low-confidence single filers to joint
        single_low_conf = [u for u in low_confidence_units if u.get('filing_status') == 'single']
        single_low_conf.sort(key=lambda x: x.get('income', 0), reverse=True)
        
        for unit in single_low_conf[:num_to_convert]:
            unit['filing_status'] = 'joint'
            unit['calibrated'] = True
            unit['calibration_method'] = 'income_weighted'
    else:
        # Convert low-confidence joint filers to single
        joint_low_conf = [u for u in low_confidence_units if u.get('filing_status') == 'joint']
        joint_low_conf.sort(key=lambda x: x.get('income', 0))
        
        for unit in joint_low_conf[:num_to_convert]:
            unit['filing_status'] = 'single'
            unit['calibrated'] = True
            unit['calibration_method'] = 'income_weighted'
    
    final_joint = sum(1 for u in tax_units_copy if u.get('filing_status') == 'joint')
    final_pct = final_joint / current_total
    calibrated_count = sum(1 for u in tax_units_copy if u.get('calibrated', False))
    low_conf_count = len(low_confidence_units)
    
    logger.info(f"Income-weighted adjustment: {current_pct:.1%} -> {final_pct:.1%} ({calibrated_count}/{low_conf_count} low-confidence units adjusted)")
    
    return tax_units_copy


def option_4_gradual_convergence(tax_units: List[Dict[str, Any]], 
                                target_joint_pct: float = 0.36,
                                convergence_rate: float = 0.3) -> List[Dict[str, Any]]:
    """
    Option 4: Gradual Convergence - Move only partway toward the target.
    
    This approach moves the distribution only a fraction of the way toward the target,
    allowing for gradual convergence over multiple iterations if needed.
    
    Args:
        tax_units: List of tax unit dictionaries
        target_joint_pct: Target percentage of joint filers
        convergence_rate: Fraction of gap to close (0.0-1.0)
        
    Returns:
        List[Dict[str, Any]]: Gradually adjusted tax units
    """
    if not tax_units:
        return tax_units
        
    current_joint = sum(1 for u in tax_units if u.get('filing_status') == 'joint')
    current_total = len(tax_units)
    current_pct = current_joint / current_total if current_total > 0 else 0
    
    gap = target_joint_pct - current_pct
    
    logger.info(f"Option 4 - Gradual Convergence: Current={current_pct:.1%}, Target={target_joint_pct:.1%}, Gap={gap:.1%}")
    
    if abs(gap) < 0.005:  # Within 0.5%
        logger.info("Already very close to target")
        return tax_units
    
    # Only close a fraction of the gap
    adjustment = gap * convergence_rate
    target_adjusted_pct = current_pct + adjustment
    
    num_to_convert = int(abs(adjustment) * current_total)
    
    logger.info(f"Gradual adjustment: {adjustment:.1%} of {gap:.1%} gap ({num_to_convert} units)")
    
    tax_units_copy = [unit.copy() for unit in tax_units]
    
    if adjustment > 0:
        # Convert some single to joint
        single_units = [u for u in tax_units_copy if u.get('filing_status') == 'single']
        # Prefer middle-income singles (most likely to actually be married)
        single_units.sort(key=lambda x: abs(x.get('income', 0) - 50000))
        
        for unit in single_units[:num_to_convert]:
            unit['filing_status'] = 'joint'
            unit['calibrated'] = True
            unit['calibration_method'] = 'gradual_convergence'
    else:
        # Convert some joint to single
        joint_units = [u for u in tax_units_copy if u.get('filing_status') == 'joint']
        # Prefer lower-income joints (less likely to be married)
        joint_units.sort(key=lambda x: x.get('income', 0))
        
        for unit in joint_units[:num_to_convert]:
            unit['filing_status'] = 'single'
            unit['calibrated'] = True
            unit['calibration_method'] = 'gradual_convergence'
    
    final_joint = sum(1 for u in tax_units_copy if u.get('filing_status') == 'joint')
    final_pct = final_joint / current_total
    calibrated_count = sum(1 for u in tax_units_copy if u.get('calibrated', False))
    
    logger.info(f"Gradual convergence: {current_pct:.1%} -> {final_pct:.1%} ({calibrated_count} units adjusted)")
    
    return tax_units_copy


def option_5_hybrid_approach(tax_units: List[Dict[str, Any]], 
                           target_joint_pct: float = 0.36,
                           max_total_adjustment: float = 0.015) -> List[Dict[str, Any]]:
    """
    Option 5: Hybrid Approach - Combine multiple gentle methods.
    
    This approach uses a combination of the above methods, prioritizing
    the gentlest adjustments first.
    
    Args:
        tax_units: List of tax unit dictionaries
        target_joint_pct: Target percentage of joint filers
        max_total_adjustment: Maximum total adjustment allowed
        
    Returns:
        List[Dict[str, Any]]: Hybrid-adjusted tax units
    """
    if not tax_units:
        return tax_units
        
    current_joint = sum(1 for u in tax_units if u.get('filing_status') == 'joint')
    current_total = len(tax_units)
    current_pct = current_joint / current_total if current_total > 0 else 0
    
    gap = target_joint_pct - current_pct
    
    logger.info(f"Option 5 - Hybrid Approach: Current={current_pct:.1%}, Target={target_joint_pct:.1%}, Gap={gap:.1%}")
    
    if abs(gap) < 0.005:
        logger.info("Already very close to target")
        return tax_units
    
    # Start with the original units
    result_units = [unit.copy() for unit in tax_units]
    
    # Step 1: Apply probability nudging (gentlest)
    if abs(gap) > 0.01:
        result_units = option_2_probability_nudging(result_units, target_joint_pct, nudge_strength=0.05)
        
        # Recalculate gap
        current_joint = sum(1 for u in result_units if u.get('filing_status') == 'joint')
        current_pct = current_joint / current_total
        gap = target_joint_pct - current_pct
        
        logger.info(f"After probability nudging: {current_pct:.1%}, remaining gap: {gap:.1%}")
    
    # Step 2: Apply income-weighted adjustment if still needed
    if abs(gap) > 0.008:
        remaining_adjustment = min(abs(gap), max_total_adjustment * 0.7)
        if gap < 0:
            remaining_adjustment = -remaining_adjustment
        
        intermediate_target = current_pct + remaining_adjustment
        result_units = option_3_income_weighted_adjustment(result_units, intermediate_target, confidence_threshold=0.6)
        
        # Recalculate gap
        current_joint = sum(1 for u in result_units if u.get('filing_status') == 'joint')
        current_pct = current_joint / current_total
        gap = target_joint_pct - current_pct
        
        logger.info(f"After income-weighted adjustment: {current_pct:.1%}, remaining gap: {gap:.1%}")
    
    # Step 3: Apply partial calibration as final step if still needed
    if abs(gap) > 0.005:
        final_adjustment = min(abs(gap), max_total_adjustment * 0.3)
        if gap < 0:
            final_adjustment = -final_adjustment
            
        final_target = current_pct + final_adjustment
        result_units = option_1_partial_calibration(result_units, final_target, max_adjustment_pct=final_adjustment)
    
    final_joint = sum(1 for u in result_units if u.get('filing_status') == 'joint')
    final_pct = final_joint / current_total
    calibrated_count = sum(1 for u in result_units if u.get('calibrated', False))
    
    logger.info(f"Hybrid approach complete: {tax_units[0].get('filing_status') if tax_units else 'N/A'} -> {final_pct:.1%} ({calibrated_count} total units adjusted)")
    
    return result_units


def compare_calibration_options(tax_units: List[Dict[str, Any]], 
                              target_joint_pct: float = 0.36) -> Dict[str, Dict[str, Any]]:
    """
    Compare all calibration options and return summary statistics.
    
    Args:
        tax_units: List of tax unit dictionaries
        target_joint_pct: Target percentage of joint filers
        
    Returns:
        Dict with comparison results for each option
    """
    if not tax_units:
        return {}
    
    original_joint = sum(1 for u in tax_units if u.get('filing_status') == 'joint')
    original_total = len(tax_units)
    original_pct = original_joint / original_total
    
    results = {
        'original': {
            'joint_pct': original_pct,
            'units_adjusted': 0,
            'adjustment_pct': 0.0,
            'distance_from_target': abs(original_pct - target_joint_pct)
        }
    }
    
    options = [
        ('option_1_partial', option_1_partial_calibration),
        ('option_2_probability', option_2_probability_nudging),
        ('option_3_income_weighted', option_3_income_weighted_adjustment),
        ('option_4_gradual', option_4_gradual_convergence),
        ('option_5_hybrid', option_5_hybrid_approach)
    ]
    
    for name, func in options:
        try:
            adjusted_units = func(tax_units, target_joint_pct)
            
            final_joint = sum(1 for u in adjusted_units if u.get('filing_status') == 'joint')
            final_pct = final_joint / original_total
            units_adjusted = sum(1 for u in adjusted_units if u.get('calibrated', False))
            adjustment_pct = units_adjusted / original_total
            
            results[name] = {
                'joint_pct': final_pct,
                'units_adjusted': units_adjusted,
                'adjustment_pct': adjustment_pct,
                'distance_from_target': abs(final_pct - target_joint_pct),
                'improvement': abs(original_pct - target_joint_pct) - abs(final_pct - target_joint_pct)
            }
            
        except Exception as e:
            logger.error(f"Error testing {name}: {e}")
            results[name] = {'error': str(e)}
    
    return results
