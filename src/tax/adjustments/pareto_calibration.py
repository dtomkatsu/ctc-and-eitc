"""
DEPRECATED: Superseded by SimultaneousCalibrator (iterative raking/IPF across all 15 brackets).
Retained for reference only.

Pareto Distribution Calibration for High-Income Filers

Uses Pareto distribution to calibrate high-income tail (AGI >= $200k)
to match DOTax distribution patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class ParetoIncomeCalibrator:
    """
    Calibrate high-income filers using Pareto distribution.
    
    The Pareto distribution is commonly used for income/wealth distributions
    where a small percentage of the population holds a large percentage of income.
    
    P(X > x) = (x_min / x)^alpha
    
    where:
    - x_min = threshold (e.g., $200,000)
    - alpha = shape parameter (controls inequality)
      - Lower alpha (1.5) = more inequality
      - Higher alpha (2.5) = less inequality
    """
    
    # DOTax Table A8 ALL BRACKET targets (SYNCHRONIZED with orchestrator.py)
    DOTAX_ALL_BRACKET_TARGETS = {
        (0, 10000): {'filers': 115285, 'tax_m': 3.0},
        (10000, 20000): {'filers': 64160, 'tax_m': 21.0},
        (20000, 30000): {'filers': 57835, 'tax_m': 51.0},
        (30000, 40000): {'filers': 58135, 'tax_m': 92.0},
        (40000, 50000): {'filers': 53555, 'tax_m': 116.0},
        (50000, 75000): {'filers': 91459, 'tax_m': 293.0},
        (75000, 100000): {'filers': 54976, 'tax_m': 261.0},
        (100000, 150000): {'filers': 62065, 'tax_m': 438.0},
        (150000, 200000): {'filers': 27976, 'tax_m': 294.0},
        (200000, 300000): {'filers': 19015, 'tax_m': 310.0},
        (300000, 400000): {'filers': 5729, 'tax_m': 153.0},
        (400000, 500000): {'filers': 2856, 'tax_m': 101.0},
        (500000, 750000): {'filers': 2549, 'tax_m': 149.0},
        (750000, 1000000): {'filers': 1004, 'tax_m': 85.0},
        (1000000, float('inf')): {'filers': 1824, 'tax_m': 663.0},
    }
    
    # Legacy high-income only targets (for backward compatibility)
    DOTAX_HIGH_INCOME_TARGETS = {
        k: v for k, v in DOTAX_ALL_BRACKET_TARGETS.items() 
        if k[0] >= 200000
    }
    
    def __init__(self, threshold: float = 200000, target_alpha: float = 1.454):
        """
        Initialize calibrator.
        
        Args:
            threshold: AGI threshold for Pareto distribution (default $200k)
            target_alpha: Target Pareto shape parameter to match DOTax
        """
        self.threshold = threshold
        self.target_alpha = target_alpha
        
    def get_pareto_weight_adjustment(self, agi: float, current_weight: float) -> float:
        """
        Calculate weight adjustment factor using Pareto distribution.
        
        Args:
            agi: Adjusted Gross Income
            current_weight: Current filer weight
            
        Returns:
            Adjusted weight
        """
        if agi < self.threshold:
            return current_weight
        
        # Pareto probability: P(X > x) = (x_min / x)^alpha
        # We use this to adjust weights to create correct distribution
        
        # Calculate relative probability at this income level
        # Higher income = lower probability (fewer filers)
        pareto_prob = (self.threshold / agi) ** self.target_alpha
        
        # Scale factor to adjust weights
        # This ensures high-income filers are properly weighted
        adjustment_factor = pareto_prob
        
        return current_weight * adjustment_factor
    
    def calibrate_high_income_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calibrate weights for high-income filers (AGI >= threshold).
        
        Strategy:
        1. Keep low/middle income filers (< $200k) as-is
        2. Adjust high-income weights using Pareto distribution
        3. Scale to match DOTax total filer counts by bracket
        
        Args:
            df: DataFrame with tax units
            
        Returns:
            DataFrame with calibrated weights
        """
        result = df.copy()
        
        # Separate low/middle vs high income
        low_income_mask = result['agi'] < self.threshold
        high_income_mask = result['agi'] >= self.threshold
        
        logger.info(f"Calibrating high-income weights (AGI >= ${self.threshold:,.0f})")
        logger.info(f"  Low/middle income filers: {result[low_income_mask]['weight'].sum():,.0f}")
        logger.info(f"  High income filers (before): {result[high_income_mask]['weight'].sum():,.0f}")
        
        # Apply Pareto-based adjustments to high-income filers
        high_income_df = result[high_income_mask].copy()
        
        # Calculate bracket-specific scale factors to match DOTax targets
        for (min_agi, max_agi), targets in self.DOTAX_HIGH_INCOME_TARGETS.items():
            bracket_mask = (high_income_df['agi'] >= min_agi) & (high_income_df['agi'] < max_agi)
            
            if bracket_mask.sum() == 0:
                logger.warning(f"  No filers in ${min_agi:,.0f}-${max_agi:,.0f} bracket")
                continue
            
            current_filers = high_income_df.loc[bracket_mask, 'weight'].sum()
            target_filers = targets['filers']
            
            if current_filers > 0:
                scale_factor = target_filers / current_filers
                high_income_df.loc[bracket_mask, 'weight'] *= scale_factor
                
                logger.info(f"  ${min_agi//1000:>4}k-${max_agi//1000 if max_agi != float('inf') else 'inf':>4}: "
                          f"{current_filers:>8,.0f} → {target_filers:>8,} (×{scale_factor:.3f})")
        
        # Update the result with calibrated high-income weights
        result.loc[high_income_mask, 'weight'] = high_income_df['weight']
        
        logger.info(f"  High income filers (after): {result[high_income_mask]['weight'].sum():,.0f}")
        logger.info(f"  Total filers: {result['weight'].sum():,.0f}")
        
        return result
    
    def calibrate_all_brackets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calibrate weights for ALL AGI brackets to match DOTAX targets.
        
        This is a comprehensive calibration that ensures filer counts match
        across all brackets, not just high-income ones.
        
        Args:
            df: DataFrame with tax units
            
        Returns:
            DataFrame with calibrated weights for all brackets
        """
        result = df.copy()
        
        logger.info(f"Calibrating ALL AGI brackets to match DOTAX targets")
        
        # Calculate bracket-specific scale factors to match ALL DOTAX targets
        for (min_agi, max_agi), targets in self.DOTAX_ALL_BRACKET_TARGETS.items():
            bracket_mask = (result['agi'] >= min_agi) & (result['agi'] < max_agi)
            
            if bracket_mask.sum() == 0:
                logger.warning(f"  No filers in ${min_agi:,.0f}-${max_agi:,.0f} bracket")
                continue
            
            current_filers = result.loc[bracket_mask, 'weight'].sum()
            target_filers = targets['filers']
            
            if current_filers > 0:
                scale_factor = target_filers / current_filers
                result.loc[bracket_mask, 'weight'] *= scale_factor
                
                bracket_label = f"${min_agi//1000:>3}k-${max_agi//1000 if max_agi != float('inf') else 'inf':>3}k"
                logger.info(f"  {bracket_label}: {current_filers:>8,.0f} → {target_filers:>8,} (×{scale_factor:.3f})")
        
        total_filers = result['weight'].sum()
        logger.info(f"  Total filers after calibration: {total_filers:,.0f}")
        
        return result
    
    def add_synthetic_extreme_high_income(self, 
                                         df: pd.DataFrame,
                                         min_agi: float = 5000000) -> pd.DataFrame:
        """
        Add synthetic filers for extremely high income (e.g., AGI > $5M).
        
        This handles cases where PUMS data doesn't reach the highest income levels.
        
        Args:
            df: DataFrame with tax units
            min_agi: Minimum AGI for synthetic filers
            
        Returns:
            DataFrame with synthetic filers added
        """
        # Check if we have filers in very high brackets
        extreme_high = df[df['agi'] >= min_agi]
        
        if len(extreme_high) > 0:
            logger.info(f"Sufficient coverage for AGI >= ${min_agi:,.0f}, skipping synthetic")
            return df
        
        logger.info(f"Adding synthetic filers for AGI >= ${min_agi:,.0f}")
        
        # Generate synthetic filers using Pareto distribution
        # We'll create a few representative filers at different income levels
        
        synthetic_incomes = [5000000, 7500000, 10000000, 15000000, 25000000]
        synthetic_filers = []
        
        for income in synthetic_incomes:
            # Calculate expected number of filers at this income using Pareto
            # P(X > x) = (threshold / x)^alpha
            prob = (self.threshold / income) ** self.target_alpha
            
            # Total high-income filers
            total_high_income = sum(t['filers'] for t in self.DOTAX_HIGH_INCOME_TARGETS.values())
            expected_filers = total_high_income * prob * 0.1  # Conservative estimate
            
            if expected_filers < 1:
                continue
            
            # Sample filing status from high-income patterns (mostly MFJ)
            filing_status = 'married_filing_jointly'  # 80% of $1M+ filers
            
            synthetic_filers.append({
                'agi': income,
                'filing_status': filing_status,
                'num_dependents': 1,  # Average for high earners
                'weight': expected_filers,
            })
        
        if synthetic_filers:
            synthetic_df = pd.DataFrame(synthetic_filers)
            logger.info(f"  Added {len(synthetic_df)} synthetic filer types")
            logger.info(f"  Total synthetic weight: {synthetic_df['weight'].sum():,.0f}")
            
            # Combine with existing data
            result = pd.concat([df, synthetic_df], ignore_index=True)
            return result
        
        return df
    
    def calibrate(self, df: pd.DataFrame, 
                 add_synthetic: bool = False,
                 calibrate_all_brackets: bool = True) -> pd.DataFrame:
        """
        Main calibration method.
        
        Args:
            df: DataFrame with tax units
            add_synthetic: Whether to add synthetic extreme high-income filers
            calibrate_all_brackets: Whether to calibrate ALL brackets (True) or just high-income (False)
            
        Returns:
            DataFrame with calibrated distribution
        """
        logger.info("=" * 80)
        if calibrate_all_brackets:
            logger.info("COMPREHENSIVE BRACKET CALIBRATION (All AGI Brackets)")
        else:
            logger.info("PARETO HIGH-INCOME CALIBRATION (High-Income Only)")
        logger.info("=" * 80)
        
        # Step 1: Calibrate brackets
        if calibrate_all_brackets:
            result = self.calibrate_all_brackets(df)
        else:
            result = self.calibrate_high_income_weights(df)
        
        # Step 2: Add synthetic filers if needed
        if add_synthetic:
            result = self.add_synthetic_extreme_high_income(result)
        
        # Verify results - check ALL brackets if comprehensive calibration
        logger.info("")
        logger.info("Calibration results:")
        
        targets_to_check = self.DOTAX_ALL_BRACKET_TARGETS if calibrate_all_brackets else self.DOTAX_HIGH_INCOME_TARGETS
        
        for (min_agi, max_agi), targets in targets_to_check.items():
            mask = (result['agi'] >= min_agi) & (result['agi'] < max_agi)
            actual_filers = result[mask]['weight'].sum()
            target_filers = targets['filers']
            
            label = f"${min_agi//1000}k-${max_agi//1000}k" if max_agi != float('inf') else f"${min_agi//1000}k+"
            match_pct = (actual_filers / target_filers * 100) if target_filers > 0 else 0
            
            status = "✅" if abs(match_pct - 100) < 5 else ("⚠️" if abs(match_pct - 100) < 10 else "❌")
            logger.info(f"  {label:<15} {actual_filers:>8,.0f} / {target_filers:>8,} ({match_pct:>6.1f}%) {status}")
        
        return result


def apply_pareto_calibration(df: pd.DataFrame, 
                             threshold: float = 200000,
                             target_alpha: float = 1.454,
                             add_synthetic: bool = False,
                             calibrate_all_brackets: bool = True) -> pd.DataFrame:
    """
    Convenience function to apply Pareto calibration.
    
    Args:
        df: DataFrame with tax units
        threshold: AGI threshold for calibration (default $200k)
        target_alpha: Target Pareto shape parameter (default 1.454 to match DOTax)
        add_synthetic: Whether to add synthetic extreme high-income filers
        calibrate_all_brackets: Whether to calibrate ALL brackets (True) or just high-income (False)
        
    Returns:
        DataFrame with calibrated weights
    """
    calibrator = ParetoIncomeCalibrator(threshold=threshold, target_alpha=target_alpha)
    return calibrator.calibrate(df, add_synthetic=add_synthetic, calibrate_all_brackets=calibrate_all_brackets)
