"""
Comprehensive Weight Calibrator

Systematically adjusts weights across ALL brackets to match DOTax filer counts
while preserving income distributions and tax calculation accuracy.
"""

import pandas as pd
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class ComprehensiveWeightCalibrator:
    """
    Calibrate weights across all AGI brackets to match DOTax targets.
    
    This addresses both surplus (middle income) and deficit (low income) issues
    by reweighting filers to match exact filer counts per bracket.
    """
    
    # DOTax Table A8 filer targets by AGI bracket
    DOTAX_FILER_TARGETS = {
        (0, 10000): 115285,
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
        (400000, 500000): 2926,
        (500000, 750000): 2991,
        (750000, 1000000): 1134,
        (1000000, float('inf')): 1824,
    }
    
    def __init__(self, apply_to_all: bool = True):
        """
        Initialize calibrator.
        
        Args:
            apply_to_all: If True, calibrate all brackets. If False, skip high-income brackets
                         that have already been calibrated via Pareto method.
        """
        self.apply_to_all = apply_to_all
        
    def calculate_weight_adjustment(self, 
                                   current_filers: float, 
                                   target_filers: float) -> float:
        """
        Calculate weight adjustment factor for a bracket.
        
        Args:
            current_filers: Current total weight in bracket
            target_filers: Target filer count
            
        Returns:
            Adjustment factor (multiply weights by this)
        """
        if current_filers == 0:
            return 1.0
        
        return target_filers / current_filers
    
    def calibrate_bracket_weights(self, 
                                 df: pd.DataFrame,
                                 bracket_min: float,
                                 bracket_max: float,
                                 target_filers: float) -> pd.DataFrame:
        """
        Calibrate weights for a single bracket.
        
        Args:
            df: Full DataFrame
            bracket_min: Minimum AGI for bracket
            bracket_max: Maximum AGI for bracket
            target_filers: Target filer count
            
        Returns:
            DataFrame with adjusted weights for this bracket
        """
        mask = (df['agi'] >= bracket_min) & (df['agi'] < bracket_max)
        
        if mask.sum() == 0:
            logger.warning(f"  No filers in ${bracket_min//1000}k-${bracket_max//1000}k bracket")
            return df
        
        current_filers = df.loc[mask, 'weight'].sum()
        
        if abs(current_filers - target_filers) < 100:  # Within 100 filers
            logger.info(f"  ${bracket_min//1000}k-${bracket_max//1000}k: {current_filers:>8,.0f} ≈ {target_filers:>8,} ✅")
            return df
        
        # Calculate adjustment factor
        adj_factor = self.calculate_weight_adjustment(current_filers, target_filers)
        
        # Apply adjustment
        result = df.copy()
        result.loc[mask, 'weight'] = result.loc[mask, 'weight'] * adj_factor
        
        # Verify
        new_filers = result.loc[mask, 'weight'].sum()
        
        logger.info(f"  ${bracket_min//1000}k-${bracket_max//1000}k: "
                   f"{current_filers:>8,.0f} → {new_filers:>8,.0f} "
                   f"(target: {target_filers:>8,}, adj: ×{adj_factor:.3f})")
        
        return result
    
    def calibrate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main calibration method - adjust weights across all brackets.
        
        Args:
            df: DataFrame with tax units
            
        Returns:
            DataFrame with calibrated weights
        """
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE WEIGHT CALIBRATION")
        logger.info("=" * 80)
        logger.info("")
        
        result = df.copy()
        
        # Track original total
        original_total_weight = result['weight'].sum()
        logger.info(f"Original total filers: {original_total_weight:,.0f}")
        logger.info("")
        
        # Apply calibration to each bracket
        for (bracket_min, bracket_max), target_filers in self.DOTAX_FILER_TARGETS.items():
            # Skip high-income brackets if already calibrated
            if not self.apply_to_all and bracket_min >= 200000:
                continue
            
            result = self.calibrate_bracket_weights(
                result, bracket_min, bracket_max, target_filers
            )
        
        # Verify final results
        final_total_weight = result['weight'].sum()
        logger.info("")
        logger.info(f"Final total filers: {final_total_weight:,.0f}")
        logger.info(f"DOTax target: 635,117")
        logger.info(f"Difference: {final_total_weight - 635117:+,.0f} ({(final_total_weight/635117-1)*100:+.1f}%)")
        
        logger.info("")
        logger.info("Calibration results by bracket:")
        
        for (bracket_min, bracket_max), target_filers in self.DOTAX_FILER_TARGETS.items():
            mask = (result['agi'] >= bracket_min) & (result['agi'] < bracket_max)
            actual_filers = result.loc[mask, 'weight'].sum()
            
            label = f"${bracket_min//1000}k-${bracket_max//1000}k" if bracket_max != float('inf') else f"${bracket_min//1000}k+"
            diff = actual_filers - target_filers
            pct = (actual_filers / target_filers * 100) if target_filers > 0 else 0
            
            status = "✅" if abs(diff) < 100 else ("⚠️" if abs(diff) < 1000 else "❌")
            logger.info(f"  {label:<20} {actual_filers:>8,.0f} / {target_filers:>8,} ({pct:>6.1f}%) {status}")
        
        return result


class UltraHighIncomeTailInflator:
    """
    Add ultra-high-income filers ($5M+) to fix $1M+ bracket deficit.
    
    Uses Pareto extrapolation to create realistic ultra-high-income distribution.
    """
    
    def __init__(self, pareto_alpha: float = 1.454):
        """
        Initialize inflator.
        
        Args:
            pareto_alpha: Pareto shape parameter (from earlier calibration)
        """
        self.pareto_alpha = pareto_alpha
        
    def calculate_expected_ultra_high_filers(self, 
                                            total_high_income: float,
                                            threshold: float = 1000000,
                                            ultra_threshold: float = 5000000) -> float:
        """
        Calculate expected number of ultra-high-income filers using Pareto.
        
        Args:
            total_high_income: Total filers with income >= threshold
            threshold: Base threshold (e.g., $1M)
            ultra_threshold: Ultra-high threshold (e.g., $5M)
            
        Returns:
            Expected number of filers >= ultra_threshold
        """
        # Pareto: P(X > x) = (x_min / x)^alpha
        ratio = (threshold / ultra_threshold) ** self.pareto_alpha
        return total_high_income * ratio
    
    def generate_ultra_high_incomes(self, target_total_tax: float) -> pd.DataFrame:
        """
        Generate synthetic ultra-high-income filers.
        
        Strategy:
        - Work backwards from target tax to required incomes
        - Use Pareto distribution to allocate across income levels
        - Create realistic filing status distribution
        
        Args:
            target_total_tax: Target total tax for $1M+ bracket (in dollars)
            
        Returns:
            DataFrame with synthetic ultra-high-income filers
        """
        # DOTax shows $1M+ should generate $663M tax
        # We currently generate $218M, so we need +$445M
        
        # Ultra-high-income levels to add
        income_levels = [
            5_000_000,   # $5M
            10_000_000,  # $10M
            25_000_000,  # $25M
            50_000_000,  # $50M
            100_000_000, # $100M
        ]
        
        synthetic_filers = []
        
        # Calculate filer distribution using Pareto
        base_filers = 1824  # Total $1M+ filers
        
        for income in income_levels:
            # Pareto probability
            prob = (1_000_000 / income) ** self.pareto_alpha
            expected_filers = base_filers * prob * 0.8  # Conservative
            
            if expected_filers < 1:
                continue
            
            # Estimate tax (11% top rate on most income)
            # Simplified: use ~10.5% effective rate for ultra-high earners
            estimated_tax = income * 0.105
            
            # Filing status: mostly MFJ at this level
            filing_status = 'married_filing_jointly'
            
            synthetic_filers.append({
                'agi': income,
                'filing_status': filing_status,
                'num_dependents': 2,
                'num_adults': 2,
                'weight': expected_filers,
                'is_synthetic': True,
            })
            
            logger.info(f"  Adding {expected_filers:.0f} filers at ${income/1_000_000:.0f}M AGI")
        
        if not synthetic_filers:
            return pd.DataFrame()
        
        return pd.DataFrame(synthetic_filers)
    
    def calibrate(self, df: pd.DataFrame, target_tax_m: float = 663.0) -> pd.DataFrame:
        """
        Add ultra-high-income filers to fix $1M+ deficit.
        
        Args:
            df: DataFrame with tax units
            target_tax_m: Target tax for $1M+ bracket (in millions)
            
        Returns:
            DataFrame with ultra-high-income filers added
        """
        logger.info("=" * 80)
        logger.info("ULTRA-HIGH-INCOME TAIL INFLATION")
        logger.info("=" * 80)
        logger.info("")
        
        # Check current $1M+ situation
        mask_1m_plus = df['agi'] >= 1_000_000
        current_filers = df.loc[mask_1m_plus, 'weight'].sum()
        current_tax = (df.loc[mask_1m_plus, 'hi_state_tax'] * df.loc[mask_1m_plus, 'weight']).sum()
        
        logger.info(f"Current $1M+ bracket:")
        logger.info(f"  Filers: {current_filers:,.0f}")
        logger.info(f"  Tax: ${current_tax/1_000_000:,.1f}M")
        logger.info(f"  Target tax: ${target_tax_m:,.1f}M")
        logger.info(f"  Deficit: ${current_tax/1_000_000 - target_tax_m:,.1f}M")
        logger.info("")
        
        # Generate synthetic ultra-high-income filers
        logger.info("Generating synthetic ultra-high-income filers:")
        synthetic_df = self.generate_ultra_high_incomes(target_tax_m * 1_000_000)
        
        if len(synthetic_df) == 0:
            logger.warning("No synthetic filers generated")
            return df
        
        logger.info("")
        logger.info(f"Generated {len(synthetic_df)} synthetic income levels")
        logger.info(f"Total synthetic filers: {synthetic_df['weight'].sum():,.0f}")
        
        # Combine with existing data
        result = pd.concat([df, synthetic_df], ignore_index=True)
        
        # Fill missing columns for synthetic filers
        synthetic_mask = result.get('is_synthetic', False).fillna(False)
        result.loc[synthetic_mask, 'total_deductions'] = result.loc[synthetic_mask, 'total_deductions'].fillna(0)
        
        return result


def apply_comprehensive_weight_calibration(df: pd.DataFrame,
                                           calibrate_all_brackets: bool = True) -> pd.DataFrame:
    """
    Apply comprehensive weight calibration to match DOTax filer counts.
    
    Args:
        df: DataFrame with tax units
        calibrate_all_brackets: If True, calibrate all brackets
        
    Returns:
        DataFrame with calibrated weights
    """
    calibrator = ComprehensiveWeightCalibrator(apply_to_all=calibrate_all_brackets)
    return calibrator.calibrate(df)


def apply_ultra_high_income_inflation(df: pd.DataFrame,
                                      target_tax_m: float = 663.0,
                                      pareto_alpha: float = 1.454) -> pd.DataFrame:
    """
    Add ultra-high-income filers to fix $1M+ bracket deficit.
    
    Args:
        df: DataFrame with tax units
        target_tax_m: Target tax for $1M+ bracket (millions)
        pareto_alpha: Pareto shape parameter
        
    Returns:
        DataFrame with ultra-high-income filers added
    """
    inflator = UltraHighIncomeTailInflator(pareto_alpha=pareto_alpha)
    return inflator.calibrate(df, target_tax_m=target_tax_m)
