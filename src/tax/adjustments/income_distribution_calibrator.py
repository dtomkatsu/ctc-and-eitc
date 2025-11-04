"""
Income Distribution Calibrator

Systematically adjusts income distributions within AGI brackets to match
DOTax effective tax rates using principled statistical methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Import AGI benchmarks from validation module
try:
    from src.tax.validation.agi_calibration import AGI_BENCHMARKS
except ImportError:
    # Fallback if import fails
    AGI_BENCHMARKS = [
        (0, 10000, 129376),
        (10000, 20000, 64160),
        (20000, 30000, 57835),
        (30000, 40000, 59827),
        (40000, 50000, 53555),
        (50000, 75000, 91459),
        (75000, 100000, 54976),
        (100000, 150000, 62065),
        (150000, 200000, 27976),
        (200000, 300000, 18937),
        (300000, 400000, 6076),
        (400000, float('inf'), 8875),
    ]


class IncomeDistributionCalibrator:
    """
    Calibrate income distributions within brackets to match target effective rates.
    
    Uses log-normal or Pareto distributions to systematically adjust income
    within brackets while preserving realistic income patterns.
    """
    
    # DOTax Table A8 targets (effective rates on taxable income, before credits)
    DOTAX_TARGETS = {
        (0, 10000): {'filers': 115285, 'tax_m': 3.0, 'eff_rate': 2.3},
        (10000, 20000): {'filers': 64160, 'tax_m': 21.0, 'eff_rate': 3.9},
        (20000, 30000): {'filers': 57835, 'tax_m': 51.0, 'eff_rate': 5.0},
        (30000, 40000): {'filers': 59827, 'tax_m': 92.0, 'eff_rate': 5.6},
        (40000, 50000): {'filers': 53555, 'tax_m': 116.0, 'eff_rate': 6.0},
        (50000, 75000): {'filers': 91459, 'tax_m': 293.0, 'eff_rate': 6.4},
        (75000, 100000): {'filers': 54976, 'tax_m': 261.0, 'eff_rate': 6.7},
        (100000, 150000): {'filers': 62065, 'tax_m': 438.0, 'eff_rate': 7.0},
        (150000, 200000): {'filers': 27976, 'tax_m': 294.0, 'eff_rate': 7.3},
        (200000, 300000): {'filers': 18937, 'tax_m': 310.0, 'eff_rate': 7.6},
        (300000, 400000): {'filers': 6076, 'tax_m': 153.0, 'eff_rate': 7.9},
        (400000, 500000): {'filers': 2926, 'tax_m': 101.0, 'eff_rate': 8.3},
        (500000, 750000): {'filers': 2991, 'tax_m': 149.0, 'eff_rate': 8.7},
        (750000, 1000000): {'filers': 1134, 'tax_m': 85.0, 'eff_rate': 9.1},
        (1000000, float('inf')): {'filers': 1824, 'tax_m': 663.0, 'eff_rate': 9.9},
    }
    
    def __init__(self, threshold: float = 100000, calibrate_agi_first: bool = True):
        """
        Initialize calibrator.
        
        Args:
            threshold: AGI threshold above which to apply effective rate calibration (default $100k)
            calibrate_agi_first: If True, calibrate AGI targets first before effective rates
        """
        self.threshold = threshold
        self.calibrate_agi_first = calibrate_agi_first
        
    def calculate_current_rate(self, df_bracket: pd.DataFrame) -> float:
        """Calculate current effective tax rate for a bracket."""
        if len(df_bracket) == 0:
            return 0.0
        
        total_tax = (df_bracket['hi_state_tax'] * df_bracket['weight']).sum()
        total_taxable = (df_bracket['hi_taxable_income'] * df_bracket['weight']).sum()
        
        return (total_tax / total_taxable * 100) if total_taxable > 0 else 0.0
    
    def calculate_target_income_shift(self, 
                                     current_rate: float, 
                                     target_rate: float,
                                     bracket_min: float,
                                     bracket_max: float) -> float:
        """
        Calculate systematic income shift needed to achieve target rate.
        
        Strategy:
        - If our rate is too LOW, shift income UP within bracket
          (higher income = higher marginal rates = higher effective rate)
        - If our rate is too HIGH, shift income DOWN within bracket
        
        Returns:
            Shift factor (1.0 = no shift, >1.0 = shift up, <1.0 = shift down)
        """
        rate_gap = target_rate - current_rate
        
        if abs(rate_gap) < 0.1:
            return 1.0  # Within tolerance, no adjustment needed
        
        # Estimate how much we need to shift income
        # Marginal rates in Hawaii range from ~1.4% to 11%
        # Higher income means higher marginal rates
        
        # For every 1pp rate gap, we need roughly a 10-15% income shift
        # This is an empirical relationship based on tax bracket structure
        shift_sensitivity = 0.12  # 12% income shift per 1pp rate gap
        
        shift_factor = 1.0 + (rate_gap * shift_sensitivity)
        
        # Bound the shift to reasonable limits (±30%)
        shift_factor = max(0.7, min(1.3, shift_factor))
        
        return shift_factor
    
    def apply_log_normal_redistribution(self,
                                       df_bracket: pd.DataFrame,
                                       shift_factor: float,
                                       bracket_min: float,
                                       bracket_max: float) -> pd.DataFrame:
        """
        Apply log-normal redistribution to shift income systematically.
        
        Log-normal is appropriate because:
        1. Income distributions are typically log-normal
        2. It's bounded below (no negative income)
        3. It has a long right tail (high earners)
        
        Args:
            df_bracket: DataFrame with filers in this bracket
            shift_factor: How much to shift (>1 = shift up, <1 = shift down)
            bracket_min: Minimum AGI for bracket
            bracket_max: Maximum AGI for bracket
            
        Returns:
            DataFrame with adjusted AGI
        """
        result = df_bracket.copy()
        
        if len(result) == 0 or shift_factor == 1.0:
            return result
        
        # Calculate current distribution parameters
        agi_values = result['agi'].values
        
        # Log-transform
        log_agi = np.log(agi_values)
        mean_log = np.mean(log_agi)
        std_log = np.std(log_agi)
        
        # Shift the mean to achieve target
        # Shifting mean in log-space shifts the median in regular space
        target_mean_log = mean_log + np.log(shift_factor)
        
        # Generate new incomes from shifted log-normal
        new_log_agi = mean_log + (log_agi - mean_log) * 1.0 + (target_mean_log - mean_log)
        new_agi = np.exp(new_log_agi)
        
        # Ensure values stay within bracket bounds
        new_agi = np.clip(new_agi, bracket_min, 
                         bracket_max if bracket_max != float('inf') else new_agi.max() * 2)
        
        result['agi'] = new_agi
        
        return result
    
    def apply_percentile_shift(self,
                              df_bracket: pd.DataFrame,
                              shift_factor: float,
                              bracket_min: float,
                              bracket_max: float) -> pd.DataFrame:
        """
        Apply percentile-based shift (simpler, more predictable).
        
        Strategy:
        - Sort incomes by percentile
        - If shift_factor > 1: Move each percentile up proportionally
        - If shift_factor < 1: Move each percentile down proportionally
        - Preserve relative ordering
        
        Args:
            df_bracket: DataFrame with filers in this bracket
            shift_factor: How much to shift (>1 = shift up, <1 = shift down)
            bracket_min: Minimum AGI for bracket
            bracket_max: Maximum AGI for bracket
            
        Returns:
            DataFrame with adjusted AGI
        """
        result = df_bracket.copy()
        
        if len(result) == 0 or shift_factor == 1.0:
            return result
        
        # Calculate current position within bracket
        agi_values = result['agi'].values
        
        # Normalize to [0, 1] within bracket
        if bracket_max == float('inf'):
            bracket_max = agi_values.max() * 1.5
        
        bracket_range = bracket_max - bracket_min
        normalized_position = (agi_values - bracket_min) / bracket_range
        
        # Apply power transformation to shift distribution
        # shift_factor > 1: push towards top (higher incomes)
        # shift_factor < 1: push towards bottom (lower incomes)
        power = np.log(shift_factor) / np.log(2) + 1  # Maps shift_factor to power
        shifted_position = normalized_position ** power
        
        # Convert back to AGI
        new_agi = bracket_min + shifted_position * bracket_range
        
        # Ensure within bounds
        new_agi = np.clip(new_agi, bracket_min, bracket_max if bracket_max != float('inf') else new_agi.max())
        
        result['agi'] = new_agi
        
        return result
    
    def calibrate_bracket(self,
                         df: pd.DataFrame,
                         bracket_min: float,
                         bracket_max: float,
                         target_rate: float,
                         method: str = 'percentile') -> pd.DataFrame:
        """
        Calibrate income distribution for a single bracket.
        
        Args:
            df: Full DataFrame
            bracket_min: Minimum AGI for bracket
            bracket_max: Maximum AGI for bracket
            target_rate: Target effective tax rate (%)
            method: 'percentile' or 'lognormal'
            
        Returns:
            DataFrame with adjusted incomes for this bracket
        """
        mask = (df['agi'] >= bracket_min) & (df['agi'] < bracket_max)
        df_bracket = df[mask].copy()
        
        if len(df_bracket) == 0:
            logger.warning(f"  No filers in ${bracket_min//1000}k-${bracket_max//1000}k bracket")
            return df
        
        # Calculate current rate
        current_rate = self.calculate_current_rate(df_bracket)
        
        if abs(current_rate - target_rate) < 0.1:
            logger.info(f"  ${bracket_min//1000}k-${bracket_max//1000}k: {current_rate:.1f}% ≈ {target_rate:.1f}% ✅")
            return df
        
        # Calculate shift needed
        shift_factor = self.calculate_target_income_shift(
            current_rate, target_rate, bracket_min, bracket_max
        )
        
        logger.info(f"  ${bracket_min//1000}k-${bracket_max//1000}k: {current_rate:.1f}% → {target_rate:.1f}% "
                   f"(shift: {shift_factor:.3f})")
        
        # Apply redistribution
        if method == 'lognormal':
            df_adjusted = self.apply_log_normal_redistribution(
                df_bracket, shift_factor, bracket_min, bracket_max
            )
        else:  # percentile
            df_adjusted = self.apply_percentile_shift(
                df_bracket, shift_factor, bracket_min, bracket_max
            )
        
        # Update the main dataframe
        result = df.copy()
        result.loc[mask, 'agi'] = df_adjusted['agi']
        
        return result
    
    def assign_agi_bracket_idx(self, agi: float) -> int:
        """Assign AGI to bracket index for AGI_BENCHMARKS."""
        for i, (min_agi, max_agi, _) in enumerate(AGI_BENCHMARKS):
            if max_agi == float('inf'):
                if agi >= min_agi:
                    return i
            else:
                if min_agi <= agi < max_agi:
                    return i
        return len(AGI_BENCHMARKS) - 1  # Default to highest bracket
    
    def calibrate_agi_targets(self, df: pd.DataFrame, weight_col: str = 'weight') -> pd.DataFrame:
        """
        Calibrate weights to match DOTAX Table 12A AGI bracket targets.
        
        This is the primary calibration step that ensures we match the official
        AGI distribution before any effective rate adjustments.
        """
        logger.info("\n" + "="*80)
        logger.info("PRIMARY AGI TARGET CALIBRATION (DOTAX Table 12A)")
        logger.info("="*80)
        logger.info("Calibrating weights to match official AGI bracket return counts...\n")
        
        result = df.copy()
        
        # Assign AGI brackets
        result['agi_bracket_idx'] = result['agi'].apply(self.assign_agi_bracket_idx)
        
        # Calculate current counts by bracket
        current_counts = result.groupby('agi_bracket_idx')[weight_col].sum()
        
        # Calculate calibration factors
        factors = {}
        for i, (min_agi, max_agi, target_count) in enumerate(AGI_BENCHMARKS):
            current = current_counts.get(i, 0)
            
            if current > 0 and target_count > 0:
                factor = target_count / current
                factors[i] = factor
            else:
                factors[i] = 1.0
                if current == 0 and target_count > 0:
                    logger.warning(f"  No filers in bracket ${min_agi//1000}k-${max_agi//1000 if max_agi != float('inf') else 'inf'}k but target is {target_count:,}")
        
        # Apply factors
        result['agi_calibration_factor'] = result['agi_bracket_idx'].map(factors)
        result[weight_col] = result[weight_col] * result['agi_calibration_factor']
        
        # Log results
        logger.info(f"{'Bracket':<20} {'Before':>12} {'After':>12} {'Target':>12} {'Factor':>8}")
        logger.info("-"*70)
        
        new_counts = result.groupby('agi_bracket_idx')[weight_col].sum()
        
        for i, (min_agi, max_agi, target_count) in enumerate(AGI_BENCHMARKS):
            before = current_counts.get(i, 0)
            after = new_counts.get(i, 0)
            factor = factors[i]
            
            if max_agi == float('inf'):
                label = f"${min_agi//1000}k+"
            else:
                label = f"${min_agi//1000}k-${max_agi//1000}k"
            
            logger.info(f"{label:<20} {before:>12,.0f} {after:>12,.0f} {target_count:>12,} {factor:>8.3f}")
        
        total_before = current_counts.sum()
        total_after = new_counts.sum()
        total_target = sum(target for _, _, target in AGI_BENCHMARKS)
        
        logger.info("-"*70)
        logger.info(f"{'TOTAL':<20} {total_before:>12,.0f} {total_after:>12,.0f} {total_target:>12,} {'':>8}")
        
        # Clean up temporary columns
        result.drop(columns=['agi_bracket_idx', 'agi_calibration_factor'], inplace=True)
        
        logger.info(f"\n✅ AGI target calibration complete. Total filers: {total_after:,.0f}")
        
        return result
    
    def calibrate(self, 
                 df: pd.DataFrame,
                 recalculate_tax: bool = True,
                 method: str = 'percentile',
                 apply_effective_rate_calibration: bool = True) -> pd.DataFrame:
        """
        Main calibration method - systematically adjust income distributions.
        
        Args:
            df: DataFrame with tax units
            recalculate_tax: Whether to recalculate taxes after adjustment
            method: 'percentile' or 'lognormal'
            
        Returns:
            DataFrame with calibrated income distributions
        """
        logger.info("=" * 80)
        logger.info("SYSTEMATIC INCOME DISTRIBUTION CALIBRATION")
        logger.info("=" * 80)
        logger.info(f"AGI Target Calibration: {'ENABLED' if self.calibrate_agi_first else 'DISABLED'}")
        logger.info(f"Effective Rate Calibration: {'ENABLED' if apply_effective_rate_calibration else 'DISABLED'}")
        logger.info(f"Method: {method}")
        logger.info(f"Threshold: AGI >= ${self.threshold:,.0f}")
        logger.info("")
        
        result = df.copy()
        
        # Step 1: Calibrate to AGI targets first (if enabled)
        if self.calibrate_agi_first:
            result = self.calibrate_agi_targets(result)
        
        # Step 2: Apply effective rate calibration (if enabled)
        if not apply_effective_rate_calibration:
            logger.info("\n⚠️  Skipping effective rate calibration as requested")
            return result
        
        logger.info("\n" + "="*80)
        logger.info("SECONDARY EFFECTIVE RATE CALIBRATION (High-Income Only)")
        logger.info("="*80)
        logger.info(f"Applying effective rate calibration to brackets >= ${self.threshold:,.0f}...\n")
        
        # Apply calibration to each high-income bracket
        for (bracket_min, bracket_max), targets in self.DOTAX_TARGETS.items():
            if bracket_min < self.threshold:
                continue  # Skip low/middle income brackets
            
            target_rate = targets['eff_rate']
            result = self.calibrate_bracket(
                result, bracket_min, bracket_max, target_rate, method
            )
        
        # Recalculate taxes with adjusted incomes
        if recalculate_tax:
            logger.info("")
            logger.info("Recalculating taxes with adjusted income distributions...")
            
            from src.tax.hawaii_calculator import HawaiiTaxCalculator
            calculator = HawaiiTaxCalculator()
            
            result = calculator.calculate_tax_units_batch(result)
        
        # Verify results
        logger.info("")
        logger.info("Calibration results:")
        
        for (bracket_min, bracket_max), targets in self.DOTAX_TARGETS.items():
            if bracket_min < self.threshold:
                continue
            
            mask = (result['agi'] >= bracket_min) & (result['agi'] < bracket_max)
            bracket_df = result[mask]
            
            if len(bracket_df) == 0:
                continue
            
            current_rate = self.calculate_current_rate(bracket_df)
            target_rate = targets['eff_rate']
            rate_gap = current_rate - target_rate
            
            label = f"${bracket_min//1000}k-${bracket_max//1000}k" if bracket_max != float('inf') else f"${bracket_min//1000}k+"
            status = "✅" if abs(rate_gap) < 0.5 else ("⚠️" if abs(rate_gap) < 1.0 else "❌")
            
            logger.info(f"  {label:<20} {current_rate:>6.2f}% vs {target_rate:>6.2f}% ({rate_gap:>+6.2f}pp) {status}")
        
        return result


def apply_income_distribution_calibration(df: pd.DataFrame,
                                          threshold: float = 100000,
                                          method: str = 'percentile',
                                          recalculate_tax: bool = True,
                                          calibrate_agi_first: bool = True,
                                          apply_effective_rate_calibration: bool = True) -> pd.DataFrame:
    """
    Convenience function to apply systematic income distribution calibration.
    
    Args:
        df: DataFrame with tax units
        threshold: AGI threshold above which to calibrate effective rates (default $100k)
        method: 'percentile' or 'lognormal'
        recalculate_tax: Whether to recalculate taxes after adjustment
        calibrate_agi_first: Whether to calibrate AGI targets first
        apply_effective_rate_calibration: Whether to apply effective rate calibration
        
    Returns:
        DataFrame with calibrated income distributions
    """
    calibrator = IncomeDistributionCalibrator(threshold=threshold, calibrate_agi_first=calibrate_agi_first)
    return calibrator.calibrate(df, recalculate_tax=recalculate_tax, method=method, 
                               apply_effective_rate_calibration=apply_effective_rate_calibration)
