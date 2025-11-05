"""
Comprehensive Weight Calibrator

Systematically adjusts weights across ALL brackets to match DOTax filer counts
while preserving income distributions and tax calculation accuracy.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class ComprehensiveWeightCalibrator:
    """
    Calibrate tax liabilities to match DOTAX Table A2-2 targets while preserving AGI counts.
    
    Table A2-2 provides tax liability (before credits) by filing status and AGI bracket.
    We apply tax multipliers (not weight changes) within each status/AGI cell, using
    sub-bracket percentile clustering with smoothing caps to preserve AGI distributions.
    """

    # Hybrid tax liability targets:
    # - Tables A9-2, A9-3, A9-4 for detailed brackets under $150k
    # - Table A2-2 for high-income brackets ≥$150k
    
    @classmethod
    def _load_hybrid_targets(cls):
        """Load hybrid targets from detailed benchmarks + A2-2 high-income."""
        try:
            from src.tax.validation.detailed_tax_calibration import load_detailed_benchmarks
            detailed_df = load_detailed_benchmarks()
        except (ImportError, FileNotFoundError) as e:
            logger.warning(f"Could not load detailed benchmarks: {e}")
            logger.warning("Falling back to Table A2-2 targets only")
            return cls._get_a22_fallback_targets()
        
        # Convert detailed benchmarks to target format
        targets = {}
        
        # Map DOTAX filing status to PUMS filing status
        status_map = {
            'joint': 'married_filing_jointly',
            'single': 'single', 
            'hoh': 'head_of_household'
        }
        
        for _, row in detailed_df.iterrows():
            dotax_status = row['filing_status']
            pums_status = status_map.get(dotax_status)
            
            if pums_status is None:
                continue
                
            if pums_status not in targets:
                targets[pums_status] = {}
            
            bracket = (float(row['agi_min']), float(row['agi_max']))
            # Use tax_after_credits for more realistic targets
            tax_millions = float(row['tax_after_credits_millions'])
            
            # Only include positive tax targets
            if tax_millions > 0:
                targets[pums_status][bracket] = tax_millions
        
        # Add high-income brackets from Table A2-2 (≥$150k)
        high_income_a22 = {
            'married_filing_jointly': {
                (150_000, 200_000): 230.0,
                (200_000, 300_000): 251.0,
                (300_000, 400_000): 118.0,
                (400_000, float('inf')): 596.0,
            },
            'single': {
                (150_000, 200_000): 52.0,
                (200_000, 300_000): 50.0,
                (300_000, 400_000): 29.0,
                (400_000, float('inf')): 375.0,
            },
            'head_of_household': {
                (150_000, 200_000): 12.0,
                (200_000, 300_000): 10.0,
                (300_000, 400_000): 5.0,
                (400_000, float('inf')): 26.0,
            },
        }
        
        # Merge high-income targets
        for status, brackets in high_income_a22.items():
            if status not in targets:
                targets[status] = {}
            targets[status].update(brackets)
        
        return targets
    
    @classmethod
    def _get_a22_fallback_targets(cls):
        """Fallback to Table A2-2 targets if detailed benchmarks unavailable."""
        return {
            'married_filing_jointly': {
                (0, 10_000): 0.01,
                (10_000, 20_000): 0.9,
                (20_000, 30_000): 4.0,
                (30_000, 40_000): 10.0,
                (40_000, 50_000): 17.0,
                (50_000, 75_000): 69.0,
                (75_000, 100_000): 109.0,
                (100_000, 150_000): 269.0,
                (150_000, 200_000): 230.0,
                (200_000, 300_000): 251.0,
                (300_000, 400_000): 118.0,
                (400_000, float('inf')): 596.0,
            },
            'single': {
                (0, 10_000): 3.0,
                (10_000, 20_000): 19.0,
                (20_000, 30_000): 41.0,
                (30_000, 40_000): 67.0,
                (40_000, 50_000): 80.0,
                (50_000, 75_000): 182.0,
                (75_000, 100_000): 122.0,
                (100_000, 150_000): 135.0,
                (150_000, 200_000): 52.0,
                (200_000, 300_000): 50.0,
                (300_000, 400_000): 29.0,
                (400_000, float('inf')): 375.0,
            },
            'head_of_household': {
                (0, 10_000): 0.1,
                (10_000, 20_000): 2.0,
                (20_000, 30_000): 6.0,
                (30_000, 40_000): 14.0,
                (40_000, 50_000): 19.0,
                (50_000, 75_000): 42.0,
                (75_000, 100_000): 31.0,
                (100_000, 150_000): 34.0,
                (150_000, 200_000): 12.0,
                (200_000, 300_000): 10.0,
                (300_000, 400_000): 5.0,
                (400_000, float('inf')): 26.0,
            },
        }

    STATUS_LABELS = {
        'married_filing_jointly': 'Joint',
        'single': 'Single',
        'head_of_household': 'Head of Household',
    }

    def __init__(self, apply_to_all: bool = True, n_percentiles: int = 5, 
                 multiplier_cap: float = 0.6, min_sample_size: int = 10,
                 max_iterations: int = 3, convergence_tol: float = 0.01):
        """
        Initialize calibrator.

        Args:
            apply_to_all: Backward compatibility flag (currently unused; retained for API).
            n_percentiles: Number of sub-clusters within each bracket (default 5 = quintiles)
            multiplier_cap: Maximum deviation from 1.0 for tax multipliers (default 0.6 = ±60%)
            min_sample_size: Minimum records needed for sub-clustering (default 10)
            max_iterations: Maximum passes over all brackets (default 3)
            convergence_tol: Convergence tolerance as a fraction (default 0.01 = 1%)
        """
        self.apply_to_all = apply_to_all
        self.n_percentiles = n_percentiles
        self.multiplier_cap = multiplier_cap
        self.min_sample_size = min_sample_size
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol

        # Load hybrid targets (A9 detailed + A2-2 high-income)
        self.tax_targets = self._load_hybrid_targets()

    def _max_gap_percentage(self, df: pd.DataFrame) -> float:
        """Compute maximum percentage gap between model and targets."""
        max_gap = 0.0
        for filing_status, targets in self.tax_targets.items():
            for bracket, target_tax_m in targets.items():
                bracket_min, bracket_max = bracket
                mask = (
                    (df['filing_status'] == filing_status)
                    & (df['agi'] >= bracket_min)
                    & (df['agi'] < bracket_max)
                )

                if mask.sum() == 0 or target_tax_m <= 0:
                    continue

                model_tax_m = (df.loc[mask, 'hi_state_tax'] * df.loc[mask, 'weight']).sum() / 1_000_000
                if target_tax_m > 0:
                    gap_pct = ((model_tax_m - target_tax_m) / target_tax_m) * 100
                    max_gap = max(max_gap, abs(gap_pct))

        return max_gap

    def _format_bracket_label(self, bracket_min: float, bracket_max: float) -> str:
        if bracket_max == float('inf'):
            return f"${bracket_min//1000}k+"
        return f"${bracket_min//1000}k-${bracket_max//1000}k"

    def _apply_percentile_clustering(self, bracket_df: pd.DataFrame) -> pd.DataFrame:
        """Split bracket into percentile clusters for fine-grained adjustments."""
        if len(bracket_df) < self.min_sample_size:
            # Too few records - use single cluster
            bracket_df = bracket_df.copy()
            bracket_df['_percentile_cluster'] = 0
            return bracket_df
        
        # Calculate percentile positions within bracket (similar to income calibrator)
        agi_values = bracket_df['agi'].values
        percentiles = np.linspace(0, 100, self.n_percentiles + 1)
        
        # Assign each unit to a percentile cluster
        percentile_thresholds = np.percentile(agi_values, percentiles)
        cluster_assignments = np.digitize(agi_values, percentile_thresholds) - 1
        
        # Ensure valid cluster indices
        cluster_assignments = np.clip(cluster_assignments, 0, self.n_percentiles - 1)
        
        bracket_df = bracket_df.copy()
        bracket_df['_percentile_cluster'] = cluster_assignments
        return bracket_df
    
    def _calculate_smoothed_multipliers(self, bracket_df: pd.DataFrame, 
                                       target_tax_m: float) -> Dict[int, float]:
        """Calculate tax multipliers for each percentile cluster with smoothing."""
        
        # Calculate overall bracket multiplier as baseline
        current_total = (bracket_df['hi_state_tax'] * bracket_df['weight']).sum() / 1_000_000
        if current_total <= 0:
            return {cluster: 1.0 for cluster in bracket_df['_percentile_cluster'].unique()}
        
        bracket_multiplier = target_tax_m / current_total
        
        # Calculate per-cluster multipliers
        cluster_multipliers = {}
        
        for cluster in bracket_df['_percentile_cluster'].unique():
            cluster_mask = bracket_df['_percentile_cluster'] == cluster
            cluster_df = bracket_df[cluster_mask]
            
            if len(cluster_df) == 0:
                cluster_multipliers[cluster] = bracket_multiplier
                continue
            
            # Current tax for this cluster
            cluster_current = (cluster_df['hi_state_tax'] * cluster_df['weight']).sum() / 1_000_000
            
            if cluster_current <= 0:
                cluster_multipliers[cluster] = bracket_multiplier
                continue
            
            # Target tax for this cluster (proportional to current share)
            cluster_share = cluster_current / current_total
            cluster_target = target_tax_m * cluster_share
            
            # Raw multiplier
            raw_multiplier = cluster_target / cluster_current if cluster_current > 0 else bracket_multiplier
            
            # Apply smoothing: blend with bracket multiplier and cap
            smoothing_weight = 0.7  # 70% cluster-specific, 30% bracket-wide
            blended_multiplier = (smoothing_weight * raw_multiplier + 
                                (1 - smoothing_weight) * bracket_multiplier)
            
            # Apply caps
            lower_bound = max(0.2, 1.0 - self.multiplier_cap)
            upper_bound = 1.0 + self.multiplier_cap
            capped_multiplier = np.clip(blended_multiplier, lower_bound, upper_bound)
            
            cluster_multipliers[cluster] = capped_multiplier
        
        return cluster_multipliers
    
    def _calibrate_group(
        self,
        df: pd.DataFrame,
        filing_status: str,
        bracket: Tuple[float, float],
        target_tax_m: float
    ) -> pd.DataFrame:
        """Calibrate tax liabilities for a specific filing-status/AGI bracket cell."""

        result = df.copy()

        bracket_min, bracket_max = bracket
        mask = (
            (result['filing_status'] == filing_status)
            & (result['agi'] >= bracket_min)
            & (result['agi'] < bracket_max)
        )

        if mask.sum() == 0:
            logger.debug(
                "  %s %s: no filers present; skipping",
                self.STATUS_LABELS.get(filing_status, filing_status),
                self._format_bracket_label(bracket_min, bracket_max),
            )
            return result

        bracket_df = result[mask].copy()
        current_tax = (bracket_df['hi_state_tax'] * bracket_df['weight']).sum() / 1_000_000

        if target_tax_m <= 0 or np.isclose(target_tax_m, 0.0):
            logger.debug(
                "  %s %s: target tax <= 0 (%.3f); skipping",
                self.STATUS_LABELS.get(filing_status, filing_status),
                self._format_bracket_label(bracket_min, bracket_max),
                target_tax_m,
            )
            return result

        if current_tax <= 0:
            logger.warning(
                "  %s %s: current tax is zero but target is %.3fM",
                self.STATUS_LABELS.get(filing_status, filing_status),
                self._format_bracket_label(bracket_min, bracket_max),
                target_tax_m,
            )
            return result

        if abs(current_tax - target_tax_m) / target_tax_m < 0.01:
            logger.info(
                "  %s %s: %.2fM ≈ %.2fM ",
                self.STATUS_LABELS.get(filing_status, filing_status),
                self._format_bracket_label(bracket_min, bracket_max),
                current_tax,
                target_tax_m,
            )
            return result

        # Apply percentile clustering
        bracket_df = self._apply_percentile_clustering(bracket_df)
        
        # Calculate smoothed multipliers
        cluster_multipliers = self._calculate_smoothed_multipliers(bracket_df, target_tax_m)
        
        # Apply tax multipliers (preserve weights, scale tax amounts)
        for cluster, multiplier in cluster_multipliers.items():
            cluster_mask_bracket = bracket_df['_percentile_cluster'] == cluster
            cluster_indices = bracket_df[cluster_mask_bracket].index
            
            # Scale tax liability columns
            tax_columns = ['hi_state_tax']
            
            for col in tax_columns:
                if col in result.columns:
                    result.loc[cluster_indices, col] = result.loc[cluster_indices, col] * multiplier
        
        # Verify final result
        final_tax = (result.loc[mask, 'hi_state_tax'] * result.loc[mask, 'weight']).sum() / 1_000_000
        
        logger.info(
            "  %s %s: %.2fM → %.2fM (target %.2fM, %d clusters)",
            self.STATUS_LABELS.get(filing_status, filing_status),
            self._format_bracket_label(bracket_min, bracket_max),
            current_tax,
            final_tax,
            target_tax_m,
            len(cluster_multipliers)
        )

        return result

    def calibrate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main calibration method - adjust weights to match Table A2-2 tax liability targets.

        Args:
            df: DataFrame with tax units

        Returns:
            DataFrame with calibrated weights
        """
        logger.info("=" * 80)
        logger.info("HYBRID TAX CALIBRATION (A9 DETAILED + A2-2 HIGH-INCOME - PRESERVES AGI COUNTS)")
        logger.info("=" * 80)
        logger.info(f"Sub-bracket clustering: {self.n_percentiles} percentiles")
        logger.info(f"Multiplier caps: ±{self.multiplier_cap*100:.0f}%")
        logger.info(f"Minimum sample size: {self.min_sample_size}")
        logger.info(f"Max iterations: {self.max_iterations} (tol {self.convergence_tol*100:.1f}% max gap)")
        logger.info("")

        result = df.copy()

        original_weight = result['weight'].sum()
        logger.info(f"Original total weighted filers: {original_weight:,.0f}")
        logger.info("")

        converged = False
        for iteration in range(1, self.max_iterations + 1):
            logger.info("--- Iteration %d/%d ---", iteration, self.max_iterations)

            for filing_status, targets in self.tax_targets.items():
                logger.info(
                    "Filing status: %s",
                    self.STATUS_LABELS.get(filing_status, filing_status)
                )
                for bracket, target_tax_m in targets.items():
                    # Optionally skip high-income brackets if apply_to_all is False
                    if not self.apply_to_all and bracket[0] >= 200_000:
                        continue

                    result = self._calibrate_group(result, filing_status, bracket, target_tax_m)

                logger.info("")

            max_gap_pct = self._max_gap_percentage(result)
            logger.info("Max gap after iteration %d: %.2f%%", iteration, max_gap_pct)

            if max_gap_pct <= self.convergence_tol * 100:
                logger.info("Converged within tolerance; stopping early")
                converged = True
                break

        if not converged and self.max_iterations > 1:
            logger.info("Reached max iterations without meeting tolerance")

        # Summary diagnostics
        final_weight = result['weight'].sum()
        logger.info("Total weighted filers after calibration: %,.0f (Δ %+,.0f, %+0.2f%%)",
                    final_weight,
                    final_weight - original_weight,
                    (final_weight / original_weight - 1) * 100)

        logger.info("")
        logger.info("Verification against Table A2-2 targets:")

        for filing_status, targets in self.tax_targets.items():
            for bracket, target_tax_m in targets.items():
                bracket_min, bracket_max = bracket
                mask = (
                    (result['filing_status'] == filing_status)
                    & (result['agi'] >= bracket_min)
                    & (result['agi'] < bracket_max)
                )

                model_tax_m = (result.loc[mask, 'hi_state_tax'] * result.loc[mask, 'weight']).sum() / 1_000_000
                gap_pct = ((model_tax_m - target_tax_m) / target_tax_m * 100) if target_tax_m > 0 else 0

                status = "✅" if abs(gap_pct) < 5 else ("⚠️" if abs(gap_pct) < 15 else "❌")

                logger.info(
                    "  %-20s %-12s Model: %7.2fM  Target: %7.2fM  Gap: %+6.2f%% %s",
                    self.STATUS_LABELS.get(filing_status, filing_status),
                    self._format_bracket_label(bracket_min, bracket_max),
                    model_tax_m,
                    target_tax_m,
                    gap_pct,
                    status,
                )

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
