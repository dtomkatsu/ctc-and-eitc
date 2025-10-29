"""
Multi-Stage IPF (Iterative Proportional Fitting) Calibration Orchestrator

Implements simultaneous calibration to multiple targets using IPF/Raking algorithm.
This approach handles overlapping targets better than pure sequential calibration.

IPF Algorithm:
1. Define multiple target distributions (margins)
2. Iteratively adjust to each margin while holding others constant
3. Repeat until convergence (all margins within tolerance)

References:
- Deming & Stephan (1940) - Original IPF algorithm
- TPC/CBO microsimulation models - Multi-dimensional calibration
- Creedy (2003) - Survey reweighting with IPF
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CalibrationTarget:
    """Represents a single calibration target (margin)."""
    name: str
    dimension: str  # 'weight', 'agi', 'filing_status', 'tax'
    targets: Dict[Tuple, float]  # Mapping of bracket/category to target value
    weight: float = 1.0  # Relative importance (for prioritization)


class IPFCalibrationOrchestrator:
    """
    Multi-stage IPF calibrator for tax microsimulation.
    
    Handles multiple overlapping calibration targets:
    1. Filer counts by AGI bracket
    2. Tax totals by AGI bracket
    3. Filing status distribution
    4. Income distribution within brackets
    
    Uses IPF/raking to iteratively adjust weights and other dimensions
    until all targets converge simultaneously.
    """
    
    # DOTax Table A8 targets (in millions $)
    DOTAX_TAX_TARGETS = {
        (0, 10000): 3.0,
        (10000, 20000): 21.0,
        (20000, 30000): 51.0,
        (30000, 40000): 92.0,
        (40000, 50000): 116.0,
        (50000, 75000): 293.0,
        (75000, 100000): 261.0,
        (100000, 150000): 438.0,
        (150000, 200000): 294.0,
        (200000, 300000): 310.0,
        (300000, 400000): 153.0,
        (400000, 500000): 101.0,
        (500000, 750000): 149.0,
        (750000, 1000000): 85.0,
        (1000000, float('inf')): 663.0,
    }
    
    # DOTax filer count targets (CANONICAL TOTAL: 618,423)
    DOTAX_FILER_TARGETS = {
        (0, 10000): 115285,
        (10000, 20000): 64160,
        (20000, 30000): 57835,
        (30000, 40000): 58135,
        (40000, 50000): 53555,
        (50000, 75000): 91459,
        (75000, 100000): 54976,
        (100000, 150000): 62065,
        (150000, 200000): 27976,
        (200000, 300000): 19015,
        (300000, 400000): 5729,
        (400000, 500000): 2856,
        (500000, 750000): 2549,
        (750000, 1000000): 1004,
        (1000000, float('inf')): 1824,
    }
    
    # DOTax filing status targets (adjusted to match filer count total of 618,423)
    DOTAX_FILING_STATUS_TARGETS = {
        'single': 326470,
        'married_filing_jointly': 210724,
        'head_of_household': 65638,
        'married_filing_separately': 15591,
    }
    
    def __init__(self, 
                 max_iterations: int = 20,
                 tolerance: float = 0.01,
                 convergence_criterion: str = 'chi_squared'):
        """
        Initialize IPF calibration orchestrator.
        
        Args:
            max_iterations: Maximum IPF iterations
            tolerance: Convergence tolerance (1% default for IPF)
            convergence_criterion: 'chi_squared' or 'max_deviation'
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.convergence_criterion = convergence_criterion
        self.calibration_history = []
        
    def calculate_chi_squared(self, 
                              current: Dict[Tuple, float],
                              target: Dict[Tuple, float]) -> float:
        """
        Calculate chi-squared statistic for convergence testing.
        
        χ² = Σ ((current - target)² / target)
        """
        chi_sq = 0
        for key in target:
            if key in current and target[key] > 0:
                chi_sq += (current[key] - target[key])**2 / target[key]
        return chi_sq
    
    def calculate_max_deviation(self,
                                 current: Dict[Tuple, float],
                                 target: Dict[Tuple, float]) -> float:
        """
        Calculate maximum relative deviation for convergence testing.
        
        max_dev = max(|current - target| / target)
        """
        max_dev = 0
        for key in target:
            if key in current and target[key] > 0:
                dev = abs(current[key] - target[key]) / target[key]
                max_dev = max(max_dev, dev)
        return max_dev
    
    def ipf_adjust_weights_to_filer_counts(self,
                                           df: pd.DataFrame,
                                           targets: Dict[Tuple, float],
                                           agi_col: str = 'agi',
                                           weight_col: str = 'weight') -> pd.DataFrame:
        """
        IPF step: Adjust weights to match filer count targets by AGI bracket.
        
        This is one "margin" in the IPF algorithm.
        """
        result = df.copy()
        
        for (min_agi, max_agi), target_count in targets.items():
            mask = (result[agi_col] >= min_agi) & (result[agi_col] < max_agi)
            current_count = result.loc[mask, weight_col].sum()
            
            if current_count > 0 and abs(current_count - target_count) > 1:
                adjustment_factor = target_count / current_count
                result.loc[mask, weight_col] *= adjustment_factor
                logger.debug(f"  Adjusted ${min_agi//1000}k-${max_agi//1000}k: "
                           f"{current_count:>8,.0f} → {target_count:>8,.0f} (×{adjustment_factor:.4f})")
        
        return result
    
    def ipf_adjust_weights_to_tax_totals(self,
                                         df: pd.DataFrame,
                                         targets: Dict[Tuple, float],
                                         agi_col: str = 'agi',
                                         tax_col: str = 'hi_state_tax',
                                         weight_col: str = 'weight',
                                         max_adjustment: float = 1.5) -> pd.DataFrame:
        """
        IPF step: Adjust weights to match tax total targets by AGI bracket.
        
        This is another "margin" in the IPF algorithm.
        Uses constrained adjustment to avoid extreme weight changes.
        """
        result = df.copy()
        
        for (min_agi, max_agi), target_tax_m in targets.items():
            mask = (result[agi_col] >= min_agi) & (result[agi_col] < max_agi)
            current_tax_m = (result.loc[mask, tax_col] * result.loc[mask, weight_col]).sum() / 1_000_000
            
            if current_tax_m > 0 and abs(current_tax_m - target_tax_m) > 0.1:
                adjustment_factor = target_tax_m / current_tax_m
                # Constrain adjustment to avoid extreme changes
                adjustment_factor = np.clip(adjustment_factor, 1/max_adjustment, max_adjustment)
                result.loc[mask, weight_col] *= adjustment_factor
                new_tax_m = (result.loc[mask, tax_col] * result.loc[mask, weight_col]).sum() / 1_000_000
                logger.debug(f"  Adjusted ${min_agi//1000}k-${max_agi//1000}k tax: "
                           f"${current_tax_m:>7.1f}M → ${new_tax_m:>7.1f}M (target: ${target_tax_m:>7.1f}M)")
        
        return result
    
    def ipf_adjust_weights_to_filing_status(self,
                                            df: pd.DataFrame,
                                            targets: Dict[str, float],
                                            filing_status_col: str = 'filing_status',
                                            weight_col: str = 'weight') -> pd.DataFrame:
        """
        IPF step: Adjust weights to match filing status distribution.
        
        This is the third "margin" in the IPF algorithm.
        """
        result = df.copy()
        
        for status, target_count in targets.items():
            mask = result[filing_status_col] == status
            current_count = result.loc[mask, weight_col].sum()
            
            if current_count > 0 and abs(current_count - target_count) > 1:
                adjustment_factor = target_count / current_count
                result.loc[mask, weight_col] *= adjustment_factor
                logger.debug(f"  Adjusted {status}: "
                           f"{current_count:>8,.0f} → {target_count:>8,.0f} (×{adjustment_factor:.4f})")
        
        return result
    
    def calibrate_ipf(self,
                      df: pd.DataFrame,
                      calibrate_filer_counts: bool = True,
                      calibrate_tax_totals: bool = True,
                      calibrate_filing_status: bool = True) -> pd.DataFrame:
        """
        Main IPF calibration loop.
        
        Iteratively adjusts to each margin until convergence:
        1. Filer counts by AGI bracket
        2. Tax totals by AGI bracket
        3. Filing status distribution
        
        Args:
            df: Input DataFrame with tax units
            calibrate_filer_counts: Whether to calibrate to filer count targets
            calibrate_tax_totals: Whether to calibrate to tax total targets
            calibrate_filing_status: Whether to calibrate to filing status targets
            
        Returns:
            Calibrated DataFrame
        """
        logger.info("=" * 80)
        logger.info("MULTI-STAGE IPF CALIBRATION")
        logger.info("=" * 80)
        logger.info(f"Max iterations: {self.max_iterations}")
        logger.info(f"Tolerance: {self.tolerance * 100}%")
        logger.info(f"Convergence criterion: {self.convergence_criterion}")
        
        result = df.copy()
        
        # Initial validation
        logger.info("\nInitial state:")
        filer_dev = self._validate_filer_counts(result)
        tax_dev = self._validate_tax_totals(result)
        status_dev = self._validate_filing_status(result)
        
        # IPF iteration loop
        for iteration in range(self.max_iterations):
            logger.info(f"\n{'='*80}")
            logger.info(f"IPF ITERATION {iteration + 1}/{self.max_iterations}")
            logger.info(f"{'='*80}")
            
            # Store previous weights for convergence check
            prev_weights = result['weight'].copy()
            
            # Margin 1: Filer counts by AGI bracket
            if calibrate_filer_counts:
                logger.info("\n1. Adjusting to filer count targets...")
                result = self.ipf_adjust_weights_to_filer_counts(
                    result, 
                    self.DOTAX_FILER_TARGETS
                )
            
            # Margin 2: Tax totals by AGI bracket  
            if calibrate_tax_totals:
                logger.info("\n2. Adjusting to tax total targets...")
                result = self.ipf_adjust_weights_to_tax_totals(
                    result,
                    self.DOTAX_TAX_TARGETS
                )
            
            # Margin 3: Filing status distribution
            if calibrate_filing_status:
                logger.info("\n3. Adjusting to filing status targets...")
                result = self.ipf_adjust_weights_to_filing_status(
                    result,
                    self.DOTAX_FILING_STATUS_TARGETS
                )
            
            # Check convergence
            weight_change = abs(result['weight'] - prev_weights).max() / prev_weights.mean()
            
            logger.info(f"\nMax weight change: {weight_change:.4f}")
            
            # Validate current state
            filer_dev = self._validate_filer_counts(result)
            tax_dev = self._validate_tax_totals(result)
            status_dev = self._validate_filing_status(result)
            
            # Calculate convergence metrics
            if self.convergence_criterion == 'chi_squared':
                conv_metric = filer_dev['chi_squared']
                logger.info(f"Chi-squared: {conv_metric:.4f}")
            else:
                conv_metric = filer_dev['max_deviation']
                logger.info(f"Max deviation: {conv_metric:.4f}")
            
            # Check for convergence
            if weight_change < self.tolerance:
                logger.info(f"\n✅ IPF CONVERGED in {iteration + 1} iterations!")
                logger.info(f"   Weight change: {weight_change:.4f} < {self.tolerance}")
                break
        else:
            logger.warning(f"\n⚠️  IPF reached max iterations ({self.max_iterations})")
            logger.warning(f"   Final weight change: {weight_change:.4f}")
        
        # Final validation report
        logger.info("\n" + "="*80)
        logger.info("FINAL IPF CALIBRATION RESULTS")
        logger.info("="*80)
        
        self._print_validation_summary(result)
        
        return result
    
    def _validate_filer_counts(self, df: pd.DataFrame) -> Dict[str, float]:
        """Validate filer counts and return deviation metrics."""
        current_counts = {}
        total_sq_error = 0
        max_dev = 0
        
        for (min_agi, max_agi), target in self.DOTAX_FILER_TARGETS.items():
            mask = (df['agi'] >= min_agi) & (df['agi'] < max_agi)
            current = df.loc[mask, 'weight'].sum()
            current_counts[(min_agi, max_agi)] = current
            
            if target > 0:
                sq_error = (current - target)**2 / target
                total_sq_error += sq_error
                dev = abs(current - target) / target
                max_dev = max(max_dev, dev)
        
        return {
            'current': current_counts,
            'chi_squared': total_sq_error,
            'max_deviation': max_dev
        }
    
    def _validate_tax_totals(self, df: pd.DataFrame) -> Dict[str, float]:
        """Validate tax totals and return deviation metrics."""
        current_totals = {}
        total_sq_error = 0
        max_dev = 0
        
        for (min_agi, max_agi), target_m in self.DOTAX_TAX_TARGETS.items():
            mask = (df['agi'] >= min_agi) & (df['agi'] < max_agi)
            current_m = (df.loc[mask, 'hi_state_tax'] * df.loc[mask, 'weight']).sum() / 1_000_000
            current_totals[(min_agi, max_agi)] = current_m
            
            if target_m > 0:
                sq_error = (current_m - target_m)**2 / target_m
                total_sq_error += sq_error
                dev = abs(current_m - target_m) / target_m
                max_dev = max(max_dev, dev)
        
        return {
            'current': current_totals,
            'chi_squared': total_sq_error,
            'max_deviation': max_dev
        }
    
    def _validate_filing_status(self, df: pd.DataFrame) -> Dict[str, float]:
        """Validate filing status distribution and return deviation metrics."""
        current_counts = {}
        total_sq_error = 0
        max_dev = 0
        
        for status, target in self.DOTAX_FILING_STATUS_TARGETS.items():
            mask = df['filing_status'] == status
            current = df.loc[mask, 'weight'].sum()
            current_counts[status] = current
            
            if target > 0:
                sq_error = (current - target)**2 / target
                total_sq_error += sq_error
                dev = abs(current - target) / target
                max_dev = max(max_dev, dev)
        
        return {
            'current': current_counts,
            'chi_squared': total_sq_error,
            'max_deviation': max_dev
        }
    
    def _print_validation_summary(self, df: pd.DataFrame):
        """Print comprehensive validation summary."""
        
        # Filer counts
        logger.info("\n1. FILER COUNTS BY AGI BRACKET:")
        filer_results = self._validate_filer_counts(df)
        within_tolerance = 0
        total_brackets = len(self.DOTAX_FILER_TARGETS)
        
        for (min_agi, max_agi), target in self.DOTAX_FILER_TARGETS.items():
            current = filer_results['current'][(min_agi, max_agi)]
            deviation = abs(current - target) / target if target > 0 else 0
            
            if deviation <= self.tolerance:
                within_tolerance += 1
                status = "✅"
            else:
                status = "❌" if deviation > 0.10 else "⚠️"
            
            logger.info(f"   ${min_agi//1000:>4}k-${max_agi//1000:>4}k: "
                       f"{current:>8,.0f} vs {target:>8,.0f} "
                       f"({deviation:>+6.1%}) {status}")
        
        logger.info(f"\n   Within tolerance: {within_tolerance}/{total_brackets} brackets")
        logger.info(f"   Chi-squared: {filer_results['chi_squared']:.2f}")
        logger.info(f"   Max deviation: {filer_results['max_deviation']:.1%}")
        
        # Tax totals
        logger.info("\n2. TAX TOTALS BY AGI BRACKET:")
        tax_results = self._validate_tax_totals(df)
        within_tolerance = 0
        
        for (min_agi, max_agi), target_m in self.DOTAX_TAX_TARGETS.items():
            current_m = tax_results['current'][(min_agi, max_agi)]
            deviation = abs(current_m - target_m) / target_m if target_m > 0 else 0
            
            if deviation <= self.tolerance:
                within_tolerance += 1
                status = "✅"
            else:
                status = "❌" if deviation > 0.10 else "⚠️"
            
            logger.info(f"   ${min_agi//1000:>4}k-${max_agi//1000}k: "
                       f"${current_m:>7.1f}M vs ${target_m:>7.1f}M "
                       f"({deviation:>+6.1%}) {status}")
        
        logger.info(f"\n   Within tolerance: {within_tolerance}/{total_brackets} brackets")
        logger.info(f"   Chi-squared: {tax_results['chi_squared']:.2f}")
        logger.info(f"   Max deviation: {tax_results['max_deviation']:.1%}")
        
        # Filing status
        logger.info("\n3. FILING STATUS DISTRIBUTION:")
        status_results = self._validate_filing_status(df)
        
        for status, target in self.DOTAX_FILING_STATUS_TARGETS.items():
            current = status_results['current'][status]
            deviation = abs(current - target) / target if target > 0 else 0
            status_icon = "✅" if deviation <= self.tolerance else "⚠️"
            
            logger.info(f"   {status:<30}: "
                       f"{current:>8,.0f} vs {target:>8,.0f} "
                       f"({deviation:>+6.1%}) {status_icon}")


def apply_ipf_calibration(df: pd.DataFrame,
                          max_iterations: int = 20,
                          tolerance: float = 0.01,
                          calibrate_filer_counts: bool = True,
                          calibrate_tax_totals: bool = True,
                          calibrate_filing_status: bool = True) -> pd.DataFrame:
    """
    Convenience function to apply IPF calibration.
    
    Args:
        df: Input DataFrame with tax units
        max_iterations: Maximum IPF iterations (default: 20)
        tolerance: Convergence tolerance (default: 1%)
        calibrate_filer_counts: Whether to calibrate to filer counts
        calibrate_tax_totals: Whether to calibrate to tax totals
        calibrate_filing_status: Whether to calibrate to filing status
        
    Returns:
        Calibrated DataFrame
    """
    orchestrator = IPFCalibrationOrchestrator(
        max_iterations=max_iterations,
        tolerance=tolerance
    )
    
    return orchestrator.calibrate_ipf(
        df,
        calibrate_filer_counts=calibrate_filer_counts,
        calibrate_tax_totals=calibrate_tax_totals,
        calibrate_filing_status=calibrate_filing_status
    )
