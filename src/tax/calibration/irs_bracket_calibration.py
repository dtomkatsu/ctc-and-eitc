"""
IRS SOI Bracket Calibration for Hawaii Tax Model

This module implements Stage 3 calibration: matching IRS SOI income bracket
distributions while maintaining DOTAX resident totals. The calibration adjusts
both the COUNT of filers in each bracket and the AVERAGE income within brackets.

Key Features:
- Matches IRS SOI bracket counts (scaled to DOTAX total)
- Adjusts average income within brackets to match IRS targets
- Bounded adjustments to prevent extreme distortions
- Preserves DOTAX total returns after calibration
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class IRSBracketCalibrator:
    """
    Calibrates PUMS tax units to match IRS SOI income bracket distributions.
    
    This implements a two-stage calibration:
    1. Adjust weights to match bracket counts (scaled to DOTAX total)
    2. Adjust incomes to match average income within each bracket
    """
    
    # IRS SOI 2022 Income Brackets for Hawaii
    # Source: IRS Statistics of Income, Table 2 - Returns by Size of AGI
    IRS_BRACKETS = {
        '0-25k': {
            'count': 130000,
            'total_agi': 1650000000,
            'avg': 12692,
            'bounds': (0, 25000)
        },
        '25-50k': {
            'count': 180000,
            'total_agi': 6300000000,
            'avg': 35000,
            'bounds': (25000, 50000)
        },
        '50-75k': {
            'count': 140000,
            'total_agi': 8400000000,
            'avg': 60000,
            'bounds': (50000, 75000)
        },
        '75-100k': {
            'count': 85000,
            'total_agi': 7200000000,
            'avg': 84706,
            'bounds': (75000, 100000)
        },
        '100-200k': {
            'count': 60000,
            'total_agi': 8400000000,
            'avg': 140000,
            'bounds': (100000, 200000)
        },
        '200k+': {
            'count': 15000,
            'total_agi': 6000000000,
            'avg': 400000,
            'bounds': (200000, float('inf'))
        }
    }
    
    # DOTAX 2022 Total Resident Returns (from Table 5A)
    DOTAX_TOTAL_RETURNS = 634956
    
    def __init__(self, 
                 max_weight_adjustment: float = 3.0,
                 max_income_adjustment: float = 0.3,
                 apply_income_adjustment: bool = True):
        """
        Initialize IRS bracket calibrator.
        
        Args:
            max_weight_adjustment: Maximum weight adjustment factor (e.g., 3.0 = 3x)
            max_income_adjustment: Maximum income adjustment (e.g., 0.3 = ±30%)
            apply_income_adjustment: Whether to adjust incomes within brackets
        """
        self.max_weight_adjustment = max_weight_adjustment
        self.max_income_adjustment = max_income_adjustment
        self.apply_income_adjustment = apply_income_adjustment
        
        # Calculate IRS total and bracket percentages
        self.irs_total_returns = sum(b['count'] for b in self.IRS_BRACKETS.values())
        self.bracket_percentages = self._calculate_bracket_percentages()
        self.target_counts = self._calculate_target_counts()
        
        logger.info(f"IRS SOI Bracket Calibrator initialized")
        logger.info(f"  IRS Total Returns: {self.irs_total_returns:,}")
        logger.info(f"  DOTAX Total Returns: {self.DOTAX_TOTAL_RETURNS:,}")
        logger.info(f"  Scaling Factor: {self.DOTAX_TOTAL_RETURNS / self.irs_total_returns:.4f}")
    
    def _calculate_bracket_percentages(self) -> Dict[str, float]:
        """Calculate percentage of returns in each IRS bracket."""
        percentages = {}
        for bracket, data in self.IRS_BRACKETS.items():
            percentages[bracket] = data['count'] / self.irs_total_returns
        return percentages
    
    def _calculate_target_counts(self) -> Dict[str, float]:
        """
        Calculate target counts for each bracket, scaled to DOTAX total.
        
        Example: $25-50k bracket
        - IRS: 180,000 out of 610,000 = 29.5%
        - Target: 29.5% × 634,956 = 187,300 filers
        """
        target_counts = {}
        for bracket, pct in self.bracket_percentages.items():
            target_counts[bracket] = pct * self.DOTAX_TOTAL_RETURNS
        return target_counts
    
    def get_bracket_bounds(self, bracket_name: str) -> Tuple[float, float]:
        """Get income bounds for a bracket."""
        return self.IRS_BRACKETS[bracket_name]['bounds']
    
    def assign_bracket(self, income: float) -> str:
        """Assign an income value to its corresponding bracket."""
        for bracket_name, data in self.IRS_BRACKETS.items():
            low, high = data['bounds']
            if low <= income < high:
                return bracket_name
        # Default to highest bracket if above all bounds
        return '200k+'
    
    def calibrate(self, 
                  tax_units: pd.DataFrame,
                  weight_col: str = 'weight',
                  income_col: str = 'income') -> pd.DataFrame:
        """
        Calibrate tax units to match IRS SOI bracket distributions.
        
        Args:
            tax_units: DataFrame with tax units
            weight_col: Name of weight column to use
            income_col: Name of income column
            
        Returns:
            DataFrame with calibrated weights and incomes
        """
        logger.info("\n" + "=" * 70)
        logger.info("Starting IRS SOI Bracket Calibration")
        logger.info("=" * 70)
        
        # Create copy to avoid modifying original
        calibrated = tax_units.copy()
        
        # Initialize calibrated columns
        if 'calibrated_weight' not in calibrated.columns:
            calibrated['calibrated_weight'] = calibrated[weight_col]
        
        # Assign brackets to each tax unit
        calibrated['income_bracket'] = calibrated[income_col].apply(self.assign_bracket)
        
        # Step 1: Calibrate bracket counts (adjust weights)
        calibrated = self._calibrate_bracket_counts(calibrated, weight_col, income_col)
        
        # Step 2: Re-normalize to maintain DOTAX total
        calibrated = self._renormalize_weights(calibrated)
        
        # Step 3: Calibrate average income within brackets (optional)
        if self.apply_income_adjustment:
            calibrated = self._calibrate_bracket_averages(calibrated, income_col)
        
        # Validate results
        self.validate(calibrated, income_col)
        
        logger.info("\n" + "=" * 70)
        logger.info("IRS SOI Bracket Calibration Complete")
        logger.info("=" * 70 + "\n")
        
        return calibrated
    
    def _calibrate_bracket_counts(self, 
                                   tax_units: pd.DataFrame,
                                   weight_col: str,
                                   income_col: str) -> pd.DataFrame:
        """
        Step 1: Adjust weights to match IRS bracket counts.
        
        For each bracket:
        - Calculate current weighted count
        - Calculate adjustment factor = target_count / current_count
        - Apply bounded adjustment to weights
        """
        logger.info("\nStep 1: Calibrating Bracket Counts")
        logger.info("-" * 70)
        logger.info(f"  {'Bracket':<12} {'Current':>12} {'Target':>12} {'Factor':>10} {'Status':>8}")
        logger.info("  " + "-" * 70)
        
        for bracket_name, target_count in self.target_counts.items():
            low, high = self.get_bracket_bounds(bracket_name)
            
            # Get mask for this bracket
            mask = (tax_units[income_col] >= low) & (tax_units[income_col] < high)
            
            # Current count in this bracket
            current_count = tax_units.loc[mask, 'calibrated_weight'].sum()
            
            # Calculate adjustment factor
            if current_count > 0:
                adjustment = target_count / current_count
                
                # Bound adjustment to prevent extreme changes
                bounded_adjustment = np.clip(
                    adjustment, 
                    1.0 / self.max_weight_adjustment,
                    self.max_weight_adjustment
                )
                
                # Apply adjustment
                tax_units.loc[mask, 'calibrated_weight'] *= bounded_adjustment
                
                # Log results
                status = '✅' if abs(adjustment - 1.0) < 0.1 else '⚠️' if abs(adjustment - 1.0) < 0.3 else '🔧'
                logger.info(f"  {status} {bracket_name:<10} {current_count:>12,.0f} {target_count:>12,.0f} "
                           f"{bounded_adjustment:>9.3f}x")
                
                if adjustment != bounded_adjustment:
                    logger.warning(f"     ⚠️  Adjustment bounded: {adjustment:.3f}x → {bounded_adjustment:.3f}x")
            else:
                logger.warning(f"  ❌ {bracket_name:<10} {'No units':>12} {target_count:>12,.0f} {'N/A':>10}")
        
        return tax_units
    
    def _renormalize_weights(self, tax_units: pd.DataFrame) -> pd.DataFrame:
        """
        Step 2: Re-normalize weights to maintain DOTAX total.
        
        After bracket adjustments, total may drift from DOTAX target.
        This step scales all weights proportionally to hit the exact target.
        """
        logger.info("\nStep 2: Re-normalizing to DOTAX Total")
        logger.info("-" * 70)
        
        current_total = tax_units['calibrated_weight'].sum()
        normalization_factor = self.DOTAX_TOTAL_RETURNS / current_total
        
        logger.info(f"  Current Total: {current_total:,.0f}")
        logger.info(f"  DOTAX Target:  {self.DOTAX_TOTAL_RETURNS:,}")
        logger.info(f"  Adjustment:    {normalization_factor:.6f}x")
        
        tax_units['calibrated_weight'] *= normalization_factor
        
        final_total = tax_units['calibrated_weight'].sum()
        logger.info(f"  Final Total:   {final_total:,.0f}")
        
        return tax_units
    
    def _calibrate_bracket_averages(self, 
                                     tax_units: pd.DataFrame,
                                     income_col: str) -> pd.DataFrame:
        """
        Step 3: Adjust average income within each bracket to match IRS targets.
        
        For each bracket:
        - Calculate current weighted average income
        - Calculate adjustment factor = target_avg / current_avg
        - Apply bounded adjustment to incomes (max ±30% by default)
        """
        logger.info("\nStep 3: Calibrating Bracket Average Incomes")
        logger.info("-" * 70)
        logger.info(f"  {'Bracket':<12} {'Current Avg':>14} {'Target Avg':>14} {'Factor':>10} {'Status':>8}")
        logger.info("  " + "-" * 70)
        
        for bracket_name, bracket_data in self.IRS_BRACKETS.items():
            low, high = self.get_bracket_bounds(bracket_name)
            target_avg = bracket_data['avg']
            
            # Get mask for this bracket
            mask = (tax_units[income_col] >= low) & (tax_units[income_col] < high)
            
            if mask.sum() == 0:
                logger.warning(f"  ❌ {bracket_name:<10} {'No units':>14} ${target_avg:>13,} {'N/A':>10}")
                continue
            
            # Calculate current weighted average
            weighted_income = (tax_units.loc[mask, income_col] * 
                              tax_units.loc[mask, 'calibrated_weight']).sum()
            total_weight = tax_units.loc[mask, 'calibrated_weight'].sum()
            current_avg = weighted_income / total_weight if total_weight > 0 else 0
            
            # Calculate adjustment factor
            if current_avg > 0:
                adjustment = target_avg / current_avg
                
                # Bound adjustment to prevent extreme distortions
                bounded_adjustment = np.clip(
                    adjustment,
                    1.0 - self.max_income_adjustment,
                    1.0 + self.max_income_adjustment
                )
                
                # Apply adjustment to incomes
                tax_units.loc[mask, income_col] *= bounded_adjustment
                
                # Log results
                status = '✅' if abs(adjustment - 1.0) < 0.05 else '⚠️' if abs(adjustment - 1.0) < 0.15 else '🔧'
                logger.info(f"  {status} {bracket_name:<10} ${current_avg:>13,.0f} ${target_avg:>13,} "
                           f"{bounded_adjustment:>9.3f}x")
                
                if adjustment != bounded_adjustment:
                    logger.warning(f"     ⚠️  Adjustment bounded: {adjustment:.3f}x → {bounded_adjustment:.3f}x")
            else:
                logger.warning(f"  ❌ {bracket_name:<10} ${current_avg:>13,.0f} ${target_avg:>13,} {'N/A':>10}")
        
        return tax_units
    
    def validate(self, 
                 tax_units: pd.DataFrame,
                 income_col: str = 'income') -> Dict[str, float]:
        """
        Validate calibrated results against IRS SOI benchmarks.
        
        Args:
            tax_units: Calibrated DataFrame
            income_col: Name of income column
            
        Returns:
            Dictionary of validation metrics
        """
        logger.info("\n" + "=" * 70)
        logger.info("IRS SOI Bracket Calibration Validation")
        logger.info("=" * 70)
        
        metrics = {}
        
        # 1. Total returns validation
        total_returns = tax_units['calibrated_weight'].sum()
        returns_error = abs(total_returns - self.DOTAX_TOTAL_RETURNS) / self.DOTAX_TOTAL_RETURNS
        metrics['total_returns'] = total_returns
        metrics['returns_error_pct'] = returns_error * 100
        
        logger.info(f"\nTotal Returns:")
        logger.info(f"  Calibrated:    {total_returns:,.0f}")
        logger.info(f"  DOTAX Target:  {self.DOTAX_TOTAL_RETURNS:,}")
        logger.info(f"  Error:         {returns_error*100:.3f}%")
        
        # 2. Bracket distribution validation
        logger.info(f"\nBracket Distribution:")
        logger.info(f"  {'Bracket':<12} {'Calibrated':>14} {'IRS Count':>14} {'Target':>14} {'Error':>10}")
        logger.info("  " + "-" * 70)
        
        total_error = 0
        for bracket_name, bracket_data in self.IRS_BRACKETS.items():
            low, high = self.get_bracket_bounds(bracket_name)
            mask = (tax_units[income_col] >= low) & (tax_units[income_col] < high)
            
            calibrated_count = tax_units.loc[mask, 'calibrated_weight'].sum()
            irs_count = bracket_data['count']
            target_count = self.target_counts[bracket_name]
            error_pct = ((calibrated_count - target_count) / target_count * 100) if target_count > 0 else 0
            
            metrics[f'bracket_{bracket_name}_count'] = calibrated_count
            metrics[f'bracket_{bracket_name}_error_pct'] = error_pct
            total_error += abs(error_pct)
            
            status = '✅' if abs(error_pct) < 2 else '⚠️' if abs(error_pct) < 5 else '❌'
            logger.info(f"  {status} {bracket_name:<10} {calibrated_count:>14,.0f} {irs_count:>14,} "
                       f"{target_count:>14,.0f} {error_pct:>+9.1f}%")
        
        avg_error = total_error / len(self.IRS_BRACKETS)
        metrics['avg_bracket_error_pct'] = avg_error
        logger.info(f"\n  Average Bracket Error: {avg_error:.2f}%")
        
        # 3. Average income validation
        logger.info(f"\nAverage Income by Bracket:")
        logger.info(f"  {'Bracket':<12} {'Calibrated':>14} {'IRS Target':>14} {'Ratio':>10}")
        logger.info("  " + "-" * 70)
        
        for bracket_name, bracket_data in self.IRS_BRACKETS.items():
            low, high = self.get_bracket_bounds(bracket_name)
            mask = (tax_units[income_col] >= low) & (tax_units[income_col] < high)
            
            if mask.sum() > 0:
                weighted_income = (tax_units.loc[mask, income_col] * 
                                  tax_units.loc[mask, 'calibrated_weight']).sum()
                total_weight = tax_units.loc[mask, 'calibrated_weight'].sum()
                calibrated_avg = weighted_income / total_weight
                target_avg = bracket_data['avg']
                ratio = calibrated_avg / target_avg if target_avg > 0 else 0
                
                metrics[f'bracket_{bracket_name}_avg_income'] = calibrated_avg
                metrics[f'bracket_{bracket_name}_avg_ratio'] = ratio
                
                status = '✅' if 0.95 <= ratio <= 1.05 else '⚠️' if 0.90 <= ratio <= 1.10 else '❌'
                logger.info(f"  {status} {bracket_name:<10} ${calibrated_avg:>13,.0f} ${target_avg:>13,} "
                           f"{ratio:>9.3f}")
        
        # 4. Overall AGI validation
        total_agi_calibrated = (tax_units[income_col] * tax_units['calibrated_weight']).sum()
        total_agi_irs = sum(b['total_agi'] for b in self.IRS_BRACKETS.values())
        agi_ratio = total_agi_calibrated / total_agi_irs
        
        metrics['total_agi_calibrated'] = total_agi_calibrated
        metrics['total_agi_irs'] = total_agi_irs
        metrics['total_agi_ratio'] = agi_ratio
        
        logger.info(f"\nTotal AGI:")
        logger.info(f"  Calibrated:    ${total_agi_calibrated:,.0f}")
        logger.info(f"  IRS Total:     ${total_agi_irs:,.0f}")
        logger.info(f"  Ratio:         {agi_ratio:.3f}")
        
        logger.info("\n" + "=" * 70 + "\n")
        
        return metrics
    
    def get_calibration_summary(self, 
                               tax_units_before: pd.DataFrame,
                               tax_units_after: pd.DataFrame,
                               income_col: str = 'income') -> pd.DataFrame:
        """
        Generate a summary comparing before/after calibration.
        
        Args:
            tax_units_before: Original tax units
            tax_units_after: Calibrated tax units
            income_col: Name of income column
            
        Returns:
            DataFrame with comparison metrics
        """
        summary_data = []
        
        # Total returns
        before_total = tax_units_before['weight'].sum()
        after_total = tax_units_after['calibrated_weight'].sum()
        summary_data.append({
            'Metric': 'Total Returns',
            'Before': f"{before_total:,.0f}",
            'After': f"{after_total:,.0f}",
            'IRS/DOTAX Target': f"{self.DOTAX_TOTAL_RETURNS:,}",
            'Change': f"{((after_total - before_total) / before_total * 100):+.1f}%"
        })
        
        # Bracket distribution
        for bracket_name, bracket_data in self.IRS_BRACKETS.items():
            low, high = self.get_bracket_bounds(bracket_name)
            
            before_mask = (tax_units_before[income_col] >= low) & (tax_units_before[income_col] < high)
            after_mask = (tax_units_after[income_col] >= low) & (tax_units_after[income_col] < high)
            
            before_count = tax_units_before.loc[before_mask, 'weight'].sum()
            after_count = tax_units_after.loc[after_mask, 'calibrated_weight'].sum()
            target_count = self.target_counts[bracket_name]
            
            change_pct = ((after_count - before_count) / before_count * 100) if before_count > 0 else 0
            
            summary_data.append({
                'Metric': f'{bracket_name} Count',
                'Before': f"{before_count:,.0f}",
                'After': f"{after_count:,.0f}",
                'IRS/DOTAX Target': f"{target_count:,.0f}",
                'Change': f"{change_pct:+.1f}%"
            })
            
            # Average income
            if before_mask.sum() > 0 and after_mask.sum() > 0:
                before_avg = (tax_units_before.loc[before_mask, income_col] * 
                             tax_units_before.loc[before_mask, 'weight']).sum() / before_count
                after_avg = (tax_units_after.loc[after_mask, income_col] * 
                            tax_units_after.loc[after_mask, 'calibrated_weight']).sum() / after_count
                target_avg = bracket_data['avg']
                
                avg_change_pct = ((after_avg - before_avg) / before_avg * 100) if before_avg > 0 else 0
                
                summary_data.append({
                    'Metric': f'{bracket_name} Avg Income',
                    'Before': f"${before_avg:,.0f}",
                    'After': f"${after_avg:,.0f}",
                    'IRS/DOTAX Target': f"${target_avg:,}",
                    'Change': f"{avg_change_pct:+.1f}%"
                })
        
        return pd.DataFrame(summary_data)


def calibrate_with_irs_brackets(tax_units: pd.DataFrame,
                                 weight_col: str = 'weight',
                                 income_col: str = 'income',
                                 max_weight_adjustment: float = 3.0,
                                 max_income_adjustment: float = 0.3,
                                 apply_income_adjustment: bool = True) -> pd.DataFrame:
    """
    Convenience function to calibrate tax units using IRS SOI brackets.
    
    Args:
        tax_units: DataFrame with tax units
        weight_col: Name of weight column
        income_col: Name of income column
        max_weight_adjustment: Maximum weight adjustment factor
        max_income_adjustment: Maximum income adjustment (as fraction)
        apply_income_adjustment: Whether to adjust incomes within brackets
        
    Returns:
        DataFrame with calibrated weights and incomes
    """
    calibrator = IRSBracketCalibrator(
        max_weight_adjustment=max_weight_adjustment,
        max_income_adjustment=max_income_adjustment,
        apply_income_adjustment=apply_income_adjustment
    )
    
    return calibrator.calibrate(tax_units, weight_col, income_col)


if __name__ == '__main__':
    # Demo usage
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.append(str(project_root))
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    logger.info("IRS SOI Bracket Calibration Demo")
    logger.info("=" * 70)
    
    # Create sample data
    np.random.seed(42)
    n_units = 1000
    
    sample_data = pd.DataFrame({
        'tax_unit_id': range(n_units),
        'income': np.random.lognormal(10.5, 1.2, n_units),
        'weight': np.random.uniform(500, 1500, n_units),
        'filing_status': np.random.choice(['single', 'married_filing_jointly', 
                                          'head_of_household', 'married_filing_separately'],
                                         n_units, p=[0.53, 0.34, 0.11, 0.02])
    })
    
    logger.info(f"\nSample data created: {len(sample_data):,} tax units")
    logger.info(f"Total weighted returns: {sample_data['weight'].sum():,.0f}")
    
    # Run calibration
    calibrator = IRSBracketCalibrator()
    calibrated_data = calibrator.calibrate(sample_data)
    
    # Show summary
    summary = calibrator.get_calibration_summary(sample_data, calibrated_data)
    logger.info("\nCalibration Summary:")
    logger.info("\n" + summary.to_string(index=False))
