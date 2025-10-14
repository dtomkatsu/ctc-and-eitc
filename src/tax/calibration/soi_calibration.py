"""
SOI-based calibration for Hawaii tax model.

Calibrates PUMS-based tax units to match Hawaii DOTAX SOI 2022 benchmarks.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class SOICalibrator:
    """
    Calibrate tax units to Hawaii DOTAX SOI 2022 benchmarks.
    
    This class provides methods to align PUMS-based tax unit estimates with
    actual Statistics of Income data from Hawaii Department of Taxation.
    """
    
    # SOI 2022 Benchmarks from Hawaii DOTAX Table 5A
    SOI_FILING_STATUS = {
        'single': 52.8,
        'married_filing_jointly': 34.1,
        'married_filing_separately': 2.5,
        'head_of_household': 10.6
    }
    
    SOI_TOTAL_RETURNS = 635117
    
    SOI_AVG_AGI = {
        'married_filing_jointly': 120767,
        'single': 41907,
        'married_filing_separately': 195040,
        'head_of_household': 55273
    }
    
    SOI_EFFECTIVE_RATES = {
        'married_filing_jointly': 5.94,
        'single': 5.82,
        'married_filing_separately': 7.98,
        'head_of_household': 4.75
    }
    
    SOI_CREDIT_RATES = {
        'married_filing_jointly': 0.073,  # 7.3% reduction
        'single': 0.053,  # 5.3% reduction
        'married_filing_separately': 0.138,  # 13.8% reduction
        'head_of_household': 0.124  # 12.4% reduction
    }
    
    def __init__(self, apply_weight_scaling: bool = True, 
                 apply_filing_status_calibration: bool = True,
                 apply_credit_calibration: bool = True):
        """
        Initialize SOI calibrator.
        
        Args:
            apply_weight_scaling: Scale weights to match total returns
            apply_filing_status_calibration: Adjust filing status distribution
            apply_credit_calibration: Apply SOI-based credit rates
        """
        self.apply_weight_scaling = apply_weight_scaling
        self.apply_filing_status_calibration = apply_filing_status_calibration
        self.apply_credit_calibration = apply_credit_calibration
        
    def calibrate(self, tax_units: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all enabled calibrations to tax units.
        
        Args:
            tax_units: DataFrame with tax unit data
            
        Returns:
            Calibrated DataFrame with new 'calibrated_weight' column
        """
        logger.info("Starting SOI calibration...")
        
        # Create copy to avoid modifying original
        calibrated = tax_units.copy()
        
        # Initialize calibrated_weight with original weight
        calibrated['calibrated_weight'] = calibrated['weight']
        
        # Apply calibrations in sequence
        if self.apply_weight_scaling:
            calibrated = self.calibrate_total_returns(calibrated)
            
        if self.apply_filing_status_calibration:
            calibrated = self.calibrate_filing_status(calibrated)
            
        if self.apply_credit_calibration:
            calibrated = self.apply_credit_rates(calibrated)
            
        # Validate results
        self.validate(calibrated)
        
        logger.info("SOI calibration complete")
        return calibrated
    
    def calibrate_total_returns(self, tax_units: pd.DataFrame) -> pd.DataFrame:
        """
        Scale weights so total returns match SOI count.
        
        Args:
            tax_units: DataFrame with tax unit data
            
        Returns:
            DataFrame with scaled weights
        """
        current_total = tax_units['calibrated_weight'].sum()
        scaling_factor = self.SOI_TOTAL_RETURNS / current_total
        
        logger.info(f"Scaling weights: {current_total:,.0f} → {self.SOI_TOTAL_RETURNS:,} "
                   f"(factor: {scaling_factor:.4f})")
        
        tax_units['calibrated_weight'] = tax_units['calibrated_weight'] * scaling_factor
        
        return tax_units
    
    def calibrate_filing_status(self, tax_units: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust filing status distribution to match SOI targets.
        
        Args:
            tax_units: DataFrame with tax unit data
            
        Returns:
            DataFrame with adjusted weights
        """
        logger.info("Calibrating filing status distribution...")
        
        # Calculate current weighted distribution
        current = tax_units.groupby('filing_status')['calibrated_weight'].sum()
        total = current.sum()
        current_pct = (current / total) * 100
        
        # Calculate adjustment factors
        adjustments = {}
        for status, target_pct in self.SOI_FILING_STATUS.items():
            current_pct_val = current_pct.get(status, 0)
            if current_pct_val > 0:
                adjustment = target_pct / current_pct_val
                adjustments[status] = adjustment
                logger.info(f"  {status}: {current_pct_val:.1f}% → {target_pct:.1f}% "
                           f"(factor: {adjustment:.4f})")
            else:
                adjustments[status] = 1.0
                logger.warning(f"  {status}: No units found, cannot calibrate")
        
        # Apply adjustments to weights
        for status, factor in adjustments.items():
            mask = tax_units['filing_status'] == status
            tax_units.loc[mask, 'calibrated_weight'] = \
                tax_units.loc[mask, 'calibrated_weight'] * factor
        
        return tax_units
    
    def apply_credit_rates(self, tax_units: pd.DataFrame) -> pd.DataFrame:
        """
        Apply SOI-calibrated credit rates to tax calculations.
        
        Args:
            tax_units: DataFrame with tax unit data
            
        Returns:
            DataFrame with calibrated credits and tax after credits
        """
        logger.info("Applying SOI-calibrated credit rates...")
        
        # Check if tax_before_credits exists, if not use 'tax' column
        if 'tax_before_credits' not in tax_units.columns:
            if 'tax' in tax_units.columns:
                tax_units['tax_before_credits'] = tax_units['tax']
            else:
                logger.warning("No tax columns found, skipping credit calibration")
                return tax_units
        
        # Apply credit rates by filing status
        for status, rate in self.SOI_CREDIT_RATES.items():
            mask = tax_units['filing_status'] == status
            if mask.sum() > 0:
                # Calculate credits as percentage of tax before credits
                tax_units.loc[mask, 'soi_credits'] = \
                    tax_units.loc[mask, 'tax_before_credits'] * rate
                tax_units.loc[mask, 'tax_after_credits'] = \
                    tax_units.loc[mask, 'tax_before_credits'] * (1 - rate)
                
                logger.info(f"  {status}: {rate*100:.1f}% credit rate applied")
        
        return tax_units
    
    def validate(self, tax_units: pd.DataFrame) -> Dict[str, float]:
        """
        Validate calibrated results against SOI benchmarks.
        
        Args:
            tax_units: Calibrated DataFrame
            
        Returns:
            Dictionary of validation metrics
        """
        logger.info("\n=== SOI Calibration Validation ===")
        
        metrics = {}
        
        # 1. Total returns
        total_returns = tax_units['calibrated_weight'].sum()
        returns_error = abs(total_returns - self.SOI_TOTAL_RETURNS) / self.SOI_TOTAL_RETURNS
        metrics['total_returns'] = total_returns
        metrics['returns_error_pct'] = returns_error * 100
        
        logger.info(f"\nTotal Returns:")
        logger.info(f"  Calibrated: {total_returns:,.0f}")
        logger.info(f"  SOI Target: {self.SOI_TOTAL_RETURNS:,}")
        logger.info(f"  Error: {returns_error*100:.2f}%")
        
        # 2. Filing status distribution
        logger.info(f"\nFiling Status Distribution:")
        logger.info(f"  {'Status':<30} {'Calibrated':>10} {'SOI Target':>10} {'Error':>8}")
        logger.info("  " + "-" * 60)
        
        current = tax_units.groupby('filing_status')['calibrated_weight'].sum()
        current_pct = (current / total_returns) * 100
        
        for status, target_pct in self.SOI_FILING_STATUS.items():
            cal_pct = current_pct.get(status, 0)
            error = cal_pct - target_pct
            metrics[f'filing_status_{status}_error'] = error
            
            symbol = '✅' if abs(error) < 1 else '⚠️' if abs(error) < 2 else '❌'
            logger.info(f"  {symbol} {status:<27} {cal_pct:>9.1f}% {target_pct:>9.1f}% {error:>+7.1f}pp")
        
        # 3. Average AGI validation (if income column exists)
        if 'income' in tax_units.columns:
            logger.info(f"\nAverage AGI by Filing Status:")
            logger.info(f"  {'Status':<30} {'Calibrated':>12} {'SOI Target':>12} {'Ratio':>8}")
            logger.info("  " + "-" * 65)
            
            for status, target_agi in self.SOI_AVG_AGI.items():
                status_units = tax_units[tax_units['filing_status'] == status]
                if len(status_units) > 0:
                    cal_avg_agi = (status_units['income'] * status_units['calibrated_weight']).sum() / \
                                  status_units['calibrated_weight'].sum()
                    ratio = cal_avg_agi / target_agi
                    metrics[f'avg_agi_{status}_ratio'] = ratio
                    
                    symbol = '✅' if 0.9 <= ratio <= 1.1 else '⚠️' if 0.8 <= ratio <= 1.2 else '❌'
                    logger.info(f"  {symbol} {status:<27} ${cal_avg_agi:>11,.0f} ${target_agi:>11,.0f} {ratio:>7.2f}")
        
        # 4. Credit impact validation (if tax columns exist)
        if 'tax_before_credits' in tax_units.columns and 'tax_after_credits' in tax_units.columns:
            logger.info(f"\nCredit Impact by Filing Status:")
            logger.info(f"  {'Status':<30} {'Calibrated':>10} {'SOI Target':>10} {'Error':>8}")
            logger.info("  " + "-" * 60)
            
            for status, target_rate in self.SOI_CREDIT_RATES.items():
                status_units = tax_units[tax_units['filing_status'] == status]
                if len(status_units) > 0:
                    total_before = (status_units['tax_before_credits'] * status_units['calibrated_weight']).sum()
                    total_after = (status_units['tax_after_credits'] * status_units['calibrated_weight']).sum()
                    
                    if total_before > 0:
                        cal_rate = (total_before - total_after) / total_before
                        error = (cal_rate - target_rate) * 100
                        metrics[f'credit_rate_{status}_error'] = error
                        
                        symbol = '✅' if abs(error) < 1 else '⚠️' if abs(error) < 2 else '❌'
                        logger.info(f"  {symbol} {status:<27} {cal_rate*100:>9.1f}% {target_rate*100:>9.1f}% {error:>+7.1f}pp")
        
        logger.info("\n" + "=" * 70 + "\n")
        
        return metrics
    
    def get_calibration_summary(self, tax_units_before: pd.DataFrame, 
                               tax_units_after: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a summary comparing before/after calibration.
        
        Args:
            tax_units_before: Original tax units
            tax_units_after: Calibrated tax units
            
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
            'SOI Target': f"{self.SOI_TOTAL_RETURNS:,}",
            'Change': f"{((after_total - before_total) / before_total * 100):+.1f}%"
        })
        
        # Filing status distribution
        before_dist = tax_units_before.groupby('filing_status')['weight'].sum()
        before_pct = (before_dist / before_total) * 100
        
        after_dist = tax_units_after.groupby('filing_status')['calibrated_weight'].sum()
        after_pct = (after_dist / after_total) * 100
        
        for status, target_pct in self.SOI_FILING_STATUS.items():
            before_val = before_pct.get(status, 0)
            after_val = after_pct.get(status, 0)
            summary_data.append({
                'Metric': f'{status} %',
                'Before': f"{before_val:.1f}%",
                'After': f"{after_val:.1f}%",
                'SOI Target': f"{target_pct:.1f}%",
                'Change': f"{(after_val - before_val):+.1f}pp"
            })
        
        return pd.DataFrame(summary_data)
