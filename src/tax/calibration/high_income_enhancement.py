"""
High-Income Enhancement for Hawaii Tax Model

This module implements Stage 4 calibration: generating synthetic high-income
tax units to address PUMS undersampling of wealthy households. Uses IRS SOI
$200k+ bracket data and percentile floors to create realistic synthetic records.

Key Features:
- Identifies high-income gap between PUMS and IRS
- Generates synthetic records using Pareto distribution
- Matches IRS average income and percentile floors
- Re-calibrates to maintain DOTAX total
"""

import pandas as pd
import numpy as np
from scipy.stats import pareto
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class HighIncomeEnhancer:
    """
    Enhances PUMS tax units by adding synthetic high-income records.
    
    PUMS significantly undersamples high-income households due to:
    - Survey response bias (wealthy less likely to respond)
    - Top-coding of income variables
    - Small sample size for rare cases
    
    This class generates synthetic records to fill the gap using:
    - IRS SOI $200k+ bracket data (count and average)
    - IRS percentile floors (e.g., top 1% starts at $650k)
    - Pareto distribution fitted to match IRS targets
    """
    
    # IRS SOI 2022 High-Income Bracket Data
    IRS_HIGH_INCOME_THRESHOLD = 200000
    IRS_HIGH_INCOME_COUNT = 15000      # Returns > $200k
    IRS_HIGH_INCOME_AVG = 400000       # Average AGI in $200k+ bracket
    IRS_HIGH_INCOME_TOTAL = 6000000000 # Total AGI in $200k+ bracket
    
    # IRS Percentile Floors (from SOI percentile data)
    IRS_TOP1_FLOOR = 650000   # Top 1% starts at $650k
    IRS_TOP5_FLOOR = 300000   # Top 5% starts at $300k
    IRS_TOP10_FLOOR = 200000  # Top 10% starts at $200k
    
    # DOTAX Total (to maintain after enhancement)
    DOTAX_TOTAL_RETURNS = 634956
    
    def __init__(self, 
                 pareto_shape: float = 2.3,
                 pareto_scale: float = 50000,
                 verify_targets: bool = True,
                 max_iterations: int = 10):
        """
        Initialize high-income enhancer.
        
        Args:
            pareto_shape: Shape parameter for Pareto distribution (alpha)
            pareto_scale: Scale parameter for Pareto distribution
            verify_targets: Whether to verify synthetic data matches IRS targets
            max_iterations: Max iterations for Pareto parameter tuning
        """
        self.pareto_shape = pareto_shape
        self.pareto_scale = pareto_scale
        self.verify_targets = verify_targets
        self.max_iterations = max_iterations
        
        logger.info("High-Income Enhancer initialized")
        logger.info(f"  IRS $200k+ count: {self.IRS_HIGH_INCOME_COUNT:,}")
        logger.info(f"  IRS $200k+ avg: ${self.IRS_HIGH_INCOME_AVG:,}")
        logger.info(f"  IRS top 1% floor: ${self.IRS_TOP1_FLOOR:,}")
    
    def calculate_high_income_gap(self, 
                                   tax_units: pd.DataFrame,
                                   weight_col: str = 'calibrated_weight',
                                   income_col: str = 'income') -> Dict[str, float]:
        """
        Calculate gap between PUMS and IRS high-income counts.
        
        Args:
            tax_units: DataFrame with tax units
            weight_col: Name of weight column
            income_col: Name of income column
            
        Returns:
            Dictionary with gap statistics
        """
        logger.info("\n" + "=" * 70)
        logger.info("Calculating High-Income Gap")
        logger.info("=" * 70)
        
        # Current PUMS high-income count
        high_income_mask = tax_units[income_col] > self.IRS_HIGH_INCOME_THRESHOLD
        pums_high_count = tax_units.loc[high_income_mask, weight_col].sum()
        
        # Calculate gap
        gap = self.IRS_HIGH_INCOME_COUNT - pums_high_count
        gap_pct = (gap / self.IRS_HIGH_INCOME_COUNT) * 100
        
        # Current PUMS high-income average
        if high_income_mask.sum() > 0:
            pums_high_avg = (
                tax_units.loc[high_income_mask, income_col] * 
                tax_units.loc[high_income_mask, weight_col]
            ).sum() / pums_high_count
        else:
            pums_high_avg = 0
        
        # Statistics
        stats = {
            'irs_target_count': self.IRS_HIGH_INCOME_COUNT,
            'pums_current_count': pums_high_count,
            'gap_count': gap,
            'gap_pct': gap_pct,
            'irs_target_avg': self.IRS_HIGH_INCOME_AVG,
            'pums_current_avg': pums_high_avg,
            'irs_top1_floor': self.IRS_TOP1_FLOOR
        }
        
        logger.info(f"\nHigh-Income Gap Analysis (>${self.IRS_HIGH_INCOME_THRESHOLD:,}):")
        logger.info(f"  IRS Target Count:  {self.IRS_HIGH_INCOME_COUNT:>12,}")
        logger.info(f"  PUMS Current:      {pums_high_count:>12,.0f}")
        logger.info(f"  Gap:               {gap:>12,.0f} ({gap_pct:+.1f}%)")
        logger.info(f"\n  IRS Target Avg:    ${self.IRS_HIGH_INCOME_AVG:>11,}")
        logger.info(f"  PUMS Current Avg:  ${pums_high_avg:>11,.0f}")
        logger.info(f"\n  IRS Top 1% Floor:  ${self.IRS_TOP1_FLOOR:>11,}")
        
        if gap > 0:
            logger.info(f"\n  ⚠️  Need to add {gap:,.0f} synthetic high-income records")
        else:
            logger.info(f"\n  ✅ PUMS already has sufficient high-income records")
        
        return stats
    
    def fit_pareto_distribution(self, 
                                target_avg: float,
                                target_p99: float,
                                loc: float = 200000) -> Tuple[float, float]:
        """
        Fit Pareto distribution to match IRS targets.
        
        The Pareto distribution is commonly used to model high-income distributions
        due to its heavy right tail. We fit it to match:
        - Average income in $200k+ bracket
        - 99th percentile (top 1% floor)
        
        Args:
            target_avg: Target average income
            target_p99: Target 99th percentile
            loc: Location parameter (minimum value)
            
        Returns:
            Tuple of (shape, scale) parameters
        """
        logger.info("\nFitting Pareto Distribution:")
        logger.info(f"  Target Average: ${target_avg:,}")
        logger.info(f"  Target P99:     ${target_p99:,}")
        logger.info(f"  Location:       ${loc:,}")
        
        best_shape = self.pareto_shape
        best_scale = self.pareto_scale
        best_error = float('inf')
        
        # Grid search for best parameters
        for shape in np.linspace(1.5, 3.5, 20):
            for scale in np.linspace(30000, 100000, 15):
                # Generate sample
                sample = pareto.rvs(shape, loc=loc, scale=scale, size=10000)
                
                # Calculate metrics
                sample_avg = sample.mean()
                sample_p99 = np.percentile(sample, 99)
                
                # Calculate error
                avg_error = abs(sample_avg - target_avg) / target_avg
                p99_error = abs(sample_p99 - target_p99) / target_p99
                total_error = avg_error + p99_error
                
                if total_error < best_error:
                    best_error = total_error
                    best_shape = shape
                    best_scale = scale
        
        # Verify best fit
        sample = pareto.rvs(best_shape, loc=loc, scale=best_scale, size=10000)
        sample_avg = sample.mean()
        sample_p99 = np.percentile(sample, 99)
        
        logger.info(f"\n  Best Fit Parameters:")
        logger.info(f"    Shape: {best_shape:.3f}")
        logger.info(f"    Scale: ${best_scale:,.0f}")
        logger.info(f"\n  Verification:")
        logger.info(f"    Sample Avg: ${sample_avg:,.0f} (target: ${target_avg:,})")
        logger.info(f"    Sample P99: ${sample_p99:,.0f} (target: ${target_p99:,})")
        logger.info(f"    Avg Error:  {abs(sample_avg - target_avg) / target_avg * 100:.1f}%")
        logger.info(f"    P99 Error:  {abs(sample_p99 - target_p99) / target_p99 * 100:.1f}%")
        
        return best_shape, best_scale
    
    def generate_synthetic_records(self,
                                    n_records: int,
                                    reference_units: pd.DataFrame,
                                    income_col: str = 'income') -> pd.DataFrame:
        """
        Generate synthetic high-income tax unit records.
        
        Args:
            n_records: Number of synthetic records to generate
            reference_units: Reference tax units for demographic patterns
            income_col: Name of income column
            
        Returns:
            DataFrame with synthetic tax units
        """
        logger.info("\n" + "=" * 70)
        logger.info(f"Generating {n_records:,.0f} Synthetic High-Income Records")
        logger.info("=" * 70)
        
        # Fit Pareto distribution if needed
        if self.verify_targets:
            shape, scale = self.fit_pareto_distribution(
                target_avg=self.IRS_HIGH_INCOME_AVG,
                target_p99=self.IRS_TOP1_FLOOR,
                loc=self.IRS_HIGH_INCOME_THRESHOLD
            )
        else:
            shape = self.pareto_shape
            scale = self.pareto_scale
        
        # Generate synthetic incomes
        synthetic_incomes = pareto.rvs(
            shape, 
            loc=self.IRS_HIGH_INCOME_THRESHOLD, 
            scale=scale, 
            size=int(n_records)
        )
        
        # Verify synthetic data
        logger.info(f"\nSynthetic Income Statistics:")
        logger.info(f"  Count:      {len(synthetic_incomes):,}")
        logger.info(f"  Mean:       ${synthetic_incomes.mean():,.0f}")
        logger.info(f"  Median:     ${np.median(synthetic_incomes):,.0f}")
        logger.info(f"  P90:        ${np.percentile(synthetic_incomes, 90):,.0f}")
        logger.info(f"  P95:        ${np.percentile(synthetic_incomes, 95):,.0f}")
        logger.info(f"  P99:        ${np.percentile(synthetic_incomes, 99):,.0f}")
        logger.info(f"  Max:        ${synthetic_incomes.max():,.0f}")
        
        # Sample demographic characteristics from high-income PUMS units
        high_income_reference = reference_units[
            reference_units[income_col] > self.IRS_HIGH_INCOME_THRESHOLD
        ]
        
        if len(high_income_reference) == 0:
            # If no high-income units, use overall distribution
            logger.warning("  ⚠️  No high-income reference units, using overall distribution")
            high_income_reference = reference_units
        
        # Sample with replacement to get demographic patterns
        sampled_indices = np.random.choice(
            high_income_reference.index,
            size=int(n_records),
            replace=True
        )
        
        synthetic_units = high_income_reference.loc[sampled_indices].copy()
        synthetic_units = synthetic_units.reset_index(drop=True)
        
        # Replace incomes with synthetic values
        synthetic_units[income_col] = synthetic_incomes
        
        # Set weight to 1 (will be calibrated later)
        synthetic_units['calibrated_weight'] = 1.0
        synthetic_units['weight'] = 1.0
        
        # Mark as synthetic
        synthetic_units['is_synthetic'] = True
        
        # Generate unique IDs
        synthetic_units['tax_unit_id'] = [
            f'synthetic_{i}' for i in range(len(synthetic_units))
        ]
        
        logger.info(f"\n✅ Generated {len(synthetic_units):,} synthetic records")
        
        return synthetic_units
    
    def enhance(self,
                tax_units: pd.DataFrame,
                weight_col: str = 'calibrated_weight',
                income_col: str = 'income') -> pd.DataFrame:
        """
        Enhance tax units by adding synthetic high-income records.
        
        Args:
            tax_units: DataFrame with tax units
            weight_col: Name of weight column
            income_col: Name of income column
            
        Returns:
            Enhanced DataFrame with synthetic records added
        """
        logger.info("\n" + "=" * 70)
        logger.info("HIGH-INCOME ENHANCEMENT - STAGE 4")
        logger.info("=" * 70)
        
        # Calculate gap
        gap_stats = self.calculate_high_income_gap(tax_units, weight_col, income_col)
        
        if gap_stats['gap_count'] <= 0:
            logger.info("\n✅ No enhancement needed - PUMS already has sufficient high-income records")
            return tax_units
        
        # Generate synthetic records
        synthetic_units = self.generate_synthetic_records(
            n_records=gap_stats['gap_count'],
            reference_units=tax_units,
            income_col=income_col
        )
        
        # Mark original units
        tax_units_marked = tax_units.copy()
        if 'is_synthetic' not in tax_units_marked.columns:
            tax_units_marked['is_synthetic'] = False
        
        # Combine original and synthetic
        logger.info("\nCombining Original and Synthetic Records:")
        logger.info(f"  Original PUMS:  {len(tax_units_marked):>10,} records")
        logger.info(f"  Synthetic:      {len(synthetic_units):>10,} records")
        
        enhanced = pd.concat([tax_units_marked, synthetic_units], ignore_index=True)
        
        logger.info(f"  Total:          {len(enhanced):>10,} records")
        
        # Re-calibrate to maintain DOTAX total
        enhanced = self._recalibrate_to_dotax(enhanced, weight_col)
        
        # Validate enhancement
        self.validate(enhanced, weight_col, income_col)
        
        logger.info("\n" + "=" * 70)
        logger.info("HIGH-INCOME ENHANCEMENT COMPLETE")
        logger.info("=" * 70 + "\n")
        
        return enhanced
    
    def _recalibrate_to_dotax(self,
                              tax_units: pd.DataFrame,
                              weight_col: str) -> pd.DataFrame:
        """
        Re-calibrate weights to maintain DOTAX total after adding synthetic records.
        
        Args:
            tax_units: DataFrame with tax units (including synthetic)
            weight_col: Name of weight column
            
        Returns:
            DataFrame with re-calibrated weights
        """
        logger.info("\nRe-calibrating to DOTAX Total:")
        
        current_total = tax_units[weight_col].sum()
        scaling_factor = self.DOTAX_TOTAL_RETURNS / current_total
        
        logger.info(f"  Current Total:  {current_total:>12,.0f}")
        logger.info(f"  DOTAX Target:   {self.DOTAX_TOTAL_RETURNS:>12,}")
        logger.info(f"  Scaling Factor: {scaling_factor:>12.6f}")
        
        tax_units[weight_col] = tax_units[weight_col] * scaling_factor
        
        final_total = tax_units[weight_col].sum()
        logger.info(f"  Final Total:    {final_total:>12,.0f}")
        
        return tax_units
    
    def validate(self,
                 tax_units: pd.DataFrame,
                 weight_col: str = 'calibrated_weight',
                 income_col: str = 'income') -> Dict[str, float]:
        """
        Validate enhanced tax units against IRS targets.
        
        Args:
            tax_units: Enhanced DataFrame
            weight_col: Name of weight column
            income_col: Name of income column
            
        Returns:
            Dictionary of validation metrics
        """
        logger.info("\n" + "=" * 70)
        logger.info("High-Income Enhancement Validation")
        logger.info("=" * 70)
        
        metrics = {}
        
        # 1. Total returns
        total_returns = tax_units[weight_col].sum()
        returns_error = abs(total_returns - self.DOTAX_TOTAL_RETURNS) / self.DOTAX_TOTAL_RETURNS
        metrics['total_returns'] = total_returns
        metrics['returns_error_pct'] = returns_error * 100
        
        logger.info(f"\nTotal Returns:")
        logger.info(f"  Enhanced:      {total_returns:>12,.0f}")
        logger.info(f"  DOTAX Target:  {self.DOTAX_TOTAL_RETURNS:>12,}")
        logger.info(f"  Error:         {returns_error*100:>12.3f}%")
        
        # 2. High-income count
        high_income_mask = tax_units[income_col] > self.IRS_HIGH_INCOME_THRESHOLD
        high_income_count = tax_units.loc[high_income_mask, weight_col].sum()
        high_count_error = abs(high_income_count - self.IRS_HIGH_INCOME_COUNT) / self.IRS_HIGH_INCOME_COUNT
        metrics['high_income_count'] = high_income_count
        metrics['high_count_error_pct'] = high_count_error * 100
        
        logger.info(f"\nHigh-Income Count (>${self.IRS_HIGH_INCOME_THRESHOLD:,}):")
        logger.info(f"  Enhanced:      {high_income_count:>12,.0f}")
        logger.info(f"  IRS Target:    {self.IRS_HIGH_INCOME_COUNT:>12,}")
        logger.info(f"  Error:         {high_count_error*100:>12.1f}%")
        
        # 3. High-income average
        if high_income_mask.sum() > 0:
            high_income_avg = (
                tax_units.loc[high_income_mask, income_col] * 
                tax_units.loc[high_income_mask, weight_col]
            ).sum() / high_income_count
            avg_ratio = high_income_avg / self.IRS_HIGH_INCOME_AVG
            metrics['high_income_avg'] = high_income_avg
            metrics['high_avg_ratio'] = avg_ratio
            
            logger.info(f"\nHigh-Income Average:")
            logger.info(f"  Enhanced:      ${high_income_avg:>11,.0f}")
            logger.info(f"  IRS Target:    ${self.IRS_HIGH_INCOME_AVG:>11,}")
            logger.info(f"  Ratio:         {avg_ratio:>12.3f}")
        
        # 4. Percentile validation
        # Calculate weighted percentiles
        sorted_units = tax_units.sort_values(income_col)
        cumulative_weight = sorted_units[weight_col].cumsum()
        total_weight = sorted_units[weight_col].sum()
        
        # Find 99th percentile
        p99_idx = (cumulative_weight >= 0.99 * total_weight).idxmax()
        p99_income = sorted_units.loc[p99_idx, income_col]
        metrics['p99_income'] = p99_income
        
        logger.info(f"\nPercentile Validation:")
        logger.info(f"  P99 (Top 1%):  ${p99_income:>11,.0f}")
        logger.info(f"  IRS Target:    ${self.IRS_TOP1_FLOOR:>11,}")
        logger.info(f"  Ratio:         {p99_income / self.IRS_TOP1_FLOOR:>12.3f}")
        
        # 5. Synthetic record statistics
        if 'is_synthetic' in tax_units.columns:
            n_synthetic = tax_units['is_synthetic'].sum()
            synthetic_weight = tax_units.loc[tax_units['is_synthetic'], weight_col].sum()
            synthetic_pct = (synthetic_weight / total_returns) * 100
            
            metrics['n_synthetic_records'] = n_synthetic
            metrics['synthetic_weight'] = synthetic_weight
            metrics['synthetic_pct'] = synthetic_pct
            
            logger.info(f"\nSynthetic Records:")
            logger.info(f"  Count:         {n_synthetic:>12,}")
            logger.info(f"  Weight:        {synthetic_weight:>12,.0f}")
            logger.info(f"  Percentage:    {synthetic_pct:>12.1f}%")
        
        logger.info("\n" + "=" * 70 + "\n")
        
        return metrics


def enhance_high_income(tax_units: pd.DataFrame,
                        weight_col: str = 'calibrated_weight',
                        income_col: str = 'income',
                        pareto_shape: float = 2.3,
                        pareto_scale: float = 50000) -> pd.DataFrame:
    """
    Convenience function to enhance tax units with synthetic high-income records.
    
    Args:
        tax_units: DataFrame with tax units
        weight_col: Name of weight column
        income_col: Name of income column
        pareto_shape: Shape parameter for Pareto distribution
        pareto_scale: Scale parameter for Pareto distribution
        
    Returns:
        Enhanced DataFrame with synthetic records
    """
    enhancer = HighIncomeEnhancer(
        pareto_shape=pareto_shape,
        pareto_scale=pareto_scale,
        verify_targets=True
    )
    
    return enhancer.enhance(tax_units, weight_col, income_col)


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
    
    logger.info("High-Income Enhancement Demo")
    logger.info("=" * 70)
    
    # Create sample data with underrepresented high-income
    np.random.seed(42)
    n_units = 5000
    
    # Generate incomes with insufficient high earners
    incomes = np.random.lognormal(10.5, 1.0, n_units)
    
    sample_data = pd.DataFrame({
        'tax_unit_id': range(n_units),
        'income': incomes,
        'calibrated_weight': np.random.uniform(100, 150, n_units),
        'filing_status': np.random.choice(
            ['single', 'married_filing_jointly', 'head_of_household'],
            n_units, p=[0.53, 0.37, 0.10]
        )
    })
    
    logger.info(f"\nSample data created: {len(sample_data):,} tax units")
    logger.info(f"Total weighted returns: {sample_data['calibrated_weight'].sum():,.0f}")
    
    # Run enhancement
    enhancer = HighIncomeEnhancer()
    enhanced_data = enhancer.enhance(sample_data)
    
    logger.info(f"\nEnhancement complete!")
    logger.info(f"Final count: {len(enhanced_data):,} tax units")
    logger.info(f"Final weighted returns: {enhanced_data['calibrated_weight'].sum():,.0f}")
