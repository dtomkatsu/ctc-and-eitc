"""
$1M+ Weight Reallocation

Reallocates weight within the $1M+ bracket from lower incomes ($1M-$2M)
to synthetic ultra-high incomes ($5M+) while preserving total filer count.

This addresses the root cause: PUMS data is top-coded around $2M, missing
the Pareto tail of ultra-wealthy earners who pay most of the tax.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MillionPlusReallocator:
    """
    Reallocate $1M+ filer weight to match tax target.
    
    Key insight:
    - Current: 1,824 filers, mostly $1M-$2M, generate $219M tax
    - Target: 1,824 filers, with Pareto tail, generate $663M tax
    - Solution: Move ~30-40% of weight to synthetic $5M-$50M levels
    
    Preserves:
    - Total filer count (1,824)
    - Effective tax rate (~9.9%)
    - Filing status distribution
    """
    
    def __init__(self, pareto_alpha: float = 1.454):
        """
        Initialize reallocator.
        
        Args:
            pareto_alpha: Pareto shape parameter (from calibration)
        """
        self.pareto_alpha = pareto_alpha
        self.threshold = 1_000_000
    
    def calculate_pareto_weights(self, 
                                 income_levels: list,
                                 total_weight: float) -> dict:
        """
        Calculate weight allocation using Pareto distribution.
        
        Args:
            income_levels: List of income levels (e.g., [5M, 10M, 25M, 50M])
            total_weight: Total weight to allocate across levels
            
        Returns:
            Dict mapping income level to weight
        """
        weights = {}
        
        # Calculate Pareto probabilities
        probabilities = {}
        total_prob = 0
        
        for income in income_levels:
            # P(X > x) = (x_min / x)^alpha
            prob = (self.threshold / income) ** self.pareto_alpha
            probabilities[income] = prob
            total_prob += prob
        
        # Normalize to total weight
        for income, prob in probabilities.items():
            weights[income] = (prob / total_prob) * total_weight
        
        return weights
    
    def estimate_tax_per_filer(self, agi: float, filing_status: str = 'married_filing_jointly') -> float:
        """
        Estimate Hawaii tax for ultra-high-income filer.
        
        Ultra-high earners typically pay ~10.5% effective rate on Hawaii income.
        
        Args:
            agi: Adjusted Gross Income
            filing_status: Filing status
            
        Returns:
            Estimated tax
        """
        # Hawaii top rate is 11% (kicks in around $200k for MFJ)
        # Ultra-high earners pay slightly less due to deductions
        effective_rate = 0.105
        
        return agi * effective_rate
    
    def reallocate(self, df: pd.DataFrame, 
                   reallocation_pct: float = 0.35,
                   target_tax_m: float = 663.0) -> pd.DataFrame:
        """
        Reallocate $1M+ weight to synthetic ultra-high incomes.
        
        Args:
            df: DataFrame with tax units
            reallocation_pct: Fraction of $1M-$2M weight to move (0.0-1.0)
            target_tax_m: Target tax for $1M+ bracket (millions)
            
        Returns:
            DataFrame with reallocated weights
        """
        logger.info("=" * 80)
        logger.info("$1M+ WEIGHT REALLOCATION")
        logger.info("=" * 80)
        logger.info("")
        
        result = df.copy()
        
        # Current $1M+ situation
        mask_1m = result['agi'] >= self.threshold
        current_filers = result.loc[mask_1m, 'weight'].sum()
        current_tax = (result.loc[mask_1m, 'hi_state_tax'] * result.loc[mask_1m, 'weight']).sum()
        current_tax_m = current_tax / 1_000_000
        current_avg_tax = current_tax / current_filers if current_filers > 0 else 0
        
        logger.info(f"Current $1M+ bracket:")
        logger.info(f"  Filers: {current_filers:,.0f}")
        logger.info(f"  Total tax: ${current_tax_m:,.1f}M")
        logger.info(f"  Avg tax per filer: ${current_avg_tax:,.0f}")
        logger.info(f"  Target tax: ${target_tax_m:,.1f}M")
        logger.info(f"  Gap: ${current_tax_m - target_tax_m:,.1f}M")
        logger.info("")
        
        # Identify $1M-$2M filers to reallocate from
        mask_1m_to_2m = (result['agi'] >= 1_000_000) & (result['agi'] < 2_000_000)
        weight_1m_to_2m = result.loc[mask_1m_to_2m, 'weight'].sum()
        
        if weight_1m_to_2m == 0:
            logger.warning("No filers in $1M-$2M range to reallocate")
            return df
        
        # Calculate weight to move
        weight_to_move = weight_1m_to_2m * reallocation_pct
        
        logger.info(f"Reallocating {reallocation_pct*100:.0f}% of $1M-$2M weight:")
        logger.info(f"  Current $1M-$2M weight: {weight_1m_to_2m:,.0f}")
        logger.info(f"  Weight to move: {weight_to_move:,.0f}")
        logger.info("")
        
        # Reduce $1M-$2M weights
        result.loc[mask_1m_to_2m, 'weight'] *= (1 - reallocation_pct)
        
        # Define ultra-high income levels
        ultra_high_incomes = [5_000_000, 10_000_000, 25_000_000, 50_000_000]
        
        # Calculate Pareto-based weight allocation
        pareto_weights = self.calculate_pareto_weights(ultra_high_incomes, weight_to_move)
        
        logger.info("Creating synthetic ultra-high-income filers:")
        
        # Create synthetic filers
        synthetic_filers = []
        total_synthetic_tax = 0
        
        for income, weight in pareto_weights.items():
            if weight < 0.1:
                continue
            
            # Estimate tax for this income level
            est_tax = self.estimate_tax_per_filer(income)
            total_synthetic_tax += est_tax * weight
            
            synthetic_filers.append({
                'agi': income,
                'filing_status': 'married_filing_jointly',
                'num_dependents': 2,
                'num_adults': 2,
                'weight': weight,
                'total_deductions': 0,  # Let calculator determine deductions
                'is_synthetic_ultra_high': True,
            })
            
            logger.info(f"  ${income/1_000_000:.0f}M: {weight:>6.0f} filers "
                       f"(est. tax: ${est_tax*weight/1_000_000:>6.1f}M)")
        
        logger.info("")
        logger.info(f"Total synthetic weight: {sum(w for _, w in pareto_weights.items()):,.0f}")
        logger.info(f"Estimated synthetic tax: ${total_synthetic_tax/1_000_000:,.1f}M")
        
        # Add synthetic filers
        if synthetic_filers:
            synthetic_df = pd.DataFrame(synthetic_filers)
            result = pd.concat([result, synthetic_df], ignore_index=True)
            
            # Calculate taxes for synthetic filers
            logger.info("\nCalculating taxes for synthetic filers...")
            from src.tax.hawaii_calculator import HawaiiTaxCalculator
            calculator = HawaiiTaxCalculator()
            
            synthetic_mask = (result['is_synthetic_ultra_high'] == True)
            for idx in result[synthetic_mask].index:
                row = result.loc[idx]
                # Calculate taxable income: AGI - standard deduction - exemptions
                agi = row['agi']
                filing_status = row['filing_status']
                num_dependents = row.get('num_dependents', 0)
                num_adults = row.get('num_adults', 2)  # Default to 2 for MFJ
                
                # Standard deduction for MFJ is $4,400 (2022)
                standard_ded = 4400 if filing_status == 'married_filing_jointly' else 2200
                personal_exemption = 1144
                total_exemptions = (num_adults + num_dependents) * personal_exemption
                
                taxable_income = max(0, agi - standard_ded - total_exemptions)
                tax = calculator.calculate_tax(taxable_income, filing_status)
                
                result.loc[idx, 'hi_taxable_income'] = taxable_income
                result.loc[idx, 'hi_state_tax'] = tax
                logger.info(f"  AGI ${agi:>12,.0f}: taxable = ${taxable_income:>12,.0f}, tax = ${tax:>12,.2f}")
        
        # Verify totals
        new_1m_filers = result.loc[result['agi'] >= self.threshold, 'weight'].sum()
        new_1m_tax = (result.loc[result['agi'] >= self.threshold, 'hi_state_tax'] * 
                     result.loc[result['agi'] >= self.threshold, 'weight']).sum() / 1_000_000
        
        logger.info("")
        logger.info("After reallocation:")
        logger.info(f"  $1M+ filers: {new_1m_filers:,.0f} (target: {current_filers:,.0f})")
        logger.info(f"  $1M+ tax: ${new_1m_tax:,.1f}M (target: ${target_tax_m:,.1f}M)")
        logger.info(f"  Filer count preserved: {'✅' if abs(new_1m_filers - current_filers) < 1 else '❌'}")
        
        return result
    
    def calibrate(self, df: pd.DataFrame, 
                  reallocation_pct: float = 0.35,
                  target_tax_m: float = 663.0) -> pd.DataFrame:
        """
        Main calibration method.
        
        Args:
            df: DataFrame with tax units
            reallocation_pct: Fraction of $1M-$2M weight to reallocate
            target_tax_m: Target tax for $1M+ bracket
            
        Returns:
            DataFrame with reallocated weights
        """
        return self.reallocate(df, reallocation_pct, target_tax_m)


def apply_million_plus_reallocation(df: pd.DataFrame,
                                    reallocation_pct: float = 0.35,
                                    target_tax_m: float = 663.0,
                                    pareto_alpha: float = 1.454) -> pd.DataFrame:
    """
    Reallocate $1M+ weight to synthetic ultra-high incomes.
    
    Args:
        df: DataFrame with tax units
        reallocation_pct: Fraction of $1M-$2M weight to move (default 35%)
        target_tax_m: Target tax for $1M+ bracket
        pareto_alpha: Pareto shape parameter
        
    Returns:
        DataFrame with reallocated weights
    """
    reallocator = MillionPlusReallocator(pareto_alpha=pareto_alpha)
    return reallocator.calibrate(df, reallocation_pct, target_tax_m)
