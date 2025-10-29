"""
Ultra-High-Income Filer Synthesizer

Adds realistic ultra-high-income filers ($5M+) to fill the gap in the $1M+ bracket
while preserving the existing filer count and effective rate calibration.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class UltraHighIncomeSynthesizer:
    """
    Synthesize ultra-high-income filers to match DOTax tax totals.
    
    Strategy:
    - DOTax $1M+ bracket: 1,824 filers generate $663M tax
    - Our model: 1,824 filers generate $215M tax
    - Gap: $448M
    - Solution: Reallocate some of the 1,824 filer weight to ultra-high incomes
    """
    
    def __init__(self):
        """Initialize synthesizer."""
        pass
    
    def calculate_needed_ultra_high_weight(self,
                                          current_tax: float,
                                          target_tax: float,
                                          avg_ultra_high_tax: float) -> float:
        """
        Calculate how much filer weight needs to be at ultra-high incomes.
        
        Args:
            current_tax: Current total tax from $1M+ bracket
            target_tax: Target total tax for $1M+ bracket
            avg_ultra_high_tax: Average tax per ultra-high-income filer
            
        Returns:
            Weight (number of filers) needed at ultra-high incomes
        """
        tax_gap = target_tax - current_tax
        
        if tax_gap <= 0:
            return 0
        
        # How many ultra-high-income filers needed to fill gap
        needed_weight = tax_gap / avg_ultra_high_tax
        
        return needed_weight
    
    def redistribute_within_million_plus(self,
                                        df: pd.DataFrame,
                                        target_tax_m: float = 663.0) -> pd.DataFrame:
        """
        Redistribute filers within $1M+ bracket to match tax target.
        
        Strategy:
        - Keep total filer count at 1,824 (already calibrated)
        - Move some weight from $1M-$2M range to $5M+ range
        - Use realistic income levels: $5M, $10M, $25M, $50M
        
        Args:
            df: DataFrame with tax units
            target_tax_m: Target tax for $1M+ bracket (millions)
            
        Returns:
            DataFrame with redistributed ultra-high-income filers
        """
        logger.info("=" * 80)
        logger.info("ULTRA-HIGH-INCOME REDISTRIBUTION")
        logger.info("=" * 80)
        logger.info("")
        
        # Current $1M+ situation
        mask_1m = df['agi'] >= 1_000_000
        
        if mask_1m.sum() == 0:
            logger.warning("No filers in $1M+ bracket")
            return df
        
        current_filers = df.loc[mask_1m, 'weight'].sum()
        current_tax = (df.loc[mask_1m, 'hi_state_tax'] * df.loc[mask_1m, 'weight']).sum()
        current_tax_m = current_tax / 1_000_000
        
        logger.info(f"Current $1M+ bracket:")
        logger.info(f"  Filers: {current_filers:,.0f}")
        logger.info(f"  Tax: ${current_tax_m:,.1f}M")
        logger.info(f"  Target: ${target_tax_m:,.1f}M")
        logger.info(f"  Gap: ${current_tax_m - target_tax_m:,.1f}M")
        logger.info("")
        
        tax_gap = target_tax_m * 1_000_000 - current_tax
        
        if tax_gap <= 0:
            logger.info("No gap to fill, skipping redistribution")
            return df
        
        # Ultra-high-income levels and their estimated taxes
        # Hawaii 11% top rate kicks in at ~$200k for MFJ
        # Ultra-high earners pay roughly 10.5-10.8% effective
        ultra_high_specs = [
            {'agi': 5_000_000, 'est_tax': 525_000, 'filing_status': 'married_filing_jointly'},
            {'agi': 10_000_000, 'est_tax': 1_070_000, 'filing_status': 'married_filing_jointly'},
            {'agi': 25_000_000, 'est_tax': 2_675_000, 'filing_status': 'married_filing_jointly'},
            {'agi': 50_000_000, 'est_tax': 5_350_000, 'filing_status': 'married_filing_jointly'},
        ]
        
        # Calculate weight needed at each level using Pareto distribution
        pareto_alpha = 1.454  # From earlier calibration
        
        total_weight_to_move = 0
        synthetic_filers = []
        
        for spec in ultra_high_specs:
            # Pareto probability relative to $1M threshold
            prob = (1_000_000 / spec['agi']) ** pareto_alpha
            
            # Allocate weight proportional to Pareto probability
            weight_at_level = current_filers * prob * 0.15  # Conservative factor
            
            if weight_at_level < 0.1:
                continue
            
            synthetic_filers.append({
                'agi': spec['agi'],
                'filing_status': spec['filing_status'],
                'num_dependents': 2,
                'num_adults': 2,
                'weight': weight_at_level,
                'is_synthetic_ultra_high': True,
            })
            
            total_weight_to_move += weight_at_level
            
            logger.info(f"  ${spec['agi']/1_000_000:.0f}M: {weight_at_level:.0f} filers "
                       f"(est. ${weight_at_level * spec['est_tax'] / 1_000_000:.1f}M tax)")
        
        logger.info("")
        logger.info(f"Total weight to redistribute: {total_weight_to_move:.0f} filers")
        
        # Reduce weight of lower $1M+ filers proportionally
        if total_weight_to_move > 0:
            # Find filers in $1M-$5M range to reduce
            mask_1m_to_5m = (df['agi'] >= 1_000_000) & (df['agi'] < 5_000_000)
            weight_1m_to_5m = df.loc[mask_1m_to_5m, 'weight'].sum()
            
            if weight_1m_to_5m > total_weight_to_move:
                # Reduce proportionally
                reduction_factor = (weight_1m_to_5m - total_weight_to_move) / weight_1m_to_5m
                
                result = df.copy()
                result.loc[mask_1m_to_5m, 'weight'] *= reduction_factor
                
                logger.info(f"Reduced $1M-$5M filer weight by {(1-reduction_factor)*100:.1f}%")
                
                # Add synthetic ultra-high-income filers
                if synthetic_filers:
                    synthetic_df = pd.DataFrame(synthetic_filers)
                    
                    # Fill in missing columns to match main DataFrame
                    synthetic_df['total_deductions'] = 0
                    synthetic_df['filing_status_hawaii'] = 'Joint_Surviving_Spouse'  # MFJ mapping
                    
                    # Fill any other missing columns with defaults
                    for col in result.columns:
                        if col not in synthetic_df.columns:
                            if col in ['weight', 'agi', 'filing_status', 'num_dependents', 'num_adults']:
                                continue  # Already set
                            elif 'tax' in col.lower():
                                synthetic_df[col] = 0  # Will be recalculated
                            else:
                                synthetic_df[col] = 0  # Default to 0
                    
                    result = pd.concat([result, synthetic_df], ignore_index=True)
                    
                    logger.info(f"Added {len(synthetic_df)} ultra-high-income levels")
                    logger.info(f"Total synthetic weight: {synthetic_df['weight'].sum():,.0f}")
                
                # Verify total filer count preserved
                new_total_1m = result.loc[result['agi'] >= 1_000_000, 'weight'].sum()
                logger.info("")
                logger.info(f"Final $1M+ filer count: {new_total_1m:,.0f} (target: {current_filers:,.0f})")
                
                return result
            else:
                logger.warning("Not enough $1M-$5M filers to redistribute")
                return df
        
        return df
    
    def calibrate(self, df: pd.DataFrame, target_tax_m: float = 663.0) -> pd.DataFrame:
        """
        Main calibration method.
        
        Args:
            df: DataFrame with tax units
            target_tax_m: Target tax for $1M+ bracket (millions)
            
        Returns:
            DataFrame with ultra-high-income filers added
        """
        return self.redistribute_within_million_plus(df, target_tax_m)


def apply_ultra_high_income_synthesis(df: pd.DataFrame,
                                      target_tax_m: float = 663.0) -> pd.DataFrame:
    """
    Add ultra-high-income filers to match $1M+ bracket tax target.
    
    Args:
        df: DataFrame with tax units
        target_tax_m: Target tax for $1M+ bracket (millions)
        
    Returns:
        DataFrame with ultra-high-income filers
    """
    synthesizer = UltraHighIncomeSynthesizer()
    return synthesizer.calibrate(df, target_tax_m=target_tax_m)
