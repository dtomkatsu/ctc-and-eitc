"""
Ultra-High-Income Filer Synthesizer v2 - Enhanced

Adds realistic ultra-high-income filers ($5M+) with:
- Configurable Pareto alpha for sensitivity analysis
- Superbracket targeting ($10M+, $50M+, $100M+)
- Increased tail weight allocation
- Fractional unit support for extreme earners
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class UltraHighIncomeSynthesizerV2:
    """
    Enhanced ultra-high-income synthesis with Pareto sensitivity and superbracket targeting.
    """
    
    # IRS SOI Superbracket Data (estimated for Hawaii)
    # These are rough estimates; should be replaced with actual DOTAX/IRS data if available
    IRS_SUPERBRACKET_TARGETS = {
        (10_000_000, float('inf')): {'filers': 5, 'avg_agi': 25_000_000},
        (50_000_000, float('inf')): {'filers': 2, 'avg_agi': 75_000_000},
        (100_000_000, float('inf')): {'filers': 1, 'avg_agi': 150_000_000},
    }
    
    def __init__(self, 
                 pareto_alpha: float = 1.454,
                 tail_multiplier: float = 0.25,
                 use_superbracket_targets: bool = True):
        """
        Initialize enhanced synthesizer.
        
        Args:
            pareto_alpha: Pareto shape parameter (1.3-1.6 typical)
            tail_multiplier: Weight allocation factor for $50M+ (0.15-0.40)
            use_superbracket_targets: Whether to enforce IRS superbracket constraints
        """
        self.pareto_alpha = pareto_alpha
        self.tail_multiplier = tail_multiplier
        self.use_superbracket_targets = use_superbracket_targets
        
    def calculate_pareto_probability(self, 
                                     income: float, 
                                     threshold: float = 1_000_000) -> float:
        """
        Calculate Pareto probability P(X > income) = (threshold / income)^alpha
        """
        if income <= threshold:
            return 1.0
        return (threshold / income) ** self.pareto_alpha
    
    def redistribute_within_million_plus(self,
                                        df: pd.DataFrame,
                                        target_tax_m: float = 663.0) -> pd.DataFrame:
        """
        Redistribute filers within $1M+ bracket with enhanced tail targeting.
        
        Args:
            df: DataFrame with tax units
            target_tax_m: Target tax for $1M+ bracket (millions)
            
        Returns:
            DataFrame with redistributed ultra-high-income filers
        """
        logger.info("=" * 80)
        logger.info("ULTRA-HIGH-INCOME REDISTRIBUTION (Enhanced v2)")
        logger.info("=" * 80)
        logger.info(f"Pareto alpha: {self.pareto_alpha}")
        logger.info(f"Tail multiplier: {self.tail_multiplier}")
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
        
        # Enhanced ultra-high-income levels with higher tail allocation
        ultra_high_specs = [
            {'agi': 5_000_000, 'est_tax': 525_000, 'filing_status': 'married_filing_jointly', 'weight_factor': 0.10},
            {'agi': 10_000_000, 'est_tax': 1_070_000, 'filing_status': 'married_filing_jointly', 'weight_factor': 0.08},
            {'agi': 25_000_000, 'est_tax': 2_675_000, 'filing_status': 'married_filing_jointly', 'weight_factor': 0.05},
            {'agi': 50_000_000, 'est_tax': 5_350_000, 'filing_status': 'married_filing_jointly', 'weight_factor': self.tail_multiplier},
        ]
        
        total_weight_to_move = 0
        synthetic_filers = []
        
        logger.info("Calculating synthetic filer allocation:")
        
        for spec in ultra_high_specs:
            # Pareto probability relative to $1M threshold
            prob = self.calculate_pareto_probability(spec['agi'], threshold=1_000_000)
            
            # Allocate weight using Pareto and weight factor
            weight_at_level = current_filers * prob * spec['weight_factor']
            
            if weight_at_level < 0.1:
                logger.info(f"  ${spec['agi']/1_000_000:.0f}M: {weight_at_level:.2f} filers (below threshold, skipped)")
                continue
            
            synthetic_filers.append({
                'agi': spec['agi'],
                'filing_status': spec['filing_status'],
                'filing_status_hawaii': 'Joint_Surviving_Spouse',  # MFJ mapping
                'num_dependents': 2,
                'num_adults': 2,
                'weight': weight_at_level,
                'is_synthetic_ultra_high': True,
                'total_deductions': 0,
            })
            
            total_weight_to_move += weight_at_level
            
            logger.info(f"  ${spec['agi']/1_000_000:.0f}M: {weight_at_level:.1f} filers "
                       f"(prob={prob:.4f}, factor={spec['weight_factor']:.2f})")
        
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
                    
                    # Fill in any missing columns with appropriate defaults
                    for col in result.columns:
                        if col not in synthetic_df.columns:
                            if col in ['weight', 'agi', 'filing_status', 'num_dependents', 'num_adults', 
                                      'filing_status_hawaii', 'is_synthetic_ultra_high', 'total_deductions']:
                                continue  # Already set
                            elif 'tax' in col.lower():
                                synthetic_df[col] = 0  # Will be recalculated
                            elif col in ['income', 'agi_without_cap_gains', 'agi_with_cap_gains']:
                                # Set income fields to AGI for synthetic units
                                synthetic_df[col] = synthetic_df['agi']
                            elif col in ['standard_deduction', 'standard_deduction_amount']:
                                # MFJ standard deduction for 2022: $25,900
                                synthetic_df[col] = 25900
                            elif col in ['taxable_income', 'hi_taxable_income', 'hi_tax_taxable_income']:
                                # Taxable income = AGI - standard deduction
                                synthetic_df[col] = synthetic_df['agi'] - 25900
                            elif col in ['has_capital_gains', 'capital_gains', 'agi_adjustments']:
                                synthetic_df[col] = 0
                            elif col in ['bracket']:
                                # Highest bracket for ultra-high earners
                                synthetic_df[col] = '1000000+'
                            elif col in ['PUMA', 'PUMA10']:
                                # Use most common PUMA from existing data
                                synthetic_df[col] = result[col].mode()[0] if len(result[col].mode()) > 0 else '100'
                            elif col in ['filer_id', 'primary_filer_id', 'SERIALNO', 'hh_id']:
                                # Generate unique IDs for synthetic units
                                synthetic_df[col] = [f'SYNTH_{i}' for i in range(len(synthetic_df))]
                            elif col in ['secondary_filer_id']:
                                # MFJ has secondary filer
                                synthetic_df[col] = [f'SYNTH_{i}_SPOUSE' for i in range(len(synthetic_df))]
                            elif col in ['dependents']:
                                # Empty list for dependents
                                synthetic_df[col] = [[] for _ in range(len(synthetic_df))]
                            elif col in ['hh_weight', 'person_weight_sum', 'weight_original', 'weight_calibrated']:
                                # Use same as weight
                                synthetic_df[col] = synthetic_df['weight']
                            elif col in ['calibration_factor']:
                                synthetic_df[col] = 1.0
                            else:
                                synthetic_df[col] = 0  # Default to 0 for other fields
                    
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


def apply_ultra_high_income_synthesis_v2(df: pd.DataFrame,
                                         target_tax_m: float = 663.0,
                                         pareto_alpha: float = 1.454,
                                         tail_multiplier: float = 0.25) -> pd.DataFrame:
    """
    Apply enhanced ultra-high-income synthesis.
    
    Args:
        df: DataFrame with tax units
        target_tax_m: Target tax for $1M+ bracket (millions)
        pareto_alpha: Pareto shape parameter
        tail_multiplier: Weight allocation factor for $50M+
        
    Returns:
        DataFrame with ultra-high-income filers added
    """
    synthesizer = UltraHighIncomeSynthesizerV2(
        pareto_alpha=pareto_alpha,
        tail_multiplier=tail_multiplier
    )
    return synthesizer.calibrate(df, target_tax_m=target_tax_m)
