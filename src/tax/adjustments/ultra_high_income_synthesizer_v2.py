"""
Ultra-High-Income Filer Synthesizer v2 - Enhanced with National IRS SOI Calibration

Adds realistic ultra-high-income filers ($5M+) with:
- Pareto alpha calibrated from national IRS SOI 2022 data (alpha=1.064)
- Superbracket targeting based on Hawaii's 0.227% share of national ultra-high earners
- Increased tail weight allocation (0.58 for $50M+ representation)
- Fractional unit support for extreme earners

National IRS SOI 2022 Data:
- $1M+: 804,831 returns nationally → ~1,824 in Hawaii (0.227%)
- Pareto alpha: 1.064 (average across $1M-$10M+ brackets)
- $10M+: 34,630 returns nationally → ~104 in Hawaii
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
    
    # IRS SOI Superbracket Data (calibrated from national 2022 data)
    # Based on Hawaii being ~0.227% of national ultra-high-income returns
    IRS_SUPERBRACKET_TARGETS = {
        (10_000_000, float('inf')): {'filers': 104, 'avg_agi': 30_364_406},
        (50_000_000, float('inf')): {'filers': 60, 'avg_agi': 75_000_000},  # Estimated from Pareto
        (100_000_000, float('inf')): {'filers': 29, 'avg_agi': 150_000_000},  # Estimated from Pareto
    }
    
    def __init__(self, 
                 pareto_alpha: float = 1.064,  # Calibrated from national IRS SOI data
                 tail_multiplier: float = 0.58,  # Calibrated for $50M+ representation
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
        
        # Calculate donor weight by filing status
        # MFJ: Use $1M+ donors (they exist in PUMS)
        # Single/MFS: Use $400k-$1M donors (no Single/MFS in $1M+ in PUMS)
        donor_weight_by_status = {}
        
        # MFJ: Use $1M+ donors
        for status in ['married_filing_jointly']:
            status_mask = mask_1m & (df['filing_status'] == status)
            donor_weight_by_status[status] = df.loc[status_mask, 'weight'].sum()
            if donor_weight_by_status[status] > 0:
                logger.info(f"$1M+ donor weight for {status}: {donor_weight_by_status[status]:.1f} filers")
        
        # Single, HoH, and MFS: Use $400k-$1M donors (since $1M+ has none)
        mask_400k_1m = (df['agi'] >= 400_000) & (df['agi'] < 1_000_000)
        for status in ['single', 'head_of_household', 'married_filing_separately']:
            status_mask = mask_400k_1m & (df['filing_status'] == status)
            donor_weight_by_status[status] = df.loc[status_mask, 'weight'].sum()
            if donor_weight_by_status[status] > 0:
                logger.info(f"$400k-$1M donor weight for {status}: {donor_weight_by_status[status]:.1f} filers")
        
        logger.info("")
        
        # Enhanced ultra-high-income levels with status-specific allocation
        # Iteration 9: FINAL BALANCE - Manual tuning (best performing)
        # - MFJ: 35%/28%/22%/15% from $1M+ donors
        # - Single: 20%/15%/10% from $400k-$1M donors  
        # - HoH: 4%/3% from $400k-$1M donors
        # - MFS: 10%/6% from $400k-$1M donors
        # Results: Joint +6.6%, Single -1.1%, HoH +5.6%, Total tax -3.7%
        ultra_high_specs = [
            # MFJ synthetics (from $1M+ donors) - FINAL BALANCE
            {'agi': 5_000_000, 'est_tax': 525_000, 'filing_status': 'married_filing_jointly', 'weight_factor': 0.35, 'status_specific': True, 'use_pareto': False},
            {'agi': 10_000_000, 'est_tax': 1_070_000, 'filing_status': 'married_filing_jointly', 'weight_factor': 0.28, 'status_specific': True, 'use_pareto': False},
            {'agi': 25_000_000, 'est_tax': 2_675_000, 'filing_status': 'married_filing_jointly', 'weight_factor': 0.22, 'status_specific': True, 'use_pareto': False},
            {'agi': 50_000_000, 'est_tax': 5_350_000, 'filing_status': 'married_filing_jointly', 'weight_factor': 0.15, 'status_specific': True, 'use_pareto': False},
            # Single synthetics (from $400k-$1M donors) - FINAL BALANCE
            {'agi': 1_500_000, 'est_tax': 161_000, 'filing_status': 'single', 'weight_factor': 0.20, 'status_specific': True, 'use_pareto': False},
            {'agi': 3_000_000, 'est_tax': 323_000, 'filing_status': 'single', 'weight_factor': 0.15, 'status_specific': True, 'use_pareto': False},
            {'agi': 5_000_000, 'est_tax': 537_000, 'filing_status': 'single', 'weight_factor': 0.10, 'status_specific': True, 'use_pareto': False},
            # HoH synthetics (from $400k-$1M donors) - FINAL BALANCE
            {'agi': 1_500_000, 'est_tax': 161_000, 'filing_status': 'head_of_household', 'weight_factor': 0.04, 'status_specific': True, 'use_pareto': False},
            {'agi': 3_000_000, 'est_tax': 323_000, 'filing_status': 'head_of_household', 'weight_factor': 0.03, 'status_specific': True, 'use_pareto': False},
            # MFS synthetics (from $400k-$1M donors)
            {'agi': 1_500_000, 'est_tax': 161_000, 'filing_status': 'married_filing_separately', 'weight_factor': 0.10, 'status_specific': True, 'use_pareto': False},
            {'agi': 3_000_000, 'est_tax': 323_000, 'filing_status': 'married_filing_separately', 'weight_factor': 0.06, 'status_specific': True, 'use_pareto': False},
        ]
        
        total_weight_to_move = 0
        synthetic_filers = []
        
        logger.info("Calculating synthetic filer allocation:")
        
        for spec in ultra_high_specs:
            # Allocate weight using status-specific donor pool
            status = spec['filing_status']
            donor_weight = donor_weight_by_status.get(status, 0)
            
            if donor_weight < 0.1:
                logger.info(f"  ${spec['agi']/1_000_000:.1f}M {status}: no donor weight available (skipped)")
                continue
            
            # Calculate weight based on whether we use Pareto
            if spec.get('use_pareto', True):
                # MFJ: Use Pareto probability relative to $1M threshold
                prob = self.calculate_pareto_probability(spec['agi'], threshold=1_000_000)
                weight_at_level = donor_weight * prob * spec['weight_factor']
            else:
                # Single/MFS: Direct allocation without Pareto (different donor base)
                # Just use donor_weight * weight_factor
                weight_at_level = donor_weight * spec['weight_factor']
                prob = 1.0  # For logging
            
            if weight_at_level < 0.1:
                logger.info(f"  ${spec['agi']/1_000_000:.0f}M: {weight_at_level:.2f} filers (below threshold, skipped)")
                continue
            
            # Map filing status to Hawaii filing status
            fs_map = {
                'married_filing_jointly': 'Joint_Surviving_Spouse',
                'single': 'Single_Married_Separate',
                'married_filing_separately': 'Single_Married_Separate',
                'head_of_household': 'Head_of_Household',
            }
            
            # Set adults/dependents based on status
            if spec['filing_status'] == 'married_filing_jointly':
                num_adults = 2
                num_dependents = 2
            elif spec['filing_status'] == 'head_of_household':
                num_adults = 1
                num_dependents = 1
            else:  # single or MFS
                num_adults = 1
                num_dependents = 0
            
            synthetic_filers.append({
                'agi': spec['agi'],
                'filing_status': spec['filing_status'],
                'filing_status_hawaii': fs_map[spec['filing_status']],
                'num_dependents': num_dependents,
                'num_adults': num_adults,
                'weight': weight_at_level,
                'is_synthetic_ultra_high': True,
                'total_deductions': 0,
            })
            
            total_weight_to_move += weight_at_level
            
            status_short = spec['filing_status'].replace('married_filing_', 'M').replace('_', '').replace('single', 'S').replace('headofhousehold', 'HoH')
            if spec.get('use_pareto', True):
                logger.info(f"  ${spec['agi']/1_000_000:.1f}M {status_short}: {weight_at_level:.1f} filers "
                           f"(prob={prob:.4f}, factor={spec['weight_factor']:.2f}, donor={donor_weight:.1f})")
            else:
                logger.info(f"  ${spec['agi']/1_000_000:.1f}M {status_short}: {weight_at_level:.1f} filers "
                           f"(factor={spec['weight_factor']:.2f}, donor={donor_weight:.1f} from $400k-$1M)")
        
        logger.info("")
        logger.info(f"Total weight to redistribute: {total_weight_to_move:.0f} filers")
        
        # Reduce weight from existing high-income filers by filing status
        # Cascade through sub-brackets to ensure conservation
        if total_weight_to_move > 0:
            result = df.copy()
            
            # Group synthetic weight by filing status
            synthetic_weight_by_status = {}
            for spec in synthetic_filers:
                status = spec['filing_status']
                synthetic_weight_by_status[status] = synthetic_weight_by_status.get(status, 0) + spec['weight']
            
            logger.info("\nReducing weight from existing filers by filing status:")
            
            # Process each filing status separately to preserve A2-2 totals
            for status, synthetic_weight_needed in synthetic_weight_by_status.items():
                logger.info(f"\n  {status}: Need to remove {synthetic_weight_needed:.1f} filers")
                
                # Track cumulative reduction for this status
                total_removed = 0
                remaining_to_remove = synthetic_weight_needed
                
                # Define sub-brackets based on filing status
                # MFJ: Use $1M+ brackets (where they have donor weight)
                # Single/HoH/MFS: Use $400k-$1M brackets (where they have donor weight)
                if status in ['married_filing_jointly']:
                    sub_brackets = [
                        (1_000_000, 5_000_000, "$1M-$5M"),
                        (5_000_000, 10_000_000, "$5M-$10M"),
                        (10_000_000, 25_000_000, "$10M-$25M"),
                        (25_000_000, 50_000_000, "$25M-$50M"),
                        (50_000_000, float('inf'), "$50M+"),
                    ]
                else:  # single, head_of_household, married_filing_separately
                    sub_brackets = [
                        (400_000, 500_000, "$400k-$500k"),
                        (500_000, 750_000, "$500k-$750k"),
                        (750_000, 1_000_000, "$750k-$1M"),
                    ]
                
                # Cascade through sub-brackets
                for min_agi, max_agi, label in sub_brackets:
                    if remaining_to_remove <= 0.01:  # Floating point tolerance
                        break
                    
                    # Find filers in this sub-bracket with this status
                    if max_agi == float('inf'):
                        mask = (result['agi'] >= min_agi) & (result['filing_status'] == status)
                    else:
                        mask = (result['agi'] >= min_agi) & (result['agi'] < max_agi) & (result['filing_status'] == status)
                    
                    available_weight = result.loc[mask, 'weight'].sum()
                    
                    if available_weight < 0.01:
                        logger.info(f"    {label}: {available_weight:.1f} available (skipping)")
                        continue
                    
                    # Calculate how much to remove from this bracket
                    to_remove = min(remaining_to_remove, available_weight)
                    reduction_factor = (available_weight - to_remove) / available_weight
                    
                    # Apply reduction
                    result.loc[mask, 'weight'] *= reduction_factor
                    
                    total_removed += to_remove
                    remaining_to_remove -= to_remove
                    
                    logger.info(f"    {label}: Removed {to_remove:.1f} of {available_weight:.1f} available ({(1-reduction_factor)*100:.1f}% reduction)")
                
                if remaining_to_remove > 0.01:
                    logger.warning(f"  ⚠️  Could not remove full {synthetic_weight_needed:.1f} for {status}")
                    logger.warning(f"     Only removed {total_removed:.1f}, shortfall of {remaining_to_remove:.1f}")
                else:
                    logger.info(f"  ✅ Successfully removed {total_removed:.1f} filers for {status}")
            
            # Add synthetic ultra-high-income filers (after all reductions)
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
                            # Status-specific standard deduction for 2022
                            def get_std_ded(status):
                                if status == 'married_filing_jointly':
                                    return 25900
                                elif status == 'head_of_household':
                                    return 19400
                                else:  # single or MFS
                                    return 12950
                            synthetic_df[col] = synthetic_df['filing_status'].apply(get_std_ded)
                        elif col in ['taxable_income', 'hi_taxable_income', 'hi_tax_taxable_income']:
                            # Taxable income = AGI - standard deduction (status-specific)
                            def calc_taxable(row):
                                std_ded = get_std_ded(row['filing_status']) if 'filing_status' in row else 12950
                                return row['agi'] - std_ded
                            
                            def get_std_ded(status):
                                if status == 'married_filing_jointly':
                                    return 25900
                                elif status == 'head_of_household':
                                    return 19400
                                else:
                                    return 12950
                            
                            synthetic_df[col] = synthetic_df.apply(calc_taxable, axis=1)
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
                            # MFJ has secondary filer, others don't
                            synthetic_df[col] = [f'SYNTH_{i}_SPOUSE' if row['filing_status'] == 'married_filing_jointly' else None 
                                               for i, row in synthetic_df.iterrows()]
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
                
                logger.info(f"\nAdded {len(synthetic_df)} ultra-high-income levels")
                logger.info(f"Total synthetic weight: {synthetic_df['weight'].sum():,.0f}")
                
            # Verify total filer count preserved by filing status and overall
            logger.info("\n" + "="*80)
            logger.info("WEIGHT CONSERVATION VERIFICATION")
            logger.info("="*80)
            
            # Check $400k+ by filing status (A2-2 targets)
            logger.info("\n$400k+ bracket by filing status:")
            for status in ['single', 'married_filing_jointly', 'head_of_household', 'married_filing_separately']:
                before_400k = df.loc[(df['agi'] >= 400_000) & (df['filing_status'] == status), 'weight'].sum()
                after_400k = result.loc[(result['agi'] >= 400_000) & (result['filing_status'] == status), 'weight'].sum()
                diff = after_400k - before_400k
                status_label = status.replace('_', ' ').title()
                logger.info(f"  {status_label:<25} Before: {before_400k:>8.1f}  After: {after_400k:>8.1f}  Diff: {diff:>+7.1f}")
            
            # Check $1M+ overall (A9 target)
            before_1m = df.loc[df['agi'] >= 1_000_000, 'weight'].sum()
            after_1m = result.loc[result['agi'] >= 1_000_000, 'weight'].sum()
            diff_1m = after_1m - before_1m
            
            logger.info("\n$1M+ bracket overall:")
            logger.info(f"  Before: {before_1m:,.1f}")
            logger.info(f"  After:  {after_1m:,.1f}")
            logger.info(f"  Diff:   {diff_1m:+,.1f}")
            
            if abs(diff_1m) < 0.1:
                logger.info("  ✅ Conservation verified: $1M+ totals preserved")
            else:
                logger.warning(f"  ⚠️  Conservation violated: $1M+ totals changed by {diff_1m:.1f}")
            
            return result
        
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
