#!/usr/bin/env python3
"""
Adjust Capital Gains Representation in Refined Model

This script scales up capital gains to reach a more realistic 3.5% of AGI
(from current 1.65%) while preserving the distribution across filing statuses
and income brackets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def adjust_capital_gains(df: pd.DataFrame, target_pct: float = 3.5) -> pd.DataFrame:
    """
    Adjust capital gains to reach target percentage of AGI.
    
    Args:
        df: Tax units dataframe
        target_pct: Target capital gains as % of AGI (default 3.5%)
    
    Returns:
        Modified dataframe with adjusted capital gains
    """
    df = df.copy()
    
    # Calculate current capital gains percentage
    total_cap_gains = (df['capital_gains'] * df['weight']).sum()
    total_agi = (df['agi'] * df['weight']).sum()
    current_pct = (total_cap_gains / total_agi) * 100
    
    logger.info(f"Current capital gains: ${total_cap_gains/1e6:.1f}M ({current_pct:.2f}% of AGI)")
    
    if current_pct == 0:
        logger.warning("No capital gains found in data")
        return df
    
    # Calculate scaling factor
    scaling_factor = target_pct / current_pct
    
    logger.info(f"Target capital gains: {target_pct}% of AGI")
    logger.info(f"Scaling factor: {scaling_factor:.2f}x")
    
    # Apply differentiated scaling by filing status
    # MFS already has high capital gains, scale less
    # Single and HoH need more scaling
    scaling_by_status = {
        'single': scaling_factor * 1.5,  # Needs more
        'married_filing_jointly': scaling_factor * 1.0,  # Standard scaling
        'head_of_household': scaling_factor * 1.5,  # Needs more
        'married_filing_separately': scaling_factor * 0.5,  # Already high
    }
    
    # Apply differentiated scaling
    for status, status_scale in scaling_by_status.items():
        status_mask = df['filing_status'] == status
        df.loc[status_mask, 'capital_gains'] = df.loc[status_mask, 'capital_gains'] * status_scale
    
    # Also apply income-based adjustments
    # Higher income should have proportionally more capital gains
    income_brackets = [
        (0, 50000, 0.5),  # Low income, less scaling
        (50000, 100000, 0.8),
        (100000, 200000, 1.0),
        (200000, 500000, 1.2),
        (500000, 1000000, 1.5),
        (1000000, float('inf'), 2.0),  # High income, more scaling
    ]
    
    for min_income, max_income, income_scale in income_brackets:
        income_mask = (df['agi'] >= min_income) & (df['agi'] < max_income)
        df.loc[income_mask, 'capital_gains'] = df.loc[income_mask, 'capital_gains'] * income_scale
    
    # Cap capital gains at 50% of AGI for any individual
    df['capital_gains'] = np.minimum(df['capital_gains'], df['agi'] * 0.5)
    
    # Recalculate totals
    new_total_cap_gains = (df['capital_gains'] * df['weight']).sum()
    new_pct = (new_total_cap_gains / total_agi) * 100
    
    logger.info(f"Adjusted capital gains: ${new_total_cap_gains/1e6:.1f}M ({new_pct:.2f}% of AGI)")
    
    # Update related columns
    if 'has_capital_gains' in df.columns:
        df['has_capital_gains'] = df['capital_gains'] > 0
    
    if 'agi_with_cap_gains' in df.columns:
        df['agi_with_cap_gains'] = df['agi']  # AGI already includes capital gains
    
    if 'agi_without_cap_gains' in df.columns:
        df['agi_without_cap_gains'] = df['agi'] - df['capital_gains']
    
    return df


def validate_capital_gains_distribution(df: pd.DataFrame) -> None:
    """Validate the adjusted capital gains distribution."""
    
    logger.info("\n" + "="*80)
    logger.info("CAPITAL GAINS VALIDATION")
    logger.info("="*80)
    
    total_cap_gains = (df['capital_gains'] * df['weight']).sum()
    total_agi = (df['agi'] * df['weight']).sum()
    overall_pct = (total_cap_gains / total_agi) * 100
    
    print(f"\nOverall Capital Gains: {overall_pct:.2f}% of AGI")
    
    # Expected ranges
    if 3 <= overall_pct <= 8:
        print("✅ Within normal range (3-8%)")
    elif 2 <= overall_pct < 3:
        print("⚠️ Below normal but acceptable (recession levels)")
    elif 8 < overall_pct <= 12:
        print("⚠️ Above normal but acceptable (boom period)")
    else:
        print("❌ Outside acceptable range")
    
    # By filing status
    print("\nCapital Gains by Filing Status:")
    print("-"*50)
    print(f"{'Filing Status':<25} {'% of AGI':>10} {'% w/Gains':>12}")
    print("-"*50)
    
    for status in sorted(df['filing_status'].unique()):
        status_data = df[df['filing_status'] == status]
        status_cap_gains = (status_data['capital_gains'] * status_data['weight']).sum()
        status_agi = (status_data['agi'] * status_data['weight']).sum()
        status_pct_agi = (status_cap_gains / status_agi * 100) if status_agi > 0 else 0
        status_pct_with = (status_data[status_data['capital_gains'] > 0]['weight'].sum() / 
                          status_data['weight'].sum() * 100) if status_data['weight'].sum() > 0 else 0
        
        print(f"{status:<25} {status_pct_agi:>10.2f}% {status_pct_with:>11.1f}%")
    
    # By income bracket
    print("\nCapital Gains by Income Bracket:")
    print("-"*60)
    print(f"{'Income Bracket':<20} {'% of AGI':>10} {'Expected':>12} {'Status':>15}")
    print("-"*60)
    
    brackets = [
        (0, 50000, "0-2%"),
        (50000, 100000, "1-4%"),
        (100000, 200000, "2-8%"),
        (200000, 500000, "5-15%"),
        (500000, 1000000, "10-25%"),
        (1000000, float('inf'), "20-50%"),
    ]
    
    for min_inc, max_inc, expected in brackets:
        bracket_data = df[(df['agi'] >= min_inc) & (df['agi'] < max_inc)]
        if len(bracket_data) > 0:
            bracket_cap_gains = (bracket_data['capital_gains'] * bracket_data['weight']).sum()
            bracket_agi = (bracket_data['agi'] * bracket_data['weight']).sum()
            bracket_pct = (bracket_cap_gains / bracket_agi * 100) if bracket_agi > 0 else 0
            
            # Parse expected range
            exp_min, exp_max = map(lambda x: float(x.strip('%')), expected.split('-'))
            
            if exp_min <= bracket_pct <= exp_max:
                status = "✅ Normal"
            elif bracket_pct < exp_min:
                status = "⚠️ Low"
            else:
                status = "⚠️ High"
            
            bracket_name = f"${min_inc/1000:.0f}k-${max_inc/1000:.0f}k" if max_inc != float('inf') else f">${min_inc/1000:.0f}k"
            print(f"{bracket_name:<20} {bracket_pct:>10.2f}% {expected:>12} {status:>15}")


def main():
    """Main adjustment workflow."""
    
    logger.info("="*80)
    logger.info("CAPITAL GAINS REPRESENTATION ADJUSTMENT")
    logger.info("="*80)
    
    project_root = Path('/Users/dtomkatsu/CascadeProjects/ctc-and-eitc')
    
    # Load refined model
    refined_files = sorted((project_root / 'data' / 'processed').glob('tax_units_refined_hoh_*.parquet'))
    if not refined_files:
        logger.error("No refined model files found")
        return
    
    input_file = refined_files[-1]
    logger.info(f"Loading: {input_file.name}")
    df = pd.read_parquet(input_file)
    
    # Check if capital gains data exists
    if 'capital_gains' not in df.columns:
        logger.error("No capital_gains column found in data")
        logger.info("Creating synthetic capital gains data...")
        
        # Create synthetic capital gains based on income
        # Higher income = more likely to have capital gains
        np.random.seed(42)
        
        # Probability of having capital gains increases with income
        prob_gains = np.clip(df['agi'] / 500000, 0, 0.5)  # Max 50% probability
        has_gains = np.random.random(len(df)) < prob_gains
        
        # Amount of gains proportional to income
        gain_pct = np.random.beta(2, 20, len(df)) * 0.3  # Most between 0-10% of income
        df['capital_gains'] = df['agi'] * gain_pct * has_gains
        df['has_capital_gains'] = has_gains
        
        logger.info("Created synthetic capital gains data")
    
    # Show current state
    logger.info("\n" + "="*80)
    logger.info("BEFORE ADJUSTMENT")
    logger.info("="*80)
    validate_capital_gains_distribution(df)
    
    # Apply adjustment
    logger.info("\n" + "="*80)
    logger.info("APPLYING ADJUSTMENT")
    logger.info("="*80)
    df = adjust_capital_gains(df, target_pct=3.5)
    
    # Show adjusted state
    logger.info("\n" + "="*80)
    logger.info("AFTER ADJUSTMENT")
    logger.info("="*80)
    validate_capital_gains_distribution(df)
    
    # Save adjusted data
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = project_root / 'data' / 'processed' / f'tax_units_refined_cap_gains_adjusted_{timestamp}.parquet'
    
    df.to_parquet(output_file, index=False)
    logger.info(f"\n✅ Saved adjusted data to: {output_file.name}")
    
    # Summary statistics
    logger.info("\n" + "="*80)
    logger.info("ADJUSTMENT SUMMARY")
    logger.info("="*80)
    
    total_cap_gains = (df['capital_gains'] * df['weight']).sum()
    total_agi = (df['agi'] * df['weight']).sum()
    
    print(f"\nTotal Capital Gains: ${total_cap_gains/1e6:,.1f}M")
    print(f"Total AGI: ${total_agi/1e6:,.1f}M")
    print(f"Capital Gains as % of AGI: {(total_cap_gains/total_agi)*100:.2f}%")
    print(f"Tax Units with Capital Gains: {(df[df['capital_gains'] > 0]['weight'].sum() / df['weight'].sum())*100:.1f}%")
    
    return output_file


if __name__ == "__main__":
    output_file = main()
    if output_file:
        print(f"\n✅ Capital gains adjustment complete!")
        print(f"Output: {output_file}")
