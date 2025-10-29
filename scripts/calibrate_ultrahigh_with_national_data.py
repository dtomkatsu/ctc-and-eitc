#!/usr/bin/env python
"""
Calibrate Ultra-High-Income Distribution Using National IRS SOI Data

Uses national AGI distribution to estimate Hawaii's ultra-high-income tail
by applying proportional scaling based on Hawaii's share of national income.
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_national_ultrahigh_data():
    """Load national IRS SOI ultra-high-income distribution."""
    df = pd.read_csv('data/external/irs_soi_national_ultrahigh_2022.csv')
    return df


def estimate_hawaii_ultrahigh_distribution(national_df, hawaii_scaling_factor=0.004):
    """
    Estimate Hawaii's ultra-high-income distribution from national data.
    
    Args:
        national_df: National IRS SOI data
        hawaii_scaling_factor: Hawaii's share of national income/population
                               Default 0.004 (~0.4%) based on Hawaii being ~0.4% of US population
    
    Returns:
        DataFrame with estimated Hawaii distribution
    """
    hawaii_df = national_df.copy()
    
    # Scale returns and AGI by Hawaii's share
    hawaii_df['hawaii_returns'] = (hawaii_df['returns'] * hawaii_scaling_factor).round(1)
    hawaii_df['hawaii_agi_thousands'] = (hawaii_df['agi_thousands'] * hawaii_scaling_factor).round(0)
    
    # Average AGI stays the same (it's a ratio)
    hawaii_df['hawaii_avg_agi'] = hawaii_df['avg_agi']
    
    return hawaii_df


def calculate_pareto_parameters(df):
    """
    Calculate Pareto distribution parameters from the data.
    
    The Pareto distribution: P(X > x) = (x_min / x)^alpha
    
    We can estimate alpha from the ratio of filers in consecutive brackets.
    """
    logger.info("\nCalculating Pareto parameters from national data:")
    logger.info("="*80)
    
    # Use $1M+ brackets to estimate alpha
    brackets = df[df['min_agi'] >= 1_000_000].copy()
    
    alphas = []
    for i in range(len(brackets) - 1):
        x1 = brackets.iloc[i]['midpoint_agi']
        x2 = brackets.iloc[i+1]['midpoint_agi']
        n1 = brackets.iloc[i]['returns']
        n2 = brackets.iloc[i+1]['returns']
        
        if n1 > 0 and n2 > 0 and x2 > x1:
            # From Pareto: n2/n1 = (x1/x2)^alpha
            # So: alpha = log(n2/n1) / log(x1/x2)
            alpha = np.log(n2 / n1) / np.log(x1 / x2)
            alphas.append(alpha)
            
            logger.info(f"  {brackets.iloc[i]['bracket']:<35} → {brackets.iloc[i+1]['bracket']:<35}")
            logger.info(f"    Midpoints: ${x1:>12,.0f} → ${x2:>12,.0f}")
            logger.info(f"    Returns:   {n1:>12,.0f} → {n2:>12,.0f}")
            logger.info(f"    Alpha:     {alpha:>12.4f}")
            logger.info("")
    
    avg_alpha = np.mean(alphas) if alphas else 1.5
    logger.info(f"Average Pareto alpha: {avg_alpha:.4f}")
    logger.info(f"Recommended range: {avg_alpha - 0.1:.4f} - {avg_alpha + 0.1:.4f}")
    
    return avg_alpha, alphas


def generate_synthetic_distribution(hawaii_df, pareto_alpha, num_synthetic_levels=10):
    """
    Generate synthetic ultra-high-income distribution using Pareto parameters.
    
    Args:
        hawaii_df: Hawaii distribution data
        pareto_alpha: Pareto exponent
        num_synthetic_levels: Number of synthetic income levels to generate
    
    Returns:
        DataFrame with synthetic distribution
    """
    logger.info("\nGenerating synthetic ultra-high-income distribution:")
    logger.info("="*80)
    
    # Start from $10M+ bracket
    base_bracket = hawaii_df[hawaii_df['min_agi'] == 10_000_000].iloc[0]
    base_returns = base_bracket['hawaii_returns']
    base_agi = base_bracket['midpoint_agi']
    
    logger.info(f"Base bracket: $10M+")
    logger.info(f"  Returns: {base_returns:.1f}")
    logger.info(f"  Base AGI: ${base_agi:,.0f}")
    logger.info(f"  Pareto alpha: {pareto_alpha:.4f}")
    logger.info("")
    
    # Generate synthetic levels
    synthetic_levels = []
    agi_levels = [10_000_000, 25_000_000, 50_000_000, 100_000_000, 250_000_000, 500_000_000]
    
    logger.info(f"{'AGI Level':<20} {'Expected Returns':<20} {'Weight Factor':<15}")
    logger.info("-"*60)
    
    for agi in agi_levels:
        # Pareto formula: P(X > x) = (x_min / x)^alpha
        prob = (base_agi / agi) ** pareto_alpha
        expected_returns = base_returns * prob
        weight_factor = expected_returns / base_returns
        
        logger.info(f"${agi/1_000_000:>6.0f}M {expected_returns:>18.2f} {weight_factor:>14.4f}")
        
        synthetic_levels.append({
            'agi': agi,
            'expected_returns': expected_returns,
            'weight_factor': weight_factor,
            'pareto_prob': prob
        })
    
    return pd.DataFrame(synthetic_levels)


def compare_with_current_hawaii_data():
    """Compare national-based estimates with current Hawaii DOTAX data."""
    logger.info("\nComparing with Hawaii DOTAX data:")
    logger.info("="*80)
    
    # Hawaii DOTAX target for $1M+ bracket
    hawaii_1m_target_tax = 663.0  # Million dollars
    hawaii_1m_target_returns = 1824  # From DOTAX
    
    logger.info(f"Hawaii DOTAX $1M+ bracket:")
    logger.info(f"  Target returns: {hawaii_1m_target_returns:,.0f}")
    logger.info(f"  Target tax: ${hawaii_1m_target_tax:.1f}M")
    logger.info(f"  Implied avg tax: ${hawaii_1m_target_tax * 1_000_000 / hawaii_1m_target_returns:,.0f}")
    
    return hawaii_1m_target_returns


def main():
    """Main execution."""
    print("="*80)
    print("ULTRA-HIGH-INCOME CALIBRATION USING NATIONAL IRS SOI DATA")
    print("="*80)
    
    # Load national data
    logger.info("\n1. Loading national IRS SOI data...")
    national_df = load_national_ultrahigh_data()
    
    logger.info(f"\nNational ultra-high-income distribution:")
    logger.info(f"  Total $1M+ returns: {national_df['returns'].sum():,.0f}")
    logger.info(f"  Total $1M+ AGI: ${national_df['agi_thousands'].sum() * 1000:,.0f}")
    
    # Calculate Hawaii's scaling factor
    # Hawaii is approximately 0.4% of US population
    # But ultra-high-income earners may be concentrated differently
    # Use 0.3% as conservative estimate (Hawaii has fewer ultra-wealthy)
    hawaii_scaling_factor = 0.003
    
    logger.info(f"\n2. Estimating Hawaii distribution (scaling factor: {hawaii_scaling_factor:.1%})...")
    hawaii_df = estimate_hawaii_ultrahigh_distribution(national_df, hawaii_scaling_factor)
    
    logger.info(f"\nEstimated Hawaii ultra-high-income distribution:")
    logger.info(f"{'Bracket':<35} {'Returns':<12} {'AGI ($M)':<15} {'Avg AGI ($)':<20}")
    logger.info("-"*85)
    
    for _, row in hawaii_df.iterrows():
        logger.info(f"{row['bracket']:<35} {row['hawaii_returns']:>10.1f} "
                   f"${row['hawaii_agi_thousands']/1000:>12.1f} "
                   f"${row['hawaii_avg_agi']:>18,.0f}")
    
    total_hawaii_1m = hawaii_df['hawaii_returns'].sum()
    logger.info(f"\nTotal estimated $1M+ returns: {total_hawaii_1m:.0f}")
    
    # Calculate Pareto parameters
    logger.info(f"\n3. Calculating Pareto parameters...")
    avg_alpha, alphas = calculate_pareto_parameters(national_df)
    
    # Generate synthetic distribution
    logger.info(f"\n4. Generating synthetic distribution for Hawaii...")
    synthetic_df = generate_synthetic_distribution(hawaii_df, avg_alpha)
    
    # Compare with current Hawaii data
    logger.info(f"\n5. Comparing with current Hawaii DOTAX data...")
    hawaii_target_returns = compare_with_current_hawaii_data()
    
    logger.info(f"\nComparison:")
    logger.info(f"  National-based estimate: {total_hawaii_1m:.0f} returns")
    logger.info(f"  DOTAX target: {hawaii_target_returns:,.0f} returns")
    logger.info(f"  Ratio: {total_hawaii_1m / hawaii_target_returns:.2f}x")
    
    # Save results
    output_file = 'data/analysis/hawaii_ultrahigh_national_calibration.csv'
    hawaii_df.to_csv(output_file, index=False)
    logger.info(f"\n✅ Saved Hawaii estimates to: {output_file}")
    
    synthetic_file = 'data/analysis/hawaii_ultrahigh_synthetic_distribution.csv'
    synthetic_df.to_csv(synthetic_file, index=False)
    logger.info(f"✅ Saved synthetic distribution to: {synthetic_file}")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR ULTRA-HIGH-INCOME SYNTHESIZER")
    print("="*80)
    print(f"\n1. **Pareto Alpha**: Use {avg_alpha:.4f} (range: {avg_alpha-0.1:.4f} - {avg_alpha+0.1:.4f})")
    print(f"   Current value: 1.3 (from sensitivity analysis)")
    print(f"   National data suggests: {avg_alpha:.4f}")
    
    print(f"\n2. **Scaling Factor**: Adjust Hawaii scaling from 0.3% to match DOTAX")
    print(f"   Current DOTAX target: {hawaii_target_returns:,.0f} returns")
    print(f"   National-based estimate: {total_hawaii_1m:.0f} returns")
    print(f"   Suggested scaling: {hawaii_target_returns / national_df['returns'].sum():.4%}")
    
    print(f"\n3. **Synthetic Levels**: Use the following weight factors:")
    for _, row in synthetic_df.iterrows():
        print(f"   ${row['agi']/1_000_000:>6.0f}M: weight_factor={row['weight_factor']:.4f} "
              f"(~{row['expected_returns']:.1f} returns)")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
