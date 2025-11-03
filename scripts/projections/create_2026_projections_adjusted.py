#!/usr/bin/env python3
"""
Create 2026 Revenue Projections with Adjustments

This script creates realistic 2026 projections by:
1. Using capital gains adjusted model as baseline
2. Applying historical CAGR growth rates
3. Capping MFS growth at 15% CAGR (not 55% historical anomaly)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_adjusted_model() -> pd.DataFrame:
    """Load the capital gains adjusted model."""
    project_root = Path(__file__).parent.parent.parent
    
    # Find latest adjusted file
    adjusted_files = sorted((project_root / 'data' / 'processed').glob('tax_units_refined_cap_gains_adjusted_*.parquet'))
    
    if not adjusted_files:
        logger.error("No adjusted model files found")
        raise FileNotFoundError("Run adjust_capital_gains_representation.py first")
    
    file_path = adjusted_files[-1]
    logger.info(f"Loading: {file_path.name}")
    
    return pd.read_parquet(file_path)


def load_historical_cagr() -> pd.DataFrame:
    """Load historical CAGR data."""
    project_root = Path(__file__).parent.parent.parent
    file_path = project_root / 'data' / 'processed' / 'dotax_historical' / 'dotax_historical_cagr.csv'
    
    return pd.read_csv(file_path)


def create_2026_projections(model_df: pd.DataFrame, cagr_df: pd.DataFrame, 
                            mfs_cap: float = 15.0) -> pd.DataFrame:
    """
    Create 2026 projections with MFS growth capped.
    
    Args:
        model_df: Current model (2022 baseline)
        cagr_df: Historical CAGR data
        mfs_cap: Maximum MFS growth rate (default 15% CAGR)
    
    Returns:
        DataFrame with projections
    """
    
    projections = []
    
    # Map filing status names
    status_map = {
        'single': 'Single',
        'married_filing_jointly': 'Married Filing Jointly',
        'married_filing_separately': 'Married Fil. Separately',
        'head_of_household': 'Head of Household',
    }
    
    for status in model_df['filing_status'].unique():
        status_data = model_df[model_df['filing_status'] == status]
        
        # Get current metrics (2022 baseline)
        current_returns = status_data['weight'].sum()
        current_agi = (status_data['agi'] * status_data['weight']).sum() / 1_000_000
        current_tax = (status_data['hi_state_tax'] * status_data['weight']).sum() / 1_000_000
        current_cap_gains = (status_data['capital_gains'] * status_data['weight']).sum() / 1_000_000
        
        # Get historical CAGR
        cagr_status = status_map.get(status, status)
        cagr_row = cagr_df[cagr_df['filing_status'] == cagr_status]
        
        if len(cagr_row) == 0:
            logger.warning(f"No CAGR data for {status}, using overall average")
            # Use overall averages
            returns_cagr = 0.3
            agi_cagr = 8.1
            tax_cagr = 11.1
        else:
            returns_cagr = cagr_row.iloc[0]['returns_cagr']
            agi_cagr = cagr_row.iloc[0]['agi_cagr']
            tax_cagr = cagr_row.iloc[0]['tax_cagr']
        
        # Apply MFS cap if needed
        if status == 'married_filing_separately':
            logger.info(f"MFS historical CAGR: Returns {returns_cagr:.1f}%, AGI {agi_cagr:.1f}%, Tax {tax_cagr:.1f}%")
            logger.info(f"Capping MFS growth at {mfs_cap}% CAGR")
            
            returns_cagr = min(returns_cagr, mfs_cap)
            agi_cagr = min(agi_cagr, mfs_cap)
            tax_cagr = min(tax_cagr, mfs_cap)
        
        # Project forward 4 years (2022 -> 2026)
        years = 4
        
        projected_returns = current_returns * ((1 + returns_cagr/100) ** years)
        projected_agi = current_agi * ((1 + agi_cagr/100) ** years)
        projected_tax = current_tax * ((1 + tax_cagr/100) ** years)
        projected_cap_gains = current_cap_gains * ((1 + agi_cagr/100) ** years)
        
        projections.append({
            'filing_status': status,
            '2022_returns': current_returns,
            '2026_returns': projected_returns,
            'returns_growth_pct': ((projected_returns / current_returns - 1) * 100),
            '2022_agi_millions': current_agi,
            '2026_agi_millions': projected_agi,
            'agi_growth_pct': ((projected_agi / current_agi - 1) * 100),
            '2022_tax_millions': current_tax,
            '2026_tax_millions': projected_tax,
            'tax_growth_pct': ((projected_tax / current_tax - 1) * 100),
            '2022_cap_gains_millions': current_cap_gains,
            '2026_cap_gains_millions': projected_cap_gains,
            'cagr_used_returns': returns_cagr,
            'cagr_used_agi': agi_cagr,
            'cagr_used_tax': tax_cagr,
        })
    
    return pd.DataFrame(projections)


def create_scenario_comparison(projections_df: pd.DataFrame) -> pd.DataFrame:
    """Create scenario comparison: uncapped vs capped MFS growth."""
    
    # Calculate totals for capped scenario
    capped_total = projections_df['2026_tax_millions'].sum()
    baseline_total = projections_df['2022_tax_millions'].sum()
    
    # Calculate uncapped scenario (for comparison)
    # MFS with historical 55% CAGR
    mfs_row = projections_df[projections_df['filing_status'] == 'married_filing_separately']
    
    if len(mfs_row) > 0:
        mfs_2022 = mfs_row.iloc[0]['2022_tax_millions']
        mfs_uncapped = mfs_2022 * ((1 + 0.551) ** 4)  # 55.1% CAGR
        mfs_capped = mfs_row.iloc[0]['2026_tax_millions']
        
        # Uncapped total
        uncapped_total = capped_total - mfs_capped + mfs_uncapped
    else:
        uncapped_total = capped_total
        mfs_uncapped = 0
        mfs_capped = 0
    
    scenarios = [
        {
            'scenario': 'Baseline (2022)',
            'total_revenue_millions': baseline_total,
            'mfs_revenue_millions': mfs_row.iloc[0]['2022_tax_millions'] if len(mfs_row) > 0 else 0,
            'notes': 'Refined model with capital gains adjusted'
        },
        {
            'scenario': 'Uncapped Growth (2026)',
            'total_revenue_millions': uncapped_total,
            'mfs_revenue_millions': mfs_uncapped,
            'notes': 'Using historical CAGR (MFS at 55%)'
        },
        {
            'scenario': 'Capped Growth (2026) - RECOMMENDED',
            'total_revenue_millions': capped_total,
            'mfs_revenue_millions': mfs_capped,
            'notes': 'MFS capped at 15% CAGR'
        },
    ]
    
    comparison_df = pd.DataFrame(scenarios)
    comparison_df['growth_from_baseline_pct'] = ((comparison_df['total_revenue_millions'] / baseline_total - 1) * 100)
    
    return comparison_df


def main():
    """Main projection workflow."""
    
    logger.info("="*100)
    logger.info("2026 REVENUE PROJECTIONS WITH ADJUSTMENTS")
    logger.info("="*100)
    
    # Load data
    model_df = load_adjusted_model()
    cagr_df = load_historical_cagr()
    
    # Create projections
    logger.info("\nCreating 2026 projections with MFS growth capped at 15% CAGR...")
    projections_df = create_2026_projections(model_df, cagr_df, mfs_cap=15.0)
    
    # Display results
    print("\n" + "="*100)
    print("2026 REVENUE PROJECTIONS (MFS CAPPED AT 15% CAGR)")
    print("="*100)
    print("\n### By Filing Status")
    print("-"*100)
    print(f"{'Filing Status':<30} {'2022 ($M)':>12} {'2026 ($M)':>12} {'Growth':>10} {'CAGR':>8}")
    print("-"*100)
    
    for _, row in projections_df.iterrows():
        print(f"{row['filing_status']:<30} ${row['2022_tax_millions']:>11,.1f} "
              f"${row['2026_tax_millions']:>11,.1f} "
              f"{row['tax_growth_pct']:>9.1f}% {row['cagr_used_tax']:>7.1f}%")
    
    print("-"*100)
    total_2022 = projections_df['2022_tax_millions'].sum()
    total_2026 = projections_df['2026_tax_millions'].sum()
    total_growth = ((total_2026 / total_2022 - 1) * 100)
    avg_cagr = total_growth / 4
    print(f"{'TOTAL':<30} ${total_2022:>11,.1f} ${total_2026:>11,.1f} {total_growth:>9.1f}% {avg_cagr:>7.1f}%")
    
    # Scenario comparison
    print("\n" + "="*100)
    print("SCENARIO COMPARISON")
    print("="*100)
    
    comparison_df = create_scenario_comparison(projections_df)
    
    print("\n### Revenue Scenarios")
    print("-"*100)
    print(f"{'Scenario':<40} {'Total Revenue ($M)':>20} {'MFS ($M)':>12} {'Growth':>10}")
    print("-"*100)
    
    for _, row in comparison_df.iterrows():
        marker = "✅" if "RECOMMENDED" in row['scenario'] else "  "
        print(f"{marker} {row['scenario']:<37} ${row['total_revenue_millions']:>19,.0f} "
              f"${row['mfs_revenue_millions']:>11,.0f} {row['growth_from_baseline_pct']:>9.1f}%")
    
    print("\n" + "-"*100)
    print("Notes:")
    for _, row in comparison_df.iterrows():
        print(f"• {row['scenario']}: {row['notes']}")
    
    # Capital gains projection
    print("\n" + "="*100)
    print("CAPITAL GAINS PROJECTIONS")
    print("="*100)
    
    total_cap_gains_2022 = projections_df['2022_cap_gains_millions'].sum()
    total_cap_gains_2026 = projections_df['2026_cap_gains_millions'].sum()
    cap_gains_growth = ((total_cap_gains_2026 / total_cap_gains_2022 - 1) * 100)
    
    print(f"\n2022 Capital Gains: ${total_cap_gains_2022:,.0f}M")
    print(f"2026 Capital Gains: ${total_cap_gains_2026:,.0f}M")
    print(f"Growth: {cap_gains_growth:.1f}%")
    
    total_agi_2022 = projections_df['2022_agi_millions'].sum()
    total_agi_2026 = projections_df['2026_agi_millions'].sum()
    
    print(f"\nCapital Gains as % of AGI:")
    print(f"  2022: {(total_cap_gains_2022/total_agi_2022)*100:.2f}%")
    print(f"  2026: {(total_cap_gains_2026/total_agi_2026)*100:.2f}%")
    
    # Save projections
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / 'data' / 'processed' / 'projections'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    projections_file = output_dir / f'2026_revenue_projections_adjusted_{timestamp}.csv'
    projections_df.to_csv(projections_file, index=False)
    
    comparison_file = output_dir / f'2026_scenario_comparison_{timestamp}.csv'
    comparison_df.to_csv(comparison_file, index=False)
    
    logger.info(f"\n✅ Projections saved to {output_dir}")
    
    # Summary
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    print("\n### Recommended 2026 Revenue Projection")
    print(f"Total Revenue: ${total_2026:,.0f}M")
    print(f"Growth from 2022: {total_growth:.1f}%")
    print(f"Average Annual Growth: {avg_cagr:.1f}%")
    
    print("\n### Key Adjustments Applied")
    print("✅ Capital gains adjusted from 1.65% to 3.31% of AGI")
    print("✅ MFS growth capped at 15% CAGR (from 55% historical)")
    print("✅ Realistic projections based on sustainable growth rates")
    
    return projections_df, comparison_df


if __name__ == "__main__":
    projections_df, comparison_df = main()
    print("\n✅ 2026 projections with adjustments complete!")
