#!/usr/bin/env python3
"""
Demo: BLS OES Wage Growth Options

This script demonstrates the available BLS OES data and wage growth calculation
options for Phase 1 of growth forecasting (2022 → 2024).

Shows:
- Available BLS OES data files
- Occupation-level growth rates
- Industry-level growth rates (if available)
- Different adjustment strategies
- Comparison of methods
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
import logging

from src.data.bls_oes_loader import BLSOESLoader
from src.tax.calibration.wage_growth_adjustment import WageGrowthAdjuster

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def check_available_data():
    """Check what BLS OES data is available."""
    logger.info("=" * 70)
    logger.info("AVAILABLE BLS OES DATA")
    logger.info("=" * 70)
    
    data_dir = Path("data/external/bls_oes")
    
    if not data_dir.exists():
        logger.warning(f"BLS OES data directory not found: {data_dir}")
        return []
    
    # Check for files
    files = list(data_dir.glob("*.xlsx")) + list(data_dir.glob("*.parquet"))
    
    logger.info(f"\nFound {len(files)} BLS OES data files:")
    
    available_years = []
    for file in sorted(files):
        size_mb = file.stat().st_size / (1024 * 1024)
        logger.info(f"  • {file.name:40s} ({size_mb:>6.1f} MB)")
        
        # Extract year from filename
        if 'M2022' in file.name or '2022' in file.name:
            available_years.append(2022)
        elif 'M2023' in file.name or '2023' in file.name:
            available_years.append(2023)
        elif 'M2024' in file.name or '2024' in file.name:
            available_years.append(2024)
    
    available_years = sorted(set(available_years))
    
    logger.info(f"\nAvailable years: {available_years}")
    
    return available_years


def show_occupation_growth_rates(adjuster: WageGrowthAdjuster,
                                 start_year: int = 2022,
                                 end_year: int = 2024):
    """Show occupation-specific growth rates."""
    logger.info("\n" + "=" * 70)
    logger.info(f"OCCUPATION-SPECIFIC GROWTH RATES ({start_year} → {end_year})")
    logger.info("=" * 70)
    
    # Calculate growth rates
    growth_df = adjuster.calculate_growth_rates_phase1(start_year, end_year)
    
    # Show top 10 by employment
    logger.info("\n📊 Top 10 Occupations by Employment:")
    logger.info(f"{'Occupation':<50} {'Employment':>12} {'Growth':>10}")
    logger.info("-" * 75)
    
    top10 = growth_df.nlargest(10, 'employment')
    for _, row in top10.iterrows():
        occ_title = row['occupation_title'][:48]
        employment = row['employment']
        growth = row['total_growth_rate']
        logger.info(f"{occ_title:<50} {employment:>12,} {growth:>9.1%}")
    
    # Show fastest growing
    logger.info("\n📈 Top 10 Fastest Growing Occupations:")
    logger.info(f"{'Occupation':<50} {'Employment':>12} {'Growth':>10}")
    logger.info("-" * 75)
    
    fastest = growth_df.nlargest(10, 'total_growth_rate')
    for _, row in fastest.iterrows():
        occ_title = row['occupation_title'][:48]
        employment = row['employment']
        growth = row['total_growth_rate']
        logger.info(f"{occ_title:<50} {employment:>12,} {growth:>9.1%}")
    
    # Show slowest growing
    logger.info("\n📉 Top 10 Slowest Growing Occupations:")
    logger.info(f"{'Occupation':<50} {'Employment':>12} {'Growth':>10}")
    logger.info("-" * 75)
    
    slowest = growth_df.nsmallest(10, 'total_growth_rate')
    for _, row in slowest.iterrows():
        occ_title = row['occupation_title'][:48]
        employment = row['employment']
        growth = row['total_growth_rate']
        logger.info(f"{occ_title:<50} {employment:>12,} {growth:>9.1%}")


def compare_adjustment_methods(adjuster: WageGrowthAdjuster,
                               start_year: int = 2022,
                               end_year: int = 2024):
    """Compare different wage adjustment methods."""
    logger.info("\n" + "=" * 70)
    logger.info("ADJUSTMENT METHOD COMPARISON")
    logger.info("=" * 70)
    
    # Get growth rates
    growth_df = adjuster.calculate_growth_rates_phase1(start_year, end_year)
    
    # Calculate different averages
    simple_avg = growth_df['total_growth_rate'].mean()
    median_growth = growth_df['total_growth_rate'].median()
    
    # Employment-weighted average
    total_employment = growth_df['employment'].sum()
    weighted_avg = (
        growth_df['total_growth_rate'] * growth_df['employment']
    ).sum() / total_employment
    
    logger.info("\n📊 Different Adjustment Methods:")
    logger.info(f"\n1. Simple Average (all occupations equal weight):")
    logger.info(f"   Adjustment factor: {1 + simple_avg:.4f} ({simple_avg:+.2%})")
    logger.info(f"   Use case: When occupation data not available")
    
    logger.info(f"\n2. Median Growth:")
    logger.info(f"   Adjustment factor: {1 + median_growth:.4f} ({median_growth:+.2%})")
    logger.info(f"   Use case: Robust to outliers")
    
    logger.info(f"\n3. Employment-Weighted Average (RECOMMENDED):")
    logger.info(f"   Adjustment factor: {1 + weighted_avg:.4f} ({weighted_avg:+.2%})")
    logger.info(f"   Use case: Reflects actual workforce composition")
    
    logger.info(f"\n4. Occupation-Specific (MOST ACCURATE):")
    logger.info(f"   Adjustment factor: Varies by occupation")
    logger.info(f"   Range: {growth_df['adjustment_factor'].min():.4f} to {growth_df['adjustment_factor'].max():.4f}")
    logger.info(f"   Use case: When PUMS occupation codes available")
    
    # Show impact on $50k wage
    test_wage = 50000
    logger.info(f"\n💰 Impact on ${test_wage:,} wage:")
    logger.info(f"   Simple average:     ${test_wage * (1 + simple_avg):>10,.0f}")
    logger.info(f"   Median:             ${test_wage * (1 + median_growth):>10,.0f}")
    logger.info(f"   Weighted average:   ${test_wage * (1 + weighted_avg):>10,.0f}")
    logger.info(f"   Min (occupation):   ${test_wage * growth_df['adjustment_factor'].min():>10,.0f}")
    logger.info(f"   Max (occupation):   ${test_wage * growth_df['adjustment_factor'].max():>10,.0f}")


def show_major_occupation_groups(adjuster: WageGrowthAdjuster,
                                 start_year: int = 2022,
                                 end_year: int = 2024):
    """Show growth rates by major occupation group."""
    logger.info("\n" + "=" * 70)
    logger.info("MAJOR OCCUPATION GROUP GROWTH RATES")
    logger.info("=" * 70)
    
    # Get growth rates
    growth_df = adjuster.calculate_growth_rates_phase1(start_year, end_year)
    
    # Filter to major groups (occupation codes ending in -0000)
    major_groups = growth_df[growth_df['occupation_code'].str.endswith('-0000')].copy()
    
    # Sort by employment
    major_groups = major_groups.sort_values('employment', ascending=False)
    
    logger.info(f"\n{'Major Occupation Group':<50} {'Employment':>12} {'Growth':>10} {'Adj Factor':>12}")
    logger.info("-" * 88)
    
    for _, row in major_groups.iterrows():
        occ_title = row['occupation_title'][:48]
        employment = row['employment']
        growth = row['total_growth_rate']
        adj_factor = row['adjustment_factor']
        logger.info(f"{occ_title:<50} {employment:>12,} {growth:>9.1%} {adj_factor:>11.4f}")
    
    logger.info("\n💡 These major groups can be used as fallbacks when specific occupation codes are not available.")


def demonstrate_phase2_projection(adjuster: WageGrowthAdjuster):
    """Demonstrate Phase 2 projection (2024 → 2026)."""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: PROJECTION TO 2026")
    logger.info("=" * 70)
    
    logger.info("\nPhase 2 uses historical growth trends to project forward:")
    logger.info("  • Historical period: 2022-2024 (2 years, actual BLS data)")
    logger.info("  • Projection period: 2024-2026 (2 years, projected)")
    logger.info("  • Method: Apply historical annual growth rate")
    
    # Project to 2026
    projected_df = adjuster.project_growth_phase2(2022, 2024, 2026)
    
    # Show examples
    logger.info("\n📊 Example Projections (Top 5 by employment):")
    logger.info(f"{'Occupation':<40} {'2022-24':>10} {'2024-26':>10} {'Total':>10}")
    logger.info("-" * 75)
    
    top5 = projected_df.nlargest(5, 'employment')
    for _, row in top5.iterrows():
        occ_title = row['occupation_title'][:38]
        hist_growth = row['total_growth_rate']
        proj_growth = row['projected_total_growth']
        total_growth = (1 + hist_growth) * (1 + proj_growth) - 1
        logger.info(f"{occ_title:<40} {hist_growth:>9.1%} {proj_growth:>9.1%} {total_growth:>9.1%}")
    
    logger.info("\n⚠️  Note: Phase 2 projections are less certain than Phase 1 actual data.")


def main():
    """Main demonstration function."""
    logger.info("=" * 70)
    logger.info("BLS OES WAGE GROWTH OPTIONS DEMO")
    logger.info("=" * 70)
    
    # Check available data
    available_years = check_available_data()
    
    if not available_years:
        logger.error("\n❌ No BLS OES data found. Please download data first.")
        logger.info("\nTo download BLS OES data:")
        logger.info("  1. Visit: https://www.bls.gov/oes/tables.htm")
        logger.info("  2. Download state-level Excel files for Hawaii")
        logger.info("  3. Place in: data/external/bls_oes/")
        return
    
    # Initialize adjuster
    logger.info("\n" + "=" * 70)
    logger.info("INITIALIZING WAGE GROWTH ADJUSTER")
    logger.info("=" * 70)
    
    adjuster = WageGrowthAdjuster(
        use_occupation_specific=True,
        use_industry_specific=False
    )
    
    # Determine years to use
    if 2022 in available_years and 2024 in available_years:
        start_year, end_year = 2022, 2024
        logger.info(f"\n✅ Using actual BLS data: {start_year} → {end_year}")
    elif len(available_years) >= 2:
        start_year = min(available_years)
        end_year = max(available_years)
        logger.info(f"\n⚠️  Using available years: {start_year} → {end_year}")
    else:
        logger.error("\n❌ Need at least 2 years of data for growth calculation")
        return
    
    # Show different options
    show_occupation_growth_rates(adjuster, start_year, end_year)
    show_major_occupation_groups(adjuster, start_year, end_year)
    compare_adjustment_methods(adjuster, start_year, end_year)
    
    # Show Phase 2 projection
    if end_year == 2024:
        demonstrate_phase2_projection(adjuster)
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY: AVAILABLE OPTIONS")
    logger.info("=" * 70)
    
    logger.info("\n✅ Phase 1 (2022 → 2024): Use actual BLS OES data")
    logger.info("   Options:")
    logger.info("   1. Overall adjustment (simple, less accurate)")
    logger.info("   2. Occupation-specific adjustment (RECOMMENDED)")
    logger.info("   3. Industry-specific adjustment (if PUMS has industry codes)")
    
    if end_year == 2024:
        logger.info("\n✅ Phase 2 (2024 → 2026): Project using historical trends")
        logger.info("   Method: Apply historical annual growth rates")
        logger.info("   Uncertainty: Higher than Phase 1")
    
    logger.info("\n💡 Recommendation:")
    logger.info("   • Use occupation-specific adjustment if PUMS has occupation codes")
    logger.info("   • Use employment-weighted average as fallback")
    logger.info("   • Phase 1 more reliable than Phase 2 (actual vs projected)")
    
    logger.info("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"\n❌ Error during demo: {e}", exc_info=True)
        sys.exit(1)
