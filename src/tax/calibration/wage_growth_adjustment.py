"""
Wage Growth Adjustment for Hawaii Tax Model

This module implements temporal adjustment of income data using BLS OES wage growth rates.
Critical for aligning historical PUMS data (2022) to current/future years (2024, 2026).

Key Features:
- Phase 1: 2022 → 2024 adjustment using actual BLS OES data
- Phase 2: 2024 → 2026 projection using trend analysis
- Occupation-specific growth rates
- Industry-specific adjustments
- Fallback to overall wage growth for missing occupations
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.data.bls_oes_loader import BLSOESLoader

logger = logging.getLogger(__name__)


class WageGrowthAdjuster:
    """
    Adjusts wage income for temporal changes using BLS OES data.
    
    Implements two-phase approach:
    - Phase 1: Historical adjustment (2022 → 2024) using actual BLS data
    - Phase 2: Forward projection (2024 → 2026) using trend analysis
    """
    
    def __init__(self, 
                 bls_data_dir: Optional[Path] = None,
                 use_occupation_specific: bool = True,
                 use_industry_specific: bool = False):
        """
        Initialize wage growth adjuster.
        
        Args:
            bls_data_dir: Directory with BLS OES data
            use_occupation_specific: Use occupation-specific growth rates
            use_industry_specific: Use industry-specific growth rates (if available)
        """
        self.bls_loader = BLSOESLoader(data_dir=bls_data_dir)
        self.use_occupation_specific = use_occupation_specific
        self.use_industry_specific = use_industry_specific
        
        # Cache for growth rates
        self._growth_rate_cache: Dict[Tuple[int, int], pd.DataFrame] = {}
        
        logger.info("Wage Growth Adjuster initialized")
        logger.info(f"  Occupation-specific: {use_occupation_specific}")
        logger.info(f"  Industry-specific: {use_industry_specific}")
    
    def calculate_growth_rates_phase1(self,
                                      start_year: int = 2022,
                                      end_year: int = 2024) -> pd.DataFrame:
        """
        Calculate actual wage growth rates from BLS OES data (Phase 1).
        
        Args:
            start_year: Starting year (default: 2022, PUMS base year)
            end_year: Ending year (default: 2024, current year)
            
        Returns:
            DataFrame with occupation codes and growth rates
        """
        logger.info("\n" + "=" * 70)
        logger.info(f"PHASE 1: CALCULATING WAGE GROWTH RATES ({start_year} → {end_year})")
        logger.info("=" * 70)
        
        # Check cache
        cache_key = (start_year, end_year)
        if cache_key in self._growth_rate_cache:
            logger.info("Using cached growth rates")
            return self._growth_rate_cache[cache_key]
        
        # Load BLS data for both years
        logger.info(f"\nLoading BLS OES data for {start_year}...")
        df_start = self.bls_loader.load_hawaii_wages(start_year)
        logger.info(f"  Loaded {len(df_start):,} occupation records")
        
        logger.info(f"\nLoading BLS OES data for {end_year}...")
        df_end = self.bls_loader.load_hawaii_wages(end_year)
        logger.info(f"  Loaded {len(df_end):,} occupation records")
        
        # Filter for all-industry totals (avoid double-counting)
        df_start_all = df_start[
            (df_start['industry_code'].isna()) | 
            (df_start['industry_code'] == '000000')
        ].copy()
        
        df_end_all = df_end[
            (df_end['industry_code'].isna()) | 
            (df_end['industry_code'] == '000000')
        ].copy()
        
        logger.info(f"\nFiltered to all-industry totals:")
        logger.info(f"  {start_year}: {len(df_start_all):,} occupations")
        logger.info(f"  {end_year}: {len(df_end_all):,} occupations")
        
        # Merge on occupation code
        merged = df_start_all.merge(
            df_end_all,
            on='occupation_code',
            suffixes=('_start', '_end'),
            how='inner'
        )
        
        logger.info(f"\nMatched {len(merged):,} occupations across both years")
        
        # Calculate growth rates
        merged['wage_growth_rate'] = (
            (merged['mean_wage_annual_end'] - merged['mean_wage_annual_start']) /
            merged['mean_wage_annual_start']
        )
        
        # Calculate annualized growth rate
        years_diff = end_year - start_year
        merged['annual_growth_rate'] = (
            (1 + merged['wage_growth_rate']) ** (1 / years_diff) - 1
        )
        
        # Calculate adjustment factor (multiply wages by this)
        merged['adjustment_factor'] = 1 + merged['wage_growth_rate']
        
        # Select relevant columns
        result = merged[[
            'occupation_code',
            'occupation_title_start',
            'employment_start',
            'mean_wage_annual_start',
            'mean_wage_annual_end',
            'wage_growth_rate',
            'annual_growth_rate',
            'adjustment_factor'
        ]].copy()
        
        result.columns = [
            'occupation_code',
            'occupation_title',
            'employment',
            f'mean_wage_{start_year}',
            f'mean_wage_{end_year}',
            'total_growth_rate',
            'annual_growth_rate',
            'adjustment_factor'
        ]
        
        # Sort by employment (most common occupations first)
        result = result.sort_values('employment', ascending=False)
        
        # Cache the results
        self._growth_rate_cache[cache_key] = result
        
        # Log summary statistics
        self._log_growth_summary(result, start_year, end_year)
        
        return result
    
    def _log_growth_summary(self, 
                           growth_df: pd.DataFrame,
                           start_year: int,
                           end_year: int):
        """Log summary statistics of growth rates."""
        logger.info("\n" + "=" * 70)
        logger.info("WAGE GROWTH SUMMARY")
        logger.info("=" * 70)
        
        # Overall statistics
        avg_growth = growth_df['total_growth_rate'].mean()
        median_growth = growth_df['total_growth_rate'].median()
        
        # Employment-weighted average
        total_employment = growth_df['employment'].sum()
        weighted_growth = (
            growth_df['total_growth_rate'] * growth_df['employment']
        ).sum() / total_employment
        
        logger.info(f"\nOverall Growth ({start_year} → {end_year}):")
        logger.info(f"  Average growth:           {avg_growth:>8.2%}")
        logger.info(f"  Median growth:            {median_growth:>8.2%}")
        logger.info(f"  Employment-weighted:      {weighted_growth:>8.2%}")
        
        # Annual growth
        years_diff = end_year - start_year
        annual_avg = (1 + avg_growth) ** (1 / years_diff) - 1
        annual_weighted = (1 + weighted_growth) ** (1 / years_diff) - 1
        
        logger.info(f"\nAnnualized Growth:")
        logger.info(f"  Average annual:           {annual_avg:>8.2%}")
        logger.info(f"  Weighted annual:          {annual_weighted:>8.2%}")
        
        # Distribution
        logger.info(f"\nGrowth Rate Distribution:")
        logger.info(f"  Min:                      {growth_df['total_growth_rate'].min():>8.2%}")
        logger.info(f"  25th percentile:          {growth_df['total_growth_rate'].quantile(0.25):>8.2%}")
        logger.info(f"  50th percentile (median): {growth_df['total_growth_rate'].quantile(0.50):>8.2%}")
        logger.info(f"  75th percentile:          {growth_df['total_growth_rate'].quantile(0.75):>8.2%}")
        logger.info(f"  Max:                      {growth_df['total_growth_rate'].max():>8.2%}")
        
        # Top/bottom movers
        logger.info(f"\nTop 5 Growing Occupations:")
        top5 = growth_df.nlargest(5, 'total_growth_rate')
        for idx, row in top5.iterrows():
            logger.info(f"  {row['occupation_title'][:50]:50s} {row['total_growth_rate']:>8.2%}")
        
        logger.info(f"\nBottom 5 Growing Occupations:")
        bottom5 = growth_df.nsmallest(5, 'total_growth_rate')
        for idx, row in bottom5.iterrows():
            logger.info(f"  {row['occupation_title'][:50]:50s} {row['total_growth_rate']:>8.2%}")
        
        logger.info("\n" + "=" * 70)
    
    def adjust_wages_phase1(self,
                           tax_units: pd.DataFrame,
                           start_year: int = 2022,
                           end_year: int = 2024,
                           wage_col: str = 'wage_income',
                           occupation_col: Optional[str] = None) -> pd.DataFrame:
        """
        Adjust wage income from start_year to end_year using BLS growth rates (Phase 1).
        
        Args:
            tax_units: DataFrame with tax units
            start_year: Starting year (PUMS base year)
            end_year: Target year
            wage_col: Name of wage income column
            occupation_col: Name of occupation code column (if available)
            
        Returns:
            DataFrame with adjusted wage income
        """
        logger.info("\n" + "=" * 70)
        logger.info(f"PHASE 1: ADJUSTING WAGES ({start_year} → {end_year})")
        logger.info("=" * 70)
        
        # Create copy
        tax_units_adjusted = tax_units.copy()
        
        # Check if wage column exists
        if wage_col not in tax_units_adjusted.columns:
            logger.warning(f"Wage column '{wage_col}' not found. Skipping wage adjustment.")
            return tax_units_adjusted
        
        # Get growth rates
        growth_df = self.calculate_growth_rates_phase1(start_year, end_year)
        
        # Calculate overall adjustment factor (fallback)
        weight_col = 'calibrated_weight' if 'calibrated_weight' in tax_units_adjusted.columns else 'weight'
        total_employment = growth_df['employment'].sum()
        overall_adjustment = (
            growth_df['adjustment_factor'] * growth_df['employment']
        ).sum() / total_employment
        
        logger.info(f"\nOverall adjustment factor: {overall_adjustment:.4f} ({(overall_adjustment-1)*100:+.2f}%)")
        
        # Apply adjustment
        if occupation_col and occupation_col in tax_units_adjusted.columns and self.use_occupation_specific:
            logger.info(f"\nApplying occupation-specific adjustments...")
            tax_units_adjusted = self._apply_occupation_specific_adjustment(
                tax_units_adjusted,
                growth_df,
                wage_col,
                occupation_col,
                overall_adjustment
            )
        else:
            logger.info(f"\nApplying overall adjustment factor to all wages...")
            original_total = (tax_units_adjusted[wage_col] * tax_units_adjusted[weight_col]).sum()
            
            tax_units_adjusted[f'{wage_col}_unadjusted'] = tax_units_adjusted[wage_col]
            tax_units_adjusted[wage_col] = tax_units_adjusted[wage_col] * overall_adjustment
            
            adjusted_total = (tax_units_adjusted[wage_col] * tax_units_adjusted[weight_col]).sum()
            
            logger.info(f"  Original total wages: ${original_total:,.0f}")
            logger.info(f"  Adjusted total wages: ${adjusted_total:,.0f}")
            logger.info(f"  Change: ${adjusted_total - original_total:,.0f} ({(adjusted_total/original_total - 1)*100:+.2f}%)")
        
        # Update total income
        if 'income' in tax_units_adjusted.columns:
            self._update_total_income(tax_units_adjusted, wage_col)
        
        logger.info("\n" + "=" * 70)
        logger.info("WAGE ADJUSTMENT COMPLETE")
        logger.info("=" * 70 + "\n")
        
        return tax_units_adjusted
    
    def _apply_occupation_specific_adjustment(self,
                                             tax_units: pd.DataFrame,
                                             growth_df: pd.DataFrame,
                                             wage_col: str,
                                             occupation_col: str,
                                             fallback_factor: float) -> pd.DataFrame:
        """Apply occupation-specific wage adjustments."""
        
        # Create occupation code lookup
        growth_lookup = growth_df.set_index('occupation_code')['adjustment_factor'].to_dict()
        
        # Map PUMS occupation codes to major groups (first 2 digits)
        def get_major_group(occ_code):
            if pd.isna(occ_code):
                return None
            occ_str = str(occ_code)
            if len(occ_str) >= 2:
                return occ_str[:2] + '-0000'
            return None
        
        tax_units['_occ_major_group'] = tax_units[occupation_col].apply(get_major_group)
        
        # Apply adjustments
        def get_adjustment_factor(occ_code):
            if pd.isna(occ_code):
                return fallback_factor
            
            # Try exact match
            if occ_code in growth_lookup:
                return growth_lookup[occ_code]
            
            # Try major group
            major_group = get_major_group(occ_code)
            if major_group and major_group in growth_lookup:
                return growth_lookup[major_group]
            
            # Fallback to overall
            return fallback_factor
        
        tax_units['_adjustment_factor'] = tax_units['_occ_major_group'].apply(get_adjustment_factor)
        
        # Store original
        tax_units[f'{wage_col}_unadjusted'] = tax_units[wage_col]
        
        # Apply adjustment
        tax_units[wage_col] = tax_units[wage_col] * tax_units['_adjustment_factor']
        
        # Log results
        weight_col = 'calibrated_weight' if 'calibrated_weight' in tax_units.columns else 'weight'
        
        n_matched = (~tax_units['_occ_major_group'].isna()).sum()
        n_total = len(tax_units)
        
        logger.info(f"  Matched {n_matched:,} / {n_total:,} units to occupation codes ({n_matched/n_total*100:.1f}%)")
        
        original_total = (tax_units[f'{wage_col}_unadjusted'] * tax_units[weight_col]).sum()
        adjusted_total = (tax_units[wage_col] * tax_units[weight_col]).sum()
        
        logger.info(f"  Original total wages: ${original_total:,.0f}")
        logger.info(f"  Adjusted total wages: ${adjusted_total:,.0f}")
        logger.info(f"  Change: ${adjusted_total - original_total:,.0f} ({(adjusted_total/original_total - 1)*100:+.2f}%)")
        
        # Clean up temporary columns
        tax_units.drop(columns=['_occ_major_group', '_adjustment_factor'], inplace=True)
        
        return tax_units
    
    def _update_total_income(self, tax_units: pd.DataFrame, wage_col: str):
        """Update total income after wage adjustment."""
        
        source_columns = ['wage_income', 'dividend_income', 'interest_income',
                         'business_income', 'capital_gains', 'pension_income',
                         'other_income']
        
        # Check if we have income source breakdown
        has_sources = all(col in tax_units.columns for col in source_columns)
        
        if has_sources:
            logger.info(f"\nUpdating total income from source components...")
            tax_units['income'] = tax_units[source_columns].sum(axis=1)
        else:
            logger.info(f"\nUpdating total income (wage adjustment only)...")
            # Calculate wage change
            if f'{wage_col}_unadjusted' in tax_units.columns:
                wage_change = tax_units[wage_col] - tax_units[f'{wage_col}_unadjusted']
                tax_units['income'] = tax_units['income'] + wage_change
    
    def project_growth_phase2(self,
                             historical_start: int = 2022,
                             historical_end: int = 2024,
                             projection_year: int = 2026) -> pd.DataFrame:
        """
        Project wage growth rates for future years (Phase 2).
        
        Uses historical growth trends to project forward.
        
        Args:
            historical_start: Start of historical period
            historical_end: End of historical period
            projection_year: Year to project to
            
        Returns:
            DataFrame with projected growth rates
        """
        logger.info("\n" + "=" * 70)
        logger.info(f"PHASE 2: PROJECTING WAGE GROWTH ({historical_end} → {projection_year})")
        logger.info("=" * 70)
        
        # Get historical growth rates
        historical_growth = self.calculate_growth_rates_phase1(historical_start, historical_end)
        
        # Calculate projection period
        historical_years = historical_end - historical_start
        projection_years = projection_year - historical_end
        
        logger.info(f"\nHistorical period: {historical_years} years ({historical_start}-{historical_end})")
        logger.info(f"Projection period: {projection_years} years ({historical_end}-{projection_year})")
        
        # Project using historical annual growth rates
        projected = historical_growth.copy()
        
        # Calculate projected adjustment factor
        projected['projected_adjustment_factor'] = (
            (1 + projected['annual_growth_rate']) ** projection_years
        )
        
        projected['projected_total_growth'] = projected['projected_adjustment_factor'] - 1
        
        # Log projection summary
        avg_projected = projected['projected_total_growth'].mean()
        weighted_projected = (
            projected['projected_total_growth'] * projected['employment']
        ).sum() / projected['employment'].sum()
        
        logger.info(f"\nProjected Growth ({historical_end} → {projection_year}):")
        logger.info(f"  Average:              {avg_projected:>8.2%}")
        logger.info(f"  Employment-weighted:  {weighted_projected:>8.2%}")
        
        logger.info("\n" + "=" * 70)
        
        return projected


def generate_growth_rate_report(adjuster: WageGrowthAdjuster,
                                start_year: int = 2022,
                                end_year: int = 2024,
                                output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Generate detailed wage growth rate report.
    
    Args:
        adjuster: Initialized WageGrowthAdjuster
        start_year: Starting year
        end_year: Ending year
        output_path: Optional path to save report
        
    Returns:
        DataFrame with growth rate details
    """
    logger.info("Generating wage growth rate report...")
    
    # Get growth rates
    growth_df = adjuster.calculate_growth_rates_phase1(start_year, end_year)
    
    # Add additional analysis columns
    growth_df['growth_category'] = pd.cut(
        growth_df['total_growth_rate'],
        bins=[-np.inf, 0, 0.05, 0.10, 0.15, np.inf],
        labels=['Negative', 'Low (0-5%)', 'Moderate (5-10%)', 'High (10-15%)', 'Very High (>15%)']
    )
    
    # Save if path provided
    if output_path:
        growth_df.to_csv(output_path, index=False)
        logger.info(f"  ✅ Saved report to: {output_path}")
    
    return growth_df


if __name__ == '__main__':
    # Demo usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    logger.info("=" * 70)
    logger.info("WAGE GROWTH ADJUSTMENT DEMO")
    logger.info("=" * 70)
    
    # Initialize adjuster
    adjuster = WageGrowthAdjuster()
    
    # Phase 1: Calculate 2022 → 2024 growth rates
    growth_df = adjuster.calculate_growth_rates_phase1(2022, 2024)
    
    logger.info(f"\n✅ Calculated growth rates for {len(growth_df):,} occupations")
    logger.info(f"\nSample growth rates:")
    logger.info(growth_df.head(10).to_string(index=False))
    
    # Phase 2: Project 2024 → 2026
    projected_df = adjuster.project_growth_phase2(2022, 2024, 2026)
    
    logger.info(f"\n✅ Projected growth rates for {len(projected_df):,} occupations")
