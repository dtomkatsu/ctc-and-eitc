#!/usr/bin/env python3
"""
Apply HoH Refinement to Tax Units

Reduces Head of Household average AGI by 10% to improve model accuracy
based on validation against 2021 actual DOTAX data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_hoh_agi_adjustment(df: pd.DataFrame, reduction_pct: float = 10.0) -> pd.DataFrame:
    """
    Reduce HoH average AGI by specified percentage.
    
    Args:
        df: Tax units dataframe
        reduction_pct: Percentage to reduce HoH AGI (default 10%)
    
    Returns:
        Modified dataframe with adjusted HoH AGI
    """
    df = df.copy()
    
    # Identify HoH filers
    hoh_mask = df['filing_status'] == 'head_of_household'
    hoh_count = hoh_mask.sum()
    
    if hoh_count == 0:
        logger.warning("No Head of Household filers found")
        return df
    
    # Get original values
    original_agi = df.loc[hoh_mask, 'agi'].copy()
    original_avg = (original_agi * df.loc[hoh_mask, 'weight']).sum() / df.loc[hoh_mask, 'weight'].sum()
    
    # Apply reduction
    reduction_factor = 1.0 - (reduction_pct / 100.0)
    df.loc[hoh_mask, 'agi'] = df.loc[hoh_mask, 'agi'] * reduction_factor
    
    # Recalculate related columns if they exist
    if 'agi_without_cap_gains' in df.columns:
        df.loc[hoh_mask, 'agi_without_cap_gains'] = df.loc[hoh_mask, 'agi_without_cap_gains'] * reduction_factor
    
    if 'agi_with_cap_gains' in df.columns:
        df.loc[hoh_mask, 'agi_with_cap_gains'] = df.loc[hoh_mask, 'agi_with_cap_gains'] * reduction_factor
    
    # Get new values
    new_agi = df.loc[hoh_mask, 'agi'].copy()
    new_avg = (new_agi * df.loc[hoh_mask, 'weight']).sum() / df.loc[hoh_mask, 'weight'].sum()
    
    logger.info(f"Applied {reduction_pct}% AGI reduction to {hoh_count:,} Head of Household filers")
    logger.info(f"  Original average AGI: ${original_avg:,.2f}")
    logger.info(f"  New average AGI: ${new_avg:,.2f}")
    logger.info(f"  Actual reduction: {(1 - new_avg/original_avg)*100:.2f}%")
    
    return df


def recalculate_taxes(df: pd.DataFrame) -> pd.DataFrame:
    """Recalculate Hawaii state taxes after AGI adjustment."""
    from src.tax.brackets.hawaii_tax import load_tax_data
    
    logger.info("Recalculating Hawaii state taxes with adjusted AGI...")
    
    # Load tax calculator
    calculator = load_tax_data()
    
    # Map filing status
    filing_status_map = {
        'single': 'Single_Married_Separate',
        'married_filing_jointly': 'Joint_Surviving_Spouse',
        'married_filing_separately': 'Single_Married_Separate',
        'head_of_household': 'Head_of_Household',
        'qualifying_widow': 'Joint_Surviving_Spouse'
    }
    
    df['filing_status_hawaii'] = df['filing_status'].map(filing_status_map)
    
    # Calculate taxes using 2022 brackets
    tax_results = calculator.calculate_tax_for_dataframe(
        df,
        income_col='agi',
        filing_status_col='filing_status_hawaii',
        year=2022
    )
    
    # Update tax columns
    df['hi_tax_tax_liability'] = tax_results['hi_tax_tax_liability']
    df['hi_tax_taxable_income'] = tax_results['hi_tax_taxable_income']
    df['hi_tax_effective_rate'] = tax_results['hi_tax_effective_rate']
    df['hi_state_tax'] = tax_results['hi_tax_tax_liability']
    df['hi_taxable_income'] = tax_results['hi_tax_taxable_income']
    df['hi_effective_rate'] = tax_results['hi_tax_effective_rate']
    
    logger.info("Tax recalculation complete")
    
    return df


def apply_systematic_calibration(df: pd.DataFrame) -> pd.DataFrame:
    """Apply systematic calibration after AGI adjustment."""
    from src.tax.adjustments.agi_calibrator import AGICalibrator
    from src.tax.adjustments.tax_revenue_calibrator import TaxRevenueCalibrator
    from src.tax.adjustments.filing_status_calibrator import FilingStatusCalibrator
    
    logger.info("\n" + "="*80)
    logger.info("Applying Systematic Calibration")
    logger.info("="*80)
    
    # Step 1: AGI distribution calibration
    logger.info("\nStep 1: AGI Distribution Calibration")
    agi_calibrator = AGICalibrator()
    df = agi_calibrator.calibrate(df)
    
    # Step 2: Filing status share calibration
    logger.info("\nStep 2: Filing Status Share Calibration")
    fs_calibrator = FilingStatusCalibrator()
    df = fs_calibrator.calibrate(df)
    
    # Step 3: Tax revenue calibration
    logger.info("\nStep 3: Tax Revenue Calibration")
    tax_calibrator = TaxRevenueCalibrator()
    df = tax_calibrator.calibrate(df)
    
    return df


def main():
    """Main refinement workflow."""
    import sys
    sys.path.append('/Users/dtomkatsu/CascadeProjects/ctc-and-eitc')
    
    logger.info("="*80)
    logger.info("HEAD OF HOUSEHOLD AGI REFINEMENT")
    logger.info("="*80)
    
    project_root = Path(__file__).parent.parent.parent
    
    # Load current calibrated data
    input_file = project_root / 'data' / 'processed' / 'tax_units_systematically_calibrated_20251103_085550.parquet'
    logger.info(f"\nLoading: {input_file}")
    df = pd.read_parquet(input_file)
    
    logger.info(f"Loaded {len(df):,} tax units")
    
    # Apply HoH AGI adjustment
    logger.info("\n" + "="*80)
    logger.info("APPLYING HOH AGI ADJUSTMENT")
    logger.info("="*80)
    df = apply_hoh_agi_adjustment(df, reduction_pct=10.0)
    
    # Recalculate taxes
    logger.info("\n" + "="*80)
    logger.info("RECALCULATING TAXES")
    logger.info("="*80)
    df = recalculate_taxes(df)
    
    # Apply systematic calibration
    df = apply_systematic_calibration(df)
    
    # Save refined data
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = project_root / 'data' / 'processed' / f'tax_units_refined_hoh_{timestamp}.parquet'
    
    df.to_parquet(output_file, index=False)
    logger.info(f"\nSaved refined tax units to: {output_file}")
    
    # Validation summary
    logger.info("\n" + "="*80)
    logger.info("REFINEMENT SUMMARY")
    logger.info("="*80)
    
    # Calculate totals by filing status
    summary = []
    for status in df['filing_status'].unique():
        status_data = df[df['filing_status'] == status]
        summary.append({
            'filing_status': status,
            'returns': status_data['weight'].sum(),
            'agi_millions': (status_data['agi'] * status_data['weight']).sum() / 1_000_000,
            'tax_millions': (status_data['hi_state_tax'] * status_data['weight']).sum() / 1_000_000,
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df['avg_agi'] = (summary_df['agi_millions'] * 1_000_000) / summary_df['returns']
    summary_df['avg_tax'] = (summary_df['tax_millions'] * 1_000_000) / summary_df['returns']
    summary_df['effective_rate'] = (summary_df['tax_millions'] / summary_df['agi_millions']) * 100
    
    print("\nRefined Model Summary:")
    print(summary_df.to_string(index=False))
    
    print(f"\nTotal Returns: {summary_df['returns'].sum():,.0f}")
    print(f"Total AGI: ${summary_df['agi_millions'].sum():,.1f}M")
    print(f"Total Tax: ${summary_df['tax_millions'].sum():,.1f}M")
    print(f"Overall Effective Rate: {(summary_df['tax_millions'].sum() / summary_df['agi_millions'].sum()) * 100:.2f}%")
    
    return output_file


if __name__ == "__main__":
    output_file = main()
    print(f"\n✅ Refinement complete! Output: {output_file}")
