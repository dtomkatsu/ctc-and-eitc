#!/usr/bin/env python3
"""
Simple HoH AGI Refinement

Reduces Head of Household average AGI by 10% and recalculates taxes.
Does NOT re-run systematic calibration to preserve return count accuracy.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

sys.path.append('/Users/dtomkatsu/CascadeProjects/ctc-and-eitc')

from src.tax.brackets.hawaii_tax import load_tax_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Apply HoH AGI refinement."""
    logger.info("="*80)
    logger.info("HEAD OF HOUSEHOLD AGI REFINEMENT (Simple)")
    logger.info("="*80)
    
    project_root = Path('/Users/dtomkatsu/CascadeProjects/ctc-and-eitc')
    
    # Load current calibrated data
    input_file = project_root / 'data' / 'processed' / 'tax_units_systematically_calibrated_20251103_085550.parquet'
    logger.info(f"\nLoading: {input_file.name}")
    df = pd.read_parquet(input_file)
    
    logger.info(f"Loaded {len(df):,} tax units")
    
    # Identify HoH filers
    hoh_mask = df['filing_status'] == 'head_of_household'
    hoh_count = hoh_mask.sum()
    
    logger.info(f"\nFound {hoh_count:,} Head of Household filers")
    
    # Get original HoH statistics
    original_avg_agi = (df.loc[hoh_mask, 'agi'] * df.loc[hoh_mask, 'weight']).sum() / df.loc[hoh_mask, 'weight'].sum()
    original_total_tax = (df.loc[hoh_mask, 'hi_state_tax'] * df.loc[hoh_mask, 'weight']).sum() / 1_000_000
    
    logger.info(f"\nOriginal HoH Statistics:")
    logger.info(f"  Average AGI: ${original_avg_agi:,.2f}")
    logger.info(f"  Total Tax: ${original_total_tax:,.1f}M")
    
    # Apply 10% AGI reduction to HoH
    logger.info(f"\nApplying 10% AGI reduction to HoH filers...")
    reduction_factor = 0.90
    
    df.loc[hoh_mask, 'agi'] = df.loc[hoh_mask, 'agi'] * reduction_factor
    
    if 'agi_without_cap_gains' in df.columns:
        df.loc[hoh_mask, 'agi_without_cap_gains'] = df.loc[hoh_mask, 'agi_without_cap_gains'] * reduction_factor
    
    if 'agi_with_cap_gains' in df.columns:
        df.loc[hoh_mask, 'agi_with_cap_gains'] = df.loc[hoh_mask, 'agi_with_cap_gains'] * reduction_factor
    
    # Update income column if it exists
    if 'income' in df.columns:
        df.loc[hoh_mask, 'income'] = df.loc[hoh_mask, 'agi']
    
    new_avg_agi = (df.loc[hoh_mask, 'agi'] * df.loc[hoh_mask, 'weight']).sum() / df.loc[hoh_mask, 'weight'].sum()
    logger.info(f"  New average AGI: ${new_avg_agi:,.2f}")
    logger.info(f"  Actual reduction: {(1 - new_avg_agi/original_avg_agi)*100:.1f}%")
    
    # Recalculate taxes for ALL filers (not just HoH) to be consistent
    logger.info(f"\nRecalculating Hawaii state taxes for all filers...")
    
    calculator = load_tax_data()
    
    filing_status_map = {
        'single': 'Single_Married_Separate',
        'married_filing_jointly': 'Joint_Surviving_Spouse',
        'married_filing_separately': 'Single_Married_Separate',
        'head_of_household': 'Head_of_Household',
        'qualifying_widow': 'Joint_Surviving_Spouse'
    }
    
    df['filing_status_hawaii'] = df['filing_status'].map(filing_status_map)
    
    # Fill any missing filing statuses
    df['filing_status_hawaii'] = df['filing_status_hawaii'].fillna('Single_Married_Separate')
    
    tax_results = calculator.calculate_tax_for_dataframe(
        df,
        income_col='agi',
        filing_status_col='filing_status_hawaii',
        year=2022
    )
    
    # Update tax columns
    df['hi_state_tax'] = tax_results['hi_tax_tax_liability']
    df['hi_taxable_income'] = tax_results['hi_tax_taxable_income']
    df['hi_effective_rate'] = tax_results['hi_tax_effective_rate']
    df['hi_tax_tax_liability'] = tax_results['hi_tax_tax_liability']
    df['hi_tax_taxable_income'] = tax_results['hi_tax_taxable_income']
    df['hi_tax_effective_rate'] = tax_results['hi_tax_effective_rate']
    
    # Get new HoH statistics
    new_total_tax = (df.loc[hoh_mask, 'hi_state_tax'] * df.loc[hoh_mask, 'weight']).sum() / 1_000_000
    
    logger.info(f"\nNew HoH Statistics:")
    logger.info(f"  Average AGI: ${new_avg_agi:,.2f}")
    logger.info(f"  Total Tax: ${new_total_tax:,.1f}M")
    logger.info(f"  Tax reduction: ${original_total_tax - new_total_tax:,.1f}M ({(1 - new_total_tax/original_total_tax)*100:.1f}%)")
    
    # Save refined data
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = project_root / 'data' / 'processed' / f'tax_units_refined_hoh_{timestamp}.parquet'
    
    df.to_parquet(output_file, index=False)
    logger.info(f"\n✅ Saved refined tax units to: {output_file.name}")
    
    # Summary by filing status
    logger.info("\n" + "="*80)
    logger.info("REFINED MODEL SUMMARY")
    logger.info("="*80)
    
    summary = []
    for status in sorted(df['filing_status'].unique()):
        status_data = df[df['filing_status'] == status]
        summary.append({
            'filing_status': status,
            'returns': status_data['weight'].sum(),
            'agi_millions': (status_data['agi'] * status_data['weight']).sum() / 1_000_000,
            'tax_millions': (status_data['hi_state_tax'] * status_data['weight']).sum() / 1_000_000,
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df['avg_agi'] = (summary_df['agi_millions'] * 1_000_000) / summary_df['returns']
    summary_df['effective_rate'] = (summary_df['tax_millions'] / summary_df['agi_millions']) * 100
    
    print("\n" + "="*80)
    print("Summary by Filing Status:")
    print("="*80)
    print(f"{'Filing Status':<30} {'Returns':>12} {'AGI ($M)':>12} {'Tax ($M)':>12} {'Avg AGI':>12} {'Eff Rate':>10}")
    print("-"*80)
    
    for _, row in summary_df.iterrows():
        print(f"{row['filing_status']:<30} {row['returns']:>12,.0f} ${row['agi_millions']:>11,.1f} "
              f"${row['tax_millions']:>11,.1f} ${row['avg_agi']:>11,.0f} {row['effective_rate']:>9.2f}%")
    
    print("-"*80)
    print(f"{'TOTAL':<30} {summary_df['returns'].sum():>12,.0f} ${summary_df['agi_millions'].sum():>11,.1f} "
          f"${summary_df['tax_millions'].sum():>11,.1f} ${(summary_df['agi_millions'].sum() * 1_000_000 / summary_df['returns'].sum()):>11,.0f} "
          f"{(summary_df['tax_millions'].sum() / summary_df['agi_millions'].sum()) * 100:>9.2f}%")
    
    return output_file


if __name__ == "__main__":
    output_file = main()
    print(f"\n✅ Refinement complete!")
    print(f"Output file: {output_file}")
