#!/usr/bin/env python3
"""
Validate Refined Model Against 2022 DOTAX SOI Data

Compares the HoH-refined model against the actual 2022 DOTAX SOI targets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_2022_dotax_targets() -> pd.DataFrame:
    """Load 2022 DOTAX SOI targets from Table 5A."""
    project_root = Path(__file__).parent.parent.parent
    file_path = project_root / 'data' / 'raw' / 'Dotax Soi 2022 - 5A.csv'
    
    # Manually parse the DOTAX data
    data = {
        'filing_status': [
            'Married Filing Jointly',
            'Single', 
            'Married Filing Separately',
            'Head of Household',
            'Qualifying Widow(er)'
        ],
        'num_returns': [216358, 335198, 16007, 67393, 161],
        'agi_millions': [26551 - 422, 14297 - 250, 3149 - 27, 3744 - 19, 9 - 0],  # Net AGI
        'tax_millions': [1551, 818, 249, 177, 0],  # After credits
    }
    
    df = pd.DataFrame(data)
    df['effective_rate'] = (df['tax_millions'] / df['agi_millions']) * 100
    df['avg_agi'] = (df['agi_millions'] * 1_000_000) / df['num_returns']
    
    logger.info("Loaded 2022 DOTAX SOI targets")
    return df


def load_refined_model() -> pd.DataFrame:
    """Load the refined model data."""
    project_root = Path(__file__).parent.parent.parent
    
    # Find the latest refined file
    refined_files = sorted((project_root / 'data' / 'processed').glob('tax_units_refined_hoh_*.parquet'))
    
    if not refined_files:
        raise FileNotFoundError("No refined model files found")
    
    file_path = refined_files[-1]
    logger.info(f"Loading refined model: {file_path.name}")
    
    df = pd.read_parquet(file_path)
    return df


def aggregate_model_data(model_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate model data by filing status."""
    
    # Standardize filing status names
    status_map = {
        'single': 'Single',
        'married_filing_jointly': 'Married Filing Jointly',
        'married_filing_separately': 'Married Filing Separately',
        'head_of_household': 'Head of Household',
        'qualifying_widow': 'Qualifying Widow(er)'
    }
    
    model_df['filing_status_standard'] = model_df['filing_status'].apply(
        lambda x: status_map.get(x.lower(), x)
    )
    
    # Aggregate
    results = []
    for status in model_df['filing_status_standard'].unique():
        status_data = model_df[model_df['filing_status_standard'] == status]
        
        result = {
            'filing_status': status,
            'num_returns': status_data['weight'].sum(),
            'agi_millions': (status_data['agi'] * status_data['weight']).sum() / 1_000_000,
            'tax_millions': (status_data['hi_state_tax'] * status_data['weight']).sum() / 1_000_000,
        }
        results.append(result)
    
    model_agg = pd.DataFrame(results)
    model_agg['effective_rate'] = (model_agg['tax_millions'] / model_agg['agi_millions']) * 100
    model_agg['avg_agi'] = (model_agg['agi_millions'] * 1_000_000) / model_agg['num_returns']
    
    return model_agg


def compare_model_vs_targets(model_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """Compare refined model against 2022 DOTAX targets."""
    
    comparison = model_df.merge(
        target_df,
        on='filing_status',
        how='outer',
        suffixes=('_model', '_target')
    )
    
    # Calculate gaps
    comparison['returns_gap'] = comparison['num_returns_model'] - comparison['num_returns_target']
    comparison['returns_gap_pct'] = (comparison['num_returns_model'] / comparison['num_returns_target'] - 1) * 100
    
    comparison['agi_gap'] = comparison['agi_millions_model'] - comparison['agi_millions_target']
    comparison['agi_gap_pct'] = (comparison['agi_millions_model'] / comparison['agi_millions_target'] - 1) * 100
    
    comparison['tax_gap'] = comparison['tax_millions_model'] - comparison['tax_millions_target']
    comparison['tax_gap_pct'] = (comparison['tax_millions_model'] / comparison['tax_millions_target'] - 1) * 100
    
    comparison['effective_rate_gap'] = comparison['effective_rate_model'] - comparison['effective_rate_target']
    
    return comparison


def print_validation_report(comparison: pd.DataFrame, model_name: str = "Refined Model"):
    """Print formatted validation report."""
    
    print("\n" + "="*100)
    print(f"{model_name} vs 2022 DOTAX SOI TARGETS")
    print("="*100)
    
    # Returns Comparison
    print("\n### Returns Comparison")
    print("-"*100)
    print(f"{'Filing Status':<30} {'Model':>12} {'Target':>12} {'Gap':>12} {'Gap %':>10}")
    print("-"*100)
    
    for _, row in comparison.iterrows():
        if pd.notna(row['num_returns_model']) and pd.notna(row['num_returns_target']):
            print(f"{row['filing_status']:<30} {row['num_returns_model']:>12,.0f} {row['num_returns_target']:>12,.0f} "
                  f"{row['returns_gap']:>12,.0f} {row['returns_gap_pct']:>9.1f}%")
    
    print("-"*100)
    model_total_returns = comparison['num_returns_model'].sum()
    target_total_returns = comparison['num_returns_target'].sum()
    print(f"{'TOTAL':<30} {model_total_returns:>12,.0f} {target_total_returns:>12,.0f} "
          f"{model_total_returns - target_total_returns:>12,.0f} "
          f"{(model_total_returns / target_total_returns - 1) * 100:>9.1f}%")
    
    # AGI Comparison
    print("\n### AGI Comparison ($M)")
    print("-"*100)
    print(f"{'Filing Status':<30} {'Model':>12} {'Target':>12} {'Gap':>12} {'Gap %':>10}")
    print("-"*100)
    
    for _, row in comparison.iterrows():
        if pd.notna(row['agi_millions_model']) and pd.notna(row['agi_millions_target']):
            print(f"{row['filing_status']:<30} ${row['agi_millions_model']:>11,.1f} ${row['agi_millions_target']:>11,.1f} "
                  f"${row['agi_gap']:>11,.1f} {row['agi_gap_pct']:>9.1f}%")
    
    print("-"*100)
    model_total_agi = comparison['agi_millions_model'].sum()
    target_total_agi = comparison['agi_millions_target'].sum()
    print(f"{'TOTAL':<30} ${model_total_agi:>11,.1f} ${target_total_agi:>11,.1f} "
          f"${model_total_agi - target_total_agi:>11,.1f} "
          f"{(model_total_agi / target_total_agi - 1) * 100:>9.1f}%")
    
    # Tax Revenue Comparison
    print("\n### Tax Revenue Comparison ($M)")
    print("-"*100)
    print(f"{'Filing Status':<30} {'Model':>12} {'Target':>12} {'Gap':>12} {'Gap %':>10}")
    print("-"*100)
    
    for _, row in comparison.iterrows():
        if pd.notna(row['tax_millions_model']) and pd.notna(row['tax_millions_target']):
            print(f"{row['filing_status']:<30} ${row['tax_millions_model']:>11,.1f} ${row['tax_millions_target']:>11,.1f} "
                  f"${row['tax_gap']:>11,.1f} {row['tax_gap_pct']:>9.1f}%")
    
    print("-"*100)
    model_total_tax = comparison['tax_millions_model'].sum()
    target_total_tax = comparison['tax_millions_target'].sum()
    print(f"{'TOTAL':<30} ${model_total_tax:>11,.1f} ${target_total_tax:>11,.1f} "
          f"${model_total_tax - target_total_tax:>11,.1f} "
          f"{(model_total_tax / target_total_tax - 1) * 100:>9.1f}%")
    
    # Effective Rate Comparison
    print("\n### Effective Tax Rate Comparison")
    print("-"*100)
    print(f"{'Filing Status':<30} {'Model':>12} {'Target':>12} {'Gap':>12}")
    print("-"*100)
    
    for _, row in comparison.iterrows():
        if pd.notna(row['effective_rate_model']) and pd.notna(row['effective_rate_target']):
            print(f"{row['filing_status']:<30} {row['effective_rate_model']:>11.2f}% {row['effective_rate_target']:>11.2f}% "
                  f"{row['effective_rate_gap']:>11.2f}pp")
    
    print("-"*100)
    model_overall = model_total_tax / model_total_agi * 100
    target_overall = target_total_tax / target_total_agi * 100
    print(f"{'OVERALL':<30} {model_overall:>11.2f}% {target_overall:>11.2f}% {model_overall - target_overall:>11.2f}pp")
    
    # Assessment
    print("\n" + "="*100)
    print("ASSESSMENT")
    print("="*100)
    
    total_tax_gap_pct = (model_total_tax / target_total_tax - 1) * 100
    
    if abs(total_tax_gap_pct) < 2:
        status = "✅ EXCELLENT"
    elif abs(total_tax_gap_pct) < 5:
        status = "✅ VERY GOOD"
    elif abs(total_tax_gap_pct) < 10:
        status = "✓ GOOD"
    elif abs(total_tax_gap_pct) < 20:
        status = "⚠️ MODERATE"
    else:
        status = "❌ NEEDS IMPROVEMENT"
    
    print(f"\nOverall Tax Revenue Gap: {total_tax_gap_pct:+.1f}% - {status}")
    
    # Identify largest gaps
    print("\nGaps by Filing Status (Tax Revenue):")
    gaps = comparison[['filing_status', 'tax_gap_pct']].copy()
    gaps = gaps.dropna()
    gaps['abs_gap'] = gaps['tax_gap_pct'].abs()
    gaps = gaps.sort_values('abs_gap', ascending=False)
    
    for _, row in gaps.iterrows():
        if abs(row['tax_gap_pct']) < 5:
            marker = "✅"
        elif abs(row['tax_gap_pct']) < 10:
            marker = "✓"
        elif abs(row['tax_gap_pct']) < 20:
            marker = "⚠️"
        else:
            marker = "❌"
        print(f"  {marker} {row['filing_status']}: {row['tax_gap_pct']:+.1f}%")


def main():
    """Main validation workflow."""
    
    logger.info("="*100)
    logger.info("REFINED MODEL VALIDATION AGAINST 2022 DOTAX SOI TARGETS")
    logger.info("="*100)
    
    # Load 2022 DOTAX targets
    targets_2022 = load_2022_dotax_targets()
    
    # Load refined model
    model_refined = load_refined_model()
    model_agg = aggregate_model_data(model_refined)
    
    # Compare
    comparison = compare_model_vs_targets(model_agg, targets_2022)
    
    # Print report
    print_validation_report(comparison, "HoH-Refined Model")
    
    # Also load and compare original model for reference
    logger.info("\n" + "="*100)
    logger.info("LOADING ORIGINAL MODEL FOR COMPARISON")
    logger.info("="*100)
    
    project_root = Path(__file__).parent.parent.parent
    original_file = project_root / 'data' / 'processed' / 'tax_units_systematically_calibrated_20251103_085550.parquet'
    
    if original_file.exists():
        logger.info(f"Loading original model: {original_file.name}")
        model_original = pd.read_parquet(original_file)
        original_agg = aggregate_model_data(model_original)
        comparison_original = compare_model_vs_targets(original_agg, targets_2022)
        
        print("\n" + "="*100)
        print("FOR COMPARISON:")
        print("="*100)
        print_validation_report(comparison_original, "Original Model (Before Refinement)")
    
    # Save comparison
    output_dir = project_root / 'data' / 'processed' / 'validation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison.to_csv(output_dir / 'refined_model_vs_2022_dotax.csv', index=False)
    logger.info(f"\nValidation results saved to {output_dir}")


if __name__ == "__main__":
    main()
