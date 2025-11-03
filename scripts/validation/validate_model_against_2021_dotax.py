#!/usr/bin/env python3
"""
Validate Tax Model Against 2021 Actual DOTAX Data

This script compares the model's estimates/projections against actual 2021 DOTAX data
to assess model accuracy and identify systematic biases.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_2021_actual_data() -> pd.DataFrame:
    """Load actual 2021 DOTAX data."""
    project_root = Path(__file__).parent.parent.parent
    file_path = project_root / 'data' / 'processed' / 'dotax_historical' / 'dotax_historical_filing_status_clean.csv'
    
    df = pd.read_csv(file_path)
    df_2021 = df[df['year'] == 2021].copy()
    
    logger.info("Loaded 2021 actual DOTAX data")
    return df_2021


def load_model_2022_calibrated_data() -> pd.DataFrame:
    """Load the model's 2022 calibrated tax units."""
    project_root = Path(__file__).parent.parent.parent
    file_path = project_root / 'data' / 'processed' / 'tax_units_systematically_calibrated_20251103_085550.parquet'
    
    df = pd.read_parquet(file_path)
    
    logger.info(f"Loaded model's 2022 calibrated data: {len(df):,} tax units")
    return df


def standardize_filing_status(status: str) -> str:
    """Standardize filing status names for comparison."""
    status_map = {
        'single': 'Single',
        'married_filing_jointly': 'Married Filing Jointly',
        'married_filing_separately': 'Married Filing Separately',
        'head_of_household': 'Head of Household',
        'qualifying_widow': 'Qualifying Widow(er)'
    }
    return status_map.get(status.lower(), status)


def aggregate_model_data(model_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate model data by filing status to match DOTAX format."""
    
    # Standardize filing status names
    model_df['filing_status_standard'] = model_df['filing_status'].apply(standardize_filing_status)
    
    # Aggregate by filing status
    agg_dict = {
        'weight': 'sum',  # Total returns (weighted)
        'agi': lambda x: (x * model_df.loc[x.index, 'weight']).sum(),  # Weighted AGI sum
        'hi_state_tax': lambda x: (x * model_df.loc[x.index, 'weight']).sum(),  # Weighted tax sum
    }
    
    grouped = model_df.groupby('filing_status_standard').agg({
        'weight': 'sum',
        'filer_id': 'count'
    }).reset_index()
    
    # Calculate weighted sums manually to ensure correct weighting
    results = []
    for status in model_df['filing_status_standard'].unique():
        status_data = model_df[model_df['filing_status_standard'] == status]
        
        result = {
            'filing_status': status,
            'num_returns': status_data['weight'].sum(),
            'agi_millions': (status_data['agi'] * status_data['weight']).sum() / 1_000_000,
            'tax_millions': (status_data['hi_state_tax'] * status_data['weight']).sum() / 1_000_000,
            'unweighted_count': len(status_data)
        }
        results.append(result)
    
    model_agg = pd.DataFrame(results)
    
    # Calculate derived metrics
    model_agg['effective_rate'] = (model_agg['tax_millions'] / model_agg['agi_millions']) * 100
    model_agg['avg_agi_per_return'] = (model_agg['agi_millions'] * 1_000_000) / model_agg['num_returns']
    model_agg['avg_tax_per_return'] = (model_agg['tax_millions'] * 1_000_000) / model_agg['num_returns']
    
    return model_agg


def back_project_to_2021(model_2022: pd.DataFrame, cagr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Back-project model's 2022 estimates to 2021 using historical CAGR.
    
    Assumes 2022 = 2021 × (1 + CAGR)
    Therefore: 2021 = 2022 / (1 + CAGR)
    """
    logger.info("Back-projecting model's 2022 data to 2021 using historical CAGR...")
    
    model_2021 = model_2022.copy()
    
    # Map filing status names
    status_map = {
        'Single': 'Single',
        'Married Filing Jointly': 'Married Filing Jointly',
        'Married Filing Separately': 'Married Fil. Separately',
        'Head of Household': 'Head of Household',
        'Qualifying Widow(er)': 'Qualifying Widow(er)'
    }
    
    for idx, row in model_2021.iterrows():
        status = row['filing_status']
        cagr_status = status_map.get(status, status)
        
        # Get CAGR for this filing status
        cagr_row = cagr_df[cagr_df['filing_status'] == cagr_status]
        
        if len(cagr_row) > 0:
            returns_cagr = cagr_row.iloc[0]['returns_cagr'] / 100
            agi_cagr = cagr_row.iloc[0]['agi_cagr'] / 100
            tax_cagr = cagr_row.iloc[0]['tax_cagr'] / 100
            
            # Back-project: 2021 = 2022 / (1 + CAGR)
            model_2021.at[idx, 'num_returns'] = row['num_returns'] / (1 + returns_cagr)
            model_2021.at[idx, 'agi_millions'] = row['agi_millions'] / (1 + agi_cagr)
            model_2021.at[idx, 'tax_millions'] = row['tax_millions'] / (1 + tax_cagr)
            
            # Recalculate derived metrics
            model_2021.at[idx, 'effective_rate'] = (model_2021.at[idx, 'tax_millions'] / model_2021.at[idx, 'agi_millions']) * 100
            model_2021.at[idx, 'avg_agi_per_return'] = (model_2021.at[idx, 'agi_millions'] * 1_000_000) / model_2021.at[idx, 'num_returns']
            model_2021.at[idx, 'avg_tax_per_return'] = (model_2021.at[idx, 'tax_millions'] * 1_000_000) / model_2021.at[idx, 'num_returns']
        else:
            logger.warning(f"No CAGR data found for {status}, using 2022 values as-is")
    
    return model_2021


def compare_model_vs_actual(model_df: pd.DataFrame, actual_df: pd.DataFrame) -> pd.DataFrame:
    """Compare model estimates against actual DOTAX data."""
    
    # Prepare actual data
    actual_compare = actual_df[['filing_status', 'num_returns', 'agi_net_millions', 
                                 'tax_after_credits_millions', 'effective_rate_after_credits',
                                 'avg_agi_per_return', 'avg_tax_per_return']].copy()
    actual_compare = actual_compare.rename(columns={
        'agi_net_millions': 'agi_millions',
        'tax_after_credits_millions': 'tax_millions',
        'effective_rate_after_credits': 'effective_rate'
    })
    
    # Merge model and actual
    comparison = model_df.merge(
        actual_compare,
        on='filing_status',
        how='outer',
        suffixes=('_model', '_actual')
    )
    
    # Calculate gaps
    comparison['returns_gap'] = comparison['num_returns_model'] - comparison['num_returns_actual']
    comparison['returns_gap_pct'] = (comparison['num_returns_model'] / comparison['num_returns_actual'] - 1) * 100
    
    comparison['agi_gap'] = comparison['agi_millions_model'] - comparison['agi_millions_actual']
    comparison['agi_gap_pct'] = (comparison['agi_millions_model'] / comparison['agi_millions_actual'] - 1) * 100
    
    comparison['tax_gap'] = comparison['tax_millions_model'] - comparison['tax_millions_actual']
    comparison['tax_gap_pct'] = (comparison['tax_millions_model'] / comparison['tax_millions_actual'] - 1) * 100
    
    comparison['effective_rate_gap'] = comparison['effective_rate_model'] - comparison['effective_rate_actual']
    
    return comparison


def print_validation_report(comparison: pd.DataFrame, title: str = "Model Validation Report"):
    """Print formatted validation report."""
    
    print("\n" + "="*100)
    print(f"{title}")
    print("="*100)
    
    # Summary by filing status
    print("\n### Returns Comparison")
    print("-"*100)
    print(f"{'Filing Status':<30} {'Model':>12} {'Actual':>12} {'Gap':>12} {'Gap %':>10}")
    print("-"*100)
    
    for _, row in comparison.iterrows():
        print(f"{row['filing_status']:<30} {row['num_returns_model']:>12,.0f} {row['num_returns_actual']:>12,.0f} "
              f"{row['returns_gap']:>12,.0f} {row['returns_gap_pct']:>9.1f}%")
    
    print("-"*100)
    print(f"{'TOTAL':<30} {comparison['num_returns_model'].sum():>12,.0f} "
          f"{comparison['num_returns_actual'].sum():>12,.0f} "
          f"{comparison['returns_gap'].sum():>12,.0f} "
          f"{(comparison['num_returns_model'].sum() / comparison['num_returns_actual'].sum() - 1) * 100:>9.1f}%")
    
    # AGI Comparison
    print("\n### AGI Comparison ($M)")
    print("-"*100)
    print(f"{'Filing Status':<30} {'Model':>12} {'Actual':>12} {'Gap':>12} {'Gap %':>10}")
    print("-"*100)
    
    for _, row in comparison.iterrows():
        print(f"{row['filing_status']:<30} ${row['agi_millions_model']:>11,.1f} ${row['agi_millions_actual']:>11,.1f} "
              f"${row['agi_gap']:>11,.1f} {row['agi_gap_pct']:>9.1f}%")
    
    print("-"*100)
    print(f"{'TOTAL':<30} ${comparison['agi_millions_model'].sum():>11,.1f} "
          f"${comparison['agi_millions_actual'].sum():>11,.1f} "
          f"${comparison['agi_gap'].sum():>11,.1f} "
          f"{(comparison['agi_millions_model'].sum() / comparison['agi_millions_actual'].sum() - 1) * 100:>9.1f}%")
    
    # Tax Revenue Comparison
    print("\n### Tax Revenue Comparison ($M)")
    print("-"*100)
    print(f"{'Filing Status':<30} {'Model':>12} {'Actual':>12} {'Gap':>12} {'Gap %':>10}")
    print("-"*100)
    
    for _, row in comparison.iterrows():
        print(f"{row['filing_status']:<30} ${row['tax_millions_model']:>11,.1f} ${row['tax_millions_actual']:>11,.1f} "
              f"${row['tax_gap']:>11,.1f} {row['tax_gap_pct']:>9.1f}%")
    
    print("-"*100)
    print(f"{'TOTAL':<30} ${comparison['tax_millions_model'].sum():>11,.1f} "
          f"${comparison['tax_millions_actual'].sum():>11,.1f} "
          f"${comparison['tax_gap'].sum():>11,.1f} "
          f"{(comparison['tax_millions_model'].sum() / comparison['tax_millions_actual'].sum() - 1) * 100:>9.1f}%")
    
    # Effective Rate Comparison
    print("\n### Effective Tax Rate Comparison")
    print("-"*100)
    print(f"{'Filing Status':<30} {'Model':>12} {'Actual':>12} {'Gap':>12}")
    print("-"*100)
    
    for _, row in comparison.iterrows():
        print(f"{row['filing_status']:<30} {row['effective_rate_model']:>11.2f}% {row['effective_rate_actual']:>11.2f}% "
              f"{row['effective_rate_gap']:>11.2f}pp")
    
    print("-"*100)
    model_overall = comparison['tax_millions_model'].sum() / comparison['agi_millions_model'].sum() * 100
    actual_overall = comparison['tax_millions_actual'].sum() / comparison['agi_millions_actual'].sum() * 100
    print(f"{'OVERALL':<30} {model_overall:>11.2f}% {actual_overall:>11.2f}% {model_overall - actual_overall:>11.2f}pp")
    
    # Assessment
    print("\n" + "="*100)
    print("ASSESSMENT")
    print("="*100)
    
    total_tax_gap_pct = (comparison['tax_millions_model'].sum() / comparison['tax_millions_actual'].sum() - 1) * 100
    
    if abs(total_tax_gap_pct) < 5:
        status = "✅ EXCELLENT"
    elif abs(total_tax_gap_pct) < 10:
        status = "✓ GOOD"
    elif abs(total_tax_gap_pct) < 20:
        status = "⚠️ MODERATE"
    else:
        status = "❌ NEEDS IMPROVEMENT"
    
    print(f"\nOverall Tax Revenue Gap: {total_tax_gap_pct:+.1f}% - {status}")
    
    # Identify largest gaps
    print("\nLargest Gaps by Filing Status:")
    gaps = comparison[['filing_status', 'tax_gap_pct']].copy()
    gaps['abs_gap'] = gaps['tax_gap_pct'].abs()
    gaps = gaps.sort_values('abs_gap', ascending=False)
    
    for _, row in gaps.iterrows():
        if abs(row['tax_gap_pct']) > 10:
            print(f"  ⚠️ {row['filing_status']}: {row['tax_gap_pct']:+.1f}%")
        elif abs(row['tax_gap_pct']) > 5:
            print(f"  • {row['filing_status']}: {row['tax_gap_pct']:+.1f}%")


def main():
    """Main validation workflow."""
    
    logger.info("="*100)
    logger.info("MODEL VALIDATION AGAINST 2021 ACTUAL DOTAX DATA")
    logger.info("="*100)
    
    # Load actual 2021 DOTAX data
    actual_2021 = load_2021_actual_data()
    
    # Load model's 2022 calibrated data
    model_2022_raw = load_model_2022_calibrated_data()
    model_2022_agg = aggregate_model_data(model_2022_raw)
    
    # Load CAGR for back-projection
    project_root = Path(__file__).parent.parent.parent
    cagr_file = project_root / 'data' / 'processed' / 'dotax_historical' / 'dotax_historical_cagr.csv'
    cagr_df = pd.read_csv(cagr_file)
    
    # Back-project 2022 model to 2021
    model_2021 = back_project_to_2021(model_2022_agg, cagr_df)
    
    # Compare model vs actual
    comparison = compare_model_vs_actual(model_2021, actual_2021)
    
    # Print report
    print_validation_report(comparison, "2021 Model Back-Projection vs Actual DOTAX Data")
    
    # Also compare 2022 model directly (without back-projection)
    comparison_2022 = compare_model_vs_actual(model_2022_agg, actual_2021)
    print_validation_report(comparison_2022, "2022 Model (No Back-Projection) vs 2021 Actual")
    
    # Save comparison
    output_dir = project_root / 'data' / 'processed' / 'validation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison.to_csv(output_dir / 'model_vs_2021_actual_backprojected.csv', index=False)
    comparison_2022.to_csv(output_dir / 'model_2022_vs_2021_actual.csv', index=False)
    
    logger.info(f"\nValidation results saved to {output_dir}")


if __name__ == "__main__":
    main()
