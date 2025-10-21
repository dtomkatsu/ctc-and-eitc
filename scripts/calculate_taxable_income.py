#!/usr/bin/env python3
"""
Calculate Taxable Income for All Tax Units

Applies deductions and exemptions to AGI to calculate taxable income.
Uses benchmark assignment approach for accuracy.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import logging

from src.tax.deductions import TaxableIncomeCalculator, DeductionPolicy
from src.tax.deductions.calculator import estimate_num_exemptions
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def get_nontaxable_thresholds() -> Dict[str, float]:
    """
    Get AGI thresholds for nontaxable returns by filing status.
    Based on Hawaii tax brackets and standard deductions for 2022.
    
    Returns:
        Dictionary mapping filing status to AGI threshold for nontaxable returns
    """
    # Standard deductions for 2022
    standard_deductions = {
        'single': 2200,
        'joint': 4400,
        'hoh': 3200,
        'mfs': 2200  # Same as single for simplicity
    }
    
    # Personal exemption amount per person (2022)
    personal_exemption = 1144
    
    # Additional standard deduction for 65+ or blind (2022)
    additional_std_deduction = 1095
    
    # Calculate thresholds for each filing status
    thresholds = {}
    
    # Single (1 exemption)
    thresholds['single'] = standard_deductions['single'] + personal_exemption
    
    # Married Filing Jointly (2 exemptions)
    thresholds['joint'] = standard_deductions['joint'] + (2 * personal_exemption)
    
    # Head of Household (1.5 exemptions on average)
    thresholds['hoh'] = standard_deductions['hoh'] + (1.5 * personal_exemption)
    
    # Married Filing Separately (1 exemption)
    thresholds['mfs'] = standard_deductions['mfs'] + personal_exemption
    
    # Add 20% buffer to account for potential itemized deductions
    thresholds = {k: v * 1.2 for k, v in thresholds.items()}
    
    # Round to nearest $100 for simplicity
    return {k: round(v / 100) * 100 for k, v in thresholds.items()}

def estimate_exemptions_for_tax_units(tax_units_df, exemption_benchmarks):
    """
    Estimate number of exemptions for each tax unit.
    
    Uses household composition when available, otherwise AGI bracket averages.
    """
    logger.info("Estimating exemptions for tax units...")
    
    exemptions = []
    
    for _, tax_unit in tax_units_df.iterrows():
        num_exemptions = estimate_num_exemptions(tax_unit, exemption_benchmarks)
        exemptions.append(num_exemptions)
    
    return exemptions


def main():
    """Calculate taxable income for all tax units."""
    logger.info("="*80)
    logger.info("CALCULATING TAXABLE INCOME FOR ALL TAX UNITS")
    logger.info("="*80)
    
    # Load tax units with revenue-calibrated weights
    logger.info("\nLoading tax units...")
    tax_units = pd.read_parquet('data/processed/tax_units_revenue_calibrated.parquet')
    logger.info(f"  Loaded {len(tax_units):,} tax units")
    
    # Use revenue-calibrated weights for accuracy
    if 'weight_revenue_calibrated' in tax_units.columns:
        tax_units['weight'] = tax_units['weight_revenue_calibrated']
        logger.info(f"  Using revenue-calibrated weights")
        logger.info(f"  Total weighted returns: {tax_units['weight'].sum():,.0f}")
        logger.info(f"  Total AGI (weighted): ${(tax_units['agi'] * tax_units['weight']).sum()/1e9:.2f}B")
    else:
        # If weight_revenue_calibrated doesn't exist, create a weight column of 1s
        tax_units['weight'] = 1.0
        logger.info("  Using unweighted data (weight=1 for all records)")
    
    # Load exemption benchmarks for estimation
    exemption_benchmarks = pd.read_csv('data/processed/exemption_benchmarks.csv')
    
    # Estimate exemptions if not present
    if 'num_exemptions' not in tax_units.columns:
        logger.info("\nEstimating exemptions from household composition...")
        tax_units['num_exemptions'] = estimate_exemptions_for_tax_units(
            tax_units, exemption_benchmarks
        )
        logger.info(f"  Average exemptions per return: {tax_units['num_exemptions'].mean():.2f}")
    
    # Add age_65_plus_count if not present (simplified - would need age data from PUMS)
    if 'age_65_plus_count' not in tax_units.columns:
        tax_units['age_65_plus_count'] = 0  # Conservative estimate
    
    # Get nontaxable thresholds by filing status
    nontaxable_thresholds = get_nontaxable_thresholds()
    
    # Initialize calculator with Hawaii deduction benchmarks
    logger.info("\nInitializing taxable income calculator...")
    deduction_benchmarks_path = Path('data/processed/deduction_benchmarks.csv')
    calculator = TaxableIncomeCalculator.from_files(
        deduction_benchmarks_path=str(deduction_benchmarks_path)
    )
    
    # Calculate taxable income with nontaxable handling
    logger.info("\nCalculating taxable income with nontaxable return handling...")
    
    # Process tax units in batches to show progress
    total_units = len(tax_units)
    batch_size = 1000
    results = []
    
    for i in range(0, total_units, batch_size):
        batch = tax_units.iloc[i:i+batch_size].copy()
        batch_results = []
        
        for _, row in batch.iterrows():
            # Get filing status and AGI
            filing_status = str(row.get('filing_status', 'single')).lower()
            agi = float(row.get('agi', 0))
            
            # Check if return is likely nontaxable based on AGI threshold
            threshold = nontaxable_thresholds.get(filing_status, nontaxable_thresholds['single'])
            is_nontaxable = agi <= threshold
            
            if is_nontaxable:
                # For nontaxable returns, set taxable income to 0
                result = {
                    **row.to_dict(),
                    'taxable_income': 0.0,
                    'deduction': 0.0,
                    'deduction_type': 'nontaxable',
                    'exemption_amount': 0.0,
                    'is_nontaxable': True
                }
            else:
                # For taxable returns, use the calculator (preserve original row data including weights)
                result = {**row.to_dict(), **calculator.calculate(row.to_dict()), 'is_nontaxable': False}
            
            batch_results.append(result)
        
        # Add batch results to main results
        results.extend(batch_results)
        
        # Log progress
        processed = min(i + batch_size, total_units)
        logger.info(f"  Processed {processed:,} / {total_units:,} tax units")
    
    # Convert results to DataFrame
    results = pd.DataFrame(results)
    
    # Summary statistics
    logger.info("\n" + "="*80)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*80)
    
    # Calculate nontaxable statistics
    nontaxable_mask = results['is_nontaxable']
    total_weight = results['weight'].sum()
    
    nontaxable_count = nontaxable_mask.sum()
    nontaxable_pct = (nontaxable_count / len(results)) * 100
    
    # Handle case where there are no nontaxable returns
    if nontaxable_count > 0:
        nontaxable_weight = results.loc[nontaxable_mask, 'weight'].sum()
        nontaxable_weight_pct = (nontaxable_weight / total_weight) * 100
    else:
        nontaxable_weight = 0
        nontaxable_weight_pct = 0.0
    
    logger.info(f"\nNontaxable Returns:")
    logger.info(f"  Count: {nontaxable_count:,} ({nontaxable_pct:.1f}% of returns)")
    logger.info(f"  Weighted: {nontaxable_weight:,.0f} ({nontaxable_weight_pct:.1f}% of total weight)")
    
    # Income components for all returns
    logger.info(f"\nIncome Components (All Returns, Weighted):")
    logger.info(f"  Total AGI: ${(results['agi'] * results['weight']).sum()/1e9:.2f}B")
    logger.info(f"  Total Deductions: ${(results['deduction'] * results['weight']).sum()/1e9:.2f}B")
    logger.info(f"  Total Exemptions: ${(results['exemption_amount'] * results['weight']).sum()/1e9:.2f}B")
    logger.info(f"  Total Taxable Income: ${(results['taxable_income'] * results['weight']).sum()/1e9:.2f}B")
    
    # Income components for taxable returns only
    taxable_mask = ~nontaxable_mask
    if taxable_mask.any():
        taxable_results = results[taxable_mask].copy()
        taxable_weight = taxable_results['weight'].sum()
    else:
        taxable_results = pd.DataFrame()
        taxable_weight = 0
    
    if len(taxable_results) > 0:
        logger.info(f"\nIncome Components (Taxable Returns Only, Weighted):")
        logger.info(f"  Total AGI: ${(taxable_results['agi'] * taxable_results['weight']).sum()/1e9:.2f}B")
        logger.info(f"  Total Deductions: ${(taxable_results['deduction'] * taxable_results['weight']).sum()/1e9:.2f}B")
        logger.info(f"  Total Exemptions: ${(taxable_results['exemption_amount'] * taxable_results['weight']).sum()/1e9:.2f}B")
        logger.info(f"  Total Taxable Income: ${(taxable_results['taxable_income'] * taxable_results['weight']).sum()/1e9:.2f}B")
    
    # Calculate averages for all returns
    total_weight = results['weight'].sum()
    logger.info(f"\nAverage per Return (All Returns):")
    logger.info(f"  Avg AGI: ${(results['agi'] * results['weight']).sum() / total_weight:,.0f}")
    logger.info(f"  Avg Deduction: ${(results['deduction'] * results['weight']).sum() / total_weight:,.0f}")
    logger.info(f"  Avg Exemptions: ${(results['exemption_amount'] * results['weight']).sum() / total_weight:,.0f}")
    logger.info(f"  Avg Taxable Income: ${(results['taxable_income'] * results['weight']).sum() / total_weight:,.0f}")
    
    # Calculate averages for taxable returns only
    if len(taxable_results) > 0 and taxable_weight > 0:
        weight_col = 'weight_revenue_calibrated' if 'weight_revenue_calibrated' in taxable_results.columns else 'weight'
        taxable_avg_agi = (taxable_results['agi'] * taxable_results[weight_col]).sum() / taxable_weight
        taxable_avg_deduction = (taxable_results['deduction'] * taxable_results[weight_col]).sum() / taxable_weight
        taxable_avg_exemptions = (taxable_results['exemption_amount'] * taxable_results[weight_col]).sum() / taxable_weight
        taxable_avg_taxable_income = (taxable_results['taxable_income'] * taxable_results[weight_col]).sum() / taxable_weight
        
        logger.info(f"\nAverage per Return (Taxable Returns Only):")
        logger.info(f"  Avg AGI: ${taxable_avg_agi:,.0f}")
        logger.info(f"  Avg Deduction: ${taxable_avg_deduction:,.0f}")
        logger.info(f"  Avg Exemptions: ${taxable_avg_exemptions:,.0f}")
        logger.info(f"  Avg Taxable Income: ${taxable_avg_taxable_income:,.0f}")
    else:
        logger.info("\nNo taxable returns to calculate averages for.")
    
    # Deduction type distribution (taxable returns only)
    if len(taxable_results) > 0:
        deduction_type_counts = taxable_results['deduction_type'].value_counts()
        deduction_type_pct = deduction_type_counts / len(taxable_results) * 100
        
        deduction_type_weighted = taxable_results.groupby('deduction_type')['weight'].sum()
        deduction_type_weighted_pct = (deduction_type_weighted / taxable_results['weight'].sum()) * 100
        
        logger.info(f"\nDeduction Type Distribution (Taxable Returns Only):")
        for dtype in deduction_type_counts.index:
            if dtype != 'nontaxable':  # Skip nontaxable since we're already showing that separately
                logger.info(f"  {dtype.capitalize()}: {deduction_type_counts[dtype]:,} ({deduction_type_pct[dtype]:.1f}%)")
        
        logger.info("\nDeduction Type Distribution (Weighted, Taxable Returns Only):")
        for dtype in deduction_type_weighted.index:
            if dtype != 'nontaxable':  # Skip nontaxable since we're already showing that separately
                logger.info(f"  {dtype.capitalize()}: {deduction_type_weighted[dtype]:,.0f} ({deduction_type_weighted_pct[dtype]:.1f}%)")
    
    # Nontaxable by filing status
    if 'filing_status' in results.columns:
        logger.info("\nNontaxable Returns by Filing Status (Weighted):")
        status_counts = results[results['is_nontaxable']].groupby('filing_status')['weight'].sum()
        status_pct = (status_counts / status_counts.sum()) * 100
        
        for status, count in status_counts.items():
            logger.info(f"  {status.upper()}: {count:,.0f} ({status_pct[status]:.1f}% of nontaxable returns)")
        
        # Distribution by filing status (taxable returns only)
        if 'filing_status' in results.columns and not taxable_results.empty:
            logger.info("\nBy Filing Status (Taxable Returns Only):")
            weight_col = 'weight_revenue_calibrated' if 'weight_revenue_calibrated' in results.columns else 'weight'
            
            for status in results[taxable_mask]['filing_status'].unique():
                status_mask = (results['filing_status'] == status) & (taxable_mask)
                status_df = results[status_mask]
                status_weight = status_df[weight_col].sum()
                
                logger.info(f"\n  {status.upper()}:")
                logger.info(f"    Count: {len(status_df):,}")
                logger.info(f"    Weighted: {status_weight:,.0f} returns")
                
                if status_weight > 0:
                    avg_agi = (status_df['agi'] * status_df[weight_col]).sum() / status_weight
                    avg_deduction = (status_df['deduction'] * status_df[weight_col]).sum() / status_weight
                    avg_taxable_income = (status_df['taxable_income'] * status_df[weight_col]).sum() / status_weight
                    
                    logger.info(f"    Avg AGI: ${avg_agi:,.0f}")
                    logger.info(f"    Avg Deduction: ${avg_deduction:,.0f}")
                    logger.info(f"    Avg Taxable Income: ${avg_taxable_income:,.0f}")
                    
                    if 'deduction_type' in status_df.columns and 'itemized' in status_df['deduction_type'].values:
                        itemized_count = (status_df['deduction_type'] == 'itemized').sum()
                        itemized_pct = (itemized_count / len(status_df)) * 100
                        logger.info(f"    Itemize rate: {itemized_pct:.1f}%")
                else:
                    logger.info("    No weighted returns for this status")
    
    # Save results
    output_path = Path('data/processed/tax_units_with_taxable_income.parquet')
    results.to_parquet(output_path)
    
    logger.info("\n" + "="*80)
    logger.info(f"✅ Saved results to: {output_path}")
    logger.info("="*80)
    
    logger.info("\nNext Steps:")
    logger.info("  1. Integrate with Hawaii tax calculator to compute tax liability")
    logger.info("  2. Compare total revenue to SOI benchmarks")
    logger.info("  3. Run policy scenarios to model revenue impacts")
    
    return results


if __name__ == '__main__':
    main()
