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
import argparse

from src.tax.deductions import TaxableIncomeCalculator, DeductionPolicy
from src.tax.deductions.calculator import estimate_num_exemptions
from src.tax.deductions.parsers import clean_currency
from src.tax.units.status.irs_based import adjust_high_income_statuses
from typing import Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_nontaxable_targets_from_table() -> Dict[str, float]:
    """Load nontaxable return counts by filing status from DOTAX Table A2-1."""
    table_path = Path('data/raw/Selected Resident Return Data/Dotax Soi 2022 - A2-1.csv')
    if not table_path.exists():
        logger.warning("DOTAX Table A2-1 not found; falling back to formula-based thresholds")
        return {}

    try:
        df = pd.read_csv(table_path, header=None)
    except Exception as exc:
        logger.warning(f"Unable to read DOTAX Table A2-1: {exc}; falling back to formula thresholds")
        return {}

    match = df[df.iloc[:, 0] == 'TOTAL RESIDENT NONTAXABLE']
    if match.empty:
        logger.warning("TOTAL RESIDENT NONTAXABLE row not found in Table A2-1; using formula thresholds")
        return {}

    row = match.iloc[0]

    def parse(value):
        parsed = clean_currency(value)
        return float(parsed) if pd.notna(parsed) else 0.0

    return {
        'joint': parse(row[4]),
        'single_mfs': parse(row[8]),
        'hoh': parse(row[12])
    }


def derive_threshold_from_targets(
    tax_units: pd.DataFrame,
    status_mask: pd.Series,
    target_weight: float,
    weight_col: str
) -> float:
    """Compute AGI threshold that captures the target weighted count."""
    if target_weight <= 0:
        return 0.0

    subset = tax_units.loc[status_mask, ['agi', weight_col]].dropna(subset=['agi'])
    if subset.empty:
        return 0.0

    subset = subset.sort_values('agi')
    subset['cum_weight'] = subset[weight_col].cumsum()
    cutoff = subset[subset['cum_weight'] >= target_weight]

    threshold = cutoff.iloc[0]['agi'] if not cutoff.empty else subset['agi'].max()
    # Round to nearest $100 for stability
    return round(float(threshold) / 100) * 100

def assign_forced_deductions(
    tax_units: pd.DataFrame,
    deduction_benchmarks: pd.DataFrame,
    weight_col: str
) -> pd.Series:
    """Deterministically assign deduction types to match DOTAX itemization targets."""
    forced = pd.Series('standard', index=tax_units.index, dtype='object')

    if deduction_benchmarks is None or deduction_benchmarks.empty:
        return forced

    agi_series = tax_units['agi']

    for _, row in deduction_benchmarks.iterrows():
        agi_min = row.get('agi_min', 0)
        agi_max = row.get('agi_max', float('inf'))
        total_returns = row.get('total_returns', 0)
        itemized_returns = row.get('itemized_returns', 0)

        if total_returns <= 0:
            continue

        if np.isfinite(agi_max):
            mask = (agi_series >= agi_min) & (agi_series < agi_max)
        else:
            mask = agi_series >= agi_min

        subset = tax_units.loc[mask, ['agi', weight_col]].copy()
        if subset.empty:
            continue

        total_weight = subset[weight_col].sum()
        if total_weight <= 0:
            continue

        target_share = max(0.0, min(1.0, itemized_returns / total_returns))
        if target_share == 0:
            continue

        # Apply HoH-specific adjustment to reduce over-itemization
        # HoH filers should itemize less frequently than joint/single in mid-income brackets
        hoh_mask = subset.index.isin(tax_units[
            tax_units['filing_status'].str.lower().isin([
                'head_of_household', 'qualifying_widow', 'qualifying_widower'
            ])
        ].index)
        
        if hoh_mask.any() and agi_min >= 20000 and agi_max <= 100000:
            # Reduce HoH itemization target by 25-40% in mid-income brackets
            hoh_reduction = 0.35 if agi_min < 50000 else 0.25
            adjusted_target_share = target_share * (1 - hoh_reduction)
            
            # Calculate separate targets for HoH and non-HoH
            hoh_subset = subset.loc[hoh_mask]
            non_hoh_subset = subset.loc[~hoh_mask]
            
            hoh_weight = hoh_subset[weight_col].sum()
            non_hoh_weight = non_hoh_subset[weight_col].sum()
            
            # HoH gets reduced target, non-HoH gets standard target
            hoh_target_weight = hoh_weight * adjusted_target_share
            non_hoh_target_weight = non_hoh_weight * target_share
            
            target_weight = hoh_target_weight + non_hoh_target_weight
        else:
            target_weight = min(total_weight, total_weight * target_share)

        subset = subset.sort_values(['agi', weight_col], ascending=[False, False])
        subset['cum_weight'] = subset[weight_col].cumsum()

        itemized_idx = subset.index[subset['cum_weight'] <= target_weight]

        if itemized_idx.empty:
            first_idx = subset.head(1).index
            itemized_idx = itemized_idx.union(first_idx)
        else:
            reached = subset.loc[itemized_idx, 'cum_weight'].max()
            if reached < target_weight:
                next_idx = subset[subset['cum_weight'] > target_weight].head(1).index
                itemized_idx = itemized_idx.union(next_idx)

        forced.loc[itemized_idx] = 'itemized'

    return forced




def get_nontaxable_thresholds(tax_units: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """
    Get AGI thresholds for nontaxable returns by filing status.
    Based on Hawaii tax brackets and standard deductions for 2022.
    
    Returns:
        Dictionary mapping filing status to AGI threshold for nontaxable returns
    """
    if tax_units is not None:
        targets = load_nontaxable_targets_from_table()
        if targets:
            thresholds: Dict[str, float] = {}
            weight_col = 'weight_revenue_calibrated' if 'weight_revenue_calibrated' in tax_units.columns else 'weight'

            if weight_col not in tax_units.columns:
                tax_units = tax_units.copy()
                tax_units[weight_col] = 1.0

            normalized_status = tax_units['filing_status'].astype(str).str.lower()
            normalized_status = normalized_status.replace({'nan': ''})

            # Joint threshold
            joint_mask = normalized_status == 'married_filing_jointly'
            thresholds['joint'] = derive_threshold_from_targets(
                tax_units, joint_mask, targets.get('joint', 0), weight_col
            )

            # Head of household (includes qualifying widow(er))
            hoh_mask = normalized_status.isin(['head_of_household', 'qualifying_widow', 'qualifying_widower'])
            thresholds['hoh'] = derive_threshold_from_targets(
                tax_units, hoh_mask, targets.get('hoh', 0), weight_col
            )

            # Single + MFS share a combined target in Table A2-1
            single_mask = normalized_status.isin(['single', 'married_filing_separately'])
            single_threshold = derive_threshold_from_targets(
                tax_units, single_mask, targets.get('single_mfs', 0), weight_col
            )
            thresholds['single'] = single_threshold
            thresholds['mfs'] = single_threshold

            # Ensure defaults for any status not computed
            for status in ['single', 'joint', 'hoh', 'mfs']:
                thresholds.setdefault(status, 0.0)

            return thresholds

    # Fallback to formula-based thresholds if data-driven approach is unavailable
    standard_deductions = {
        'single': 2200,
        'joint': 4400,
        'hoh': 3200,
        'mfs': 2200
    }
    personal_exemption = 1144

    thresholds = {
        'single': standard_deductions['single'] + personal_exemption,
        'joint': standard_deductions['joint'] + (2 * personal_exemption),
        'hoh': standard_deductions['hoh'] + (1.5 * personal_exemption),
        'mfs': standard_deductions['mfs'] + personal_exemption
    }

    thresholds = {k: v * 1.2 for k, v in thresholds.items()}
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate taxable income for all tax units')
    parser.add_argument('--high-income-tweak', action='store_true', 
                       help='Apply high-income filing status fine-tuning to improve revenue accuracy')
    parser.add_argument('--agi-threshold', type=float, default=200_000,
                       help='AGI threshold for high-income adjustment (default: $200,000)')
    args = parser.parse_args()
    
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

    # Ensure HoH units retain at least one dependent for deduction/exemption logic
    normalized_status = tax_units['filing_status'].astype(str).str.lower()
    if 'num_dependents' not in tax_units.columns:
        tax_units['num_dependents'] = 0
    missing_dependents_mask = normalized_status.isin([
        'head_of_household',
        'qualifying_widow',
        'qualifying_widower'
    ]) & (tax_units['num_dependents'].fillna(0) < 1)
    if missing_dependents_mask.any():
        logger.info(
            f"\nAdjusting HoH dependents to minimum of 1 for {missing_dependents_mask.sum():,} units"
        )
        tax_units.loc[missing_dependents_mask, 'num_dependents'] = 1

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
    nontaxable_thresholds = get_nontaxable_thresholds(tax_units)
    logger.info("\nNontaxable thresholds (AGI):")
    for status, threshold in nontaxable_thresholds.items():
        logger.info(f"  {status.upper()}: ${threshold:,.0f}")
    
    # Initialize calculator with Hawaii deduction benchmarks
    logger.info("\nInitializing taxable income calculator...")
    deduction_benchmarks_path = Path('data/processed/deduction_benchmarks.csv')
    deduction_benchmarks_df = pd.read_csv(deduction_benchmarks_path)

    logger.info('Assigning deduction types to match DOTAX itemization targets...')
    weight_col = 'weight' if 'weight' in tax_units.columns else ('weight_revenue_calibrated' if 'weight_revenue_calibrated' in tax_units.columns else 'weight')
    tax_units['forced_deduction_type'] = assign_forced_deductions(
        tax_units, deduction_benchmarks_df, weight_col
    )

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

    # Apply high-income fine-tuning if requested
    if args.high_income_tweak:
        logger.info("\nApplying high-income filing status fine-tuning...")
        results = adjust_high_income_statuses(
            results, 
            weight_col='weight',
            agi_threshold=args.agi_threshold
        )

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
