"""
Build complete Hawaii tax unit population: Residents + Nonresidents.

This script:
1. Loads PUMS data and constructs resident tax units
2. Synthesizes nonresident tax units from DOTAX Table 17A
3. Combines both populations
4. Validates against DOTAX SOI totals (743,109 returns)
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.pums_loader import PUMSDataLoader
from src.tax.units.constructor import TaxUnitConstructor
from src.tax.units.nonresident_synthesizer import (
    NonresidentSynthesizer,
    combine_resident_nonresident_units
)
from src.tax.calibration.dotax_soi_parser import DOTAXSOIParser


def build_full_population(save_output: bool = True) -> pd.DataFrame:
    """
    Build complete Hawaii tax unit population using SOI-calibrated resident units
    and DOTAX-based nonresident units.
    
    Returns:
        DataFrame with all tax units (residents + nonresidents)
        
    Note:
        - Resident units are constructed from PUMS data with SOI calibration
        - Nonresident units are synthesized from DOTAX Table 17A
        - Combined total matches DOTAX SOI 2022 total of 742,948 returns
    """
    print("="*80)
    print("BUILDING FULL HAWAII TAX UNIT POPULATION")
    print("="*80)
    
    # Step 1: Load PUMS and construct resident tax units with SOI calibration
    print("\n### Step 1: Constructing Resident Tax Units from PUMS with SOI Calibration ###")
    loader = PUMSDataLoader(year=2022, state='HI')
    
    # Load person and household data
    person_data = loader.load_data()
    hh_data = loader.load_households()
    
    # Load SOI benchmarks for calibration
    print("Loading SOI benchmarks for calibration...")
    from src.tax.units.soi_calibration import load_dotax_benchmarks, load_irs_benchmarks
    dotax_benchmarks = load_dotax_benchmarks()
    irs_benchmarks = load_irs_benchmarks()
    
    # Initialize constructor with SOI calibration
    print("Initializing TaxUnitConstructor with SOI calibration...")
    constructor = TaxUnitConstructor(
        person_df=person_data,
        hh_df=hh_data,
        use_soi_calibration=True,
        soi_calibration_method='filing_status',
        dotax_benchmarks=dotax_benchmarks,
        irs_benchmarks=irs_benchmarks
    )
    
    # Create tax units with SOI calibration
    resident_units = constructor.create_rule_based_units()
    
    print(f"✅ Created {len(resident_units):,} resident tax units")
    print(f"   Weighted total: {resident_units['weight'].sum():,.0f}")
    
    # Step 2: Synthesize nonresident tax units
    print("\n### Step 2: Synthesizing Nonresident Tax Units ###")
    synthesizer = NonresidentSynthesizer(data_dir='data/raw')
    
    # Get target number of nonresident returns from DOTAX benchmarks
    parser = DOTAXSOIParser(data_dir='data/raw')
    combined_totals = parser.get_combined_totals()
    target_nonresident_returns = int(combined_totals['nonresident_returns'])
    
    # Scale down for synthesis (we'll weight them back up)
    sample_size = min(10000, target_nonresident_returns)
    nonresident_units = synthesizer.synthesize_nonresident_units(num_samples=sample_size)
    
    # Adjust weights to match target nonresident count
    nonresident_units['weight'] = nonresident_units['weight'] * (target_nonresident_returns / sample_size)
    
    print(f"✅ Created {len(nonresident_units):,} nonresident tax units")
    print(f"   Weighted total: {nonresident_units['weight'].sum():,.0f}")
    
    # Step 3: Combine populations
    print("\n### Step 3: Combining Populations ###")
    
    # Ensure both dataframes have the same columns
    common_columns = list(set(resident_units.columns) & set(nonresident_columns))
    resident_units = resident_units[common_columns]
    nonresident_units = nonresident_units[common_columns]
    
    full_population = combine_resident_nonresident_units(resident_units, nonresident_units)
    
    # Add is_resident flag if not present
    if 'is_resident' not in full_population.columns:
        full_population['is_resident'] = full_population.index.isin(resident_units.index)
    
    print(f"✅ Combined population: {len(full_population):,} tax units")
    print(f"   Weighted total: {full_population['weight'].sum():,.0f}")
    
    # Step 4: Validate against DOTAX totals
    print("\n### Step 4: Validation Against DOTAX SOI ###")
    validate_full_population(full_population)
    
    # Step 5: Save output
    if save_output:
        output_path = project_root / 'data' / 'processed' / 'full_tax_units.parquet'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        full_population.to_parquet(output_path, index=False)
        print(f"\n✅ Saved to: {output_path}")
    
    return full_population


def validate_full_population(full_population: pd.DataFrame):
    """Validate combined population against DOTAX SOI totals."""
    parser = DOTAXSOIParser(data_dir='data/raw')
    combined_totals = parser.get_combined_totals()
    
    # Total returns
    model_total = full_population['weight'].sum()
    target_total = combined_totals['total_returns']
    
    print(f"\nTotal Returns:")
    print(f"  Model:  {model_total:,.0f}")
    print(f"  Target: {target_total:,}")
    print(f"  Diff:   {(model_total - target_total) / target_total * 100:+.1f}%")
    
    # Resident vs Nonresident split
    resident_count = full_population[full_population['is_resident']]['weight'].sum()
    nonresident_count = full_population[~full_population['is_resident']]['weight'].sum()
    
    print(f"\nResident Returns:")
    print(f"  Model:  {resident_count:,.0f}")
    print(f"  Target: {combined_totals['resident_returns']:,}")
    print(f"  Diff:   {(resident_count - combined_totals['resident_returns']) / combined_totals['resident_returns'] * 100:+.1f}%")
    
    print(f"\nNonresident Returns:")
    print(f"  Model:  {nonresident_count:,.0f}")
    print(f"  Target: {combined_totals['nonresident_returns']:,}")
    print(f"  Diff:   {(nonresident_count - combined_totals['nonresident_returns']) / combined_totals['nonresident_returns'] * 100:+.1f}%")
    
    print(f"\nNonresident Percentage:")
    model_pct = nonresident_count / model_total
    target_pct = combined_totals['pct_nonresident']
    print(f"  Model:  {model_pct:.2%}")
    print(f"  Target: {target_pct:.2%}")
    
    # Filing status distribution (combined)
    print(f"\n### Filing Status Distribution (Combined) ###")
    filing_dist = full_population.groupby('filing_status')['weight'].sum()
    filing_dist_pct = filing_dist / filing_dist.sum()
    
    # Get resident-only targets (we don't have combined targets)
    resident_totals = parser.get_resident_totals()
    
    print(f"{'Status':<10} {'Model Count':>15} {'Model %':>10}")
    print("-" * 40)
    for status in ['single', 'joint', 'mfs', 'hoh']:
        if status in filing_dist.index:
            count = filing_dist[status]
            pct = filing_dist_pct[status]
            print(f"{status:<10} {count:>15,.0f} {pct:>9.1%}")
    
    # Tax revenue
    if 'tax_after_credits' in full_population.columns:
        model_tax = (full_population['tax_after_credits'] * full_population['weight']).sum()
        target_tax = combined_totals['total_tax_after'] * 1_000_000  # Convert from millions
        
        print(f"\nTotal Tax Revenue (After Credits):")
        print(f"  Model:  ${model_tax / 1e6:,.0f}M")
        print(f"  Target: ${target_tax / 1e6:,.0f}M")
        print(f"  Diff:   {(model_tax - target_tax) / target_tax * 100:+.1f}%")


def analyze_resident_nonresident_differences(full_population: pd.DataFrame):
    """Analyze differences between resident and nonresident populations."""
    print("\n" + "="*80)
    print("RESIDENT VS NONRESIDENT ANALYSIS")
    print("="*80)
    
    residents = full_population[full_population['is_resident']]
    nonresidents = full_population[~full_population['is_resident']]
    
    # Average AGI
    res_avg_agi = (residents['agi'] * residents['weight']).sum() / residents['weight'].sum()
    nonres_avg_agi = (nonresidents['agi'] * nonresidents['weight']).sum() / nonresidents['weight'].sum()
    
    print(f"\n### Average AGI ###")
    print(f"Residents:    ${res_avg_agi:,.0f}")
    print(f"Nonresidents: ${nonres_avg_agi:,.0f}")
    print(f"Ratio:        {nonres_avg_agi / res_avg_agi:.2f}x")
    
    # Average tax
    if 'tax_after_credits' in full_population.columns:
        res_avg_tax = (residents['tax_after_credits'] * residents['weight']).sum() / residents['weight'].sum()
        nonres_avg_tax = (nonresidents['tax_after_credits'] * nonresidents['weight']).sum() / nonresidents['weight'].sum()
        
        print(f"\n### Average Tax (After Credits) ###")
        print(f"Residents:    ${res_avg_tax:,.0f}")
        print(f"Nonresidents: ${nonres_avg_tax:,.0f}")
        print(f"Ratio:        {nonres_avg_tax / res_avg_tax:.2f}x")
    
    # Filing status distribution
    print(f"\n### Filing Status Distribution ###")
    res_filing = residents.groupby('filing_status')['weight'].sum()
    nonres_filing = nonresidents.groupby('filing_status')['weight'].sum()
    
    res_filing_pct = res_filing / res_filing.sum()
    nonres_filing_pct = nonres_filing / nonres_filing.sum()
    
    print(f"{'Status':<10} {'Residents':>12} {'Nonresidents':>15}")
    print("-" * 40)
    for status in ['single', 'joint', 'mfs', 'hoh']:
        res_pct = res_filing_pct.get(status, 0)
        nonres_pct = nonres_filing_pct.get(status, 0)
        print(f"{status:<10} {res_pct:>11.1%} {nonres_pct:>14.1%}")


def main():
    """Build and analyze full population."""
    # Build full population
    full_population = build_full_population(save_output=True)
    
    # Analyze differences
    analyze_resident_nonresident_differences(full_population)
    
    print("\n" + "="*80)
    print("✅ FULL POPULATION BUILD COMPLETE")
    print("="*80)
    print(f"\nTotal tax units: {len(full_population):,}")
    print(f"Weighted returns: {full_population['weight'].sum():,.0f}")
    print(f"Target: 743,109")
    print(f"\nData saved to: data/processed/full_tax_units.parquet")


if __name__ == '__main__':
    main()
