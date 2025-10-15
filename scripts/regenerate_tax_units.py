#!/usr/bin/env python3
"""
Regenerate Tax Units with Updated MFS Logic

This script regenerates the tax units file with:
1. MFS filers included (updated thresholds)
2. Strict married couple identification
3. All filing statuses aligned to DOTAX 2022 benchmarks
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import logging
from datetime import datetime

from src.tax.units.constructor import TaxUnitConstructor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    """Main execution."""
    logger.info("="*80)
    logger.info("REGENERATING TAX UNITS WITH UPDATED LOGIC")
    logger.info("="*80)
    
    # Load PUMS data
    logger.info("\nLoading PUMS data...")
    persons = pd.read_csv('data/raw/pums/psam_p15.csv')
    households = pd.read_csv('data/raw/pums/psam_h15.csv')
    
    logger.info(f"  Loaded {len(persons):,} persons")
    logger.info(f"  Loaded {len(households):,} households")
    
    # Initialize constructor
    logger.info("\nInitializing TaxUnitConstructor...")
    constructor = TaxUnitConstructor(
        person_df=persons,
        hh_df=households,
        use_soi_calibration=False  # We'll validate manually
    )
    
    # Create tax units
    logger.info("\nCreating tax units...")
    tax_units = constructor.create_rule_based_units()
    
    logger.info(f"\n✅ Created {len(tax_units):,} tax units")
    
    # Analyze filing status distribution
    logger.info("\n" + "="*80)
    logger.info("FILING STATUS DISTRIBUTION")
    logger.info("="*80)
    
    # Determine weight column
    weight_col = 'weight' if 'weight' in tax_units.columns else 'PWGTP'
    
    status_counts = tax_units.groupby('filing_status')[weight_col].sum()
    total_filers = status_counts.sum()
    
    print("\nStatus                    | Count      | % of Total | DOTAX Target | Gap")
    print("--------------------------|------------|------------|--------------|--------")
    
    dotax_targets = {
        'single': (335198, 0.528),
        'married_filing_jointly': (216358, 0.341),
        'head_of_household': (67393, 0.106),
        'married_filing_separately': (16007, 0.025)
    }
    
    for status, (target_count, target_pct) in dotax_targets.items():
        current_count = status_counts.get(status, 0)
        current_pct = current_count / total_filers if total_filers > 0 else 0
        gap = current_count - target_count
        gap_pct = (gap / target_count * 100) if target_count > 0 else 0
        
        status_name = status.replace('_', ' ').title()
        print(f"{status_name:<25} | {current_count:>10,.0f} | {current_pct:>9.1%} | {target_count:>12,} | {gap:>+7,.0f} ({gap_pct:>+5.1f}%)")
    
    print(f"\nTOTAL                     | {total_filers:>10,.0f} | 100.0%     | 635,117      | {total_filers-635117:>+7,.0f}")
    
    # Save output
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'data/processed/tax_units_regenerated_{timestamp}.parquet'
    
    logger.info(f"\nSaving to: {output_file}")
    tax_units.to_parquet(output_file, index=False)
    
    logger.info("\n" + "="*80)
    logger.info("✅ TAX UNITS REGENERATION COMPLETE")
    logger.info("="*80)
    
    # Validation summary
    logger.info("\nVALIDATION SUMMARY:")
    
    mfs_count = status_counts.get('married_filing_separately', 0)
    mfs_pct = (mfs_count / total_filers * 100) if total_filers > 0 else 0
    
    if mfs_count > 0:
        logger.info(f"  ✅ MFS filers created: {mfs_count:,.0f} ({mfs_pct:.2f}%)")
    else:
        logger.info(f"  ❌ MFS filers: STILL ZERO!")
    
    logger.info(f"  Total filers: {total_filers:,.0f} (target: 635,117)")
    logger.info(f"  Coverage: {total_filers/635117*100:.1f}%")
    
    logger.info(f"\nOutput file: {output_file}")


if __name__ == '__main__':
    main()
