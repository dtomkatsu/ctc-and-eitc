"""
Investigate $50k-$75k Bracket Over-Representation

This script analyzes why the $50k-$75k AGI bracket is over-represented in PUMS data
and develops solutions to correct it.

Key Issues:
- Model: 113,994 returns vs DOTAX: 91,459 (+24.6%)
- Model: $419.0M tax vs DOTAX: $293.0M (+43.0%)
"""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.pums_loader import PUMSDataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_50k_75k_bracket(tax_units: pd.DataFrame):
    """Analyze the $50k-$75k bracket in detail."""
    
    logger.info("="*80)
    logger.info("INVESTIGATING $50K-$75K BRACKET OVER-REPRESENTATION")
    logger.info("="*80)
    
    # Define the bracket
    bracket_min, bracket_max = 50000, 75000
    
    # Filter to this bracket
    in_bracket = (tax_units['agi'] >= bracket_min) & (tax_units['agi'] < bracket_max)
    bracket_df = tax_units[in_bracket].copy()
    
    logger.info(f"\nBracket: ${bracket_min:,} - ${bracket_max:,}")
    logger.info(f"  Units in bracket: {len(bracket_df):,}")
    logger.info(f"  Weighted count: {bracket_df['weight'].sum():,.0f}")
    logger.info(f"  DOTAX benchmark: 91,459")
    logger.info(f"  Over-representation: +{bracket_df['weight'].sum() - 91459:,.0f} (+{(bracket_df['weight'].sum() - 91459)/91459*100:.1f}%)")
    
    # Analyze by filing status
    logger.info("\n" + "-"*80)
    logger.info("BY FILING STATUS")
    logger.info("-"*80)
    logger.info(f"{'Status':<30} {'Count':>12} {'% of Bracket':>15} {'Avg AGI':>15} {'Avg Tax':>15}")
    logger.info("-"*80)
    
    for status in ['single', 'married_filing_jointly', 'head_of_household', 'married_filing_separately']:
        status_df = bracket_df[bracket_df['filing_status'] == status]
        if len(status_df) > 0:
            count = status_df['weight'].sum()
            pct = count / bracket_df['weight'].sum() * 100
            avg_agi = (status_df['agi'] * status_df['weight']).sum() / count
            avg_tax = (status_df['hi_state_tax'] * status_df['weight']).sum() / count if 'hi_state_tax' in status_df.columns else 0
            
            logger.info(f"{status:<30} {count:>12,.0f} {pct:>14.1f}% ${avg_agi:>14,.0f} ${avg_tax:>14,.0f}")
    
    # Analyze income sources
    logger.info("\n" + "-"*80)
    logger.info("INCOME COMPOSITION ANALYSIS")
    logger.info("-"*80)
    
    # Check what income components we have
    income_cols = [col for col in bracket_df.columns if any(x in col.lower() for x in ['wage', 'salary', 'business', 'retirement', 'interest', 'dividend'])]
    
    if income_cols:
        logger.info(f"Available income columns: {', '.join(income_cols)}")
    else:
        logger.info("No detailed income breakdown available")
    
    # Analyze household composition
    logger.info("\n" + "-"*80)
    logger.info("HOUSEHOLD COMPOSITION")
    logger.info("-"*80)
    
    if 'num_adults' in bracket_df.columns and 'num_dependents' in bracket_df.columns:
        logger.info(f"{'Adults':>8} {'Dependents':>12} {'Count':>12} {'% of Bracket':>15}")
        logger.info("-"*80)
        
        composition = bracket_df.groupby(['num_adults', 'num_dependents'])['weight'].sum().reset_index()
        composition = composition.sort_values('weight', ascending=False)
        
        for _, row in composition.head(10).iterrows():
            adults = int(row['num_adults'])
            deps = int(row['num_dependents'])
            count = row['weight']
            pct = count / bracket_df['weight'].sum() * 100
            logger.info(f"{adults:>8} {deps:>12} {count:>12,.0f} {pct:>14.1f}%")
    
    # Distribution within bracket
    logger.info("\n" + "-"*80)
    logger.info("AGI DISTRIBUTION WITHIN BRACKET")
    logger.info("-"*80)
    
    sub_brackets = [
        (50000, 55000, "$50k-$55k"),
        (55000, 60000, "$55k-$60k"),
        (60000, 65000, "$60k-$65k"),
        (65000, 70000, "$65k-$70k"),
        (70000, 75000, "$70k-$75k"),
    ]
    
    logger.info(f"{'Sub-Bracket':<15} {'Count':>12} {'% of Bracket':>15} {'Avg Tax':>15}")
    logger.info("-"*80)
    
    for min_agi, max_agi, label in sub_brackets:
        sub_df = bracket_df[(bracket_df['agi'] >= min_agi) & (bracket_df['agi'] < max_agi)]
        count = sub_df['weight'].sum()
        pct = count / bracket_df['weight'].sum() * 100 if bracket_df['weight'].sum() > 0 else 0
        avg_tax = (sub_df['hi_state_tax'] * sub_df['weight']).sum() / count if count > 0 and 'hi_state_tax' in sub_df.columns else 0
        
        logger.info(f"{label:<15} {count:>12,.0f} {pct:>14.1f}% ${avg_tax:>14,.0f}")
    
    # Compare to adjacent brackets
    logger.info("\n" + "-"*80)
    logger.info("COMPARISON TO ADJACENT BRACKETS")
    logger.info("-"*80)
    
    brackets_compare = [
        (40000, 50000, "$40k-$50k", 53555),
        (50000, 75000, "$50k-$75k", 91459),
        (75000, 100000, "$75k-$100k", 54976),
    ]
    
    logger.info(f"{'Bracket':<15} {'Model':>12} {'DOTAX':>12} {'Difference':>12} {'% Diff':>10}")
    logger.info("-"*80)
    
    for min_agi, max_agi, label, dotax_count in brackets_compare:
        in_range = (tax_units['agi'] >= min_agi) & (tax_units['agi'] < max_agi)
        model_count = tax_units[in_range]['weight'].sum()
        diff = model_count - dotax_count
        pct = (diff / dotax_count * 100) if dotax_count > 0 else 0
        
        logger.info(f"{label:<15} {model_count:>12,.0f} {dotax_count:>12,} {diff:>+12,.0f} {pct:>+9.1f}%")
    
    return bracket_df


def develop_correction_options(tax_units: pd.DataFrame):
    """Develop options for correcting the $50k-$75k over-representation."""
    
    logger.info("\n" + "="*80)
    logger.info("CORRECTION OPTIONS FOR $50K-$75K OVER-REPRESENTATION")
    logger.info("="*80)
    
    bracket_min, bracket_max = 50000, 75000
    in_bracket = (tax_units['agi'] >= bracket_min) & (tax_units['agi'] < bracket_max)
    bracket_df = tax_units[in_bracket].copy()
    
    current_count = bracket_df['weight'].sum()
    target_count = 91459
    excess = current_count - target_count
    reduction_factor = target_count / current_count
    
    logger.info(f"\nCurrent count: {current_count:,.0f}")
    logger.info(f"Target count: {target_count:,.0f}")
    logger.info(f"Excess: {excess:,.0f} (+{excess/target_count*100:.1f}%)")
    logger.info(f"Required reduction factor: {reduction_factor:.4f}")
    
    # Option 1: Uniform weight reduction
    logger.info("\n" + "-"*80)
    logger.info("OPTION 1: UNIFORM WEIGHT REDUCTION")
    logger.info("-"*80)
    logger.info(f"Multiply all weights in bracket by {reduction_factor:.4f}")
    logger.info(f"  Pros: Simple, preserves within-bracket distribution")
    logger.info(f"  Cons: Treats all households equally (may not reflect reality)")
    
    # Option 2: Graduated reduction (higher incomes get larger reduction)
    logger.info("\n" + "-"*80)
    logger.info("OPTION 2: GRADUATED REDUCTION (TOP-HEAVY)")
    logger.info("-"*80)
    logger.info("Reduce weights more for higher incomes within bracket")
    logger.info("  Example: $50-55k → 0.90x, $55-60k → 0.85x, ..., $70-75k → 0.70x")
    logger.info("  Pros: Addresses potential PUMS over-sampling of upper-middle incomes")
    logger.info("  Cons: More complex, requires assumption about where excess occurs")
    
    # Option 3: Filing status-specific reduction
    logger.info("\n" + "-"*80)
    logger.info("OPTION 3: FILING STATUS-SPECIFIC REDUCTION")
    logger.info("-"*80)
    logger.info("Apply different reduction factors by filing status")
    
    for status in ['single', 'married_filing_jointly', 'head_of_household', 'married_filing_separately']:
        status_df = bracket_df[bracket_df['filing_status'] == status]
        if len(status_df) > 0:
            status_count = status_df['weight'].sum()
            status_pct = status_count / current_count * 100
            logger.info(f"  {status:<30}: {status_count:>10,.0f} ({status_pct:>5.1f}%)")
    
    logger.info("  Pros: Can target specific filing statuses if they're over-represented")
    logger.info("  Cons: Need benchmarks by status AND income bracket")
    
    # Option 4: Hybrid with adjacent brackets
    logger.info("\n" + "-"*80)
    logger.info("OPTION 4: HYBRID MULTI-BRACKET CALIBRATION")
    logger.info("-"*80)
    logger.info("Calibrate $40k-$50k, $50k-$75k, and $75k-$100k simultaneously")
    logger.info("  Benefits:")
    logger.info("    - Prevents creating gaps in adjacent brackets")
    logger.info("    - More natural income distribution")
    logger.info("    - Better handles 'boundary effects'")
    
    # Show what this would look like
    brackets_multi = [
        (40000, 50000, 53555, 60115),
        (50000, 75000, 91459, 113994),
        (75000, 100000, 54976, 64505),
    ]
    
    logger.info("\n  Current vs Target:")
    logger.info(f"  {'Bracket':<15} {'Current':>12} {'Target':>12} {'Factor':>10}")
    logger.info("  " + "-"*50)
    
    for min_agi, max_agi, target, current in brackets_multi:
        label = f"${min_agi//1000}k-${max_agi//1000}k"
        factor = target / current if current > 0 else 1.0
        logger.info(f"  {label:<15} {current:>12,.0f} {target:>12,} {factor:>9.3f}x")
    
    # Option 5: Income source-based reduction
    logger.info("\n" + "-"*80)
    logger.info("OPTION 5: INCOME SOURCE-BASED REDUCTION")
    logger.info("-"*80)
    logger.info("Reduce weights based on primary income source")
    logger.info("  Rationale: PUMS may over-represent certain income types")
    logger.info("  Example: Dual-income households, wage earners vs business income")
    logger.info("  Limitation: Requires detailed income composition data")
    
    # Recommendation
    logger.info("\n" + "="*80)
    logger.info("RECOMMENDATION")
    logger.info("="*80)
    logger.info("\n✅ RECOMMENDED: Option 4 - Hybrid Multi-Bracket Calibration")
    logger.info("\nRationale:")
    logger.info("  1. The $50k-$75k issue is part of a broader pattern:")
    logger.info("     - $40k-$50k: +12.2% over (needs reduction)")
    logger.info("     - $50k-$75k: +24.6% over (needs reduction)")
    logger.info("     - $75k-$100k: +17.3% over (needs reduction)")
    logger.info("  2. These three brackets together represent middle-income over-representation")
    logger.info("  3. Calibrating them together prevents unnatural 'jumps' at boundaries")
    logger.info("  4. Already implemented in bracket_calibration.py infrastructure")
    logger.info("\nImplementation:")
    logger.info("  - Use existing SOI calibration factors from DOTAX Table 12A")
    logger.info("  - Apply smooth transitions at bracket boundaries")
    logger.info("  - Validate against total tax liability benchmarks")


def propose_implementation(tax_units: pd.DataFrame):
    """Propose specific implementation for correction."""
    
    logger.info("\n" + "="*80)
    logger.info("IMPLEMENTATION PLAN")
    logger.info("="*80)
    
    logger.info("\nStep 1: Enhance bracket_calibration.py")
    logger.info("  - Already has AGI bracket targets from Table 12A")
    logger.info("  - Currently calibrates by filing status + taxable income brackets")
    logger.info("  - Need to apply to AGI brackets as well")
    
    logger.info("\nStep 2: Create dual-layer calibration")
    logger.info("  - Layer 1: Filing status totals (currently working)")
    logger.info("  - Layer 2: AGI bracket distribution (new)")
    
    logger.info("\nStep 3: Apply calibration factors")
    
    # Calculate what the factors would be
    dotax_brackets = [
        (0, 10000, 129376),
        (10000, 20000, 64160),
        (20000, 30000, 57835),
        (30000, 40000, 59827),
        (40000, 50000, 53555),
        (50000, 75000, 91459),
        (75000, 100000, 54976),
        (100000, 150000, 62065),
        (150000, 200000, 27976),
        (200000, 300000, 18937),
        (300000, 400000, 6076),
        (400000, float('inf'), 8875),
    ]
    
    logger.info("\n  Proposed calibration factors:")
    logger.info(f"  {'AGI Bracket':<20} {'Current':>12} {'Target':>12} {'Factor':>10}")
    logger.info("  " + "-"*60)
    
    for min_agi, max_agi, target in dotax_brackets:
        if max_agi == float('inf'):
            in_range = tax_units['agi'] >= min_agi
            label = f"${min_agi//1000}k+"
        else:
            in_range = (tax_units['agi'] >= min_agi) & (tax_units['agi'] < max_agi)
            label = f"${min_agi//1000}k-${max_agi//1000}k"
        
        current = tax_units[in_range]['weight'].sum()
        factor = target / current if current > 0 else 1.0
        
        # Highlight the $50k-$75k bracket
        marker = " ⚠️" if 50000 <= min_agi < 75000 else ""
        logger.info(f"  {label:<20} {current:>12,.0f} {target:>12,} {factor:>9.3f}x{marker}")
    
    logger.info("\nStep 4: Validate results")
    logger.info("  - Re-run Table 12A validation")
    logger.info("  - Check tax liability by bracket")
    logger.info("  - Verify no unintended side effects on filing status distribution")


def main():
    """Main analysis function."""
    
    # Load most recent calibrated tax units
    import glob
    files = glob.glob('data/processed/tax_units_calibrated_*.parquet')
    if not files:
        logger.error("No calibrated tax units found!")
        return
    
    latest_file = max(files)
    logger.info(f"Loading: {latest_file}")
    
    tax_units = pd.read_parquet(latest_file)
    logger.info(f"Loaded {len(tax_units):,} tax units with total weight {tax_units['weight'].sum():,.0f}")
    
    # Run analyses
    bracket_df = analyze_50k_75k_bracket(tax_units)
    develop_correction_options(tax_units)
    propose_implementation(tax_units)
    
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info("\nNext steps:")
    logger.info("  1. Review proposed correction options")
    logger.info("  2. Decide on implementation approach")
    logger.info("  3. Modify bracket_calibration.py to add AGI-based calibration")
    logger.info("  4. Re-run regenerate_tax_units.py with new calibration")
    logger.info("  5. Validate results against Table 12A")


if __name__ == '__main__':
    main()
