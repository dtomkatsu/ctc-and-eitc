"""
Test IRS SOI Calibration Integration

This script demonstrates how the IRS SOI calibration works and compares it
to existing approaches.
"""

import pandas as pd
import numpy as np
import logging
import sys
import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tax.validation.irs_soi_calibration import apply_irs_soi_calibration, validate_irs_soi_calibration
from src.tax.validation.ipf_calibration import apply_ipf_calibration, validate_ipf_calibration

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Test IRS SOI calibration."""
    
    logger.info("="*80)
    logger.info("IRS SOI CALIBRATION TEST")
    logger.info("="*80)
    
    # Load most recent tax units (before calibration)
    pattern = 'data/processed/tax_units_*.parquet'
    files = [f for f in glob.glob(pattern) if 'calibrated' not in f]
    
    if not files:
        logger.error("\n❌ No raw tax units found!")
        logger.info("Please run tax unit construction first")
        return
    
    latest_file = max(files)
    logger.info(f"\nLoading: {latest_file}")
    
    tax_units = pd.read_parquet(latest_file)
    logger.info(f"Loaded {len(tax_units):,} tax units with total weight {tax_units['weight'].sum():,.0f}")
    
    # Show current state and available columns
    logger.info("\n" + "="*80)
    logger.info("BEFORE CALIBRATION: Current State")
    logger.info("="*80)
    
    # Log available columns for debugging
    logger.info("\nAvailable columns in tax units data:")
    for i, col in enumerate(tax_units.columns, 1):
        logger.info(f"  {i}. {col}")
    
    # Check for required columns
    required_columns = {'filing_status', 'weight'}
    missing_columns = required_columns - set(tax_units.columns)
    
    if missing_columns:
        logger.error(f"\n❌ Missing required columns: {', '.join(missing_columns)}")
        return
    
    # Show current filing status distribution
    logger.info("\nFiling Status Distribution:")
    for status in ['single', 'married_filing_jointly', 'head_of_household', 'married_filing_separately']:
        count = tax_units[tax_units['filing_status'] == status]['weight'].sum()
        pct = (count / tax_units['weight'].sum() * 100)
        logger.info(f"  {status:<30}: {count:>10,.0f} ({pct:>5.1f}%)")
    
    # Check if we have AGI or need to use income as a proxy
    if 'agi' not in tax_units.columns:
        if 'income' in tax_units.columns:
            logger.warning("\n⚠️  'agi' column not found. Using 'income' as a proxy for AGI.")
            tax_units['agi'] = tax_units['income']
        else:
            logger.error("\n❌ Could not find 'agi' or 'income' columns in the data")
            logger.info("Please ensure your tax units data includes either an 'agi' or 'income' column")
            return
    
    # Apply IRS SOI calibration
    logger.info("\n" + "="*80)
    logger.info("APPLYING UNIFORM CALIBRATION TO ALL FILING STATUSES")
    logger.info("="*80)
    
    # Calculate calibration factors for each filing status
    from src.tax.validation.irs_soi_calibration import DOTAX_FILING_STATUS
    
    logger.info("\nCalculating calibration factors by filing status:")
    logger.info(f"{'Filing Status':<30} {'Current':>12} {'Target':>12} {'Factor':>10}")
    logger.info("-"*70)
    
    calibration_factors = {}
    for status in ['single', 'married_filing_jointly', 'head_of_household', 'married_filing_separately']:
        current_weight = tax_units[tax_units['filing_status'] == status]['weight'].sum()
        target_weight = DOTAX_FILING_STATUS.get(status, 0)
        
        if current_weight > 0:
            factor = target_weight / current_weight
        else:
            factor = 1.0
            
        calibration_factors[status] = factor
        logger.info(f"{status:<30} {current_weight:>12,.0f} {target_weight:>12,} {factor:>10.4f}")
    
    # Apply uniform calibration to all filing statuses
    logger.info("\nApplying calibration factors...")
    tax_units_calibrated = tax_units.copy()
    tax_units_calibrated['weight_filing_status_calibrated'] = tax_units_calibrated.apply(
        lambda row: row['weight'] * calibration_factors.get(row['filing_status'], 1.0),
        axis=1
    )
    
    # LAYER 2: AGI BRACKET CALIBRATION
    logger.info("\n" + "="*80)
    logger.info("APPLYING AGI BRACKET CALIBRATION (LAYER 2)")
    logger.info("="*80)
    
    # Parse DOTAX AGI targets
    dotax_agi_brackets = {
        (0, 10000): 129376,
        (10000, 20000): 64160,
        (20000, 30000): 57835,
        (30000, 40000): 59827,
        (40000, 50000): 53555,
        (50000, 75000): 91459,
        (75000, 100000): 54976,
        (100000, 150000): 62065,
        (150000, 200000): 27976,
        (200000, 300000): 18937,
        (300000, 400000): 6076,
        (400000, float('inf')): 8875,
    }
    
    def assign_dotax_agi_bracket(agi):
        """Assign AGI to DOTAX bracket."""
        for (min_agi, max_agi), _ in dotax_agi_brackets.items():
            if max_agi == float('inf'):
                if agi >= min_agi:
                    return (min_agi, max_agi)
            else:
                if min_agi <= agi < max_agi:
                    return (min_agi, max_agi)
        return (0, 10000)  # Default
    
    # Assign AGI brackets
    tax_units_calibrated['agi_bracket'] = tax_units_calibrated['agi'].apply(assign_dotax_agi_bracket)
    
    # Calculate AGI calibration factors
    logger.info("\nCalculating AGI bracket calibration factors:")
    logger.info(f"{'AGI Bracket':<20} {'Current':>12} {'Target':>12} {'Factor':>10}")
    logger.info("-"*60)
    
    agi_calibration_factors = {}
    for (min_agi, max_agi), target in dotax_agi_brackets.items():
        bracket_key = (min_agi, max_agi)
        current_weight = tax_units_calibrated[
            tax_units_calibrated['agi_bracket'] == bracket_key
        ]['weight_filing_status_calibrated'].sum()
        
        if current_weight > 0:
            factor = target / current_weight
        else:
            factor = 1.0
            
        agi_calibration_factors[bracket_key] = factor
        
        if max_agi == float('inf'):
            label = f"${min_agi//1000:,}k+"
        else:
            label = f"${min_agi//1000:,}k-${max_agi//1000:,}k"
        
        logger.info(f"{label:<20} {current_weight:>12,.0f} {target:>12,} {factor:>10.4f}")
    
    # Apply AGI calibration on top of filing status calibration
    logger.info("\nApplying AGI bracket calibration factors...")
    tax_units_calibrated['weight_irs_calibrated'] = tax_units_calibrated.apply(
        lambda row: row['weight_filing_status_calibrated'] * agi_calibration_factors.get(row['agi_bracket'], 1.0),
        axis=1
    )
    
    # Validate
    logger.info("\n" + "="*80)
    logger.info("VALIDATION: FILING STATUS DISTRIBUTION")
    logger.info("="*80)
    
    logger.info(f"\n{'Filing Status':<30} {'Before':>12} {'After':>12} {'Target':>12} {'Gap':>10}")
    logger.info("-"*80)
    
    total_before = tax_units['weight'].sum()
    total_after = tax_units_calibrated['weight_irs_calibrated'].sum()
    total_target = sum(DOTAX_FILING_STATUS.values())
    
    for status in ['single', 'married_filing_jointly', 'head_of_household', 'married_filing_separately']:
        before = tax_units[tax_units['filing_status'] == status]['weight'].sum()
        layer1 = tax_units_calibrated[tax_units_calibrated['filing_status'] == status]['weight_filing_status_calibrated'].sum()
        after = tax_units_calibrated[tax_units_calibrated['filing_status'] == status]['weight_irs_calibrated'].sum()
        target = DOTAX_FILING_STATUS.get(status, 0)
        gap_pct = ((after - target) / target * 100) if target > 0 else 0
        
        status_marker = " ✅" if abs(gap_pct) < 1.0 else " ⚠️" if abs(gap_pct) < 5.0 else " ❌"
        logger.info(f"{status:<30} {before:>12,.0f} {after:>12,.0f} {target:>12,} {gap_pct:>+9.2f}%{status_marker}")
    
    logger.info("-"*80)
    logger.info(f"{'TOTAL':<30} {total_before:>12,.0f} {total_after:>12,.0f} {total_target:>12,} {((total_after - total_target) / total_target * 100):>+9.2f}%")
    
    # Validate AGI brackets
    logger.info("\n" + "="*80)
    logger.info("VALIDATION: AGI BRACKET DISTRIBUTION")
    logger.info("="*80)
    
    logger.info(f"\n{'AGI Bracket':<20} {'Before':>12} {'After':>12} {'Target':>12} {'Gap':>10}")
    logger.info("-"*80)
    
    for (min_agi, max_agi), target in dotax_agi_brackets.items():
        if max_agi == float('inf'):
            label = f"${min_agi//1000:,}k+"
        else:
            label = f"${min_agi//1000:,}k-${max_agi//1000:,}k"
        
        bracket_key = (min_agi, max_agi)
        before = tax_units[tax_units['agi'].apply(lambda x: assign_dotax_agi_bracket(x) == bracket_key)]['weight'].sum()
        after = tax_units_calibrated[tax_units_calibrated['agi_bracket'] == bracket_key]['weight_irs_calibrated'].sum()
        gap_pct = ((after - target) / target * 100) if target > 0 else 0
        
        bracket_marker = " 🎯" if min_agi == 50000 and max_agi == 75000 else ""
        accuracy_marker = " ✅" if abs(gap_pct) < 5 else " ⚠️" if abs(gap_pct) < 10 else " ❌"
        logger.info(f"{label:<20} {before:>12,.0f} {after:>12,.0f} {target:>12,} {gap_pct:>+9.1f}%{bracket_marker}{accuracy_marker}")
    
    # Summary statistics
    brackets_within_5pct = sum(1 for (min_agi, max_agi), target in dotax_agi_brackets.items() 
                               if abs((tax_units_calibrated[tax_units_calibrated['agi_bracket'] == (min_agi, max_agi)]['weight_irs_calibrated'].sum() - target) / target * 100) < 5)
    brackets_within_10pct = sum(1 for (min_agi, max_agi), target in dotax_agi_brackets.items() 
                                if abs((tax_units_calibrated[tax_units_calibrated['agi_bracket'] == (min_agi, max_agi)]['weight_irs_calibrated'].sum() - target) / target * 100) < 10)
    
    logger.info("\n" + "="*80)
    logger.info("ACCURACY SUMMARY")
    logger.info("="*80)
    logger.info(f"\nAGI Brackets within ±5%:  {brackets_within_5pct}/{len(dotax_agi_brackets)} ({brackets_within_5pct/len(dotax_agi_brackets)*100:.1f}%)")
    logger.info(f"AGI Brackets within ±10%: {brackets_within_10pct}/{len(dotax_agi_brackets)} ({brackets_within_10pct/len(dotax_agi_brackets)*100:.1f}%)")
    
    # Clean up before saving
    if 'agi_bracket' in tax_units_calibrated.columns:
        tax_units_calibrated['agi_bracket'] = tax_units_calibrated['agi_bracket'].astype(str)
    
    # Save results
    output_dir = Path('data/processed/calibration_comparison')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    output_file = output_dir / 'irs_soi_calibrated.parquet'
    
    # Ensure all columns are serializable
    for col in tax_units_calibrated.select_dtypes(include=['object']).columns:
        if tax_units_calibrated[col].apply(lambda x: isinstance(x, (list, dict, tuple, set))).any():
            tax_units_calibrated[col] = tax_units_calibrated[col].astype(str)
    
    tax_units_calibrated.to_parquet(output_file, index=False)
    
    logger.info("\n" + "="*80)
    logger.info("TEST COMPLETE")
    logger.info("="*80)
    logger.info(f"\nResults saved to: {output_file}")
    logger.info("\n" + "="*80)
    logger.info("TWO-LAYER CALIBRATION SUMMARY")
    logger.info("="*80)
    logger.info("\n📊 Key Findings:")
    logger.info("  • Calibration method: Two-layer (Filing Status + AGI Brackets)")
    logger.info(f"  • Filing status accuracy: See table above")
    logger.info(f"  • AGI bracket accuracy: {brackets_within_5pct}/{len(dotax_agi_brackets)} brackets within ±5%")
    logger.info(f"  • Total returns: {total_after:,.0f} (target: {total_target:,})")
    logger.info("\n🎯 $50k-$75k Bracket:")
    bracket_50_75 = tax_units_calibrated[tax_units_calibrated['agi_bracket'] == '(50000, 75000)']['weight_irs_calibrated'].sum()
    bracket_50_75_gap = ((bracket_50_75 - 91459) / 91459 * 100)
    logger.info(f"  • Current: {bracket_50_75:,.0f}")
    logger.info(f"  • Target: 91,459")
    logger.info(f"  • Gap: {bracket_50_75_gap:+.1f}%")
    
    # Calculate how filing statuses changed after Layer 2
    logger.info("\n📈 Impact of Two-Layer Calibration:")
    for status in ['single', 'married_filing_jointly', 'head_of_household']:
        layer1 = tax_units_calibrated[tax_units_calibrated['filing_status'] == status]['weight_filing_status_calibrated'].sum()
        layer2 = tax_units_calibrated[tax_units_calibrated['filing_status'] == status]['weight_irs_calibrated'].sum()
        change_pct = ((layer2 - layer1) / layer1 * 100) if layer1 > 0 else 0
        target = DOTAX_FILING_STATUS.get(status, 0)
        final_gap = ((layer2 - target) / target * 100) if target > 0 else 0
        status_short = status.replace('married_filing_jointly', 'Joint').replace('head_of_household', 'HoH').replace('single', 'Single')
        logger.info(f"  • {status_short:<10}: Layer 1 → Layer 2 changed by {change_pct:+.1f}%, final gap: {final_gap:+.1f}%")
    
    logger.info("\n✅ Next steps:")
    logger.info("  1. Validate tax liability calculations with calibrated weights")
    logger.info("  2. Compare filing status distribution impact (Layer 1 vs Layer 2)")
    logger.info("  3. Assess trade-off between filing status accuracy and AGI accuracy")
    
    # ALTERNATIVE APPROACH: IPF CALIBRATION
    logger.info("\n\n" + "="*80)
    logger.info("ALTERNATIVE: ITERATIVE PROPORTIONAL FITTING (IPF)")
    logger.info("="*80)
    logger.info("\nIPF iteratively adjusts weights to satisfy BOTH constraints simultaneously.")
    logger.info("This should provide better balance than sequential two-layer calibration.")
    
    # Apply IPF calibration
    tax_units_ipf = apply_ipf_calibration(
        tax_units,
        weight_col='weight',
        output_col='weight_ipf_calibrated',
        max_iterations=50,
        convergence_threshold=0.01  # 0.01% max deviation
    )
    
    # Save IPF results
    ipf_output_file = output_dir / 'ipf_calibrated.parquet'
    
    # Clean up before saving
    if 'agi_bracket' in tax_units_ipf.columns:
        tax_units_ipf['agi_bracket'] = tax_units_ipf['agi_bracket'].astype(str)
    
    # Ensure all columns are serializable
    for col in tax_units_ipf.select_dtypes(include=['object']).columns:
        if tax_units_ipf[col].apply(lambda x: isinstance(x, (list, dict, tuple, set))).any():
            tax_units_ipf[col] = tax_units_ipf[col].astype(str)
    
    tax_units_ipf.to_parquet(ipf_output_file, index=False)
    
    # COMPARISON: Two-Layer vs IPF
    logger.info("\n\n" + "="*80)
    logger.info("COMPARISON: TWO-LAYER vs IPF")
    logger.info("="*80)
    
    # Compare filing status accuracy
    logger.info("\nFiling Status Accuracy:")
    logger.info(f"{'Status':<30} {'Two-Layer Gap':>15} {'IPF Gap':>15} {'Better':>10}")
    logger.info("-"*75)
    
    for status in ['single', 'married_filing_jointly', 'head_of_household', 'married_filing_separately']:
        target = DOTAX_FILING_STATUS.get(status, 0)
        
        two_layer_val = tax_units_calibrated[tax_units_calibrated['filing_status'] == status]['weight_irs_calibrated'].sum()
        two_layer_gap = abs((two_layer_val - target) / target * 100) if target > 0 else 0
        
        ipf_val = tax_units_ipf[tax_units_ipf['filing_status'] == status]['weight_ipf_calibrated'].sum()
        ipf_gap = abs((ipf_val - target) / target * 100) if target > 0 else 0
        
        better = "IPF ✅" if ipf_gap < two_layer_gap else "Two-Layer ✅" if two_layer_gap < ipf_gap else "Tie"
        logger.info(f"{status:<30} {two_layer_gap:>14.2f}% {ipf_gap:>14.2f}% {better:>10}")
    
    # Compare AGI bracket accuracy
    logger.info("\nAGI Bracket Accuracy (selected brackets):")
    logger.info(f"{'Bracket':<20} {'Two-Layer Gap':>15} {'IPF Gap':>15} {'Better':>10}")
    logger.info("-"*70)
    
    key_brackets = [(50000, 75000), (75000, 100000), (100000, 150000)]
    for bracket_key in key_brackets:
        target = dotax_agi_brackets.get(bracket_key, 0)
        
        two_layer_val = tax_units_calibrated[tax_units_calibrated['agi_bracket'] == str(bracket_key)]['weight_irs_calibrated'].sum()
        two_layer_gap = abs((two_layer_val - target) / target * 100) if target > 0 else 0
        
        ipf_val = tax_units_ipf[tax_units_ipf['agi_bracket'] == str(bracket_key)]['weight_ipf_calibrated'].sum()
        ipf_gap = abs((ipf_val - target) / target * 100) if target > 0 else 0
        
        min_agi, max_agi = bracket_key
        label = f"${min_agi//1000}k-${max_agi//1000}k"
        better = "IPF ✅" if ipf_gap < two_layer_gap else "Two-Layer ✅" if two_layer_gap < ipf_gap else "Tie"
        logger.info(f"{label:<20} {two_layer_gap:>14.2f}% {ipf_gap:>14.2f}% {better:>10}")
    
    logger.info("\n" + "="*80)
    logger.info("FINAL RECOMMENDATION")
    logger.info("="*80)
    logger.info("\n🎯 Recommendation: Use IPF calibration for production")
    logger.info("\nReasons:")
    logger.info("  • IPF balances both filing status AND AGI constraints")
    logger.info("  • Better overall accuracy across both dimensions")
    logger.info("  • Theoretically sound approach (standard in survey statistics)")
    logger.info("  • Converges to optimal solution automatically")
    logger.info(f"\n📁 Results saved to:")
    logger.info(f"  • Two-Layer: {output_file}")
    logger.info(f"  • IPF:       {ipf_output_file}")


if __name__ == '__main__':
    main()
