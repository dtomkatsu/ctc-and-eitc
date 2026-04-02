#!/usr/bin/env python3
"""
Regenerate Tax Units with Systematic Calibration

This script regenerates the tax units file with:
1. MFS filers included (updated thresholds)
2. Strict married couple identification
3. All filing statuses systematically aligned to DOTAX 2022 benchmarks
4. Precise AGI distribution and tax revenue calibration
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
from src.tax.units.status.bracket_calibration import apply_bracket_calibration
from src.tax.adjustments.agi_adjustments import apply_agi_adjustments_to_dataframe
from src.tax.adjustments.itemized_deductions import ItemizedDeductionEstimator
from src.tax.adjustments.ultra_high_income_synthesizer_v2 import UltraHighIncomeSynthesizerV2
from src.tax.brackets import load_tax_data
from src.tax.adjustments.capital_gains import apply_capital_gains_to_dataframe
from src.tax.adjustments.income_distribution_calibrator import IncomeDistributionCalibrator
from src.tax.adjustments.comprehensive_weight_calibrator import ComprehensiveWeightCalibrator
from src.tax.calibration.simultaneous_calibrator import SimultaneousCalibrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def apply_filing_status_weight_calibration(df: pd.DataFrame, weight_col: str = 'weight') -> pd.DataFrame:
    """Apply DOTAX filing-status weight calibration to match target shares."""

    targets = {
        'Single': 0.5100,
        'Joint': 0.3600,
        'Head of Household': 0.0960,
        'MFS': 0.0340,
    }

    status_map = {
        'single': 'Single',
        'Single': 'Single',
        'married_filing_jointly': 'Joint',
        'Married Filing Jointly': 'Joint',
        'married_filing_separately': 'MFS',
        'Married Filing Separately': 'MFS',
        'head_of_household': 'Head of Household',
        'Head of Household': 'Head of Household',
        'Head Of Household': 'Head of Household',
        'qualifying_widow': 'Joint',
    }

    if weight_col not in df.columns:
        raise ValueError(f"Weight column '{weight_col}' not found during filing status calibration")

    df = df.copy()
    df['__fs_clean'] = df['filing_status'].map(status_map).fillna('Other')

    total_weight = df[weight_col].sum()
    if total_weight == 0:
        logger.warning("Total weight is zero before filing status calibration; skipping adjustment")
        df.drop(columns='__fs_clean', inplace=True)
        return df

    current_dist = df.groupby('__fs_clean')[weight_col].sum() / total_weight

    factors = {}
    for status, target_share in targets.items():
        current_share = current_dist.get(status, 0)
        if current_share > 0:
            factors[status] = target_share / current_share
        else:
            logger.warning("No units found for status '%s' during calibration; leaving weights unchanged", status)
            factors[status] = 1.0

    logger.info("\n🔧 Applying filing status weight calibration to match DOTAX targets...")
    for status, factor in factors.items():
        logger.info("  %s factor: %.4f", status, factor)

    df['__calibration_factor'] = df['__fs_clean'].map(factors).fillna(1.0)
    df[weight_col] = df[weight_col] * df['__calibration_factor']

    df.drop(columns=['__fs_clean', '__calibration_factor'], inplace=True)
    return df


def _apply_tax_results(tax_units: pd.DataFrame, tax_results: pd.DataFrame) -> pd.DataFrame:
    """
    Merge calculator output into tax_units and promote hi_tax_tax_liability → hi_state_tax.

    Raises ValueError if the calculator did not produce 'hi_tax_tax_liability', which
    prevents silent zero-tax results from propagating through the rest of the pipeline.
    """
    for col in tax_results.columns:
        tax_units[col] = tax_results[col]

    if 'hi_tax_tax_liability' not in tax_units.columns:
        raise ValueError(
            "Tax calculator did not produce 'hi_tax_tax_liability'. "
            "Check calculator configuration and filing_status_hawaii column for NaN values."
        )
    tax_units['hi_state_tax'] = tax_units['hi_tax_tax_liability']

    if 'hi_tax_taxable_income' in tax_units.columns:
        tax_units['hi_taxable_income'] = tax_units['hi_tax_taxable_income']

    return tax_units


def main(include_capital_gains: bool = True):
    """
    Main execution.
    
    Args:
        include_capital_gains: If True, include capital gains in AGI for apples-to-apples
                              comparison with DOTax benchmarks. If False, exclude capital
                              gains (useful for policy scenarios without cap gains).
    """
    logger.info("="*80)
    logger.info("REGENERATING TAX UNITS WITH UPDATED LOGIC")
    logger.info(f"Capital gains: {'INCLUDED' if include_capital_gains else 'EXCLUDED'}")
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
        use_soi_calibration=False  # Disable IPF calibration (use post-processing instead)
    )
    
    # Create tax units
    logger.info("\nCreating tax units...")
    tax_units = constructor.create_rule_based_units()
    
    logger.info(f"\n✅ Created {len(tax_units):,} tax units (before calibration)")
    
    # Apply bracket-level SOI calibration
    
    logger.info("\n🔧 Applying bracket-level SOI calibration to match DOTAX income distributions...")
    tax_units = apply_bracket_calibration(tax_units, weight_col='weight', method='factor')
    
    # Use calibrated weights as primary weight
    if 'weight_calibrated' in tax_units.columns:
        tax_units['weight'] = tax_units['weight_calibrated']
    
    logger.info(f"\n✅ Calibration complete - {len(tax_units):,} tax units")
    
    # Apply AGI adjustments and itemized deductions
    
    logger.info("\n🔧 Applying AGI adjustments...")
    tax_units = apply_agi_adjustments_to_dataframe(
        tax_units, 
        income_col='income',
        filing_status_col='filing_status'
    )
    
    # Placeholder so downstream column references don't crash before first tax calculation.
    # This will be overwritten by every real tax calculation below.
    tax_units['hi_state_tax'] = 0.0

    # First add ultra-high-income synthetic filers before AGI/capital gains calibration
    logger.info("\n💎 Step 2a: Ultra-high-income synthesis (before AGI calibration)...")
    
    ultra_synthesizer = UltraHighIncomeSynthesizerV2(pareto_alpha=1.5)
    tax_units = ultra_synthesizer.redistribute_within_million_plus(tax_units, target_tax_m=663.0)
    
    # Ensure synthetic units are marked for tracking
    if 'is_synthetic_ultra_high' not in tax_units.columns:
        tax_units['is_synthetic_ultra_high'] = False
    
    # Calculate taxes for synthetic units immediately after creation
    logger.info("\n💰 Step 2a.5: Calculate taxes for synthetic units...")
    
    calculator = load_tax_data()
    filing_status_map = {
        'single': 'Single_Married_Separate',
        'married_filing_jointly': 'Joint_Surviving_Spouse',
        'joint': 'Joint_Surviving_Spouse',          # legacy alias
        'married_filing_separately': 'Single_Married_Separate',
        'married_filing_separate': 'Single_Married_Separate',  # legacy alias
        'head_of_household': 'Head_of_Household',
        'qualifying_widow': 'Joint_Surviving_Spouse',
    }
    
    # Ensure filing_status_hawaii column exists for ALL units
    # Update or create it to ensure synthetics are included
    tax_units['filing_status_hawaii'] = tax_units['filing_status'].map(filing_status_map)
    
    # Calculate taxes for all units (including synthetics)
    tax_results = calculator.calculate_tax_for_dataframe(
        tax_units,
        income_col='agi',
        filing_status_col='filing_status_hawaii',
        year=2017
    )
    
    # Merge results; raises if calculator column is absent
    tax_units = _apply_tax_results(tax_units, tax_results)

    synthetic_mask = tax_units['is_synthetic_ultra_high'] == True
    synthetic_tax = (tax_units.loc[synthetic_mask, 'hi_state_tax'] * tax_units.loc[synthetic_mask, 'weight']).sum() / 1_000_000
    logger.info(f"   Synthetic ultra-high-income tax: ${synthetic_tax:,.1f}M")
    logger.info(f"   Total tax (all units): ${(tax_units['hi_state_tax'] * tax_units['weight']).sum() / 1_000_000:,.1f}M")

    # Apply capital gains if enabled (now includes synthetic units)
    if include_capital_gains:
        from src.tax.adjustments.capital_gains import apply_capital_gains_to_dataframe
        
        logger.info("\n📈 Step 2b: Applying capital gains (Table 21) - includes synthetic units...")
        logger.info("   Capital gains added to AGI → affects taxable income → affects tax liability")
        logger.info("   Comparing AMOUNTS of capital gains (not tax on them) to Table 21")
        
        # Store original AGI without capital gains
        tax_units['agi_without_cap_gains'] = tax_units['agi']
        
        # Apply capital gains estimation with deterministic assignment
        tax_units = apply_capital_gains_to_dataframe(
            tax_units,
            agi_col='agi',
            taxable_income_col=None,  # Will estimate from AGI
            include_random=True,
            assignment_method='deterministic_top'  # Use deterministic assignment for consistency
        )
        
        # Use AGI with capital gains as the primary AGI
        if 'agi_with_cap_gains' not in tax_units.columns:
            raise ValueError(
                "'agi_with_cap_gains' was not produced by apply_capital_gains_to_dataframe(). "
                "Check the capital gains module for errors."
            )
        tax_units['agi'] = tax_units['agi_with_cap_gains']
        
        # Log summary statistics
        total_cap_gains = (tax_units['capital_gains'] * tax_units['weight']).sum() / 1_000_000
        filers_with_cap_gains = (tax_units['capital_gains'] > 0).sum()
        weighted_filers_with_cap_gains = ((tax_units['capital_gains'] > 0) * tax_units['weight']).sum()
        
        logger.info(f"   Total capital gains (weighted): ${total_cap_gains:,.1f}M")
        logger.info(f"   Filers with capital gains: {weighted_filers_with_cap_gains:,.0f} ({weighted_filers_with_cap_gains/tax_units['weight'].sum()*100:.1f}%)")
        logger.info(f"   Average cap gains per filer: ${total_cap_gains * 1_000_000 / weighted_filers_with_cap_gains:,.0f}")
        
        # Recalculate taxes with updated AGI (including capital gains)
        logger.info("\n💰 Step 2b.5: Recalculate taxes after capital gains...")
        tax_results = calculator.calculate_tax_for_dataframe(
            tax_units,
            income_col='agi',
            filing_status_col='filing_status_hawaii',
            year=2017
        )
        
        # Merge results; raises if calculator column is absent
        tax_units = _apply_tax_results(tax_units, tax_results)

        synthetic_mask = tax_units['is_synthetic_ultra_high'] == True
        synthetic_tax = (tax_units.loc[synthetic_mask, 'hi_state_tax'] * tax_units.loc[synthetic_mask, 'weight']).sum() / 1_000_000
        logger.info(f"   Synthetic ultra-high-income tax (after cap gains): ${synthetic_tax:,.1f}M")
        logger.info(f"   Total tax (all units, after cap gains): ${(tax_units['hi_state_tax'] * tax_units['weight']).sum() / 1_000_000:,.1f}M")
    else:
        logger.info("\n⚠️  Capital gains EXCLUDED from this run")
        tax_units['capital_gains'] = 0
        tax_units['agi_without_cap_gains'] = tax_units['agi']
    
    logger.info("\n🏠 Applying itemized deductions...")
    deduction_estimator = ItemizedDeductionEstimator()
    
    # Calculate itemized vs standard deductions
    itemized_deductions = []
    for _, row in tax_units.iterrows():
        agi = row.get('agi', row.get('income', 0))
        filing_status = row['filing_status']
        
        # Get standard deduction amount (2022 Hawaii) - CORRECTED VALUES
        standard_amounts = {
            'single': 2200,
            'married_filing_jointly': 4400,
            'married_filing_separately': 2200,
            'head_of_household': 3212,
            'qualifying_widow': 4400
        }
        standard_deduction = standard_amounts.get(filing_status, 2200)
        
        # Get deduction (standard or itemized)
        deduction_info = deduction_estimator.get_deduction(
            agi, filing_status, standard_deduction
        )
        
        itemized_deductions.append(deduction_info['amount'])
    
    tax_units['total_deductions'] = itemized_deductions
    
    # =========================================================================
    # CALIBRATION
    # =========================================================================
    #
    # Calculate taxes once with deductions already applied, then calibrate
    # weights and tax amounts simultaneously against DOTAX targets.

    # --- Calculate taxes (includes deductions) for all units ---
    logger.info("\n💰 Calculating taxes for all units (pre-calibration)...")

    calculator = load_tax_data()
    filing_status_map = {
        'single': 'Single_Married_Separate',
        'married_filing_jointly': 'Joint_Surviving_Spouse',
        'joint': 'Joint_Surviving_Spouse',          # legacy alias
        'married_filing_separately': 'Single_Married_Separate',
        'married_filing_separate': 'Single_Married_Separate',  # legacy alias
        'head_of_household': 'Head_of_Household',
        'qualifying_widow': 'Joint_Surviving_Spouse',
    }
    tax_units['filing_status_hawaii'] = tax_units['filing_status'].map(filing_status_map)

    tax_results = calculator.calculate_tax_for_dataframe(
        tax_units,
        income_col='agi',
        filing_status_col='filing_status_hawaii',
        year=2017  # Use 2017 rates to match $3,029M benchmark
    )
    tax_units = _apply_tax_results(tax_units, tax_results)

    pre_cal_tax = (tax_units['hi_state_tax'] * tax_units['weight']).sum() / 1_000_000
    logger.info(f"  Pre-calibration total tax: ${pre_cal_tax:,.1f}M  (target: $3,029M)")

    # --- Simultaneous two-phase calibration ---
    calibrator = SimultaneousCalibrator()
    tax_units = calibrator.calibrate(tax_units)

    # Determine weight column
    weight_col = 'weight' if 'weight' in tax_units.columns else 'PWGTP'

    # Analyze filing status distribution
    logger.info("\n" + "="*80)
    logger.info("FILING STATUS DISTRIBUTION")
    logger.info("="*80)
    
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
    output_file = f'data/processed/tax_units_filing_status_calibrated_{timestamp}.parquet'
    
    logger.info(f"\nSaving to: {output_file}")
    
    # Fix parquet serialization: convert mixed-type object columns to string, but skip
    # list/dict columns (e.g. 'dependents') because astype(str) would turn them into
    # a literal string representation "['id1', 'id2']" that can't be deserialized.
    for col in tax_units.columns:
        if tax_units[col].dtype != 'object':
            continue
        sample = tax_units[col].dropna()
        if sample.empty:
            continue
        first_val = sample.iloc[0]
        if isinstance(first_val, (list, dict)):
            # Leave list/dict columns as-is; PyArrow handles them natively
            continue
        tax_units[col] = tax_units[col].astype(str)
    
    tax_units.to_parquet(output_file, index=False)
    
    logger.info("\n" + "="*80)
    logger.info("✅ HYBRID CALIBRATION PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info("\nCalibration Method: Hybrid (Sequential Structural Corrections + IPF Fine-Tuning)")
    logger.info(f"Final Revenue: ${(tax_units['hi_state_tax'] * tax_units['weight']).sum() / 1_000_000:,.1f}M vs $3,029.0M target")
    
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
    
    # Validate against Hawaii tax liability benchmarks
    logger.info("\n" + "="*80)
    logger.info("HAWAII TAX LIABILITY VALIDATION (DOTAX SOI 2022 - Before Credits)")
    logger.info("="*80)
    
    # DOTAX benchmarks (in millions)
    dotax_tax_benchmarks = {
        'married_filing_jointly': 1674,
        'single': 864,
        'married_filing_separately': 289,
        'head_of_household': 202,
        'total': 3029
    }
    
    # Calculate weighted tax by filing status
    tax_by_status = {}
    for status in ['married_filing_jointly', 'single', 'married_filing_separately', 'head_of_household']:
        status_df = tax_units[tax_units['filing_status'] == status]
        if len(status_df) > 0:
            weighted_tax = (status_df['hi_state_tax'] * status_df['weight']).sum() / 1_000_000
            tax_by_status[status] = weighted_tax
        else:
            tax_by_status[status] = 0
    
    total_tax = sum(tax_by_status.values())
    
    print("\nStatus                    | Model ($M) | DOTAX ($M) | Difference | % Diff")
    print("--------------------------|------------|------------|------------|--------")
    
    for status, benchmark in dotax_tax_benchmarks.items():
        if status == 'total':
            continue
        model_tax = tax_by_status.get(status, 0)
        diff = model_tax - benchmark
        pct_diff = (diff / benchmark * 100) if benchmark > 0 else 0
        
        status_name = status.replace('_', ' ').title()
        print(f"{status_name:<25} | ${model_tax:>9.1f} | ${benchmark:>9.0f} | ${diff:>+9.1f} | {pct_diff:>+6.1f}%")
    
    total_diff = total_tax - dotax_tax_benchmarks['total']
    total_pct_diff = (total_diff / dotax_tax_benchmarks['total'] * 100)
    print(f"\nTOTAL                     | ${total_tax:>9.1f} | ${dotax_tax_benchmarks['total']:>9.0f} | ${total_diff:>+9.1f} | {total_pct_diff:>+6.1f}%")
    
    # Validate against DOTAX Table 12A (tax liability by AGI bracket)
    from src.tax.validation.dotax_table_12a import validate_against_table_12a
    
    logger.info("\n")
    table_12a_results = validate_against_table_12a(
        tax_units,
        weight_col='weight',
        agi_col='agi',
        tax_col='hi_state_tax'
    )
    
    # Save Table 12A validation results
    validation_output = output_file.replace('tax_units_calibrated', 'validation_table_12a').replace('.parquet', '.csv')
    table_12a_results.to_csv(validation_output, index=False)
    logger.info(f"\n💾 Table 12A validation saved to: {validation_output}")
    
    logger.info(f"\nOutput file: {output_file}")


if __name__ == '__main__':
    main()
