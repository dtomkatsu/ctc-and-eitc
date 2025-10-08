#!/usr/bin/env python3
"""
Implement district-specific filing status calibration for Hawaii tax unit construction.
Uses IRS SOI district benchmarks to calibrate tax units at the legislative district level.
"""

import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from tax.units.constructor import TaxUnitConstructor
from tax.units.status.irs_based import calibrate_to_soi_totals
from data.pums_loader import PUMSDataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_district_benchmarks():
    """Load IRS SOI district benchmarks."""
    logger.info("Loading district benchmarks...")
    
    benchmarks_path = Path('data/irs_soi/district_benchmarks.json')
    if not benchmarks_path.exists():
        logger.error(f"District benchmarks not found: {benchmarks_path}")
        return None
    
    with open(benchmarks_path, 'r') as f:
        benchmarks = json.load(f)
    
    logger.info(f"Loaded benchmarks for {len(benchmarks.get('house_districts', {}))} House districts")
    logger.info(f"Loaded benchmarks for {len(benchmarks.get('senate_districts', {}))} Senate districts")
    
    return benchmarks

def load_zip_district_mapping():
    """Load ZIP-to-district mapping for tax unit assignment."""
    logger.info("Loading ZIP-district mapping...")
    
    # Load the original PUMA-district crosswalk directly
    crosswalk_path = Path('data/crosswalks/hawaii_puma_districts_official_2022.csv')
    if not crosswalk_path.exists():
        logger.error(f"PUMA-district crosswalk not found: {crosswalk_path}")
        return None
    
    df = pd.read_csv(crosswalk_path)
    
    # Create mapping from PUMA to districts with multiple format support
    puma_district_map = {}
    
    for _, row in df.iterrows():
        puma = row['PUMA']
        house_dist = row['house_district'] 
        senate_dist = row['senate_district']
        
        # Support multiple PUMA formats to handle tax units format
        puma_formats = [
            str(puma).zfill(4),  # 4-digit format like '0301' (tax units format)
            str(puma).zfill(3),  # 3-digit format like '301'
            str(puma),           # Original format
            f"{puma}.0"          # Float format
        ]
        
        for puma_fmt in puma_formats:
            puma_district_map[puma_fmt] = {
                'house_district': house_dist,
                'senate_district': senate_dist
            }
    
    logger.info(f"Created PUMA-district mapping for {len(set(df['PUMA']))} unique PUMAs with format variations")
    
    # Log sample mappings for verification
    sample_pumas = ['0301', '0200', '0304', '0100']  # Common tax units PUMA formats
    logger.info("Sample PUMA mappings:")
    for puma in sample_pumas:
        if puma in puma_district_map:
            mapping = puma_district_map[puma]
            logger.info(f"  {puma} -> House: {mapping['house_district']}, Senate: {mapping['senate_district']}")
    
    return puma_district_map

def assign_districts_to_tax_units(tax_units_df, puma_district_map):
    """Assign legislative districts to tax units based on PUMA."""
    logger.info("Assigning districts to tax units...")
    
    # Ensure PUMA column exists and is properly formatted
    if 'PUMA' not in tax_units_df.columns:
        logger.error("PUMA column not found in tax units data")
        return tax_units_df
    
    # Use PUMA as-is (already in correct 4-digit format for tax units)
    tax_units_df['puma_str'] = tax_units_df['PUMA'].astype(str)
    
    # Map districts using the PUMA format from tax units
    tax_units_df['house_district'] = tax_units_df['puma_str'].map(
        lambda x: puma_district_map.get(x, {}).get('house_district', None)
    )
    tax_units_df['senate_district'] = tax_units_df['puma_str'].map(
        lambda x: puma_district_map.get(x, {}).get('senate_district', None)
    )
    
    # Report assignment success
    house_assigned = tax_units_df['house_district'].notna().sum()
    senate_assigned = tax_units_df['senate_district'].notna().sum()
    total_units = len(tax_units_df)
    
    logger.info(f"Assigned House districts to {house_assigned}/{total_units} tax units ({house_assigned/total_units*100:.1f}%)")
    logger.info(f"Assigned Senate districts to {senate_assigned}/{total_units} tax units ({senate_assigned/total_units*100:.1f}%)")
    
    return tax_units_df

def calibrate_district_filing_status(tax_units_df, benchmarks, district_type='house'):
    """Calibrate filing status distribution by district using IRS SOI targets."""
    logger.info(f"Calibrating {district_type} district filing status...")
    
    district_col = f'{district_type}_district'
    benchmark_key = f'{district_type}_districts'
    
    if benchmark_key not in benchmarks:
        logger.error(f"No benchmarks found for {district_type} districts")
        return tax_units_df
    
    district_benchmarks = benchmarks[benchmark_key]
    calibrated_units = []
    
    # Get unique districts in the data
    districts_in_data = tax_units_df[district_col].dropna().unique()
    logger.info(f"Calibrating {len(districts_in_data)} {district_type} districts")
    
    for district_id in districts_in_data:
        district_str = str(int(district_id))
        district_units = tax_units_df[tax_units_df[district_col] == district_id].copy()
        
        if district_str in district_benchmarks:
            # Get IRS SOI targets for this district
            targets = district_benchmarks[district_str]['filing_status_targets']
            confidence = district_benchmarks[district_str]['data_quality']['confidence_score']
            
            # Only calibrate districts with reasonable confidence
            if confidence >= 30:  # Minimum confidence threshold
                logger.info(f"Calibrating {district_type} district {district_id} with {len(district_units)} units (confidence: {confidence})")
                
                # Apply district-specific calibration (using joint percentage only for now)
                # Convert DataFrame to list of dicts for calibration function
                district_units_list = district_units.to_dict('records')
                calibrated_units_list = calibrate_to_soi_totals(
                    district_units_list,
                    target_joint_pct=targets['joint_pct']/100
                )
                # Convert back to DataFrame
                calibrated_district_units = pd.DataFrame(calibrated_units_list)
                
                # Mark as district-calibrated
                calibrated_district_units['calibration_level'] = f'{district_type}_district'
                calibrated_district_units['calibration_confidence'] = confidence
                
                calibrated_units.append(calibrated_district_units)
            else:
                logger.warning(f"Skipping {district_type} district {district_id} due to low confidence ({confidence})")
                district_units['calibration_level'] = 'uncalibrated'
                district_units['calibration_confidence'] = confidence
                calibrated_units.append(district_units)
        else:
            logger.warning(f"No IRS benchmarks found for {district_type} district {district_id}")
            district_units['calibration_level'] = 'no_benchmark'
            district_units['calibration_confidence'] = 0
            calibrated_units.append(district_units)
    
    # Handle units without district assignment
    unassigned_units = tax_units_df[tax_units_df[district_col].isna()].copy()
    if len(unassigned_units) > 0:
        logger.warning(f"Found {len(unassigned_units)} tax units without {district_type} district assignment")
        unassigned_units['calibration_level'] = 'unassigned'
        unassigned_units['calibration_confidence'] = 0
        calibrated_units.append(unassigned_units)
    
    # Combine all calibrated units
    result_df = pd.concat(calibrated_units, ignore_index=True)
    
    logger.info(f"Completed {district_type} district calibration for {len(result_df)} tax units")
    return result_df

def validate_district_calibration(calibrated_units, benchmarks, district_type='house'):
    """Validate the district calibration results against IRS SOI targets."""
    logger.info(f"Validating {district_type} district calibration...")
    
    district_col = f'{district_type}_district'
    benchmark_key = f'{district_type}_districts'
    
    validation_results = []
    
    # Group by district and calculate actual filing status distributions
    district_groups = calibrated_units.groupby(district_col)
    
    for district_id, group in district_groups:
        if pd.isna(district_id):
            continue
            
        district_str = str(int(district_id))
        
        # Calculate actual distributions
        filing_status_counts = group['filing_status'].value_counts()
        total_units = len(group)
        
        actual_pcts = {
            'single': filing_status_counts.get('single', 0) / total_units * 100,
            'joint': filing_status_counts.get('married_filing_jointly', 0) / total_units * 100,
            'hoh': filing_status_counts.get('head_of_household', 0) / total_units * 100,
            'mfs': filing_status_counts.get('married_filing_separate', 0) / total_units * 100
        }
        
        # Get IRS targets if available
        if district_str in benchmarks.get(benchmark_key, {}):
            targets = benchmarks[benchmark_key][district_str]['filing_status_targets']
            
            # Calculate differences
            differences = {
                'single': actual_pcts['single'] - targets['single_pct'],
                'joint': actual_pcts['joint'] - targets['joint_pct'],
                'hoh': actual_pcts['hoh'] - targets['hoh_pct'],
                'mfs': actual_pcts['mfs'] - targets['mfs_pct']
            }
            
            # Calculate overall accuracy (average absolute difference)
            accuracy = 100 - np.mean([abs(diff) for diff in differences.values()])
            
            validation_results.append({
                'district_type': district_type,
                'district_id': district_id,
                'total_units': total_units,
                'accuracy_score': accuracy,
                'single_diff': differences['single'],
                'joint_diff': differences['joint'],
                'hoh_diff': differences['hoh'],
                'mfs_diff': differences['mfs'],
                'calibration_level': group['calibration_level'].iloc[0] if 'calibration_level' in group.columns else 'unknown'
            })
    
    validation_df = pd.DataFrame(validation_results)
    
    if not validation_df.empty:
        logger.info(f"Validation results for {len(validation_df)} {district_type} districts:")
        logger.info(f"  Average accuracy: {validation_df['accuracy_score'].mean():.1f}%")
        logger.info(f"  Districts with >90% accuracy: {(validation_df['accuracy_score'] > 90).sum()}")
        logger.info(f"  Districts with >80% accuracy: {(validation_df['accuracy_score'] > 80).sum()}")
        
        # Show worst performing districts
        worst_districts = validation_df.nsmallest(3, 'accuracy_score')
        if len(worst_districts) > 0:
            logger.warning("Districts with lowest accuracy:")
            for _, row in worst_districts.iterrows():
                logger.warning(f"  District {int(row['district_id'])}: {row['accuracy_score']:.1f}% accuracy")
    
    return validation_df

def main():
    """Main calibration implementation."""
    logger.info("Starting district-specific filing status calibration...")
    
    # Load required data
    benchmarks = load_district_benchmarks()
    if benchmarks is None:
        return
    
    puma_district_map = load_zip_district_mapping()
    if puma_district_map is None:
        return
    
    # Load existing tax units
    logger.info("Loading existing tax units...")
    tax_units_path = Path('src/data/processed/tax_units_weight_split_raked.parquet')
    if not tax_units_path.exists():
        logger.error(f"Tax units file not found: {tax_units_path}")
        return
    
    tax_units_df = pd.read_parquet(tax_units_path)
    logger.info(f"Loaded {len(tax_units_df)} tax units")
    
    # Assign districts to tax units
    tax_units_with_districts = assign_districts_to_tax_units(tax_units_df, puma_district_map)
    
    # Calibrate by House districts
    house_calibrated = calibrate_district_filing_status(
        tax_units_with_districts, benchmarks, district_type='house'
    )
    
    # Calibrate by Senate districts (optional - can use House calibration as base)
    senate_calibrated = calibrate_district_filing_status(
        house_calibrated, benchmarks, district_type='senate'
    )
    
    # Validate calibration results
    house_validation = validate_district_calibration(
        house_calibrated, benchmarks, district_type='house'
    )
    senate_validation = validate_district_calibration(
        senate_calibrated, benchmarks, district_type='senate'
    )
    
    # Save calibrated tax units
    output_path = Path('src/data/processed/tax_units_district_calibrated.parquet')
    senate_calibrated.to_parquet(output_path)
    logger.info(f"Saved district-calibrated tax units: {output_path}")
    
    # Save validation results
    if not house_validation.empty:
        house_validation.to_csv('data/irs_soi/house_district_validation.csv', index=False)
        logger.info("Saved House district validation results")
    
    if not senate_validation.empty:
        senate_validation.to_csv('data/irs_soi/senate_district_validation.csv', index=False)
        logger.info("Saved Senate district validation results")
    
    # Summary report
    logger.info("\n" + "="*60)
    logger.info("DISTRICT CALIBRATION COMPLETE")
    logger.info("="*60)
    
    # Report calibration coverage
    calibration_summary = senate_calibrated['calibration_level'].value_counts()
    logger.info("Calibration coverage:")
    for level, count in calibration_summary.items():
        pct = count / len(senate_calibrated) * 100
        logger.info(f"  {level}: {count:,} units ({pct:.1f}%)")
    
    # Report overall filing status distribution
    final_filing_status = senate_calibrated['filing_status'].value_counts()
    total_units = len(senate_calibrated)
    logger.info("\nFinal filing status distribution:")
    for status, count in final_filing_status.items():
        pct = count / total_units * 100
        logger.info(f"  {status}: {count:,} units ({pct:.1f}%)")
    
    logger.info(f"\nNext steps:")
    logger.info("1. Review validation results and accuracy scores")
    logger.info("2. Update district CTC analysis pipeline to use calibrated tax units")
    logger.info("3. Generate comprehensive district-level CTC estimates")
    
    return {
        'calibrated_tax_units': senate_calibrated,
        'house_validation': house_validation,
        'senate_validation': senate_validation,
        'benchmarks': benchmarks
    }

if __name__ == "__main__":
    results = main()
