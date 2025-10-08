#!/usr/bin/env python3
"""
Generate comprehensive district-level CTC estimates using IRS SOI-calibrated tax units.
Includes validation against IRS benchmarks and comprehensive reporting.
"""

import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
import sys
import os
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from tax.credits.ctc import ChildTaxCredit
from analysis.district_ctc import DistrictCTCAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_district_calibrated_tax_units():
    """Load the district-calibrated tax units."""
    logger.info("Loading district-calibrated tax units...")
    
    tax_units_path = Path('src/data/processed/tax_units_district_calibrated.parquet')
    if not tax_units_path.exists():
        logger.error(f"District-calibrated tax units not found: {tax_units_path}")
        return None
    
    df = pd.read_parquet(tax_units_path)
    logger.info(f"Loaded {len(df)} district-calibrated tax units")
    
    # Report calibration coverage
    calibration_summary = df['calibration_level'].value_counts()
    logger.info("Calibration coverage:")
    for level, count in calibration_summary.items():
        pct = count / len(df) * 100
        logger.info(f"  {level}: {count:,} units ({pct:.1f}%)")
    
    return df

def load_irs_benchmarks():
    """Load IRS SOI benchmarks for validation."""
    logger.info("Loading IRS benchmarks...")
    
    benchmarks_path = Path('data/irs_soi/district_benchmarks.json')
    if not benchmarks_path.exists():
        logger.error(f"IRS benchmarks not found: {benchmarks_path}")
        return None
    
    with open(benchmarks_path, 'r') as f:
        benchmarks = json.load(f)
    
    logger.info(f"Loaded benchmarks for {len(benchmarks.get('house_districts', {}))} House districts")
    logger.info(f"Loaded benchmarks for {len(benchmarks.get('senate_districts', {}))} Senate districts")
    
    return benchmarks

def calculate_district_ctc_estimates(tax_units_df):
    """Calculate CTC estimates by district."""
    logger.info("Calculating district-level CTC estimates...")
    
    # Initialize CTC calculator
    ctc_calculator = ChildTaxCredit()
    
    # Calculate CTC for each tax unit
    logger.info("Computing CTC for all tax units...")
    tax_units_with_ctc = []
    
    for idx, unit in tax_units_df.iterrows():
        if idx % 5000 == 0:
            logger.info(f"Processed {idx:,}/{len(tax_units_df):,} tax units ({idx/len(tax_units_df)*100:.1f}%)")
        
        # Extract tax unit information
        filing_status = unit.get('filing_status', 'single')
        income = float(unit.get('income', 0))
        dependents = unit.get('dependents', [])
        
        # Handle dependents - could be stored as string, list, or count
        if isinstance(dependents, str):
            try:
                dependents = eval(dependents) if dependents else []
            except:
                dependents = []
        elif pd.isna(dependents):
            dependents = []
        elif isinstance(dependents, (int, float)):
            # If it's a count, create dummy dependent list
            dependents = [{'age': 10} for _ in range(int(dependents))]
        
        # Calculate CTC
        try:
            ctc_result = ctc_calculator.calculate_credit(
                filing_status=filing_status,
                agi=income,
                dependents=dependents
            )
            
            unit_with_ctc = unit.to_dict()
            unit_with_ctc.update({
                'ctc_amount': ctc_result['total_credit'],
                'ctc_refundable': ctc_result['refundable_credit'],
                'ctc_nonrefundable': ctc_result['nonrefundable_credit'],
                'qualifying_children': len(dependents),
                'has_ctc': ctc_result['total_credit'] > 0
            })
            
            tax_units_with_ctc.append(unit_with_ctc)
            
        except Exception as e:
            logger.warning(f"Error calculating CTC for unit {idx}: {e}")
            unit_with_ctc = unit.to_dict()
            unit_with_ctc.update({
                'ctc_amount': 0,
                'ctc_refundable': 0,
                'ctc_nonrefundable': 0,
                'qualifying_children': 0,
                'has_ctc': False
            })
            tax_units_with_ctc.append(unit_with_ctc)
    
    logger.info("Completed CTC calculations")
    return pd.DataFrame(tax_units_with_ctc)

def aggregate_by_districts(tax_units_with_ctc):
    """Aggregate CTC estimates by legislative districts."""
    logger.info("Aggregating CTC estimates by districts...")
    
    results = {}
    
    # House Districts
    house_districts = tax_units_with_ctc.groupby('house_district').agg({
        'ctc_amount': ['sum', 'mean', 'count'],
        'ctc_refundable': 'sum',
        'ctc_nonrefundable': 'sum',
        'qualifying_children': 'sum',
        'has_ctc': 'sum',
        'income': ['mean', 'median'],
        'household_weight': 'sum',
        'calibration_level': lambda x: x.mode().iloc[0] if len(x) > 0 else 'unknown'
    }).round(2)
    
    # Flatten column names
    house_districts.columns = [
        'total_ctc_amount', 'avg_ctc_per_unit', 'total_tax_units',
        'total_ctc_refundable', 'total_ctc_nonrefundable', 
        'total_qualifying_children', 'ctc_recipient_units',
        'avg_income', 'median_income', 'total_population_weight',
        'calibration_method'
    ]
    
    # Calculate additional metrics
    house_districts['avg_ctc_per_recipient'] = (
        house_districts['total_ctc_amount'] / house_districts['ctc_recipient_units']
    ).fillna(0)
    house_districts['ctc_recipient_rate'] = (
        house_districts['ctc_recipient_units'] / house_districts['total_tax_units'] * 100
    )
    house_districts['district_type'] = 'house'
    house_districts['district'] = house_districts.index
    
    results['house_districts'] = house_districts.reset_index()
    
    # Senate Districts
    senate_districts = tax_units_with_ctc.groupby('senate_district').agg({
        'ctc_amount': ['sum', 'mean', 'count'],
        'ctc_refundable': 'sum',
        'ctc_nonrefundable': 'sum',
        'qualifying_children': 'sum',
        'has_ctc': 'sum',
        'income': ['mean', 'median'],
        'household_weight': 'sum',
        'calibration_level': lambda x: x.mode().iloc[0] if len(x) > 0 else 'unknown'
    }).round(2)
    
    # Flatten column names
    senate_districts.columns = [
        'total_ctc_amount', 'avg_ctc_per_unit', 'total_tax_units',
        'total_ctc_refundable', 'total_ctc_nonrefundable',
        'total_qualifying_children', 'ctc_recipient_units',
        'avg_income', 'median_income', 'total_population_weight',
        'calibration_method'
    ]
    
    # Calculate additional metrics
    senate_districts['avg_ctc_per_recipient'] = (
        senate_districts['total_ctc_amount'] / senate_districts['ctc_recipient_units']
    ).fillna(0)
    senate_districts['ctc_recipient_rate'] = (
        senate_districts['ctc_recipient_units'] / senate_districts['total_tax_units'] * 100
    )
    senate_districts['district_type'] = 'senate'
    senate_districts['district'] = senate_districts.index
    
    results['senate_districts'] = senate_districts.reset_index()
    
    logger.info(f"Aggregated results for {len(house_districts)} House districts and {len(senate_districts)} Senate districts")
    
    return results

def validate_against_irs_benchmarks(district_results, irs_benchmarks):
    """Validate district CTC estimates against IRS SOI benchmarks."""
    logger.info("Validating district estimates against IRS benchmarks...")
    
    validation_results = []
    
    # Validate House districts
    house_df = district_results['house_districts']
    house_benchmarks = irs_benchmarks.get('house_districts', {})
    
    for _, row in house_df.iterrows():
        district_id = str(int(row['house_district']))
        
        if district_id in house_benchmarks:
            benchmark = house_benchmarks[district_id]
            
            # Compare total returns vs total tax units
            irs_returns = benchmark['total_returns']
            pums_units = row['total_tax_units']
            
            # Calculate validation metrics
            unit_ratio = pums_units / irs_returns if irs_returns > 0 else 0
            
            validation_results.append({
                'district_type': 'house',
                'district_id': int(row['house_district']),
                'irs_total_returns': irs_returns,
                'pums_total_units': pums_units,
                'unit_ratio': unit_ratio,
                'total_ctc_amount': row['total_ctc_amount'],
                'avg_ctc_per_unit': row['avg_ctc_per_unit'],
                'calibration_method': row['calibration_method'],
                'data_quality_score': benchmark['data_quality']['confidence_score']
            })
    
    # Validate Senate districts
    senate_df = district_results['senate_districts']
    senate_benchmarks = irs_benchmarks.get('senate_districts', {})
    
    for _, row in senate_df.iterrows():
        district_id = str(int(row['senate_district']))
        
        if district_id in senate_benchmarks:
            benchmark = senate_benchmarks[district_id]
            
            # Compare total returns vs total tax units
            irs_returns = benchmark['total_returns']
            pums_units = row['total_tax_units']
            
            # Calculate validation metrics
            unit_ratio = pums_units / irs_returns if irs_returns > 0 else 0
            
            validation_results.append({
                'district_type': 'senate',
                'district_id': int(row['senate_district']),
                'irs_total_returns': irs_returns,
                'pums_total_units': pums_units,
                'unit_ratio': unit_ratio,
                'total_ctc_amount': row['total_ctc_amount'],
                'avg_ctc_per_unit': row['avg_ctc_per_unit'],
                'calibration_method': row['calibration_method'],
                'data_quality_score': benchmark['data_quality']['confidence_score']
            })
    
    validation_df = pd.DataFrame(validation_results)
    
    if not validation_df.empty:
        logger.info("Validation summary:")
        logger.info(f"  Districts validated: {len(validation_df)}")
        logger.info(f"  Average unit ratio (PUMS/IRS): {validation_df['unit_ratio'].mean():.2f}")
        logger.info(f"  Unit ratio std dev: {validation_df['unit_ratio'].std():.2f}")
        
        # Identify outliers
        outliers = validation_df[
            (validation_df['unit_ratio'] < 0.5) | (validation_df['unit_ratio'] > 2.0)
        ]
        if len(outliers) > 0:
            logger.warning(f"Found {len(outliers)} districts with unusual unit ratios:")
            for _, outlier in outliers.head(5).iterrows():
                logger.warning(f"  {outlier['district_type'].title()} District {outlier['district_id']}: ratio = {outlier['unit_ratio']:.2f}")
    
    return validation_df

def create_comprehensive_report(district_results, validation_df, tax_units_with_ctc):
    """Create comprehensive district analysis report."""
    logger.info("Creating comprehensive district analysis report...")
    
    # Calculate statewide totals
    statewide_totals = {
        'total_tax_units': len(tax_units_with_ctc),
        'total_ctc_amount': tax_units_with_ctc['ctc_amount'].sum(),
        'total_ctc_recipients': tax_units_with_ctc['has_ctc'].sum(),
        'total_qualifying_children': tax_units_with_ctc['qualifying_children'].sum(),
        'avg_ctc_per_unit': tax_units_with_ctc['ctc_amount'].mean(),
        'avg_ctc_per_recipient': tax_units_with_ctc[tax_units_with_ctc['has_ctc']]['ctc_amount'].mean(),
        'ctc_recipient_rate': tax_units_with_ctc['has_ctc'].mean() * 100
    }
    
    # District-level analysis
    house_df = district_results['house_districts']
    senate_df = district_results['senate_districts']
    
    # Top performing districts
    top_house_ctc = house_df.nlargest(5, 'total_ctc_amount')[['house_district', 'total_ctc_amount', 'total_tax_units']]
    top_senate_ctc = senate_df.nlargest(5, 'total_ctc_amount')[['senate_district', 'total_ctc_amount', 'total_tax_units']]
    
    # Create summary report
    report = {
        'analysis_metadata': {
            'generated_at': datetime.now().isoformat(),
            'total_districts_analyzed': len(house_df) + len(senate_df),
            'house_districts': len(house_df),
            'senate_districts': len(senate_df),
            'validation_coverage': len(validation_df) if validation_df is not None else 0
        },
        'statewide_summary': {
            'total_tax_units': f"{statewide_totals['total_tax_units']:,}",
            'total_ctc_amount': f"${statewide_totals['total_ctc_amount']:,.0f}",
            'total_ctc_recipients': f"{statewide_totals['total_ctc_recipients']:,}",
            'total_qualifying_children': f"{statewide_totals['total_qualifying_children']:,}",
            'avg_ctc_per_unit': f"${statewide_totals['avg_ctc_per_unit']:,.0f}",
            'avg_ctc_per_recipient': f"${statewide_totals['avg_ctc_per_recipient']:,.0f}",
            'ctc_recipient_rate': f"{statewide_totals['ctc_recipient_rate']:.1f}%"
        },
        'district_analysis': {
            'house_districts': {
                'total_analyzed': len(house_df),
                'total_ctc_amount': f"${house_df['total_ctc_amount'].sum():,.0f}",
                'highest_ctc_district': int(house_df.loc[house_df['total_ctc_amount'].idxmax(), 'house_district']),
                'highest_ctc_amount': f"${house_df['total_ctc_amount'].max():,.0f}",
                'avg_ctc_per_district': f"${house_df['total_ctc_amount'].mean():,.0f}",
                'top_5_districts': [
                    {
                        'district': int(row['house_district']),
                        'ctc_amount': f"${row['total_ctc_amount']:,.0f}",
                        'tax_units': int(row['total_tax_units'])
                    }
                    for _, row in top_house_ctc.iterrows()
                ]
            },
            'senate_districts': {
                'total_analyzed': len(senate_df),
                'total_ctc_amount': f"${senate_df['total_ctc_amount'].sum():,.0f}",
                'highest_ctc_district': int(senate_df.loc[senate_df['total_ctc_amount'].idxmax(), 'senate_district']),
                'highest_ctc_amount': f"${senate_df['total_ctc_amount'].max():,.0f}",
                'avg_ctc_per_district': f"${senate_df['total_ctc_amount'].mean():,.0f}",
                'top_5_districts': [
                    {
                        'district': int(row['senate_district']),
                        'ctc_amount': f"${row['total_ctc_amount']:,.0f}",
                        'tax_units': int(row['total_tax_units'])
                    }
                    for _, row in top_senate_ctc.iterrows()
                ]
            }
        }
    }
    
    # Add validation summary if available
    if validation_df is not None and not validation_df.empty:
        report['validation_summary'] = {
            'districts_validated': len(validation_df),
            'avg_unit_ratio': f"{validation_df['unit_ratio'].mean():.2f}",
            'unit_ratio_std': f"{validation_df['unit_ratio'].std():.2f}",
            'high_quality_districts': int((validation_df['data_quality_score'] >= 80).sum()),
            'medium_quality_districts': int(((validation_df['data_quality_score'] >= 50) & (validation_df['data_quality_score'] < 80)).sum()),
            'low_quality_districts': int((validation_df['data_quality_score'] < 50).sum())
        }
    
    return report

def save_results(district_results, validation_df, comprehensive_report, tax_units_with_ctc):
    """Save all analysis results."""
    logger.info("Saving district CTC analysis results...")
    
    # Create output directory
    output_dir = Path('output/district_ctc_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save district-level results
    house_df = district_results['house_districts']
    senate_df = district_results['senate_districts']
    
    house_df.to_csv(output_dir / 'house_district_ctc_estimates.csv', index=False)
    senate_df.to_csv(output_dir / 'senate_district_ctc_estimates.csv', index=False)
    logger.info(f"Saved district CTC estimates: {output_dir}")
    
    # Save validation results
    if validation_df is not None:
        validation_df.to_csv(output_dir / 'district_validation_results.csv', index=False)
        logger.info("Saved validation results")
    
    # Save comprehensive report
    with open(output_dir / 'comprehensive_district_report.json', 'w') as f:
        json.dump(comprehensive_report, f, indent=2)
    logger.info("Saved comprehensive report")
    
    # Save detailed tax units with CTC
    tax_units_with_ctc.to_parquet(output_dir / 'tax_units_with_district_ctc.parquet')
    logger.info("Saved detailed tax units with CTC calculations")

def main():
    """Main analysis function."""
    logger.info("Starting comprehensive district-level CTC analysis with IRS validation...")
    
    # Load data
    tax_units_df = load_district_calibrated_tax_units()
    if tax_units_df is None:
        return
    
    irs_benchmarks = load_irs_benchmarks()
    
    # Calculate CTC estimates
    tax_units_with_ctc = calculate_district_ctc_estimates(tax_units_df)
    
    # Aggregate by districts
    district_results = aggregate_by_districts(tax_units_with_ctc)
    
    # Validate against IRS benchmarks
    validation_df = None
    if irs_benchmarks:
        validation_df = validate_against_irs_benchmarks(district_results, irs_benchmarks)
    
    # Create comprehensive report
    comprehensive_report = create_comprehensive_report(district_results, validation_df, tax_units_with_ctc)
    
    # Save results
    save_results(district_results, validation_df, comprehensive_report, tax_units_with_ctc)
    
    # Summary output
    logger.info("\n" + "="*60)
    logger.info("DISTRICT-LEVEL CTC ANALYSIS COMPLETE")
    logger.info("="*60)
    
    statewide = comprehensive_report['statewide_summary']
    logger.info(f"Statewide CTC Impact:")
    logger.info(f"  Total CTC Amount: {statewide['total_ctc_amount']}")
    logger.info(f"  Total Recipients: {statewide['total_ctc_recipients']}")
    logger.info(f"  Average per Recipient: {statewide['avg_ctc_per_recipient']}")
    logger.info(f"  Recipient Rate: {statewide['ctc_recipient_rate']}")
    
    house_analysis = comprehensive_report['district_analysis']['house_districts']
    senate_analysis = comprehensive_report['district_analysis']['senate_districts']
    
    logger.info(f"\nDistrict Analysis:")
    logger.info(f"  House Districts Analyzed: {house_analysis['total_analyzed']}")
    logger.info(f"  Senate Districts Analyzed: {senate_analysis['total_analyzed']}")
    logger.info(f"  Highest CTC House District: {house_analysis['highest_ctc_district']} ({house_analysis['highest_ctc_amount']})")
    logger.info(f"  Highest CTC Senate District: {senate_analysis['highest_ctc_district']} ({senate_analysis['highest_ctc_amount']})")
    
    if 'validation_summary' in comprehensive_report:
        validation = comprehensive_report['validation_summary']
        logger.info(f"\nValidation Results:")
        logger.info(f"  Districts Validated: {validation['districts_validated']}")
        logger.info(f"  High Quality Districts: {validation['high_quality_districts']}")
        logger.info(f"  Average Unit Ratio: {validation['avg_unit_ratio']}")
    
    logger.info(f"\nOutput files saved to: output/district_ctc_analysis/")
    logger.info("✅ District-level CTC reintroduction complete with IRS SOI validation!")
    
    return {
        'district_results': district_results,
        'validation_results': validation_df,
        'comprehensive_report': comprehensive_report,
        'tax_units_with_ctc': tax_units_with_ctc
    }

if __name__ == "__main__":
    results = main()
