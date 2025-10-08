#!/usr/bin/env python3
"""
Aggregate IRS SOI ZIP code data by Hawaii legislative districts.
Creates district-level benchmarks for filing status, income, and CTC patterns.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_irs_soi_data():
    """Load the cleaned IRS SOI data."""
    logger.info("Loading IRS SOI data...")
    
    soi_path = Path('data/irs_soi/hawaii_irs_soi_clean.csv')
    if not soi_path.exists():
        logger.error(f"IRS SOI data not found: {soi_path}")
        return None
    
    df = pd.read_csv(soi_path)
    logger.info(f"Loaded {len(df)} ZIP code records from IRS SOI")
    
    # Clean ZIP codes - ensure 5-digit format
    df['ZIP_CODE'] = df['ZIP_CODE'].astype(str).str.zfill(5)
    
    # Filter to valid Hawaii ZIP codes (96xxx)
    hawaii_df = df[df['ZIP_CODE'].str.match(r'^96\d{3}$')].copy()
    logger.info(f"Found {len(hawaii_df)} valid Hawaii ZIP codes")
    
    return hawaii_df

def load_zip_district_crosswalk():
    """Load the ZIP-to-district crosswalk."""
    logger.info("Loading ZIP-district crosswalk...")
    
    crosswalk_path = Path('data/crosswalks/hawaii_zip_puma_district_crosswalk.csv')
    if not crosswalk_path.exists():
        logger.error(f"ZIP-district crosswalk not found: {crosswalk_path}")
        return None
    
    df = pd.read_csv(crosswalk_path)
    logger.info(f"Loaded crosswalk with {len(df)} records")
    
    # Clean ZIP codes for matching
    df['zip_code'] = df['zip_code'].astype(str).str.zfill(5)
    
    return df

def analyze_district_coverage(soi_df, crosswalk_df):
    """Analyze which districts have adequate IRS SOI data coverage."""
    logger.info("Analyzing district coverage...")
    
    # Merge SOI data with crosswalk
    merged = crosswalk_df.merge(
        soi_df[['ZIP_CODE']].drop_duplicates(),
        left_on='zip_code',
        right_on='ZIP_CODE',
        how='left',
        indicator=True
    )
    
    # Analyze coverage by district
    coverage_stats = []
    
    for district_type in ['house_district', 'senate_district']:
        district_coverage = merged.groupby(district_type).agg({
            'zip_code': 'nunique',  # Total ZIP codes in district
            '_merge': lambda x: (x == 'both').sum()  # ZIP codes with IRS data
        }).rename(columns={'_merge': 'zips_with_irs_data'})
        
        district_coverage['coverage_pct'] = (
            district_coverage['zips_with_irs_data'] / district_coverage['zip_code'] * 100
        )
        district_coverage['district_type'] = district_type
        district_coverage['district'] = district_coverage.index
        
        coverage_stats.append(district_coverage.reset_index(drop=True))
    
    coverage_df = pd.concat(coverage_stats, ignore_index=True)
    
    logger.info("District coverage summary:")
    for district_type in ['house_district', 'senate_district']:
        subset = coverage_df[coverage_df['district_type'] == district_type]
        logger.info(f"\n{district_type.replace('_', ' ').title()}:")
        logger.info(f"  Districts with >50% coverage: {(subset['coverage_pct'] > 50).sum()}")
        logger.info(f"  Districts with >80% coverage: {(subset['coverage_pct'] > 80).sum()}")
        logger.info(f"  Average coverage: {subset['coverage_pct'].mean():.1f}%")
        
        # Show districts with low coverage
        low_coverage = subset[subset['coverage_pct'] < 50]
        if len(low_coverage) > 0:
            logger.warning(f"  Districts with <50% coverage: {low_coverage['district'].tolist()}")
    
    return coverage_df

def extract_filing_status_patterns(soi_df):
    """Extract filing status patterns from IRS SOI data."""
    logger.info("Extracting filing status patterns...")
    
    # The IRS SOI data structure needs to be analyzed to identify filing status columns
    # This is a placeholder implementation that would need to be customized based on actual column structure
    
    filing_patterns = []
    
    for _, row in soi_df.iterrows():
        zip_code = row['ZIP_CODE']
        
        # Extract numeric columns that might represent filing status counts
        numeric_cols = []
        for col in soi_df.columns[1:]:  # Skip ZIP_CODE column
            try:
                val = pd.to_numeric(row[col], errors='coerce')
                if pd.notna(val) and val > 0:
                    numeric_cols.append((col, val))
            except:
                continue
        
        if len(numeric_cols) >= 4:  # Assume we need at least 4 columns for filing statuses
            # This is a simplified extraction - would need refinement based on actual data structure
            pattern = {
                'zip_code': zip_code,
                'total_returns': numeric_cols[0][1] if numeric_cols else 0,
                'single_returns': numeric_cols[1][1] if len(numeric_cols) > 1 else 0,
                'joint_returns': numeric_cols[2][1] if len(numeric_cols) > 2 else 0,
                'hoh_returns': numeric_cols[3][1] if len(numeric_cols) > 3 else 0,
                'mfs_returns': numeric_cols[4][1] if len(numeric_cols) > 4 else 0
            }
            filing_patterns.append(pattern)
    
    if filing_patterns:
        patterns_df = pd.DataFrame(filing_patterns)
        
        # Calculate filing status percentages
        total_col = 'total_returns'
        if total_col in patterns_df.columns:
            for status in ['single_returns', 'joint_returns', 'hoh_returns', 'mfs_returns']:
                if status in patterns_df.columns:
                    patterns_df[f'{status}_pct'] = (
                        patterns_df[status] / patterns_df[total_col] * 100
                    ).fillna(0)
        
        logger.info(f"Extracted filing patterns for {len(patterns_df)} ZIP codes")
        return patterns_df
    else:
        logger.warning("Could not extract filing status patterns from IRS data")
        return pd.DataFrame()

def aggregate_by_district(soi_patterns_df, crosswalk_df):
    """Aggregate IRS SOI patterns by legislative district."""
    logger.info("Aggregating data by legislative district...")
    
    if soi_patterns_df.empty:
        logger.error("No SOI patterns data to aggregate")
        return None, None
    
    # Merge patterns with district crosswalk
    district_data = crosswalk_df.merge(
        soi_patterns_df,
        left_on='zip_code',
        right_on='zip_code',
        how='inner'
    )
    
    logger.info(f"Successfully merged {len(district_data)} ZIP-district records with SOI data")
    
    # Aggregate by House districts
    house_agg = district_data.groupby('house_district').agg({
        'total_returns': 'sum',
        'single_returns': 'sum', 
        'joint_returns': 'sum',
        'hoh_returns': 'sum',
        'mfs_returns': 'sum',
        'zip_code': 'nunique'  # Number of ZIP codes per district
    }).rename(columns={'zip_code': 'num_zip_codes'})
    
    # Calculate district-level filing status percentages
    house_agg['single_pct'] = house_agg['single_returns'] / house_agg['total_returns'] * 100
    house_agg['joint_pct'] = house_agg['joint_returns'] / house_agg['total_returns'] * 100
    house_agg['hoh_pct'] = house_agg['hoh_returns'] / house_agg['total_returns'] * 100
    house_agg['mfs_pct'] = house_agg['mfs_returns'] / house_agg['total_returns'] * 100
    
    house_agg['district_type'] = 'house'
    house_agg['district'] = house_agg.index
    
    # Aggregate by Senate districts
    senate_agg = district_data.groupby('senate_district').agg({
        'total_returns': 'sum',
        'single_returns': 'sum',
        'joint_returns': 'sum', 
        'hoh_returns': 'sum',
        'mfs_returns': 'sum',
        'zip_code': 'nunique'
    }).rename(columns={'zip_code': 'num_zip_codes'})
    
    # Calculate district-level filing status percentages
    senate_agg['single_pct'] = senate_agg['single_returns'] / senate_agg['total_returns'] * 100
    senate_agg['joint_pct'] = senate_agg['joint_returns'] / senate_agg['total_returns'] * 100
    senate_agg['hoh_pct'] = senate_agg['hoh_returns'] / senate_agg['total_returns'] * 100
    senate_agg['mfs_pct'] = senate_agg['mfs_returns'] / senate_agg['total_returns'] * 100
    
    senate_agg['district_type'] = 'senate'
    senate_agg['district'] = senate_agg.index
    
    logger.info(f"Aggregated data for {len(house_agg)} House districts and {len(senate_agg)} Senate districts")
    
    return house_agg.reset_index(), senate_agg.reset_index()

def create_district_benchmarks(house_df, senate_df):
    """Create comprehensive district benchmarks for tax unit calibration."""
    logger.info("Creating district benchmarks...")
    
    benchmarks = {
        'house_districts': {},
        'senate_districts': {},
        'summary_stats': {}
    }
    
    # Process House districts
    if house_df is not None and not house_df.empty:
        for _, row in house_df.iterrows():
            district_id = int(row['house_district'])
            benchmarks['house_districts'][district_id] = {
                'total_returns': int(row['total_returns']),
                'filing_status_targets': {
                    'single_pct': float(row['single_pct']),
                    'joint_pct': float(row['joint_pct']),
                    'hoh_pct': float(row['hoh_pct']),
                    'mfs_pct': float(row['mfs_pct'])
                },
                'data_quality': {
                    'num_zip_codes': int(row['num_zip_codes']),
                    'confidence_score': min(100, row['num_zip_codes'] * 10)  # Simple confidence metric
                }
            }
    
    # Process Senate districts
    if senate_df is not None and not senate_df.empty:
        for _, row in senate_df.iterrows():
            district_id = int(row['senate_district'])
            benchmarks['senate_districts'][district_id] = {
                'total_returns': int(row['total_returns']),
                'filing_status_targets': {
                    'single_pct': float(row['single_pct']),
                    'joint_pct': float(row['joint_pct']),
                    'hoh_pct': float(row['hoh_pct']),
                    'mfs_pct': float(row['mfs_pct'])
                },
                'data_quality': {
                    'num_zip_codes': int(row['num_zip_codes']),
                    'confidence_score': min(100, row['num_zip_codes'] * 10)
                }
            }
    
    # Calculate summary statistics
    all_districts = pd.concat([house_df, senate_df], ignore_index=True) if house_df is not None and senate_df is not None else pd.DataFrame()
    
    if not all_districts.empty:
        benchmarks['summary_stats'] = {
            'total_districts_analyzed': len(all_districts),
            'avg_filing_status_pct': {
                'single': float(all_districts['single_pct'].mean()),
                'joint': float(all_districts['joint_pct'].mean()),
                'hoh': float(all_districts['hoh_pct'].mean()),
                'mfs': float(all_districts['mfs_pct'].mean())
            },
            'district_coverage': {
                'high_confidence': int((all_districts['num_zip_codes'] >= 5).sum()),
                'medium_confidence': int(((all_districts['num_zip_codes'] >= 2) & (all_districts['num_zip_codes'] < 5)).sum()),
                'low_confidence': int((all_districts['num_zip_codes'] < 2).sum())
            }
        }
    
    return benchmarks

def save_results(coverage_df, house_df, senate_df, benchmarks):
    """Save all analysis results."""
    logger.info("Saving results...")
    
    # Create output directory
    output_dir = Path('data/irs_soi')
    output_dir.mkdir(exist_ok=True)
    
    # Save coverage analysis
    if coverage_df is not None:
        coverage_path = output_dir / 'district_coverage_analysis.csv'
        coverage_df.to_csv(coverage_path, index=False)
        logger.info(f"Saved coverage analysis: {coverage_path}")
    
    # Save district aggregations
    if house_df is not None and not house_df.empty:
        house_path = output_dir / 'house_district_soi_aggregated.csv'
        house_df.to_csv(house_path, index=False)
        logger.info(f"Saved House district data: {house_path}")
    
    if senate_df is not None and not senate_df.empty:
        senate_path = output_dir / 'senate_district_soi_aggregated.csv'
        senate_df.to_csv(senate_path, index=False)
        logger.info(f"Saved Senate district data: {senate_path}")
    
    # Save benchmarks
    if benchmarks:
        benchmarks_path = output_dir / 'district_benchmarks.json'
        with open(benchmarks_path, 'w') as f:
            json.dump(benchmarks, f, indent=2)
        logger.info(f"Saved district benchmarks: {benchmarks_path}")

def main():
    """Main analysis function."""
    logger.info("Starting IRS SOI district aggregation analysis...")
    
    # Load data
    soi_df = load_irs_soi_data()
    if soi_df is None:
        return
    
    crosswalk_df = load_zip_district_crosswalk()
    if crosswalk_df is None:
        return
    
    # Analyze coverage
    coverage_df = analyze_district_coverage(soi_df, crosswalk_df)
    
    # Extract filing patterns
    patterns_df = extract_filing_status_patterns(soi_df)
    
    # Aggregate by district
    house_df, senate_df = aggregate_by_district(patterns_df, crosswalk_df)
    
    # Create benchmarks
    benchmarks = create_district_benchmarks(house_df, senate_df)
    
    # Save results
    save_results(coverage_df, house_df, senate_df, benchmarks)
    
    # Summary report
    logger.info("\n" + "="*60)
    logger.info("IRS SOI DISTRICT AGGREGATION COMPLETE")
    logger.info("="*60)
    
    if benchmarks and 'summary_stats' in benchmarks:
        stats = benchmarks['summary_stats']
        logger.info(f"Districts analyzed: {stats.get('total_districts_analyzed', 0)}")
        
        if 'avg_filing_status_pct' in stats:
            filing_stats = stats['avg_filing_status_pct']
            logger.info("Average filing status distribution across districts:")
            logger.info(f"  Single: {filing_stats.get('single', 0):.1f}%")
            logger.info(f"  Joint: {filing_stats.get('joint', 0):.1f}%")
            logger.info(f"  Head of Household: {filing_stats.get('hoh', 0):.1f}%")
            logger.info(f"  Married Filing Separate: {filing_stats.get('mfs', 0):.1f}%")
        
        if 'district_coverage' in stats:
            coverage_stats = stats['district_coverage']
            logger.info("Data quality distribution:")
            logger.info(f"  High confidence districts: {coverage_stats.get('high_confidence', 0)}")
            logger.info(f"  Medium confidence districts: {coverage_stats.get('medium_confidence', 0)}")
            logger.info(f"  Low confidence districts: {coverage_stats.get('low_confidence', 0)}")
    
    logger.info("\nNext steps:")
    logger.info("1. Review district coverage and data quality")
    logger.info("2. Implement district-specific calibration in tax unit construction")
    logger.info("3. Update district CTC analysis pipeline")
    
    return {
        'coverage': coverage_df,
        'house_districts': house_df,
        'senate_districts': senate_df,
        'benchmarks': benchmarks
    }

if __name__ == "__main__":
    results = main()
