#!/usr/bin/env python3
"""
Generate District-Level CTC Estimates for Hawaii

This script generates comprehensive Child Tax Credit estimates by Hawaii legislative
districts, counties, and other geographic areas using PUMS data and tax unit construction.

Usage:
    python scripts/generate_district_ctc_estimates.py [--output-dir OUTPUT_DIR] [--districts DISTRICT_TYPES]
"""

import pandas as pd
import numpy as np
import logging
import argparse
from pathlib import Path
import sys
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.analysis.district_ctc import DistrictCTCAnalyzer, create_ctc_district_report

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('district_ctc_estimates.log')
    ]
)
logger = logging.getLogger(__name__)

def load_tax_units(
    tax_units_path: str = 'src/data/processed/tax_units_weight_split_raked.parquet',
    person_path: str = 'data/processed/pums_person_processed.parquet',
    household_path: str = 'data/processed/pums_household_processed.parquet'
) -> pd.DataFrame:
    """
    Load tax units and merge with person and household data.
    
    Args:
        tax_units_path: Path to tax units parquet file
        person_path: Path to person-level PUMS data
        household_path: Path to household-level PUMS data
        
    Returns:
        DataFrame with tax units and merged person/household data
    """
    try:
        # Load weight-raked tax units (includes filing status and geographic data)
        logger.info(f"Loading weight-raked tax units from {tax_units_path}")
        raked_df = pd.read_parquet(tax_units_path)
        logger.info(f"Loaded {len(raked_df):,} weight-raked tax units")
        
        # Load original tax units for income and dependents data
        original_tax_units_path = 'src/data/processed/tax_units_rule_based.parquet'
        logger.info(f"Loading original tax units from {original_tax_units_path}")
        original_df = pd.read_parquet(original_tax_units_path)
        logger.info(f"Loaded {len(original_df):,} original tax units")
        
        # Merge original income/dependents data into raked tax units (preserving raked structure)
        logger.info("Merging original tax unit data into raked tax units...")
        # Group original data by SERIALNO and take first occurrence for each household
        original_grouped = original_df.groupby('SERIALNO').first().reset_index()
        
        df = raked_df.merge(
            original_grouped[['SERIALNO', 'income', 'dependents', 'num_dependents']], 
            on='SERIALNO', 
            how='left'
        )
        
        logger.info(f"After merge: {len(df):,} tax units retained (preserving raked structure)")
        
        # Check merge success
        missing_income = df['income'].isna().sum()
        if missing_income > 0:
            logger.warning(f"{missing_income:,} tax units missing income data")
            # Fill missing income with 0
            df['income'] = df['income'].fillna(0)
        
        # Load person data for dependent processing
        logger.info(f"Loading person data from {person_path}")
        person_df = pd.read_parquet(person_path)
        logger.info(f"Loaded {len(person_df):,} person records")
        
        # Create person_id for merging (SERIALNO_SPORDER)
        person_df['person_id'] = person_df['SERIALNO'] + '_' + person_df['SPORDER'].astype(str)
        
        # Convert dependents from array of person_ids to list of dicts with person data
        logger.info("Processing dependents...")
        
        def get_dependent_info(dependent_ids):
            """Convert array of person_ids to list of dependent dicts."""
            if not isinstance(dependent_ids, (list, np.ndarray)) or len(dependent_ids) == 0:
                return []
                
            # Get person records for all dependents
            dependents = person_df[person_df['person_id'].isin(dependent_ids)]
            
            # Convert to list of dicts with required fields
            return [{
                'age': row.get('AGEP', 0),
                'relationship': row.get('RELSHIPP', ''),
                'citizenship': row.get('CIT', '1')  # Default to US citizen
            } for _, row in dependents.iterrows()]
        
        # Apply to all tax units
        if 'dependents' in df.columns:
            df['dependents'] = df['dependents'].apply(get_dependent_info)
        else:
            logger.warning("No dependents column found - creating empty dependents")
            df['dependents'] = [[] for _ in range(len(df))]
        
        # Rename weight column to match expected format
        if 'scenario_weight' in df.columns:
            df['weight'] = df['scenario_weight']
            logger.info("Renamed 'scenario_weight' to 'weight' for compatibility")
        
        logger.info(f"Successfully processed {len(df):,} tax units with income, dependents and geographic data")
        return df
        
    except Exception as e:
        logger.error(f"Error loading tax units: {e}")
        raise

def validate_tax_units(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean tax units data for district analysis."""
    logger.info("Validating tax units data...")
    
    original_count = len(df)
    
    # Check for required columns - updated for raked tax units
    required_cols = ['weight', 'filing_status', 'income']
    
    # Geographic columns should be present in raked tax units
    geo_cols = ['PUMA', 'county', 'house_district', 'senate_district']
    available_geo_cols = [col for col in geo_cols if col in df.columns]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Log available geographic data
    logger.info(f"Geographic data available: {len([col for col in ['county'] if col in df.columns])} counties")
    if 'house_district' in df.columns:
        logger.info(f"House districts available: {df['house_district'].nunique()} districts")
    if 'senate_district' in df.columns:
        logger.info(f"Senate districts available: {df['senate_district'].nunique()} districts")
    
    # Remove units with zero or negative weights
    df = df[df['weight'] > 0]
    
    # Handle PUMA formatting if PUMA column exists
    if 'PUMA' in df.columns:
        # Remove units with missing PUMA (can't assign to districts)
        df = df.dropna(subset=['PUMA'])
        # Ensure PUMA is string format (Hawaii uses 4-digit format: 0100, 0200, etc.)
        df['PUMA'] = df['PUMA'].astype(str).str.zfill(4)
    
    # Validate geographic data
    if 'county' in df.columns:
        logger.info(f"Geographic data available: {df['county'].nunique()} counties")
    if 'house_district' in df.columns:
        logger.info(f"House districts available: {df['house_district'].nunique()} districts")
    if 'senate_district' in df.columns:
        logger.info(f"Senate districts available: {df['senate_district'].nunique()} districts")
    
    logger.info(f"Validation complete: {len(df):,} units retained ({original_count - len(df):,} removed)")
    return df

def generate_summary_report(results: dict, output_dir: Path) -> None:
    """Generate summary report with key findings."""
    logger.info("Generating summary report...")
    
    # Create summary statistics
    statewide = results['statewide_summary']
    house_districts = results['house_districts']
    counties = results['counties']
    
    summary = {
        'statewide_totals': {
            'total_ctc_amount': f"${statewide['total_ctc_amount']:,.0f}",
            'total_tax_units': f"{statewide['total_tax_units']:,}",
            'ctc_recipient_rate': f"{statewide['ctc_recipient_rate']:.1%}",
            'avg_ctc_per_unit': f"${statewide['avg_ctc_per_unit']:,.0f}",
            'avg_ctc_per_recipient': f"${statewide['avg_ctc_per_recipient']:,.0f}",
            'total_qualifying_children': f"{statewide['total_qualifying_children']:,.0f}"
        },
        'district_analysis': {
            'num_house_districts': int(len(house_districts)),
            'highest_ctc_house_district': str(house_districts.loc[house_districts['total_ctc_amount'].idxmax(), 'district']),
            'highest_ctc_amount': f"${float(house_districts['total_ctc_amount'].max()):,.0f}",
            'lowest_ctc_house_district': str(house_districts.loc[house_districts['total_ctc_amount'].idxmin(), 'district']),
            'lowest_ctc_amount': f"${float(house_districts['total_ctc_amount'].min()):,.0f}"
        },
        'county_analysis': {
            'num_counties': int(len(counties)),
            'highest_ctc_county': str(counties.loc[counties['total_ctc_amount'].idxmax(), 'district']),
            'county_ctc_amounts': {
                str(row['district']): f"${float(row['total_ctc_amount']):,.0f}" 
                for _, row in counties.iterrows()
            }
        }
    }
    
    # Save summary as JSON
    summary_file = output_dir / 'hawaii_ctc_summary_report.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create human-readable summary
    readme_content = f"""# Hawaii Child Tax Credit District Analysis Summary

## Statewide Totals
- **Total CTC Amount**: {summary['statewide_totals']['total_ctc_amount']}
- **Total Tax Units**: {summary['statewide_totals']['total_tax_units']}
- **CTC Recipient Rate**: {summary['statewide_totals']['ctc_recipient_rate']}
- **Average CTC per Tax Unit**: {summary['statewide_totals']['avg_ctc_per_unit']}
- **Average CTC per Recipient**: {summary['statewide_totals']['avg_ctc_per_recipient']}
- **Total Qualifying Children**: {summary['statewide_totals']['total_qualifying_children']}

## House District Analysis
- **Number of Districts**: {summary['district_analysis']['num_house_districts']}
- **Highest CTC District**: {summary['district_analysis']['highest_ctc_house_district']} ({summary['district_analysis']['highest_ctc_amount']})
- **Lowest CTC District**: {summary['district_analysis']['lowest_ctc_house_district']} ({summary['district_analysis']['lowest_ctc_amount']})

## County Analysis
- **Number of Counties**: {summary['county_analysis']['num_counties']}
- **Highest CTC County**: {summary['county_analysis']['highest_ctc_county']}

### CTC by County:
"""
    
    for county, amount in summary['county_analysis']['county_ctc_amounts'].items():
        readme_content += f"- **{county}**: {amount}\n"
    
    readme_content += f"""
## Files Generated
- `hawaii_house_districts_ctc_summary.csv`: House district CTC estimates
- `hawaii_senate_districts_ctc_summary.csv`: Senate district CTC estimates  
- `hawaii_county_districts_ctc_summary.csv`: County CTC estimates
- `house/`, `senate/`, `county/`: Individual district profile files
- `hawaii_ctc_summary_report.json`: Machine-readable summary statistics

## Methodology
- Based on 2023 PUMS data for Hawaii
- Tax units constructed using rule-based approach with SOI calibration
- CTC calculated using 2023 tax law parameters
- Population estimates using PUMS household weights
- Geographic assignment via PUMA-to-district crosswalk

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    readme_file = output_dir / 'README.md'
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    
    logger.info(f"Summary report saved to {summary_file} and {readme_file}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Generate Hawaii district-level CTC estimates')
    parser.add_argument('--output-dir', '-o', 
                       default='output/district_ctc_analysis',
                       help='Output directory for results')
    parser.add_argument('--districts', '-d',
                       nargs='+',
                       default=['house', 'senate', 'county'],
                       choices=['house', 'senate', 'county'],
                       help='District types to analyze')
    parser.add_argument('--tax-units-file', '-t',
                       default='src/data/processed/tax_units_weight_split_raked.parquet',
                       help='Path to tax units file')
    
    args = parser.parse_args()
    
    logger.info("=== Hawaii District-Level CTC Analysis ===")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"District types: {args.districts}")
    
    try:
        # Load and validate data
        tax_units_df = load_tax_units(args.tax_units_file)
        tax_units_df = validate_tax_units(tax_units_df)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate comprehensive district analysis
        logger.info("Generating comprehensive district CTC analysis...")
        results = create_ctc_district_report(tax_units_df, str(output_dir))
        
        # Generate summary report
        generate_summary_report(results, output_dir)
        
        # Log key results
        statewide = results['statewide_summary']
        logger.info("=== KEY RESULTS ===")
        logger.info(f"Total CTC Amount: ${statewide['total_ctc_amount']:,.0f}")
        logger.info(f"CTC Recipients: {statewide['ctc_recipient_units']:,} ({statewide['ctc_recipient_rate']:.1%})")
        logger.info(f"Average CTC per Recipient: ${statewide['avg_ctc_per_recipient']:,.0f}")
        logger.info(f"Qualifying Children: {statewide['total_qualifying_children']:,.0f}")
        
        # District-specific results
        for district_type in args.districts:
            if f'{district_type}_districts' in results:
                district_data = results[f'{district_type}_districts']
                logger.info(f"{district_type.title()} Districts: {len(district_data)} analyzed")
        
        logger.info(f"Analysis complete! Results saved to: {output_dir}")
        logger.info("Check README.md for detailed summary and file descriptions.")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()
