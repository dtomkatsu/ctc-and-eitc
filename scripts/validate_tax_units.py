#!/usr/bin/env python3
"""
Validate tax units against known patterns and SOI data.

This script:
1. Loads the constructed tax units and PUMS person data
2. Applies PUMS weights to scale to population totals
3. Compares the weighted distribution to SOI data
4. Generates validation reports
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import os
from typing import Dict, Tuple, Optional

# Add the project root directory to the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('validate_tax_units.log')
    ]
)
logger = logging.getLogger(__name__)

class TaxUnitValidator:
    """Class for validating tax units against known patterns and SOI data."""
    
    def __init__(self, data_dir: str = 'data/processed', tax_units_dir: str = 'src/data/processed', n_jobs: int = -1):
        """Initialize with paths to data files and processing options.
        
        Args:
            data_dir: Directory containing processed data files
            tax_units_dir: Directory containing tax unit files
            n_jobs: Number of parallel jobs to use (-1 for all available CPUs)
        """
        self.data_dir = Path(data_dir)
        self.tax_units_dir = Path(tax_units_dir)
        self.n_jobs = n_jobs
        self.tax_units = None
        self.person_df = None
        self.hh_df = None
        self.soi_data = None
        self.construction_time = None
        
    def load_data(self) -> bool:
        """Load the required data files."""
        try:
            # Load tax units from the tax_units_dir
            tax_units_path = self.tax_units_dir / 'tax_units_rule_based.parquet'
            
            # If tax units file doesn't exist, generate it
            if not tax_units_path.exists():
                logger.info(f"Tax units file not found at {tax_units_path}. Constructing tax units...")
                if not self.construct_tax_units():
                    return False
            else:
                logger.info(f"Loading tax units from {tax_units_path}")
                self.tax_units = pd.read_parquet(tax_units_path)
            
            if self.tax_units is None or self.tax_units.empty:
                logger.error("Failed to load or construct tax units")
                return False
                
            logger.info(f"Loaded {len(self.tax_units)} tax units")
            
            # Load PUMS person data to get weights
            person_path = self.data_dir / 'pums_person_processed.parquet'
            logger.info(f"Loading person data from {person_path}")
            self.person_df = pd.read_parquet(person_path)
            
            # Load household data
            hh_path = self.data_dir / 'pums_household_processed.parquet'
            logger.info(f"Loading household data from {hh_path}")
            self.hh_df = pd.read_parquet(hh_path)
            
            # Load SOI data
            self._load_soi_data()
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}", exc_info=True)
            return False
    
    def construct_tax_units(self) -> bool:
        """Construct tax units from PUMS data using multiprocessing."""
        from src.data.pums_loader import PUMSDataLoader
        from src.tax.units.constructor import TaxUnitConstructor
        import time
        
        logger.info("Starting tax unit construction...")
        start_time = time.time()
        
        try:
            # Initialize data loader
            data_loader = PUMSDataLoader()
            
            # Load person and household data
            logger.info("Loading PUMS data...")
            person_df, hh_df = data_loader.load_data()
            
            if person_df is None or person_df.empty or hh_df is None or hh_df.empty:
                logger.error("Failed to load PUMS data")
                return False
            
            # Create tax units with multiprocessing
            logger.info(f"Constructing tax units using {self.n_jobs} processes...")
            constructor = TaxUnitConstructor(person_df=person_df, hh_df=hh_df)
            self.tax_units = constructor.create_rule_based_units(n_jobs=self.n_jobs)
            
            # Save the constructed tax units
            output_path = self.tax_units_dir / 'tax_units_rule_based.parquet'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.tax_units.to_parquet(output_path, index=False)
            logger.info(f"Saved {len(self.tax_units)} tax units to {output_path}")
            
            # Record construction time
            self.construction_time = time.time() - start_time
            logger.info(f"Tax unit construction completed in {self.construction_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during tax unit construction: {e}", exc_info=True)
            return False
            
            # Load PUMS person data to get weights
            person_path = self.data_dir / 'pums_person_processed.parquet'
            logger.info(f"Loading person data from {person_path}")
            self.person_df = pd.read_parquet(person_path)
            
            # Load household data
            hh_path = self.data_dir / 'pums_household_processed.parquet'
            logger.info(f"Loading household data from {hh_path}")
            self.hh_df = pd.read_parquet(hh_path)
            
            # Load SOI data
            self._load_soi_data()
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}", exc_info=True)
            return False
    
    def _load_soi_data(self) -> None:
        """
        Load and prepare SOI data for comparison using 2022 Hawaii DOTAX filing status distribution.
        
        Source: 2022 Hawaii DOTAX data provided by user
        Distribution:
        - Married Filing Jointly: 36.0%
        - Single: 51.0%
        - Married Filing Separately: 3.1%
        - Head of Household: 9.6%
        """
        soi_path = Path('data/23dbhawaii.xlsx')
        
        try:
            # Read the Excel file to get total individual returns
            df = pd.read_excel(soi_path, sheet_name=0, header=None)
            
            # Extract the total returns from row 3 (0-indexed as 2)
            total_returns = int(df.iloc[2, 1])  # Column B, Row 3 (0-indexed 2,1)
            individual_returns = int(df.iloc[3, 1])  # Individual income tax row
            
            # Apply 2022 Hawaii DOTAX filing status distribution
            self.soi_data = {
                'total_returns': total_returns,
                'individual_returns': individual_returns,
                'single_returns': int(individual_returns * 0.510),  # 51.0% Single
                'mfj_returns': int(individual_returns * 0.360),    # 36.0% Married Filing Jointly
                'mfs_returns': int(individual_returns * 0.031),    # 3.1% Married Filing Separately
                'hoh_returns': int(individual_returns * 0.096),    # 9.6% Head of Household
                # For backward compatibility, also include combined joint returns
                'joint_returns': int(individual_returns * 0.360)   # Just MFJ (36.0%)
            }
            
            logger.info(f"Loaded SOI data with {total_returns:,} total returns and {individual_returns:,} individual returns")
            logger.info("Using 2022 Hawaii DOTAX filing status distribution for comparison")
            
        except Exception as e:
            logger.error(f"Error loading SOI data: {e}")
            logger.error("Falling back to estimated values")
            
            # Fallback to hardcoded values if file can't be read
            self.soi_data = {
                'total_returns': 1_092_275,
                'individual_returns': 855_541,
                'single_returns': int(855_541 * 0.510),  # 51.0%
                'joint_returns': int(855_541 * 0.391),   # 36.0% + 3.1%
                'hoh_returns': int(855_541 * 0.096),     # 9.6%
                'mfs_returns': int(855_541 * 0.031),     # 3.1%
                'mfj_returns': int(855_541 * 0.360),     # 36.0%
            }
    
    def apply_weights(self) -> pd.DataFrame:
        """Apply PUMS weights to tax units."""
        if self.tax_units is None or self.person_df is None or self.hh_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        logger.info("Applying PUMS weights to tax units...")
        
        # Create a mapping from (SERIALNO, SPORDER) to weight
        weight_map = self.person_df.set_index(['SERIALNO', 'SPORDER'])['PWGTP'].to_dict()
        
        # Get the person data for filers
        filer_data = self.person_df[self.person_df.index.isin(self.tax_units['filer_id'])]
        
        # Create a mapping from filer_id to weight
        filer_weight_map = dict(zip(filer_data.index, filer_data['PWGTP']))
        
        # Add weights to tax units using filer_id
        self.tax_units['weight'] = self.tax_units['filer_id'].map(filer_weight_map)
        
        # For joint filers, the weight is already set from the primary filer
        # No need to average since we're using the primary filer's weight
        
        # Handle any missing weights (shouldn't happen with proper data)
        missing_weights = self.tax_units['weight'].isna().sum()
        if missing_weights > 0:
            logger.warning(f"{missing_weights} tax units have missing weights. Filling with 1.")
            self.tax_units['weight'] = self.tax_units['weight'].fillna(1)
        
        return self.tax_units
    
    def get_weighted_counts(self) -> Dict[str, float]:
        """Get weighted counts by filing status."""
        if 'weight' not in self.tax_units.columns:
            self.apply_weights()
        
        # Group by filing status and sum weights
        weighted_counts = self.tax_units.groupby('filing_status')['weight'].sum().to_dict()
        
        # Add total weighted count
        weighted_counts['total'] = sum(weighted_counts.values())
        
        return weighted_counts
    
    def compare_to_soi(self) -> Dict[str, Dict[str, float]]:
        """Compare weighted PUMS counts to SOI data using 2022 Hawaii DOTAX distribution."""
        weighted_counts = self.get_weighted_counts()
        
        comparison = {}
        
        # Calculate scaling factor (SOI total / PUMS weighted total)
        scaling_factor = self.soi_data['individual_returns'] / weighted_counts['total']
        
        # Compare filing status distributions
        # Map our PUMS categories to SOI categories
        # Note: 'joint' in our PUMS data corresponds to 'Married Filing Jointly' (MFJ) only
        status_map = {
            'single': 'single_returns',
            'joint': 'mfj_returns',      # Map to MFJ only (not combined with MFS)
            'head_of_household': 'hoh_returns'
        }
        
        for pums_status, soi_key in status_map.items():
            pums_count = weighted_counts.get(pums_status, 0)
            soi_count = self.soi_data.get(soi_key, 0)
            scaled_pums = pums_count * scaling_factor
            
            comparison[pums_status] = {
                'pums_weighted': pums_count,
                'pums_scaled': scaled_pums,
                'soi_actual': soi_count,
                'difference': scaled_pums - soi_count,
                'pct_difference': ((scaled_pums - soi_count) / soi_count * 100) if soi_count > 0 else float('nan')
            }
            
        # Add detailed MFS comparison (MFJ is already included as 'joint')
        mfs_soi_count = self.soi_data.get('mfs_returns', 0)
        comparison['mfs_returns'] = {
            'pums_weighted': float('nan'),
            'pums_scaled': float('nan'),
            'soi_actual': mfs_soi_count,
            'difference': float('nan'),
            'pct_difference': float('nan')
        }
        
        # Add totals
        total_pums = sum(v['pums_weighted'] for v in comparison.values())
        total_scaled = total_pums * scaling_factor
        total_soi = self.soi_data['individual_returns']
        
        comparison['total'] = {
            'pums_weighted': total_pums,
            'pums_scaled': total_scaled,
            'soi_actual': total_soi,
            'difference': total_scaled - total_soi,
            'pct_difference': ((total_scaled - total_soi) / total_soi * 100) if total_soi > 0 else float('nan')
        }
        
        return comparison
    
    def generate_validation_report(self, output_dir: str = 'analysis') -> None:
        """Generate a validation report comparing PUMS to SOI data."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get weighted counts and comparison
        weighted_counts = self.get_weighted_counts()
        comparison = self.compare_to_soi()
        
        # Create a DataFrame for the report with proper ordering
        status_order = [
            'single', 
            'joint',           # MFJ only
            'mfs_returns',     # MFS (not in PUMS data)
            'head_of_household'
        ]
        
        report_data = []
        for status in status_order:
            if status not in comparison:
                continue
                
            data = comparison[status]
            display_name = {
                'single': 'Single',
                'joint': 'Married Filing Jointly',
                'mfs_returns': 'Married Filing Separately',
                'head_of_household': 'Head of Household'
            }.get(status, status.replace('_', ' ').title())
            
            report_data.append({
                'filing_status': display_name,
                'pums_weighted': data['pums_weighted'],
                'pums_scaled': data['pums_scaled'],
                'soi_actual': data['soi_actual'],
                'difference': data['difference'],
                'pct_difference': data['pct_difference']
            })
        
        # Add totals
        total = comparison['total']
        report_data.append({
            'filing_status': 'TOTAL',
            'pums_weighted': total['pums_weighted'],
            'pums_scaled': total['pums_scaled'],
            'soi_actual': total['soi_actual'],
            'difference': total['difference'],
            'pct_difference': total['pct_difference']
        })
        
        report_df = pd.DataFrame(report_data)
        
        # Save to CSV
        csv_path = output_dir / 'tax_unit_validation.csv'
        report_df.to_csv(csv_path, index=False, float_format='%.2f')
        logger.info(f"Saved validation report to {csv_path}")
        
        # Print summary
        print("\n" + "="*100)
        print("TAX UNIT VALIDATION REPORT (2023 PUMS vs 2022 HAWAII DOTAX DISTRIBUTION)")
        print("="*100)
        
        print("\nFiling Status Comparison (Weighted):")
        print("-"*100)
        
        # Format the DataFrame for better readability
        def format_number(x):
            if pd.isna(x):
                return 'N/A'
            return f"{x:,.0f}"
            
        def format_pct(x):
            if pd.isna(x):
                return 'N/A'
            return f"{x:+.1f}%"
        
        formatted_df = report_df.copy()
        for col in ['pums_weighted', 'pums_scaled', 'soi_actual', 'difference']:
            formatted_df[col] = formatted_df[col].apply(format_number)
        formatted_df['pct_difference'] = formatted_df['pct_difference'].apply(format_pct)
        
        print(formatted_df.to_string(index=False, justify='right'))
        
        # Print scaling factor
        scaling_factor = comparison['total']['pums_scaled'] / comparison['total']['pums_weighted']
        print(f"\nScaling factor (SOI/PUMS): {scaling_factor:.2f}")
        
        # Print any large discrepancies
        large_diffs = [
            (status, data)
            for status, data in comparison.items()
            if status != 'total' and abs(data['pct_difference']) > 10
        ]
        
        if large_diffs:
            print("\nLarge discrepancies found:")
            for status, data in large_diffs:
                print(f"  - {status}: {data['pct_difference']:+.1f}%")
        
        print("\n" + "="*80)

def main():
    """Main function to run the validation."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Validate tax units with optional multiprocessing')
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='Number of parallel jobs to use (-1 for all available CPUs)')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Directory containing processed data files')
    parser.add_argument('--tax-units-dir', type=str, default='src/data/processed',
                       help='Directory containing tax unit files')
    
    args = parser.parse_args()
    
    logger.info(f"Starting validation with {args.n_jobs} parallel jobs")
    validator = TaxUnitValidator(
        data_dir=args.data_dir,
        tax_units_dir=args.tax_units_dir,
        n_jobs=args.n_jobs
    )
    
    # Load the data
    if not validator.load_data():
        return 1
    
    # Apply weights and generate report
    try:
        validator.apply_weights()
        validator.generate_validation_report()
        
        # Log performance information
        if validator.construction_time is not None:
            logger.info(f"Tax unit construction time: {validator.construction_time:.2f} seconds")
            logger.info(f"Tax units per second: {len(validator.tax_units) / validator.construction_time:.2f}")
            
        return 0
    except Exception as e:
        logger.error(f"Error during validation: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
