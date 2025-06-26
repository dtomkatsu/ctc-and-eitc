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
from typing import Dict, Tuple, Optional

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
    
    def __init__(self, data_dir: str = 'data/processed', tax_units_dir: str = 'src/data/processed'):
        """Initialize with paths to data files."""
        self.data_dir = Path(data_dir)
        self.tax_units_dir = Path(tax_units_dir)
        self.tax_units = None
        self.person_df = None
        self.hh_df = None
        self.soi_data = None
        
    def load_data(self) -> bool:
        """Load the required data files."""
        try:
            # Load tax units from the tax_units_dir
            tax_units_path = self.tax_units_dir / 'tax_units_rule_based.parquet'
            logger.info(f"Loading tax units from {tax_units_path}")
            self.tax_units = pd.read_parquet(tax_units_path)
            
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
        
        # Create a unique person ID in the person_df (SERIALNO + _ + SPORDER)
        self.person_df['person_id'] = self.person_df['SERIALNO'].astype(str) + '_' + self.person_df['SPORDER'].astype(str)
        
        # Create a mapping from person_id to weight
        weight_map = self.person_df.set_index('person_id')['PWGTP'].to_dict()
        
        # Add weights to tax units
        self.tax_units['weight'] = self.tax_units['adult1_id'].map(weight_map)
        
        # For joint filers, average the weights of both adults
        joint_mask = self.tax_units['filing_status'] == 'joint'
        if joint_mask.any():
            joint_weights = self.tax_units.loc[joint_mask, 'adult2_id'].map(weight_map)
            self.tax_units.loc[joint_mask, 'weight'] = (
                self.tax_units.loc[joint_mask, 'weight'] + joint_weights
            ) / 2
        
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
    validator = TaxUnitValidator()
    
    # Load the data
    if not validator.load_data():
        return 1
    
    # Apply weights and generate report
    try:
        validator.apply_weights()
        validator.generate_validation_report()
        return 0
    except Exception as e:
        logger.error(f"Error during validation: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
