#!/usr/bin/env python3
"""
Compare PUMS-based tax unit counts with SOI (IRS) return counts.
"""
import pandas as pd
import sys
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_pums_tax_units():
    """Load the PUMS-based tax units."""
    tax_units_file = Path("src/data/processed/tax_units_rule_based.parquet")
    if not tax_units_file.exists():
        logger.error("Tax units file not found. Please run tax unit construction first.")
        return None
    
    logger.info("Loading PUMS-based tax units...")
    return pd.read_parquet(tax_units_file)

def get_soi_returns():
    """Get the SOI return counts."""
    # From the SOI data analysis
    return {
        'total_returns': 1092275,
        'individual_returns': 855541,
        'corporate_returns': 11863,
        'estate_returns': 306,
        'estate_trust_returns': 18985,
        'gift_tax_returns': 2506,
        'tax_exempt_returns': 7538,
        'excise_tax_returns': 1499
    }

def analyze_tax_units(tax_units):
    """Analyze the PUMS-based tax units."""
    analysis = {}
    
    # Total tax units
    analysis['total_units'] = len(tax_units)
    
    # Filing status breakdown
    if 'filing_status' in tax_units.columns:
        # Convert to string to handle both numeric and string statuses
        tax_units['filing_status'] = tax_units['filing_status'].astype(str)
        
        # Import the same status constants used in the tax unit construction
        from src.tax.units import FILING_STATUS
        
        # Map status values to ensure consistency with the tax unit construction
        status_map = {
            '1': FILING_STATUS['SINGLE'],
            '2': FILING_STATUS['JOINT'],
            '3': FILING_STATUS['SEPARATE'],
            '4': FILING_STATUS['HEAD_HOUSEHOLD'],
            '5': FILING_STATUS['WIDOW']
        }
        
        # Replace numeric status codes with string values
        tax_units['filing_status'] = tax_units['filing_status'].replace(status_map)
        
        # Also ensure any direct string values match our constants
        status_mapping = {
            'single': FILING_STATUS['SINGLE'],
            'joint': FILING_STATUS['JOINT'],
            'separate': FILING_STATUS['SEPARATE'],
            'head_of_household': FILING_STATUS['HEAD_HOUSEHOLD'],
            'head of household': FILING_STATUS['HEAD_HOUSEHOLD'],  # Handle variations
            'widow': FILING_STATUS['WIDOW'],
            'widower': FILING_STATUS['WIDOW']  # Handle variations
        }
        tax_units['filing_status'] = tax_units['filing_status'].replace(status_mapping)
        
        # Get status counts
        status_counts = tax_units['filing_status'].value_counts().to_dict()
        analysis.update(status_counts)
        
        # Log the distribution of filing statuses
        logger.info("Filing status distribution:")
        for status, count in status_counts.items():
            logger.info(f"  {status}: {count} ({(count/len(tax_units))*100:.1f}%)")
    
    # Single vs joint filers
    analysis['single_filers'] = len(tax_units[tax_units['filing_status'] == 'single'])
    analysis['joint_filers'] = len(tax_units[tax_units['filing_status'] == 'joint'])
    analysis['head_of_household'] = len(tax_units[tax_units['filing_status'] == 'head_of_household'])
    
    # Log any unexpected status values
    valid_statuses = {'single', 'joint', 'separate', 'head_of_household', 'widow'}
    invalid_statuses = set(tax_units['filing_status'].unique()) - valid_statuses
    if invalid_statuses:
        logger.warning(f"Found {len(invalid_statuses)} unexpected filing status values: {invalid_statuses}")
    
    # Dependents
    if 'num_dependents' in tax_units.columns:
        analysis['total_dependents'] = tax_units['num_dependents'].sum()
    
    return analysis

def compare_to_soi(pums_analysis, soi_data):
    """Compare PUMS analysis to SOI data."""
    comparison = {}
    
    # Calculate scaling factor
    pums_total = pums_analysis['total_units']
    soi_total = soi_data['individual_returns']  # Most comparable to our PUMS data
    scaling_factor = soi_total / pums_total
    
    comparison['scaling_factor'] = scaling_factor
    
    # Compare total returns
    comparison['pums_total'] = pums_total
    comparison['soi_total'] = soi_total
    comparison['pums_scaled'] = pums_total * scaling_factor
    
    # Compare filing status (if available)
    if 'single_filers' in pums_analysis and 'joint_filers' in pums_analysis:
        # Calculate expected numbers based on PUMS distribution
        single_pct = pums_analysis['single_filers'] / pums_total
        joint_pct = pums_analysis['joint_filers'] / pums_total
        hoh_pct = pums_analysis.get('head_of_household', 0) / pums_total if 'head_of_household' in pums_analysis else 0
        
        comparison['filing_status'] = {
            'single': {
                'pums_count': pums_analysis['single_filers'],
                'pums_pct': single_pct * 100,
                'expected_soi': single_pct * soi_total
            },
            'joint': {
                'pums_count': pums_analysis['joint_filers'],
                'pums_pct': joint_pct * 100,
                'expected_soi': joint_pct * soi_total / 2  # Divide by 2 since joint returns count two people
            },
            'head_of_household': {
                'pums_count': pums_analysis.get('head_of_household', 0),
                'pums_pct': hoh_pct * 100,
                'expected_soi': hoh_pct * soi_total
            }
        }
    
    return comparison

def print_comparison(comparison):
    """Print the comparison results."""
    print("\n" + "="*80)
    print("TAX UNIT COMPARISON: PUMS vs SOI")
    print("="*80)
    
    # Basic comparison
    print(f"\n{'Metric':<30} {'PUMS':>15} {'SOI':>15} {'Ratio (SOI/PUMS)':>20}")
    print("-"*70)
    print(f"{'Total Returns:':<30} {comparison['pums_total']:15,.0f} {comparison['soi_total']:15,.0f} {comparison['scaling_factor']:20.2f}")
    
    # Print PUMS distribution
    print("\nPUMS Filing Status Distribution:")
    for status in ['single', 'joint', 'head_of_household']:
        count = comparison.get(status, 0)
        pct = (count / comparison['pums_total']) * 100 if comparison['pums_total'] > 0 else 0
        print(f"  {status.replace('_', ' ').title()}: {count:8,.0f} ({pct:5.1f}%)")
    
    # Print expected SOI distribution
    print("\nExpected SOI Distribution (based on PUMS):")
    for status, data in comparison.get('filing_status', {}).items():
        print(f"  {status.replace('_', ' ').title()}: {data['expected_soi']:8,.0f}")
    
    # Print actual SOI distribution (if available)
    if 'soi_distribution' in comparison:
        print("\nActual SOI Distribution:")
        for status, count in comparison['soi_distribution'].items():
            print(f"  {status.replace('_', ' ').title()}: {count:8,.0f}")
    
    # Filing status comparison
    if 'filing_status' in comparison:
        print("\nFiling Status Comparison:")
        print("-"*70)
        print(f"{'Status':<20} {'PUMS Count':>12} {'PUMS %':>10} {'Expected SOI':>15}")
        print("-"*70)
        
        for status, data in comparison['filing_status'].items():
            status_name = status.replace('_', ' ').title()
            print(f"{status_name:<20} {data['pums_count']:12,.0f} {data['pums_pct']:9.1f}% {data['expected_soi']:14,.0f}")
    
    print("\n" + "="*80)

def main():
    try:
        # Load the data
        tax_units = load_pums_tax_units()
        if tax_units is None:
            return 1
            
        soi_data = get_soi_returns()
        
        # Analyze the tax units
        pums_analysis = analyze_tax_units(tax_units)
        
        # Compare to SOI
        comparison = compare_to_soi(pums_analysis, soi_data)
        
        # Print the results
        print_comparison(comparison)
        
        # Save detailed comparison to CSV
        output_file = Path("analysis/tax_unit_comparison.csv")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create comparison DataFrame
        comparison_data = []
        for status, data in comparison['filing_status'].items():
            comparison_data.append({
                'filing_status': status,
                'pums_count': data['pums_count'],
                'pums_pct': data['pums_pct'],
                'expected_soi': data['expected_soi']
            })
        
        pd.DataFrame(comparison_data).to_csv(output_file, index=False)
        logger.info(f"Detailed comparison saved to {output_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error comparing to SOI data: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
