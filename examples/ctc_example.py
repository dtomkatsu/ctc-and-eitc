"""
Example: Child Tax Credit (CTC) Calculation

This example demonstrates how to calculate CTC for tax units constructed from PUMS data.
"""

import sys
import os
import pandas as pd

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from tax.credits.ctc import calculate_ctc, calculate_ctc_for_tax_units, get_ctc_summary_stats


def create_sample_tax_units():
    """Create sample tax units for demonstration."""
    return pd.DataFrame([
        {
            'filer_id': 'HH1_1',
            'filing_status': 'single',
            'income': 45000,
            'dependents': [
                {'age': 8, 'relationship': '22', 'citizenship': '1'},   # Natural child, age 8
                {'age': 14, 'relationship': '22', 'citizenship': '1'}   # Natural child, age 14
            ],
            'num_dependents': 2,
            'hh_id': 'HH1'
        },
        {
            'filer_id': 'HH2_1',
            'filing_status': 'married_filing_jointly',
            'income': 85000,
            'dependents': [
                {'age': 5, 'relationship': '22', 'citizenship': '1'},   # Natural child, age 5
                {'age': 16, 'relationship': '23', 'citizenship': '1'},  # Adopted child, age 16
                {'age': 18, 'relationship': '22', 'citizenship': '1'}   # Natural child, age 18 (too old)
            ],
            'num_dependents': 3,
            'hh_id': 'HH2'
        },
        {
            'filer_id': 'HH3_1',
            'filing_status': 'head_of_household',
            'income': 220000,  # High income - will trigger phaseout
            'dependents': [
                {'age': 12, 'relationship': '22', 'citizenship': '1'}   # Natural child, age 12
            ],
            'num_dependents': 1,
            'hh_id': 'HH3'
        },
        {
            'filer_id': 'HH4_1',
            'filing_status': 'single',
            'income': 35000,
            'dependents': [],  # No children
            'num_dependents': 0,
            'hh_id': 'HH4'
        }
    ])


def main():
    """Run CTC calculation example."""
    print("Child Tax Credit (CTC) Calculation Example")
    print("=" * 50)
    
    # Create sample tax units
    tax_units = create_sample_tax_units()
    print(f"Created {len(tax_units)} sample tax units\n")
    
    # Calculate CTC for each tax unit
    print("Individual Tax Unit CTC Calculations:")
    print("-" * 40)
    
    for idx, tax_unit in tax_units.iterrows():
        tax_unit_dict = tax_unit.to_dict()
        ctc_result = calculate_ctc(tax_unit_dict)
        
        print(f"Tax Unit: {tax_unit['filer_id']}")
        print(f"  Filing Status: {tax_unit['filing_status']}")
        print(f"  Income: ${tax_unit['income']:,}")
        print(f"  Total Dependents: {tax_unit['num_dependents']}")
        print(f"  Qualifying Children: {ctc_result['qualifying_children']}")
        print(f"  Total CTC: ${ctc_result['ctc_total']:,.2f}")
        print(f"  Refundable CTC: ${ctc_result['ctc_refundable']:,.2f}")
        print(f"  Non-refundable CTC: ${ctc_result['ctc_nonrefundable']:,.2f}")
        print()
    
    # Calculate CTC for all tax units using DataFrame method
    tax_units_with_ctc = calculate_ctc_for_tax_units(tax_units)
    
    # Get summary statistics
    stats = get_ctc_summary_stats(tax_units_with_ctc)
    
    print("CTC Summary Statistics:")
    print("-" * 25)
    print(f"Total Tax Units: {stats['total_tax_units']}")
    print(f"Units with CTC: {stats['units_with_ctc']}")
    print(f"CTC Participation Rate: {stats['units_with_ctc']/stats['total_tax_units']*100:.1f}%")
    print(f"Total CTC Amount: ${stats['total_ctc_amount']:,.2f}")
    print(f"Average CTC per Unit: ${stats['average_ctc_per_unit']:,.2f}")
    print(f"Average CTC per Recipient: ${stats['average_ctc_per_recipient']:,.2f}")
    print(f"Total Qualifying Children: {stats['total_qualifying_children']}")
    print(f"Total Refundable CTC: ${stats['total_refundable_ctc']:,.2f}")
    print(f"Total Non-refundable CTC: ${stats['total_nonrefundable_ctc']:,.2f}")
    
    # Show breakdown by filing status
    print("\nCTC by Filing Status:")
    print("-" * 22)
    filing_status_stats = tax_units_with_ctc.groupby('filing_status').agg({
        'ctc_ctc_total': ['count', 'sum', 'mean'],
        'ctc_qualifying_children': 'sum'
    }).round(2)
    
    print(filing_status_stats)


if __name__ == "__main__":
    main()
