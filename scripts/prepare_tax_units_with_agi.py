#!/usr/bin/env python3
"""
Prepare tax units with AGI adjustments for different use cases.

This script:
1. Loads the latest tax units (with filing threshold removed)
2. Creates three versions:
   - Original: total_income (for general analysis)
   - AGI: agi (for tax revenue estimates)
   - Taxable: taxable_income (for SOI comparison)
3. Saves all versions for easy access
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
from src.tax.units.income_adjustments import create_income_versions

def main():
    print("="*80)
    print("PREPARE TAX UNITS WITH AGI ADJUSTMENTS")
    print("="*80)
    
    # Find latest tax units file
    data_dir = project_root / 'data' / 'processed'
    
    # Look for calibrated files first, then regenerated files
    files = sorted(data_dir.glob('tax_units_filing_status_calibrated_*.parquet'))
    if not files:
        files = sorted(data_dir.glob('tax_units_regenerated_*.parquet'))
    
    if not files:
        print("\nERROR: No tax units files found!")
        print("Please run: python scripts/regenerate_tax_units.py")
        return
    
    tax_units_file = files[-1]
    print(f"\nLoading tax units from: {tax_units_file.name}")
    
    # Load original tax units
    tax_units = pd.read_parquet(tax_units_file)
    print(f"Loaded {len(tax_units):,} tax units")
    print(f"Total weighted: {tax_units['weight'].sum():,.0f}")
    
    # Create versions
    print("\n" + "="*80)
    print("CREATING INCOME VERSIONS")
    print("="*80)
    
    versions = create_income_versions(tax_units, tax_year=2022, verbose=True)
    
    # Save versions
    print("\n" + "="*80)
    print("SAVING VERSIONS")
    print("="*80)
    
    output_dir = project_root / 'data' / 'processed'
    
    for name, df in versions.items():
        output_file = output_dir / f'tax_units_{name}.parquet'
        df.to_parquet(output_file, index=False)
        print(f"\n✓ Saved '{name}' version to: {output_file.name}")
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        
        # Show key columns
        if name == 'original':
            print(f"  Key column: 'income' (total income)")
        elif name == 'agi':
            print(f"  Key columns: 'income' (total), 'agi', 'agi_adjustments'")
        else:  # taxable
            print(f"  Key columns: 'income' (total), 'agi', 'taxable_income', 'standard_deduction'")
    
    # Create usage guide
    print("\n" + "="*80)
    print("USAGE GUIDE")
    print("="*80)
    
    print("\n1. FOR GENERAL ANALYSIS (use total income):")
    print("   tax_units = pd.read_parquet('data/processed/tax_units_original.parquet')")
    print("   # Use 'income' column")
    
    print("\n2. FOR TAX REVENUE ESTIMATES (use AGI):")
    print("   tax_units = pd.read_parquet('data/processed/tax_units_agi.parquet')")
    print("   # Use 'agi' column for tax calculations")
    print("   # AGI = total income - adjustments (IRA, student loans, etc.)")
    
    print("\n3. FOR SOI COMPARISON (use taxable income):")
    print("   tax_units = pd.read_parquet('data/processed/tax_units_taxable.parquet')")
    print("   # Use 'taxable_income' column for bracket comparison")
    print("   # Taxable income = AGI - standard deduction")
    
    # Show income comparison
    print("\n" + "="*80)
    print("INCOME COMPARISON")
    print("="*80)
    
    comparison = pd.DataFrame({
        'Version': ['Original', 'AGI', 'Taxable'],
        'Column': ['income', 'agi', 'taxable_income'],
        'Average': [
            versions['original']['income'].mean(),
            versions['agi']['agi'].mean(),
            versions['taxable']['taxable_income'].mean()
        ],
        'Total (Weighted)': [
            (versions['original']['income'] * versions['original']['weight']).sum(),
            (versions['agi']['agi'] * versions['agi']['weight']).sum(),
            (versions['taxable']['taxable_income'] * versions['taxable']['weight']).sum()
        ]
    })
    
    print("\n" + comparison.to_string(index=False))
    
    # Calculate reductions
    print("\n" + "="*80)
    print("INCOME REDUCTIONS")
    print("="*80)
    
    avg_total = versions['original']['income'].mean()
    avg_agi = versions['agi']['agi'].mean()
    avg_taxable = versions['taxable']['taxable_income'].mean()
    
    print(f"\nAverage total income:      ${avg_total:>12,.0f}")
    print(f"Average AGI:               ${avg_agi:>12,.0f}  ({(avg_agi/avg_total)*100:.1f}% of total)")
    print(f"Average taxable income:    ${avg_taxable:>12,.0f}  ({(avg_taxable/avg_total)*100:.1f}% of total)")
    
    print(f"\nReduction from total to AGI:     ${avg_total - avg_agi:>12,.0f}  ({((avg_total-avg_agi)/avg_total)*100:.1f}%)")
    print(f"Reduction from AGI to taxable:   ${avg_agi - avg_taxable:>12,.0f}  ({((avg_agi-avg_taxable)/avg_agi)*100:.1f}%)")
    print(f"Total reduction:                 ${avg_total - avg_taxable:>12,.0f}  ({((avg_total-avg_taxable)/avg_total)*100:.1f}%)")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print("\nYou can now use the prepared tax units for:")
    print("  ✓ Tax revenue estimates (use 'agi' version)")
    print("  ✓ SOI comparison (use 'taxable' version)")
    print("  ✓ General analysis (use 'original' version)")
    print()

if __name__ == '__main__':
    main()
