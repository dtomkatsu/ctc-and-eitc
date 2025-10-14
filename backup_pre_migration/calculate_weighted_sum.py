import pandas as pd
import numpy as np
from pathlib import Path

def calculate_person_weighted_sum():
    """Calculate weighted sum using person weights (PWGTP)."""
    
    # Load the data
    print("Loading data...")
    person_df = pd.read_parquet('data/processed/pums_person_processed.parquet')
    tax_units = pd.read_parquet('src/data/processed/tax_units_rule_based.parquet')
    
    # Ensure SERIALNO is string for consistent merging
    person_df['SERIALNO'] = person_df['SERIALNO'].astype(str)
    tax_units['SERIALNO'] = tax_units['SERIALNO'].astype(str)
    
    # 1. Calculate total population using PWGTP
    total_population = person_df['PWGTP'].sum()
    print(f"\nTotal population (sum of PWGTP): {total_population:,.0f}")
    
    # 2. Calculate weighted sum by tax unit
    print("\nCalculating weighted sums by tax unit type...")
    
    # Get all filer IDs from tax units
    all_filer_ids = set(tax_units['primary_filer_id']) | \
                   set(tax_units['secondary_filer_id'].dropna())
    
    # Filter to only include people in tax units
    in_tax_units = person_df.index.astype(str).isin(all_filer_ids)
    
    # Calculate weighted sums
    results = {
        'Total Population (PWGTP)': total_population,
        'People in Tax Units (PWGTP)': person_df.loc[in_tax_units, 'PWGTP'].sum(),
        'People Not in Tax Units (PWGTP)': person_df.loc[~in_tax_units, 'PWGTP'].sum(),
        'Coverage (People in Tax Units / Total)': 
            person_df.loc[in_tax_units, 'PWGTP'].sum() / total_population * 100
    }
    
    # Print results
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k}: {v:.2f}%" if '%' in k else f"{k}: {v:,.0f}")
        else:
            print(f"{k}: {v}")
    
    # 3. Save detailed results
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # Save person-level data with tax unit info
    person_df['in_tax_unit'] = in_tax_units
    person_df.to_parquet(output_dir / 'person_with_tax_unit_status.parquet')
    
    print(f"\nDetailed results saved to: {output_dir}/")

if __name__ == "__main__":
    calculate_person_weighted_sum()
