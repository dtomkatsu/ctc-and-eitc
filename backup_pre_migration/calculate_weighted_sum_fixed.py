import pandas as pd
import numpy as np
from pathlib import Path

def calculate_person_weighted_sum():
    """Calculate weighted sum using person weights (PWGTP)."""
    
    print("Loading data...")
    # Load person data with PWGTP weights
    person_df = pd.read_parquet('data/processed/pums_person_processed.parquet')
    tax_units = pd.read_parquet('src/data/processed/tax_units_rule_based.parquet')
    
    # Ensure consistent ID types
    person_df['SERIALNO'] = person_df['SERIALNO'].astype(str)
    tax_units['SERIALNO'] = tax_units['SERIALNO'].astype(str)
    
    # Create a unique person ID in both dataframes
    person_df['person_id'] = person_df['SERIALNO'] + '_' + person_df.index.astype(str)
    
    # Get all unique person IDs from tax units
    all_filer_ids = set(tax_units['primary_filer_id'].dropna().unique())
    all_filer_ids.update(tax_units['secondary_filer_id'].dropna().unique())
    
    # Calculate total population using PWGTP
    total_population = person_df['PWGTP'].sum()
    
    # Calculate weighted sum of people in tax units
    in_tax_units = person_df['person_id'].isin(all_filer_ids)
    in_tax_units_sum = person_df.loc[in_tax_units, 'PWGTP'].sum()
    
    # Calculate coverage
    coverage = (in_tax_units_sum / total_population) * 100
    
    # Print results
    print("\n=== Person Weight (PWGTP) Analysis ===")
    print(f"Total Population (sum of PWGTP): {total_population:,.0f}")
    print(f"People in Tax Units (PWGTP): {in_tax_units_sum:,.0f}")
    print(f"People Not in Tax Units (PWGTP): {total_population - in_tax_units_sum:,.0f}")
    print(f"Coverage (People in Tax Units / Total): {coverage:.2f}%")
    
    # Additional analysis by age group
    print("\n=== Age Distribution ===")
    person_df['age_group'] = pd.cut(
        person_df['AGEP'], 
        bins=[0, 17, 24, 44, 64, 120],
        labels=['0-17', '18-24', '25-44', '45-64', '65+']
    )
    
    age_dist = person_df.groupby('age_group').agg(
        total_population=('PWGTP', 'sum'),
        in_tax_units=('PWGTP', lambda x: x[person_df.loc[x.index, 'person_id'].isin(all_filer_ids)].sum())
    )
    
    age_dist['coverage'] = (age_dist['in_tax_units'] / age_dist['total_population']) * 100
    print("\nAge Group Distribution (PWGTP):")
    print(age_dist[['total_population', 'in_tax_units', 'coverage']])
    
    # Save detailed results
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # Save person-level data with tax unit status
    person_df['in_tax_unit'] = in_tax_units
    person_df.to_parquet(output_dir / 'person_with_tax_unit_status.parquet')
    
    print(f"\nDetailed results saved to: {output_dir}/")

if __name__ == "__main__":
    calculate_person_weighted_sum()
