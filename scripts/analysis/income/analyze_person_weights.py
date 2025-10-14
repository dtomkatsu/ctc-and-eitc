import pandas as pd
import numpy as np
from pathlib import Path

def analyze_person_weights():
    """Analyze person weights (PWGTP) from PUMS person data."""
    
    print("Loading person data...")
    # Load person data with PWGTP weights
    person_df = pd.read_parquet('data/processed/pums_person_processed.parquet')
    
    # Basic statistics
    total_population = person_df['PWGTP'].sum()
    avg_weight = person_df['PWGTP'].mean()
    
    print("\n=== Person Weight (PWGTP) Analysis ===")
    print(f"Total Population (sum of PWGTP): {total_population:,.0f}")
    print(f"Number of people in sample: {len(person_df):,}")
    print(f"Average weight per person: {avg_weight:.2f}")
    
    # Age distribution
    print("\n=== Age Distribution ===")
    person_df['age_group'] = pd.cut(
        person_df['AGEP'], 
        bins=[0, 17, 24, 44, 64, 120],
        labels=['0-17', '18-24', '25-44', '45-64', '65+']
    )
    
    age_dist = person_df.groupby('age_group').agg(
        count=('PWGTP', 'count'),
        weighted_population=('PWGTP', 'sum'),
        avg_weight=('PWGTP', 'mean')
    )
    
    age_dist['pct_of_total'] = (age_dist['weighted_population'] / total_population) * 100
    
    print("\nAge Group Distribution:")
    print(age_dist[['count', 'weighted_population', 'pct_of_total', 'avg_weight']])
    
    # Save detailed results
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # Save age distribution to CSV
    age_dist.to_csv(output_dir / 'age_distribution_pwgpt.csv')
    
    # Save person data with age groups for further analysis
    person_df.to_parquet(output_dir / 'person_data_with_age_groups.parquet')
    
    print(f"\nDetailed results saved to: {output_dir}/")

if __name__ == "__main__":
    analyze_person_weights()
