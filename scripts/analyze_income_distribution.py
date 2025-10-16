#!/usr/bin/env python3
"""
Analyze income distribution by filing status and compare with DOTAX SOI tables.
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'analysis_results' / 'income_distribution'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the latest tax units file
tax_units_file = sorted((PROJECT_ROOT / 'data' / 'processed').glob('tax_units_regenerated_*.parquet'))[-1]
tax_units = pd.read_parquet(tax_units_file)

# Define income brackets (matching SOI tables)
INCOME_BRACKETS = [
    (0, 10_000),
    (10_000, 25_000),
    (25_000, 50_000),
    (50_000, 75_000),
    (75_000, 100_000),
    (100_000, 200_000),
    (200_000, 500_000),
    (500_000, 1_000_000),
    (1_000_000, 1_500_000),
    (1_500_000, 2_000_000),
    (2_000_000, 5_000_000),
    (5_000_000, 10_000_000),
    (10_000_000, float('inf'))
]

def load_soi_table(file_path, filing_status):
    """Load a single SOI table and extract relevant information."""
    # Read the CSV file with proper encoding
    df = pd.read_csv(file_path, header=None, skiprows=6, encoding='latin1')
    
    # Clean up the data - handle different formats
    df = df.dropna(how='all')
    
    # Remove the TOTAL row
    df = df[~df[3].astype(str).str.contains('TOTAL', na=False)]
    
    # Extract numeric values
    def extract_numeric(val):
        if pd.isna(val):
            return None
        try:
            # Handle cases like "82,072" or "$82,072"
            return float(str(val).replace('$', '').replace(',', '').strip())
        except:
            return None
    
    # The 5th column (index 4) contains the number of returns
    df['num_returns'] = df[4].apply(extract_numeric)
    df = df.dropna(subset=['num_returns'])
    
    # Parse income brackets
    records = []
    for idx, row in df.iterrows():
        num_returns = row['num_returns']
        
        # Check if this is the first row (Not over $X)
        if pd.notna(row[1]) and str(row[1]).strip() == 'Not':
            income_min = 0
            income_max = extract_numeric(row[3])
        # Check if this is an "Over" row with upper bound
        elif pd.notna(row[0]) and str(row[0]).strip() == 'Over':
            income_min = extract_numeric(row[1])
            # Check if there's an upper bound
            if pd.notna(row[2]) and str(row[2]).strip() == 'to':
                income_max = extract_numeric(row[3])
            else:
                # This is "Over $X" with no upper bound
                income_max = float('inf')
        # Check if this is the last row (blank, blank, Over, $X)
        elif pd.isna(row[0]) and pd.isna(row[1]) and pd.notna(row[2]) and str(row[2]).strip().lower() == 'over':
            income_min = extract_numeric(row[3])
            income_max = float('inf')
        else:
            continue
        
        if income_min is not None and income_max is not None:
            records.append({
                'income_min': income_min,
                'income_max': income_max,
                'num_returns': num_returns,
                'filing_status': filing_status
            })
    
    result = pd.DataFrame(records)
    return result

def load_soi_tables():
    """Load and combine SOI tables 13A, 13B, and 13C."""
    tables = []
    
    # Map tables to filing statuses
    table_mapping = {
        '13A': 'single',
        '13B': 'married_filing_jointly',
        '13C': 'head_of_household'
    }
    
    for table_file, status in table_mapping.items():
        file_path = DATA_DIR / 'raw' / f'Dotax Soi 2022 - {table_file}.csv'
        if file_path.exists():
            df = load_soi_table(file_path, status)
            tables.append(df)
    
    if not tables:
        raise FileNotFoundError("No SOI table files found. Please check the data/raw directory.")
    
    return pd.concat(tables, ignore_index=True)

def analyze_tax_units(tax_units, income_col='income'):
    """Analyze tax unit distribution by income bracket and filing status."""
    # Create a copy to avoid modifying the original
    df = tax_units.copy()
    
    # Define income brackets from SOI tables
    brackets = [
        (0, 2400), (2400, 4800), (4800, 9600), (9600, 14400), 
        (14400, 19200), (19200, 24000), (24000, 36000), 
        (36000, 48000), (48000, 150000), (150000, 175000),
        (175000, 200000), (200000, float('inf'))
    ]
    
    labels = [
        'Under $2,400', '$2,400-$4,800', '$4,800-$9,600', '$9,600-$14,400',
        '$14,400-$19,200', '$19,200-$24,000', '$24,000-$36,000',
        '$36,000-$48,000', '$48,000-$150,000', '$150,000-$175,000',
        '$175,000-$200,000', '$200,000+'
    ]
    
    # Create income bracket column
    df['income_bracket'] = pd.cut(
        df[income_col],
        bins=[b[0] for b in brackets] + [float('inf')],
        labels=labels,
        right=False
    )
    
    # Group by income bracket and filing status
    dist = df.groupby(['income_bracket', 'filing_status'])['weight'].sum().unstack().fillna(0)
    
    # Calculate percentages
    dist_pct = dist.div(dist.sum(axis=1), axis=0) * 100
    
    # Add total column
    dist['Total'] = dist.sum(axis=1)
    dist_pct['Total'] = 100
    
    # Ensure all brackets are represented
    for label in labels:
        if label not in dist.index:
            dist.loc[label] = 0
            dist_pct.loc[label] = 0
    
    # Sort by income bracket
    dist = dist.reindex(labels)
    dist_pct = dist_pct.reindex(labels)
    
    return dist, dist_pct

def plot_distribution(soi_dist, model_dist, title, filename):
    """Plot comparison between SOI and model distributions."""
    plt.style.use('seaborn')
    
    # Prepare data
    soi_melted = soi_dist.reset_index().melt(id_vars='income_bracket', var_name='filing_status', value_name='count')
    model_melted = model_dist.reset_index().melt(id_vars='income_bracket', var_name='filing_status', value_name='count')
    
    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))
    
    # Plot SOI Distribution
    sns.barplot(
        data=soi_melted, 
        x='income_bracket', 
        y='count', 
        hue='filing_status',
        ax=ax1,
        palette='viridis',
        edgecolor='black',
        linewidth=0.5
    )
    ax1.set_title(f'DOTAX SOI 2022 - {title}', fontsize=14, pad=20)
    ax1.set_ylabel('Number of Returns', fontsize=12)
    ax1.set_xlabel('Taxable Income Bracket', fontsize=12)
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    ax1.legend(title='Filing Status', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot Model Distribution
    sns.barplot(
        data=model_melted, 
        x='income_bracket', 
        y='count', 
        hue='filing_status',
        ax=ax2,
        palette='viridis',
        edgecolor='black',
        linewidth=0.5
    )
    ax2.set_title(f'Model Results - {title}', fontsize=14, pad=20)
    ax2.set_ylabel('Weighted Number of Returns', fontsize=12)
    ax2.set_xlabel('Income Bracket', fontsize=12)
    ax2.tick_params(axis='x', rotation=45, labelsize=10)
    ax2.legend(title='Filing Status', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()

def main():
    print(f"Analyzing tax units from: {tax_units_file}")
    
    try:
        # Load SOI data
        print("Loading DOTAX SOI tables...")
        soi = load_soi_tables()
        
        # Pivot SOI data to get counts by income bracket and filing status
        soi_dist = soi.pivot_table(
            index=['income_min', 'income_max'],
            columns='filing_status',
            values='num_returns',
            aggfunc='sum',
            fill_value=0
        ).reset_index()
        
        # Ensure we have all filing statuses
        for status in ['single', 'married_filing_jointly', 'head_of_household']:
            if status not in soi_dist.columns:
                soi_dist[status] = 0
        
        # Create income bracket labels for SOI data
        def format_income_bracket(row):
            min_val = row['income_min']
            max_val = row['income_max']
            
            # Handle special cases
            if pd.isna(min_val) or pd.isna(max_val):
                return 'Unknown'
            
            # Format the bracket string
            if min_val == 0:
                if max_val == float('inf'):
                    return 'All Incomes'
                return f'Under ${max_val:,.0f}'
            elif max_val == float('inf'):
                return f'${min_val:,.0f}+'
            else:
                return f'${min_val:,.0f}-${max_val:,.0f}'
        
        soi_dist['income_bracket'] = soi_dist.apply(format_income_bracket, axis=1)
        
        # Sort by income_min for consistent ordering
        soi_dist = soi_dist.sort_values('income_min').reset_index(drop=True)
        
        # Analyze tax units from model
        print("Analyzing tax units...")
        dist, dist_pct = analyze_tax_units(tax_units)
        
        # Save results
        print("Saving results...")
        dist.to_csv(OUTPUT_DIR / 'income_distribution_counts.csv')
        dist_pct.to_csv(OUTPUT_DIR / 'income_distribution_percentages.csv')
        
        # Ensure we have all required columns in the output
        for status in ['single', 'married_filing_jointly', 'head_of_household']:
            if status not in dist.columns:
                dist[status] = 0
            if status not in dist_pct.columns:
                dist_pct[status] = 0
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print("=" * 50)
        print(f"Total tax units in model: {dist[['single', 'married_filing_jointly', 'head_of_household']].sum().sum():,.0f}")
        print(f"Total returns in SOI data: {soi_dist[['single', 'married_filing_jointly', 'head_of_household']].sum().sum():,.0f}")
        
        # Calculate overall distribution
        print("\nOverall Distribution:")
        print("-" * 30)
        for status in ['single', 'married_filing_jointly', 'head_of_household']:
            if status in soi_dist.columns and status in dist.columns:
                soi_total = soi_dist[status].sum()
                model_total = dist[status].sum()
                pct_diff = ((model_total - soi_total) / soi_total * 100) if soi_total > 0 else float('nan')
                print(f"{status.replace('_', ' ').title()}: SOI={soi_total:,.0f}, Model={model_total:,.0f} ({pct_diff:+.1f}%)")
        
        print(f"\nAnalysis complete! Results saved to: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()
