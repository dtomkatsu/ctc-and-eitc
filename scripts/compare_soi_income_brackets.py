#!/usr/bin/env python3
"""
Compare income bracket distributions between DOTAX SOI and constructed tax units.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'analysis_results' / 'income_distribution'

def load_soi_data():
    """Load and parse SOI tables with income brackets."""
    
    def parse_soi_table(file_path, filing_status):
        """Parse a single SOI table."""
        df = pd.read_csv(file_path, header=None, skiprows=6, encoding='latin1')
        df = df.dropna(how='all')
        df = df[~df[3].astype(str).str.contains('TOTAL', na=False)]
        
        def extract_numeric(val):
            if pd.isna(val):
                return None
            try:
                return float(str(val).replace('$', '').replace(',', '').strip())
            except:
                return None
        
        df['num_returns'] = df[4].apply(extract_numeric)
        df = df.dropna(subset=['num_returns'])
        
        records = []
        for idx, row in df.iterrows():
            num_returns = row['num_returns']
            
            # Parse income brackets
            if pd.notna(row[1]) and str(row[1]).strip() == 'Not':
                income_min = 0
                income_max = extract_numeric(row[3])
                bracket_label = f"Under ${int(income_max):,}"
            elif pd.notna(row[0]) and str(row[0]).strip() == 'Over':
                income_min = extract_numeric(row[1])
                if pd.notna(row[2]) and str(row[2]).strip() == 'to':
                    income_max = extract_numeric(row[3])
                    bracket_label = f"${int(income_min):,}-${int(income_max):,}"
                else:
                    income_max = float('inf')
                    bracket_label = f"Over ${int(income_min):,}"
            elif pd.isna(row[0]) and pd.isna(row[1]) and pd.notna(row[2]) and str(row[2]).strip().lower() == 'over':
                income_min = extract_numeric(row[3])
                income_max = float('inf')
                bracket_label = f"Over ${int(income_min):,}"
            else:
                continue
            
            if income_min is not None and income_max is not None:
                records.append({
                    'income_min': income_min,
                    'income_max': income_max,
                    'bracket_label': bracket_label,
                    'num_returns': num_returns,
                    'filing_status': filing_status
                })
        
        return pd.DataFrame(records)
    
    # Load all three tables
    tables = []
    table_mapping = {
        '13A': 'single',  # Note: This includes MFS in SOI
        '13B': 'married_filing_jointly',
        '13C': 'head_of_household'
    }
    
    for table_file, status in table_mapping.items():
        file_path = DATA_DIR / 'raw' / f'Dotax Soi 2022 - {table_file}.csv'
        if file_path.exists():
            df = parse_soi_table(file_path, status)
            tables.append(df)
    
    return pd.concat(tables, ignore_index=True)

def load_model_data():
    """Load constructed tax units and aggregate by income brackets."""
    # Load the latest tax units file
    tax_units_file = sorted((DATA_DIR / 'processed').glob('tax_units_regenerated_*.parquet'))[-1]
    tax_units = pd.read_parquet(tax_units_file)
    
    print(f"Loaded tax units from: {tax_units_file.name}")
    print(f"Total tax units: {len(tax_units):,}")
    print(f"Total weighted: {tax_units['weight'].sum():,.0f}")
    
    return tax_units

def map_to_soi_brackets(tax_units, soi_data):
    """Map tax units to SOI income brackets by filing status."""
    
    results = []
    
    # Process each filing status separately
    for filing_status in ['single', 'married_filing_jointly', 'head_of_household']:
        # Get SOI brackets for this filing status
        soi_brackets = soi_data[soi_data['filing_status'] == filing_status].copy()
        soi_brackets = soi_brackets.sort_values('income_min').reset_index(drop=True)
        
        # Get tax units for this filing status
        status_units = tax_units[tax_units['filing_status'] == filing_status].copy()
        
        # Map each SOI bracket
        for _, bracket in soi_brackets.iterrows():
            min_income = bracket['income_min']
            max_income = bracket['income_max']
            
            # Filter tax units in this bracket
            if max_income == float('inf'):
                mask = status_units['income'] > min_income
            else:
                mask = (status_units['income'] > min_income) & (status_units['income'] <= max_income)
            
            bracket_units = status_units[mask]
            model_count = bracket_units['weight'].sum()
            
            results.append({
                'filing_status': filing_status,
                'bracket_label': bracket['bracket_label'],
                'income_min': min_income,
                'income_max': max_income,
                'soi_returns': bracket['num_returns'],
                'model_returns': model_count,
                'difference': model_count - bracket['num_returns'],
                'pct_difference': ((model_count - bracket['num_returns']) / bracket['num_returns'] * 100) if bracket['num_returns'] > 0 else 0
            })
    
    return pd.DataFrame(results)

def create_comparison_plots(comparison_df):
    """Create visualization comparing SOI vs Model by income bracket."""
    
    for filing_status in ['single', 'married_filing_jointly', 'head_of_household']:
        df = comparison_df[comparison_df['filing_status'] == filing_status].copy()
        df = df.sort_values('income_min')
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Absolute counts
        x = np.arange(len(df))
        width = 0.35
        
        axes[0].bar(x - width/2, df['soi_returns'], width, label='SOI', alpha=0.8, color='#2E86AB')
        axes[0].bar(x + width/2, df['model_returns'], width, label='Model', alpha=0.8, color='#A23B72')
        
        axes[0].set_xlabel('Income Bracket', fontsize=11)
        axes[0].set_ylabel('Number of Returns', fontsize=11)
        axes[0].set_title(f'{filing_status.replace("_", " ").title()}: SOI vs Model Returns by Income Bracket', 
                         fontsize=13, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(df['bracket_label'], rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (soi, model) in enumerate(zip(df['soi_returns'], df['model_returns'])):
            axes[0].text(i - width/2, soi, f'{soi:,.0f}', ha='center', va='bottom', fontsize=8)
            axes[0].text(i + width/2, model, f'{model:,.0f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Percentage difference
        colors = ['#D62828' if x < -10 else '#F77F00' if x < -5 else '#06A77D' if x < 5 else '#2E86AB' 
                  for x in df['pct_difference']]
        
        axes[1].bar(x, df['pct_difference'], color=colors, alpha=0.8)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        axes[1].axhline(y=-5, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
        axes[1].axhline(y=5, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
        
        axes[1].set_xlabel('Income Bracket', fontsize=11)
        axes[1].set_ylabel('Difference (%)', fontsize=11)
        axes[1].set_title('Percentage Difference (Model - SOI)', fontsize=12, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(df['bracket_label'], rotation=45, ha='right')
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, pct in enumerate(df['pct_difference']):
            axes[1].text(i, pct, f'{pct:+.1f}%', ha='center', 
                        va='bottom' if pct > 0 else 'top', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'bracket_comparison_{filing_status}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot: bracket_comparison_{filing_status}.png")

def main():
    print("="*80)
    print("INCOME BRACKET COMPARISON: SOI vs CONSTRUCTED TAX UNITS")
    print("="*80)
    
    # Load data
    print("\n1. Loading SOI data...")
    soi_data = load_soi_data()
    print(f"   Loaded {len(soi_data)} SOI brackets")
    print(f"   Total SOI returns: {soi_data['num_returns'].sum():,.0f}")
    
    print("\n2. Loading model data...")
    tax_units = load_model_data()
    
    print("\n3. Mapping tax units to SOI brackets...")
    comparison = map_to_soi_brackets(tax_units, soi_data)
    
    # Save detailed comparison
    comparison.to_csv(OUTPUT_DIR / 'bracket_by_bracket_comparison.csv', index=False)
    print(f"   Saved: bracket_by_bracket_comparison.csv")
    
    # Print summary by filing status
    print("\n" + "="*80)
    print("DETAILED COMPARISON BY FILING STATUS")
    print("="*80)
    
    for filing_status in ['single', 'married_filing_jointly', 'head_of_household']:
        df = comparison[comparison['filing_status'] == filing_status].copy()
        df = df.sort_values('income_min')
        
        print(f"\n{filing_status.replace('_', ' ').upper()}")
        print("-" * 80)
        print(f"{'Bracket':<25} {'SOI':>12} {'Model':>12} {'Diff':>12} {'% Diff':>10}")
        print("-" * 80)
        
        for _, row in df.iterrows():
            print(f"{row['bracket_label']:<25} {row['soi_returns']:>12,.0f} {row['model_returns']:>12,.0f} "
                  f"{row['difference']:>12,.0f} {row['pct_difference']:>9.1f}%")
        
        total_soi = df['soi_returns'].sum()
        total_model = df['model_returns'].sum()
        total_diff = total_model - total_soi
        total_pct = (total_diff / total_soi * 100) if total_soi > 0 else 0
        
        print("-" * 80)
        print(f"{'TOTAL':<25} {total_soi:>12,.0f} {total_model:>12,.0f} "
              f"{total_diff:>12,.0f} {total_pct:>9.1f}%")
        print()
    
    # Create visualizations
    print("\n4. Creating comparison plots...")
    create_comparison_plots(comparison)
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for filing_status in ['single', 'married_filing_jointly', 'head_of_household']:
        df = comparison[comparison['filing_status'] == filing_status]
        
        # Find brackets with largest discrepancies
        worst_over = df.nlargest(3, 'pct_difference')
        worst_under = df.nsmallest(3, 'pct_difference')
        
        print(f"\n{filing_status.replace('_', ' ').upper()}:")
        print(f"  Average % difference: {df['pct_difference'].mean():.1f}%")
        print(f"  Std dev % difference: {df['pct_difference'].std():.1f}%")
        print(f"  Brackets within ±5%: {len(df[df['pct_difference'].abs() <= 5])}/{len(df)}")
        print(f"  Brackets within ±10%: {len(df[df['pct_difference'].abs() <= 10])}/{len(df)}")
        
        if len(worst_over) > 0:
            print(f"\n  Largest overcounts:")
            for _, row in worst_over.iterrows():
                print(f"    {row['bracket_label']}: {row['pct_difference']:+.1f}%")
        
        if len(worst_under) > 0:
            print(f"\n  Largest undercounts:")
            for _, row in worst_under.iterrows():
                print(f"    {row['bracket_label']}: {row['pct_difference']:+.1f}%")
    
    print(f"\n{'='*80}")
    print(f"Analysis complete! Results saved to: {OUTPUT_DIR}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
