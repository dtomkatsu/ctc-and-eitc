import pandas as pd
from pathlib import Path

def analyze_high_income_tax_units(tax_units_path, output_dir=None):
    """
    Analyze tax units with $1M+ income.
    
    Args:
        tax_units_path: Path to the saved tax units file (CSV or Parquet)
        output_dir: Optional directory to save the analysis results
        
    Returns:
        Dictionary containing analysis results
    """
    print(f"Loading tax units from: {tax_units_path}")
    
    # Load the tax units data
    if str(tax_units_path).endswith('.parquet'):
        tax_units = pd.read_parquet(tax_units_path)
    else:  # Assume CSV
        tax_units = pd.read_csv(tax_units_path)
    
    print(f"Loaded {len(tax_units):,} tax units")
    
    # Ensure income is numeric
    tax_units['income'] = pd.to_numeric(tax_units['income'], errors='coerce')
    
    # Ensure weight is numeric and handle any missing values
    tax_units['weight'] = pd.to_numeric(tax_units.get('weight', 1), errors='coerce').fillna(1)
    
    # Define the high income threshold ($1M)
    HIGH_INCOME_THRESHOLD = 1_000_000
    
    # Filter for high-income tax units
    high_income = tax_units[tax_units['income'] >= HIGH_INCOME_THRESHOLD].copy()
    
    print(f"Found {len(high_income):,} tax units with income ≥${HIGH_INCOME_THRESHOLD:,}")
    
    # Calculate unweighted counts
    total_units = len(tax_units)
    high_income_count = len(high_income)
    high_income_pct = (high_income_count / total_units) * 100 if total_units > 0 else 0
    
    # Calculate weighted counts
    total_weighted = tax_units['weight'].sum()
    high_income_weighted = high_income['weight'].sum()
    high_income_weighted_pct = (high_income_weighted / total_weighted) * 100 if total_weighted > 0 else 0
    
    # Group by filing status
    if 'filing_status' in tax_units.columns:
        status_groups = tax_units.groupby('filing_status')
        high_income_by_status = high_income.groupby('filing_status')
        
        # Create a summary DataFrame
        summary = pd.DataFrame({
            'total_units': status_groups.size(),
            'total_weighted': status_groups['weight'].sum(),
            'high_income_units': high_income_by_status.size(),
            'high_income_weighted': high_income_by_status['weight'].sum()
        }).fillna(0)
        
        # Calculate percentages
        summary['pct_of_total'] = (summary['high_income_units'] / summary['total_units']) * 100
        summary['pct_of_high_income'] = (summary['high_income_weighted'] / high_income_weighted) * 100 if high_income_weighted > 0 else 0
        summary['pct_weighted'] = (summary['high_income_weighted'] / summary['total_weighted']) * 100
        
        # Sort by weighted count
        summary = summary.sort_values('high_income_weighted', ascending=False)
    else:
        summary = None
    
    # Create results dictionary
    results = {
        'total_units': total_units,
        'total_weighted': total_weighted,
        'high_income_count': high_income_count,
        'high_income_weighted': high_income_weighted,
        'high_income_pct': high_income_pct,
        'high_income_weighted_pct': high_income_weighted_pct,
        'by_filing_status': summary,
        'high_income_sample': high_income.head(10)  # Sample of high-income tax units
    }
    
    # Save results if output directory is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary to CSV
        if summary is not None:
            summary.to_csv(output_dir / 'high_income_by_status.csv')
            print(f"Saved breakdown by filing status to: {output_dir / 'high_income_by_status.csv'}")
        
        # Save full high-income records
        high_income.to_csv(output_dir / 'high_income_units.csv', index=False)
        print(f"Saved full high-income records to: {output_dir / 'high_income_units.csv'}")
        
        # Save summary statistics
        with open(output_dir / 'summary.txt', 'w') as f:
            f.write(f"Total tax units: {total_units:,}\n")
            f.write(f"Total weighted tax units: {total_weighted:,.0f}\n")
            f.write(f"High-income tax units (>${HIGH_INCOME_THRESHOLD:,}): {high_income_count:,}\n")
            f.write(f"High-income weighted count: {high_income_weighted:,.0f}\n")
            f.write(f"Percentage of high-income units: {high_income_pct:.4f}%\n")
            f.write(f"Weighted percentage of high-income units: {high_income_weighted_pct:.4f}%\n")
        
        print(f"Saved summary to: {output_dir / 'summary.txt'}")
    
    return results

def print_analysis_results(results):
    """Print the analysis results in a readable format."""
    print("\n" + "="*80)
    print("HIGH INCOME TAX UNITS ANALYSIS")
    print("="*80)
    print(f"\nTotal tax units: {results['total_units']:,}")
    print(f"Total weighted tax units: {results['total_weighted']:,.0f}")
    print(f"\nTax units with income ≥$1M: {results['high_income_count']:,}")
    print(f"Weighted count: {results['high_income_weighted']:,.0f}")
    print(f"\nPercentage of all tax units: {results['high_income_pct']:.4f}%")
    print(f"Weighted percentage: {results['high_income_weighted_pct']:.4f}%")
    
    if results['by_filing_status'] is not None:
        print("\n" + "="*80)
        print("BREAKDOWN BY FILING STATUS")
        print("="*80)
        print("\nCounts and percentages of high-income tax units by filing status:")
        # Format the DataFrame for better display
        formatted_df = results['by_filing_status'].copy()
        for col in ['total_units', 'total_weighted', 'high_income_units', 'high_income_weighted']:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:,.0f}")
        for col in ['pct_of_total', 'pct_weighted']:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}%")
        formatted_df['pct_of_high_income'] = formatted_df['pct_of_high_income'].apply(lambda x: f"{x:.2f}%")
        print(formatted_df.to_string())
        
        # Also print a summary of the high-income distribution
        print("\nHigh-Income Distribution by Filing Status:")
        print("-" * 50)
        print(f"{'Filing Status':<20} {'Count':>10} {'Weighted':>15} {'% of Total':>15}")
        print("-" * 50)
        for status, row in results['by_filing_status'].iterrows():
            print(f"{status:<20} {int(row['high_income_units']):>10,} {row['high_income_weighted']:>15,.0f} {row['pct_weighted']:>15}")
    
    print("\n" + "="*80)
    print("SAMPLE OF HIGH-INCOME TAX UNITS")
    print("="*80)
    print(results['high_income_sample'].to_string())

if __name__ == "__main__":
    # Path to the tax units file
    TAX_UNITS_PATH = "/Users/dtomkatsu/CascadeProjects/ctc-and-eitc/data/processed/tax_units_soi_calibrated.parquet"
    OUTPUT_DIR = "analysis_results/high_income_analysis"
    
    print(f"Starting analysis of high-income tax units...")
    
    # Run the analysis
    results = analyze_high_income_tax_units(TAX_UNITS_PATH, OUTPUT_DIR)
    
    # Print the results
    print_analysis_results(results)
    
    print("\nAnalysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
