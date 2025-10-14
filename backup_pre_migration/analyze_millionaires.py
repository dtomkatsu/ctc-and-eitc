import pandas as pd
import numpy as np
from pathlib import Path

def estimate_millionaires(income_brackets, total_returns, alpha=1.5):
    """
    Estimate the number of returns above $1M using a Pareto distribution.
    
    Args:
        income_brackets: List of (lower_bound, upper_bound, count) tuples
        total_returns: Total number of returns in the bracket
        alpha: Shape parameter for the Pareto distribution (default 1.5)
        
    Returns:
        Estimated number of returns above $1M
    """
    if not income_brackets:
        return 0
        
    # Get the highest bracket
    highest_bracket = max(income_brackets, key=lambda x: x[1] if x[1] is not None else float('inf'))
    lower, upper, count = highest_bracket
    
    if upper is None:  # Open-ended bracket
        # For open-ended bracket, use Pareto distribution to estimate tail
        if lower >= 1_000_000:
            return count  # Entire bracket is above $1M
        
        # Estimate number above $1M using Pareto distribution
        # P(X > x) = (x_m / x)^α where x_m is the lower bound
        # For alpha=1.5, this is a common shape parameter for income distributions
        prob_above_1m = (lower / 1_000_000) ** alpha
        return int(round(count * prob_above_1m))
    else:
        # For closed bracket, estimate based on the bracket width
        bracket_width = upper - lower
        if upper >= 1_000_000:
            # If bracket includes $1M, estimate proportion above $1M
            if lower >= 1_000_000:
                return count  # Entire bracket is above $1M
            else:
                # Linear approximation within the bracket
                prop_above = (upper - 1_000_000) / (upper - lower)
                return int(round(count * prop_above))
        else:
            return 0

def analyze_dotax_soi(file_path):
    """Analyze DOTAX SOI data to estimate number of millionaires by filing status."""
    # Read the CSV file
    df = pd.read_csv(file_path, header=None)
    
    # Parse the data (this is a simplified parser, may need adjustment)
    data = []
    current_status = None
    
    for _, row in df.iterrows():
        # Check for filing status headers
        if "Filing Status:" in str(row[0]):
            current_status = str(row[0]).split(":")[1].strip()
            continue
            
        # Check for data rows (look for rows with numbers in the count column)
        if len(row) >= 6 and str(row[4]).replace(',', '').isdigit():
            lower = None
            upper = None
            
            # Parse income range
            if "over" in str(row[0]).lower() and "to" in str(row[2]).lower():
                # Format: "Over X to Y"
                lower = float(str(row[1]).replace('$', '').replace(',', ''))
                upper = float(str(row[3]).replace('$', '').replace(',', ''))
            elif "over" in str(row[0]).lower() and "$" in str(row[2]):
                # Format: "Over X, $Y"
                lower = float(str(row[1]).replace('$', '').replace(',', ''))
                upper = None  # Open-ended
            elif "$" in str(row[0]):
                # Format: "$X"
                lower = float(str(row[0]).replace('$', '').replace(',', ''))
                upper = float(str(row[2]).replace('$', '').replace(',', ''))
                
            if lower is not None:
                count = int(str(row[4]).replace(',', ''))
                data.append({
                    'filing_status': current_status or 'Unknown',
                    'lower_bound': lower,
                    'upper_bound': upper,
                    'returns': count,
                    'marginal_rate': row[5] if len(row) > 5 else None,
                    'avg_rate_before_credits': row[6] if len(row) > 6 else None,
                    'avg_rate_after_credits': row[7] if len(row) > 7 else None
                })
    
    # Convert to DataFrame
    df_parsed = pd.DataFrame(data)
    
    # Estimate millionaires by filing status
    results = {}
    for status in df_parsed['filing_status'].unique():
        status_data = df_parsed[df_parsed['filing_status'] == status]
        brackets = list(zip(status_data['lower_bound'], 
                          status_data['upper_bound'], 
                          status_data['returns']))
        
        # Estimate millionaires
        millionaires = estimate_millionaires(brackets, status_data['returns'].sum())
        total_returns = status_data['returns'].sum()
        
        results[status] = {
            'total_returns': total_returns,
            'estimated_millionaires': millionaires,
            'pct_millionaires': (millionaires / total_returns * 100) if total_returns > 0 else 0
        }
    
    return pd.DataFrame.from_dict(results, orient='index')

if __name__ == "__main__":
    # Path to the DOTAX SOI file
    SOI_FILE = "/Users/dtomkatsu/Downloads/DOTAX/Dotax Soi 2022 - 13A.csv"
    OUTPUT_DIR = "analysis_results/millionaire_analysis"
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    print(f"Analyzing DOTAX SOI data from: {SOI_FILE}")
    
    # Run the analysis
    results = analyze_dotax_soi(SOI_FILE)
    
    # Print results
    print("\nEstimated Number of Millionaires by Filing Status:")
    print("=" * 70)
    
    # Format the results for better display
    formatted_results = results.copy()
    for col in ['total_returns', 'estimated_millionaires']:
        formatted_results[col] = formatted_results[col].apply(lambda x: f"{x:,.0f}")
    formatted_results['pct_millionaires'] = formatted_results['pct_millionaires'].apply(lambda x: f"{x:.4f}%")
    
    print(formatted_results.to_string())
    
    # Print a summary
    print("\nSummary:")
    print(f"Total Estimated Millionaires: {int(results['estimated_millionaires'].sum()):,}")
    print(f"Percentage of All Returns: {results['pct_millionaires'].mean():.4f}%")
    
    # Print by filing status
    print("\nBreakdown by Filing Status:")
    print("-" * 80)
    print(f"{'Filing Status':<40} {'Returns':>15} {'Millionaires':>15} {'% of Returns':>15}")
    print("-" * 80)
    for status, row in results.iterrows():
        print(f"{status:<40} {int(row['total_returns']):>15,} {int(row['estimated_millionaires']):>15,} {row['pct_millionaires']:>15.4f}%")
    
    # Save results
    results.to_csv(f"{OUTPUT_DIR}/millionaire_estimates.csv")
    print(f"\nResults saved to: {OUTPUT_DIR}/millionaire_estimates.csv")
