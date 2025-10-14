import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

class IncomeDistributionAnalyzer:
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.results = {}
        
    def parse_soi_table(self, file_path: Path) -> List[Dict]:
        """Parse a DOTAX SOI table file into a list of income brackets."""
        data = []
        
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                lines = f.readlines()
                
                # Skip header lines (first 5 lines)
                for line in lines[5:]:
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Split the line by commas, but be careful with quoted values
                    parts = []
                    in_quotes = False
                    current_part = []
                    
                    for char in line:
                        if char == '"':
                            in_quotes = not in_quotes
                        elif char == ',' and not in_quotes:
                            parts.append(''.join(current_part).strip())
                            current_part = []
                        else:
                            current_part.append(char)
                    
                    # Add the last part
                    if current_part:
                        parts.append(''.join(current_part).strip())
                    
                    # Process the parts
                    if len(parts) >= 5:
                        try:
                            # Handle the first line (special case)
                            if line.startswith(',Not,over'):
                                lower = None
                                upper = parts[3].strip('"$') if len(parts) > 3 else None
                                returns = parts[4].strip('"') if len(parts) > 4 else None
                                marginal_rate = parts[5].strip('"%') if len(parts) > 5 else None
                            # Handle the "Over" lines
                            elif line.startswith('Over'):
                                lower = parts[1].strip('"$') if len(parts) > 1 else None
                                upper = parts[3].strip('"$') if len(parts) > 3 else None
                                returns = parts[4].strip('"') if len(parts) > 4 else None
                                marginal_rate = parts[5].strip('"%') if len(parts) > 5 else None
                            # Handle the ",,Over," lines (last bracket)
                            elif line.startswith(',,Over'):
                                lower = parts[3].strip('"$') if len(parts) > 3 else None
                                upper = None
                                returns = parts[4].strip('"') if len(parts) > 4 else None
                                marginal_rate = parts[5].strip('"%') if len(parts) > 5 else None
                            else:
                                continue
                                
                            # Skip the TOTAL line or lines without returns
                            if not returns or returns.lower() == 'total':
                                continue
                                
                            # Clean and convert values
                            try:
                                lower_bound = float(lower.replace(',', '')) if lower and lower.lower() != 'over' else None
                                upper_bound = float(upper.replace(',', '')) if upper and upper.lower() != 'over' else None
                                returns = int(returns.replace(',', '')) if returns and returns.lower() != 'total' else 0
                                
                                # Only add if we have valid data
                                if returns > 0 and (lower_bound is not None or upper_bound is not None):
                                    data.append({
                                        'lower_bound': lower_bound,
                                        'upper_bound': upper_bound,
                                        'returns': returns,
                                        'marginal_rate': marginal_rate
                                    })
                                    
                            except (ValueError, AttributeError) as e:
                                print(f"  Could not parse values in line: {line}")
                                continue
                                
                        except Exception as e:
                            print(f"  Error parsing line: {line}")
                            continue
                            
        except Exception as e:
            print(f"  Error reading {file_path.name}: {str(e)}")
            return []
        
        return data
    
    def estimate_millionaires(self, brackets: List[Dict], alpha: float = 1.5) -> Tuple[int, float]:
        """Estimate number of returns above $1M using Pareto distribution.
        
        Args:
            brackets: List of income bracket dictionaries with 'lower_bound', 'upper_bound', and 'returns'
            alpha: Shape parameter for the Pareto distribution (default 1.5)
            
        Returns:
            Tuple of (estimated number of millionaires, percentage of returns that are millionaires)
        """
        if not brackets:
            return 0, 0.0
            
        # Filter out brackets with no lower bound and sort by lower_bound
        valid_brackets = [b for b in brackets if b.get('lower_bound') is not None]
        if not valid_brackets:
            return 0, 0.0
            
        # Sort brackets by lower bound in descending order
        valid_brackets.sort(key=lambda x: x['lower_bound'], reverse=True)
        
        # Get the highest bracket
        highest_bracket = valid_brackets[0]
        
        # If the highest bracket starts above $1M, all returns in it are millionaires
        if highest_bracket['lower_bound'] >= 1_000_000:
            return highest_bracket['returns'], 100.0
            
        # If the highest bracket is open-ended (no upper bound) and starts below $1M
        if highest_bracket.get('upper_bound') is None and highest_bracket['lower_bound'] > 0:
            lower = highest_bracket['lower_bound']
            count = highest_bracket['returns']
            
            # Use Pareto distribution to estimate the proportion above $1M
            # P(X > x) = (x_m / x)^α where x_m is the lower bound
            prob_above_1m = (lower / 1_000_000) ** alpha
            millionaires = int(round(count * prob_above_1m))
            pct = prob_above_1m * 100
            
            # If we have a very small number of returns, round up to at least 1
            if millionaires == 0 and count > 0 and prob_above_1m > 0:
                millionaires = 1
                
            return millionaires, pct
            
        # If the highest bracket is closed but includes $1M
        if (highest_bracket.get('upper_bound') is not None and 
            highest_bracket['upper_bound'] > 1_000_000):
            # Estimate the proportion of this bracket that's above $1M
            bracket_width = highest_bracket['upper_bound'] - highest_bracket['lower_bound']
            if bracket_width > 0:
                # Simple linear approximation (could be improved)
                proportion_above_1m = (highest_bracket['upper_bound'] - 1_000_000) / bracket_width
                millionaires = int(round(highest_bracket['returns'] * proportion_above_1m))
                pct = (millionaires / highest_bracket['returns']) * 100 if highest_bracket['returns'] > 0 else 0
                return max(1, millionaires), pct
                
        return 0, 0.0
    
    def analyze_filing_status(self, status_name: str, filename: str):
        """Analyze a single filing status from a file."""
        # Try exact filename first
        file_path = self.data_dir / filename
        
        # If not found, try case-insensitive search
        if not file_path.exists():
            files = list(self.data_dir.glob('*'))
            matching_files = [f for f in files if f.name.lower() == filename.lower()]
            if matching_files:
                file_path = matching_files[0]
            else:
                print(f"Warning: File not found for {status_name} ({filename})")
                return
            
        print(f"\nAnalyzing {status_name} from {file_path.name}")
        
        # Parse the data
        brackets = self.parse_soi_table(file_path)
        
        if not brackets:
            print(f"  No data found in {file_path.name}")
            return
            
        # Calculate total returns
        total_returns = sum(b['returns'] for b in brackets)
        print(f"  Total returns: {total_returns:,}")
        
        # Print debug info for the brackets
        print("\n  Parsed brackets:")
        for i, b in enumerate(brackets, 1):
            lb = f"${b['lower_bound']:,}" if b.get('lower_bound') is not None else 'None'
            ub = f"${b['upper_bound']:,}" if b.get('upper_bound') is not None else '∞'
            returns = f"{b['returns']:,}" if 'returns' in b else 'N/A'
            rate = b.get('marginal_rate', 'N/A')
            print(f"    Bracket {i}: {lb:>10} to {ub:<10} | Returns: {returns:>8} | Rate: {rate}")
        
        # Estimate millionaires
        millionaires, pct = self.estimate_millionaires(brackets)
        
        # Store results
        self.results[status_name] = {
            'total_returns': total_returns,
            'estimated_millionaires': millionaires,
            'pct_millionaires': pct,
            'brackets': brackets
        }
        
        print(f"  Estimated millionaires: {millionaires:,} ({pct:.4f}% of returns)")
    
    def analyze_all(self):
        """Analyze all filing statuses."""
        print("Starting analysis of DOTAX SOI income distributions...")
        
        # Define the files and corresponding status names
        status_files = [
            ("Single and Married Filing Separately", "Dotax Soi 2022 - 13A.csv"),
            ("Married Filing Jointly", "Dotax Soi 2022 - 13B.csv"),
            ("Head of Household", "Dotax Soi 2022 - 13C.csv")
        ]
        
        # Analyze each status
        for status_name, file_pattern in status_files:
            self.analyze_filing_status(status_name, file_pattern)
        
        # Print summary
        self.print_summary()
        
        # Save results
        self.save_results()
    
    def print_summary(self):
        """Print a summary of the analysis."""
        print("\n" + "="*80)
        print("INCOME DISTRIBUTION ANALYSIS SUMMARY")
        print("="*80)
        
        if not self.results:
            print("No results to display.")
            return
        
        # Calculate totals
        total_returns = sum(v['total_returns'] for v in self.results.values())
        total_millionaires = sum(v['estimated_millionaires'] for v in self.results.values())
        
        # Print header
        print(f"\n{'Filing Status':<40} {'Returns':>15} {'Millionaires':>15} {'% of Returns':>15} {'% of Millionaires':>20}")
        print("-" * 105)
        
        # Print each status
        for status, data in self.results.items():
            pct_of_total = (data['total_returns'] / total_returns) * 100 if total_returns > 0 else 0
            pct_of_millionaires = (data['estimated_millionaires'] / total_millionaires * 100) if total_millionaires > 0 else 0
            
            print(f"{status:<40} {data['total_returns']:>15,} {data['estimated_millionaires']:>15,} "
                  f"{data['pct_millionaires']:>14.4f}% {pct_of_millionaires:>18.2f}%")
        
        # Print totals
        print("-" * 105)
        print(f"{'TOTAL':<40} {total_returns:>15,} {total_millionaires:>15,} "
              f"{(total_millionaires / total_returns * 100) if total_returns > 0 else 0:>14.4f}% {'100.00%':>20}")
    
    def save_results(self, output_dir: str = "analysis_results/income_distribution"):
        """Save the analysis results to CSV files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        summary_data = []
        for status, data in self.results.items():
            summary_data.append({
                'filing_status': status,
                'total_returns': data['total_returns'],
                'estimated_millionaires': data['estimated_millionaires'],
                'pct_millionaires': data['pct_millionaires']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = output_dir / 'millionaire_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved summary to: {summary_path}")
        
        # Save detailed brackets for each status
        for status, data in self.results.items():
            brackets_df = pd.DataFrame(data['brackets'])
            if not brackets_df.empty:
                # Clean up the status name for filename
                clean_status = status.lower().replace(' ', '_').replace('&', 'and')
                brackets_path = output_dir / f'{clean_status}_brackets.csv'
                brackets_df.to_csv(brackets_path, index=False)
                print(f"Saved {status} brackets to: {brackets_path}")

if __name__ == "__main__":
    analyzer = IncomeDistributionAnalyzer()
    analyzer.analyze_all()
