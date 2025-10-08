#!/usr/bin/env python3
"""
Analyze tax credits from SOI data.

This script analyzes Child Tax Credit (CTC) and Earned Income Tax Credit (EITC)
data from IRS SOI files.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import json

# Project directories
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'data'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
REPORTS_DIR = PROJECT_DIR / 'reports'

# Create reports directory if it doesn't exist
REPORTS_DIR.mkdir(exist_ok=True)

class TaxCreditAnalyzer:
    """Analyze tax credits from SOI data."""
    
    def __init__(self, year: int = 2020):
        """Initialize with the tax year to analyze."""
        self.year = year
        self.data = {}
        self.results = {}
        
    def load_data(self, file_pattern: str = 'processed_*.csv') -> None:
        """Load processed SOI data files."""
        print("Loading processed data...")
        
        for filepath in PROCESSED_DATA_DIR.glob(file_pattern):
            filename = filepath.stem.replace('processed_', '')
            print(f"Loading {filename}...")
            
            try:
                df = pd.read_csv(filepath)
                self.data[filename] = df
                print(f"  Loaded {len(df):,} rows")
            except Exception as e:
                print(f"  Error loading {filepath}: {e}")
    
    def analyze_ctc(self) -> Dict:
        """Analyze Child Tax Credit data."""
        results = {}
        
        for name, df in self.data.items():
            if 'nctc' not in df.columns:
                continue
                
            print(f"\nAnalyzing CTC in {name}...")
            
            # Basic stats
            total_ctc = df['actc'].sum()
            total_returns = df['nret'].sum()
            avg_ctc = total_ctc / total_returns if total_returns > 0 else 0
            
            results[name] = {
                'total_ctc': total_ctc,
                'total_returns': total_returns,
                'avg_ctc_per_return': avg_ctc,
                'total_returns_with_ctc': df['nctc'].sum(),
                'avg_ctc_per_claim': df['actc'].sum() / df['nctc'].sum() if df['nctc'].sum() > 0 else 0,
                'refundable_pct': (df['nctcref'].sum() / df['nctc'].sum()) * 100 if df['nctc'].sum() > 0 else 0
            }
            
            print(f"  Total CTC: ${total_ctc:,.0f}")
            print(f"  Average CTC per return: ${avg_ctc:,.0f}")
        
        return results
    
    def analyze_eitc(self) -> Dict:
        """Analyze Earned Income Tax Credit data."""
        results = {}
        
        for name, df in self.data.items():
            if 'eitc' not in df.columns:
                continue
                
            print(f"\nAnalyzing EITC in {name}...")
            
            # Basic stats
            total_eitc = df['eitc'].sum()
            total_returns = df['nret'].sum()
            
            results[name] = {
                'total_eitc': total_eitc,
                'total_returns': total_returns,
                'avg_eitc_per_return': total_eitc / total_returns if total_returns > 0 else 0,
                'total_returns_with_eitc': df['nret_eitc'].sum() if 'nret_eitc' in df.columns else None,
                'total_children': df['nchild'].sum() if 'nchild' in df.columns else None
            }
            
            print(f"  Total EITC: ${total_eitc:,.0f}")
            print(f"  Average EITC per return: ${results[name]['avg_eitc_per_return']:,.0f}")
        
        return results
    
    def save_results(self, results: Dict, filename: str) -> None:
        """Save analysis results to a JSON file."""
        output_path = REPORTS_DIR / filename
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")
    
    def plot_credit_distribution(self, df: pd.DataFrame, credit_col: str, title: str, filename: str) -> None:
        """Create and save a distribution plot for a credit amount."""
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=credit_col, bins=50, kde=True)
        plt.title(title)
        plt.xlabel('Credit Amount ($)')
        plt.ylabel('Frequency')
        
        # Save the plot
        plot_path = REPORTS_DIR / filename
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {plot_path}")

def main():
    """Main function to analyze tax credits."""
    print("Starting tax credit analysis...\n")
    
    # Initialize analyzer
    analyzer = TaxCreditAnalyzer(year=2020)
    
    # Load processed data
    analyzer.load_data()
    
    if not analyzer.data:
        print("No data found. Please run process_soi_data.py first.")
        return
    
    # Analyze tax credits
    print("\n" + "="*50)
    print("ANALYZING TAX CREDITS")
    print("="*50)
    
    ctc_results = analyzer.analyze_ctc()
    eitc_results = analyzer.analyze_eitc()
    
    # Save results
    analyzer.save_results({
        'ctc_analysis': ctc_results,
        'eitc_analysis': eitc_results
    }, 'tax_credit_analysis.json')
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
