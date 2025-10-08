#!/usr/bin/env python3
"""
Generate reports and visualizations for tax credit analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from typing import Dict, List, Optional

# Project directories
PROJECT_DIR = Path(__file__).parent.parent
REPORTS_DIR = PROJECT_DIR / 'reports'

# Create reports directory if it doesn't exist
REPORTS_DIR.mkdir(exist_ok=True)

class ReportGenerator:
    """Generate reports and visualizations for tax credit analysis."""
    
    def __init__(self, results_file: str = 'tax_credit_analysis.json'):
        """Initialize with path to analysis results."""
        self.results_file = REPORTS_DIR / results_file
        self.results = {}
        
    def load_results(self) -> bool:
        """Load analysis results from JSON file."""
        try:
            with open(self.results_file, 'r') as f:
                self.results = json.load(f)
            return True
        except FileNotFoundError:
            print(f"Error: Results file not found at {self.results_file}")
            print("Please run analyze_credits.py first.")
            return False
    
    def generate_summary_report(self) -> str:
        """Generate a markdown summary report."""
        if not self.results:
            return "No results to report."
            
        report = "# Tax Credit Analysis Report\n\n"
        
        # CTC Summary
        if 'ctc_analysis' in self.results:
            report += "## Child Tax Credit (CTC) Summary\n\n"
            ctc_data = self.results['ctc_analysis']
            
            for source, data in ctc_data.items():
                report += f"### {source.upper()}\n"
                report += f"- **Total CTC Amount**: ${data['total_ctc']:,.0f}\n"
                report += f"- **Total Returns**: {data['total_returns']:,.0f}\n"
                report += f"- **Average CTC per Return**: ${data['avg_ctc_per_return']:,.0f}\n"
                report += f"- **Total Returns with CTC**: {data['total_returns_with_ctc']:,.0f}\n"
                report += f"- **Average CTC per Claim**: ${data['avg_ctc_per_claim']:,.0f}\n"
                report += f"- **Refundable Percentage**: {data['refundable_pct']:.1f}%\n\n"
        
        # EITC Summary
        if 'eitc_analysis' in self.results:
            report += "## Earned Income Tax Credit (EITC) Summary\n\n"
            eitc_data = self.results['eitc_analysis']
            
            for source, data in eitc_data.items():
                report += f"### {source.upper()}\n"
                report += f"- **Total EITC Amount**: ${data['total_eitc']:,.0f}\n"
                report += f"- **Total Returns**: {data['total_returns']:,.0f}\n"
                if data['total_returns_with_eitc'] is not None:
                    report += f"- **Returns with EITC**: {data['total_returns_with_eitc']:,.0f}\n"
                if data['total_children'] is not None:
                    report += f"- **Total Qualifying Children**: {data['total_children']:,.0f}\n"
                report += f"- **Average EITC per Return**: ${data['avg_eitc_per_return']:,.0f}\n\n"
        
        # Save the report
        report_path = REPORTS_DIR / 'tax_credit_summary.md'
        with open(report_path, 'w') as f:
            f.write(report)
            
        print(f"Report generated: {report_path}")
        return str(report_path)
    
    def create_visualizations(self) -> None:
        """Create visualizations for the analysis."""
        if not self.results:
            return
            
        # Example visualization: CTC vs EITC by source
        if 'ctc_analysis' in self.results and 'eitc_analysis' in self.results:
            sources = list(self.results['ctc_analysis'].keys())
            ctc_totals = [self.results['ctc_analysis'][s]['total_ctc'] / 1e6 for s in sources]
            eitc_totals = [self.results['eitc_analysis'].get(s, {}).get('total_eitc', 0) / 1e6 for s in sources]
            
            plt.figure(figsize=(12, 6))
            
            x = range(len(sources))
            width = 0.35
            
            plt.bar([i - width/2 for i in x], ctc_totals, width, label='CTC (Millions $)')
            plt.bar([i + width/2 for i in x], eitc_totals, width, label='EITC (Millions $)')
            
            plt.xlabel('Data Source')
            plt.ylabel('Total Amount (Millions $)')
            plt.title('Total CTC vs EITC by Data Source')
            plt.xticks(x, [s.upper() for s in sources])
            plt.legend()
            
            # Save the plot
            plot_path = REPORTS_DIR / 'ctc_vs_eitc_comparison.png'
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            print(f"Visualization saved: {plot_path}")

def main():
    """Main function to generate reports."""
    print("Generating tax credit reports...\n")
    
    # Initialize report generator
    generator = ReportGenerator()
    
    # Load results
    if not generator.load_results():
        return
    
    # Generate reports
    print("Creating summary report...")
    report_path = generator.generate_summary_report()
    
    print("\nGenerating visualizations...")
    generator.create_visualizations()
    
    print("\nReport generation complete!")
    print(f"Reports saved to: {REPORTS_DIR}")

if __name__ == "__main__":
    main()
