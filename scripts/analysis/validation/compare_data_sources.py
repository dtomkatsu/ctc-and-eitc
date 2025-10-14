"""
Compare DOTAX SOI, IRS SOI, and PUMS data to determine alignment and adjustment factors.

This script analyzes:
1. DOTAX SOI 2022 data (residents) - official Hawaii state tax data
2. IRS SOI data - federal tax data for Hawaii
3. PUMS-based tax units - constructed from Census PUMS data

Goal: Determine if PUMS data needs adjustment to align with DOTAX official numbers
for more accurate income tax revenue estimates.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from scripts.analyze_income_distribution import IncomeDistributionAnalyzer


class DataSourceComparator:
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        self.results = {}
        
    def load_dotax_soi_data(self) -> Dict:
        """Load and parse DOTAX SOI 2022 data for residents."""
        print("Loading DOTAX SOI 2022 data...")
        
        analyzer = IncomeDistributionAnalyzer(str(self.data_dir / "raw"))
        
        # Parse the three main tables
        status_files = [
            ("Single", "Dotax Soi 2022 - 13A.csv"),
            ("Married Filing Jointly", "Dotax Soi 2022 - 13B.csv"),
            ("Head of Household", "Dotax Soi 2022 - 13C.csv")
        ]
        
        dotax_data = {}
        total_returns = 0
        
        for status_name, filename in status_files:
            file_path = self.data_dir / "raw" / filename
            if file_path.exists():
                brackets = analyzer.parse_soi_table(file_path)
                returns = sum(b['returns'] for b in brackets)
                dotax_data[status_name] = {
                    'returns': returns,
                    'brackets': brackets
                }
                total_returns += returns
                print(f"  {status_name}: {returns:,} returns")
        
        # Note: 13A includes both Single and MFS, need to separate
        # For now, we'll treat 13A as primarily Single
        dotax_data['total_returns'] = total_returns
        dotax_data['year'] = 2022
        
        print(f"  Total DOTAX returns: {total_returns:,}\n")
        return dotax_data
    
    def load_irs_soi_data(self) -> Dict:
        """Load IRS SOI data for Hawaii."""
        print("Loading IRS SOI data...")
        
        # Try to load the processed SOI benchmarks
        soi_file = self.data_dir / "processed" / "soi_benchmarks_2022.csv"
        
        if soi_file.exists():
            df = pd.read_csv(soi_file)
            
            irs_data = {
                'year': int(df['year'].iloc[0]),
                'total_returns': int(df['total_returns'].iloc[0]),
                'single_returns': int(df['single_returns'].iloc[0]),
                'joint_returns': int(df['joint_returns'].iloc[0]),
                'hoh_returns': int(df['hoh_returns'].iloc[0]),
                'mfs_returns': int(df['mfs_returns'].iloc[0]),
                'total_agi': float(df['total_agi'].iloc[0]),
                'total_income': float(df['total_income'].iloc[0]),
                'avg_agi': float(df['avg_agi'].iloc[0])
            }
            
            print(f"  Year: {irs_data['year']}")
            print(f"  Total returns: {irs_data['total_returns']:,}")
            print(f"  Single: {irs_data['single_returns']:,} ({irs_data['single_returns']/irs_data['total_returns']*100:.1f}%)")
            print(f"  Joint: {irs_data['joint_returns']:,} ({irs_data['joint_returns']/irs_data['total_returns']*100:.1f}%)")
            print(f"  HoH: {irs_data['hoh_returns']:,} ({irs_data['hoh_returns']/irs_data['total_returns']*100:.1f}%)")
            print(f"  MFS: {irs_data['mfs_returns']:,} ({irs_data['mfs_returns']/irs_data['total_returns']*100:.1f}%)")
            print(f"  Total AGI: ${irs_data['total_agi']:,.0f}")
            print(f"  Average AGI: ${irs_data['avg_agi']:,.2f}\n")
            
            return irs_data
        else:
            print(f"  Warning: IRS SOI benchmarks file not found at {soi_file}\n")
            return {}
    
    def load_pums_tax_units(self) -> Dict:
        """Load PUMS-based tax units data."""
        print("Loading PUMS tax units data...")
        
        # Try to load the calibrated tax units
        pums_file = self.data_dir / "processed" / "tax_units_soi_calibrated.parquet"
        
        if pums_file.exists():
            df = pd.read_parquet(pums_file)
            
            # Calculate statistics
            total_units = len(df)
            total_weighted = df['weight'].sum()
            
            # Group by filing status
            status_counts = df.groupby('filing_status').agg({
                'weight': 'sum',
                'income': ['count', 'mean', 'sum']
            }).round(2)
            
            pums_data = {
                'total_units': total_units,
                'total_weighted': total_weighted,
                'status_distribution': status_counts,
                'avg_income': df['income'].mean(),
                'total_income': (df['income'] * df['weight']).sum()
            }
            
            print(f"  Total tax units (unweighted): {total_units:,}")
            print(f"  Total tax units (weighted): {total_weighted:,.0f}")
            print(f"  Average income: ${pums_data['avg_income']:,.2f}")
            print(f"  Total income (weighted): ${pums_data['total_income']:,.0f}")
            print(f"\n  Filing status distribution:")
            print(status_counts)
            print()
            
            return pums_data
        else:
            print(f"  Warning: PUMS tax units file not found at {pums_file}\n")
            return {}
    
    def compare_dotax_vs_irs(self, dotax_data: Dict, irs_data: Dict) -> Dict:
        """Compare DOTAX SOI with IRS SOI data."""
        print("="*80)
        print("COMPARISON: DOTAX SOI vs IRS SOI")
        print("="*80)
        
        if not dotax_data or not irs_data:
            print("Missing data for comparison\n")
            return {}
        
        comparison = {}
        
        # Compare total returns
        dotax_total = dotax_data['total_returns']
        irs_total = irs_data['total_returns']
        diff = dotax_total - irs_total
        pct_diff = (diff / irs_total) * 100
        
        print(f"\n1. Total Returns Comparison:")
        print(f"   DOTAX SOI:  {dotax_total:>10,} returns")
        print(f"   IRS SOI:    {irs_total:>10,} returns")
        print(f"   Difference: {diff:>10,} ({pct_diff:+.2f}%)")
        
        comparison['total_returns'] = {
            'dotax': dotax_total,
            'irs': irs_total,
            'difference': diff,
            'pct_difference': pct_diff
        }
        
        # Compare filing status distribution
        print(f"\n2. Filing Status Distribution:")
        print(f"   {'Status':<30} {'DOTAX':>12} {'IRS':>12} {'Difference':>12} {'% Diff':>10}")
        print(f"   {'-'*76}")
        
        # Map DOTAX to IRS categories
        dotax_single = dotax_data.get('Single', {}).get('returns', 0)
        dotax_joint = dotax_data.get('Married Filing Jointly', {}).get('returns', 0)
        dotax_hoh = dotax_data.get('Head of Household', {}).get('returns', 0)
        
        irs_single = irs_data['single_returns']
        irs_joint = irs_data['joint_returns']
        irs_hoh = irs_data['hoh_returns']
        irs_mfs = irs_data['mfs_returns']
        
        # Note: DOTAX 13A includes both Single and MFS
        # We need to estimate the split
        dotax_single_only = dotax_single - irs_mfs  # Rough estimate
        
        statuses = [
            ('Single', dotax_single_only, irs_single),
            ('Married Filing Jointly', dotax_joint, irs_joint),
            ('Head of Household', dotax_hoh, irs_hoh),
            ('Married Filing Separately', irs_mfs, irs_mfs)  # DOTAX doesn't separate
        ]
        
        comparison['filing_status'] = {}
        
        for status, dotax_val, irs_val in statuses:
            if irs_val > 0:
                diff = dotax_val - irs_val
                pct_diff = (diff / irs_val) * 100
                print(f"   {status:<30} {dotax_val:>12,} {irs_val:>12,} {diff:>12,} {pct_diff:>9.2f}%")
                
                comparison['filing_status'][status] = {
                    'dotax': dotax_val,
                    'irs': irs_val,
                    'difference': diff,
                    'pct_difference': pct_diff
                }
        
        print()
        return comparison
    
    def compare_dotax_vs_pums(self, dotax_data: Dict, pums_data: Dict) -> Dict:
        """Compare DOTAX SOI with PUMS-based tax units."""
        print("="*80)
        print("COMPARISON: DOTAX SOI vs PUMS Tax Units")
        print("="*80)
        
        if not dotax_data or not pums_data:
            print("Missing data for comparison\n")
            return {}
        
        comparison = {}
        
        # Compare total returns/units
        dotax_total = dotax_data['total_returns']
        pums_total = pums_data['total_weighted']
        diff = dotax_total - pums_total
        pct_diff = (diff / pums_total) * 100
        
        print(f"\n1. Total Tax Units Comparison:")
        print(f"   DOTAX SOI:  {dotax_total:>10,.0f} returns")
        print(f"   PUMS:       {pums_total:>10,.0f} tax units (weighted)")
        print(f"   Difference: {diff:>10,.0f} ({pct_diff:+.2f}%)")
        
        comparison['total_units'] = {
            'dotax': dotax_total,
            'pums': pums_total,
            'difference': diff,
            'pct_difference': pct_diff,
            'adjustment_factor': dotax_total / pums_total
        }
        
        print(f"   Adjustment factor needed: {comparison['total_units']['adjustment_factor']:.4f}")
        
        # Compare filing status distribution
        print(f"\n2. Filing Status Distribution:")
        
        status_dist = pums_data['status_distribution']
        
        print(f"   {'Status':<30} {'DOTAX':>12} {'PUMS':>12} {'Difference':>12} {'% Diff':>10}")
        print(f"   {'-'*76}")
        
        # Map statuses - need to handle case sensitivity
        status_mapping = {
            'Single': 'single',
            'Married Filing Jointly': 'married_filing_jointly',
            'Head of Household': 'head_of_household'
        }
        
        comparison['filing_status'] = {}
        
        for dotax_status, pums_status in status_mapping.items():
            dotax_val = dotax_data.get(dotax_status, {}).get('returns', 0)
            
            # Get PUMS value from the status distribution
            if pums_status in status_dist.index:
                pums_val = status_dist.loc[pums_status, ('weight', 'sum')]
            else:
                # Try case-insensitive match
                matching_status = [s for s in status_dist.index if s.lower().replace(' ', '_') == pums_status]
                if matching_status:
                    pums_val = status_dist.loc[matching_status[0], ('weight', 'sum')]
                else:
                    pums_val = 0
            
            if pums_val > 0:
                diff = dotax_val - pums_val
                pct_diff = (diff / pums_val) * 100
                adj_factor = dotax_val / pums_val
                
                print(f"   {dotax_status:<30} {dotax_val:>12,} {pums_val:>12,.0f} {diff:>12,.0f} {pct_diff:>9.2f}%")
                
                comparison['filing_status'][dotax_status] = {
                    'dotax': dotax_val,
                    'pums': pums_val,
                    'difference': diff,
                    'pct_difference': pct_diff,
                    'adjustment_factor': adj_factor
                }
        
        print()
        return comparison
    
    def calculate_adjustment_factors(self, comparisons: Dict) -> Dict:
        """Calculate recommended adjustment factors for PUMS data."""
        print("="*80)
        print("RECOMMENDED ADJUSTMENT FACTORS")
        print("="*80)
        
        if 'dotax_vs_pums' not in comparisons:
            print("No PUMS comparison data available\n")
            return {}
        
        pums_comp = comparisons['dotax_vs_pums']
        
        print(f"\n1. Overall Adjustment Factor:")
        overall_factor = pums_comp['total_units']['adjustment_factor']
        print(f"   Multiply all PUMS weights by: {overall_factor:.4f}")
        print(f"   This will align total tax units with DOTAX official count")
        
        print(f"\n2. Filing Status-Specific Adjustment Factors:")
        print(f"   {'Status':<30} {'Current PUMS':>15} {'Target (DOTAX)':>15} {'Factor':>10}")
        print(f"   {'-'*70}")
        
        adjustments = {}
        
        for status, data in pums_comp.get('filing_status', {}).items():
            factor = data['adjustment_factor']
            print(f"   {status:<30} {data['pums']:>15,.0f} {data['dotax']:>15,} {factor:>10.4f}")
            adjustments[status] = factor
        
        print(f"\n3. Recommendation:")
        print(f"   Apply the overall adjustment factor of {overall_factor:.4f} to all PUMS weights")
        print(f"   to align with DOTAX total returns count.")
        print(f"   ")
        print(f"   If filing status-specific adjustments are needed, use the factors above.")
        print(f"   However, this may indicate issues with filing status determination logic")
        print(f"   in the tax unit construction process.")
        
        print()
        return adjustments
    
    def compare_income_distributions(self, dotax_data: Dict, irs_data: Dict, pums_data: Dict) -> Dict:
        """Compare income distributions across all three data sources."""
        print("="*80)
        print("INCOME DISTRIBUTION COMPARISON")
        print("="*80)
        
        comparison = {}
        
        # Compare average income
        print(f"\n1. Average Income Comparison:")
        
        if irs_data:
            irs_avg = irs_data['avg_agi']
            print(f"   IRS SOI Average AGI:  ${irs_avg:>12,.2f}")
        
        if pums_data:
            pums_avg = pums_data['avg_income']
            print(f"   PUMS Average Income:  ${pums_avg:>12,.2f}")
            
            if irs_data:
                diff = pums_avg - irs_avg
                pct_diff = (diff / irs_avg) * 100
                print(f"   Difference:           ${diff:>12,.2f} ({pct_diff:+.2f}%)")
                
                comparison['avg_income'] = {
                    'irs': irs_avg,
                    'pums': pums_avg,
                    'difference': diff,
                    'pct_difference': pct_diff
                }
        
        # Compare total income/AGI
        print(f"\n2. Total Income/AGI Comparison:")
        
        if irs_data:
            irs_total = irs_data['total_agi']
            print(f"   IRS SOI Total AGI:    ${irs_total:>15,.0f}")
        
        if pums_data:
            pums_total = pums_data['total_income']
            print(f"   PUMS Total Income:    ${pums_total:>15,.0f}")
            
            if irs_data:
                diff = pums_total - irs_total
                pct_diff = (diff / irs_total) * 100
                print(f"   Difference:           ${diff:>15,.0f} ({pct_diff:+.2f}%)")
                
                comparison['total_income'] = {
                    'irs': irs_total,
                    'pums': pums_total,
                    'difference': diff,
                    'pct_difference': pct_diff
                }
        
        print()
        return comparison
    
    def generate_report(self, output_dir: str = "analysis_results/data_comparison"):
        """Generate comprehensive comparison report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all data sources
        dotax_data = self.load_dotax_soi_data()
        irs_data = self.load_irs_soi_data()
        pums_data = self.load_pums_tax_units()
        
        # Perform comparisons
        comparisons = {}
        
        if dotax_data and irs_data:
            comparisons['dotax_vs_irs'] = self.compare_dotax_vs_irs(dotax_data, irs_data)
        
        if dotax_data and pums_data:
            comparisons['dotax_vs_pums'] = self.compare_dotax_vs_pums(dotax_data, pums_data)
        
        # Compare income distributions
        if irs_data and pums_data:
            comparisons['income_comparison'] = self.compare_income_distributions(dotax_data, irs_data, pums_data)
        
        # Calculate adjustment factors
        adjustments = self.calculate_adjustment_factors(comparisons)
        
        # Save results
        self._save_results(comparisons, adjustments, output_dir)
        
        return comparisons, adjustments
    
    def _save_results(self, comparisons: Dict, adjustments: Dict, output_dir: Path):
        """Save comparison results to CSV files."""
        
        # Save DOTAX vs IRS comparison
        if 'dotax_vs_irs' in comparisons:
            comp = comparisons['dotax_vs_irs']
            
            # Total returns comparison
            total_df = pd.DataFrame([comp['total_returns']])
            total_df.to_csv(output_dir / "dotax_vs_irs_total.csv", index=False)
            
            # Filing status comparison
            if 'filing_status' in comp:
                status_data = []
                for status, data in comp['filing_status'].items():
                    status_data.append({
                        'filing_status': status,
                        **data
                    })
                status_df = pd.DataFrame(status_data)
                status_df.to_csv(output_dir / "dotax_vs_irs_filing_status.csv", index=False)
        
        # Save DOTAX vs PUMS comparison
        if 'dotax_vs_pums' in comparisons:
            comp = comparisons['dotax_vs_pums']
            
            # Total units comparison
            total_df = pd.DataFrame([comp['total_units']])
            total_df.to_csv(output_dir / "dotax_vs_pums_total.csv", index=False)
            
            # Filing status comparison
            if 'filing_status' in comp:
                status_data = []
                for status, data in comp['filing_status'].items():
                    status_data.append({
                        'filing_status': status,
                        **data
                    })
                status_df = pd.DataFrame(status_data)
                status_df.to_csv(output_dir / "dotax_vs_pums_filing_status.csv", index=False)
        
        # Save adjustment factors
        if adjustments:
            adj_df = pd.DataFrame([
                {'filing_status': status, 'adjustment_factor': factor}
                for status, factor in adjustments.items()
            ])
            adj_df.to_csv(output_dir / "pums_adjustment_factors.csv", index=False)
        
        print(f"Results saved to: {output_dir}/")


def main():
    comparator = DataSourceComparator()
    comparisons, adjustments = comparator.generate_report()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nSummary:")
    print("- Compared DOTAX SOI 2022 (official Hawaii state tax data)")
    print("- Compared IRS SOI data (federal tax data for Hawaii)")
    print("- Compared PUMS-based tax units (constructed from Census data)")
    print("\nResults saved to: analysis_results/data_comparison/")


if __name__ == "__main__":
    main()
