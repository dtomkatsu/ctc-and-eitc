#!/usr/bin/env python3
"""
Validate SOI Comparison with Taxable Income

Compare our tax units to DOTAX SOI 2022 income brackets by filing status.
Uses taxable income (after standard deduction) to match SOI definitions.

SOI Tables:
- Table 13A: Single and Married Filing Separately
- Table 13B: Married Filing Jointly
- Table 13C: Head of Household
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# Configure plotting
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def load_soi_data() -> Dict[str, pd.DataFrame]:
    """Load DOTAX SOI 2022 data for all filing statuses."""
    
    soi_data = {}
    
    # Table 13A: Single and MFS
    table_13a = pd.read_csv('data/raw/Dotax Soi 2022 - 13A.csv')
    
    # Parse Single filers
    single_data = []
    for _, row in table_13a.iterrows():
        income_range = str(row.iloc[0]).strip()
        if 'Under' in income_range or '$' in income_range or 'and over' in income_range:
            try:
                num_returns = float(str(row.iloc[1]).replace(',', ''))
                single_data.append({'income_range': income_range, 'returns': num_returns})
            except:
                pass
    
    soi_data['single'] = pd.DataFrame(single_data)
    
    # Parse MFS filers (usually in same table)
    # Note: May need to adjust based on actual table structure
    
    # Table 13B: Married Filing Jointly
    table_13b = pd.read_csv('data/raw/Dotax Soi 2022 - 13B.csv')
    
    mfj_data = []
    for _, row in table_13b.iterrows():
        income_range = str(row.iloc[0]).strip()
        if 'Under' in income_range or '$' in income_range or 'and over' in income_range:
            try:
                num_returns = float(str(row.iloc[1]).replace(',', ''))
                mfj_data.append({'income_range': income_range, 'returns': num_returns})
            except:
                pass
    
    soi_data['married_filing_jointly'] = pd.DataFrame(mfj_data)
    
    # Table 13C: Head of Household
    table_13c = pd.read_csv('data/raw/Dotax Soi 2022 - 13C.csv')
    
    hoh_data = []
    for _, row in table_13c.iterrows():
        income_range = str(row.iloc[0]).strip()
        if 'Under' in income_range or '$' in income_range or 'and over' in income_range:
            try:
                num_returns = float(str(row.iloc[1]).replace(',', ''))
                hoh_data.append({'income_range': income_range, 'returns': num_returns})
            except:
                pass
    
    soi_data['head_of_household'] = pd.DataFrame(hoh_data)
    
    return soi_data


def parse_income_range(range_str: str) -> Tuple[float, float]:
    """Parse income range string to (min, max) tuple."""
    range_str = range_str.strip()
    
    # Handle "Under $X"
    if 'Under' in range_str:
        max_val = float(range_str.split('$')[1].replace(',', ''))
        return (0, max_val)
    
    # Handle "$X and over"
    if 'and over' in range_str or 'or more' in range_str:
        min_val = float(range_str.split('$')[1].split()[0].replace(',', ''))
        return (min_val, float('inf'))
    
    # Handle "$X - $Y" or "$X under $Y"
    if '-' in range_str or 'under' in range_str:
        parts = range_str.replace('under', '-').split('-')
        min_val = float(parts[0].strip().replace('$', '').replace(',', ''))
        max_val = float(parts[1].strip().replace('$', '').replace(',', ''))
        return (min_val, max_val)
    
    # Default
    return (0, 0)


def create_soi_brackets() -> Dict[str, List[Tuple[float, float, str]]]:
    """Create standardized income brackets for each filing status."""
    
    brackets = {}
    
    # Single and MFS brackets
    brackets['single'] = [
        (0, 2400, 'Under $2,400'),
        (2400, 4800, '$2,400 - $4,800'),
        (4800, 7200, '$4,800 - $7,200'),
        (7200, 9600, '$7,200 - $9,600'),
        (9600, 12000, '$9,600 - $12,000'),
        (12000, 14400, '$12,000 - $14,400'),
        (14400, 16800, '$14,400 - $16,800'),
        (16800, 19200, '$16,800 - $19,200'),
        (19200, 24000, '$19,200 - $24,000'),
        (24000, 28800, '$24,000 - $28,800'),
        (28800, 33600, '$28,800 - $33,600'),
        (33600, 38400, '$33,600 - $38,400'),
        (38400, 48000, '$38,400 - $48,000'),
        (48000, 60000, '$48,000 - $60,000'),
        (60000, 72000, '$60,000 - $72,000'),
        (72000, 84000, '$72,000 - $84,000'),
        (84000, 96000, '$84,000 - $96,000'),
        (96000, 120000, '$96,000 - $120,000'),
        (120000, 180000, '$120,000 - $180,000'),
        (180000, 240000, '$180,000 - $240,000'),
        (240000, float('inf'), '$240,000 and over')
    ]
    
    # MFJ brackets (different thresholds)
    brackets['married_filing_jointly'] = [
        (0, 4800, 'Under $4,800'),
        (4800, 9600, '$4,800 - $9,600'),
        (9600, 14400, '$9,600 - $14,400'),
        (14400, 19200, '$14,400 - $19,200'),
        (19200, 24000, '$19,200 - $24,000'),
        (24000, 28800, '$24,000 - $28,800'),
        (28800, 33600, '$28,800 - $33,600'),
        (33600, 38400, '$33,600 - $38,400'),
        (38400, 48000, '$38,400 - $48,000'),
        (48000, 60000, '$48,000 - $60,000'),
        (60000, 72000, '$60,000 - $72,000'),
        (72000, 84000, '$72,000 - $84,000'),
        (84000, 96000, '$84,000 - $96,000'),
        (96000, 120000, '$96,000 - $120,000'),
        (120000, 180000, '$120,000 - $180,000'),
        (180000, 240000, '$180,000 - $240,000'),
        (240000, 360000, '$240,000 - $360,000'),
        (360000, 480000, '$360,000 - $480,000'),
        (480000, float('inf'), '$480,000 and over')
    ]
    
    # HoH brackets
    brackets['head_of_household'] = [
        (0, 3600, 'Under $3,600'),
        (3600, 7200, '$3,600 - $7,200'),
        (7200, 10800, '$7,200 - $10,800'),
        (10800, 14400, '$10,800 - $14,400'),
        (14400, 18000, '$14,400 - $18,000'),
        (18000, 21600, '$18,000 - $21,600'),
        (21600, 25200, '$21,600 - $25,200'),
        (25200, 28800, '$25,200 - $28,800'),
        (28800, 36000, '$28,800 - $36,000'),
        (36000, 45000, '$36,000 - $45,000'),
        (45000, 54000, '$45,000 - $54,000'),
        (54000, 63000, '$54,000 - $63,000'),
        (63000, 72000, '$63,000 - $72,000'),
        (72000, 90000, '$72,000 - $90,000'),
        (90000, 135000, '$90,000 - $135,000'),
        (135000, 180000, '$135,000 - $180,000'),
        (180000, float('inf'), '$180,000 and over')
    ]
    
    brackets['married_filing_separately'] = brackets['single']  # Same as single
    
    return brackets


def calculate_taxable_income(tax_units: pd.DataFrame) -> pd.DataFrame:
    """Calculate taxable income (total income - standard deduction)."""
    
    # Standard deductions for 2022
    standard_deductions = {
        'single': 12950,
        'married_filing_jointly': 25900,
        'married_filing_separately': 12950,
        'head_of_household': 19400
    }
    
    tax_units = tax_units.copy()
    
    # Apply standard deduction
    tax_units['standard_deduction'] = tax_units['filing_status'].map(standard_deductions)
    tax_units['taxable_income'] = (tax_units['income'] - tax_units['standard_deduction']).clip(lower=0)
    
    return tax_units


def map_to_brackets(tax_units: pd.DataFrame, brackets: Dict[str, List[Tuple[float, float, str]]]) -> pd.DataFrame:
    """Map tax units to SOI income brackets."""
    
    results = []
    
    for status in ['single', 'married_filing_jointly', 'head_of_household', 'married_filing_separately']:
        status_units = tax_units[tax_units['filing_status'] == status].copy()
        
        if len(status_units) == 0:
            continue
        
        status_brackets = brackets[status]
        
        for min_inc, max_inc, label in status_brackets:
            if max_inc == float('inf'):
                in_bracket = status_units[status_units['taxable_income'] >= min_inc]
            else:
                in_bracket = status_units[
                    (status_units['taxable_income'] >= min_inc) & 
                    (status_units['taxable_income'] < max_inc)
                ]
            
            count = in_bracket['weight'].sum()
            
            results.append({
                'filing_status': status,
                'income_range': label,
                'min_income': min_inc,
                'max_income': max_inc,
                'returns': count
            })
    
    return pd.DataFrame(results)


def compare_to_soi(model_data: pd.DataFrame, soi_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compare model data to SOI benchmarks."""
    
    comparisons = []
    
    for status in ['single', 'married_filing_jointly', 'head_of_household']:
        model_status = model_data[model_data['filing_status'] == status]
        
        if status not in soi_data or len(soi_data[status]) == 0:
            continue
        
        soi_status = soi_data[status]
        
        # Merge on income range
        for _, model_row in model_status.iterrows():
            income_range = model_row['income_range']
            model_returns = model_row['returns']
            
            # Find matching SOI row
            soi_match = soi_status[soi_status['income_range'].str.contains(income_range.replace('$', '\\$'), regex=True, na=False)]
            
            if len(soi_match) == 0:
                # Try fuzzy match
                for _, soi_row in soi_status.iterrows():
                    if income_range.replace(',', '') in soi_row['income_range'].replace(',', ''):
                        soi_returns = soi_row['returns']
                        break
                else:
                    soi_returns = 0
            else:
                soi_returns = soi_match.iloc[0]['returns']
            
            diff = model_returns - soi_returns
            pct_diff = (diff / soi_returns * 100) if soi_returns > 0 else 0
            
            comparisons.append({
                'filing_status': status,
                'income_range': income_range,
                'soi_returns': soi_returns,
                'model_returns': model_returns,
                'difference': diff,
                'pct_difference': pct_diff
            })
    
    return pd.DataFrame(comparisons)


def create_comparison_plots(comparison_df: pd.DataFrame, output_dir: Path):
    """Create visualization plots."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for status in comparison_df['filing_status'].unique():
        status_data = comparison_df[comparison_df['filing_status'] == status].copy()
        
        if len(status_data) == 0:
            continue
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Absolute comparison
        x = range(len(status_data))
        width = 0.35
        
        axes[0].bar([i - width/2 for i in x], status_data['soi_returns'], width, label='SOI', alpha=0.8)
        axes[0].bar([i + width/2 for i in x], status_data['model_returns'], width, label='Model', alpha=0.8)
        
        axes[0].set_xlabel('Income Bracket')
        axes[0].set_ylabel('Number of Returns')
        axes[0].set_title(f'{status.replace("_", " ").title()} - Returns by Income Bracket')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(status_data['income_range'], rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Percentage difference
        colors = ['red' if x < -5 else 'orange' if x < 5 else 'green' for x in status_data['pct_difference']]
        axes[1].bar(x, status_data['pct_difference'], color=colors, alpha=0.7)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1].axhline(y=5, color='orange', linestyle='--', linewidth=0.5, alpha=0.5)
        axes[1].axhline(y=-5, color='orange', linestyle='--', linewidth=0.5, alpha=0.5)
        
        axes[1].set_xlabel('Income Bracket')
        axes[1].set_ylabel('Percentage Difference (%)')
        axes[1].set_title(f'{status.replace("_", " ").title()} - Percentage Difference from SOI')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(status_data['income_range'], rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f'{status}_income_bracket_comparison.png'
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {filename}")


def generate_summary_report(comparison_df: pd.DataFrame, output_file: Path):
    """Generate detailed summary report."""
    
    with open(output_file, 'w') as f:
        f.write("# SOI Income Bracket Validation Report\n\n")
        f.write(f"**Date**: 2025-10-15\n")
        f.write(f"**Using**: Taxable Income (after standard deduction)\n\n")
        
        f.write("---\n\n")
        
        for status in comparison_df['filing_status'].unique():
            status_data = comparison_df[comparison_df['filing_status'] == status]
            
            f.write(f"## {status.replace('_', ' ').title()}\n\n")
            
            # Summary statistics
            total_soi = status_data['soi_returns'].sum()
            total_model = status_data['model_returns'].sum()
            total_diff = total_model - total_soi
            total_pct = (total_diff / total_soi * 100) if total_soi > 0 else 0
            
            f.write(f"### Summary\n\n")
            f.write(f"- **SOI Total**: {total_soi:,.0f}\n")
            f.write(f"- **Model Total**: {total_model:,.0f}\n")
            f.write(f"- **Difference**: {total_diff:+,.0f} ({total_pct:+.1f}%)\n\n")
            
            # Brackets within ±5%
            within_5pct = len(status_data[status_data['pct_difference'].abs() <= 5])
            total_brackets = len(status_data)
            pct_within = (within_5pct / total_brackets * 100) if total_brackets > 0 else 0
            
            f.write(f"- **Brackets within ±5%**: {within_5pct} / {total_brackets} ({pct_within:.1f}%)\n\n")
            
            # Detailed table
            f.write(f"### Detailed Comparison\n\n")
            f.write(f"| Income Range | SOI | Model | Difference | % Diff | Status |\n")
            f.write(f"|--------------|-----|-------|------------|--------|--------|\n")
            
            for _, row in status_data.iterrows():
                status_icon = "✅" if abs(row['pct_difference']) <= 5 else "⚠️" if abs(row['pct_difference']) <= 10 else "❌"
                f.write(f"| {row['income_range']} | {row['soi_returns']:,.0f} | {row['model_returns']:,.0f} | "
                       f"{row['difference']:+,.0f} | {row['pct_difference']:+.1f}% | {status_icon} |\n")
            
            f.write(f"\n---\n\n")
        
        # Overall summary
        f.write(f"## Overall Assessment\n\n")
        
        total_brackets = len(comparison_df)
        within_5 = len(comparison_df[comparison_df['pct_difference'].abs() <= 5])
        within_10 = len(comparison_df[comparison_df['pct_difference'].abs() <= 10])
        
        f.write(f"- **Total brackets analyzed**: {total_brackets}\n")
        f.write(f"- **Within ±5%**: {within_5} ({within_5/total_brackets*100:.1f}%)\n")
        f.write(f"- **Within ±10%**: {within_10} ({within_10/total_brackets*100:.1f}%)\n\n")
        
        if within_5 / total_brackets >= 0.8:
            f.write(f"✅ **EXCELLENT** - 80%+ of brackets within ±5%\n")
        elif within_10 / total_brackets >= 0.8:
            f.write(f"✅ **GOOD** - 80%+ of brackets within ±10%\n")
        else:
            f.write(f"⚠️ **NEEDS IMPROVEMENT** - Less than 80% of brackets within ±10%\n")


def main():
    """Main execution."""
    print("="*80)
    print("SOI INCOME BRACKET VALIDATION")
    print("="*80)
    print("\nUsing taxable income (after standard deduction) to match SOI definitions\n")
    
    # Load tax units
    tax_units_file = Path('data/processed/tax_units_final_20251015_102701.parquet')
    
    if not tax_units_file.exists():
        print(f"ERROR: Tax units file not found: {tax_units_file}")
        print("Please run the tax unit generation first.")
        return
    
    print(f"Loading: {tax_units_file.name}")
    tax_units = pd.read_parquet(tax_units_file)
    
    print(f"  Total tax units: {len(tax_units):,}")
    print(f"  Weighted total: {tax_units['weight'].sum():,.0f}")
    
    # Calculate taxable income
    print("\nCalculating taxable income...")
    tax_units = calculate_taxable_income(tax_units)
    
    print(f"  Average total income: ${tax_units['income'].mean():,.0f}")
    print(f"  Average taxable income: ${tax_units['taxable_income'].mean():,.0f}")
    print(f"  Average reduction: ${(tax_units['income'] - tax_units['taxable_income']).mean():,.0f}")
    
    # Create brackets
    print("\nMapping to SOI income brackets...")
    brackets = create_soi_brackets()
    model_data = map_to_brackets(tax_units, brackets)
    
    # Load SOI data
    print("\nLoading SOI benchmark data...")
    try:
        soi_data = load_soi_data()
        print(f"  Loaded SOI data for {len(soi_data)} filing statuses")
    except Exception as e:
        print(f"  Warning: Could not load SOI data: {e}")
        print(f"  Will generate model distribution only")
        soi_data = {}
    
    # Compare
    comparison_successful = False
    if soi_data:
        print("\nComparing to SOI benchmarks...")
        comparison_df = compare_to_soi(model_data, soi_data)
        
        if len(comparison_df) == 0:
            print("  Warning: No comparisons generated. Will show model distribution only.")
        else:
            comparison_successful = True
            # Display summary
            print("\n" + "="*80)
            print("SUMMARY BY FILING STATUS")
            print("="*80)
            
            for status in comparison_df['filing_status'].unique():
                status_data = comparison_df[comparison_df['filing_status'] == status]
                
                total_soi = status_data['soi_returns'].sum()
                total_model = status_data['model_returns'].sum()
                diff = total_model - total_soi
                pct_diff = (diff / total_soi * 100) if total_soi > 0 else 0
                
                within_5 = len(status_data[status_data['pct_difference'].abs() <= 5])
                total_brackets = len(status_data)
                
                print(f"\n{status.replace('_', ' ').title()}:")
                print(f"  SOI Total:        {total_soi:>12,.0f}")
                print(f"  Model Total:      {total_model:>12,.0f}")
                print(f"  Difference:       {diff:>+12,.0f} ({pct_diff:>+6.1f}%)")
                print(f"  Brackets ±5%:     {within_5}/{total_brackets} ({within_5/total_brackets*100:.1f}%)")
            
            # Create visualizations
            print("\n\nGenerating visualizations...")
            output_dir = Path('analysis_results/soi_validation')
            create_comparison_plots(comparison_df, output_dir)
            
            # Generate report
            print("\nGenerating detailed report...")
            report_file = output_dir / 'SOI_VALIDATION_REPORT.md'
            generate_summary_report(comparison_df, report_file)
            print(f"  Saved: {report_file}")
    
    if not comparison_successful:
        print("\nGenerating model distribution only...")
        print("\nModel Distribution by Filing Status and Income Bracket:")
        
        for status in model_data['filing_status'].unique():
            status_data = model_data[model_data['filing_status'] == status]
            print(f"\n{status.replace('_', ' ').title()}:")
            print(f"{'Income Range':<30} {'Returns':>12}")
            print("-"*45)
            for _, row in status_data.iterrows():
                print(f"{row['income_range']:<30} {row['returns']:>12,.0f}")
            print(f"{'TOTAL':<30} {status_data['returns'].sum():>12,.0f}")
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
