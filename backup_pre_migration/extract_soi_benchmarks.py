"""
Extract IRS SOI benchmarks for Hawaii (2022 data)

This script extracts key metrics from the IRS Statistics of Income data
to use as benchmarks for calibrating our tax model.
"""

import pandas as pd
from pathlib import Path

def extract_soi_data():
    """Extract SOI data from the Excel file"""
    
    # Read the SOI Excel file
    soi_file = Path(__file__).parent.parent / 'data' / 'irs_soi' / '22zp12hi.xlsx'
    df = pd.read_excel(soi_file, header=None)
    
    # Row 6 contains statewide totals
    # Columns based on header rows 3-5
    total_row = df.iloc[6]
    
    # Extract key metrics
    soi_data = {
        'year': 2022,
        'total_returns': int(total_row[2]),  # Col 2: Number of returns
        'single_returns': int(total_row[3]),  # Col 3: Number of single returns
        'joint_returns': int(total_row[4]),   # Col 4: Number of joint returns
        'hoh_returns': int(total_row[5]),     # Col 5: Number of head of household returns
        'total_agi': int(total_row[18]) * 1000,  # Col 18: AGI in thousands
        'total_income': int(total_row[20]) * 1000,  # Col 20: Total income in thousands
    }
    
    # Calculate filing status percentages
    soi_data['single_pct'] = soi_data['single_returns'] / soi_data['total_returns'] * 100
    soi_data['joint_pct'] = soi_data['joint_returns'] / soi_data['total_returns'] * 100
    soi_data['hoh_pct'] = soi_data['hoh_returns'] / soi_data['total_returns'] * 100
    
    # Calculate MFS (remaining percentage)
    soi_data['mfs_returns'] = soi_data['total_returns'] - soi_data['single_returns'] - soi_data['joint_returns'] - soi_data['hoh_returns']
    soi_data['mfs_pct'] = soi_data['mfs_returns'] / soi_data['total_returns'] * 100
    
    # Average AGI per return
    soi_data['avg_agi'] = soi_data['total_agi'] / soi_data['total_returns']
    
    return soi_data


def extract_soi_by_income_bracket():
    """Extract SOI data by income bracket"""
    
    soi_file = Path(__file__).parent.parent / 'data' / 'irs_soi' / '22zp12hi.xlsx'
    df = pd.read_excel(soi_file, header=None)
    
    # Rows 7-12 contain income bracket data
    brackets = []
    bracket_rows = {
        7: '$1 under $25,000',
        8: '$25,000 under $50,000',
        9: '$50,000 under $75,000',
        10: '$75,000 under $100,000',
        11: '$100,000 under $200,000',
        12: '$200,000 or more'
    }
    
    for row_idx, bracket_name in bracket_rows.items():
        row = df.iloc[row_idx]
        brackets.append({
            'bracket': bracket_name,
            'total_returns': int(row[2]),
            'single_returns': int(row[3]),
            'joint_returns': int(row[4]),
            'hoh_returns': int(row[5]),
            'total_agi': int(row[18]) * 1000,
        })
    
    return pd.DataFrame(brackets)


def main():
    """Extract and display SOI benchmarks"""
    
    print("="*80)
    print("IRS SOI BENCHMARKS FOR HAWAII (Tax Year 2022)")
    print("="*80)
    
    # Extract overall data
    soi = extract_soi_data()
    
    print("\n## OVERALL STATISTICS")
    print("-"*80)
    print(f"Total Returns:        {soi['total_returns']:>15,}")
    print(f"Total AGI:            ${soi['total_agi']:>15,.0f}")
    print(f"Total Income:         ${soi['total_income']:>15,.0f}")
    print(f"Average AGI:          ${soi['avg_agi']:>15,.2f}")
    
    print("\n## FILING STATUS DISTRIBUTION")
    print("-"*80)
    print(f"Single:               {soi['single_returns']:>10,}  ({soi['single_pct']:>5.1f}%)")
    print(f"Joint:                {soi['joint_returns']:>10,}  ({soi['joint_pct']:>5.1f}%)")
    print(f"Head of Household:    {soi['hoh_returns']:>10,}  ({soi['hoh_pct']:>5.1f}%)")
    print(f"Married Filing Sep:   {soi['mfs_returns']:>10,}  ({soi['mfs_pct']:>5.1f}%)")
    
    # Extract by income bracket
    brackets_df = extract_soi_by_income_bracket()
    
    print("\n## BY INCOME BRACKET")
    print("-"*80)
    print(f"{'Bracket':<25} {'Returns':>12} {'Avg AGI':>15}")
    print("-"*80)
    for _, row in brackets_df.iterrows():
        avg_agi = row['total_agi'] / row['total_returns']
        print(f"{row['bracket']:<25} {row['total_returns']:>12,} ${avg_agi:>14,.0f}")
    
    # Save to CSV
    output_file = Path(__file__).parent.parent / 'data' / 'processed' / 'soi_benchmarks_2022.csv'
    
    # Create summary DataFrame
    summary_df = pd.DataFrame([soi])
    summary_df.to_csv(output_file, index=False)
    
    print(f"\n\nBenchmarks saved to: {output_file}")
    print("="*80)
    
    return soi, brackets_df


if __name__ == '__main__':
    soi_data, brackets_data = main()
