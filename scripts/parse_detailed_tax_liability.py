#!/usr/bin/env python3
"""
Parse DOTAX SOI 2022 Table A-9 (Selected Resident Return Data)

This script parses the detailed tax liability data by AGI bracket and filing status.
Contains more detailed brackets than Table 12A and includes actual tax liability data.

Files:
- A9-2: Joint filers (Married Filing Jointly)
- A9-3: Single filers (includes Single and MFS)
- A9-4: Head of Household filers
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

def clean_currency(value):
    """Clean currency values (remove $, commas, handle negatives)."""
    if pd.isna(value) or value == '':
        return np.nan
    
    # Convert to string and clean
    value = str(value).strip()
    
    # Handle parentheses for negatives
    if value.startswith('(') and value.endswith(')'):
        value = '-' + value[1:-1]
    
    # Remove $, commas
    value = value.replace('$', '').replace(',', '')
    
    try:
        return float(value)
    except:
        return np.nan


def parse_agi_bracket(low_str, high_str):
    """Parse AGI bracket from text."""
    # Handle special cases
    if 'loss' in str(low_str).lower():
        return -999999999, 0
    if 'equal to' in str(low_str).lower():
        return 0, 0
    
    # Clean and convert
    low = clean_currency(low_str)
    
    # If high_str is NaN or None, use very large number
    if pd.isna(high_str) or high_str is None or str(high_str).strip() == '':
        high = 999999999
    else:
        high = clean_currency(high_str)
    
    return low, high


def parse_table(file_path, filing_status):
    """
    Parse one of the A-9 tables.
    
    Args:
        file_path: Path to CSV file
        filing_status: 'joint', 'single', or 'hoh'
        
    Returns:
        DataFrame with columns:
        - filing_status
        - agi_min
        - agi_max
        - returns (number of returns)
        - total_agi (millions)
        - tax_before_credits (millions)
        - tax_after_credits (millions)
        - avg_tax_before
        - avg_tax_after
        - rate_before (%)
        - rate_after (%)
    """
    df = pd.read_csv(file_path, header=None)
    
    records = []
    
    # Find data rows (skip header rows, look for actual data)
    for idx, row in df.iterrows():
        # Skip header rows and summary rows
        if idx < 7:  # Skip headers
            continue
        if 'TOTAL' in str(row[0]) or 'ALL' in str(row[0]) or 'Ratio' in str(row[0]):
            break  # Stop at summary rows
        
        try:
            # Parse AGI bracket
            # Format: [0]=low value, [1]="to under", [2]=high value
            # Special cases: [1]="Loss" or [0]=empty, [1]="Equal to", [2]="$0"
            
            # Check for special cases first
            if pd.notna(row[1]) and 'loss' in str(row[1]).lower():
                agi_min, agi_max = -999999999, 0
            elif pd.notna(row[1]) and 'equal to' in str(row[1]).lower():
                agi_min, agi_max = 0, 0
            else:
                # Standard bracket: [0]=low, [2]=high
                bracket_low = row[0]
                bracket_high = row[2] if pd.notna(row[2]) else None
                
                if pd.isna(bracket_low) or bracket_low == '':
                    continue
                
                agi_min, agi_max = parse_agi_bracket(str(bracket_low), str(bracket_high) if bracket_high else None)
            
            # Extract data (correct column indices with NaN columns)
            num_returns = clean_currency(row[4])
            total_agi = clean_currency(row[6])  # In millions
            tax_before = clean_currency(row[8])  # Total in millions
            avg_tax_before = clean_currency(row[10])
            tax_after = clean_currency(row[12])  # Total in millions
            avg_tax_after = clean_currency(row[14])
            rate_before = clean_currency(row[16])
            rate_after = clean_currency(row[18])
            
            if pd.notna(num_returns) and num_returns > 0:
                records.append({
                    'filing_status': filing_status,
                    'agi_min': agi_min,
                    'agi_max': agi_max,
                    'returns': num_returns,
                    'total_agi_millions': total_agi,
                    'tax_before_credits_millions': tax_before,
                    'tax_after_credits_millions': tax_after,
                    'avg_tax_before': avg_tax_before,
                    'avg_tax_after': avg_tax_after,
                    'effective_rate_before_pct': rate_before,
                    'effective_rate_after_pct': rate_after
                })
        except Exception as e:
            print(f"Warning: Could not parse row {idx}: {e}")
            continue
    
    return pd.DataFrame(records)


def main():
    """Parse all three tables and combine."""
    data_dir = Path(__file__).parent.parent / 'data' / 'raw' / 'Selected Resident Return Data'
    
    print("Parsing DOTAX SOI 2022 Table A-9 (Selected Resident Return Data)...\n")
    
    # Parse each filing status
    tables = {
        'joint': data_dir / 'Dotax Soi 2022 - A9-2.csv',
        'single': data_dir / 'Dotax Soi 2022 - A9-3.csv',
        'hoh': data_dir / 'Dotax Soi 2022 - A9-4.csv'
    }
    
    all_data = []
    
    for status, file_path in tables.items():
        print(f"Parsing {status} filers from {file_path.name}...")
        df = parse_table(file_path, status)
        print(f"  Found {len(df)} brackets with {df['returns'].sum():,.0f} total returns")
        print(f"  AGI range: ${df['agi_min'].min():,.0f} to ${df['agi_max'].max():,.0f}")
        print(f"  Total AGI: ${df['total_agi_millions'].sum():,.1f}M")
        print(f"  Total tax (after credits): ${df['tax_after_credits_millions'].sum():,.1f}M\n")
        all_data.append(df)
    
    # Combine all filing statuses
    combined = pd.concat(all_data, ignore_index=True)
    
    # Save to processed directory
    output_path = Path(__file__).parent.parent / 'data' / 'processed' / 'detailed_tax_liability_benchmarks.csv'
    combined.to_csv(output_path, index=False)
    
    print(f"\n✅ Saved combined data to: {output_path}")
    print(f"\nSummary:")
    print(f"  Total brackets: {len(combined)}")
    print(f"  Total returns: {combined['returns'].sum():,.0f}")
    print(f"  Total AGI: ${combined['total_agi_millions'].sum():,.1f}M")
    print(f"  Total tax (before credits): ${combined['tax_before_credits_millions'].sum():,.1f}M")
    print(f"  Total tax (after credits): ${combined['tax_after_credits_millions'].sum():,.1f}M")
    
    # Show sample
    print("\nSample data (first 10 rows):")
    print(combined.head(10).to_string())
    
    # Show by filing status
    print("\nBy filing status:")
    summary = combined.groupby('filing_status').agg({
        'returns': 'sum',
        'total_agi_millions': 'sum',
        'tax_after_credits_millions': 'sum'
    }).round(1)
    summary['avg_tax'] = (summary['tax_after_credits_millions'] * 1e6 / summary['returns']).round(0)
    print(summary)


if __name__ == '__main__':
    main()
