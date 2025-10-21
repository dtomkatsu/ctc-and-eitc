"""
Parse Hawaii SOI deduction and exemption tables.

Tables:
- A4-2: Itemized vs Standard Deductions by AGI
- A5: Personal Exemptions by AGI
"""

import pandas as pd
import numpy as np
from pathlib import Path


def clean_currency(value):
    """Clean currency values (remove $, commas, handle negatives)."""
    if pd.isna(value) or value == '' or value == '-':
        return np.nan
    
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


def parse_deduction_benchmarks():
    """
    Parse Table A4-2 for deduction benchmarks.
    
    Returns:
        DataFrame with columns:
        - agi_min, agi_max
        - total_returns
        - itemized_returns, itemized_amount_millions
        - standard_returns, standard_amount_millions
        - avg_itemized, avg_standard
        - itemize_pct
    """
    a4_2_path = Path('data/raw/Selected Resident Return Data/Dotax Soi 2022 - A4-2.csv')
    df = pd.read_csv(a4_2_path, header=None)
    
    print("Parsing Table A4-2 (Deductions)...")
    
    records = []
    
    # Define AGI brackets (taxable returns only - rows 7-18 in 0-indexed)
    # Format: (agi_min, agi_max, row_idx)
    brackets = [
        (0, 10000, 7),
        (10000, 20000, 8),
        (20000, 30000, 9),
        (30000, 40000, 10),
        (40000, 50000, 11),
        (50000, 75000, 12),
        (75000, 100000, 13),
        (100000, 150000, 14),
        (150000, 200000, 15),
        (200000, 300000, 16),
        (300000, 400000, 17),
        (400000, 999999999, 18),
    ]
    
    for agi_min, agi_max, row_idx in brackets:
        try:
            # Column indices from A4-2
            # Columns: ..., Itemized (Number, Amount), Standard (Number, Amount), Total (Number, Amount)
            itemized_returns = clean_currency(df.iloc[row_idx, 8])
            itemized_amount_m = clean_currency(df.iloc[row_idx, 10])
            standard_returns = clean_currency(df.iloc[row_idx, 12])
            standard_amount_m = clean_currency(df.iloc[row_idx, 14])
            
            if pd.notna(itemized_returns) or pd.notna(standard_returns):
                # Calculate totals and averages
                itemized_returns = itemized_returns if pd.notna(itemized_returns) else 0
                standard_returns = standard_returns if pd.notna(standard_returns) else 0
                itemized_amount_m = itemized_amount_m if pd.notna(itemized_amount_m) else 0
                standard_amount_m = standard_amount_m if pd.notna(standard_amount_m) else 0
                
                total_returns = itemized_returns + standard_returns
                
                # Calculate averages
                avg_itemized = (itemized_amount_m * 1e6 / itemized_returns) if itemized_returns > 0 else 0
                avg_standard = (standard_amount_m * 1e6 / standard_returns) if standard_returns > 0 else 0
                
                # Calculate itemization propensity
                itemize_pct = (itemized_returns / total_returns * 100) if total_returns > 0 else 0
                
                records.append({
                    'agi_min': agi_min,
                    'agi_max': agi_max,
                    'total_returns': total_returns,
                    'itemized_returns': itemized_returns,
                    'itemized_amount_millions': itemized_amount_m,
                    'standard_returns': standard_returns,
                    'standard_amount_millions': standard_amount_m,
                    'avg_itemized': avg_itemized,
                    'avg_standard': avg_standard,
                    'itemize_pct': itemize_pct
                })
        except Exception as e:
            print(f"Warning: Error parsing row {row_idx}: {e}")
            continue
    
    benchmarks = pd.DataFrame(records)
    
    print(f"  Parsed {len(benchmarks)} deduction brackets")
    print(f"  Total returns: {benchmarks['total_returns'].sum():,.0f}")
    print(f"  Total itemized: ${benchmarks['itemized_amount_millions'].sum():,.0f}M")
    print(f"  Total standard: ${benchmarks['standard_amount_millions'].sum():,.0f}M")
    
    return benchmarks


def parse_exemption_benchmarks():
    """
    Parse Table A5 for personal exemption benchmarks.
    
    Returns:
        DataFrame with columns:
        - agi_min, agi_max
        - total_returns
        - total_exemptions (regular + dependent + age)
        - avg_exemptions_per_return
        - exemption_amount_millions
        - avg_exemption_per_return
        - exemption_value (per exemption)
    """
    a5_path = Path('data/raw/Selected Resident Return Data/Dotax Soi 2022 - A5.csv')
    df = pd.read_csv(a5_path, header=None)
    
    print("\nParsing Table A5 (Exemptions)...")
    
    records = []
    
    # Define AGI brackets (taxable returns only - rows 8-19 in 0-indexed)
    brackets = [
        (0, 10000, 8),
        (10000, 20000, 9),
        (20000, 30000, 10),
        (30000, 40000, 11),
        (40000, 50000, 12),
        (50000, 75000, 13),
        (75000, 100000, 14),
        (100000, 150000, 15),
        (150000, 200000, 16),
        (200000, 300000, 17),
        (300000, 400000, 18),
        (400000, 999999999, 19),
    ]
    
    for agi_min, agi_max, row_idx in brackets:
        try:
            # Column indices from A5
            total_returns = clean_currency(df.iloc[row_idx, 4])
            regular_exemptions = clean_currency(df.iloc[row_idx, 6])
            dependent_exemptions = clean_currency(df.iloc[row_idx, 8])
            age_exemptions = clean_currency(df.iloc[row_idx, 10])
            total_exemptions_count = clean_currency(df.iloc[row_idx, 12])
            exemption_amount_m = clean_currency(df.iloc[row_idx, 22])
            
            if pd.notna(total_returns) and total_returns > 0:
                # Calculate metrics
                total_exemptions = total_exemptions_count if pd.notna(total_exemptions_count) else 0
                exemption_amount_m = exemption_amount_m if pd.notna(exemption_amount_m) else 0
                
                avg_exemptions_per_return = total_exemptions / total_returns if total_returns > 0 else 0
                avg_exemption_amount_per_return = (exemption_amount_m * 1e6 / total_returns) if total_returns > 0 else 0
                exemption_value = (exemption_amount_m * 1e6 / total_exemptions) if total_exemptions > 0 else 0
                
                records.append({
                    'agi_min': agi_min,
                    'agi_max': agi_max,
                    'total_returns': total_returns,
                    'regular_exemptions': regular_exemptions if pd.notna(regular_exemptions) else 0,
                    'dependent_exemptions': dependent_exemptions if pd.notna(dependent_exemptions) else 0,
                    'age_exemptions': age_exemptions if pd.notna(age_exemptions) else 0,
                    'total_exemptions': total_exemptions,
                    'avg_exemptions_per_return': avg_exemptions_per_return,
                    'exemption_amount_millions': exemption_amount_m,
                    'avg_exemption_amount_per_return': avg_exemption_amount_per_return,
                    'exemption_value': exemption_value
                })
        except Exception as e:
            print(f"Warning: Error parsing row {row_idx}: {e}")
            continue
    
    benchmarks = pd.DataFrame(records)
    
    print(f"  Parsed {len(benchmarks)} exemption brackets")
    print(f"  Total returns: {benchmarks['total_returns'].sum():,.0f}")
    print(f"  Total exemptions: {benchmarks['total_exemptions'].sum():,.0f}")
    print(f"  Total exemption amount: ${benchmarks['exemption_amount_millions'].sum():,.0f}M")
    print(f"  Avg exemption value: ${benchmarks['exemption_value'].mean():,.0f}")
    
    return benchmarks


def save_benchmarks():
    """Parse and save both benchmark datasets."""
    # Parse benchmarks
    deduction_benchmarks = parse_deduction_benchmarks()
    exemption_benchmarks = parse_exemption_benchmarks()
    
    # Save to processed data
    deduction_path = Path('data/processed/deduction_benchmarks.csv')
    exemption_path = Path('data/processed/exemption_benchmarks.csv')
    
    deduction_benchmarks.to_csv(deduction_path, index=False)
    exemption_benchmarks.to_csv(exemption_path, index=False)
    
    print(f"\n✅ Saved deduction benchmarks to: {deduction_path}")
    print(f"✅ Saved exemption benchmarks to: {exemption_path}")
    
    return deduction_benchmarks, exemption_benchmarks


if __name__ == '__main__':
    save_benchmarks()
