#!/usr/bin/env python3
"""
Parse DOTAX SOI 2022 Comprehensive Tax Data

Parses Table A-2 which provides complete coverage of all income levels.

Table A-2: Covers ALL income levels (100% of returns)
- Includes both taxable and nontaxable returns
- 16 brackets per filing status
- Includes returns and AGI data
- Includes tax liability (before/after credits)

Table A-9 provides more detailed brackets for AGI < $150k but is a subset of A-2.
For calibration, you can choose:
- A-2: Complete coverage (100%) with broader brackets
- A-9: More granular (15 brackets) but only covers 90% of returns
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

def clean_currency(value):
    """Clean currency values (remove $, commas, handle negatives)."""
    if pd.isna(value) or value == '':
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


def parse_a2_table():
    """
    Parse Table A-2 for complete income coverage.
    
    Returns:
        DataFrame with all income brackets including high-income.
    """
    a2_1_path = Path('data/raw/Selected Resident Return Data/Dotax Soi 2022 - A2-1.csv')
    a2_2_path = Path('data/raw/Selected Resident Return Data/Dotax Soi 2022 - A2-2.csv')
    
    # Load both parts
    df1 = pd.read_csv(a2_1_path, header=None)
    df2 = pd.read_csv(a2_2_path, header=None)
    
    print("Parsing Table A-2 (complete income coverage)...")
    
    records = []
    
    # Define brackets from the table
    # NOTE: Row index 19 is "TOTAL RESIDENT TAXABLE" - do NOT include it
    # Row indices are 0-based, so file row 8 = index 7, etc.
    brackets = [
        # TAXABLE RETURNS (file rows 8-19, indices 7-18)
        (0, 10000, 7),       # File row 8: $0 to under $10k
        (10000, 20000, 8),   # File row 9: $10k to under $20k
        (20000, 30000, 9),   # File row 10: $20k to under $30k
        (30000, 40000, 10),  # File row 11: $30k to under $40k
        (40000, 50000, 11),  # File row 12: $40k to under $50k
        (50000, 75000, 12),  # File row 13: $50k to under $75k
        (75000, 100000, 13), # File row 14: $75k to under $100k
        (100000, 150000, 14),# File row 15: $100k to under $150k
        (150000, 200000, 15),# File row 16: $150k to under $200k
        (200000, 300000, 16),# File row 17: $200k to under $300k
        (300000, 400000, 17),# File row 18: $300k to under $400k
        (400000, 999999999, 18),  # File row 19: $400k and over
        # Row index 19 is TOTAL - SKIP IT
        # NONTAXABLE RETURNS (file rows 23-26, indices 22-25)
        (-999999999, 0, 22), # File row 23: Loss
        (0, 5000, 23),       # File row 24: $0 to under $5k (nontaxable)
        (5000, 10000, 24),   # File row 25: $5k to under $10k (nontaxable)
        (10000, 999999999, 25),  # File row 26: $10k and over (nontaxable)
    ]
    
    for agi_min, agi_max, row_idx in brackets:
        # Extract data for joint, single, hoh
        try:
            # From A2-1: Number of returns and total AGI
            joint_returns = clean_currency(df1.iloc[row_idx, 4])
            single_returns = clean_currency(df1.iloc[row_idx, 8])
            hoh_returns = clean_currency(df1.iloc[row_idx, 12])
            
            # AGI columns: Joint=16, Single=18, HoH=20
            joint_agi = clean_currency(df1.iloc[row_idx, 16])
            single_agi = clean_currency(df1.iloc[row_idx, 18])
            hoh_agi = clean_currency(df1.iloc[row_idx, 20])
            
            # From A2-2: Tax liability
            if row_idx <= 18:  # Taxable returns (indices 7-18, not including 19 which is TOTAL)
                joint_tax_before = clean_currency(df2.iloc[row_idx, 8])
                single_tax_before = clean_currency(df2.iloc[row_idx, 9])
                hoh_tax_before = clean_currency(df2.iloc[row_idx, 10])
                
                joint_tax_after = clean_currency(df2.iloc[row_idx, 12])
                single_tax_after = clean_currency(df2.iloc[row_idx, 13])
                hoh_tax_after = clean_currency(df2.iloc[row_idx, 14])
            else:  # Nontaxable returns (indices 22-25) - only tax after credits available
                joint_tax_before = 0
                single_tax_before = 0
                hoh_tax_before = 0
                
                joint_tax_after = clean_currency(df2.iloc[row_idx, 12])
                single_tax_after = clean_currency(df2.iloc[row_idx, 13])
                hoh_tax_after = clean_currency(df2.iloc[row_idx, 14])
            
            # Add records for each filing status
            for status, returns, agi, tax_before, tax_after in [
                ('joint', joint_returns, joint_agi, joint_tax_before, joint_tax_after),
                ('single', single_returns, single_agi, single_tax_before, single_tax_after),
                ('hoh', hoh_returns, hoh_agi, hoh_tax_before, hoh_tax_after)
            ]:
                if pd.notna(returns) and returns > 0:
                    avg_tax_before = (tax_before * 1e6 / returns) if (pd.notna(tax_before) and returns > 0) else 0
                    avg_tax_after = (tax_after * 1e6 / returns) if (pd.notna(tax_after) and returns > 0) else 0
                    
                    records.append({
                        'filing_status': status,
                        'agi_min': agi_min,
                        'agi_max': agi_max,
                        'returns': returns,
                        'total_agi_millions': agi if pd.notna(agi) else 0,
                        'tax_before_credits_millions': tax_before if pd.notna(tax_before) else 0,
                        'tax_after_credits_millions': tax_after if pd.notna(tax_after) else 0,
                        'avg_tax_before': avg_tax_before,
                        'avg_tax_after': avg_tax_after
                    })
        except Exception as e:
            print(f"Warning: Error parsing row {row_idx}: {e}")
            continue
    
    df = pd.DataFrame(records)
    
    print(f"  Parsed {len(df)} brackets from Table A-2")
    print(f"  Total returns: {df['returns'].sum():,.0f}")
    print(f"  AGI range: ${df['agi_min'].min():,.0f} to ${df['agi_max'].max():,.0f}")
    
    return df


def combine_a2_and_a9():
    """
    Use Table A-2 for comprehensive coverage.
    
    Table A-2 already includes all returns, so we use it directly.
    Table A-9 provides more granular detail for AGI < $150k but is a subset of A-2.
    
    Returns:
        DataFrame with comprehensive benchmark data
    """
    # Parse A-2 data (all income levels, complete coverage)
    comprehensive_df = parse_a2_table()
    
    print(f"\nUsing Table A-2 for complete coverage:")
    print(f"  Note: Table A-9 provides more detail for AGI < $150k but is subset of A-2")
    
    # Sort by filing status and AGI
    comprehensive_df = comprehensive_df.sort_values(['filing_status', 'agi_min']).reset_index(drop=True)
    
    # Calculate effective rates
    comprehensive_df['effective_rate_before_pct'] = np.where(
        comprehensive_df['total_agi_millions'] > 0,
        (comprehensive_df['tax_before_credits_millions'] / comprehensive_df['total_agi_millions'] * 100),
        np.nan
    )
    comprehensive_df['effective_rate_after_pct'] = np.where(
        comprehensive_df['total_agi_millions'] > 0,
        (comprehensive_df['tax_after_credits_millions'] / comprehensive_df['total_agi_millions'] * 100),
        np.nan
    )
    
    print(f"\n✅ Comprehensive dataset (Table A-2):")
    print(f"  Total brackets: {len(comprehensive_df)}")
    print(f"  Total returns: {comprehensive_df['returns'].sum():,.0f}")
    print(f"  Total AGI: ${comprehensive_df['total_agi_millions'].sum():,.1f}M")
    print(f"  Total tax (after credits): ${comprehensive_df['tax_after_credits_millions'].sum():,.1f}M")
    print(f"  Coverage: 100% of all Hawaii resident returns")
    
    # Show breakdown by filing status
    print("\nBreakdown by filing status:")
    for status in ['joint', 'single', 'hoh']:
        status_df = comprehensive_df[comprehensive_df['filing_status'] == status]
        n_brackets = len(status_df)
        returns = status_df['returns'].sum()
        agi = status_df['total_agi_millions'].sum()
        tax = status_df['tax_after_credits_millions'].sum()
        
        print(f"  {status:<8} - {n_brackets} brackets, {returns:>7,.0f} returns, "
              f"AGI=${agi:>7,.1f}M, Tax=${tax:>6,.1f}M")
    
    return comprehensive_df


def main():
    """Parse comprehensive tax data from Table A-2."""
    print("="*80)
    print("PARSING COMPREHENSIVE TAX DATA (TABLE A-2)")
    print("="*80)
    
    # Parse A-2 data (covers all income levels)
    comprehensive_df = combine_a2_and_a9()  # Actually just parses A-2 now
    
    # Save comprehensive benchmarks
    output_path = Path('data/processed/comprehensive_tax_benchmarks.csv')
    comprehensive_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Saved comprehensive benchmarks to: {output_path}")
    
    # Show sample high-income brackets
    print("\nSample high-income brackets (AGI ≥ $150k):")
    high_income = comprehensive_df[comprehensive_df['agi_min'] >= 150000]
    print(high_income[['filing_status', 'agi_min', 'agi_max', 'returns', 
                       'avg_tax_after', 'effective_rate_after_pct']].to_string(index=False))
    
    # Validation check
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)
    
    # Compare to A-2 totals
    a2_total_returns = {
        'joint': 216358,
        'single': 351205,
        'hoh': 67554
    }
    
    print("\nTotal returns comparison (vs Table A-2):")
    for status in ['joint', 'single', 'hoh']:
        status_returns = comprehensive_df[comprehensive_df['filing_status'] == status]['returns'].sum()
        target = a2_total_returns[status]
        error = abs(status_returns - target) / target * 100
        
        print(f"  {status:<8}: Parsed={status_returns:>7,.0f}, Target={target:>7,.0f}, "
              f"Error={error:.2f}%")
    
    total_parsed = comprehensive_df['returns'].sum()
    total_target = sum(a2_total_returns.values())
    total_error = abs(total_parsed - total_target) / total_target * 100
    
    print(f"\n  TOTAL    : Parsed={total_parsed:>7,.0f}, Target={total_target:>7,.0f}, "
          f"Error={total_error:.2f}%")
    
    if total_error < 0.1:
        print("\n✅ Validation passed! Comprehensive benchmarks cover 100% of returns.")
    else:
        print(f"\n⚠️  Warning: {total_error:.2f}% error in total returns")


if __name__ == '__main__':
    main()
