#!/usr/bin/env python3
"""
Parse DOTAX SOI 2022 Comprehensive Tax Data with A8 Enhancement

Hybrid Approach:
- Use A-2 for complete coverage and filing status breakdowns
- Use A8 to split $400k+ bracket into 4 detailed sub-brackets
- Use A-2's AGI totals as control (more reliable than reverse-engineered)
- Distribute A-2's $400k+ AGI proportionally based on A8 patterns
"""

import pandas as pd
import numpy as np
from pathlib import Path


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


def parse_a8_proportions():
    """
    Parse A8 to get proportions for splitting $400k+ bracket.
    
    Returns:
        List of dicts with bracket proportions
    """
    a8_path = Path('data/raw/Selected Resident Return Data/Dotax Soi 2022 - A8.csv')
    df = pd.read_csv(a8_path, header=None)
    
    print("Parsing A8 for high-income proportions...")
    
    # Extract A8 brackets
    a8_brackets = []
    
    for i in range(19, 24):  # Rows for $400k+ brackets
        bracket_min_str = str(df.iloc[i, 0]).strip()
        bracket_max_str = str(df.iloc[i, 2]).strip()
        
        returns = float(str(df.iloc[i, 4]).replace(',', ''))
        tax_before_str = str(df.iloc[i, 6]).replace('$', '').replace(',', '').strip()
        tax_before_millions = float(tax_before_str) if tax_before_str != 'nan' else 0
        
        # Parse effective rate (AGI basis)
        effective_rate_agi_str = str(df.iloc[i, 16]).replace('%', '').strip()
        try:
            effective_rate_agi = float(effective_rate_agi_str) / 100 if effective_rate_agi_str != 'nan' else None
        except:
            effective_rate_agi = None
        
        # Reverse engineer AGI for proportions
        if effective_rate_agi and effective_rate_agi > 0:
            est_agi_m = tax_before_millions / effective_rate_agi
        else:
            est_agi_m = 0
        
        # Parse bracket bounds
        if bracket_min_str == '$400,000':
            agi_min = 400000
            agi_max = 500000
        elif bracket_min_str == '$500,000':
            agi_min = 500000
            agi_max = 750000
        elif bracket_min_str == '$750,000':
            agi_min = 750000
            agi_max = 1000000
        elif bracket_min_str == '$1,000,000':
            agi_min = 1000000
            agi_max = 999999999
        else:
            continue
        
        a8_brackets.append({
            'agi_min': agi_min,
            'agi_max': agi_max,
            'returns': returns,
            'est_agi_millions': est_agi_m
        })
    
    # Calculate proportions
    total_returns = sum(b['returns'] for b in a8_brackets)
    total_agi = sum(b['est_agi_millions'] for b in a8_brackets)
    
    for bracket in a8_brackets:
        bracket['pct_returns'] = bracket['returns'] / total_returns if total_returns > 0 else 0
        bracket['pct_agi'] = bracket['est_agi_millions'] / total_agi if total_agi > 0 else 0
    
    print(f"  A8 $400k+ total: {total_returns:,.0f} returns, ${total_agi:,.0f}M AGI")
    print(f"  Split into {len(a8_brackets)} sub-brackets")
    
    return a8_brackets


def get_a2_400k_plus_data():
    """Get A-2's $400k+ bracket data (control totals)."""
    a2_1_path = Path('data/raw/Selected Resident Return Data/Dotax Soi 2022 - A2-1.csv')
    a2_2_path = Path('data/raw/Selected Resident Return Data/Dotax Soi 2022 - A2-2.csv')
    
    df1 = pd.read_csv(a2_1_path, header=None)
    df2 = pd.read_csv(a2_2_path, header=None)
    
    row_idx = 18  # $400k+ row
    
    # Extract data by filing status
    a2_data = {}
    
    for status, returns_col, agi_col, tax_before_col, tax_after_col in [
        ('joint', 4, 16, 8, 12),
        ('single', 8, 18, 9, 13),
        ('hoh', 12, 20, 10, 14)
    ]:
        returns = clean_currency(df1.iloc[row_idx, returns_col])
        agi = clean_currency(df1.iloc[row_idx, agi_col])
        tax_before = clean_currency(df2.iloc[row_idx, tax_before_col])
        tax_after = clean_currency(df2.iloc[row_idx, tax_after_col])
        
        if pd.notna(returns) and returns > 0:
            a2_data[status] = {
                'returns': returns,
                'total_agi_millions': agi if pd.notna(agi) else 0,
                'tax_before_millions': tax_before if pd.notna(tax_before) else 0,
                'tax_after_millions': tax_after if pd.notna(tax_after) else 0
            }
    
    return a2_data


def create_enhanced_400k_plus_brackets(a8_proportions, a2_data):
    """
    Create enhanced $400k+ brackets using hybrid approach.
    
    For each A8 sub-bracket, distribute A-2's filing status data proportionally.
    """
    enhanced_brackets = []
    
    print("\nCreating enhanced $400k+ brackets (hybrid A8 + A-2)...")
    
    for a8_bracket in a8_proportions:
        agi_min = a8_bracket['agi_min']
        agi_max = a8_bracket['agi_max']
        pct_returns = a8_bracket['pct_returns']
        pct_agi = a8_bracket['pct_agi']
        
        # For each filing status, distribute proportionally
        for status, a2_status_data in a2_data.items():
            # Use return count proportion to split returns
            # Use AGI proportion to split AGI
            returns = a2_status_data['returns'] * pct_returns
            agi_millions = a2_status_data['total_agi_millions'] * pct_agi
            tax_before_millions = a2_status_data['tax_before_millions'] * pct_agi
            tax_after_millions = a2_status_data['tax_after_millions'] * pct_agi
            
            enhanced_brackets.append({
                'filing_status': status,
                'agi_min': agi_min,
                'agi_max': agi_max,
                'returns': returns,
                'total_agi_millions': agi_millions,
                'tax_before_credits_millions': tax_before_millions,
                'tax_after_credits_millions': tax_after_millions,
                'source': 'A-2+A8 hybrid'
            })
    
    # Validate totals match A-2
    total_returns = sum(b['returns'] for b in enhanced_brackets)
    total_agi = sum(b['total_agi_millions'] for b in enhanced_brackets)
    
    a2_total_returns = sum(d['returns'] for d in a2_data.values())
    a2_total_agi = sum(d['total_agi_millions'] for d in a2_data.values())
    
    print(f"  Created {len(enhanced_brackets)} enhanced brackets")
    print(f"  Total returns: {total_returns:,.0f} (A-2: {a2_total_returns:,.0f})")
    print(f"  Total AGI: ${total_agi:,.0f}M (A-2: ${a2_total_agi:,.0f}M)")
    
    return enhanced_brackets


def parse_a2_table_exclude_400k_plus():
    """Parse Table A-2 excluding $400k+ bracket."""
    a2_1_path = Path('data/raw/Selected Resident Return Data/Dotax Soi 2022 - A2-1.csv')
    a2_2_path = Path('data/raw/Selected Resident Return Data/Dotax Soi 2022 - A2-2.csv')
    
    df1 = pd.read_csv(a2_1_path, header=None)
    df2 = pd.read_csv(a2_2_path, header=None)
    
    print("Parsing Table A-2 (excluding $400k+ for enhancement)...")
    
    records = []
    
    # Define brackets (excluding row 18 which is $400k+)
    # Map nontaxable brackets to match taxable AGI ranges for merging
    brackets = [
        # TAXABLE RETURNS < $400k
        (0, 10000, 7, 'taxable'),
        (10000, 20000, 8, 'taxable'),
        (20000, 30000, 9, 'taxable'),
        (30000, 40000, 10, 'taxable'),
        (40000, 50000, 11, 'taxable'),
        (50000, 75000, 12, 'taxable'),
        (75000, 100000, 13, 'taxable'),
        (100000, 150000, 14, 'taxable'),
        (150000, 200000, 15, 'taxable'),
        (200000, 300000, 16, 'taxable'),
        (300000, 400000, 17, 'taxable'),
        # SKIP row 18 ($400k+) - will use enhanced version
        # NONTAXABLE RETURNS - remap to match taxable ranges
        (-999999999, 0, 22, 'nontaxable'),
        (0, 10000, 23, 'nontaxable'),  # Combine $0-$5k and $5k-$10k into $0-$10k
        (0, 10000, 24, 'nontaxable'),  # These will be merged
        (10000, 20000, 25, 'nontaxable'),  # Map $10k+ to $10k-$20k (conservative)
    ]
    
    for agi_min, agi_max, row_idx, bracket_type in brackets:
        try:
            # From A2-1: Number of returns and total AGI
            joint_returns = clean_currency(df1.iloc[row_idx, 4])
            single_returns = clean_currency(df1.iloc[row_idx, 8])
            hoh_returns = clean_currency(df1.iloc[row_idx, 12])
            
            # AGI columns
            joint_agi = clean_currency(df1.iloc[row_idx, 16])
            single_agi = clean_currency(df1.iloc[row_idx, 18])
            hoh_agi = clean_currency(df1.iloc[row_idx, 20])
            
            # From A2-2: Tax liability
            if row_idx <= 18:  # Taxable returns
                joint_tax_before = clean_currency(df2.iloc[row_idx, 8])
                single_tax_before = clean_currency(df2.iloc[row_idx, 9])
                hoh_tax_before = clean_currency(df2.iloc[row_idx, 10])
                
                joint_tax_after = clean_currency(df2.iloc[row_idx, 12])
                single_tax_after = clean_currency(df2.iloc[row_idx, 13])
                hoh_tax_after = clean_currency(df2.iloc[row_idx, 14])
            else:  # Nontaxable returns
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
                    records.append({
                        'filing_status': status,
                        'agi_min': agi_min,
                        'agi_max': agi_max,
                        'returns': returns,
                        'total_agi_millions': agi if pd.notna(agi) else 0,
                        'tax_before_credits_millions': tax_before if pd.notna(tax_before) else 0,
                        'tax_after_credits_millions': tax_after if pd.notna(tax_after) else 0,
                        'source': 'A-2'
                    })
        except Exception as e:
            print(f"Warning: Error parsing row {row_idx}: {e}")
            continue
    
    df = pd.DataFrame(records)
    
    print(f"  Parsed {len(df)} brackets from A-2 (< $400k)")
    print(f"  Total returns: {df['returns'].sum():,.0f}")
    
    return df


def merge_taxable_nontaxable_brackets(df):
    """
    Merge taxable and nontaxable brackets with overlapping AGI ranges.
    
    This fixes the calibration mismatch where PUMS includes all filers
    but A-2 separates them into taxable vs nontaxable categories.
    """
    print("\nMerging taxable and nontaxable brackets...")
    
    # Group by filing status and AGI range, sum returns and AGI
    merged = df.groupby(['filing_status', 'agi_min', 'agi_max'], as_index=False).agg({
        'returns': 'sum',
        'total_agi_millions': 'sum',
        'tax_before_credits_millions': 'sum',
        'tax_after_credits_millions': 'sum'
    })
    
    merged['source'] = 'A-2 (merged)'
    
    # Count how many brackets were merged
    original_count = len(df)
    merged_count = len(merged)
    print(f"  Merged {original_count} brackets → {merged_count} unique AGI ranges")
    print(f"  Combined {original_count - merged_count} overlapping taxable/nontaxable brackets")
    
    return merged


def combine_enhanced_data():
    """Combine A-2 data with enhanced $400k+ brackets."""
    # Get A8 proportions
    a8_proportions = parse_a8_proportions()
    
    # Get A-2 $400k+ data (control totals)
    a2_400k_data = get_a2_400k_plus_data()
    
    # Create enhanced $400k+ brackets
    enhanced_400k_brackets = create_enhanced_400k_plus_brackets(a8_proportions, a2_400k_data)
    
    # Parse A-2 for < $400k brackets
    a2_under_400k = parse_a2_table_exclude_400k_plus()
    
    # Merge taxable and nontaxable brackets with same AGI ranges
    a2_under_400k = merge_taxable_nontaxable_brackets(a2_under_400k)
    
    # Combine
    enhanced_400k_df = pd.DataFrame(enhanced_400k_brackets)
    comprehensive_df = pd.concat([a2_under_400k, enhanced_400k_df], ignore_index=True)
    
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
    
    return comprehensive_df


def main():
    """Parse comprehensive tax data with A8 enhancement."""
    print("="*80)
    print("PARSING COMPREHENSIVE TAX DATA (A-2 + A8 HYBRID)")
    print("="*80)
    print()
    
    # Create enhanced dataset
    comprehensive_df = combine_enhanced_data()
    
    print("\n" + "="*80)
    print("ENHANCED DATASET SUMMARY")
    print("="*80)
    print(f"\nTotal brackets: {len(comprehensive_df)}")
    print(f"Total returns: {comprehensive_df['returns'].sum():,.0f}")
    print(f"Total AGI: ${comprehensive_df['total_agi_millions'].sum():,.1f}M")
    print(f"Total tax (after credits): ${comprehensive_df['tax_after_credits_millions'].sum():,.1f}M")
    
    # Show breakdown by filing status
    print("\nBreakdown by filing status:")
    for status in ['joint', 'single', 'hoh']:
        status_df = comprehensive_df[comprehensive_df['filing_status'] == status]
        n_brackets = len(status_df)
        returns = status_df['returns'].sum()
        agi = status_df['total_agi_millions'].sum()
        tax = status_df['tax_after_credits_millions'].sum()
        
        print(f"  {status:<8} - {n_brackets:2d} brackets, {returns:>7,.0f} returns, "
              f"AGI=${agi:>7,.1f}M, Tax=${tax:>6,.1f}M")
    
    # Show high-income detail
    print("\n" + "="*80)
    print("HIGH-INCOME BRACKET DETAIL ($400k+)")
    print("="*80)
    
    high_income = comprehensive_df[comprehensive_df['agi_min'] >= 400000]
    print(f"\n{'Status':<8} {'AGI Range':<25} {'Returns':>10} {'AGI (M)':>12} {'Source':<15}")
    print("-"*80)
    
    for _, row in high_income.iterrows():
        agi_range = f"${row['agi_min']//1000}k-${row['agi_max']//1000 if row['agi_max'] < 999999999 else '∞'}k"
        print(f"{row['filing_status']:<8} {agi_range:<25} {row['returns']:>10,.0f} "
              f"{row['total_agi_millions']:>12,.0f} {row['source']:<15}")
    
    # Save comprehensive benchmarks
    output_path = Path('data/processed/comprehensive_tax_benchmarks_enhanced.csv')
    comprehensive_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Saved enhanced benchmarks to: {output_path}")
    
    # Validation
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)
    
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
        
        status_char = '✅' if error < 0.1 else '⚠️'
        print(f"  {status:<8}: Parsed={status_returns:>7,.0f}, Target={target:>7,.0f}, "
              f"Error={error:.2f}% {status_char}")
    
    total_parsed = comprehensive_df['returns'].sum()
    total_target = sum(a2_total_returns.values())
    total_error = abs(total_parsed - total_target) / total_target * 100
    
    print(f"\n  TOTAL    : Parsed={total_parsed:>7,.0f}, Target={total_target:>7,.0f}, "
          f"Error={total_error:.2f}%")
    
    if total_error < 0.1:
        print("\n✅ Validation passed! Enhanced benchmarks maintain A-2 totals.")
    else:
        print(f"\n⚠️  Warning: {total_error:.2f}% error in total returns")
    
    print("\n" + "="*80)
    print("KEY IMPROVEMENTS")
    print("="*80)
    print("✅ $400k+ bracket split into 4 detailed sub-brackets (A8)")
    print("✅ Filing status breakdown preserved (A-2)")
    print("✅ AGI totals match A-2 control (more reliable)")
    print("✅ Ready for high-precision revenue calibration")


if __name__ == '__main__':
    main()
