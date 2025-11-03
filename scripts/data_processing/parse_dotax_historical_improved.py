#!/usr/bin/env python3
"""
Improved DOTAX Historical Data Parser

Extracts clean, standardized data from DOTAX historical spreadsheets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_table_5a_improved(file_path: Path, year: int) -> pd.DataFrame:
    """
    Parse Table 5A with proper column names.
    
    Returns DataFrame with columns:
    - year
    - filing_status
    - num_returns
    - pct_returns
    - agi_positive_millions
    - pct_agi_positive
    - agi_negative_millions
    - pct_agi_negative
    - tax_before_credits_millions
    - pct_tax_before_credits
    - tax_after_credits_millions
    - pct_tax_after_credits
    """
    logger.info(f"Parsing Table 5A from {year}...")
    
    try:
        # Read the raw data - data starts at row 5 (0-indexed row 4)
        df_data = pd.read_excel(file_path, sheet_name='5A', skiprows=4)
        
        # Extract data rows (before TOTAL)
        data = []
        for idx, row in df_data.iterrows():
            filing_status = str(row.iloc[0]).strip()
            
            # Stop at TOTAL
            if 'TOTAL' in filing_status.upper() or filing_status == 'nan':
                break
            
            # Clean filing status name
            filing_status = filing_status.replace('*', '').strip()
            
            # Extract values by position
            # Typical structure: Status, Returns, %, AGI+, %, AGI-, %, Tax Before, %, Tax After, %
            try:
                record = {
                    'year': year,
                    'filing_status': filing_status,
                    'num_returns': float(row.iloc[1]) if pd.notna(row.iloc[1]) else None,
                    'pct_returns': float(row.iloc[2]) if pd.notna(row.iloc[2]) else None,
                    'agi_positive_millions': float(row.iloc[3]) if pd.notna(row.iloc[3]) else None,
                    'pct_agi_positive': float(row.iloc[4]) if pd.notna(row.iloc[4]) else None,
                    'agi_negative_millions': float(row.iloc[5]) if pd.notna(row.iloc[5]) else None,
                    'pct_agi_negative': float(row.iloc[6]) if pd.notna(row.iloc[6]) else None,
                    'tax_before_credits_millions': float(row.iloc[7]) if pd.notna(row.iloc[7]) else None,
                    'pct_tax_before_credits': float(row.iloc[8]) if pd.notna(row.iloc[8]) else None,
                    'tax_after_credits_millions': float(row.iloc[9]) if pd.notna(row.iloc[9]) else None,
                    'pct_tax_after_credits': float(row.iloc[10]) if pd.notna(row.iloc[10]) else None,
                }
                data.append(record)
            except Exception as e:
                logger.warning(f"  Could not parse row {idx}: {e}")
                continue
        
        if not data:
            logger.warning(f"  No data extracted for {year}")
            return None
        
        result_df = pd.DataFrame(data)
        
        # Calculate derived metrics
        result_df['agi_net_millions'] = result_df['agi_positive_millions'] + result_df['agi_negative_millions']
        result_df['effective_rate_before_credits'] = (
            result_df['tax_before_credits_millions'] / result_df['agi_net_millions'] * 100
        )
        result_df['effective_rate_after_credits'] = (
            result_df['tax_after_credits_millions'] / result_df['agi_net_millions'] * 100
        )
        result_df['credit_reduction_pct'] = (
            (result_df['tax_before_credits_millions'] - result_df['tax_after_credits_millions']) / 
            result_df['tax_before_credits_millions'] * 100
        )
        result_df['avg_agi_per_return'] = (
            result_df['agi_net_millions'] * 1_000_000 / result_df['num_returns']
        )
        result_df['avg_tax_per_return'] = (
            result_df['tax_after_credits_millions'] * 1_000_000 / result_df['num_returns']
        )
        
        logger.info(f"  Successfully extracted {len(result_df)} filing status categories")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error parsing Table 5A for {year}: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_growth_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate year-over-year growth rates."""
    df = df.sort_values(['filing_status', 'year'])
    
    growth_metrics = []
    
    for filing_status in df['filing_status'].unique():
        fs_data = df[df['filing_status'] == filing_status].sort_values('year')
        
        for i in range(1, len(fs_data)):
            prev = fs_data.iloc[i-1]
            curr = fs_data.iloc[i]
            
            record = {
                'filing_status': filing_status,
                'from_year': prev['year'],
                'to_year': curr['year'],
                'returns_growth_pct': ((curr['num_returns'] / prev['num_returns'] - 1) * 100) if prev['num_returns'] > 0 else None,
                'agi_growth_pct': ((curr['agi_net_millions'] / prev['agi_net_millions'] - 1) * 100) if prev['agi_net_millions'] > 0 else None,
                'tax_growth_pct': ((curr['tax_after_credits_millions'] / prev['tax_after_credits_millions'] - 1) * 100) if prev['tax_after_credits_millions'] > 0 else None,
            }
            growth_metrics.append(record)
    
    return pd.DataFrame(growth_metrics)


def calculate_cagr(df: pd.DataFrame, filing_status: str = None) -> dict:
    """Calculate Compound Annual Growth Rate."""
    if filing_status:
        df = df[df['filing_status'] == filing_status]
    
    df = df.sort_values('year')
    
    if len(df) < 2:
        return None
    
    first_year = df.iloc[0]
    last_year = df.iloc[-1]
    years = last_year['year'] - first_year['year']
    
    cagr = {
        'filing_status': filing_status if filing_status else 'ALL',
        'start_year': first_year['year'],
        'end_year': last_year['year'],
        'years': years,
        'returns_cagr': ((last_year['num_returns'] / first_year['num_returns']) ** (1/years) - 1) * 100 if first_year['num_returns'] > 0 else None,
        'agi_cagr': ((last_year['agi_net_millions'] / first_year['agi_net_millions']) ** (1/years) - 1) * 100 if first_year['agi_net_millions'] > 0 else None,
        'tax_cagr': ((last_year['tax_after_credits_millions'] / first_year['tax_after_credits_millions']) ** (1/years) - 1) * 100 if first_year['tax_after_credits_millions'] > 0 else None,
    }
    
    return cagr


def main():
    """Main parsing workflow."""
    logger.info("="*80)
    logger.info("IMPROVED DOTAX HISTORICAL DATA PARSER")
    logger.info("="*80)
    
    # Set paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / 'data' / 'raw' / 'dotax_historical_data'
    output_dir = project_root / 'data' / 'processed' / 'dotax_historical'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse all years
    all_data = []
    year_files = sorted(data_dir.glob('*indinc_tables.xlsx'))
    
    for file_path in year_files:
        year = int(file_path.stem[:4])
        df = parse_table_5a_improved(file_path, year)
        if df is not None:
            all_data.append(df)
    
    if not all_data:
        logger.error("No data parsed successfully")
        return
    
    # Combine all years
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sort_values(['year', 'filing_status'])
    
    # Save main dataset
    output_file = output_dir / 'dotax_historical_filing_status_clean.csv'
    combined_df.to_csv(output_file, index=False)
    logger.info(f"\nSaved clean data to {output_file}")
    
    # Calculate growth rates
    growth_df = calculate_growth_rates(combined_df)
    growth_file = output_dir / 'dotax_historical_growth_rates.csv'
    growth_df.to_csv(growth_file, index=False)
    logger.info(f"Saved growth rates to {growth_file}")
    
    # Calculate CAGR for each filing status
    cagr_data = []
    for fs in combined_df['filing_status'].unique():
        cagr = calculate_cagr(combined_df, fs)
        if cagr:
            cagr_data.append(cagr)
    
    cagr_df = pd.DataFrame(cagr_data)
    cagr_file = output_dir / 'dotax_historical_cagr.csv'
    cagr_df.to_csv(cagr_file, index=False)
    logger.info(f"Saved CAGR to {cagr_file}")
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"\nYears parsed: {sorted(combined_df['year'].unique())}")
    logger.info(f"Filing statuses: {sorted(combined_df['filing_status'].unique())}")
    logger.info(f"Total records: {len(combined_df)}")
    
    # Print latest year summary
    latest_year = combined_df['year'].max()
    latest_data = combined_df[combined_df['year'] == latest_year]
    
    logger.info(f"\n{latest_year} Summary:")
    logger.info(f"  Total Returns: {latest_data['num_returns'].sum():,.0f}")
    logger.info(f"  Total AGI: ${latest_data['agi_net_millions'].sum():,.1f}M")
    logger.info(f"  Total Tax (After Credits): ${latest_data['tax_after_credits_millions'].sum():,.1f}M")
    logger.info(f"  Average Effective Rate: {(latest_data['tax_after_credits_millions'].sum() / latest_data['agi_net_millions'].sum() * 100):.2f}%")
    
    # Print CAGR summary
    logger.info(f"\nCAGR ({cagr_df['start_year'].iloc[0]:.0f}-{cagr_df['end_year'].iloc[0]:.0f}):")
    for _, row in cagr_df.iterrows():
        logger.info(f"  {row['filing_status']:25s}: Returns {row['returns_cagr']:+.2f}%, AGI {row['agi_cagr']:+.2f}%, Tax {row['tax_cagr']:+.2f}%")


if __name__ == "__main__":
    main()
