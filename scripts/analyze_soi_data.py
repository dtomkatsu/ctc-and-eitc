#!/usr/bin/env python3
"""
Analyze the SOI (Statistics of Income) data for Hawaii.
"""
import pandas as pd
import sys
from pathlib import Path

def clean_soi_data(df):
    """Clean and structure the SOI data."""
    # Set column names from the first two rows
    df.columns = [
        'category',
        'num_returns',
        'num_efile_returns',
        'gross_collections',
        'num_refunds',
        'refund_amount'
    ]
    
    # Remove the first two rows (headers and units)
    df = df.iloc[2:].reset_index(drop=True)
    
    # Clean up the data
    for col in df.columns[1:]:  # Skip the first column (category)
        # Remove commas and convert to numeric, coerce errors to NaN
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.replace(' ', ''), 
                              errors='coerce')
    
    return df

def main():
    # Path to the SOI data file
    soi_file = Path("/Users/dtomkatsu/Downloads/23dbhawaii.xlsx")
    
    try:
        print(f"Analyzing SOI data file: {soi_file}")
        
        # Read the Excel file
        df = pd.read_excel(soi_file, header=None)
        
        # Clean and structure the data
        df_clean = clean_soi_data(df.copy())
        
        # Display the cleaned data
        print("\nCleaned SOI Data:")
        print("=" * 50)
        print(df_clean.to_string())
        
        # Basic statistics
        print("\nSummary Statistics:")
        print("=" * 50)
        
        # Total returns and collections
        total_returns = df_clean.loc[df_clean['category'].str.contains('Total', na=False), 'num_returns'].sum()
        total_collections = df_clean.loc[df_clean['category'].str.contains('Total', na=False), 'gross_collections'].sum()
        
        print(f"\nTotal Returns: {total_returns:,.0f}")
        print(f"Total Collections: ${total_collections:,.0f} (thousands)")
        
        # Breakdown by category
        print("\nBreakdown by Category:")
        print("-" * 50)
        for _, row in df_clean.iterrows():
            if pd.notna(row['num_returns']):
                print(f"\n{row['category']}:")
                print(f"  Returns: {row['num_returns']:,.0f}")
                print(f"  e-File Rate: {row['num_efile_returns']/row['num_returns']*100:.1f}%" if pd.notna(row['num_efile_returns']) else "  e-File Rate: N/A")
                print(f"  Collections: ${row['gross_collections']:,.0f} (thousands)" if pd.notna(row['gross_collections']) else "  Collections: N/A")
        
        # Refund information
        total_refunds = df_clean.loc[df_clean['category'].str.contains('Total', na=False), 'num_refunds'].sum()
        total_refund_amt = df_clean.loc[df_clean['category'].str.contains('Total', na=False), 'refund_amount'].sum()
        
        if total_refunds > 0 and total_refund_amt > 0:
            avg_refund = total_refund_amt * 1000 / total_refunds  # Convert to dollars from thousands
            print(f"\nAverage Refund: ${avg_refund:,.2f}")
        
        return 0
        
    except Exception as e:
        print(f"Error analyzing SOI data: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
