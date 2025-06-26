#!/usr/bin/env python3
"""
Examine the SOI (Statistics of Income) data for Hawaii.
"""
import pandas as pd
import sys
from pathlib import Path

def main():
    # Path to the SOI data file
    soi_file = Path("/Users/dtomkatsu/Downloads/23dbhawaii.xlsx")
    
    try:
        # Read the Excel file
        print(f"Examining SOI data file: {soi_file}")
        
        # Get all sheet names
        xl = pd.ExcelFile(soi_file)
        print("\nSheets in the Excel file:")
        for sheet in xl.sheet_names:
            print(f"- {sheet}")
        
        # Read and display the first few rows of each sheet
        for sheet_name in xl.sheet_names:
            print(f"\nExamining sheet: {sheet_name}")
            print("=" * (len(sheet_name) + 10))
            
            # Read the sheet
            df = pd.read_excel(soi_file, sheet_name=sheet_name)
            
            # Basic info
            print(f"Shape: {df.shape}")
            print("\nFirst 5 rows:")
            print(df.head().to_string())
            
            print("\nColumn names:")
            print("\n".join([f"- {col}" for col in df.columns]))
            
            print("\nData types:")
            print(df.dtypes)
            
            # Check for summary statistics if the data looks numeric
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                print("\nSummary statistics for numeric columns:")
                print(df[numeric_cols].describe().to_string())
        
        return 0
        
    except Exception as e:
        print(f"Error examining SOI data: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
