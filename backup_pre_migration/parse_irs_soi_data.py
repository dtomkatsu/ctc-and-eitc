#!/usr/bin/env python3
"""
Parse IRS SOI ZIP code data for Hawaii with proper header handling.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def examine_excel_structure():
    """Examine the Excel file structure to understand the layout."""
    data_path = Path('data/irs_soi/22zp12hi.xlsx')
    
    # Read without headers to see raw structure
    df_raw = pd.read_excel(data_path, header=None)
    logger.info(f"Raw data shape: {df_raw.shape}")
    
    # Show first 10 rows to understand structure
    logger.info("First 10 rows of raw data:")
    for i in range(min(10, len(df_raw))):
        row_data = df_raw.iloc[i].head(10).tolist()  # First 10 columns
        logger.info(f"Row {i}: {row_data}")
    
    return df_raw

def find_header_row(df_raw):
    """Find the actual header row in the Excel file."""
    # Look for rows that contain ZIP code indicators
    for i in range(min(20, len(df_raw))):
        row_str = str(df_raw.iloc[i].tolist()).lower()
        if 'zip' in row_str or 'postal' in row_str or 'code' in row_str:
            logger.info(f"Potential header row found at index {i}")
            logger.info(f"Header content: {df_raw.iloc[i].tolist()[:10]}")
            return i
    
    # Look for rows with many non-null values (likely headers)
    for i in range(min(10, len(df_raw))):
        non_null_count = df_raw.iloc[i].notna().sum()
        if non_null_count > 10:  # Assuming headers have many columns
            logger.info(f"Row {i} has {non_null_count} non-null values")
            logger.info(f"Content: {df_raw.iloc[i].head(10).tolist()}")
    
    return None

def parse_irs_data_properly():
    """Parse the IRS data with proper header handling."""
    data_path = Path('data/irs_soi/22zp12hi.xlsx')
    
    # First examine structure
    df_raw = examine_excel_structure()
    
    # Try to find header row
    header_row = find_header_row(df_raw)
    
    if header_row is not None:
        # Read with proper header
        df = pd.read_excel(data_path, header=header_row)
        logger.info(f"Data loaded with header at row {header_row}")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)[:10]}")
    else:
        # Try different approaches
        logger.info("Trying to read with skiprows...")
        for skip in [1, 2, 3, 4, 5]:
            try:
                df = pd.read_excel(data_path, skiprows=skip)
                logger.info(f"Skiprows {skip} - Shape: {df.shape}")
                logger.info(f"First column name: {df.columns[0]}")
                logger.info(f"First few values in col 0: {df.iloc[:3, 0].tolist()}")
                
                # Check if first column looks like ZIP codes
                first_col = df.iloc[:, 0].astype(str)
                if any(val.isdigit() and len(val) == 5 for val in first_col.head(10)):
                    logger.info(f"Found ZIP codes with skiprows={skip}")
                    break
            except Exception as e:
                logger.info(f"Skiprows {skip} failed: {e}")
    
    return df

def analyze_irs_columns(df):
    """Analyze the columns to identify key data fields."""
    logger.info("\n=== COLUMN ANALYSIS ===")
    
    # Show all column names
    logger.info("All columns:")
    for i, col in enumerate(df.columns):
        logger.info(f"  {i+1:3d}. {col}")
    
    # Look for specific patterns
    patterns = {
        'zip_codes': ['zip', 'postal', 'code'],
        'returns': ['return', 'count', 'number'],
        'filing_status': ['single', 'joint', 'married', 'head', 'filing'],
        'income': ['agi', 'income', 'wages', 'salary'],
        'credits': ['ctc', 'credit', 'child', 'dependent'],
        'amounts': ['amount', 'total', 'sum']
    }
    
    categorized = {}
    for category, keywords in patterns.items():
        matching_cols = []
        for col in df.columns:
            col_str = str(col).lower()
            if any(keyword in col_str for keyword in keywords):
                matching_cols.append(col)
        categorized[category] = matching_cols
        if matching_cols:
            logger.info(f"{category.upper()}: {matching_cols}")
    
    return categorized

def examine_data_content(df, categorized):
    """Examine the actual data content."""
    logger.info("\n=== DATA CONTENT ANALYSIS ===")
    
    # Check first column (likely ZIP codes)
    first_col = df.columns[0]
    logger.info(f"First column '{first_col}' sample values:")
    print(df[first_col].head(10))
    
    # Check for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    logger.info(f"Numeric columns ({len(numeric_cols)}): {list(numeric_cols)[:10]}")
    
    # Show basic stats for first few numeric columns
    for col in numeric_cols[:5]:
        logger.info(f"\nStats for {col}:")
        print(df[col].describe())

def create_clean_dataset(df):
    """Create a clean, properly formatted dataset."""
    logger.info("\n=== CREATING CLEAN DATASET ===")
    
    # Try to identify ZIP code column
    zip_col = None
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if values look like ZIP codes
            sample_vals = df[col].dropna().astype(str).head(10)
            if any(val.isdigit() and len(val) == 5 for val in sample_vals):
                zip_col = col
                logger.info(f"Identified ZIP code column: {col}")
                break
    
    if zip_col is None:
        # Use first column as ZIP
        zip_col = df.columns[0]
        logger.info(f"Using first column as ZIP: {zip_col}")
    
    # Create clean dataset
    clean_df = df.copy()
    
    # Rename ZIP column
    if zip_col != 'ZIP_CODE':
        clean_df = clean_df.rename(columns={zip_col: 'ZIP_CODE'})
    
    # Clean ZIP codes
    clean_df['ZIP_CODE'] = clean_df['ZIP_CODE'].astype(str).str.strip()
    
    # Remove rows with invalid ZIP codes
    valid_zips = clean_df['ZIP_CODE'].str.match(r'^\d{5}$')
    clean_df = clean_df[valid_zips]
    
    logger.info(f"Clean dataset shape: {clean_df.shape}")
    logger.info(f"Valid ZIP codes: {len(clean_df)}")
    
    return clean_df

def main():
    """Main parsing function."""
    logger.info("Parsing IRS SOI data for Hawaii...")
    
    # Parse the data properly
    df = parse_irs_data_properly()
    
    # Analyze columns
    categorized = analyze_irs_columns(df)
    
    # Examine content
    examine_data_content(df, categorized)
    
    # Create clean dataset
    clean_df = create_clean_dataset(df)
    
    # Save clean data
    output_path = Path('data/irs_soi/hawaii_irs_soi_clean.csv')
    clean_df.to_csv(output_path, index=False)
    logger.info(f"Clean data saved to: {output_path}")
    
    return clean_df, categorized

if __name__ == "__main__":
    clean_df, categorized = main()
