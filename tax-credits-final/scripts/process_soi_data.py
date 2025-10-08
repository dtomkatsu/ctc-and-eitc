#!/usr/bin/env python3
"""
Process IRS SOI data for tax credit analysis.

This script loads, cleans, and processes IRS Statistics of Income (SOI) data
for Child Tax Credit (CTC) and Earned Income Tax Credit (EITC) analysis.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import yaml
from tqdm import tqdm

# Project directories
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

class SOIDataProcessor:
    """Process IRS SOI data for tax credit analysis."""
    
    def __init__(self, year: int = 2020):
        """Initialize with the tax year to analyze."""
        self.year = year
        self.data = {}
        self.metadata = {}
        
    def load_zip_data(self, filepath: Path) -> pd.DataFrame:
        """Load and clean ZIP code level SOI data."""
        # Read the CSV file
        df = pd.read_csv(filepath, encoding='latin1')
        
        # Basic cleaning
        df = df.rename(columns=str.lower)
        
        # Convert ZIP code to string, handle leading zeros
        if 'zipcode' in df.columns:
            df['zipcode'] = df['zipcode'].astype(str).str.zfill(5)
            
        return df
    
    def process_ctc_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process CTC-related columns."""
        ctc_cols = [
            'nctc', 'actc', 'nctcref', 'actcref', 'nctcnon', 'actcnoni',
            'nctcpart', 'actcpart', 'nctcfor', 'actcfor'
        ]
        
        # Only keep columns that exist in the dataframe
        ctc_cols = [col for col in ctc_cols if col in df.columns]
        
        if not ctc_cols:
            return df
            
        # Calculate per-return averages
        if 'nret' in df.columns:
            for amount_col in [col for col in ctc_cols if col.startswith('a')]:
                count_col = 'n' + amount_col[1:]
                if count_col in df.columns:
                    df[f'avg_{amount_col[1:]}'] = df[amount_col] / df[count_col].replace(0, np.nan)
        
        return df
    
    def process_eitc_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process EITC-related columns."""
        eitc_cols = ['eitc', 'nret', 'nchild']
        
        # Only keep columns that exist in the dataframe
        eitc_cols = [col for col in eitc_cols if col in df.columns]
        
        if not eitc_cols:
            return df
            
        # Calculate per-return averages
        if 'nret' in df.columns and 'eitc' in df.columns:
            df['avg_eitc'] = df['eitc'] / df['nret'].replace(0, np.nan)
            
        return df
    
    def aggregate_by_geo(self, df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
        """Aggregate data by geographic level."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        agg_dict = {col: 'sum' for col in numeric_cols}
        
        # For average columns, take the weighted average
        avg_cols = [col for col in numeric_cols if col.startswith('avg_')]
        for col in avg_cols:
            base_col = col[4:]  # Remove 'avg_'
            if base_col in df.columns and 'nret' in df.columns:
                agg_dict[col] = lambda x: np.average(x, weights=df.loc[x.index, 'nret'])
        
        return df.groupby(group_cols, as_index=False).agg(agg_dict)
    
    def process_all_data(self):
        """Process all available SOI data files."""
        print(f"Processing SOI data for {self.year}...")
        
        # Process each file in the raw data directory
        for filepath in tqdm(list(RAW_DATA_DIR.glob('*.csv'))):
            filename = filepath.stem.lower()
            
            # Skip non-data files
            if not any(x in filename for x in ['zip', 'county', 'agi']):
                continue
                
            print(f"Processing {filepath.name}...")
            
            # Load and process the data
            df = self.load_zip_data(filepath)
            
            # Process tax credit data
            df = self.process_ctc_data(df)
            df = self.process_eitc_data(df)
            
            # Save processed data
            output_file = PROCESSED_DATA_DIR / f"processed_{filepath.name}"
            df.to_csv(output_file, index=False)
            
            print(f"Saved processed data to {output_file}")

def main():
    """Main function to process SOI data."""
    processor = SOIDataProcessor(year=2020)
    processor.process_all_data()
    
    print("\nProcessing complete!")
    print(f"Processed data saved to: {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    main()
