"""
PUMS Data Loader for CTC and EITC Analysis

This module handles loading and preprocessing PUMS data for Hawaii,
with specific attention to variables needed for CTC and EITC calculations.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

# Configure logger
logger = logging.getLogger(__name__)

# Default values
DEFAULT_PUMS_YEAR = 2022  # Most recent year available
DEFAULT_STATE = '15'  # Hawaii FIPS code
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'pums'

class PUMSDataLoader:
    """Handles loading and processing of PUMS data for tax benefit analysis."""
    
    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR):
        """Initialize the PUMS data loader.
        
        Args:
            data_dir: Directory containing PUMS data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Define required columns for person and household data
        self.person_columns = {
            'SERIALNO': str, 'SPORDER': int, 'PUMA': str, 'ST': str,
            'AGEP': int, 'SEX': int, 'HISP': int, 'RAC1P': int,
            'SCHL': int, 'DIS': int, 'MIL': int, 'MILITARY': int,
            'WAGP': float, 'SEMP': float, 'INTP': float, 'RETP': float,
            'SSP': float, 'SSIP': float, 'PAP': float, 'OIP': float,
            'POVPIP': float, 'PWGTP': int, 'NP': int, 'MAR': int,
            'NOC': int, 'PINCP': float, 'ADJINC': float, 'RELSHIPP': int,
            'MSP': int, 'HICOV': int, 'HINS1': int, 'HINS2': int,
            'HINS3': int, 'HINS4': int, 'MIG': int
        }
        
        self.household_columns = {
            'SERIALNO': str, 'PUMA': str, 'ST': str, 'HINCP': float,
            'ADJINC': float, 'WGTP': int, 'NP': int, 'TEN': int,
            'HHT': int, 'NOC': int, 'NRC': int, 'BLD': int,
            'ACCESS': int, 'BROADBND': int, 'COMPOTHX': int, 'HFL': int,
            'FS': int, 'HHL': int, 'LNGI': int, 'MULTG': int,
            'PARTNER': int, 'PLM': int, 'PSF': int, 'R18': int,
            'R65': int, 'RESMODE': int, 'SMX': int, 'SRNT': int,
            'TAXP': int, 'WIF': int, 'WKEXREL': int, 'WORKSTAT': int,
            'YBL': int
        }
    
    def get_total_households(self, state: str = DEFAULT_STATE) -> int:
        """Get the total number of households in the dataset.
        
        Args:
            state: State FIPS code (default: '15' for Hawaii)
            
        Returns:
            Total number of households
        """
        hh_file = self.data_dir / f'psam_h{state}.csv'
        if not hh_file.exists():
            raise FileNotFoundError(f"Household PUMS file not found: {hh_file}")
            
        # Use wc -l to quickly count lines (subtract 1 for header)
        import subprocess
        result = subprocess.run(['wc', '-l', str(hh_file)], capture_output=True, text=True)
        num_lines = int(result.stdout.strip().split()[0])
        return max(0, num_lines - 1)  # Subtract header row
    
    def load_households_batch(
        self, 
        batch_size: int = 1000, 
        offset: int = 0,
        state: str = DEFAULT_STATE
    ) -> pd.DataFrame:
        """Load a batch of household records.
        
        Args:
            batch_size: Number of records to load
            offset: Starting record number
            state: State FIPS code (default: '15' for Hawaii)
            
        Returns:
            DataFrame with household data
        """
        hh_file = self.data_dir / f'psam_h{state}.csv'
        if not hh_file.exists():
            raise FileNotFoundError(f"Household PUMS file not found: {hh_file}")
        
        # Read batch of records
        logger.debug(f"Loading household batch: offset={offset}, size={batch_size}")
        hh_df = pd.read_csv(
            hh_file,
            skiprows=range(1, offset + 1),  # +1 to skip header
            nrows=batch_size,
            dtype=self.household_columns
        )
        
        if hh_df.empty:
            return pd.DataFrame()
            
        # Apply income adjustments and filters
        hh_df = self._adjust_income(hh_df)
        if 'ST' in hh_df.columns:
            hh_df = hh_df[hh_df['ST'] == state].copy()
            
        return hh_df
    
    def load_persons_for_households(
        self, 
        serialnos: List[str],
        state: str = DEFAULT_STATE
    ) -> pd.DataFrame:
        """Load person records for specific households.
        
        Args:
            serialnos: List of SERIALNO values to load
            state: State FIPS code (default: '15' for Hawaii)
            
        Returns:
            DataFrame with person data for the specified households
        """
        # Check if serialnos is empty or None
        if serialnos is None or (hasattr(serialnos, '__len__') and len(serialnos) == 0):
            return pd.DataFrame()
            
        person_file = self.data_dir / f'psam_p{state}.csv'
        if not person_file.exists():
            raise FileNotFoundError(f"Person PUMS file not found: {person_file}")
        
        # Convert serialnos to a set for faster lookups
        if hasattr(serialnos, 'tolist'):  # Handle numpy array or pandas Series
            serialnos = serialnos.tolist()
        serialno_set = set(str(sn) for sn in serialnos)
        
        # Use chunking to process large files
        chunks = []
        chunk_size = 10000
        
        for chunk in pd.read_csv(person_file, chunksize=chunk_size, dtype=self.person_columns):
            # Filter for our households
            chunk = chunk[chunk['SERIALNO'].astype(str).isin(serialno_set)]
            if not chunk.empty:
                chunks.append(chunk)
        
        if not chunks:
            return pd.DataFrame()
            
        # Combine chunks and process
        person_df = pd.concat(chunks, ignore_index=True)
        person_df = self._adjust_income(person_df)
        
        if 'ST' in person_df.columns:
            person_df = person_df[person_df['ST'] == state].copy()
            
        return person_df
    
    def load_data(
        self, 
        year: int = DEFAULT_PUMS_YEAR,
        state: str = DEFAULT_STATE,
        sample_size: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and process PUMS data for the specified year and state.
        
        Args:
            year: Year of PUMS data to load
            state: State FIPS code (default: '15' for Hawaii)
            sample_size: If provided, return a random sample of this size
            
        Returns:
            Tuple of (persons_df, households_df) DataFrames
        """
        logger.info(f"Loading PUMS data for state {state} from {year}...")
        
        # Load all households first by getting total count and loading in batches
        total_households = self.get_total_households(state=state)
        logger.info(f"Loading all {total_households} households...")
        
        # Load in batches to manage memory
        batch_size = 10000
        hh_batches = []
        
        for offset in range(0, total_households, batch_size):
            current_batch_size = min(batch_size, total_households - offset)
            logger.debug(f"Loading household batch: {offset} to {offset + current_batch_size}")
            batch = self.load_households_batch(
                batch_size=current_batch_size,
                offset=offset,
                state=state
            )
            hh_batches.append(batch)
            
        hh_df = pd.concat(hh_batches, ignore_index=True)
        
        # Load all persons for these households
        person_df = self.load_persons_for_households(
            serialnos=hh_df['SERIALNO'].tolist(),
            state=state
        )
        
        # Take sample if requested
        if sample_size and sample_size < len(person_df):
            logger.info(f"Taking random sample of {sample_size} persons...")
            person_df = person_df.sample(n=sample_size, random_state=42)
            
        return person_df, hh_df
    
    def _adjust_income(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare income columns by filling NA values with 0.
        
        Note: We don't apply the ADJINC factor here because it will be applied
        in the _calculate_income method of the TaxUnitConstructor.
        """
        df = df.copy()
        
        # Define monetary columns that need NA filling
        monetary_cols = [
            'WAGP', 'SEMP', 'INTP', 'DIV', 'RETP', 'SSP', 'SSIP', 
            'OIP', 'PAP', 'PINCP', 'HINCP', 'PERNP', 'POVPIP'
        ]
        
        # Fill NA with 0 for all monetary columns
        for col in monetary_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
                
        # Ensure ADJINC is a float and fill NA with 1.0 (no adjustment)
        if 'ADJINC' in df.columns:
            df['ADJINC'] = df['ADJINC'].fillna(1.0).astype(float)
        
        return df
    
    def create_tax_units(self, person_df: pd.DataFrame, 
                        hh_df: pd.DataFrame) -> pd.DataFrame:
        """Construct tax units from person and household data.
        
        Args:
            person_df: DataFrame of person records
            hh_df: DataFrame of household records
            
        Returns:
            DataFrame of tax units with relevant variables for CTC/EITC
        """
        logger.info("Constructing tax units...")
        
        # Merge person and household data
        merged = pd.merge(
            person_df,
            hh_df[['SERIALNO', 'WGTP', 'HINCP', 'TEN']],
            on='SERIALNO',
            how='left'
        )
        
        # Identify tax filers (simplified)
        merged['is_filer'] = (merged['PINCP'] > 0) | (merged['WAGP'] > 0)
        
        # Group by household to identify potential tax units
        tax_units = merged.groupby('SERIALNO').apply(self._create_tax_unit).reset_index()
        
        return tax_units
    
    def _create_tax_unit(self, hh_group: pd.DataFrame) -> pd.Series:
        """Create a tax unit from a household group."""
        # This is a simplified version - would need to be expanded
        # based on actual tax unit construction rules
        
        # Get householder (person with RELSHIP = 20 or lowest SPORDER if not found)
        householders = hh_group[hh_group['RELSHIPP'] == 20]
        if len(householders) == 0:
            householders = hh_group[hh_group['SPORDER'] == hh_group['SPORDER'].min()]
        
        if len(householders) == 0:
            return pd.Series()
            
        householder = householders.iloc[0]
        
        # Count dependents
        dependents = hh_group[
            (hh_group['AGEP'] < 19) | 
            ((hh_group['AGEP'] < 24) & (hh_group['SCHL'].between(1, 15, inclusive='both')))
        ]
        
        # Create tax unit record
        tax_unit = {
            'serialno': householder['SERIALNO'],
            'filer_age': householder['AGEP'],
            'filer_sex': householder['SEX'],
            'filer_race': householder['RAC1P'],
            'filer_hispanic': householder['HISP'] > 1,
            'marital_status': householder['MAR'],
            'num_dependents': len(dependents),
            'wages': hh_group['WAGP'].sum(),
            'self_emp_income': hh_group['SEMP'].sum(),
            'investment_income': hh_group[['INTP', 'DIV', 'RENT']].sum().sum(),
            'retirement_income': hh_group[['RETP', 'SSP', 'SSIP']].sum().sum(),
            'public_assistance': hh_group[['PAP', 'OIP']].sum().sum(),
            'household_weight': householder.get('WGTP', 1)
        }
        
        return pd.Series(tax_unit)

def load_pums_data(year: int = DEFAULT_PUMS_YEAR,
                  state: str = DEFAULT_STATE,
                  sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience function to load PUMS data.
    
    Args:
        year: Year of PUMS data to load
        state: State FIPS code (default: '15' for Hawaii)
        sample_size: If provided, return a random sample of this size
        
    Returns:
        Tuple of (persons_df, households_df) DataFrames
    """
    loader = PUMSDataLoader()
    return loader.load_data(year=year, state=state, sample_size=sample_size)
