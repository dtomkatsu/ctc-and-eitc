#!/usr/bin/env python3
"""
Test script to verify the PUMS loader works with 2023 data.
"""
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.pums_loader import PUMSDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_pums_loader():
    """Test loading 2023 PUMS data."""
    try:
        # Initialize loader with 2023 data
        loader = PUMSDataLoader(year=2023)
        
        # Load a small batch of households
        logger = logging.getLogger(__name__)
        logger.info("Loading first 10 households from 2023 PUMS data...")
        
        # Reset batch state and load first batch
        loader.reset_batch_state()
        hh_batch = loader.load_households_batch(batch_size=10, year=2023)
        
        if hh_batch.empty:
            logger.error("No household data loaded. Check if data files exist.")
            return False
            
        logger.info(f"Successfully loaded {len(hh_batch)} households")
        logger.info("Sample household data:")
        logger.info(hh_batch[['SERIALNO', 'NP', 'TYPE', 'ADJINC']].head())
        
        # Load persons for these households
        person_df = loader.load_persons_for_households(
            serialnos=hh_batch['SERIALNO'].tolist(),
            year=2023
        )
        
        if person_df.empty:
            logger.error("No person data loaded. Check if data files exist.")
            return False
            
        logger.info(f"Successfully loaded {len(person_df)} persons")
        logger.info("Sample person data:")
        logger.info(person_df[['SERIALNO', 'SPORDER', 'AGEP', 'SEX', 'HISP', 'RAC1P']].head())
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing PUMS loader: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_pums_loader()
    sys.exit(0 if success else 1)
