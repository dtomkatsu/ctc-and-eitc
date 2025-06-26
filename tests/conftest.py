"""
Pytest configuration and fixtures for testing.
"""
import pytest
import pandas as pd
from pathlib import Path
import sys

# Add src to path to allow importing from tax package
sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture(scope="session")
def sample_household_data():
    """Create sample household data for testing."""
    data = [
        # Married couple
        {
            'SERIALNO': '1', 'SPORDER': '1', 'RELSHIPP': '20', 'SEX': '1', 
            'AGE': '35', 'MAR': '1', 'PINCP': '50000', 'ADJINC': '1000000'
        },
        {
            'SERIALNO': '1', 'SPORDER': '2', 'RELSHIPP': '20', 'SEX': '2', 
            'AGE': '32', 'MAR': '1', 'PINCP': '45000', 'ADJINC': '1000000'
        },
        # Single parent with child
        {
            'SERIALNO': '2', 'SPORDER': '1', 'RELSHIPP': '20', 'SEX': '2', 
            'AGE': '30', 'MAR': '5', 'PINCP': '40000', 'ADJINC': '1000000'
        },
        {
            'SERIALNO': '2', 'SPORDER': '2', 'RELSHIPP': '03', 'SEX': '1', 
            'AGE': '8', 'MAR': '6', 'PINCP': '0', 'ADJINC': '1000000'
        }
    ]
    return pd.DataFrame(data)
