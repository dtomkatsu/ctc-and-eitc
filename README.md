# CTC and EITC Analysis for Hawaii

This project analyzes the Child Tax Credit (CTC) and Earned Income Tax Credit (EITC) using PUMS microdata for Hawaii at various geographic levels (state, county, and legislative district). The system includes robust tax unit construction with support for various filing statuses and Hawaii-specific rules.

## Project Structure

```
ctc-and-eitc/
├── data/                           # Data storage
│   ├── raw/                       # Raw data files (e.g., PUMS microdata)
│   └── processed/                 # Processed and cleaned data
├── notebooks/                     # Jupyter notebooks for analysis
├── src/                           # Source code
│   ├── tax/                       # Core tax calculation logic
│   │   ├── units/                 # Tax unit construction
│   │   │   ├── __init__.py        # Package initialization
│   │   │   ├── base.py            # Base tax unit constructor
│   │   │   ├── constructor.py     # Main tax unit constructor
│   │   │   ├── dependencies.py    # Dependency determination
│   │   │   ├── filers/            # Filer type implementations
│   │   │   ├── income.py          # Income calculations
│   │   │   ├── relationships.py   # Relationship mapping
│   │   │   ├── status/            # Filing status determination
│   │   │   └── utils.py           # Utility functions
│   │   └── credits/               # Tax credit calculations
│   ├── data/                      # Data processing scripts
│   └── analysis/                  # Analysis scripts
├── tests/                         # Test suite
│   ├── test_smoke.py              # Smoke tests for core functionality
│   └── units/                     # Unit tests for tax unit construction
├── config/                        # Configuration files
└── docs/                          # Documentation
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` and add your Census API key:
     ```
     CENSUS_API_KEY=your_api_key_here
     ```
   - You can get a free API key from: https://api.census.gov/data/key_signup.html

4. Download PUMS data:
   ```bash
   python scripts/download_pums.py
   ```
   This will download the data to `data/raw/pums/`.

   Alternatively, you can specify options:
   ```bash
   python scripts/download_pums.py --year 2022 --state 15 --data-dir data/raw/pums
   ```

## Key Features

- **Tax Unit Construction**: Robust construction of tax units from PUMS data
- **Multiple Filing Statuses**: Support for Single, Head of Household, Married Filing Jointly, and Married Filing Separately
- **Hawaii-Specific Rules**: Custom logic for Hawaii's unique family structures
- **Comprehensive Testing**: Extensive test suite including unit and smoke tests

## Data Sources

- PUMS microdata from the U.S. Census Bureau
- Geographic boundary files for Hawaii
- American Community Survey (ACS) data for validation

## Running Tests

To run the test suite:

```bash
# Run all tests
pytest

# Run smoke tests only
pytest tests/test_smoke.py -v

# Run unit tests for tax unit construction
pytest tests/units/ -v
```

## Usage

Basic usage example:

```python
from tax.units import TaxUnitConstructor
import pandas as pd

# Load your PUMS data
person_df = pd.read_csv('path/to/person_data.csv')
hh_df = pd.read_csv('path/to/household_data.csv')

# Create tax units
constructor = TaxUnitConstructor(person_df, hh_df)
tax_units = constructor.create_rule_based_units()

# Analyze the results
print(f"Created {len(tax_units)} tax units")
print(tax_units[['filing_status', 'income', 'num_dependents']].head())
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
