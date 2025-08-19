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

## Understanding PUMS Weights and Sample Scaling

PUMS (Public Use Microdata Sample) data represents a sample of the population, not the complete count. Each record in the sample has an associated weight that indicates how many people/households in the population that record represents.

### Key Points About Weights:
- **Household Weights (WGTP)**: Each household in the sample has a weight indicating how many similar households it represents in the population.
- **Sample vs. Population**: The raw count of tax units in our sample (e.g., ~47,000) is much smaller than the actual population count because it's just a sample.
- **Weighted Analysis**: To get population-level estimates, always use the weight column when performing calculations.

### Example: Calculating Total Tax Units
To get the estimated total number of tax units in the population:

```python
import pandas as pd

# Load the tax units data
tax_units = pd.read_parquet('data/processed/tax_units_rule_based.parquet')

# Calculate weighted total
total_weighted_units = tax_units['weight'].sum()
print(f"Estimated total tax units in population: {total_weighted_units:,.0f}")
```

## Tax Unit Construction

This project constructs tax units from PUMS (Public Use Microdata Sample) data to model Hawaii's tax filing units. The tax unit construction process involves several key steps:

### 1. Data Preprocessing
- Person and household data is loaded from PUMS
- Each person is assigned a unique ID based on household (SERIALNO) and person number (SPORDER)
- Adults (age 18+) are identified and flagged
- Household data is merged with person records

### 2. Household Processing
For each household, the system identifies and creates different types of tax units in this priority order:

1. **Married Filing Jointly**
   - Identifies married couples living together
   - Combines their incomes and dependents into a single tax unit
   - Applies joint filing rules and income thresholds

2. **Married Filing Separately**
   - Some married couples may choose to file separately
   - Determined based on income differences, self-employment status, and other factors
   - Each spouse gets their own tax unit with their own dependents

3. **Head of Household**
   - Single adults with qualifying dependents
   - Must provide more than half the cost of maintaining the household
   - More favorable tax rates than single filers

4. **Single Filers**
   - Adults not in any other category
   - May or may not have dependents

### 3. Dependent Assignment
- Dependents are assigned to the most appropriate tax unit
- Qualifying dependents must be:
  - Under age 19 (or 24 if a student)
  - Related to the filer
  - Not filing their own return
- Special rules apply for qualifying relatives and children of divorced/separated parents

### 4. Income Calculation
- Combines all income sources for each tax unit
- Adjusts for inflation using ADJINC factor
- Handles various income types (wages, self-employment, investments, etc.)
- Calculates adjusted gross income (AGI)

### 5. Validation
- Ensures all household members are properly assigned
- Validates income calculations
- Checks for duplicate or missing information
- Verifies dependent relationships

### Output
- The final output is a DataFrame containing all tax units with their attributes:
  - Filing status
  - Income
  - Number of dependents
  - Dependents list
  - Household ID
  - Weights for population scaling

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
