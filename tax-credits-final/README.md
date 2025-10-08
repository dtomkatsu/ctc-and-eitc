# Tax Credits Final

This project focuses on analyzing and adjusting IRS SOI (Statistics of Income) data for tax credit analysis, specifically for the Child Tax Credit (CTC) and Earned Income Tax Credit (EITC) at the district level in Hawaii.

## SOI Data Structure

### Main SOI Data Files
1. **Individual Income Tax Returns (SOI)**: Contains tax return data including:
   - Filing status
   - Adjusted Gross Income (AGI)
   - Number of dependents
   - Tax credits claimed (CTC, EITC, etc.)
   - Income sources
   - Demographic information (limited)

2. **ZIP Code Data (ZIP-AGI)**:
   - Number of returns by AGI range
   - Number of exemptions
   - Wages and salaries
   - Taxable interest
   - Tax credits claimed (CTC, EITC)

3. **County Data**:
   - Similar to ZIP data but aggregated at county level
   - Includes additional demographic breakdowns

### Key Variables for CTC/EITC Analysis
- **CTC Related**:
  - `nctc`: Number of CTC claims
  - `actc`: Total CTC amount claimed
  - `nctcref`: Number of refundable CTC claims
  - `actcref`: Total refundable CTC amount
  - `nctcnon`: Number of non-refundable CTC claims
  - `actcnoni`: Total non-refundable CTC amount

- **EITC Related**:
  - `eitc`: Total EITC amount
  - `nret`: Number of returns claiming EITC
  - `nchild`: Number of qualifying children for EITC

### Data Hierarchy
1. **National Level** → **State Level** → **County Level** → **ZIP Code Level**
2. **Aggregation Levels**:
   - By AGI ranges
   - By filing status (Single, HoH, MFJ, MFS)
   - By number of dependents

## Project Structure

```
tax-credits-final/
├── data/                    # Data directory
│   ├── raw/                 # Original data files
│   └── processed/           # Cleaned and processed data
├── docs/                    # Documentation
├── notebooks/               # Jupyter notebooks for analysis
├── scripts/                 # Python scripts for data processing
└── src/                     # Source code
    ├── data/                # Data loading and processing
    ├── analysis/            # Analysis modules
    └── visualization/       # Visualization tools
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

3. Place SOI data files in `data/raw/`

## Usage

1. Run data processing pipeline:
   ```bash
   python scripts/process_soi_data.py
   ```

2. Run analysis:
   ```bash
   python scripts/analyze_credits.py
   ```

3. Generate reports:
   ```bash
   python scripts/generate_reports.py
   ```

## Data Sources
- IRS SOI Data: [https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics-zip-code-data-soi](https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics-zip-code-data-soi)
- Hawaii State Data Book: [http://dbedt.hawaii.gov/economic/databook/](http://dbedt.hawaii.gov/economic/databook/)

## License
This project is licensed under the MIT License - see the LICENSE file for details.
