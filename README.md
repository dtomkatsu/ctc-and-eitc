# CTC and EITC Analysis for Hawaii

This project analyzes the Child Tax Credit (CTC) and Earned Income Tax Credit (EITC) using PUMS microdata for Hawaii at various geographic levels (state, county, and legislative district).

## Project Structure

```
ctc-and-eitc/
├── data/                   # Data storage
│   ├── raw/               # Raw data files (e.g., PUMS microdata)
│   └── processed/         # Processed and cleaned data
├── notebooks/             # Jupyter notebooks for analysis
├── src/                   # Source code
│   ├── data/              # Data processing scripts
│   ├── analysis/          # Analysis scripts
│   └── visualization/     # Visualization scripts
├── config/                # Configuration files
└── docs/                  # Documentation
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

## Data Sources

- PUMS microdata from the U.S. Census Bureau
- Geographic boundary files for Hawaii

## Usage

[Instructions will be added as the project develops]
