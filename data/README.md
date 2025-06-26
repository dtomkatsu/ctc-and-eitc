# Data Directory

This directory contains the raw and processed data for the CTC and EITC analysis project.

## Directory Structure

```
data/
├── raw/                   # Raw data files (not version controlled)
│   └── pums/             # PUMS microdata files
│       ├── psam_h15.csv   # Hawaii household records
│       └── psam_p15.csv   # Hawaii person records
└── processed/             # Processed and cleaned data
```

## Data Sources

### PUMS Data

1. **Source**: U.S. Census Bureau's American Community Survey (ACS) Public Use Microdata Sample (PUMS)
2. **Website**: [https://www.census.gov/programs-surveys/acs/microdata.html](https://www.census.gov/programs-surveys/acs/microdata.html)
3. **Files Needed**:
   - `psam_h15.csv` - Household records for Hawaii
   - `psam_p15.csv` - Person records for Hawaii

### How to Download PUMS Data

1. Visit the [Census Bureau's PUMS data page](https://www.census.gov/programs-surveys/acs/microdata.html)
2. Select the desired year
3. Download the 1-year or 5-year estimates for Hawaii (state code 15)
4. Place the files in the `data/raw/pums/` directory

### Expected File Structure

After downloading, your data directory should look like this:

```
data/raw/pums/
├── psam_h15.csv    # Household records
└── psam_p15.csv    # Person records
```

## Data Processing

Raw data is processed using the `PUMSDataLoader` class in `src/data/pums_loader.py`. This class handles:

- Loading and merging household and person records
- Adjusting income values using the ADJINC factor
- Creating tax units for analysis
- Calculating derived variables for CTC/EITC eligibility

## Notes

- Raw data files are not version controlled (see `.gitignore`)
- Processed data files should be saved in the `processed/` directory
- Always document any transformations or cleaning steps applied to the data
