---
name: ctc-and-eitc
description: CTC and EITC tax data pipeline and analysis
---

# CTC and EITC Data Pipeline

Analyzes Child Tax Credit (CTC) and Earned Income Tax Credit (EITC) distribution and impact using Hawaii tax microdata.

## Project Overview

- **Purpose**: Model and analyze CTC/EITC distribution across Hawaii households, inform policy on tax credits
- **Data source**: Hawaii DOTAX SOI data, PUMS microdata
- **Pipeline**: Monthly DOTAX scraper, tax model calibration, analysis reports
- **Location**: `/Users/devinthomas/ctc-and-eitc/`

## Key Files

- `src/tax/` — Tax calculation and modeling
- `src/data/` — Data loading and processing
- `logs/pipeline/dotax_scrape.log` — Monthly scraper logs
- `data/` — Raw and processed datasets

## Recurring Tasks

- **Monthly scrape**: DOTAX data pull via automated pipeline (check logs for errors)
- **Model calibration**: Adjust tax model to match official revenue benchmarks
- **Analysis**: Generate reports on credit distribution by income, geography, household type

## Key Metrics to Track

- Total CTC and EITC revenue (Hawaii resident only)
- Distribution across income brackets
- Impact on effective tax rates
- Regional variation (island, district, county)

## Important Notes

- Model estimates **resident-only revenue** (~91% of total)
- Non-resident revenue (8.8%) not captured by PUMS — must be noted in reports
- Revenue estimates as ranges with uncertainty bounds
- All model outputs require comparison to DOTAX official benchmarks

## Output Style

- Data-driven, show methodology
- Always include uncertainty/caveats
- Compare model output to DOTAX official figures
- Highlight gaps or discrepancies for investigation
