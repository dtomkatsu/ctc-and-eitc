# DOTAX SOI 2022 Data Integration - Quick Start Guide

## What Was Done

✅ **Copied all DOTAX SOI 2022 files to project:**
- `data/raw/Dotax Soi 2022 - 5A.csv` - Filing Status Summary (Residents)
- `data/raw/Dotax Soi 2022 - 13A.csv` - Tax Rates: Single/MFS (Residents)
- `data/raw/Dotax Soi 2022 - 13B.csv` - Tax Rates: MFJ/QW (Residents)
- `data/raw/Dotax Soi 2022 - 13C.csv` - Tax Rates: HoH (Residents)
- `data/raw/Dotax Soi 2022 - 17A.csv` - Nonresident Tax Liability

✅ **Created comprehensive parser:**
- `src/tax/calibration/dotax_soi_parser.py` - Parses all DOTAX tables

✅ **Created documentation:**
- `docs/DOTAX_SOI_2022_INTEGRATION.md` - Integration strategy
- `docs/DOTAX_DATA_SUMMARY.md` - Complete data summary

✅ **Created validation script:**
- `scripts/validate_income_distributions.py` - Compare model to SOI

## 🚨 CRITICAL FINDING

**14.5% of Hawaii tax returns are from NONRESIDENTS** (107,992 out of 743,109 total returns)

**Implications:**
- PUMS data only captures Hawaii **residents** (635,117 returns)
- Our model should target **resident-only** benchmarks
- Nonresidents contribute $271M in tax revenue (8.8% of total)
- This partially explains income gaps in our model

## Quick Usage

### 1. Parse DOTAX Data

```python
from src.tax.calibration.dotax_soi_parser import DOTAXSOIParser

# Initialize parser
parser = DOTAXSOIParser(data_dir='data/raw')

# Get resident totals
resident_totals = parser.get_resident_totals()
print(f"Target: {resident_totals['total_returns']:,} resident returns")

# Get filing status distribution
filing_status = parser.parse_filing_status_summary()
print(filing_status)

# Get income distribution by filing status
mfs_income = parser.parse_tax_rates_by_income('single_mfs')
hoh_income = parser.parse_tax_rates_by_income('hoh')
mfj_income = parser.parse_tax_rates_by_income('mfj_qw')

# Get nonresident data
nonres_totals = parser.get_nonresident_totals()
print(f"Nonresidents: {nonres_totals['total_returns']:,}")
```

### 2. Validate Income Distributions

```bash
# Run validation script
python scripts/validate_income_distributions.py
```

This will:
- Compare model's MFS income distribution to Table 13A
- Compare model's HoH income distribution to Table 13C
- Identify which income brackets are under/over-represented
- Provide specific recommendations for fixing gaps

### 3. Test the Parser

```bash
# Run parser demo
python src/tax/calibration/dotax_soi_parser.py
```

Output shows:
- Filing status summary
- Tax rates by income for all filing statuses
- Nonresident data
- Combined totals with resident/nonresident breakdown

## Key Statistics

### Resident Filing Status (Target for PUMS Model)

| Filing Status | Returns | % | Avg AGI |
|--------------|---------|---|---------|
| Single | 335,198 | 52.8% | $42,649 |
| Married Filing Jointly | 216,358 | 34.1% | $122,682 |
| Head of Household | 67,393 | 10.6% | $55,273 |
| Married Filing Separately | 16,007 | 2.5% | $195,040 |
| **TOTAL RESIDENTS** | **635,117** | **100%** | **$75,188** |

### Nonresidents (NOT in PUMS)

| Metric | Value |
|--------|-------|
| Total Returns | 107,992 |
| % of All Returns | 14.5% |
| Tax After Credits | $271M |
| Avg Tax | $2,506 |

## How This Helps Solve Income Gaps

### MFS Income Gap (40% below SOI)

**From Table 13A, we can see:**
- Only 2.2% of Single/MFS filers earn >$150k
- But they represent significant tax revenue
- Our model may be missing these high earners

**Validation:**
```bash
python scripts/validate_income_distributions.py
```

Will show exactly which income brackets are missing.

### HoH Income Gap (34% below SOI)

**From Table 13C, we can see:**
- 14.1% of HoH filers earn >$72k
- Only 1.1% earn >$225k
- Income is concentrated in $36-54k range

**Validation:**
Same script will identify if we're missing high-income HoH filers.

## Next Steps

### Immediate (Today)

1. ✅ **DONE:** Copy DOTAX files to project
2. ✅ **DONE:** Create parser
3. ⏳ **TODO:** Run validation script:
   ```bash
   python scripts/validate_income_distributions.py
   ```
4. ⏳ **TODO:** Review income distribution gaps

### Short-Term (This Week)

5. ⏳ **TODO:** Update `SOICalibrator` to use resident-only targets
6. ⏳ **TODO:** Adjust MFS logic to capture high-income filers
7. ⏳ **TODO:** Adjust HoH logic to capture high-income filers
8. ⏳ **TODO:** Re-run validation to confirm improvements

### Medium-Term (Next Week)

9. ⏳ **TODO:** Add effective tax rate validation
10. ⏳ **TODO:** Document model scope (residents only)
11. ⏳ **TODO:** Add nonresident adjustment factor for policy analysis

## Files Reference

### Data Files
- `data/raw/Dotax Soi 2022 - 5A.csv` - Main filing status summary
- `data/raw/Dotax Soi 2022 - 13A.csv` - Single/MFS income distribution
- `data/raw/Dotax Soi 2022 - 13B.csv` - MFJ/QW income distribution
- `data/raw/Dotax Soi 2022 - 13C.csv` - HoH income distribution
- `data/raw/Dotax Soi 2022 - 17A.csv` - Nonresident data

### Code Files
- `src/tax/calibration/dotax_soi_parser.py` - Parser for all tables
- `src/tax/calibration/soi_calibration.py` - Calibration logic (needs update)
- `scripts/validate_income_distributions.py` - Validation script

### Documentation
- `docs/DOTAX_SOI_2022_INTEGRATION.md` - Detailed integration guide
- `docs/DOTAX_DATA_SUMMARY.md` - Complete data summary
- `README_DOTAX_DATA.md` - This quick start guide

## Questions?

**Q: Why are there two different total return counts?**
A: 635,117 residents + 107,992 nonresidents = 743,109 total. PUMS only has residents.

**Q: Should I calibrate to 635,117 or 743,109?**
A: **635,117** (residents only). PUMS doesn't include nonresidents.

**Q: How do I account for nonresident tax revenue?**
A: Add 8.8% adjustment factor ($271M / $3,066M) for policy analysis, or document as limitation.

**Q: Why is MFS average income so high ($195k)?**
A: High earners use MFS for tax optimization (itemized deductions, student loans, etc.)

**Q: Can I model nonresidents?**
A: Not with PUMS data. Consider synthetic population (future work) or accept limitation.

## Summary

✅ All DOTAX SOI 2022 data is now in the project
✅ Parser is ready to use
✅ Validation script is ready to run
✅ Documentation is complete

**Next action:** Run `python scripts/validate_income_distributions.py` to see exactly where the income gaps are!
