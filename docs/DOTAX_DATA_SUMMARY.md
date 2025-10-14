# Hawaii DOTAX SOI 2022 - Complete Data Summary

## Executive Summary

**CRITICAL FINDING:** 14.5% of Hawaii tax returns are from **NONRESIDENTS**. This has major implications for our PUMS-based tax modeling, as PUMS only captures Hawaii residents.

## Data Files Copied to Project

All files are now in `data/raw/`:

1. ✅ **Dotax Soi 2022 - 5A.csv** - Filing Status Summary (Residents)
2. ✅ **Dotax Soi 2022 - 13A.csv** - Tax Rates: Single/MFS (Residents)
3. ✅ **Dotax Soi 2022 - 13B.csv** - Tax Rates: MFJ/QW (Residents)
4. ✅ **Dotax Soi 2022 - 13C.csv** - Tax Rates: HoH (Residents)
5. ✅ **Dotax Soi 2022 - 17A.csv** - Nonresident Tax Liability

## Key Statistics

### Total Returns (2022)

| Category | Returns | % of Total |
|----------|---------|------------|
| **Residents** | 635,117 | 85.5% |
| **Nonresidents** | 107,992 | 14.5% |
| **TOTAL** | 743,109 | 100% |

### Resident Filing Status Distribution (Table 5A)

| Filing Status | Returns | % of Residents | Avg AGI |
|--------------|---------|----------------|---------|
| Single | 335,198 | 52.8% | $42,649 |
| Married Filing Jointly | 216,358 | 34.1% | $122,682 |
| Head of Household | 67,393 | 10.6% | $55,273 |
| Married Filing Separately | 16,007 | 2.5% | $195,040 |
| Qualifying Widow(er) | 161 | 0.0% | $55,901 |

### Tax Revenue (2022)

| Category | Before Credits | After Credits | Effective Rate |
|----------|----------------|---------------|----------------|
| **Residents** | $3,029M | $2,795M | 7.64% |
| **Nonresidents** | $300M | $271M | 9.73% |
| **TOTAL** | $3,329M | $3,066M | 7.78% |

**Key Insight:** Nonresidents have higher effective tax rates (9.73% vs 7.64%) despite representing only 14.5% of returns.

## Income Distribution by Filing Status

### Single/MFS (Table 13A) - 351,205 Returns

| Income Bracket | Returns | % of Total | Effective Rate After Credits |
|----------------|---------|------------|------------------------------|
| Under $2,400 | 82,072 | 23.4% | -41.53% (refundable credits) |
| $2,400 - $4,800 | 13,773 | 3.9% | 0.06% |
| $4,800 - $9,600 | 24,317 | 6.9% | 1.94% |
| $9,600 - $14,400 | 20,638 | 5.9% | 3.54% |
| $14,400 - $19,200 | 18,647 | 5.3% | 4.45% |
| $19,200 - $24,000 | 18,863 | 5.4% | 5.01% |
| $24,000 - $36,000 | 47,514 | 13.5% | 5.89% |
| $36,000 - $48,000 | 40,454 | 11.5% | 6.45% |
| $48,000 - $150,000 | 76,914 | 21.9% | 7.12% |
| $150,000 - $175,000 | 2,060 | 0.6% | 7.47% |
| $175,000 - $200,000 | 1,220 | 0.3% | 7.57% |
| Over $200,000 | 4,733 | 1.3% | 8.57% |

**Key Finding:** Only 2.2% of Single/MFS filers earn over $150k, but they represent a significant portion of tax revenue.

### MFJ/QW (Table 13B) - 216,358 Returns

| Income Bracket | Returns | % of Total | Effective Rate After Credits |
|----------------|---------|------------|------------------------------|
| Under $4,800 | 40,236 | 18.6% | -56.19% (refundable credits) |
| $4,800 - $9,600 | 5,632 | 2.6% | -0.47% |
| $9,600 - $19,200 | 10,798 | 5.0% | 1.64% |
| $19,200 - $28,800 | 11,317 | 5.2% | 2.96% |
| $28,800 - $38,400 | 11,443 | 5.3% | 4.20% |
| $38,400 - $48,000 | 10,914 | 5.0% | 5.10% |
| $48,000 - $72,000 | 27,330 | 12.6% | 5.80% |
| $72,000 - $96,000 | 26,226 | 12.1% | 6.30% |
| $96,000 - $300,000 | 62,358 | 28.8% | 6.97% |
| $300,000 - $350,000 | 2,401 | 1.1% | 7.30% |
| $350,000 - $400,000 | 1,588 | 0.7% | 7.41% |
| Over $400,000 | 6,115 | 2.8% | 8.11% |

**Key Finding:** 33.4% of MFJ filers earn over $96k, showing higher income concentration than single filers.

### Head of Household (Table 13C) - 67,393 Returns

| Income Bracket | Returns | % of Total | Effective Rate After Credits |
|----------------|---------|------------|------------------------------|
| Under $3,600 | 6,671 | 9.9% | -48.28% (refundable credits) |
| $3,600 - $7,200 | 2,619 | 3.9% | -2.84% |
| $7,200 - $14,400 | 5,543 | 8.2% | -0.16% |
| $14,400 - $21,600 | 5,904 | 8.8% | 0.86% |
| $21,600 - $28,800 | 7,376 | 10.9% | 2.49% |
| $28,800 - $36,000 | 7,732 | 11.5% | 4.16% |
| $36,000 - $54,000 | 13,878 | 20.6% | 5.68% |
| $54,000 - $72,000 | 7,835 | 11.6% | 6.37% |
| $72,000 - $225,000 | 9,117 | 13.5% | 6.96% |
| $225,000 - $262,500 | 176 | 0.3% | 7.49% |
| $262,500 - $300,000 | 118 | 0.2% | 7.64% |
| Over $300,000 | 424 | 0.6% | 8.15% |

**Key Finding:** HoH filers have lower average income, with 53.3% earning under $36k. Only 1.1% earn over $225k.

### Nonresidents (Table 17A) - 107,992 Returns

| AGI Bracket | Returns | % of Total | Avg Tax After Credits |
|-------------|---------|------------|-----------------------|
| Under $10,000 | 12,005 | 11.1% | $117 |
| $10,000 - $20,000 | 7,205 | 6.7% | $173 |
| $20,000 - $30,000 | 6,665 | 6.2% | $368 |
| $30,000 - $40,000 | 5,869 | 5.4% | $519 |
| $40,000 - $50,000 | 4,966 | 4.6% | $653 |
| $50,000 - $75,000 | 10,582 | 9.8% | $860 |
| $75,000 - $100,000 | 8,407 | 7.8% | $1,104 |
| $100,000 - $150,000 | 12,068 | 11.2% | $1,465 |
| $150,000 - $200,000 | 7,726 | 7.2% | $1,778 |
| $200,000 - $300,000 | 8,628 | 8.0% | $2,549 |
| $300,000 - $400,000 | 4,806 | 4.5% | $3,218 |
| Over $400,000 | 17,037 | 15.8% | $8,543 |

**Key Finding:** Nonresidents are heavily concentrated in high income brackets - 46.7% earn over $100k vs. 28.8% of resident MFJ filers.

## Implications for PUMS-Based Modeling

### ✅ What We Can Model (Residents Only)

1. **Filing Status Distribution** - PUMS captures resident households accurately
2. **Income Distribution** - PUMS has resident income data
3. **Tax Unit Construction** - Can build resident tax units from PUMS
4. **Resident Tax Liability** - Can estimate $2,795M in resident tax revenue

### ❌ What We CANNOT Model (Nonresidents)

1. **Nonresident Returns** - PUMS doesn't include people who don't live in Hawaii
2. **Nonresident Income** - Missing Hawaii-source income from non-residents
3. **Nonresident Tax Revenue** - Missing $271M (8.8% of total)
4. **Property Owners** - Second homes, investment properties owned by non-residents
5. **Remote Workers** - People working for Hawaii companies but living elsewhere

### 🎯 Recommended Approach

**For Model Validation:**
- ✅ Use **resident-only** benchmarks from Table 5A
- ✅ Target **635,117** total returns (not 743,109)
- ✅ Compare filing status distribution to resident percentages
- ✅ Validate income distributions using Tables 13A/B/C

**For Policy Analysis:**
- ⚠️ Clearly state: "Model estimates RESIDENT tax liability only"
- ⚠️ Add 8.8% adjustment factor for nonresident revenue if needed
- ⚠️ Document that 14.5% of returns are excluded (nonresidents)
- ⚠️ Consider creating synthetic nonresident population (future work)

## How to Use This Data

### 1. Parse the Data

```python
from src.tax.calibration.dotax_soi_parser import DOTAXSOIParser

parser = DOTAXSOIParser(data_dir='data/raw')

# Get resident totals
resident_totals = parser.get_resident_totals()
print(f"Target returns: {resident_totals['total_returns']:,}")

# Get filing status distribution
filing_status = parser.parse_filing_status_summary()

# Get income distribution by filing status
single_mfs_income = parser.parse_tax_rates_by_income('single_mfs')
mfj_income = parser.parse_tax_rates_by_income('mfj_qw')
hoh_income = parser.parse_tax_rates_by_income('hoh')

# Get nonresident data
nonres_totals = parser.get_nonresident_totals()
```

### 2. Update Calibration Targets

```python
# In src/tax/calibration/soi_calibration.py

class SOICalibrator:
    # Use RESIDENT-ONLY totals
    SOI_TOTAL_RETURNS = 635117  # NOT 743109!
    
    SOI_FILING_STATUS = {
        'single': 0.528,
        'joint': 0.341,
        'mfs': 0.025,
        'hoh': 0.106,
        'widow': 0.0003
    }
```

### 3. Validate Income Distributions

```python
# Compare model's income distribution to SOI
model_mfs_high_income = (tax_units[tax_units['filing_status'] == 'mfs']['agi'] > 150000).sum()
model_mfs_total = (tax_units['filing_status'] == 'mfs').sum()
model_pct_high_income = model_mfs_high_income / model_mfs_total

# From Table 13A: (2060 + 1220 + 4733) / 351205 = 2.2% of Single/MFS earn >$150k
soi_pct_high_income = 0.022

print(f"Model: {model_pct_high_income:.1%} vs SOI: {soi_pct_high_income:.1%}")
```

### 4. Document Model Scope

Add to all reports:

> **Model Scope:** This analysis models Hawaii RESIDENT tax units only, based on PUMS data. 
> Nonresidents represent an additional 14.5% of tax returns (107,992 returns) and 8.8% of 
> tax revenue ($271M after credits). The model targets 635,117 resident returns and $2,795M 
> in resident tax revenue.

## Next Steps

1. ✅ **DONE:** Copy all DOTAX files to project
2. ✅ **DONE:** Create parser for all tables
3. ✅ **DONE:** Document resident vs. nonresident distinction
4. ⏳ **TODO:** Update `SOICalibrator` to use resident-only targets
5. ⏳ **TODO:** Add income distribution validation
6. ⏳ **TODO:** Investigate MFS and HoH income gaps using Tables 13A/C
7. ⏳ **TODO:** Add effective tax rate validation
8. ⏳ **TODO:** Consider synthetic nonresident population (optional)

## Files Created

- ✅ `data/raw/Dotax Soi 2022 - 5A.csv`
- ✅ `data/raw/Dotax Soi 2022 - 13A.csv`
- ✅ `data/raw/Dotax Soi 2022 - 13B.csv`
- ✅ `data/raw/Dotax Soi 2022 - 13C.csv`
- ✅ `data/raw/Dotax Soi 2022 - 17A.csv`
- ✅ `src/tax/calibration/dotax_soi_parser.py`
- ✅ `docs/DOTAX_SOI_2022_INTEGRATION.md`
- ✅ `docs/DOTAX_DATA_SUMMARY.md`
