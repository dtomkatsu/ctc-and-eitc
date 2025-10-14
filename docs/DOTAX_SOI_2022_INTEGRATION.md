# Hawaii DOTAX SOI 2022 Data Integration Guide

## Overview

This document explains how to incorporate Hawaii Department of Taxation (DOTAX) Statistics of Income (SOI) 2022 data into the tax modeling system. The data provides critical benchmarks for validating and calibrating our PUMS-based tax unit construction.

## Data Files

All files are located in `data/raw/`:

### Table 5A: Filing Status Summary (Residents Only)
**File:** `Dotax Soi 2022 - 5A.csv`

**Content:**
- Number of returns by filing status
- AGI (positive and negative) by filing status
- Tax liability (before and after credits) by filing status

**Key Metrics:**
- Total Resident Returns: **635,117**
- Filing Status Distribution:
  - Married Filing Jointly: 34.1% (216,358)
  - Single: 52.8% (335,198)
  - Married Filing Separately: 2.5% (16,007)
  - Head of Household: 10.6% (67,393)
  - Qualifying Widow(er): 0.0% (161)

### Table 13A: Tax Rates - Single/MFS (Residents Only)
**File:** `Dotax Soi 2022 - 13A.csv`

**Content:**
- Income distribution by taxable income brackets
- Marginal tax rates by bracket
- Average effective tax rates (before and after credits)

**Key Insights:**
- Total Single + MFS Returns: **351,205**
- Income distribution shows concentration in $24k-$150k range
- Average effective rate after credits: **7.07%**

### Table 13B: Tax Rates - MFJ/QW (Residents Only)
**File:** `Dotax Soi 2022 - 13B.csv`

**Content:**
- Income distribution for joint filers
- Marginal and effective tax rates

**Key Insights:**
- Total MFJ + QW Returns: **216,358**
- Higher income concentration than single filers
- Average effective rate after credits: **6.94%**

### Table 13C: Tax Rates - HoH (Residents Only)
**File:** `Dotax Soi 2022 - 13C.csv`

**Content:**
- Income distribution for Head of Household filers
- Tax rates by income bracket

**Key Insights:**
- Total HoH Returns: **67,393**
- Lower average income than other filing statuses
- Average effective rate after credits: **5.79%**
- Significant concentration in lower income brackets

### Table 17A: Nonresident Tax Liability
**File:** `Dotax Soi 2022 - 17A.csv`

**Content:**
- Number of nonresident returns by AGI bracket
- Tax liability before and after credits
- Average tax per return

**CRITICAL INSIGHT:**
- Total Nonresident Returns: **107,992**
- **14.5% of all Hawaii tax returns are from NONRESIDENTS**
- This is a major factor our PUMS-based model cannot capture!

## Key Findings: Residents vs. Nonresidents

### Combined Totals (2022)

| Metric | Residents | Nonresidents | Total | % Nonresident |
|--------|-----------|--------------|-------|---------------|
| Returns | 635,117 | 107,992 | 743,109 | **14.5%** |
| Tax Before Credits | $3,029M | $300M | $3,329M | 9.0% |
| Tax After Credits | $2,795M | $271M | $3,066M | 8.8% |

### Why This Matters

**PUMS data only captures Hawaii RESIDENTS**, not nonresidents who:
- Own property in Hawaii but live elsewhere
- Have Hawaii-source income (rental, business, investments)
- Are military stationed in Hawaii but claim residency elsewhere
- Work remotely for Hawaii companies

This explains some of our model discrepancies:
1. **Total Returns**: Our model should target **635,117** (residents only), not 743,109
2. **Filing Status Distribution**: Should be based on resident returns only
3. **Income Distributions**: Nonresidents skew higher income (avg tax $2,506 vs residents)

## Integration Strategy

### Phase 1: Update Calibration Targets (IMMEDIATE)

**Current Issue:** We're calibrating to totals that include nonresidents

**Fix:** Update `SOICalibrator` to use **resident-only** benchmarks:

```python
# In src/tax/calibration/soi_calibration.py

class SOICalibrator:
    # OLD (includes nonresidents):
    SOI_TOTAL_RETURNS = 743109  # WRONG
    
    # NEW (residents only):
    SOI_TOTAL_RETURNS = 635117  # CORRECT
    
    # Filing status distribution (from Table 5A - residents only)
    SOI_FILING_STATUS = {
        'single': 0.528,      # 335,198 / 635,117
        'joint': 0.341,       # 216,358 / 635,117
        'mfs': 0.025,         # 16,007 / 635,117
        'hoh': 0.106,         # 67,393 / 635,117
        'widow': 0.0003       # 161 / 635,117
    }
```

### Phase 2: Income Distribution Calibration (HIGH PRIORITY)

**Use Tables 13A/B/C to validate income distributions:**

```python
# Example: Validate MFS income distribution
parser = DOTAXSOIParser()
mfs_rates = parser.parse_tax_rates_by_income('single_mfs')

# Check if our model's MFS filers match SOI income distribution
# Expected: 29.3% of MFS earn >$150k (from Table 13A)
# Current model: Need to verify
```

**Action Items:**
1. Parse income brackets from Tables 13A/B/C
2. Compare model's income distribution to SOI by filing status
3. Identify which income brackets are under/over-represented
4. Adjust MFS/HoH identification logic to capture high-income filers

### Phase 3: Effective Tax Rate Validation (MEDIUM PRIORITY)

**Use effective tax rates to validate our tax calculations:**

From the tables:
- Single/MFS: 7.07% effective rate after credits
- MFJ/QW: 6.94% effective rate after credits
- HoH: 5.79% effective rate after credits

**Validation:**
```python
# Calculate effective rates from our model
model_effective_rate = (tax_after_credits / agi).mean()

# Compare to SOI benchmarks
soi_effective_rates = {
    'single_mfs': 0.0707,
    'mfj_qw': 0.0694,
    'hoh': 0.0579
}
```

### Phase 4: Nonresident Modeling (FUTURE/OPTIONAL)

**Challenge:** PUMS doesn't include nonresidents

**Options:**
1. **Accept the limitation**: Model residents only, document clearly
2. **Synthetic nonresident population**: 
   - Create synthetic tax units based on Table 17A distribution
   - Weight to match 107,992 nonresident returns
   - Use simplified tax calculation (no credits, higher income)
3. **Adjustment factor**: 
   - Scale up total tax revenue by 8.8% to account for nonresidents
   - Document that return counts are residents only

**Recommendation:** Option 1 (accept limitation) for now, Option 3 for policy analysis

## Implementation Checklist

- [x] Copy all DOTAX SOI files to `data/raw/`
- [x] Create `DOTAXSOIParser` class
- [ ] Update `SOICalibrator` to use resident-only targets
- [ ] Add income distribution validation by filing status
- [ ] Add effective tax rate validation
- [ ] Update documentation to clarify resident vs. nonresident scope
- [ ] Add nonresident adjustment factor for policy analysis (optional)

## Usage Example

```python
from src.tax.calibration.dotax_soi_parser import DOTAXSOIParser

# Initialize parser
parser = DOTAXSOIParser(data_dir='data/raw')

# Get resident totals
resident_totals = parser.get_resident_totals()
print(f"Total resident returns: {resident_totals['total_returns']:,}")

# Get filing status distribution
filing_status_df = parser.parse_filing_status_summary()
print(filing_status_df)

# Get income distribution for MFS filers
mfs_income = parser.parse_tax_rates_by_income('single_mfs')
print(mfs_income)

# Get nonresident statistics
nonres_totals = parser.get_nonresident_totals()
print(f"Nonresident returns: {nonres_totals['total_returns']:,}")

# Get combined totals
combined = parser.get_combined_totals()
print(f"Nonresident percentage: {combined['pct_nonresident']:.1%}")
```

## Critical Insights for Model Validation

### 1. **Our Model Should Target Residents Only**
- PUMS = Hawaii residents
- SOI Table 5A = Hawaii residents
- SOI Table 17A = Nonresidents (separate)
- **Do NOT mix resident and nonresident benchmarks**

### 2. **Income Gaps May Be Partially Explained**
- Nonresidents have higher average income
- If we were accidentally calibrating to combined totals, this would explain some income gaps
- Need to verify we're using resident-only benchmarks

### 3. **Filing Status Distribution is Resident-Only**
- Our current SOI targets (from Table 5A) are already resident-only ✅
- This is correct and should not change

### 4. **Tax Revenue Estimates Need Adjustment**
- Our model estimates resident tax revenue only
- For policy analysis, add 8.8% to account for nonresident revenue
- Or scale by (635,117 / 743,109) = 85.5% if using combined benchmarks

## Next Steps

1. **Run the parser** to validate all data loads correctly:
   ```bash
   python src/tax/calibration/dotax_soi_parser.py
   ```

2. **Update SOI calibrator** to ensure resident-only targets

3. **Add income distribution validation** to identify which income brackets are missing

4. **Document model scope** clearly: "This model estimates Hawaii RESIDENT tax liability only. Nonresidents represent an additional 14.5% of returns and 8.8% of tax revenue."

## References

- Hawaii DOTAX SOI 2022 Tables: `data/raw/Dotax Soi 2022 - *.csv`
- Parser Implementation: `src/tax/calibration/dotax_soi_parser.py`
- Calibration Module: `src/tax/calibration/soi_calibration.py`
