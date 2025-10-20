# Detailed Tax Liability Calibration

## Overview

The detailed tax liability calibration uses **DOTAX SOI 2022 Table A-9** (Selected Resident Return Data) instead of Table 12A. This provides more granular AGI brackets and includes actual tax liability data for more accurate microsimulation.

## Why Use Detailed Calibration?

### Table A-9 vs Table 12A

| Feature | Table 12A (Old) | Table A-9 (New) |
|---------|-----------------|-----------------|
| **AGI Brackets** | 12 broad brackets | 15 detailed brackets |
| **Granularity** | Standard IRS brackets | More detailed |
| **Tax Liability** | Not included | Included (before/after credits) |
| **Coverage** | All returns | AGI < $150k (90.3%) |
| **Filing Status Detail** | Basic | Detailed breakdown |
| **Effective Rates** | Not provided | Provided by bracket |

### Key Advantages

1. **More Granular Brackets**: 15 brackets per filing status provides better income distribution
2. **Tax Liability Data**: Actual tax liability by bracket for validation
3. **Better Low-Income Coverage**: More detailed brackets under $50k
4. **Effective Rates**: Can validate tax calculations against actual rates

### Example: Joint Filers

**Table 12A** (broad):
- $0-$10k
- $10k-$25k
- $25k-$50k
- $50k-$75k
- ...

**Table A-9** (detailed):
- $0-$1k
- $1k-$5k
- $5k-$10k
- $10k-$15k
- $15k-$20k
- $20k-$30k
- $30k-$40k
- $40k-$50k
- $50k-$60k
- $60k-$75k
- ...

## Data Source

**DOTAX SOI 2022 Table A-9**: "Selected Data from Resident Tax Returns with Hawai'i AGI Under $150,000 by Filing Status"

Three separate tables by filing status:
- **A9-2**: Joint filers (Married Filing Jointly) - 166,461 returns
- **A9-3**: Single filers (includes Single + MFS) - 341,399 returns
- **A9-4**: Head of Household (includes QW) - 65,393 returns

**Total Coverage**: 573,253 returns (90.3% of all Hawaii returns)

## Implementation

### 1. Data Parsing

Run the parser to extract benchmarks:

```bash
python scripts/parse_detailed_tax_liability.py
```

This creates: `data/processed/detailed_tax_liability_benchmarks.csv`

### 2. Apply Calibration

```python
from src.tax.validation.detailed_tax_calibration import apply_detailed_tax_calibration

# Load tax units
tax_units = pd.read_parquet('data/processed/tax_units.parquet')

# Apply detailed calibration
tax_units_calibrated = apply_detailed_tax_calibration(
    tax_units,
    weight_col='weight',
    agi_col='agi'
)

# New column: weight_detailed_calibrated
```

### 3. Validate Results

```python
from src.tax.validation.detailed_tax_calibration import validate_detailed_calibration

validation = validate_detailed_calibration(
    tax_units_calibrated,
    weight_col='weight_detailed_calibrated'
)

# Check accuracy
print(f"Total error: {validation['total_error_pct']:.3f}%")
print(f"Max bracket error: {validation['max_error_pct']:.3f}%")
```

## Benchmark Summary

### By Filing Status

| Filing Status | Returns | Total AGI | Total Tax | Avg AGI | Avg Tax | Eff. Rate |
|---------------|---------|-----------|-----------|---------|---------|-----------|
| **Joint** | 166,461 | $9.9B | $446M | $59,331 | $2,680 | 4.5% |
| **Single** | 341,399 | $11.5B | $623M | $33,806 | $1,824 | 5.4% |
| **HoH** | 65,393 | $3.0B | $128M | $46,585 | $1,960 | 4.2% |
| **TOTAL** | 573,253 | $24.5B | $1,197M | $42,675 | $2,088 | 4.9% |

## Detailed AGI Brackets

### Joint Filers (15 brackets)

| AGI Bracket | Returns | Avg Tax | Eff. Rate |
|-------------|---------|---------|-----------|
| Loss to $0 | 5,055 | -$274 | n/a |
| $0 | 3,641 | -$201 | n/a |
| $1-$1k | 5,806 | -$174 | n/a |
| $1k-$5k | 6,274 | -$183 | n/a |
| $5k-$10k | 6,374 | -$187 | n/a |
| $10k-$15k | 6,181 | -$163 | -8.4% |
| $15k-$20k | 6,043 | -$85 | -1.7% |
| $20k-$30k | 11,515 | $125 | 1.1% |
| $30k-$40k | 11,769 | $547 | 2.7% |
| $40k-$50k | 11,478 | $1,195 | 4.0% |
| $50k-$60k | 10,984 | $1,971 | 5.0% |
| $60k-$75k | 15,637 | $2,833 | 5.6% |
| $75k-$100k | 25,352 | $4,152 | 6.1% |
| $100k-$125k | 22,833 | $5,746 | 6.4% |
| $125k-$150k | 17,519 | $7,367 | 6.7% |

### Single Filers (15 brackets)

Similar structure with 15 brackets from negative AGI to $150k.

### Head of Household (15 brackets)

Similar structure with 15 brackets from negative AGI to $150k.

## Coverage and Limitations

### What's Covered (90.3%)

- **All returns with AGI < $150k**: 573,253 returns
- **Detailed calibration**: 15 brackets per filing status
- **Tax liability data**: Actual tax before/after credits

### What's Not Covered (9.7%)

- **Returns with AGI ≥ $150k**: ~62,000 returns (estimated)
- **High-income returns**: Retain original PUMS weights
- **Non-residents**: Not included in Table A-9

This is acceptable because:
1. 90.3% coverage is excellent for microsimulation
2. High-income returns have better PUMS representation
3. Low-income calibration is more important for policy analysis

## Calibration Method

Uses **Iterative Proportional Fitting (IPF)** to adjust weights:

1. **Categories**: filing_status × AGI_bracket (45 categories total)
2. **Targets**: DOTAX SOI return counts by category
3. **Iterations**: Typically converges in 30-50 iterations
4. **Tolerance**: < 0.1% error per bracket

### IPF Process

```
Initial weights → IPF adjustment → Calibrated weights

45 categories:
- joint × 15 brackets = 15 categories
- single × 15 brackets = 15 categories  
- hoh × 15 brackets = 15 categories
```

## Validation

### Expected Accuracy

- **Total returns**: < 0.1% error
- **Per-bracket**: < 1% error typically
- **Filing status**: Near-perfect match
- **AGI distribution**: Highly accurate

### Validation Checks

1. **Total returns**: Should match DOTAX total (573,253)
2. **Filing status distribution**: Joint, Single, HoH percentages
3. **AGI distribution**: Returns per bracket
4. **Tax liability**: Can validate calculated tax against benchmarks

## Use Cases

### 1. Tax Policy Analysis

More accurate for analyzing:
- Low-income tax credits (EITC, CTC)
- Bracket-specific policy changes
- Distributional effects

### 2. Revenue Estimation

Better estimates for:
- State tax revenue
- Credit costs
- Policy reforms

### 3. Tax Liability Validation

Can validate microsimulation against actual tax data:
```python
# Calculate tax
tax_units['calculated_tax'] = calculate_taxes(tax_units)

# Compare to benchmark averages
benchmarks = load_detailed_benchmarks()
for bracket in benchmarks:
    actual_avg = bracket['avg_tax_after']
    model_avg = calculate_model_avg(tax_units, bracket)
    error = abs(model_avg - actual_avg) / actual_avg
```

## Comparison: Old vs New Calibration

### Old Approach (Table 12A)

```python
from src.tax.validation.irs_soi_calibration import apply_irs_soi_calibration

tax_units = apply_irs_soi_calibration(tax_units)
# Uses 12 broad AGI brackets
# No tax liability data
```

**Pros**:
- Covers all income ranges
- Simple to implement
- Well-documented

**Cons**:
- Broader brackets (less granular)
- No tax liability validation
- Less detail for low-income analysis

### New Approach (Table A-9)

```python
from src.tax.validation.detailed_tax_calibration import apply_detailed_tax_calibration

tax_units = apply_detailed_tax_calibration(tax_units)
# Uses 15 detailed AGI brackets
# Includes tax liability data
```

**Pros**:
- More granular (15 brackets)
- Tax liability data included
- Better low-income coverage
- Effective rate validation

**Cons**:
- Only covers AGI < $150k
- Slightly more complex
- Newer implementation

## Recommendation

**Use detailed calibration for:**
- Policy analysis focused on low/middle income
- Tax credit analysis (EITC, CTC)
- Revenue estimation
- When tax liability validation is important

**Use standard calibration (Table 12A) for:**
- High-income policy analysis
- When 100% coverage is required
- Quick analyses
- When simplicity is preferred

## Example Usage

### Complete Pipeline

```python
import pandas as pd
from src.tax.validation.detailed_tax_calibration import (
    apply_detailed_tax_calibration,
    validate_detailed_calibration,
    get_benchmark_summary
)

# 1. Load tax units
tax_units = pd.read_parquet('data/processed/tax_units.parquet')

# 2. Apply detailed calibration
tax_units_calibrated = apply_detailed_tax_calibration(
    tax_units,
    weight_col='weight',
    agi_col='agi'
)

# 3. Validate
validation = validate_detailed_calibration(
    tax_units_calibrated,
    weight_col='weight_detailed_calibrated'
)

# 4. View benchmark summary
summary = get_benchmark_summary()
print(summary)

# 5. Use calibrated weights for analysis
total_tax = (
    tax_units_calibrated['tax_liability'] * 
    tax_units_calibrated['weight_detailed_calibrated']
).sum()
```

### Test Script

```bash
# Run test to see detailed calibration in action
python scripts/test_detailed_calibration.py
```

This will:
- Load detailed benchmarks
- Apply calibration to tax units
- Validate accuracy
- Show detailed results

## Files

### Data Files
- `data/raw/Selected Resident Return Data/Dotax Soi 2022 - A9-2.csv` - Joint filers
- `data/raw/Selected Resident Return Data/Dotax Soi 2022 - A9-3.csv` - Single filers
- `data/raw/Selected Resident Return Data/Dotax Soi 2022 - A9-4.csv` - HoH filers
- `data/processed/detailed_tax_liability_benchmarks.csv` - Parsed benchmarks

### Code Files
- `src/tax/validation/detailed_tax_calibration.py` - Main module
- `scripts/parse_detailed_tax_liability.py` - Data parser
- `scripts/test_detailed_calibration.py` - Test script

### Documentation
- `docs/DETAILED_TAX_CALIBRATION.md` - This file

## References

- **DOTAX SOI 2022**: Hawaii Department of Taxation Statistics of Income
- **Table A-9**: Selected Data from Resident Tax Returns (AGI < $150k)
- **IPF Method**: Iterative Proportional Fitting (raking)
