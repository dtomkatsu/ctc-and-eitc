# Calibration Data Comparison: A-2 vs A-9

## Overview

You now have **two calibration options** for Hawaii tax data:

1. **Table A-9**: More granular detail (15 brackets), 90.3% coverage
2. **Table A-2**: Complete coverage (16 brackets), 100% of returns

## Data Summary

### Table A-9 (Detailed, Partial Coverage)

**Source**: "Selected Resident Return Data with Hawai'i AGI Under $150,000"

| Filing Status | Brackets | Returns | Coverage | Total AGI | Total Tax |
|---------------|----------|---------|----------|-----------|-----------|
| **Joint** | 15 | 166,461 | 76.9% | $9.9B | $446M |
| **Single** | 15 | 341,399 | 97.2% | $11.5B | $623M |
| **HoH** | 15 | 65,393 | 96.8% | $3.0B | $128M |
| **TOTAL** | 45 | 573,253 | **90.3%** | $24.5B | $1,197M |

**Key Features**:
- ✅ More granular brackets (15 per status)
- ✅ Detailed low-income coverage (8 brackets under $50k)
- ✅ Tax liability before and after credits
- ❌ Only covers AGI < $150k
- ❌ Missing 9.7% of returns (high-income)

**Best for**:
- Low/middle-income policy analysis
- Tax credit modeling (EITC, CTC, etc.)
- Detailed income distribution studies

---

### Table A-2 (Complete Coverage)

**Source**: "Selected Data from Resident Tax Returns by Filing Status and Hawai'i AGI Class"

| Filing Status | Brackets | Returns | Coverage | Total AGI | Total Tax |
|---------------|----------|---------|----------|-----------|-----------|
| **Joint** | 16 | 216,358 | 100% | $26.1B | $1,551M |
| **Single** | 16 | 351,205 | 100% | $17.2B | $1,067M |
| **HoH** | 16 | 67,554 | 100% | $3.7B | $177M |
| **TOTAL** | 48 | 635,117 | **100%** | $47.0B | $2,795M |

**Key Features**:
- ✅ Complete coverage (100% of returns)
- ✅ Includes all income levels
- ✅ Covers high-income brackets (AGI ≥ $150k)
- ✅ Taxable and nontaxable returns
- ❌ Broader brackets (less granular)
- ❌ Less detail for low-income ranges

**Best for**:
- Complete state-wide estimates
- High-income policy analysis
- Revenue estimation
- When 100% coverage is required

---

## Detailed Comparison

### Bracket Structure

**Table A-9 (AGI < $150k only)**:
```
Loss to $0
$0
$1 - $1k
$1k - $5k
$5k - $10k
$10k - $15k
$15k - $20k
$20k - $30k
$30k - $40k
$40k - $50k
$50k - $60k
$60k - $75k
$75k - $100k
$100k - $125k
$125k - $150k
```

**Table A-2 (All income levels)**:
```
TAXABLE:
$0 - $10k
$10k - $20k
$20k - $30k
$30k - $40k
$40k - $50k
$50k - $75k
$75k - $100k
$100k - $150k
$150k - $200k    ← Additional
$200k - $300k    ← Additional
$300k - $400k    ← Additional
$400k and over   ← Additional

NONTAXABLE:
Loss
$0 - $5k
$5k - $10k
$10k and over
```

### Coverage Gap Analysis

**What A-9 is Missing** (9.7% of returns, ~62,000 returns):

| Filing Status | Missing Returns | Missing AGI | % of Status |
|---------------|-----------------|-------------|-------------|
| **Joint** | 49,897 | $16.2B | 23.1% |
| **Single** | 9,806 | $5.7B | 2.8% |
| **HoH** | 2,161 | $0.7B | 3.2% |
| **TOTAL** | 61,864 | $22.5B | 9.7% |

**Key Finding**: Joint filers have the largest gap because high-income earners are more likely to file jointly.

---

## Calibration Accuracy

### Expected IPF Performance

**Table A-9** (45 categories):
- Total error: < 0.1%
- Per-bracket error: < 1%
- Convergence: ~30-50 iterations

**Table A-2** (48 categories):
- Total error: < 0.1%
- Per-bracket error: < 2% (broader brackets)
- Convergence: ~30-50 iterations

---

## Recommendation Matrix

| Use Case | Recommended | Reason |
|----------|-------------|--------|
| **Tax credit analysis** (EITC, CTC) | Table A-9 | More detail in low-income ranges |
| **Revenue estimation** (all brackets) | Table A-2 | Complete coverage needed |
| **High-income policy** (AGI > $150k) | Table A-2 | Only source with high-income data |
| **Distributional analysis** (detailed) | Table A-9 | More granular brackets |
| **State-wide totals** | Table A-2 | 100% coverage required |
| **Research/academic** | Both | Use A-9 for detail, A-2 for validation |

---

## Implementation

### Using Table A-9 (Detailed)

```python
from src.tax.validation.detailed_tax_calibration import apply_detailed_tax_calibration

# More granular, 90.3% coverage
tax_units = apply_detailed_tax_calibration(
    tax_units,
    weight_col='weight',
    agi_col='agi'
)
```

**Output**: `weight_detailed_calibrated` (90.3% of returns)

### Using Table A-2 (Complete)

```python
from src.tax.validation.comprehensive_tax_calibration import apply_comprehensive_calibration

# Complete coverage, 100% of returns
tax_units = apply_comprehensive_calibration(
    tax_units,
    weight_col='weight',
    agi_col='agi'
)
```

**Output**: `weight_comprehensive_calibrated` (100% of returns)

---

## Hybrid Approach

For maximum accuracy, you could use a **hybrid approach**:

1. **Use A-9 for AGI < $150k** (more granular)
2. **Use A-2 for AGI ≥ $150k** (only source)

This would give you:
- 15 detailed brackets for < $150k (90.3% of returns)
- 4 additional brackets for ≥ $150k (9.7% of returns)
- Total: 19 brackets per filing status

However, this adds complexity and the benefit is marginal since:
- A-2 already provides good coverage
- High-income brackets have fewer policy implications
- The difference in low-income brackets is modest

**Recommendation**: Start with **Table A-2** for simplicity and complete coverage. Use **Table A-9** if you need more granular low-income analysis for specific policy questions.

---

## Files Available

### Data Files
1. **A-9 Benchmarks**: `data/processed/detailed_tax_liability_benchmarks.csv`
   - 45 brackets, 90.3% coverage
   
2. **A-2 Benchmarks**: `data/processed/comprehensive_tax_benchmarks.csv`
   - 48 brackets, 100% coverage

### Code Modules
1. **A-9 Calibration**: `src/tax/validation/detailed_tax_calibration.py`
2. **A-2 Calibration**: (to be created) `src/tax/validation/comprehensive_tax_calibration.py`

### Documentation
1. **A-9 Details**: `docs/DETAILED_TAX_CALIBRATION.md`
2. **Comparison** (this file): `docs/CALIBRATION_DATA_COMPARISON.md`
3. **Summary**: `docs/CALIBRATION_UPDATE_SUMMARY.md`

---

## Quick Decision Guide

**Question 1**: Do you need to analyze high-income taxpayers (AGI > $150k)?
- **Yes** → Use Table A-2
- **No** → Continue to Question 2

**Question 2**: Do you need very detailed low-income brackets?
- **Yes** → Use Table A-9
- **No** → Use Table A-2 (simpler, complete)

**Question 3**: Is this for official state revenue estimates?
- **Yes** → Use Table A-2 (100% coverage required)
- **No** → Either works

---

## Summary

Both tables are excellent for calibration. The choice depends on your specific needs:

- **Table A-9**: Best for detailed policy analysis of low/middle-income households
- **Table A-2**: Best for complete state-wide estimates and simplicity

For most use cases, **Table A-2 is recommended** because:
1. ✅ Complete coverage (100%)
2. ✅ Simpler (one source)
3. ✅ Includes high-income data
4. ✅ Still provides good granularity

Use **Table A-9** when you specifically need the additional detail in low-income brackets for tax credit or distributional analysis.
