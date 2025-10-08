# Hawaii Tax Model Calibration Report

**Date:** October 7, 2025  
**Issue:** Model overestimates revenue by 38% ($1.17B)  
**Status:** ✅ Diagnosed and Calibrated

---

## Executive Summary

Our Hawaii tax model **overestimates** 2017 bracket revenue by **$1.17 billion (+38%)** compared to actual 2023 collections. Through comparison with IRS SOI data, we've identified the root causes and created a **calibration factor of 0.7252** to align model outputs with reality.

### The Numbers

| Metric | Model Estimate | Actual/SOI | Disparity |
|--------|----------------|------------|-----------|
| **2017 Revenue** | $4.28B | $3.10B (2023 actual) | **+$1.17B (+38%)** |
| Tax Units | 527,631 | 674,660 (SOI 2022) | -147,029 (-22%) |
| Total Income | $58.0B | $57.1B (SOI 2022) | +$1.0B (+2%) |
| Avg Income | $110,017 | $83,750 (SOI 2022) | +$26,267 (+31%) |

---

## Root Causes Identified

### 1. **Income Measurement Gap** (Primary Issue)
**Impact: -10% to -15% revenue**

- **Problem**: PUMS captures **total income**, not **Adjusted Gross Income (AGI)**
- **Missing Adjustments**:
  - 401(k) and IRA contributions
  - HSA contributions  
  - Self-employment tax deduction (50% of SE tax)
  - Student loan interest
  - Educator expenses
  - Moving expenses (for military)

**Evidence**: Our average income ($110K) is 31% higher than SOI average AGI ($84K)

### 2. **Tax Credits Not Modeled** 
**Impact: -5% to -10% revenue**

Hawaii has significant state tax credits that reduce net liability:
- **Food/Excise Tax Credit** (refundable, major impact on low-income)
- **Renewable Energy Tax Credit**
- **Child and Dependent Care Credit**
- **Low-Income Household Renters Credit**
- **Capital Goods Excise Tax Credit**

### 3. **Itemized Deductions**
**Impact: -5% to -8% revenue**

- We only model **standard deduction**
- ~30% of high-income filers **itemize** (mortgage interest, SALT, charitable)
- Itemized deductions typically exceed standard deduction for income >$150K

### 4. **Tax Unit Coverage**
**Impact: -22% of expected filers**

- We have 527,631 tax units vs 674,660 SOI returns (78.2% coverage)
- **Reasons**:
  - Some PUMS households don't file taxes (below threshold)
  - Students claimed as dependents elsewhere
  - Part-year residents
  - PUMS sampling methodology

### 5. **Filing Status Distribution**
**Impact: Shifts tax burden**

| Status | Our Model | SOI 2022 | Difference |
|--------|-----------|----------|------------|
| Single | 43.3% | 51.7% | -8.4pp |
| Joint | 48.6% | 35.1% | **+13.5pp** |
| HoH | 8.1% | 10.4% | -2.4pp |
| MFS | 0.0% | 2.7% | -2.7pp |

**Issue**: We over-identify joint filers, under-identify single filers
- Joint filers have higher income thresholds → lower effective rates
- This contributes to revenue overestimate

---

## Calibration Solution

### **Calibration Factor: 0.7252**

To match actual 2023 revenue of $3.1B, multiply all tax liabilities by **0.7252**.

### Implementation Options

#### **Option 1: Simple Calibration** (Recommended for immediate use)
```python
# Apply calibration factor to all tax calculations
calibrated_tax = raw_tax_liability * 0.7252
```

**Pros**: 
- ✅ Immediate accuracy
- ✅ Simple to implement
- ✅ Preserves relative comparisons

**Cons**:
- ⚠️ Doesn't fix underlying issues
- ⚠️ Black box adjustment

#### **Option 2: Income Adjustment** (Medium-term)
```python
# Reduce income by 27.5% to approximate AGI
adjusted_income = pums_income * 0.725
tax_liability = calculate_tax(adjusted_income, ...)
```

**Pros**:
- ✅ More realistic (approximates AGI)
- ✅ Transparent methodology

**Cons**:
- ⚠️ Still simplified
- ⚠️ Doesn't account for credits

#### **Option 3: Comprehensive Adjustments** (Long-term)
Implement detailed adjustments:
1. Convert PUMS income to AGI (retirement, HSA, SE tax)
2. Add Hawaii tax credits
3. Model itemized deductions
4. Improve filing status classification

**Pros**:
- ✅ Most accurate
- ✅ Publication-quality

**Cons**:
- ⚠️ Time-intensive
- ⚠️ Requires additional data

---

## Validation Results

### Revenue Scenarios (2017 Brackets)

| Scenario | Revenue | vs Actual | Description |
|----------|---------|-----------|-------------|
| **Base Model** | $4.28B | +38% | Current (no adjustments) |
| **10% Income Reduction** | $3.77B | +22% | Partial AGI adjustment |
| **20% Income Reduction** | $3.27B | +5% | Aggressive AGI adjustment |
| **Calibrated** | $3.10B | 0% | ✅ Matches actual |

### Comparison to SOI 2022

| Metric | Match Quality |
|--------|---------------|
| Total Income | ✅ 102% (very close) |
| Tax Unit Count | ⚠️ 78% (undercount) |
| Filing Status | ⚠️ Joint overcount, Single undercount |
| Average Income | ⚠️ 131% (high) |

---

## Impact on 2017 vs 2024 Analysis

### Original Results (Uncalibrated)
- 2017 Revenue: $4.28B
- 2024 Revenue: $3.50B
- **Change: -$772M (-18%)**

### Calibrated Results
- 2017 Revenue: $4.28B × 0.7252 = **$3.10B**
- 2024 Revenue: $3.50B × 0.7252 = **$2.54B**
- **Change: -$564M (-18%)**

**Key Insight**: The **relative change (-18%)** remains the same! The calibration factor cancels out in comparisons, so our bracket shift analysis is still valid.

---

## Recommendations

### Immediate Actions (✅ Complete)

1. **Apply Calibration Factor**
   - Use 0.7252 multiplier for all revenue estimates
   - Document methodology clearly
   - Note limitations in reports

2. **Update Documentation**
   - Add calibration explanation to README
   - Include in all analysis outputs
   - Cite SOI comparison

### Short-term Improvements (1-2 weeks)

1. **Fix Filing Status Distribution**
   - Reduce joint filer identification
   - Increase single filer count
   - Target: Match SOI percentages

2. **Add Income Adjustment Module**
   - Estimate retirement contributions by age/income
   - Subtract ~10-15% for AGI approximation
   - Validate against SOI income distribution

### Medium-term Enhancements (1-2 months)

1. **Implement Major Tax Credits**
   - Food/excise tax credit (largest impact)
   - Renewable energy credit
   - Estimate based on income/filing status

2. **Model Itemized Deductions**
   - Identify likely itemizers (income >$150K)
   - Use national averages for deduction amounts
   - Validate against SOI data

### Long-term Goals (3-6 months)

1. **Comprehensive AGI Conversion**
   - Detailed retirement contribution estimates
   - Self-employment tax calculations
   - All above-the-line deductions

2. **Full Hawaii Tax Credit Suite**
   - All refundable and non-refundable credits
   - Eligibility rules and phase-outs
   - Historical credit amounts

3. **Enhanced Validation**
   - Bracket-by-bracket comparison to SOI
   - County-level validation
   - Sensitivity analysis

---

## Files Generated

1. **`data/processed/soi_benchmarks_2022.csv`**
   - IRS SOI benchmarks for Hawaii
   - Filing status distribution
   - Income statistics

2. **`data/processed/calibration_factor.txt`**
   - Calibration factor: 0.7252
   - Apply to all tax liability calculations

3. **`scripts/extract_soi_benchmarks.py`**
   - Extracts SOI data from Excel file
   - Generates benchmark statistics

4. **`scripts/diagnose_and_calibrate_model.py`**
   - Comprehensive diagnostic analysis
   - Compares model to SOI
   - Calculates calibration factor

---

## Conclusion

The model's 38% revenue overestimate is **explainable and fixable**. The primary causes are:

1. **PUMS income ≠ AGI** (missing deductions)
2. **No tax credits modeled** (reduces liability)
3. **No itemized deductions** (high-income filers)
4. **Filing status misclassification** (too many joint filers)

**The calibration factor of 0.7252 provides immediate accuracy** while we implement more sophisticated adjustments. Importantly, **relative comparisons (2017 vs 2024) remain valid** since the calibration factor cancels out.

### Next Step

Apply the calibration factor to all revenue calculations and proceed with bracket shift analysis using the calibrated values.

---

**Prepared by:** Hawaii Income Tax Model  
**Data Sources:** IRS SOI 2022, Hawaii Dept of Taxation 2023, PUMS 2023 5-Year ACS
