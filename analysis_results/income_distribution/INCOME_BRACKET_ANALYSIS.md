# Income Bracket Distribution Analysis: Model vs DOTAX SOI 2022

## Executive Summary

While the **overall filing status totals** are excellent (within 5%), the **income bracket distributions** show **significant misalignment**, particularly at the extremes:

### Critical Findings

❌ **MAJOR ISSUE**: Severe undercounting in **lowest income brackets** (under $5K)
✅ **GOOD**: Middle income brackets generally well-aligned  
⚠️ **CONCERN**: Overcounting in **high income brackets** ($150K+)

---

## Detailed Findings by Filing Status

### 1. SINGLE FILERS

**Overall**: -4.7% (334,676 vs 351,205) ✅

**Income Bracket Performance**:

| Income Bracket | SOI | Model | Difference | % Diff |
|----------------|-----|-------|------------|--------|
| **Under $2,400** | 82,072 | 902 | -81,170 | **-98.9%** ❌ |
| **$2,400-$4,800** | 13,773 | 527 | -13,246 | **-96.2%** ❌ |
| $4,800-$9,600 | 24,317 | 23,665 | -652 | -2.7% ✅ |
| $9,600-$14,400 | 20,638 | 26,370 | +5,732 | +27.8% ⚠️ |
| $14,400-$19,200 | 18,647 | 22,475 | +3,828 | +20.5% ⚠️ |
| $19,200-$24,000 | 18,863 | 21,062 | +2,199 | +11.7% ⚠️ |
| $24,000-$36,000 | 47,514 | 53,936 | +6,422 | +13.5% ⚠️ |
| $36,000-$48,000 | 40,454 | 44,521 | +4,067 | +10.1% ⚠️ |
| **$48,000-$150,000** | 76,914 | 126,408 | +49,494 | **+64.3%** ❌ |
| **$150,000-$175,000** | 2,060 | 4,304 | +2,244 | **+108.9%** ❌ |
| **$175,000-$200,000** | 1,220 | 2,939 | +1,719 | **+140.9%** ❌ |
| **Over $200,000** | 4,733 | 7,567 | +2,834 | **+59.9%** ❌ |

**Key Issues**:
- ❌ **Missing 94,416 very low-income single filers** (under $5K)
- ❌ **Overcounting middle-high income** by 49,494 ($48K-$150K)
- ❌ **Overcounting high income** by 6,797 ($150K+)
- Only **1 out of 12 brackets** within ±5%

---

### 2. MARRIED FILING JOINTLY

**Overall**: -1.0% (214,209 vs 216,358) ✅

**Income Bracket Performance**:

| Income Bracket | SOI | Model | Difference | % Diff |
|----------------|-----|-------|------------|--------|
| **Under $4,800** | 40,236 | 1,023 | -39,213 | **-97.5%** ❌ |
| **$4,800-$9,600** | 5,632 | 1,621 | -4,011 | **-71.2%** ❌ |
| **$9,600-$19,200** | 10,798 | 4,921 | -5,877 | **-54.4%** ❌ |
| **$19,200-$28,800** | 11,317 | 6,466 | -4,851 | **-42.9%** ❌ |
| **$28,800-$38,400** | 11,443 | 7,420 | -4,023 | **-35.2%** ❌ |
| $38,400-$48,000 | 10,914 | 8,871 | -2,043 | -18.7% ⚠️ |
| $48,000-$72,000 | 27,330 | 25,414 | -1,916 | -7.0% ⚠️ |
| $72,000-$96,000 | 26,226 | 30,102 | +3,876 | +14.8% ⚠️ |
| **$96,000-$300,000** | 62,358 | 114,956 | +52,598 | **+84.3%** ❌ |
| **$300,000-$350,000** | 2,401 | 3,944 | +1,543 | **+64.3%** ❌ |
| $350,000-$400,000 | 1,588 | 2,261 | +673 | +42.4% ⚠️ |
| Over $400,000 | 6,115 | 7,210 | +1,095 | +17.9% ⚠️ |

**Key Issues**:
- ❌ **Missing 57,975 very low-income joint filers** (under $38K)
- ❌ **Massive overcount of $96K-$300K bracket** (+52,598, +84.3%)
- ❌ **Overcounting high income** by 5,311 ($300K+)
- **0 out of 12 brackets** within ±5%

---

### 3. HEAD OF HOUSEHOLD

**Overall**: +0.1% (67,448 vs 67,393) ✅

**Income Bracket Performance**:

| Income Bracket | SOI | Model | Difference | % Diff |
|----------------|-----|-------|------------|--------|
| **Under $3,600** | 6,671 | 45 | -6,626 | **-99.3%** ❌ |
| **$3,600-$7,200** | 2,619 | 1,262 | -1,357 | **-51.8%** ❌ |
| $7,200-$14,400 | 5,543 | 4,006 | -1,537 | -27.7% ⚠️ |
| $14,400-$21,600 | 5,904 | 3,631 | -2,273 | -38.5% ⚠️ |
| **$21,600-$28,800** | 7,376 | 4,418 | -2,958 | **-40.1%** ❌ |
| $28,800-$36,000 | 7,732 | 5,379 | -2,353 | -30.4% ⚠️ |
| $36,000-$54,000 | 13,878 | 13,766 | -112 | -0.8% ✅ |
| $54,000-$72,000 | 7,835 | 7,966 | +131 | +1.7% ✅ |
| **$72,000-$225,000** | 9,117 | 25,167 | +16,050 | **+176.0%** ❌ |
| $225,000-$262,500 | 176 | 269 | +93 | +52.6% ⚠️ |
| $262,500-$300,000 | 118 | 159 | +41 | +34.6% ⚠️ |
| **Over $300,000** | 424 | 1,380 | +956 | **+225.6%** ❌ |

**Key Issues**:
- ❌ **Missing 10,610 very low-income HoH filers** (under $14K)
- ❌ **Massive overcount of $72K-$225K bracket** (+16,050, +176.0%)
- ❌ **Overcounting very high income** (+956, +225.6% over $300K)
- Only **2 out of 12 brackets** within ±5%

---

## Root Cause Analysis

### Problem 1: Missing Very Low-Income Filers ❌

**Total missing across all statuses**: ~152,000 filers under $5K

**Possible causes**:
1. **PUMS income reporting**: PUMS may not capture very low-income individuals who file taxes
2. **Filing threshold logic**: Tax units may be excluding people below filing thresholds
3. **Dependent classification**: Low-income adults may be incorrectly classified as dependents
4. **PUMS coverage**: PUMS may undersample very low-income populations

**Impact**: This is the primary driver of the overall -2.7% gap in total returns.

### Problem 2: Income Shifting to Higher Brackets ❌

**Pattern**: Consistent undercounting in low brackets, overcounting in middle-high brackets

**Possible causes**:
1. **Income definition mismatch**: PUMS "income" vs SOI "taxable income"
   - PUMS uses total income (wages + benefits + etc.)
   - SOI uses taxable income (after deductions/adjustments)
   - PUMS income is likely **higher** than taxable income
2. **AGI adjustments not applied**: Model may not be reducing income for:
   - IRA contributions
   - Student loan interest
   - Self-employment deductions
   - Other above-the-line deductions
3. **Standard deduction**: SOI brackets are based on **taxable income** (after standard deduction)

### Problem 3: High-Income Overcounting ⚠️

**Pattern**: Consistent overcounting in $150K+ brackets across all statuses

**Possible causes**:
1. **Income inflation**: PUMS income > taxable income (see above)
2. **Itemized deductions**: High-income filers itemize, reducing taxable income
3. **Business income**: PUMS may overstate business income vs actual taxable income

---

## Recommendations

### Priority 1: Fix Income Definition ⭐⭐⭐ CRITICAL

**Issue**: PUMS "income" ≠ SOI "taxable income"

**Solution**: Apply income adjustments to match SOI definition:

```python
# In tax unit construction, adjust income to approximate taxable income
taxable_income = pums_income - standard_deduction - agi_adjustments
```

**Expected Impact**: 
- Shift distributions down by $12K-$25K (standard deduction)
- Move ~100K filers from middle brackets to lower brackets
- Better alignment across all brackets

### Priority 2: Investigate Low-Income Coverage ⭐⭐ HIGH

**Issue**: Missing 152,000 very low-income filers

**Actions**:
1. Check if PUMS excludes non-filers
2. Review dependent classification logic
3. Verify filing threshold logic
4. Compare PUMS vs SOI universe definitions

### Priority 3: Apply AGI Adjustments ⭐⭐ HIGH

**Issue**: PUMS total income > AGI

**Solution**: Implement AGI adjustment module (may already exist per memory):
- IRA contributions (~1-1.5% of income)
- Self-employed health insurance
- Student loan interest
- Other above-the-line deductions

**Expected Impact**: Reduce income by ~1-2%, shifting distributions down

### Priority 4: Consider Itemized Deductions ⭐ MEDIUM

**Issue**: High-income filers itemize, reducing taxable income

**Solution**: Estimate itemized deductions for high-income filers:
- Mortgage interest
- State/local taxes (SALT)
- Charitable contributions

---

## Summary Statistics

| Filing Status | Brackets within ±5% | Brackets within ±10% | Avg % Diff | Std Dev |
|---------------|---------------------|----------------------|------------|---------|
| Single | 1/12 (8%) | 1/12 (8%) | +21.6% | 70.4% |
| MFJ | 0/12 (0%) | 1/12 (8%) | -8.6% | 55.3% |
| HoH | 2/12 (17%) | 2/12 (17%) | +16.8% | 95.2% |

**Overall Assessment**: 
- ❌ **Income bracket alignment is POOR**
- ✅ **Total counts are GOOD** (within 5%)
- ⚠️ **Systematic bias**: Income shifting from low to high brackets

---

## Impact on Tax Credit Calculations

### CTC/EITC Implications

The income bracket misalignment will **significantly impact** tax credit estimates:

1. **EITC**: Concentrated in $10K-$50K range
   - Missing low-income filers = **underestimate EITC**
   - Overcounting middle income = **overestimate EITC**

2. **CTC**: Phases out at $200K (single) / $400K (joint)
   - Overcounting high income = **underestimate CTC** (more phaseouts)
   - Income shifting = **distorted phase-out patterns**

3. **Overall**: Income definition mismatch creates **systematic bias** in all calculations

**Recommendation**: **Fix income definition BEFORE running CTC/EITC analysis**

---

*Generated: 2025-10-15*
*Analysis: Detailed Income Bracket Comparison - Model vs DOTAX SOI 2022*
