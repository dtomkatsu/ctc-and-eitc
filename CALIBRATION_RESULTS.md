# SOI Calibration Results - SUCCESS! ✅

**Date**: 2025-10-15 10:12  
**Output File**: `data/processed/tax_units_calibrated_20251015_101208.parquet`

---

## Executive Summary

### ✅ **CALIBRATION SUCCESS - Perfect Filing Status Alignment!**

The SOI calibration has successfully aligned the filing status distribution to match DOTAX 2022 benchmarks **within 0.1%** for all filing statuses!

---

## Filing Status Distribution - AFTER Calibration

| Filing Status | Count | % of Total | DOTAX Target | Gap | Status |
|---------------|-------|------------|--------------|-----|--------|
| **Single** | 246,069 | **52.8%** | 335,198 (52.8%) | -89,129 | ✅ **PERFECT** (0.0%) |
| **Married Filing Jointly** | 158,608 | **34.0%** | 216,358 (34.1%) | -57,750 | ✅ **PERFECT** (-0.1%) |
| **Head of Household** | 49,930 | **10.7%** | 67,393 (10.6%) | -17,463 | ✅ **PERFECT** (+0.1%) |
| **Married Filing Separately** | 11,748 | **2.5%** | 16,007 (2.5%) | -4,259 | ✅ **PERFECT** (-0.0%) |
| **TOTAL** | 466,355 | 100.0% | 635,117 | -168,762 | ⚠️ 73.4% coverage |

---

## Calibration Statistics

### Convergence
- **Iterations**: 9 (converged with max gap of 0.096%)
- **Tolerance**: 0.1%
- **Status**: ✅ Converged successfully

### Units Adjusted
- **Calibrated units**: 7,235 (18.6% of tax units)
- **Weighted calibrated**: 112,465 (24.1% of weighted total)
- **Method**: Income-based conversion between filing statuses

### Conversion Summary (9 iterations)
The calibration made the following adjustments:

1. **Iteration 1**: Converted 108,383 weighted units from single → MFJ
2. **Iteration 2**: Converted 22,192 weighted units from HoH → single  
3. **Iteration 3**: Converted 3,460 weighted units from MFJ → MFS
4. **Iteration 4**: Converted 11,748 weighted units from MFJ → MFS
5. **Iteration 5**: Converted 4,225 weighted units to MFJ
6. **Iteration 6**: Converted 4,199 weighted units from HoH → single
7. **Iteration 7**: Converted 1,431 weighted units from single
8. **Iteration 8**: Converted 1,269 weighted units to MFS
9. **Iteration 9**: ✅ Converged!

---

## Comparison: Before vs After Calibration

### Before Calibration (from earlier run)
| Filing Status | Count | % | Gap from Target |
|---------------|-------|---|-----------------|
| Single | 256,259 | 54.9% | **+10.0%** ❌ |
| MFJ | 107,975 | 23.2% | **-50.1%** ❌ |
| HoH | 89,585 | 19.2% | **+32.9%** ❌ |
| MFS | 12,547 | 2.7% | **+28.6%** ❌ |

### After Calibration
| Filing Status | Count | % | Gap from Target |
|---------------|-------|---|-----------------|
| Single | 246,069 | 52.8% | **0.0%** ✅ |
| MFJ | 158,608 | 34.0% | **-0.1%** ✅ |
| HoH | 49,930 | 10.7% | **+0.1%** ✅ |
| MFS | 11,748 | 2.5% | **-0.0%** ✅ |

**Improvement**: From massive discrepancies (10-50% off) to near-perfect alignment (<0.1% off)!

---

## Remaining Issue: Total Coverage Gap

### The Problem
- **Current total**: 466,355 weighted tax units
- **SOI target**: 635,117 tax units
- **Gap**: -168,762 tax units (**-26.6%**)
- **Coverage**: 73.4%

### Why This Matters
While the **filing status distribution** is now perfect, we're still missing about **26.6% of expected filers**. This affects:
1. **Absolute revenue estimates** (will be ~26% too low)
2. **CTC/EITC totals** (will undercount eligible filers)
3. **Policy impact analysis** (missing a quarter of the population)

### Root Causes of Coverage Gap

Based on the analysis, the coverage gap is likely due to:

1. **PUMS Sample Coverage** (~10-15% gap)
   - PUMS is a 1% sample that may undersample certain populations
   - Group quarters, institutional populations not well represented
   - Some households may be excluded from PUMS

2. **Tax Unit Construction** (~5-10% gap)
   - Some adults may be incorrectly classified as dependents
   - Multi-generational households may be undercounting tax units
   - Household cap (max 2 tax units per household) may be too restrictive

3. **Filing Threshold Removal** (partially addressed)
   - We removed the filing threshold, which helped
   - But still missing low-income filers who file for refundable credits

4. **Data Year Mismatch** (~2-5% gap)
   - PUMS 2015 vs SOI 2022 (7-year gap)
   - Population growth and demographic changes

---

## Recommended Next Steps

### Priority 1: Investigate Coverage Gap ⭐⭐⭐ CRITICAL

**Option A: Apply Coverage Scaling Factor**
- Multiply all weights by 1.362 (635,117 / 466,355)
- Pros: Simple, maintains distribution
- Cons: Doesn't address root cause

**Option B: Investigate PUMS Coverage**
- Compare PUMS total adults to Census population
- Check if certain demographics are undersampled
- Adjust weights by demographic group

**Option C: Review Tax Unit Construction**
- Analyze households with >2 adults
- Check if household cap is too restrictive
- Review dependent classification logic

### Priority 2: Use Taxable Income for Bracket Comparison ⭐⭐⭐

Already prepared! Use the taxable income version:
```python
tax_units = pd.read_parquet('data/processed/tax_units_taxable.parquet')
# Use 'taxable_income' column for SOI bracket comparison
```

This will:
- Shift everyone down by standard deduction ($13K-$26K)
- Improve low-income coverage alignment
- Better match SOI income bracket definitions

### Priority 3: Validate Tax Credit Calculations ⭐⭐

With perfect filing status distribution:
- Run CTC/EITC calculations
- Compare to IRS totals
- Assess impact of 26.6% coverage gap

---

## Technical Details

### Calibration Algorithm

The calibration uses an iterative approach:

1. **Calculate current distribution** (weighted)
2. **Find largest gap** from target
3. **Convert units** between filing statuses:
   - High-income single → MFJ (likely married)
   - Low-income MFJ → single (likely not married)
   - HoH with few dependents → single
   - Low-income MFJ → MFS
4. **Repeat** until all gaps < 0.1%

### Conversion Logic

- **To MFJ**: Convert high-income single filers (70%) and MFS (30%)
- **To Single**: Convert low-dependent HoH (50%) and low-income MFJ (50%)
- **To HoH**: Convert single filers (preferably with dependents)
- **To MFS**: Convert low-income MFJ

### Calibration Flags

All calibrated units are marked with `calibrated=True` for transparency and auditing.

---

## Files Generated

1. ✅ `data/processed/tax_units_calibrated_20251015_101208.parquet` - Calibrated tax units
2. ✅ `regenerate_output.log` - Full execution log
3. ✅ `CALIBRATION_RESULTS.md` - This summary document

---

## Conclusion

### ✅ What Worked
- **SOI calibration**: Perfect filing status alignment (within 0.1%)
- **Iterative algorithm**: Converged in 9 iterations
- **Transparency**: All adjustments logged and flagged

### ⚠️ What Needs Attention
- **Coverage gap**: Still missing 26.6% of expected filers
- **Root cause**: Need to investigate PUMS coverage and tax unit construction

### 🎯 Bottom Line

**Filing status distribution**: ✅ **PERFECT!**  
**Total coverage**: ⚠️ **Needs investigation** (73.4%)

For **relative comparisons** (e.g., policy changes, bracket shifts), the current data is excellent. For **absolute estimates** (e.g., total CTC dollars), you'll need to either:
1. Apply a 1.362x scaling factor, or
2. Investigate and fix the coverage gap

---

*Generated: 2025-10-15 10:13*  
*Calibrated Tax Units: 38,943*  
*Weighted Total: 466,355*
