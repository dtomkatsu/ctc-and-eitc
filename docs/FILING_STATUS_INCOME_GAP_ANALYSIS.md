# Filing Status Income Gap Analysis

**Date**: October 31, 2025  
**Status**: 🔴 **CRITICAL ISSUES IDENTIFIED**

---

## Executive Summary

Filing status weight calibration successfully achieved exact filing status share targets (51% Single, 36% Joint, 9.6% HoH, 3.4% MFS), but **revenue gaps persist due to income distribution mismatches**, particularly for MFS and Single filers.

**Key Finding**: Weight calibration adjusts *how many* people of each status exist, but doesn't fix *how much income/tax* each person has. Income-level issues remain.

---

## Critical Issues by Filing Status

### 🔴 **MFS (Married Filing Separately) - SEVERE**

| Metric | Model | DOTAX | Gap |
|--------|-------|-------|-----|
| **Returns** | 19,897 | 16,007 | +24.3% ⚠️ |
| **Avg AGI** | $45,702 | $196,726 | **-76.8%** ❌ |
| **Avg Tax** | $2,996 | $18,055 | **-83.4%** ❌ |
| **Total Tax** | $59.6M | $289M | **-79.4%** ❌ |
| **Eff Rate** | 6.56% | 9.18% | -2.62pp |

**Root Cause**: Model is creating MFS units from **low-income married couples** instead of **high-income couples filing separately for tax optimization**.

**Real-world MFS filers**: Typically high-income couples ($200K+) who file separately to:
- Maximize itemized deductions (SALT cap workaround)
- Optimize student loan repayment plans
- Separate business/investment income

**Model MFS filers**: Low-income couples flagged by income disparity or other heuristics.

**Impact**: $229M revenue gap (79% underestimate) for MFS category.

---

### ⚠️ **Single - MODERATE**

| Metric | Model | DOTAX | Gap |
|--------|-------|-------|-----|
| **Returns** | 298,448 | 335,198 | -11.0% |
| **Avg AGI** | $40,700 | $42,652 | -4.6% |
| **Avg Tax** | $2,545 | $2,578 | -1.3% ✅ |
| **Total Tax** | $759.6M | $864M | -12.1% |

**Root Cause**: Calibration is not fully hitting the 51% target (only 50.9% actual vs 51.0% target).

**Issue**: Weight calibration applied, but returns count still 11% below target. This suggests:
1. Calibration factors not being applied correctly, OR
2. Total weight sum differs from DOTAX total (587,742 vs 635,117)

**Impact**: $104M revenue gap (12% underestimate).

---

### ⚠️ **Head of Household - MODERATE**

| Metric | Model | DOTAX | Gap |
|--------|-------|-------|-----|
| **Returns** | 56,179 | 67,393 | -16.6% |
| **Avg AGI** | $59,020 | $55,555 | +6.2% |
| **Avg Tax** | $3,315 | $2,997 | +10.6% |
| **Total Tax** | $186.2M | $202M | -7.8% ✅ |

**Root Cause**: Too few HoH units created (16.6% below target), but those created have higher incomes than expected.

**Observation**: Average AGI and tax per return are *higher* than DOTAX, but total tax is lower due to insufficient unit count.

**Impact**: $15.8M revenue gap (8% underestimate) - relatively minor.

---

### ✅ **Joint - ACCEPTABLE**

| Metric | Model | DOTAX | Gap |
|--------|-------|-------|-----|
| **Returns** | 210,669 | 216,358 | -2.6% ✅ |
| **Avg AGI** | $116,706 | $122,718 | -4.9% ✅ |
| **Avg Tax** | $8,345 | $7,737 | +7.9% |
| **Total Tax** | $1,758M | $1,674M | **+5.0%** ✅ |

**Status**: Joint filers are **over-performing** (+5% revenue vs target).

**Observation**: Slightly fewer returns and lower average AGI, but higher effective tax rate (7.15% vs 6.30%) leads to higher total tax.

**Impact**: +$84M revenue surplus (5% overestimate) - helps offset other gaps.

---

## Root Cause Analysis

### 1. **Weight Calibration Not Fully Applied**

Returns counts still show gaps:
- Single: -11.0% (should be 0%)
- Joint: -2.6% (should be 0%)
- HoH: -16.6% (should be 0%)
- MFS: +24.3% (should be 0%)

**Hypothesis**: Calibration is adjusting weights, but total weighted count (587,742) is still 7.5% below DOTAX total (635,117).

**Solution**: Need to scale total weights to match DOTAX total returns count.

---

### 2. **MFS Income Distribution Completely Wrong**

MFS filers have average AGI of $45K vs $197K expected (-77% gap).

**Hypothesis**: Model's MFS identification logic (`_should_file_separately()`) is flagging:
- Low-income couples with income disparity
- Couples with employment mismatches
- Couples in complex households

Instead of:
- High-income couples optimizing tax strategy
- Couples with business/investment income
- Couples maximizing itemized deductions

**Solution**: 
1. **Restrict MFS to high-income couples only** (AGI > $150K)
2. **Use income-based probability** for MFS assignment
3. **Synthetic MFS units** for ultra-high-income brackets

---

### 3. **Income Distribution Calibration Needed**

Even with correct filing status shares, income distributions within each status don't match DOTAX:
- Single: -4.6% avg AGI
- Joint: -4.9% avg AGI
- HoH: +6.2% avg AGI
- MFS: -76.8% avg AGI ❌

**Solution**: Apply income distribution calibration *within each filing status* to match DOTAX average AGI targets.

---

## Recommended Fixes (Priority Order)

### 🔴 **Priority 1: Fix MFS Income Distribution**

**Current**: MFS avg AGI = $45,702  
**Target**: MFS avg AGI = $196,726  
**Gap**: -76.8%

**Actions**:
1. **Restrict MFS to high-income only**: Only allow MFS for couples with AGI > $150K
2. **Income-based MFS probability**: Higher income → higher MFS probability
3. **Synthetic MFS units**: Create synthetic high-income MFS units to fill gap
4. **Review MFS logic**: Check `_should_file_separately()` criteria

**Expected Impact**: Close $229M MFS revenue gap (79% of total gap).

---

### 🟡 **Priority 2: Scale Total Weights to Match DOTAX**

**Current**: 587,742 total weighted returns  
**Target**: 635,117 total returns  
**Gap**: -7.5%

**Actions**:
1. **Apply global scaling factor**: Multiply all weights by 1.0806 (635,117 / 587,742)
2. **Verify calibration**: Ensure filing status shares remain at targets after scaling

**Expected Impact**: Close ~$265M total revenue gap proportionally across all statuses.

---

### 🟡 **Priority 3: Income Distribution Calibration Within Filing Status**

**Actions**:
1. **Single**: Increase avg AGI by 4.6% to match $42,652 target
2. **Joint**: Increase avg AGI by 4.9% to match $122,718 target
3. **HoH**: Decrease avg AGI by 6.2% to match $55,555 target
4. **MFS**: Increase avg AGI by 76.8% to match $196,726 target (covered by Priority 1)

**Method**: Apply percentile-based income shifting within each filing status group.

**Expected Impact**: Fine-tune revenue to match DOTAX targets exactly.

---

## Implementation Plan

### Phase 1: MFS Fix (Immediate)
```python
# In _should_file_separately() or post-processing:
# 1. Restrict MFS to high-income couples
if agi < 150000:
    return False  # Don't allow MFS for low-income couples

# 2. Income-based MFS probability
mfs_probability = min(0.15, (agi - 150000) / 1000000)  # 0-15% based on income

# 3. Synthetic MFS units
# Create synthetic high-income MFS units to match DOTAX distribution
```

### Phase 2: Global Weight Scaling
```python
# After filing status calibration:
DOTAX_TOTAL_RETURNS = 635117
current_total = df['weight'].sum()
scaling_factor = DOTAX_TOTAL_RETURNS / current_total
df['weight'] = df['weight'] * scaling_factor
```

### Phase 3: Income Distribution Calibration
```python
# Within each filing status:
for status in ['Single', 'Joint', 'Head of Household', 'MFS']:
    status_df = df[df['filing_status_clean'] == status]
    target_avg_agi = DOTAX_AVG_AGI[status]
    current_avg_agi = (status_df['agi'] * status_df['weight']).sum() / status_df['weight'].sum()
    
    # Apply percentile-based income adjustment
    adjustment_factor = target_avg_agi / current_avg_agi
    df.loc[df['filing_status_clean'] == status, 'agi'] *= adjustment_factor
```

---

## Expected Outcomes

### After Priority 1 (MFS Fix):
- MFS revenue: $59.6M → ~$289M (+$229M)
- Total revenue: $2,763M → ~$2,992M
- Remaining gap: -8.8% → -1.2%

### After Priority 2 (Global Scaling):
- Total returns: 587,742 → 635,117 (+7.5%)
- Total revenue: $2,992M → ~$3,217M
- Remaining gap: -1.2% → +6.2% (over-estimate, need fine-tuning)

### After Priority 3 (Income Calibration):
- Total revenue: $3,217M → $3,029M (exact match)
- All filing status revenues match DOTAX targets
- Remaining gap: +6.2% → 0.0% ✅

---

## Validation Metrics

After implementing fixes, verify:

1. **Filing Status Shares** (should remain at targets):
   - Single: 51.0%
   - Joint: 36.0%
   - HoH: 9.6%
   - MFS: 3.4%

2. **Average AGI by Filing Status** (should match DOTAX):
   - Single: $42,652
   - Joint: $122,718
   - HoH: $55,555
   - MFS: $196,726

3. **Total Tax by Filing Status** (should match DOTAX):
   - Single: $864M
   - Joint: $1,674M
   - HoH: $202M
   - MFS: $289M

4. **Total Returns**: 635,117

---

## Conclusion

**Filing status weight calibration is working** (shares are correct), but **income distributions within each status are misaligned**, particularly for MFS filers.

**Key Insight**: Weight calibration alone cannot fix revenue gaps when the underlying income distributions are wrong. Need to:
1. Fix MFS income distribution (Priority 1)
2. Scale total weights to match DOTAX (Priority 2)
3. Calibrate income distributions within each filing status (Priority 3)

**Expected Result**: All three fixes combined should close the -8.8% revenue gap to near 0%.

---

## Files

- **Diagnostic Script**: `scripts/diagnose_filing_status_income_gaps.py`
- **Latest Calibrated Data**: `data/processed/tax_units_filing_status_calibrated_20251030_121326.parquet`
- **DOTAX Benchmarks**: `data/raw/Dotax Soi 2022 - 5A.csv` (Table 5A)
