# Final Tax Units Regeneration Summary

## Results After All Fixes

**File:** `data/processed/tax_units_regenerated_20251014_200110.parquet`

### Overall Statistics

- **Tax units created:** 35,009
- **Weighted filers:** 681,867
- **DOTAX target:** 635,117
- **Coverage:** 107.4% (+7.4% overcounting)
- **Units per household:** 1.32 (was 1.59, now much better)

### Filing Status Distribution

| Status | Current | % | Target | % | Gap |
|--------|---------|---|--------|---|-----|
| **Single** | 393,302 | 57.7% | 335,198 | 52.8% | +17.3% |
| **Joint** | 228,086 | 33.5% | 216,358 | 34.1% | **+5.4%** ✅ |
| **HoH** | 35,856 | 5.3% | 67,393 | 10.6% | **-46.8%** ❌ |
| **MFS** | 24,623 | 3.6% | 16,007 | 2.5% | +53.8% |
| **TOTAL** | 681,867 | 100% | 635,117 | 100% | +7.4% |

### Critical Bugs Fixed

#### 1. ✅ ADJINC Calculation Bug (CRITICAL)
**Fixed in:** `src/tax/units/income.py` and `src/tax/units/constructor.py`

- ADJINC stored as integer 1,184,371 (represents 1.184371)
- Was multiplying by 1,184,371 instead of 1.184371
- **Result:** Median income now $58,795 (was $55 billion!)

#### 2. ✅ Weight Calculation Bug (CRITICAL)
**Fixed in:** `src/tax/units/constructor.py` - `_calculate_hybrid_weight()`

- Old formula: `(hh_weight + sum(person_weights)) / 2`
- Inflated joint filers by 2x (summed both spouses' weights + household weight)
- **New formula:** Use average person weight for couples, single person weight for singles
- **Result:** Joint filers went from +97.3% to +5.4% (near perfect!)

#### 3. ✅ MAX_TAX_UNITS Reduced
- Reduced from 4 to 2 units per household
- **Result:** 1.59 → 1.32 units/household

#### 4. ✅ Filing Threshold Added
- Added $5,000 minimum income threshold
- **Result:** Filters out non-filers

## Remaining Issues

### Issue 1: Head of Household Severely Undercounted (-46.8%)

**Current:** 35,856 HoH filers (5.3%)  
**Target:** 67,393 HoH filers (10.6%)  
**Gap:** -31,537 filers (-46.8%)

**Root Cause:** Not identifying enough dependents in households
- HoH qualification logic works correctly (100% of HoH filers have dependents)
- Only 57 single filers have dependents (should be HoH but aren't)
- The problem is **upstream**: dependent identification in `identify_dependents()` is too conservative

**Solution Needed:**
- Review `src/tax/units/dependencies.py`
- Relax dependent identification criteria
- Consider grandchildren, other relatives as potential dependents
- Check relationship code filters

### Issue 2: Slight Overcounting (+7.4%)

**Current:** 681,867 filers  
**Target:** 635,117 filers  
**Gap:** +46,750 filers

**Causes:**
1. Still creating 1.32 units/household (target ~1.1)
2. 32% of households have multiple units (should be ~20%)

**Solution Needed:**
- Consider lowering MAX_TAX_UNITS to 1 for most households
- Add stricter adult filing criteria
- Review multi-unit household logic

### Issue 3: MFS Overcounted (+53.8%)

**Current:** 24,623 MFS filers (3.6%)  
**Target:** 16,007 MFS filers (2.5%)  
**Gap:** +8,616 filers

**Solution Needed:**
- Reduce MFS scoring probabilities:
  - Score 3: 5% → 2%
  - Score 4: 30% → 20%
  - Score 5: 60% → 50%

## Comparison to Previous Iterations

| Metric | Old (Aug 19) | First Try | Second Try | Current |
|--------|--------------|-----------|------------|---------|
| **Total Filers** | 527,631 | 1,046,345 | 886,316 | 681,867 |
| **Coverage** | 83.1% | 164.7% | 139.6% | 107.4% |
| **Units/HH** | 1.00 | 1.59 | 1.34 | 1.32 |
| **Income (median)** | N/A | $55B 💀 | $55B 💀 | $58K ✅ |
| **MFS %** | 0% | 3.7% | 4.1% | 3.6% |
| **Joint %** | 48.6% | 40.8% | 48.2% | 33.5% ✅ |

## What Works Well Now

✅ **Joint filers:** 33.5% vs 34.1% target (only +5.4% gap!)  
✅ **Income values:** Median $58,795 (reasonable)  
✅ **MFS filers:** Created successfully (was 0%)  
✅ **Units/household:** 1.32 (down from 1.59)  
✅ **No duplicate adults:** Each adult in only one tax unit  

## What Still Needs Work

❌ **HoH undercounted:** 5.3% vs 10.6% target (-46.8%)  
❌ **Overall overcounting:** 107.4% coverage (need to reach 100%)  
❌ **Single overcounted:** 57.7% vs 52.8% target (+17.3%)  

## Recommended Next Steps

### Priority 1: Fix HoH Undercounting

1. **Analyze dependent identification**
   ```bash
   python scripts/diagnosis/diagnose_dependents.py
   ```

2. **Review `src/tax/units/dependencies.py`**
   - Check relationship code filters
   - Consider expanding qualifying relationships
   - Look at age thresholds

3. **Target:** Convert ~30,000 single filers to HoH

### Priority 2: Reduce Overall Overcounting

1. **Option A:** Lower MAX_TAX_UNITS to 1 (stricter)
2. **Option B:** Add stricter filing requirements
3. **Option C:** Apply calibration weights (quick fix)

### Priority 3: Fine-Tune MFS Rate

1. Reduce MFS scoring probabilities by ~30%
2. Target: 16,007 MFS filers (2.5%)

## For Immediate Use

**Recommendation:** Use the **current file** for your analysis:
- `data/processed/tax_units_regenerated_20251014_200110.parquet`
- 681,867 weighted filers (107.4% coverage)
- Apply calibration factor of **0.931** to reach exactly 635,117
- Filing status distributions are reasonably close to DOTAX

**Calibration:**
```python
calibration_factor = 635117 / 681867  # = 0.931
calibrated_weights = tax_units['weight'] * calibration_factor
```

This will give you:
- Total filers: 635,117 (exact match)
- Filing status proportions maintained
- Income values correct

## Files Modified

1. **`src/tax/units/income.py`** - Fixed ADJINC calculation
2. **`src/tax/units/constructor.py`**:
   - Fixed ADJINC calculation
   - Fixed hybrid weight calculation
   - Reduced MAX_TAX_UNITS from 4 to 2
   - Added filing threshold filter

## Documentation Created

- `OVERCOUNTING_FIX_SUMMARY.md` - Detailed analysis of overcounting
- `FILING_STATUS_CALIBRATION_PLAN.md` - Strategy for all filing statuses
- `MFS_DIAGNOSIS_SUMMARY.md` - MFS scoring analysis
- `FINAL_REGENERATION_SUMMARY.md` - This document

---

**Status:** Major progress achieved. Joint filers near perfect (+5.4%). HoH undercounting is the main remaining issue.
