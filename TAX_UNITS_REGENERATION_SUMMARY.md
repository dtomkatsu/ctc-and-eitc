# Tax Units Regeneration Summary

## Results

### ✅ MFS Filers Successfully Created!

**MFS Filers:** 38,662 (3.70% of total)  
**DOTAX Target:** 16,007 (2.5%)  
**Status:** MFS logic is working, but creating 2.4x too many

### ❌ Critical Overcounting Issue

**Total Filers:** 1,046,345  
**DOTAX Target:** 635,117  
**Overcounting:** +411,228 filers (164.7% of target!)

## Filing Status Distribution

| Status | Current | % | DOTAX Target | % | Gap |
|--------|---------|---|--------------|---|-----|
| **Single** | 524,157 | 50.1% | 335,198 | 52.8% | +188,959 (+56.4%) |
| **Joint** | 426,934 | 40.8% | 216,358 | 34.1% | +210,576 (+97.3%) |
| **HoH** | 56,591 | 5.4% | 67,393 | 10.6% | -10,802 (-16.0%) |
| **MFS** | 38,662 | 3.7% | 16,007 | 2.5% | +22,656 (+141.5%) |
| **TOTAL** | **1,046,345** | **100%** | **635,117** | **100%** | **+411,228** |

## Root Cause

The constructor is creating **1.59 tax units per household on average** (should be ~1.1):
- 28,896 households
- 45,686 tax units created
- 10,354 households (35.8%) have multiple tax units

This suggests the constructor is:
1. **Over-splitting households** into too many tax units
2. **Not properly consolidating** related adults
3. **Creating phantom tax units** for dependents or non-filers

## What Worked

✅ **MFS Logic:** Successfully creates MFS filers (was 0%, now 3.7%)  
✅ **Updated Thresholds:** Score-based MFS probabilities are functioning  
✅ **Strict Pairing:** `_are_married()` only pairs RELSHIPP 20/21

## What Needs Fixing

❌ **Household Consolidation:** Too many tax units per household  
❌ **Adult Assignment:** Some adults being assigned to multiple units  
❌ **Dependent Handling:** Dependents may be creating separate units  
❌ **Filing Threshold:** Not properly filtering non-filers

## Comparison to Previous File

**Old File (Aug 19):**
- 29,060 tax units
- 527,631 weighted filers (83.1% of target)
- 0% MFS filers
- **Undercounting problem**

**New File (Oct 14):**
- 45,686 tax units  
- 1,046,345 weighted filers (164.7% of target)
- 3.7% MFS filers
- **Overcounting problem**

## Recommended Next Steps

### Option 1: Use Old File + Manual MFS Adjustment
- Start with old file (527,631 filers)
- Manually convert some joint filers to MFS using calibration weights
- Simpler and faster

### Option 2: Fix Overcounting in Constructor
- Debug why 1.59 units/household (should be ~1.1)
- Fix adult assignment logic
- Add filing threshold checks
- More accurate long-term

### Option 3: Hybrid Calibration Approach
- Use new file's filing status distribution (proportions)
- Scale total down to 635,117 using calibration weights
- Quick fix but masks underlying issues

## Immediate Action

**For your current analysis (age-specific population growth):**

Use the **old file** (`hawaii_ctc_full_population_20250819_132333.parquet`):
- 527,631 filers is closer to target than 1,046,345
- Filing status distribution is more reasonable
- Can apply calibration weights if needed

**The MFS issue is solved in the code** - just need to fix the overcounting before regenerating.

## Files Created

- `data/processed/tax_units_regenerated_20251014_194028.parquet` - New file (overcounted)
- `scripts/regenerate_tax_units.py` - Regeneration script
- `COVERAGE_GAP_REASSESSMENT_DOTAX.md` - Gap analysis
- `MFS_DIAGNOSIS_SUMMARY.md` - MFS investigation
- `FILING_STATUS_CALIBRATION_PLAN.md` - Calibration strategy
- `TAX_UNITS_REGENERATION_SUMMARY.md` - This file

## Conclusion

**MFS filers are now being created** (3.7% vs 2.5% target), proving the updated logic works. However, the constructor has an **overcounting issue** that needs to be resolved before the new file can be used.

**For now, continue using the old file** for your analysis. The overcounting issue is a separate problem that requires debugging the household processing logic.
