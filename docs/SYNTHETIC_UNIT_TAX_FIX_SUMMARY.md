# Synthetic Unit Tax Calculation - Fix Summary

**Date**: October 28, 2025  
**Status**: ✅ **COMPLETE**

---

## Problem Statement

Synthetic ultra-high-income units ($5M, $10M, $25M, $50M) were being created by `UltraHighIncomeSynthesizerV2` but had **$0 tax liability** in the final output, despite having proper AGI values.

---

## Root Cause Analysis

### Issue 1: Incomplete Field Initialization
**Location**: `src/tax/adjustments/ultra_high_income_synthesizer_v2.py` (lines 164-173)

**Problem**: Synthetic units were created with minimal fields, and missing fields were filled with zeros:
```python
# Old code
else:
    synthetic_df[col] = 0  # Default to 0 for other fields
```

**Impact**: Critical fields like `income`, `taxable_income`, `standard_deduction` were set to 0, causing the tax calculator to produce $0 tax.

### Issue 2: Final Gap Closer Only Handled NaN
**Location**: `src/tax/adjustments/final_gap_closer.py` (line 118)

**Problem**: The `step3_calculate_synthetic_taxes` method only recalculated taxes for synthetic units with **NaN** tax values, not **zero** values:
```python
# Old code
needs_calc = synthetic_mask & nan_mask  # Only NaN, not zero
```

**Impact**: Synthetic units with $0 tax were skipped during the final gap closer step.

---

## Solution Implemented

### Fix 1: Proper Field Initialization
**File**: `src/tax/adjustments/ultra_high_income_synthesizer_v2.py`

**Changes**:
- Set `income`, `agi_without_cap_gains`, `agi_with_cap_gains` to AGI value
- Set `standard_deduction` to $25,900 (MFJ 2022)
- Set `taxable_income` to AGI - standard deduction
- Generate unique IDs for synthetic units
- Set appropriate defaults for all other fields

**Code**:
```python
elif col in ['income', 'agi_without_cap_gains', 'agi_with_cap_gains']:
    synthetic_df[col] = synthetic_df['agi']
elif col in ['standard_deduction', 'standard_deduction_amount']:
    synthetic_df[col] = 25900  # MFJ 2022
elif col in ['taxable_income', 'hi_taxable_income', 'hi_tax_taxable_income']:
    synthetic_df[col] = synthetic_df['agi'] - 25900
# ... etc
```

### Fix 2: Handle Zero Tax Values
**File**: `src/tax/adjustments/final_gap_closer.py`

**Changes**:
- Check for both NaN **and** zero tax values
- Use `load_tax_data()` calculator instead of `HawaiiTaxCalculator()`
- Calculate tax from AGI and filing status (not from taxable_income)
- Update multiple tax-related columns

**Code**:
```python
# Check for NaN or zero tax
nan_mask = result['hi_state_tax'].isna()
zero_mask = (result['hi_state_tax'] == 0) | (result['hi_state_tax'] == '0')

needs_calc = synthetic_mask & (nan_mask | zero_mask)

# Calculate tax using AGI and filing status
tax_result = calculator.calculate_tax(
    income=agi,
    year=2022,
    filing_status=filing_status
)
```

---

## Results

### Before Fix
```
Synthetic Unit Details:
AGI       Weight    Tax         Weighted Tax    Effective Rate
$5M       22.5      $0          $0.00M          0.00%
$10M      7.3       $0          $0.00M          0.00%
$25M      1.4       $0          $0.00M          0.00%
$50M      5.1       $0          $0.00M          0.00%

$1M+ bracket: $206.0M (vs $663M target, -$457M gap)
Total gap: -19.7%
```

### After Fix
```
Synthetic Unit Details:
AGI       Weight    Tax         Weighted Tax    Effective Rate
$5M       22.5      $529,464    $11.92M         10.59%
$10M      7.3       $1,079,464  $7.89M          10.79%
$25M      1.4       $2,729,464  $3.79M          10.92%
$50M      5.1       $5,479,464  $27.82M         10.96%

$1M+ bracket: $253.3M (vs $663M target, -$409.7M gap)
Total gap: -18.6%
Synthetic contribution: $51.4M (20.3% of $1M+ tax)
```

### Improvement
- **Synthetic tax contribution**: $0M → $51.4M ✅
- **$1M+ bracket**: $206.0M → $253.3M (+$47.3M, +23% improvement)
- **Total gap**: -19.7% → -18.6% (+1.1 percentage points)
- **Synthetic units**: 4 units contributing 2.09% of total tax

---

## Validation

### Tax Calculation Accuracy
All synthetic units now have realistic tax liabilities:
- Effective rates: 10.59% - 10.96% (appropriate for ultra-high incomes)
- Tax amounts match Hawaii tax calculator expectations
- Weighted contributions properly included in totals

### Integration with Pipeline
- Synthetic units persist through all calibration steps
- Final Gap Closer correctly identifies and processes them
- Output file contains proper tax values

---

## Files Modified

1. **`src/tax/adjustments/ultra_high_income_synthesizer_v2.py`**
   - Lines 164-204: Enhanced field initialization logic
   - Proper defaults for all DataFrame columns

2. **`src/tax/adjustments/final_gap_closer.py`**
   - Lines 99-151: Enhanced `step3_calculate_synthetic_taxes` method
   - Handles both NaN and zero tax values
   - Uses proper tax calculator

3. **`scripts/regenerate_tax_units.py`**
   - No changes required (fix handled in modules)

---

## Key Learnings

1. **Field initialization matters**: Synthetic units need all fields properly initialized, not just AGI and filing status.

2. **Zero vs NaN**: Tax values of $0 are different from NaN - both need to be handled.

3. **Tax calculator usage**: Must use `load_tax_data()` and pass AGI + filing status, not just taxable income.

4. **Multi-column updates**: When fixing tax, update all related columns (`hi_state_tax`, `hi_tax_tax_liability`, `hi_taxable_income`, etc.).

---

## Impact on Overall Calibration

### $1M+ Bracket
- **Before**: $206.0M (-69.3% from $663M target)
- **After**: $253.3M (-61.8% from $663M target)
- **Improvement**: +$47.3M (+7.5 percentage points)

### Total Tax
- **Before**: $2,425.6M (-19.9% from $3,029M target)
- **After**: $2,465.7M (-18.6% from $3,029M target)
- **Improvement**: +$40.1M (+1.3 percentage points)

### Remaining Gap
The remaining $409.7M gap in the $1M+ bracket is due to:
1. **PUMS data limitation**: Top-coded around $2M, missing $100M+, $500M+ earners
2. **Conservative synthesis**: Only 4 synthetic units (36.4 weighted filers)
3. **Need for IRS SOI data**: Actual superbracket counts ($10M+, $50M+, $100M+)

---

## Next Steps (Optional)

1. **Increase synthetic unit allocation**: Boost tail multiplier from 0.45 to 0.60-0.80
2. **Add more income tiers**: Include $100M, $250M, $500M synthetic units
3. **Integrate IRS SOI superbracket data**: Use actual counts for validation
4. **Validate against DOTAX**: Compare with Hawaii administrative records

---

## Conclusion

✅ **Synthetic unit tax calculation is now working correctly**

The fix successfully addresses the root cause of zero tax values for synthetic units. The $51.4M contribution from synthetic units represents a significant improvement in the $1M+ bracket calibration, though further enhancement requires integration of IRS SOI or DOTAX administrative data for ultra-high earners.

**Status**: Ready for production use with optional enhancements available.

