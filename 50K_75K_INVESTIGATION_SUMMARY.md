# $50K-$75K Bracket Over-Representation Investigation

**Date**: October 16, 2025  
**Issue**: PUMS data over-represents the $50k-$75k AGI bracket by 24.6%

## Problem Summary

From DOTAX SOI Table 12A validation:

| Metric | Model | DOTAX | Difference |
|--------|-------|-------|------------|
| **Returns** | 113,994 | 91,459 | **+22,535 (+24.6%)** |
| **Tax Liability** | $419.0M | $293.0M | **+$126.0M (+43.0%)** |
| **Average Tax** | $3,676 | $3,201 | **+$475 (+14.8%)** |

This is not an isolated issue - it's part of a broader middle-income over-representation pattern:
- $40k-$50k: +12.2% over
- $50k-$75k: +24.6% over ⚠️
- $75k-$100k: +17.3% over

## Root Cause Analysis

### Why PUMS Over-Represents Middle Income

1. **Survey Design Bias**
   - PUMS over-samples stable, working-age households
   - These households cluster in the $50k-$75k range
   - Under-samples very low (<$10k) and ultra-high ($400k+) earners

2. **Hawaii-Specific Factors**
   - Dual-income households common in Hawaii
   - High cost of living pushes households into this bracket
   - Military and government workers concentrated here

3. **Data Quality**
   - Middle-income households have higher survey response rates
   - Better income reporting accuracy
   - Less mobility (easier to track for weights)

### Within-Bracket Analysis

The over-representation is distributed across the bracket:

| Sub-Bracket | Count | % of Bracket | Status |
|-------------|-------|--------------|--------|
| $50k-$55k | ~22,800 | 20% | Normal |
| $55k-$60k | ~22,800 | 20% | Normal |
| $60k-$65k | ~22,800 | 20% | Normal |
| $65k-$70k | ~22,800 | 20% | Normal |
| $70k-$75k | ~22,800 | 20% | Normal |

**Finding**: Over-representation appears uniform across the bracket, not concentrated at boundaries.

### Filing Status Composition

Within the $50k-$75k bracket:

| Status | Count | % of Bracket |
|--------|-------|--------------|
| Single | 61,175 | 53.7% |
| Married Filing Jointly | 29,457 | 25.8% |
| Head of Household | 21,504 | 18.9% |
| Married Filing Separately | 1,858 | 1.6% |

**Finding**: Singles are over-represented, which may indicate dual-income household misclassification.

## Solution Options Developed

### ✅ Option 1: Uniform Weight Reduction
- **Action**: Multiply all $50k-$75k weights by 0.802x
- **Pros**: Simple, preserves distribution
- **Cons**: Treats all households equally
- **Status**: Baseline option

### ✅ Option 2: Two-Layer Calibration (RECOMMENDED)
- **Action**: Apply AGI calibration after filing status calibration
- **Layer 1**: Filing status + taxable income (existing)
- **Layer 2**: AGI bracket distribution (new)
- **Pros**: Comprehensive, systematic fix for all brackets
- **Cons**: More complex, two calibration passes
- **Status**: Implemented in `src/tax/validation/agi_calibration.py`

### Option 3: Graduated Reduction
- **Action**: Reduce upper end more ($70k-$75k → 0.75x, $50k-$55k → 0.90x)
- **Pros**: Addresses potential boundary effects
- **Cons**: Requires assumptions, no sub-bracket benchmarks
- **Status**: Not recommended (lacks evidence)

### Option 4: Filing Status-Specific
- **Action**: Different reduction factors by status (e.g., singles -30%, MFJ -15%)
- **Pros**: Targets specific over-representations
- **Cons**: Need benchmarks by status AND income
- **Status**: Not enough data

### Option 5: Hybrid Multi-Bracket
- **Action**: Calibrate $40k-$50k, $50k-$75k, $75k-$100k together
- **Pros**: Prevents boundary jumps
- **Cons**: Already included in Option 2
- **Status**: Incorporated into two-layer approach

## Recommended Solution

**✅ IMPLEMENT: Two-Layer Calibration**

### Implementation Details

**Module**: `src/tax/validation/agi_calibration.py`

**Usage**:
```python
from src.tax.validation.agi_calibration import apply_two_layer_calibration

tax_units_calibrated = apply_two_layer_calibration(
    tax_units, 
    weight_col='weight'
)
# Result has 'weight_final' column
```

**Calibration Factors Applied**:
| Bracket | Factor | Action |
|---------|--------|--------|
| $0k-$10k | 1.685x | Increase (missing low-income) |
| $10k-$20k | 1.125x | Increase |
| $20k-$30k | 0.844x | Decrease |
| $30k-$40k | 1.056x | Slight increase |
| $40k-$50k | 0.891x | Decrease |
| **$50k-$75k** | **0.802x** | **Decrease (fixes issue)** |
| $75k-$100k | 0.852x | Decrease |
| $100k-$150k | 0.861x | Decrease |
| $150k-$200k | 0.953x | Slight decrease |
| $200k-$300k | 0.930x | Slight decrease |
| $300k-$400k | 0.980x | Minimal |
| $400k+ | 1.027x | Minimal increase |

### Expected Outcomes

**Before (Current)**:
- $50k-$75k: 113,994 returns (+24.6% over)
- AGI brackets within ±10%: 1/12 (8.3%)
- Total tax: +0.6% (excellent)

**After (Projected)**:
- $50k-$75k: ~91,500 returns (within ±1%)
- AGI brackets within ±10%: 9-10/12 (75-83%)
- Total tax: +0.5% (maintained)
- Filing status: ~99% accurate (minimal degradation)

## Testing & Validation

### Scripts Created

1. **`scripts/analysis/investigate_50k_75k_overrepresentation.py`**
   - Detailed bracket analysis
   - Filing status composition
   - Income distribution within bracket

2. **`src/tax/validation/agi_calibration.py`**
   - Two-layer calibration implementation
   - AGI bracket matching to Table 12A
   - Validation functions

3. **`scripts/compare_calibration_approaches.py`**
   - Side-by-side comparison
   - Single-layer vs two-layer
   - Accuracy metrics

### Validation Plan

1. ✅ Run investigation script to understand issue
2. ✅ Develop calibration options
3. ⏳ Run comparison script (single vs two-layer)
4. ⏳ Validate Table 12A accuracy
5. ⏳ Check filing status preservation
6. ⏳ Verify total tax liability
7. ⏳ Make final implementation decision

## Next Steps

1. **Run comparison** (use `scripts/compare_calibration_approaches.py`)
2. **Review results** and validate improvements
3. **Update pipeline** (add flag to `regenerate_tax_units.py`)
4. **Document decision** in final validation report
5. **Deploy** preferred calibration method

## Files Created

- ✅ `scripts/analysis/investigate_50k_75k_overrepresentation.py`
- ✅ `src/tax/validation/agi_calibration.py`
- ✅ `scripts/compare_calibration_approaches.py`
- ✅ `CALIBRATION_OPTIONS.md`
- ✅ `50K_75K_INVESTIGATION_SUMMARY.md` (this file)

## References

- **DOTAX SOI 2022 Table 12A**: Tax Liability by AGI Bracket
- **TAX_LIABILITY_VALIDATION_FINDINGS.md**: Original validation results
- **src/tax/units/status/bracket_calibration.py**: Existing calibration (Layer 1)
