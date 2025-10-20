# Calibration Options for Hawaii Tax Unit Construction

## ✅ CURRENT STATUS: IPF Calibration Implemented (October 2025)

**The IRS SOI calibration now uses Iterative Proportional Fitting (IPF) by default**, achieving:
- **Filing Status Accuracy**: <0.1% error on all categories
- **AGI Bracket Accuracy**: <0.1% error on all brackets
- **Convergence**: Typically within 50 iterations
- **Method**: Simultaneous balancing of both filing status and AGI distributions

This document is maintained for historical reference and to explain the evolution of the calibration approach.

---

## Problem Statement (Historical)

The $50k-$75k AGI bracket is significantly over-represented in PUMS data:
- **Model**: 113,994 returns (+24.6%)
- **DOTAX**: 91,459 returns (benchmark)
- **Tax Liability**: $419.0M vs $293.0M (+43.0%)

This is part of a broader middle-income over-representation pattern affecting $40k-$100k brackets.

## Root Cause

PUMS (Public Use Microdata Sample) systematically over-represents middle-income households:
1. **Survey design**: PUMS over-samples certain demographics that cluster in middle income
2. **Geographic weighting**: Hawaii's unique population distribution may not align with standard PUMS weights
3. **Income reporting**: Middle-income households may have better response rates

## Correction Options

### Option 1: Single-Layer Calibration (Current Implementation)

**What it does:**
- Calibrates by **filing status** (Single, MFJ, HoH, MFS)
- Uses **taxable income brackets** within each filing status
- Matches DOTAX Table 13A/13B/13C benchmarks

**Strengths:**
- ✅ Perfect match on filing status totals
- ✅ Well-tested and stable
- ✅ Preserves within-status income distribution

**Weaknesses:**
- ❌ Doesn't directly address AGI distribution
- ❌ $50k-$75k still over-represented (+24.6%)
- ❌ Overall income distribution doesn't match Table 12A

**Current Results:**
| Metric | Result |
|--------|--------|
| Filing status accuracy | ✅ 100% (perfect) |
| AGI bracket accuracy | ⚠️ 8.3% within ±10% |
| Total tax liability | ✅ +0.6% (excellent) |

---

### Option 2: Two-Layer Calibration (Recommended)

**What it does:**
- **Layer 1**: Filing status + taxable income (current)
- **Layer 2**: AGI bracket distribution (new)
- Applies sequential calibration to match both benchmarks

**How it works:**
```
Step 1: Apply filing status calibration (Table 13A/13B/13C)
  → Ensures correct Single/MFJ/HoH/MFS totals
  → Preserves within-status income patterns

Step 2: Apply AGI bracket calibration (Table 12A)
  → Adjusts weights to match AGI distribution
  → Fixes $50k-$75k over-representation
  → Required factor: 0.802x (reduces by ~20%)
```

**Calibration Factors (Layer 2):**
| AGI Bracket | Current | Target | Factor | Action |
|-------------|---------|--------|--------|--------|
| $0k-$10k | 76,769 | 129,376 | 1.685x | Increase |
| $10k-$20k | 57,010 | 64,160 | 1.125x | Increase |
| $20k-$30k | 68,552 | 57,835 | 0.844x | Decrease |
| $30k-$40k | 56,674 | 59,827 | 1.056x | Slight increase |
| $40k-$50k | 60,115 | 53,555 | 0.891x | Decrease |
| **$50k-$75k** | **113,994** | **91,459** | **0.802x** | **Decrease** |
| $75k-$100k | 64,505 | 54,976 | 0.852x | Decrease |
| $100k-$150k | 72,087 | 62,065 | 0.861x | Decrease |
| $150k-$200k | 29,365 | 27,976 | 0.953x | Slight decrease |
| $200k-$300k | 20,353 | 18,937 | 0.930x | Slight decrease |
| $300k-$400k | 6,198 | 6,076 | 0.980x | Minimal change |
| $400k+ | 8,644 | 8,875 | 1.027x | Minimal increase |

**Strengths:**
- ✅ Fixes $50k-$75k over-representation
- ✅ Matches both filing status AND AGI distributions
- ✅ Addresses middle-income over-sampling systematically
- ✅ Smooth transitions between brackets

**Potential Concerns:**
- ⚠️ Two calibration layers may compound adjustment errors
- ⚠️ Need to validate filing status distribution remains accurate
- ⚠️ More complex to debug if issues arise

**Expected Results:**
| Metric | Expected |
|--------|----------|
| Filing status accuracy | ✅ ~99% (minimal degradation) |
| AGI bracket accuracy | ✅ ~90% within ±10% |
| $50k-$75k accuracy | ✅ Near-perfect match |
| Total tax liability | ✅ +0.5% or better |

---

### Option 3: Uniform AGI Calibration (Simple Alternative)

**What it does:**
- Apply AGI calibration only (skip filing status calibration)
- Single-pass adjustment based purely on Table 12A

**Strengths:**
- ✅ Simplest approach
- ✅ Directly targets AGI distribution
- ✅ Easy to understand and validate

**Weaknesses:**
- ❌ May not preserve filing status accuracy
- ❌ Less granular than two-layer approach
- ❌ Could create filing status distribution issues

---

### Option 4: Graduated Reduction Within Bracket

**What it does:**
- Apply different reduction factors within the $50k-$75k bracket
- Higher incomes get larger reductions (e.g., $70k-$75k reduced more than $50k-$55k)

**Rationale:**
- PUMS may over-sample upper end of middle-income range
- Addresses "bracket boundary effects"

**Implementation:**
- $50k-$55k: 0.90x reduction
- $55k-$60k: 0.85x reduction
- $60k-$65k: 0.82x reduction
- $65k-$70k: 0.78x reduction
- $70k-$75k: 0.75x reduction

**Strengths:**
- ✅ More nuanced within-bracket adjustment
- ✅ Addresses potential boundary artifacts

**Weaknesses:**
- ❌ Requires assumption about where excess occurs
- ❌ More complex implementation
- ❌ Harder to validate without sub-bracket benchmarks

---

## Recommendation (UPDATED: October 2025)

**✅ IMPLEMENTED: IPF Calibration (Iterative Proportional Fitting)**

### Why IPF Won

After testing all options, IPF emerged as the clear winner:

1. **Superior Accuracy**: Achieves <0.1% error on both filing status AND AGI brackets simultaneously
2. **Theoretically Sound**: Standard method in survey statistics for balancing multiple constraints
3. **Automatic Convergence**: No manual tuning of calibration factors needed
4. **Robust**: Handles edge cases and missing data gracefully
5. **Proven**: Used by Census Bureau, BLS, and other statistical agencies

### Comparison Results

| Method | Filing Status Accuracy | AGI Bracket Accuracy | Complexity |
|--------|----------------------|---------------------|------------|
| Single-Layer | ✅ 100% | ⚠️ 8.3% within ±10% | Low |
| Two-Layer | ⚠️ ~95% | ✅ ~90% within ±10% | Medium |
| **IPF** | **✅ >99.9%** | **✅ >99.9%** | **Medium** |

### Implementation

The IPF calibration is now the default in `apply_irs_soi_calibration()`:

```python
from src.tax.validation.irs_soi_calibration import apply_irs_soi_calibration

# Simple usage - IPF is applied automatically
tax_units_calibrated = apply_irs_soi_calibration(
    tax_units,
    weight_col='weight'
)

# Use 'weight_irs_calibrated' column for downstream analysis
```

### Success Criteria (ACHIEVED)

| Metric | Target | Actual |
|--------|--------|--------|
| $50k-$75k returns | Within ±5% of 91,459 | ✅ <0.1% error |
| Filing status totals | Within ±2% of DOTAX | ✅ <0.1% error |
| Total tax liability | Within ±2% of $3,029M | ✅ <1% error |
| AGI brackets within ±1% | >75% | ✅ 100% |

## Historical Context

This document originally proposed several calibration approaches:
1. **Single-Layer**: Filing status only (original implementation)
2. **Two-Layer**: Sequential filing status + AGI calibration
3. **Uniform AGI**: AGI calibration only
4. **Graduated Reduction**: Within-bracket adjustments

After implementation and testing, **IPF proved superior to all alternatives** and is now the production standard.
