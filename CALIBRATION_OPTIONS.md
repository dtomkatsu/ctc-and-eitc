# Calibration Options for Hawaii Tax Unit Construction

## Problem Statement

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

## Recommendation

**✅ IMPLEMENT: Option 2 - Two-Layer Calibration**

### Rationale

1. **Comprehensive Solution**: Addresses both filing status AND AGI distribution simultaneously
2. **Proven Infrastructure**: Builds on existing bracket_calibration.py framework
3. **Systematic Fix**: Corrects not just $50k-$75k, but entire middle-income over-representation pattern
4. **Maintains Accuracy**: Preserves excellent filing status distribution while improving AGI matching

### Implementation Steps

1. **Create AGI calibration module** ✅ (Complete)
   - File: `src/tax/validation/agi_calibration.py`
   - Function: `apply_two_layer_calibration()`

2. **Update regenerate_tax_units.py**
   - Add option flag: `--two-layer-calibration`
   - Default: Keep current single-layer
   - Optional: Enable two-layer for improved AGI matching

3. **Validate results**
   - Run Table 12A validation
   - Check filing status distribution (should remain ~100%)
   - Verify $50k-$75k accuracy improvement
   - Confirm total tax liability remains accurate

4. **Compare approaches**
   - Generate side-by-side comparison report
   - Document trade-offs and final decision

### Success Criteria

| Metric | Target |
|--------|--------|
| $50k-$75k returns | Within ±5% of 91,459 |
| $50k-$75k tax liability | Within ±10% of $293M |
| Filing status totals | Within ±2% of DOTAX |
| Total tax liability | Within ±2% of $3,029M |
| AGI brackets within ±10% | >75% (vs current 8.3%) |

## Implementation Code

The two-layer calibration is now available:

```python
from src.tax.validation.agi_calibration import apply_two_layer_calibration

# Apply both layers
tax_units_calibrated = apply_two_layer_calibration(
    tax_units, 
    weight_col='weight'
)

# Use 'weight_final' column for downstream analysis
```

## Next Steps

1. ✅ Review this document and approve approach
2. ⏳ Update `regenerate_tax_units.py` with flag for two-layer calibration
3. ⏳ Run comparison: single-layer vs two-layer
4. ⏳ Validate results and document findings
5. ⏳ Make decision on default calibration method
