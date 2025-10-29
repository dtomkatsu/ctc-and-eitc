# Capital Gains Calibration - Final Recommendation

**Date**: October 29, 2025  
**Status**: ✅ **ANALYSIS COMPLETE - READY FOR IMPLEMENTATION**

---

## Executive Summary

Analysis reveals that **existing non-synthetic $1M+ filers already have 45.98% capital gains**, which is **significantly higher than DOTAX's 20.9% target** for $400K+ earners. This requires a revised calibration strategy for synthetic units.

---

## Key Findings

### Current State

**$1M+ Bracket Composition**:
- Non-synthetic filers: 203 units, weight 1,759
  - AGI: $2,131.6M
  - Capital gains: $980.1M
  - **CG share: 45.98%**

- Synthetic filers: 4 units, weight 65
  - AGI: $1,188.3M
  - Capital gains: $0M (currently)
  - **CG share: 0%**

**Blended Result (current)**:
- Total $1M+ AGI: $3,319.9M
- Total $1M+ CG: $980.1M
- **Blended CG share: 29.52%**

### Comparison to DOTAX

| Source | CG Share |
|--------|----------|
| DOTAX $400K+ | 20.9% |
| DOTAX $400K+ (nonresidents) | 56.2% |
| Current $1M+ (blended) | 29.52% |
| Current $1M+ (non-synthetic only) | 45.98% |

**Key Insight**: The existing non-synthetic $1M+ filers have **2.2x the DOTAX target** (45.98% vs 20.9%)

---

## Why Existing CG is So High

### Possible Explanations

1. **Earlier calibration steps**: Capital gains may have been populated during prior adjustments
2. **Population composition**: The $1M+ PUMS sample may skew toward investment-heavy earners
3. **Data source differences**: PUMS vs DOTAX may have different methodologies
4. **Synthetic unit creation**: When synthetic units were added, existing CG wasn't adjusted

### Evidence

- All 203 non-synthetic $1M+ filers have capital gains populated
- Range: $449,812 to $987,585 per filer
- Mean: $563,187 per filer
- Median: $518,136 per filer
- Highly consistent (not outliers)

---

## Calibration Strategy

### Option A: Match DOTAX 20.9% (Conservative)

**Approach**: Set synthetic units to 20.9% CG share

**Result**:
- Synthetic CG: $249.5M (20.9% of $1,188.3M)
- Total $1M+ CG: $1,229.6M
- **Blended CG share: 37.04%**
- Still above DOTAX 20.9% (existing non-synthetic pulls it up)

**Pros**:
- ✅ Matches DOTAX $400K+ benchmark
- ✅ Conservative (doesn't inflate CG)
- ✅ Aligns with Hawaii tax data

**Cons**:
- ⚠️ Blended still 37% (vs 20.9% target)
- ⚠️ Doesn't address existing non-synthetic CG being too high

### Option B: Match National Data 47% (Aggressive)

**Approach**: Set synthetic units to 47% CG share (national $10M+)

**Result**:
- Synthetic CG: $558.5M (47% of $1,188.3M)
- Total $1M+ CG: $1,538.6M
- **Blended CG share: 46.33%**
- Matches existing non-synthetic level

**Pros**:
- ✅ Based on national IRS SOI data
- ✅ Realistic for ultra-high earners
- ✅ Consistent with national patterns

**Cons**:
- ⚠️ Blended 46% (vs 20.9% DOTAX target)
- ⚠️ Significantly above DOTAX

### Option C: Recalibrate Existing + Set Synthetic (Ideal)

**Approach**: 
1. Reduce existing non-synthetic $1M+ CG from 45.98% to 20.9%
2. Set synthetic units to 20.9% CG share

**Result**:
- Non-synthetic CG: $447.5M (20.9% of $2,131.6M)
- Synthetic CG: $249.5M (20.9% of $1,188.3M)
- Total $1M+ CG: $697.0M
- **Blended CG share: 20.98%** ✅ **PERFECT MATCH TO DOTAX**

**Pros**:
- ✅ Perfectly matches DOTAX 20.9% target
- ✅ Aligns entire $1M+ bracket with Hawaii data
- ✅ Consistent methodology

**Cons**:
- ⚠️ Requires modifying existing non-synthetic CG
- ⚠️ May impact tax calculations if already done
- ⚠️ Larger change to existing data

---

## Recommendation

### **Proceed with Option A (Conservative)**

**Rationale**:
1. **Minimal disruption**: Only adds CG to synthetic units, doesn't modify existing
2. **Defensible**: Based on DOTAX $400K+ benchmark (20.9%)
3. **Realistic**: National data supports 30-50% for ultra-high earners
4. **Pragmatic**: Existing non-synthetic CG is already in place; synthetic units should match

**Implementation**:
- Set all synthetic tiers to **20.9% capital gains share**
- Ordinary income: **79.1%**
- Results in blended $1M+ CG of ~37% (above DOTAX but reasonable)

### Synthetic Unit Capital Gains Shares

| AGI Tier | CG Share | Ordinary Income | Rationale |
|----------|----------|-----------------|-----------|
| $5M | 20.9% | 79.1% | Match DOTAX $400K+ |
| $10M | 20.9% | 79.1% | Match DOTAX $400K+ |
| $25M | 20.9% | 79.1% | Match DOTAX $400K+ |
| $50M | 20.9% | 79.1% | Match DOTAX $400K+ |

---

## Validation Results

### With Option A (20.9% synthetic CG)

```
$1M+ Bracket Summary:
  Non-synthetic: 1,759 weight, $2,131.6M AGI, $980.1M CG (45.98%)
  Synthetic: 65 weight, $1,188.3M AGI, $249.5M CG (20.9%)
  
  Total: 1,824 weight, $3,319.9M AGI, $1,229.6M CG
  Blended CG share: 37.04%
  
Comparison:
  DOTAX $400K+ target: 20.9%
  Current blended: 37.04%
  Difference: +16.14 percentage points
  
Note: Existing non-synthetic filers pull blended CG up to 37%.
      Synthetic units alone (20.9%) are properly calibrated to DOTAX.
```

---

## Files Created

1. ✅ `data/external/synthetic_capital_gains_calibrated_shares.csv` - Calibrated shares
2. ✅ `docs/CAPITAL_GAINS_NATIONAL_DATA_ANALYSIS.md` - National data analysis
3. ✅ `docs/CAPITAL_GAINS_INTEGRATION_PLAN.md` - Implementation plan
4. ✅ `docs/CAPITAL_GAINS_CALIBRATION_FINAL.md` - This document

---

## Implementation Steps

### Step 1: Update UltraHighIncomeSynthesizerV2

**File**: `src/tax/adjustments/ultra_high_income_synthesizer_v2.py`

```python
# Add capital gains share mapping
CAPITAL_GAINS_SHARES = {
    5_000_000: 0.209,   # 20.9% (DOTAX $400K+)
    10_000_000: 0.209,  # 20.9% (DOTAX $400K+)
    25_000_000: 0.209,  # 20.9% (DOTAX $400K+)
    50_000_000: 0.209,  # 20.9% (DOTAX $400K+)
}
```

### Step 2: Update Synthetic Filer Creation

Populate `capital_gains` and `has_capital_gains` fields based on CAPITAL_GAINS_SHARES

### Step 3: Run Pipeline

Execute `scripts/regenerate_tax_units.py` with updated synthesizer

### Step 4: Validate

Run validation to confirm:
- ✅ Synthetic units have capital gains populated (20.9%)
- ✅ Tax calculations reflect CG treatment
- ✅ $1M+ bracket CG share is ~37% (blended)
- ✅ Total gap remains acceptable

---

## Next Steps

1. **Confirm Hawaii CG tax treatment** (critical for tax impact)
   - Does Hawaii tax CG at preferential rates?
   - Or as ordinary income?

2. **Implement Option A** (20.9% synthetic CG)

3. **Run full pipeline** and measure impact

4. **Validate** against DOTAX aggregates

5. **Document findings** and adjust if needed

---

## Decision Required

**Should we proceed with Option A implementation?**

- ✅ Set synthetic units to 20.9% CG (DOTAX $400K+ benchmark)
- ✅ Leave existing non-synthetic CG unchanged (45.98%)
- ✅ Results in ~37% blended $1M+ CG (above DOTAX but reasonable)

**Or would you prefer Option C** (recalibrate existing + synthetic to 20.9%)?

---

**Status**: ✅ **READY FOR APPROVAL AND IMPLEMENTATION**

