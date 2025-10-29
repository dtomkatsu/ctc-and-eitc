# Ultra-High-Income Synthesis Enhancement Plan

## Executive Summary

The current ultra-high-income synthesis achieves a **-13.0% tax gap**, but the $1M+ bracket remains **$449M short** of the $663M target. This document outlines a systematic approach to close this gap through:

1. **Pareto sensitivity analysis** (test alpha 1.3-1.6)
2. **Superbracket targeting** (use IRS SOI $10M+, $50M+, $100M+ data)
3. **Increased tail weight** (boost multiplier from 0.15 to 0.25-0.45)
4. **Synthetic unit persistence** (ensure synthetic filers survive to output)
5. **Fractional unit support** (allow non-integer weights for extreme earners)

---

## Current State Analysis

### $1M+ Bracket Performance
```
Current:  $213.9M tax from 1,824 filers
Target:   $663.0M tax
Gap:      -$449.1M (-67.7%)
```

### Root Causes
1. **Synthetic units not persisting** – `is_synthetic_ultra_high` column missing from output
2. **Conservative tail multiplier** – 0.15 factor leaves $50M+ underrepresented
3. **Fixed Pareto alpha** – No sensitivity analysis on exponent (1.3-1.6 range)
4. **No superbracket constraints** – Not using IRS SOI $10M+, $50M+, $100M+ targets
5. **Missing extreme earners** – No $50M+ filers in current output

---

## Enhancement Strategy

### Phase 1: Immediate Fixes (High Priority)

#### 1.1 Fix Synthetic Unit Persistence
**Problem**: Synthetic ultra-high-income units created but not saved to output file

**Solution**: 
- Ensure `is_synthetic_ultra_high` column is preserved through all calibration stages
- Add column to DataFrame before final save
- Verify in `scripts/regenerate_tax_units.py`

**Implementation**:
```python
# In regenerate_tax_units.py, after ultra-high synthesis:
if 'is_synthetic_ultra_high' not in tax_units.columns:
    tax_units['is_synthetic_ultra_high'] = False
# Ensure synthetic units are marked
tax_units.loc[tax_units['agi'] >= 5_000_000, 'is_synthetic_ultra_high'] = True
```

#### 1.2 Increase Tail Weight Multiplier
**Problem**: Current 0.15 multiplier too conservative for $50M+ tier

**Solution**: Boost to 0.25-0.40 range based on Pareto sensitivity analysis

**Expected Impact**: +$100-200M in $1M+ tax (10-30% improvement)

### Phase 2: Pareto Sensitivity Analysis (Medium Priority)

#### 2.1 Test Alpha Values
**Approach**: Run `scripts/analyze_pareto_sensitivity.py` with:
- Alpha values: 1.3, 1.4, 1.454, 1.5, 1.6
- Tail multipliers: 0.15, 0.25, 0.35, 0.45

**Expected Output**:
```
Pareto Alpha: 1.5
Tail Multiplier: 0.35
Estimated $1M+ Tax: $380M
Gap to Target: -$283M (57% improvement vs baseline)
```

**Files**:
- `scripts/analyze_pareto_sensitivity.py` – Sensitivity analysis script
- `data/analysis/pareto_sensitivity_results.csv` – Results table

#### 2.2 Select Optimal Configuration
Based on sensitivity analysis, choose (alpha, tail_mult) that minimizes gap to $663M target.

### Phase 3: Superbracket Targeting (Medium Priority)

#### 3.1 Integrate IRS SOI Data
**Data Needed** (from DOTAX or IRS SOI):
```python
IRS_SUPERBRACKET_TARGETS = {
    (10_000_000, float('inf')): {'filers': 5, 'avg_agi': 25_000_000},
    (50_000_000, float('inf')): {'filers': 2, 'avg_agi': 75_000_000},
    (100_000_000, float('inf')): {'filers': 1, 'avg_agi': 150_000_000},
}
```

**Implementation**:
- Add superbracket constraints to `UltraHighIncomeSynthesizerV2`
- Ensure synthetic filer counts match IRS targets exactly
- Validate in output diagnostics

#### 3.2 Validate Synthetic Unit Counts
**Verification**:
```python
# After synthesis, check:
assert (df['agi'] >= 10_000_000).sum() >= 5  # At least 5 filers
assert (df['agi'] >= 50_000_000).sum() >= 2  # At least 2 filers
assert (df['agi'] >= 100_000_000).sum() >= 1 # At least 1 filer
```

### Phase 4: Fractional Unit Support (Low Priority)

#### 4.1 Allow Non-Integer Weights
**Rationale**: SOI public use files use fractional weights for extreme earners

**Implementation**:
- Remove integer constraints on synthetic filer weights
- Allow weights like 0.5, 1.5, 2.3 (already supported in current code)
- Document in code comments

---

## Implementation Roadmap

### Step 1: Fix Synthetic Unit Persistence (Today)
```bash
# Edit scripts/regenerate_tax_units.py
# Add column preservation before final save
```

### Step 2: Run Pareto Sensitivity Analysis (Today)
```bash
python scripts/analyze_pareto_sensitivity.py
# Review results and select optimal (alpha, tail_mult)
```

### Step 3: Update Ultra-High Synthesis (Tomorrow)
```bash
# Use UltraHighIncomeSynthesizerV2 with optimal parameters
# Update regenerate_tax_units.py to use new synthesizer
```

### Step 4: Integrate Superbracket Data (This Week)
```bash
# Add IRS SOI $10M+, $50M+, $100M+ targets
# Validate synthetic unit counts
```

### Step 5: Validate and Benchmark (This Week)
```bash
# Run full pipeline with enhancements
# Compare $1M+ bracket performance
# Document improvements
```

---

## Expected Outcomes

### Conservative Estimate (Phase 1-2)
- **$1M+ tax**: $213.9M → $350-400M
- **Total gap**: -13.0% → -8.0% to -10.0%
- **Improvement**: +$136-186M (+4-5 percentage points)

### Optimistic Estimate (All Phases)
- **$1M+ tax**: $213.9M → $450-500M
- **Total gap**: -13.0% → -5.0% to -7.0%
- **Improvement**: +$236-286M (+6-8 percentage points)

---

## Files Created

1. **`src/tax/adjustments/ultra_high_income_synthesizer_v2.py`**
   - Enhanced synthesizer with configurable Pareto alpha
   - Tail multiplier support
   - Superbracket targeting framework

2. **`scripts/analyze_pareto_sensitivity.py`**
   - Sensitivity analysis for Pareto alpha (1.3-1.6)
   - Tail multiplier optimization (0.15-0.45)
   - Results saved to CSV

3. **`docs/ULTRA_HIGH_INCOME_ENHANCEMENT_PLAN.md`** (this file)
   - Comprehensive enhancement strategy
   - Implementation roadmap
   - Expected outcomes

---

## Next Steps

1. **Immediate**: Fix synthetic unit persistence in `regenerate_tax_units.py`
2. **Today**: Run Pareto sensitivity analysis
3. **Tomorrow**: Update ultra-high synthesis with optimal parameters
4. **This week**: Integrate IRS SOI superbracket data
5. **Next week**: Full validation and benchmarking

---

## References

- **Current Performance**: -13.0% gap, $213.9M $1M+ tax vs $663M target
- **Pareto Distribution**: P(X > x) = (x_min / x)^alpha
- **IRS SOI**: Actual tax return data for superbracket validation
- **DOTAX**: Hawaii Department of Taxation administrative data

