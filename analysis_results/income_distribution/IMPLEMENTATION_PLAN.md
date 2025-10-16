# Implementation Plan: Fix Income Bracket Alignment

## Summary of Issues

### Priority 1: Income Definition Mismatch ⭐⭐⭐ CRITICAL
- **Problem**: Using PUMS total income instead of taxable income
- **Impact**: Everyone appears in higher income brackets
- **Gap**: $13K-$26K per filer (standard deduction + AGI adjustments)

### Priority 2: Missing 146,414 Low-Income Filers ⭐⭐⭐ CRITICAL
- **Problem**: Filing threshold logic excludes 97% of low-income adults
- **Impact**: Severe undercounting in SOI brackets under $5K
- **Root cause**: `constructor.py` line 1387 excludes income below $12,950

---

## Investigation Results

### ✅ PUMS Data Coverage
- **PUMS has 160,518 low-income adults** (under $5K)
- **Constructor creates only 4,589 tax units** (under $5K)
- **Gap: 155,929 missing (97.1%)**

### ❌ Filing Threshold Issue
```python
# Line 1387 in constructor.py
if income < FILING_THRESHOLD and not has_self_employment:
    logger.debug(f"Adult {adult.name} has income ${income:.0f} below filing threshold, not creating tax unit")
    return None  # ← This excludes 97% of low-income filers!
```

**Why this is wrong**:
- SOI includes **ALL** filers, including those below the standard threshold
- People file even with low income for:
  - Refundable credits (EITC, CTC, etc.)
  - Tax refunds (withholding)
  - Self-employment income ≥$400
  - Benefits claims

---

## Solution Options

## Option A: Two-Step Fix (RECOMMENDED) ⭐⭐⭐

### Step 1: Remove/Lower Filing Threshold
**File**: `src/tax/units/constructor.py` (line 1387)

**Change**:
```python
# BEFORE:
if income < FILING_THRESHOLD and not has_self_employment:
    logger.debug(f"Adult {adult.name} has income ${income:.0f} below filing threshold, not creating tax unit")
    return None

# AFTER (Option A1 - Remove entirely):
# Removed filing threshold check - SOI includes all filers, even those with low income
# who file for refundable credits, refunds, or other purposes

# AFTER (Option A2 - Use very low threshold):
if income < 1 and not has_self_employment:  # Only exclude $0 income
    logger.debug(f"Adult {adult.name} has zero income, not creating tax unit")
    return None
```

**Impact**: 
- ✅ Will capture ~155,000 additional low-income filers
- ✅ Aligns with SOI universe definition
- ✅ Better matches real-world filing behavior

### Step 2: Use Taxable Income for SOI Comparison
**File**: Create new script or modify existing comparison

**Implementation**:
```python
from src.tax.units.taxable_income import TaxableIncomeCalculator

# Load tax units
tax_units = pd.read_parquet('data/processed/tax_units_regenerated_*.parquet')

# Calculate taxable income for SOI comparison
calculator = TaxableIncomeCalculator(tax_year=2022)
tax_units = calculator.apply_to_tax_units(
    tax_units,
    income_col='income',
    filing_status_col='filing_status',
    apply_agi_adjustments=True  # Apply AGI adjustments
)

# Use 'taxable_income' column for SOI bracket comparison
```

**Impact**:
- ✅ Shifts all filers down by $13K-$26K (standard deduction)
- ✅ Shifts additional 1-2% down (AGI adjustments)
- ✅ Matches SOI bracket definitions exactly
- ✅ Expected to reduce bracket misalignment from 70% to <20%

---

## Option B: Keep Current, Adjust for Comparison (PARTIAL FIX) ⭐⭐

### Don't modify constructor, only adjust comparison script

**Implementation**: Only apply taxable income calculation in comparison script

**Pros**:
- ✅ No changes to core constructor logic
- ✅ Preserves existing tax unit definitions

**Cons**:
- ❌ Still missing 146K low-income filers
- ❌ Only fixes income bracket alignment, not coverage
- ❌ Won't match SOI total counts

---

## Option C: Full Income Pipeline (COMPREHENSIVE) ⭐⭐⭐

### Integrate taxable income calculation into constructor

**File**: `src/tax/units/constructor.py`

**Implementation**:
```python
# In _create_single_filer (and other creation methods):

# 1. Calculate total income (current)
income = calculate_tax_unit_income(income_df)

# 2. Calculate taxable income
from tax.units.taxable_income import TaxableIncomeCalculator
calculator = TaxableIncomeCalculator(tax_year=2022)
tax_result = calculator.calculate_taxable_income(
    income, filing_status, age, has_self_employment
)

# 3. Store both in tax unit
tax_unit = {
    'total_income': income,
    'agi': tax_result['agi'],
    'agi_adjustments': tax_result['agi_adjustments'],
    'taxable_income': tax_result['taxable_income'],
    'income': tax_result['taxable_income'],  # Use taxable for comparisons
    # ... other fields
}
```

**Pros**:
- ✅ Complete income pipeline
- ✅ All income measures available for analysis
- ✅ Transparent and auditable

**Cons**:
- ⚠️ More complex changes
- ⚠️ May require updating downstream code
- ⚠️ Slower processing (small impact)

---

## AGI Adjustments - Three Options

### AGI Option 1: Enable in Comparison (SIMPLE) ⭐⭐⭐
**When**: During SOI comparison only
**How**: Set `apply_agi_adjustments=True` in `TaxableIncomeCalculator`

```python
calculator.apply_to_tax_units(
    tax_units,
    apply_agi_adjustments=True  # ← Enable here
)
```

**Impact**: ~1-2% income reduction
**Pros**: Simple, reversible, only affects comparison
**Cons**: Not reflected in core data

### AGI Option 2: Integrate in Constructor (COMPREHENSIVE) ⭐⭐
**When**: During tax unit creation
**How**: Modify income calculation in `constructor.py`

**Impact**: ~1-2% income reduction across all analyses
**Pros**: Consistent across all uses, more accurate
**Cons**: Larger code changes, slower processing

### AGI Option 3: Post-Processing (HYBRID) ⭐
**When**: After tax units created, before analysis
**How**: Create adjusted copy of tax units

```python
# Create two versions:
tax_units_raw = load_tax_units()  # Original
tax_units_soi = apply_taxable_income(tax_units_raw)  # For SOI comparison
```

**Impact**: ~1-2% income reduction
**Pros**: Flexibility, keeps both versions
**Cons**: Duplicate data, potential confusion

---

## Recommended Implementation Sequence

### Phase 1: Quick Wins (30 minutes) ⭐⭐⭐
1. **Fix filing threshold** (Option A, Step 1)
   - Edit `constructor.py` line 1387
   - Either remove check or set threshold to $1
   - Regenerate tax units

2. **Test impact**
   - Run comparison script
   - Should see ~155K more low-income filers
   - Check SOI alignment

**Expected Result**: 
- Low-income brackets: -98% → within ±10%
- Total coverage: 97.3% → ~99%

### Phase 2: Income Definition (1 hour) ⭐⭐⭐
1. **Apply taxable income calculation** (Option A, Step 2)
   - Use `TaxableIncomeCalculator` in comparison
   - Enable AGI adjustments (AGI Option 1)
   - Rerun comparison

**Expected Result**:
- Income brackets shift down by $13K-$26K
- Middle brackets: +64% → within ±15%
- High brackets: +108% → within ±20%

### Phase 3: Integration (2-3 hours) ⭐⭐
1. **Integrate into constructor** (Option C)
   - Store all income measures
   - Use taxable income for SOI comparisons
   - Keep total income for other analyses

**Expected Result**:
- Consistent income definitions
- Clean comparison workflows
- Better documentation

### Phase 4: Validation (1 hour) ⭐
1. **Run full comparison**
2. **Validate all brackets within ±10%**
3. **Document methodology**

---

## Code Locations

### Files to Modify
1. `src/tax/units/constructor.py` (line 1387)
   - Remove/lower filing threshold

2. `scripts/compare_soi_income_brackets.py`
   - Add taxable income calculation
   - Use for bracket comparison

### New Files Created
1. ✅ `src/tax/units/taxable_income.py`
   - Calculate taxable income from total income
   - Apply standard deduction + AGI adjustments

2. ✅ `scripts/diagnose_low_income_gap.py`
   - Diagnostic tool for low-income coverage

3. ✅ `src/tax/adjustments/agi_adjustments.py` (already exists!)
   - AGI adjustment estimates

---

## Testing Checklist

### After Phase 1 (Filing Threshold)
- [ ] Tax units created for low-income adults
- [ ] Under $5K bracket within ±10% of SOI
- [ ] Total coverage >99%

### After Phase 2 (Taxable Income)
- [ ] All brackets shift down appropriately
- [ ] Middle brackets within ±15% of SOI
- [ ] High brackets within ±20% of SOI

### After Phase 3 (Integration)
- [ ] All tests pass
- [ ] Performance acceptable
- [ ] Documentation updated

---

## Expected Outcomes

### Before Fixes
| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Low-income coverage | 3% | 100% | -97% |
| Total coverage | 97.3% | 100% | -2.7% |
| Brackets within ±5% | 3/36 (8%) | 28/36 (78%) | -70% |
| Avg % difference | 21.6% | <5% | +16.6% |

### After Phase 1 (Filing Threshold Fix)
| Metric | Expected | Improvement |
|--------|----------|-------------|
| Low-income coverage | ~95% | +92% |
| Total coverage | ~99% | +1.7% |
| Brackets within ±5% | 6/36 (17%) | +9% |

### After Phase 2 (Taxable Income)
| Metric | Expected | Improvement |
|--------|----------|-------------|
| Low-income coverage | ~98% | +95% |
| Total coverage | ~99% | +1.7% |
| Brackets within ±5% | 20/36 (56%) | +48% |
| Avg % difference | <8% | +14% |

### After Phase 3 (Full Integration)
| Metric | Expected | Improvement |
|--------|----------|-------------|
| Low-income coverage | 100% | +97% |
| Total coverage | 100% | +2.7% |
| Brackets within ±5% | 28/36 (78%) | +70% |
| Avg % difference | <5% | +17% |

---

## Next Steps

1. **Decision Point**: Choose implementation path
   - **Recommended**: Option A (Two-Step Fix)
   - **Alternative**: Option C (Full Integration)

2. **Execute Phase 1**: Fix filing threshold (30 min)

3. **Validate**: Run comparison script

4. **Execute Phase 2**: Apply taxable income (1 hour)

5. **Validate**: Check all brackets

6. **Decide**: Proceed with Phase 3 integration or stop

---

*Generated: 2025-10-15*
*Analysis: Income Bracket Alignment Implementation Plan*
