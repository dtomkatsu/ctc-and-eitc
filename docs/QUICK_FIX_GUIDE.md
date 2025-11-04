# Quick Fix Guide: Income Bracket Alignment

## Problem Summary
- **Missing 146K low-income filers** (97% undercounted)
- **Income brackets misaligned** by 20-140% due to using total income vs taxable income

## ✅ What's Ready

### 1. Diagnostic Tools
- ✅ `scripts/diagnose_low_income_gap.py` - Identifies root causes
- ✅ `scripts/compare_soi_income_brackets.py` - Detailed bracket comparison

### 2. Solution Modules  
- ✅ `src/tax/units/taxable_income.py` - Calculates taxable income from total income
- ✅ `src/tax/adjustments/agi_adjustments.py` - Already exists! AGI adjustments

### 3. Analysis Reports
- ✅ `analysis_results/income_distribution/INCOME_BRACKET_ANALYSIS.md`
- ✅ `analysis_results/income_distribution/IMPLEMENTATION_PLAN.md`

---

## 🚀 Quick Fix (30 Minutes)

### Step 1: Fix Filing Threshold

**File**: `src/tax/units/constructor.py`  
**Line**: 1387

**Change this:**
```python
if income < FILING_THRESHOLD and not has_self_employment:
    logger.debug(f"Adult {adult.name} has income ${income:.0f} below filing threshold, not creating tax unit")
    return None
```

**To this (Option 1 - Remove completely):**
```python
# Removed filing threshold check - SOI includes all filers
# Many low-income individuals file for refundable credits, refunds, or other purposes
```

**Or this (Option 2 - Lower to $1):**
```python
if income < 1 and not has_self_employment:  # Only exclude $0 income
    logger.debug(f"Adult {adult.name} has zero income, not creating tax unit")
    return None
```

### Step 2: Regenerate Tax Units
```bash
python scripts/generate_tax_units.py
```

### Step 3: Test Results
```bash
python scripts/compare_soi_income_brackets.py
```

**Expected**: Low-income brackets improve from -98% to within ±10%

---

## 🎯 Complete Fix (90 Minutes)

### After Step 1-3 above, add taxable income calculation:

**File**: `scripts/compare_soi_income_brackets.py`

**Add after loading tax units:**
```python
# Add at top of file:
from src.tax.units.taxable_income import TaxableIncomeCalculator

# In main() function, after loading tax_units:
def main():
    # ... existing code to load tax_units ...
    
    # Convert to taxable income for SOI comparison
    print("Calculating taxable income for SOI comparison...")
    calculator = TaxableIncomeCalculator(tax_year=2022)
    tax_units = calculator.apply_to_tax_units(
        tax_units,
        income_col='income',
        filing_status_col='filing_status',
        apply_agi_adjustments=True  # Enable AGI adjustments
    )
    
    # Use 'taxable_income' column instead of 'income' for brackets
    # Change this line in map_to_soi_brackets():
    # OLD: mask = (status_units['income'] > min_income) & (status_units['income'] <= max_income)
    # NEW: mask = (status_units['taxable_income'] > min_income) & (status_units['taxable_income'] <= max_income)
```

### Test Complete Fix
```bash
python scripts/compare_soi_income_brackets.py
```

**Expected**: All brackets within ±15% of SOI targets

---

## 📊 AGI Adjustment Options

### Option 1: Enable in Comparison Only (RECOMMENDED) ⭐
Already done in the code above! Set `apply_agi_adjustments=True`

**Pros**: Simple, reversible, only affects SOI comparison  
**Impact**: ~1-2% income reduction

### Option 2: Integrate in Constructor
**File**: `src/tax/units/constructor.py`

Add to income calculation:
```python
# In _create_single_filer and similar methods:
from tax.units.taxable_income import TaxableIncomeCalculator

# After calculating income:
calculator = TaxableIncomeCalculator(tax_year=2022)
tax_result = calculator.calculate_taxable_income(
    income, filing_status, age, has_self_employment,
    apply_agi_adjustments=True
)

# Store both:
tax_unit['total_income'] = income
tax_unit['agi'] = tax_result['agi']
tax_unit['taxable_income'] = tax_result['taxable_income']
tax_unit['income'] = tax_result['taxable_income']  # Use for comparisons
```

**Pros**: Consistent everywhere, more accurate  
**Cons**: Larger code change, affects all analyses

### Option 3: Post-Processing
Create adjusted copy only when needed for SOI comparison

**Pros**: Flexibility, keeps original data  
**Cons**: Duplicate data, potential confusion

---

## 🔍 Validation

### After Quick Fix:
```bash
# Should show:
# - Low-income brackets: -98% → within ±10%
# - Total coverage: 97.3% → ~99%
```

### After Complete Fix:
```bash
# Should show:
# - All brackets: within ±15% of SOI
# - Low-income: within ±5%
# - Middle-income: within ±10%
# - High-income: within ±20%
```

---

## 📝 Summary

| Issue | Fix | Time | Impact |
|-------|-----|------|--------|
| Missing 146K low-income filers | Remove filing threshold | 5 min | +97% low-income coverage |
| Income bracket misalignment | Use taxable income | 30 min | Brackets align within ±15% |
| AGI adjustments | Enable in comparison | 5 min | +1-2% accuracy |
| **TOTAL** | | **40 min** | **Near-perfect SOI alignment** |

---

## 🎯 Recommended Approach

1. ✅ **Do Quick Fix first** (30 min)
2. ✅ **Validate results**
3. ✅ **If good, proceed with Complete Fix** (1 hour)
4. ✅ **Choose AGI Option 1** (enable in comparison only)

---

## 📞 Need Help?

- **Full details**: `analysis_results/income_distribution/IMPLEMENTATION_PLAN.md`
- **Diagnostic script**: `python scripts/diagnose_low_income_gap.py`
- **Comparison script**: `python scripts/compare_soi_income_brackets.py`

---

*Last updated: 2025-10-15*
