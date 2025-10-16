# AGI Quick Reference Card

## ✅ What's Done

1. **Filing threshold removed** - Will capture ~155K more low-income filers
2. **AGI adjustments module created** - Post-processing approach (Option 3)
3. **Three income versions prepared** - Original, AGI, Taxable
4. **All code tested and working** ✅

---

## 📊 Three Income Versions

| Version | File | Income Column | Use For |
|---------|------|---------------|---------|
| **Original** | `tax_units_original.parquet` | `income` | General analysis |
| **AGI** ⭐ | `tax_units_agi.parquet` | `agi` | **Tax revenue estimates** |
| **Taxable** | `tax_units_taxable.parquet` | `taxable_income` | SOI comparison |

---

## 🚀 For Tax Revenue Estimates (Use AGI)

```python
import pandas as pd

# Load tax units with AGI
tax_units = pd.read_parquet('data/processed/tax_units_agi.parquet')

# Use 'agi' column for tax calculations
tax_units['tax_liability'] = calculate_tax(
    tax_units['agi'],  # ← Use AGI, not 'income'
    tax_units['filing_status']
)

# Calculate total revenue
total_revenue = (tax_units['tax_liability'] * tax_units['weight']).sum()
```

**Why AGI?**
- ✅ Standard base for tax calculations
- ✅ Tax brackets based on AGI
- ✅ Credits/deductions phase out based on AGI
- ✅ Matches IRS methodology

---

## 📈 Income Reductions

```
Total Income:     $89,141  (100.0%)
    ↓ AGI adjustments (-1.1%)
AGI:              $88,173  (98.9%)
    ↓ Standard deduction (-19.5%)
Taxable Income:   $71,008  (79.7%)
```

**Total reduction: 20.3%** ($18,133 per tax unit)

---

## 🔄 Next Steps

### 1. Regenerate Tax Units (IMPORTANT!)
```bash
python scripts/generate_tax_units.py
```
**Why**: Filing threshold removed, will create ~155K more tax units

### 2. Prepare Income Versions (Already Done!)
```bash
python scripts/prepare_tax_units_with_agi.py
```
**Output**: Three parquet files in `data/processed/`

### 3. Update Your Tax Revenue Scripts
```python
# OLD:
tax_units = pd.read_parquet('data/processed/tax_units_regenerated_*.parquet')
tax = calculate_tax(tax_units['income'], ...)

# NEW:
tax_units = pd.read_parquet('data/processed/tax_units_agi.parquet')
tax = calculate_tax(tax_units['agi'], ...)  # ← Use AGI
```

---

## 📁 Key Files

### Code
- `src/tax/units/constructor.py` - Filing threshold removed (line 1375)
- `src/tax/units/income_adjustments.py` - AGI post-processing module
- `src/tax/adjustments/agi_adjustments.py` - AGI calculation logic

### Scripts
- `scripts/prepare_tax_units_with_agi.py` - Create income versions
- `scripts/diagnose_low_income_gap.py` - Diagnostic tool

### Documentation
- `AGI_IMPLEMENTATION_SUMMARY.md` - Full details
- `QUICK_FIX_GUIDE.md` - Step-by-step instructions
- `analysis_results/income_distribution/IMPLEMENTATION_PLAN.md` - Technical plan

---

## 💡 Quick Tips

### Load the Right Version
```python
# For tax revenue estimates
from src.tax.units.income_adjustments import load_and_prepare_tax_units
tax_units = load_and_prepare_tax_units(use_version='agi')
```

### Check Available Columns
```python
# AGI version has:
# - income (total income)
# - agi (adjusted gross income) ← Use this for tax calculations
# - agi_adjustments (amount of adjustments)
# - agi_adjustment_details (breakdown)
```

### Verify Income Reduction
```python
# Should show ~1.1% reduction
print(f"AGI is {(tax_units['agi'] / tax_units['income']).mean() * 100:.1f}% of total income")
# Expected: 98.9%
```

---

## ⚠️ Important Notes

1. **Always use AGI for tax calculations** - Not total income
2. **Regenerate tax units** - Filing threshold removed, need fresh data
3. **Three versions available** - Use the right one for your analysis
4. **AGI ≠ Taxable Income** - AGI is before standard deduction

---

## 🎯 Expected Results

### After Regeneration
- **Total tax units**: ~50,000 (up from 34,887)
- **Low-income coverage**: ~98% (up from 3%)
- **Total coverage**: ~99% (up from 97.3%)

### Tax Revenue Estimates
- **More accurate** - Using AGI instead of total income
- **Better aligned** - Matches IRS methodology
- **Properly adjusted** - Accounts for IRA, student loans, etc.

---

*Quick Reference - Last Updated: 2025-10-15*
