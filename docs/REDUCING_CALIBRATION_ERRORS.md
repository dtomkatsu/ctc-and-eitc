# Reducing Calibration Errors in High-Income Brackets

## Problem Statement

High-income brackets have **high calibration errors** (37-125%) due to:
- **Small PUMS sample sizes** (e.g., 52 HoH records for $150k-$200k bracket)
- **Limited degrees of freedom** for 2D IPF (matching both returns and AGI)
- **Sample representativeness issues** in rare populations

## Solutions Ranked by Effectiveness

### 1. ⭐ Synthetic Household Generation (RECOMMENDED)

**What it does**: Creates additional synthetic tax units in under-sampled brackets by resampling and adding controlled noise.

**Pros**:
- ✅ Significantly improves calibration accuracy (expected 30-60% error reduction)
- ✅ Maintains realistic distributions
- ✅ Flexible and controllable
- ✅ Works for any bracket size

**Cons**:
- ⚠️ Adds complexity
- ⚠️ Requires validation to ensure synthetic records are realistic

**Implementation**:
```python
from src.tax.validation.synthetic_augmentation import augment_undersampled_brackets

# Augment under-sampled brackets
augmented = augment_undersampled_brackets(
    tax_units,
    min_sample_size=100,      # Augment if < 100 samples
    target_sample_size=300,    # Target 300 samples
    noise_factor=0.03,         # ±3% noise on AGI
    synthetic_weight_factor=0.1  # 10% weight for synthetic records
)

# Then apply calibration
calibrated = apply_hybrid_tax_calibration(augmented)
```

**Expected Improvement**:
- HoH $150k-$200k: 125% → **40-60%** error
- Single $150k-$200k: 57% → **20-30%** error
- Joint $150k-$200k: 37% → **10-20%** error

---

### 2. Bracket Consolidation

**What it does**: Combine adjacent small brackets into larger brackets to increase sample size.

**Pros**:
- ✅ Simple to implement
- ✅ Guaranteed to reduce errors
- ✅ No synthetic data needed

**Cons**:
- ⚠️ Loses granularity
- ⚠️ May not match policy analysis needs

**Implementation**:
```python
# Instead of: $150k-$200k, $200k-$300k, $300k-$400k, $400k+
# Use: $150k-$300k, $300k+

consolidated_brackets = [
    (150000, 300000),  # Combines 2 brackets
    (300000, 999999999)  # Combines 2 brackets
]
```

**Expected Improvement**:
- Combining $150k-$200k + $200k-$300k: ~50% error reduction
- Trade-off: Lose ability to analyze $150k-$200k separately

---

### 3. Hierarchical Calibration

**What it does**: Calibrate at aggregate level first, then disaggregate proportionally.

**Pros**:
- ✅ Ensures total accuracy
- ✅ Reduces oscillation in IPF
- ✅ Works well for nested structures

**Cons**:
- ⚠️ May not match individual bracket targets exactly
- ⚠️ Complex implementation

**Implementation**:
```python
# Step 1: Calibrate to filing status totals
calibrate_to_filing_status_totals(tax_units)

# Step 2: Within each filing status, calibrate to AGI brackets
for status in ['single', 'joint', 'hoh']:
    calibrate_within_status(tax_units, status)
```

**Expected Improvement**: 20-40% error reduction

---

### 4. External Data Augmentation

**What it does**: Use additional data sources (IRS Public Use File, Tax Policy Center microsimulation, etc.) to supplement PUMS.

**Pros**:
- ✅ Real data (not synthetic)
- ✅ May have better high-income coverage
- ✅ Can validate PUMS patterns

**Cons**:
- ⚠️ Requires data acquisition and harmonization
- ⚠️ May have different definitions/years
- ⚠️ Complex integration

**Sources**:
- IRS SOI Public Use File (PUF)
- Tax Policy Center Microsimulation Model
- CBO Tax Model data

---

### 5. Relaxed Tolerance for Small Brackets

**What it does**: Accept higher errors for brackets with < 100 samples.

**Pros**:
- ✅ Simple
- ✅ Acknowledges data limitations
- ✅ No code changes needed

**Cons**:
- ⚠️ Doesn't actually improve accuracy
- ⚠️ Still have high errors for policy analysis

**Implementation**:
```python
# Set different tolerances by bracket size
if sample_size < 50:
    acceptable_error = 100%  # Very small samples
elif sample_size < 100:
    acceptable_error = 50%   # Small samples
elif sample_size < 500:
    acceptable_error = 20%   # Medium samples
else:
    acceptable_error = 10%   # Large samples
```

---

## Recommended Approach

**For maximum accuracy with manageable complexity**:

1. **Use Synthetic Augmentation** (Solution #1)
   - Augment brackets with < 100 samples to target 200-300 samples
   - Use conservative noise factor (3-5%)
   - Weight synthetic records at 10% of original

2. **Validate Synthetic Records**
   - Check that median/mean AGI matches original
   - Ensure synthetic records span full bracket range
   - Verify total weight doesn't dominate

3. **Monitor Convergence**
   - Use increased iterations (500+)
   - Tighten tolerance (0.0005)
   - Log problematic brackets

4. **For Extremely Small Brackets** (< 20 samples):
   - Consider bracket consolidation
   - Or flag as "low confidence" in outputs

## Testing Synthetic Augmentation

Run the test script to compare baseline vs synthetic:
```bash
python scripts/test_synthetic_augmentation.py
```

This will show:
- Sample size increases
- Error rate improvements
- Overall accuracy gains

## Expected Final Accuracy

With synthetic augmentation:
- **Large brackets** (> 1000 samples): 1-5% error
- **Medium brackets** (100-1000 samples): 5-15% error
- **Small brackets** (20-100 samples): 15-40% error
- **Very small brackets** (< 20 samples): 40-100% error (consider consolidating)

## References

- Deville, J.C., & Särndal, C.E. (1992). "Calibration Estimators in Survey Sampling"
- Little, R.J.A. & Rubin, D.B. (2019). "Statistical Analysis with Missing Data"
- Synthetic data generation: CART, hot-deck imputation, parametric bootstrapping
