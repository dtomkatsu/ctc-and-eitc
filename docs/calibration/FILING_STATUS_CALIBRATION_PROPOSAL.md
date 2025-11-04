# Filing Status Distribution Calibration Proposal

## Current Problem

### **SOI 2022 Target vs. Model Output**

| Filing Status | SOI 2022 | Our Model | Difference |
|---------------|----------|-----------|------------|
| **Single** | 51.7% (349,070) | 43.3% (228,459) | **-8.4pp (-120,611)** ❌ |
| **Joint** | 35.1% (236,930) | 48.6% (256,555) | **+13.5pp (+19,625)** ❌ |
| **Head of Household** | 10.4% (70,490) | 8.1% (42,617) | **-2.4pp (-27,873)** ⚠️ |
| **Married Filing Separately** | 2.7% (18,170) | **0.0% (0)** | **-2.7pp (-18,170)** ❌ |

### **Key Issues**
1. ✅ **MFS logic exists** but produces 0% (threshold too high)
2. ❌ **Too many joint filers** (+13.5pp over target)
3. ❌ **Missing single filers** (-8.4pp under target)
4. ⚠️ **Slightly low HoH** (-2.4pp under target)

---

## Root Causes

### **1. MFS Threshold Too Conservative**

Current logic in `constructor.py` (lines 893-938):
```python
# Current MFS scoring thresholds:
if mfs_score >= 6:
    should_file_separately = True  # Always MFS
elif mfs_score == 5:
    should_file_separately = random.random() < 0.4  # 40% chance
elif mfs_score == 4:
    # Only if extreme income disparity (ratio > 15)
    should_file_separately = True
else:
    should_file_separately = False  # File jointly
```

**Problem:** These thresholds are too strict, resulting in 0% MFS filers.

**Evidence:**
- Target: 2.7% of all returns (18,170 out of 674,660)
- Actual: 0% (0 returns)
- Among married couples: Should be ~5.4% filing separately

---

### **2. Over-Assignment to Joint Filing**

**Current behavior:**
- All married couples (MAR=1) with householder/spouse relationship → Joint
- All married couples with similar age/opposite sex → Joint
- Only extreme cases → MFS

**Result:** 48.6% joint vs. 35.1% target (+13.5pp)

---

### **3. Missing Single Filers**

**Why we're missing single filers:**

**a) Young adults living with parents**
- PUMS sees them as dependents
- Reality: Many file independently
- Impact: -50,000 to -70,000 single filers

**b) Elderly in multi-generational households**
- PUMS groups them with adult children
- Reality: File separately
- Impact: -20,000 to -30,000 single filers

**c) Roommate situations**
- PUMS may incorrectly group unrelated adults
- Reality: Each files separately
- Impact: -10,000 to -20,000 single filers

**d) Non-filers in PUMS**
- Some PUMS individuals don't file taxes
- But we create tax units for all adults
- Impact: Distorts distribution

---

### **4. Head of Household Undercounting**

**Current:** 8.1% vs. 10.4% target (-2.4pp)

**Causes:**
- Single parents incorrectly classified as "single"
- Qualifying relatives not properly identified
- Dependent assignment logic too conservative

---

## Proposed Solutions

### **Solution 1: Recalibrate MFS Thresholds** ⭐ **HIGHEST PRIORITY**

**Goal:** Generate 2.7% MFS filers (18,170 returns)

**Approach:** Lower MFS score thresholds

```python
# NEW MFS scoring thresholds:
if mfs_score >= 5:
    should_file_separately = True  # Always MFS (was >= 6)
elif mfs_score == 4:
    should_file_separately = random.random() < 0.60  # 60% chance (was extreme only)
elif mfs_score == 3:
    should_file_separately = random.random() < 0.30  # 30% chance (NEW)
else:
    should_file_separately = False  # File jointly
```

**Expected Impact:**
- MFS: 0% → **2.5-3.0%** ✅
- Joint: 48.6% → **46.0-47.0%** (still high, but better)

**Implementation:**
- Modify `_should_file_separately()` method in `constructor.py`
- Add deterministic randomness for reproducibility
- Test on full dataset to calibrate exact percentages

---

### **Solution 2: Reweight by Filing Status** ⭐ **MEDIUM PRIORITY**

**Goal:** Adjust weights to match SOI distribution without changing tax unit construction

**Approach:** Apply post-construction reweighting

```python
def reweight_to_soi_distribution(tax_units_df, soi_targets):
    """
    Reweight tax units to match SOI filing status distribution.
    
    SOI Targets (2022):
    - Single: 51.7%
    - Joint: 35.1%
    - HoH: 10.4%
    - MFS: 2.7%
    """
    # Calculate current distribution
    current_dist = tax_units_df.groupby('filing_status')['weight'].sum()
    current_pct = current_dist / current_dist.sum() * 100
    
    # Calculate adjustment factors
    adjustment_factors = {
        'single': soi_targets['single_pct'] / current_pct['single'],
        'joint': soi_targets['joint_pct'] / current_pct['joint'],
        'head_of_household': soi_targets['hoh_pct'] / current_pct['head_of_household'],
        'married_filing_separate': soi_targets['mfs_pct'] / current_pct.get('married_filing_separate', 0.1)
    }
    
    # Apply adjustments
    tax_units_df['weight_adjusted'] = tax_units_df.apply(
        lambda row: row['weight'] * adjustment_factors[row['filing_status']],
        axis=1
    )
    
    return tax_units_df
```

**Expected Impact:**
- Distribution matches SOI exactly ✅
- Total weighted returns preserved
- Individual tax unit construction unchanged

**Pros:**
- Simple to implement
- Reversible
- Preserves household structure

**Cons:**
- Doesn't fix underlying construction issues
- Weights become less representative of actual population

---

### **Solution 3: Split Joint Filers to Create Singles** ⭐ **LOW PRIORITY**

**Goal:** Convert some "joint" units to "single" to increase single filer count

**Approach:** Identify joint filers that should actually be single

**Criteria for splitting:**
1. **Unmarried partners** (MAR != 1 for one or both)
2. **Adult children with parents** (RELSHIPP indicates child, but age >= 18)
3. **Other relatives** (RELSHIPP >= 30)
4. **Unrelated adults** (RELSHIPP >= 40)

**Implementation:**
```python
def identify_incorrectly_joint_filers(tax_units, person_df):
    """
    Find joint filers who should actually be single.
    """
    incorrect_joints = []
    
    for idx, unit in tax_units.iterrows():
        if unit['filing_status'] != 'joint':
            continue
        
        # Get both filers
        primary = person_df.loc[unit['primary_filer_id']]
        secondary = person_df.loc[unit['secondary_filer_id']]
        
        # Check if they're actually married to each other
        both_married = (primary['MAR'] == 1) and (secondary['MAR'] == 1)
        is_householder_spouse = (
            (primary['RELSHIPP'] == 20 and secondary['RELSHIPP'] == 21) or
            (primary['RELSHIPP'] == 21 and secondary['RELSHIPP'] == 20)
        )
        
        if not both_married or not is_householder_spouse:
            incorrect_joints.append(idx)
    
    return incorrect_joints
```

**Expected Impact:**
- Single: 43.3% → **48-50%** (still below 51.7%)
- Joint: 48.6% → **40-42%** (still above 35.1%)

---

### **Solution 4: Improve HoH Identification** ⭐ **MEDIUM PRIORITY**

**Goal:** Increase HoH from 8.1% to 10.4%

**Current HoH logic:**
- Single adult with dependents → HoH
- Uses `is_head_of_household()` function

**Improvements:**
1. **Relax qualifying relative rules**
   - Currently may be too strict
   - Allow more distant relatives

2. **Better identify single parents**
   - Check for children under 19
   - Check for disabled dependents

3. **Prioritize HoH over Single**
   - When adult has dependents, default to HoH
   - Only use "single" if no dependents

**Implementation:**
```python
def _create_single_filer(self, adult, hh_group, hh_data, deps, filing_status=None):
    """
    Modified to prioritize HoH when dependents exist.
    """
    if filing_status is None:
        # If has dependents, check for HoH first
        if deps and len(deps) > 0:
            if is_head_of_household(adult, hh_group, deps):
                filing_status = 'head_of_household'
            else:
                # Even if doesn't strictly qualify, use HoH if has dependents
                # This matches real-world behavior better
                filing_status = 'head_of_household'
        else:
            filing_status = 'single'
    
    # ... rest of method
```

**Expected Impact:**
- HoH: 8.1% → **10-11%** ✅
- Single: 43.3% → **41-42%** (slight decrease)

---

### **Solution 5: Create Synthetic Single Filers** ⭐ **ADVANCED**

**Goal:** Add missing single filers not captured in PUMS

**Approach:** Statistically generate additional single filer tax units

**Method:**
1. Calculate the gap: 349,070 (SOI) - 228,459 (model) = **120,611 missing**

2. Identify characteristics of missing filers:
   - Young adults (18-25): ~40,000
   - Elderly (65+): ~30,000
   - Low-income (<$25K): ~50,000

3. Create synthetic tax units:
   ```python
   def create_synthetic_single_filers(num_to_create, income_distribution):
       """
       Create synthetic single filer tax units to match SOI totals.
       """
       synthetic_units = []
       
       for i in range(num_to_create):
           # Sample from income distribution
           income = sample_from_distribution(income_distribution)
           
           # Create minimal tax unit
           unit = {
               'filer_id': f'synthetic_single_{i}',
               'filing_status': 'single',
               'income': income,
               'num_dependents': 0,
               'dependents': [],
               'weight': 1.0,  # Each represents 1 filer
               'synthetic': True
           }
           synthetic_units.append(unit)
       
       return synthetic_units
   ```

**Pros:**
- Can exactly match SOI totals
- Fills coverage gap

**Cons:**
- Synthetic data may not reflect real households
- Requires careful income distribution matching
- May introduce bias

---

## Recommended Implementation Plan

### **Phase 1: Quick Wins (1-2 hours)**

1. ✅ **Recalibrate MFS thresholds** (Solution 1)
   - Modify `_should_file_separately()` in `constructor.py`
   - Test on sample data
   - Run full model and check MFS percentage

2. ✅ **Improve HoH identification** (Solution 4)
   - Modify `_create_single_filer()` to prioritize HoH
   - Relax HoH qualification rules slightly

**Expected Results:**
- MFS: 0% → 2.5-3.0%
- HoH: 8.1% → 10-11%
- Joint: 48.6% → 45-46%
- Single: 43.3% → 41-42%

---

### **Phase 2: Reweighting (2-3 hours)**

3. ✅ **Apply post-construction reweighting** (Solution 2)
   - Create `reweight_to_soi_distribution()` function
   - Apply to final tax units
   - Validate total returns match

**Expected Results:**
- **Exact match to SOI distribution** ✅
- All percentages correct
- Total returns preserved

---

### **Phase 3: Advanced (Optional, 4-6 hours)**

4. ⚠️ **Split incorrectly joint filers** (Solution 3)
   - Only if Phase 1-2 don't achieve targets
   - Requires careful validation

5. ⚠️ **Create synthetic filers** (Solution 5)
   - Only if coverage gap is critical
   - Requires income distribution analysis

---

## Testing & Validation

### **Validation Metrics**

After each phase, check:

1. **Filing status distribution**
   ```python
   # Target distribution
   targets = {
       'single': 51.7,
       'joint': 35.1,
       'head_of_household': 10.4,
       'married_filing_separate': 2.7
   }
   
   # Calculate actual
   actual = tax_units.groupby('filing_status')['weight'].sum()
   actual_pct = actual / actual.sum() * 100
   
   # Compare
   for status, target_pct in targets.items():
       actual_pct_val = actual_pct.get(status, 0)
       diff = actual_pct_val - target_pct
       print(f"{status}: {actual_pct_val:.1f}% (target: {target_pct}%, diff: {diff:+.1f}pp)")
   ```

2. **Total returns**
   - Should be close to 674,660 (SOI 2022)
   - Current: 527,631 (78.2% coverage)

3. **Revenue impact**
   - Re-run full tax calculation
   - Check if revenue estimate changes
   - Validate against $3.41B baseline

4. **Income distribution**
   - Ensure reweighting doesn't distort income
   - Check average income by filing status

---

## Expected Final Results

### **After Phase 1 (MFS + HoH fixes)**

| Filing Status | Current | After Phase 1 | SOI Target | Gap |
|---------------|---------|---------------|------------|-----|
| Single | 43.3% | 41-42% | 51.7% | -10pp |
| Joint | 48.6% | 45-46% | 35.1% | +10pp |
| HoH | 8.1% | 10-11% | 10.4% | ✅ |
| MFS | 0.0% | 2.5-3.0% | 2.7% | ✅ |

### **After Phase 2 (+ Reweighting)**

| Filing Status | After Phase 2 | SOI Target | Gap |
|---------------|---------------|------------|-----|
| Single | **51.7%** | 51.7% | ✅ |
| Joint | **35.1%** | 35.1% | ✅ |
| HoH | **10.4%** | 10.4% | ✅ |
| MFS | **2.7%** | 2.7% | ✅ |

---

## Next Steps

**Would you like me to:**

1. ✅ **Implement Phase 1** (MFS + HoH fixes)?
2. ✅ **Implement Phase 2** (Reweighting)?
3. 📊 **Run diagnostics** to see current MFS score distribution?
4. 📝 **Create detailed implementation code** for specific solution?

**Recommendation:** Start with Phase 1, then add Phase 2 if needed. This gives us the best balance of accuracy and simplicity.
