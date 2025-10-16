# Coverage Gap Analysis - Critical Findings

**Date**: 2025-10-15  
**Gap**: 168,762 missing filers (26.6% under SOI target)

---

## 🚨 CRITICAL FINDINGS

### 1. **Household Cap is TOO RESTRICTIVE** ⚠️⚠️⚠️

**Impact**: **228,078 tax units lost** (135% of the gap!)

The current 2-tax-unit-per-household cap is **artificially limiting** tax unit creation:

| Adults per HH | Weighted HH | Potential Units | Capped Units | **Lost Units** |
|---------------|-------------|-----------------|--------------|----------------|
| 1 | 171,828 | 171,828 | 171,828 | 0 |
| 2 | 244,473 | 488,946 | 488,946 | 0 |
| **3** | 37,877 | 113,632 | 75,755 | **37,877** |
| **4** | 13,969 | 55,877 | 27,939 | **27,939** |
| **5** | 5,008 | 25,040 | 10,016 | **15,024** |
| **6+** | 7,424 | 59,392 | 14,848 | **44,544** |
| **TOTAL** | 480,579 | **914,715** | **686,637** | **228,078** |

**Key Insight**: We're losing 228,078 potential tax units due to the household cap, which is **more than the entire coverage gap**!

### 2. **Massive Adult Assignment Problem** ⚠️⚠️⚠️

**Impact**: **508,738 unassigned adults** (301% of the gap!)

Only **55.6% of adults** are assigned to tax units:

- **Total PUMS adults**: 1,145,448
- **Adults in tax units**: 636,710
- **Unassigned adults**: 508,738 (44.4%!)

**This is the smoking gun** - nearly half of all adults are not being assigned to any tax unit!

### 3. **Dependent Classification Issues** ⚠️

**Impact**: 64,068 missing dependents (21% under PUMS children)

- **Children in PUMS**: 305,141
- **Dependents in tax units**: 241,073
- **Missing**: 64,068 (21%)

Only 79% of children are being assigned as dependents.

### 4. **PUMS Has Enough Adults** ✅

- **PUMS adults**: 1,145,448
- **SOI filers**: 635,117
- **Surplus**: 510,331 adults

PUMS has **1.8x more adults** than SOI filers, which is correct (not everyone files taxes). The problem is in our tax unit construction logic.

---

## Root Cause Analysis

### Why is the Household Cap So Restrictive?

The 2-tax-unit cap was likely implemented to prevent over-counting, but it's **too aggressive**:

1. **Multi-generational households**: Grandparents, adult children, etc. should file separately
2. **Roommates**: Unrelated adults in same household should file separately
3. **Extended family**: Siblings, cousins, etc. living together

**Example**: A household with 2 parents, 2 adult children, and 1 grandparent (5 adults) could have:
- Parents filing jointly (1 tax unit)
- Adult child 1 filing single (1 tax unit)
- Adult child 2 filing single (1 tax unit)
- Grandparent filing single (1 tax unit)
- **Total**: 4 tax units

But the cap limits this to **2 tax units**, losing 2 potential filers.

### Why Are So Many Adults Unassigned?

Possible reasons:

1. **Household cap prevents creation**: Adults in large households can't create tax units
2. **Logic errors**: Tax unit construction logic may be skipping adults
3. **Dependent misclassification**: Some adults incorrectly classified as dependents
4. **Relationship code issues**: Some relationship codes may not be handled properly

---

## Recommended Solutions

### Priority 1: Remove Household Cap ⭐⭐⭐ CRITICAL

**Action**: Remove or significantly increase the 2-tax-unit-per-household cap

**Implementation**:
```python
# In constructor.py, find the household cap logic and either:
# Option A: Remove it entirely
# Option B: Increase to 5-6 tax units per household
# Option C: Make it proportional to number of adults (e.g., max = num_adults / 1.5)
```

**Expected Impact**:
- Add 228,078 tax units
- Close 135% of the gap
- **This alone would EXCEED the SOI target!**

**Risk**: May over-count if not careful. Need to ensure:
- Each adult only appears in one tax unit
- Dependents properly assigned
- No duplicate tax units

### Priority 2: Fix Adult Assignment Logic ⭐⭐⭐ CRITICAL

**Action**: Debug why 508,738 adults (44%) are unassigned

**Investigation Steps**:
1. Check if unassigned adults are in large households (>2 adults)
2. Review relationship codes of unassigned adults
3. Check if they're being incorrectly classified as dependents
4. Verify the `_process_household` logic handles all adults

**Expected Impact**:
- Assign remaining 508,738 adults to tax units
- Could create 300,000+ additional tax units
- **This would MASSIVELY exceed SOI target**

**Key Insight**: The fact that we have 508K unassigned adults AND 228K lost to cap suggests:
- **Most unassigned adults are in large households**
- **Removing the cap will automatically assign most of them**

### Priority 3: Improve Dependent Assignment ⭐⭐

**Action**: Ensure all children are properly assigned as dependents

**Expected Impact**:
- Assign 64,068 missing children
- Improve HoH and dependent counts
- Better align with SOI dependent statistics

---

## Testing Strategy

### Step 1: Remove Household Cap (Test)

1. **Backup current code**
2. **Remove or increase cap** to 10 tax units per household
3. **Regenerate tax units**
4. **Check results**:
   - Total tax units (expect ~650K-700K)
   - Filing status distribution (should still be calibrated)
   - Adult assignment rate (expect >90%)

### Step 2: Validate Results

Compare to SOI benchmarks:
- Total filers: Should be close to 635,117
- Filing status distribution: Should remain perfect (calibration handles this)
- Income distribution: Should improve
- Dependent counts: Should improve

### Step 3: Fine-Tune if Needed

If we over-count:
- Implement smarter cap (e.g., based on household composition)
- Add validation to prevent duplicate tax units
- Review relationship codes for edge cases

---

## Expected Outcomes

### Scenario A: Remove Cap Entirely

**Optimistic**:
- Add 228,078 tax units → **694,433 total** (109% of SOI)
- Need to reduce by ~60K units (apply stricter logic or calibration)

**Realistic**:
- Add 150,000 tax units → **616,355 total** (97% of SOI)
- Close enough to SOI target

**Pessimistic**:
- Add 100,000 tax units → **566,355 total** (89% of SOI)
- Still need more work

### Scenario B: Increase Cap to 5

**Expected**:
- Add 100,000-150,000 tax units
- Get to 90-95% of SOI target
- More conservative, less risk of over-counting

### Scenario C: Proportional Cap (num_adults / 1.5)

**Expected**:
- Add 120,000-180,000 tax units
- Get to 92-97% of SOI target
- Balanced approach

---

## Code Changes Required

### File: `src/tax/units/constructor.py`

**Find the household cap logic** (likely in `_process_household` or `create_rule_based_units`):

```python
# Current (restrictive):
MAX_TAX_UNITS_PER_HH = 2

# Option A: Remove cap
# MAX_TAX_UNITS_PER_HH = None  # No cap

# Option B: Increase cap
MAX_TAX_UNITS_PER_HH = 5

# Option C: Proportional cap
def get_max_tax_units(num_adults):
    return max(2, int(num_adults / 1.5))
```

**Search for cap enforcement**:
```python
# Find code like:
if len(tax_units) > MAX_TAX_UNITS_PER_HH:
    # Remove excess units
    
# And either remove it or adjust the logic
```

---

## Validation Checklist

After removing/adjusting the cap:

- [ ] Total tax units within ±5% of SOI (603K-667K)
- [ ] Filing status distribution still calibrated (within 0.1%)
- [ ] Adult assignment rate >90%
- [ ] No duplicate tax units (check filer_id uniqueness)
- [ ] Dependent assignment rate >90%
- [ ] Income distribution aligns with SOI brackets
- [ ] Each adult appears in exactly one tax unit

---

## Timeline

1. **Remove/adjust household cap**: 30 minutes
2. **Regenerate tax units**: 5 minutes
3. **Validate results**: 15 minutes
4. **Fine-tune if needed**: 1-2 hours
5. **Final validation**: 30 minutes

**Total**: 2-3 hours to fully resolve the coverage gap

---

## Conclusion

### The Good News ✅

1. **PUMS has enough adults** (1.1M vs 635K needed)
2. **Filing status distribution is perfect** (calibration works!)
3. **Root cause identified**: Household cap and adult assignment

### The Bad News ⚠️

1. **Household cap is way too restrictive** (losing 228K units)
2. **44% of adults unassigned** (508K adults)
3. **21% of children not assigned as dependents** (64K children)

### The Solution 🎯

**Remove or significantly increase the household cap**. This single change will:
- Add 150,000-228,000 tax units
- Close 90-135% of the coverage gap
- Automatically assign most unassigned adults
- Get us to 95-109% of SOI target

The household cap is the **primary bottleneck**. Fix this first, then validate and fine-tune.

---

*Analysis Date: 2025-10-15*  
*Current Coverage: 73.4% (466,355 / 635,117)*  
*Target Coverage: 100% (635,117)*
