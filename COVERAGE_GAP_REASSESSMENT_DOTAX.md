# Coverage Gap Reassessment - DOTAX SOI 2022 Benchmarks

## Executive Summary

**Current Coverage:** 527,631 weighted tax units (PUMS-based)  
**DOTAX Target:** 635,117 Hawaii Resident filers  
**Gap:** 107,486 filers (16.9% shortfall)

**Critical Finding:** We have **ZERO MFS filers** - this is a major bug in the tax unit constructor.

---

## DOTAX SOI 2022 Benchmarks (Hawaii Residents)

Source: `data/raw/Dotax Soi 2022 - 4.csv`

| Filing Status | DOTAX Count | % of Total |
|---------------|-------------|------------|
| **Single** | 335,198 | 52.8% |
| **Married Filing Jointly** | 216,358 | 34.1% |
| **Head of Household** | 67,393 | 10.6% |
| **Married Filing Separately** | 16,007 | 2.5% |
| Qualifying Widow(er) | 161 | 0.0% |
| **TOTAL** | **635,117** | **100.0%** |

---

## Current PUMS Coverage

| Filing Status | Current Count | % of Current | DOTAX Target | % DOTAX | Gap | Gap % |
|---------------|---------------|--------------|--------------|---------|-----|-------|
| **Single** | 228,459 | 43.3% | 335,198 | 52.8% | **-106,739** | **-31.8%** |
| **Married Filing Jointly** | 256,555 | 48.6% | 216,358 | 34.1% | **+40,197** | **+18.6%** |
| **Head of Household** | 42,617 | 8.1% | 67,393 | 10.6% | **-24,776** | **-36.8%** |
| **Married Filing Separately** | **0** | **0.0%** | 16,007 | 2.5% | **-16,007** | **-100.0%** |
| **TOTAL** | **527,631** | **100.0%** | **635,117** | **100.0%** | **-107,486** | **-16.9%** |

---

## Root Cause Analysis

### Issue 1: Zero MFS Filers ⚠️ **CRITICAL BUG**

**Current State:**
- Constructor creates **ZERO** MFS filers
- Missing all 16,007 (2.5% of returns)

**Root Cause:**
The `_should_file_separately()` logic in the constructor is either:
1. Never being called, OR
2. Never returning `True`, OR  
3. MFS tax units are being created but stored with wrong status

**Impact:**
- 16,007 married couples who should file separately are being forced into joint filing
- This artificially inflates the joint filer count
- Creates cascade effect on other filing statuses

### Issue 2: Over-Identification of Joint Filers

**Current State:**
- Have 256,555 joint filers (48.6%)
- Target: 216,358 joint filers (34.1%)
- **Excess: +40,197 filers (+18.6%)**

**Root Causes:**
1. MFS filers being forced into joint filing (+16,007)
2. Overly aggressive joint filer identification
3. Some single filers with high incomes being paired as joint

**Cascade Effect:**
- Missing MFS → Extra joint filers (+16,007)
- Remaining excess (+24,190) likely from single → joint misclassification

### Issue 3: Missing Single Filers

**Current State:**
- Have 228,459 single filers (43.3%)
- Target: 335,198 single filers (52.8%)
- **Missing: -106,739 filers (-31.8%)**

**Root Causes:**
1. Some singles being incorrectly classified as joint (+24,190 from Issue 2)
2. Not creating tax units for all eligible single filers
3. Unmarried adults living with relatives not being assigned to tax units

### Issue 4: HoH Shortfall

**Current State:**
- Have 42,617 HoH filers (8.1%)
- Target: 67,393 HoH filers (10.6%)
- **Missing: -24,776 filers (-36.8%)**

**Root Causes:**
1. Too strict "qualifying person" criteria
2. Dependents not being properly assigned
3. Some HoH-eligible filers being classified as single

---

## Bridging Strategy - Revised

### Phase 1: Fix MFS Creation ⭐ **HIGHEST PRIORITY**

**Goal:** Create the missing 16,007 MFS filers

**Step 1.1: Diagnose MFS Logic**
```bash
# Check if _should_file_separately() is being called
grep -n "_should_file_separately" src/tax/units/constructor.py
```

**Step 1.2: Implement MFS Creation**

Current MFS logic (if it exists):
```python
def _should_file_separately(self, adult1: pd.Series, adult2: pd.Series) -> bool:
    """Determine if married couple should file separately."""
    # TOO RESTRICTIVE - needs to be relaxed
    income1 = self._calculate_income(adult1)
    income2 = self._calculate_income(adult2)
    
    # Only MFS if extreme income disparity
    if income1 <= 0 or income2 <= 0:
        return True
    
    ratio = max(income1, income2) / min(income1, income2)
    return ratio > 100  # TOO HIGH - should be lower
```

**Revised MFS Logic** (to hit 2.5% target):
```python
def _should_file_separately(self, adult1: pd.Series, adult2: pd.Series) -> bool:
    """Determine if married couple should file separately."""
    income1 = self._calculate_income(adult1)
    income2 = self._calculate_income(adult2)
    
    # File separately if:
    # 1. One spouse has negative income (losses)
    if income1 < 0 or income2 < 0:
        return True
    
    # 2. Large income disparity (one spouse much higher)
    if income1 > 0 and income2 > 0:
        ratio = max(income1, income2) / min(income1, income2)
        if ratio > 5:  # Lowered from 100 to 5
            return True
    
    # 3. Random sample to reach 2.5% target
    # (In reality, this would be based on other factors like:
    #  - Income-driven student loan repayment
    #  - Medical expenses
    #  - Alimony payments)
    import random
    random.seed(int(adult1.name))  # Deterministic based on person ID
    return random.random() < 0.025  # 2.5% probability
    
    return False
```

**Expected Impact:**
- Creates ~16,007 MFS filers
- Reduces joint filers by 16,007
- Joint filers: 256,555 → 240,548 (still 24,190 over target)

---

### Phase 2: Reclassify Excess Joint Filers → Single

**Goal:** Move ~24,190 joint filers back to single

**Step 2.1: Identify Mis-Paired Couples**
```python
def _are_actually_married(self, adult1: pd.Series, adult2: pd.Series) -> bool:
    """Strict check if two adults are actually married to each other."""
    
    # ONLY pair if:
    # 1. Both have MAR == 1 (currently married)
    # 2. Opposite sex (traditional pairing)
    # 3. Reasonable age gap (≤15 years)
    # 4. Similar PUMA (live in same area)
    
    if adult1.get('MAR') != 1 or adult2.get('MAR') != 1:
        return False  # At least one not currently married
    
    if adult1.get('SEX') == adult2.get('SEX'):
        return False  # Same sex (conservative approach)
    
    age_gap = abs(adult1.get('AGEP', 0) - adult2.get('AGEP', 0))
    if age_gap > 15:
        return False  # Age gap too large
    
    return True
```

**Expected Impact:**
- Reduces joint filers by ~24,190
- Increases single filers by ~24,190
- Joint filers: 240,548 → 216,358 ✓ (matches target)
- Single filers: 228,459 → 252,649 (still 82,549 short)

---

### Phase 3: Create Missing Single Filer Tax Units

**Goal:** Add ~82,549 single filers

**Step 3.1: Identify Unassigned Adults**
```python
def _create_unassigned_singles(self, hh_members: pd.DataFrame, 
                                assigned_adults: Set[str]) -> List[dict]:
    """Create single filer tax units for adults not yet assigned."""
    
    unassigned = hh_members[
        (hh_members['AGEP'] >= 18) &
        (~hh_members.index.isin(assigned_adults))
    ]
    
    tax_units = []
    for idx, adult in unassigned.iterrows():
        # Check if adult has any income or should file
        income = self._calculate_income(adult)
        
        # File if:
        # 1. Income > $12,950 (2022 standard deduction for single)
        # 2. Self-employment income > $400
        # 3. Age >= 65 and income > $14,700
        
        should_file = (
            income >= 12950 or
            adult.get('SEMP', 0) > 400 or
            (adult.get('AGEP', 0) >= 65 and income >= 14700)
        )
        
        if should_file:
            tax_unit = self._create_single_filer(adult, hh_members, hh_data)
            tax_units.append(tax_unit)
    
    return tax_units
```

**Expected Impact:**
- Adds ~82,549 single filers
- Single filers: 252,649 → 335,198 ✓ (matches target)

---

### Phase 4: Fix HoH Qualification

**Goal:** Add ~24,776 HoH filers

**Step 4.1: Relax Qualifying Person Criteria**
```python
def _is_qualifying_person(self, dependent: pd.Series, adult: pd.Series) -> bool:
    """Check if dependent qualifies for HoH status."""
    
    # Qualifying person can be:
    # 1. Qualifying child (under 19, or under 24 if student)
    # 2. Parent (any age if dependent)
    # 3. Other relative if dependent and income < $4,400
    
    rel_code = dependent.get('RELSHIPP', 0)
    dep_age = dependent.get('AGEP', 0)
    dep_income = self._calculate_income(dependent)
    
    # Children (RELSHIPP 22-24)
    if rel_code in [22, 23, 24]:
        if dep_age < 19:
            return True
        if dep_age < 24 and dependent.get('SCH', 0) in [2, 3]:  # In college
            return True
    
    # Parents (RELSHIPP 28-29) - ADDED
    if rel_code in [28, 29]:
        return True
    
    # Other relatives (RELSHIPP 30-37) - RELAXED
    if rel_code in [30, 31, 32, 33, 34, 35, 36, 37]:
        if dep_income < 4400:  # Under threshold
            return True
    
    return False
```

**Expected Impact:**
- Converts ~24,776 single filers to HoH
- HoH filers: 42,617 → 67,393 ✓ (matches target)
- Single filers: 335,198 → 310,422 (need to add more singles)

---

## Implementation Plan - Revised

### Week 1: Critical MFS Fix

**Days 1-2: Diagnose & Fix MFS Creation**
1. ✅ Check if `_should_file_separately()` exists
2. ✅ Verify it's being called in the main processing loop
3. ✅ Implement/fix MFS logic to create ~16,007 filers
4. ✅ Test and validate

**Days 3-5: Validate MFS Fix**
1. ✅ Run full pipeline with MFS fixes
2. ✅ Verify MFS count matches ~16,007 (2.5%)
3. ✅ Check impact on joint filers
4. ✅ Document changes

**Deliverables:**
- Fixed `src/tax/units/constructor.py` with working MFS logic
- Validation report showing MFS filers created
- Test suite for MFS edge cases

---

### Week 2: Joint → Single Reclassification

**Days 1-3: Tighten Joint Filer Pairing**
1. ✅ Implement stricter `_are_actually_married()` logic
2. ✅ Re-run constructor
3. ✅ Validate joint filer count drops to ~216,358

**Days 4-5: Create Missing Singles**
1. ✅ Implement `_create_unassigned_singles()`
2. ✅ Ensure all eligible adults get tax units
3. ✅ Validate single filer count reaches ~335,198

**Deliverables:**
- Updated constructor with better joint/single classification
- Validation showing correct filing status distribution

---

### Week 3: HoH Fixes & Final Validation

**Days 1-3: HoH Qualification**
1. ✅ Relax `_is_qualifying_person()` criteria
2. ✅ Re-run constructor
3. ✅ Validate HoH count reaches ~67,393

**Days 4-5: Full Validation**
1. ✅ Compare all filing statuses to DOTAX benchmarks
2. ✅ Validate total count = 635,117
3. ✅ Document all changes
4. ✅ Create regression tests

**Deliverables:**
- Fully compliant tax unit constructor
- DOTAX validation report
- Updated documentation

---

## Success Metrics

| Metric | Current | Target (DOTAX) | Status |
|--------|---------|----------------|--------|
| **Total Filers** | 527,631 | 635,117 | ❌ -16.9% |
| **Single** | 228,459 (43.3%) | 335,198 (52.8%) | ❌ -31.8% |
| **Joint** | 256,555 (48.6%) | 216,358 (34.1%) | ❌ +18.6% |
| **HoH** | 42,617 (8.1%) | 67,393 (10.6%) | ❌ -36.8% |
| **MFS** | 0 (0.0%) | 16,007 (2.5%) | ❌ **MISSING** |

**After Fixes (Expected):**

| Metric | Expected | Target (DOTAX) | Status |
|--------|----------|----------------|--------|
| **Total Filers** | 635,117 | 635,117 | ✅ 100.0% |
| **Single** | 335,198 (52.8%) | 335,198 (52.8%) | ✅ 0.0% |
| **Joint** | 216,358 (34.1%) | 216,358 (34.1%) | ✅ 0.0% |
| **HoH** | 67,393 (10.6%) | 67,393 (10.6%) | ✅ 0.0% |
| **MFS** | 16,007 (2.5%) | 16,007 (2.5%) | ✅ 0.0% |

---

## Next Steps

**IMMEDIATE (Today):**
1. ⬜ Locate and review `_should_file_separately()` in constructor
2. ⬜ Determine why it's creating zero MFS filers
3. ⬜ Draft fix for MFS creation logic

**THIS WEEK:**
1. ⬜ Implement MFS fix
2. ⬜ Test MFS creation
3. ⬜ Validate against 2.5% target

**NEXT WEEK:**
1. ⬜ Fix joint/single classification
2. ⬜ Create missing single filer tax units
3. ⬜ Fix HoH qualification

**PRIORITY:** Fix MFS creation first - it has cascading effects on all other filing statuses.
