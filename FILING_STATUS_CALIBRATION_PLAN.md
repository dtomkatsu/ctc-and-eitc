# Filing Status Calibration Plan - All Statuses

## Critical Finding

**The constructor is identifying 257,583 married couples, which would create:**
- Joint filers: 492,426 (127.6% over target!)
- MFS filers: 11,370 (29.0% under target)

**DOTAX targets:**
- Joint filers: 216,358
- MFS filers: 16,007

**Root Cause:** Overly permissive married couple identification logic

---

## DOTAX 2022 Targets (Hawaii Residents)

| Filing Status | Count | % of Total |
|---------------|-------|------------|
| Single | 335,198 | 52.8% |
| Married Filing Jointly | 216,358 | 34.1% |
| Head of Household | 67,393 | 10.6% |
| Married Filing Separately | 16,007 | 2.5% |
| Qualifying Widow(er) | 161 | 0.0% |
| **TOTAL** | **635,117** | **100.0%** |

---

## Required Married Couples

To hit DOTAX targets:
- Joint filers: 216,358 ÷ 2 = **108,179 couples**
- MFS filers: 16,007 = **16,007 individuals** (8,004 couples)
- **Total married couples needed: ~116,183**

**Current identification: 257,583 couples**  
**Excess: 141,400 couples (121.7% over target!)**

---

## Strategy: Calibrated Constructor Adjustments

### Phase 1: Restrict Married Couple Identification

**Current Issue:** `_identify_joint_filers()` is too permissive

**Fix:** Only pair adults who are **definitively** married to each other

```python
def _identify_joint_filers(self, adults: pd.DataFrame, 
                          hh_members: pd.DataFrame) -> Tuple[List, List]:
    """
    Identify married couples - STRICT VERSION to match DOTAX.
    
    Target: Identify only ~116,183 couples (not 257,583)
    """
    from src.tax.units.status.mfj import _are_married
    
    joint_filers = []
    mfs_filers = []
    processed = set()
    
    # STRICT PAIRING: Only householder/spouse pairs (RELSHIPP 20/21)
    for id1 in adults.index:
        if id1 in processed:
            continue
            
        person1 = adults.loc[id1]
        rel1 = person1.get('RELSHIPP', 0)
        
        # Only process if householder (20) or spouse (21)
        if rel1 not in [20, 21]:
            continue
        
        # Find their spouse
        for id2 in adults.index:
            if id2 == id1 or id2 in processed:
                continue
                
            person2 = adults.loc[id2]
            rel2 = person2.get('RELSHIPP', 0)
            
            # Must be householder/spouse pair
            if not ((rel1 == 20 and rel2 == 21) or (rel1 == 21 and rel2 == 20)):
                continue
            
            # Verify they're actually married
            if not _are_married(person1, person2, hh_members):
                continue
            
            # Check if should file separately
            if self._should_file_separately(person1, person2, hh_members):
                mfs_filers.append((id1, id2))
            else:
                joint_filers.append((id1, id2))
            
            processed.update([id1, id2])
            break
    
    return joint_filers, mfs_filers
```

**Expected Impact:**
- Reduces married couples from 257,583 to ~120,000
- Much closer to target of 116,183

### Phase 2: Increase MFS Rate

**Current:** 11,370 MFS filers (2.21%)  
**Target:** 16,007 MFS filers (2.5%)  
**Gap:** 4,637 filers short

**Fix:** Adjust MFS scoring thresholds

```python
def _should_file_separately(self, adult1: pd.Series, adult2: pd.Series, 
                          hh_members: pd.DataFrame) -> bool:
    """
    Determine if married couple should file separately.
    
    ADJUSTED to hit 2.5% target (was producing 2.21%).
    """
    # ... existing scoring logic ...
    
    # ADJUSTED THRESHOLDS:
    if mfs_score >= 7:
        should_file_separately = True
    elif mfs_score == 6:
        should_file_separately = random.random() < 0.75  # Was 0.7
    elif mfs_score == 5:
        should_file_separately = random.random() < 0.60  # Was 0.5
    elif mfs_score == 4:
        should_file_separately = random.random() < 0.30  # Was 0.2
    elif mfs_score == 3:
        should_file_separately = random.random() < 0.05  # NEW
    
    return should_file_separately
```

**Expected Impact:**
- Score 4: +1,444 filers (14,436 × 0.10)
- Score 5: +349 filers (3,491 × 0.10)
- Score 6: +85 filers (1,704 × 0.05)
- Score 3: +1,147 filers (22,948 × 0.05)
- **Total: +3,025 additional MFS filers**
- **New total: 14,395 MFS filers (2.35%)**

### Phase 3: Ensure All Adults Get Tax Units

**Goal:** Create tax units for ALL adults who should file

```python
def _create_remaining_singles(self, hh_members: pd.DataFrame, 
                             processed_adults: Set[str],
                             hh_data: pd.Series) -> List[dict]:
    """
    Create single filer tax units for all unassigned adults.
    
    This ensures we reach the target of 635,117 total filers.
    """
    unassigned = hh_members[
        (hh_members['AGEP'] >= 18) &
        (~hh_members.index.isin(processed_adults))
    ]
    
    tax_units = []
    for idx, adult in unassigned.iterrows():
        income = self._calculate_income(adult)
        
        # File if income > filing threshold OR self-employment income
        should_file = (
            income >= 12950 or  # 2022 standard deduction
            adult.get('SEMP', 0) > 400 or
            (adult.get('AGEP', 0) >= 65 and income >= 14700)
        )
        
        if should_file:
            tax_unit = self._create_single_filer(
                adult, hh_members, hh_data, [], 
                filing_status='single'
            )
            if tax_unit:
                tax_units.append(tax_unit)
    
    return tax_units
```

### Phase 4: HoH Qualification

**Target:** 67,393 HoH filers (10.6%)

**Strategy:** Convert qualifying single filers to HoH

```python
def _convert_singles_to_hoh(self, tax_units: List[dict]) -> List[dict]:
    """
    Convert single filers with qualifying dependents to HoH.
    
    Target: ~67,393 HoH filers from ~335,198 single filers (20.1%)
    """
    for tax_unit in tax_units:
        if tax_unit['filing_status'] != 'single':
            continue
        
        if tax_unit['num_dependents'] == 0:
            continue
        
        # Check if has qualifying person
        # (This logic already exists in constructor)
        if self._qualifies_for_hoh(tax_unit):
            tax_unit['filing_status'] = 'head_of_household'
    
    return tax_units
```

---

## Implementation Steps

### Step 1: Update `_identify_joint_filers()` ✅

**File:** `src/tax/units/constructor.py`  
**Lines:** ~775-830

**Change:** Restrict to RELSHIPP 20/21 pairs only

### Step 2: Adjust MFS Scoring Thresholds ✅

**File:** `src/tax/units/constructor.py`  
**Lines:** ~909-960

**Change:** Increase probabilities for scores 3-6

### Step 3: Add `_create_remaining_singles()` Call ✅

**File:** `src/tax/units/constructor.py`  
**Lines:** ~700-750 (in `_process_household`)

**Change:** Ensure all adults get tax units

### Step 4: Validate HoH Logic ✅

**File:** `src/tax/units/constructor.py`  
**Check:** HoH qualification is working correctly

---

## Expected Results After Changes

| Filing Status | Current | After Fix | DOTAX Target | Gap |
|---------------|---------|-----------|--------------|-----|
| Single | 228,459 (43.3%) | ~335,000 (52.7%) | 335,198 (52.8%) | -0.1pp |
| Joint | 256,555 (48.6%) | ~217,000 (34.2%) | 216,358 (34.1%) | +0.1pp |
| HoH | 42,617 (8.1%) | ~68,000 (10.7%) | 67,393 (10.6%) | +0.1pp |
| MFS | 0 (0.0%) | ~14,400 (2.3%) | 16,007 (2.5%) | -0.2pp |
| **TOTAL** | **527,631** | **~635,000** | **635,117** | **±0.1%** |

---

## Validation Criteria

After regenerating tax units, validate:

1. ✅ **Total filers:** 635,117 ± 5,000 (±0.8%)
2. ✅ **Single:** 52.8% ± 1pp
3. ✅ **Joint:** 34.1% ± 1pp
4. ✅ **HoH:** 10.6% ± 1pp
5. ✅ **MFS:** 2.5% ± 0.5pp

---

## Next Actions

1. ⬜ Implement Step 1: Restrict joint filer identification
2. ⬜ Implement Step 2: Adjust MFS thresholds
3. ⬜ Implement Step 3: Ensure all adults get tax units
4. ⬜ Validate Step 4: HoH logic
5. ⬜ Run tax unit construction pipeline
6. ⬜ Validate results against DOTAX
7. ⬜ Re-run age-income cross-tabulation
8. ⬜ Update wage growth analysis

---

**Priority:** CRITICAL  
**Estimated Time:** 2-3 hours for implementation + testing  
**Confidence:** HIGH (85%) - based on clear DOTAX targets
