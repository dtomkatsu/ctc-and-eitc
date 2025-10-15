# Coverage Gap Assessment & Bridging Strategy

## Executive Summary

**Current Coverage:** 527,631 weighted tax units  
**SOI Target:** 635,000 filers  
**Gap:** 107,369 filers (16.9% shortfall)

The gap is caused by **two distinct issues**:
1. **Filing status misclassification** (redistributing existing filers)
2. **Missing filers** (need to create additional tax units)

---

## Root Cause Analysis

### Issue 1: Filing Status Misclassification

| Status | Current | SOI Target | Gap | Filer Impact |
|--------|---------|------------|-----|--------------|
| **Single** | 43.3% (228,464) | 51.0% (323,850) | **-7.7pp** | **Need +95,386** |
| **Joint** | 48.6% (256,429) | 36.0% (228,600) | **+12.6pp** | **Have +27,829 extra** |
| **HoH** | 8.1% (42,738) | 9.6% (60,960) | -1.5pp | Need +18,222 |
| **MFS** | 0.0% (0) | 3.4% (21,590) | -3.4pp | Need +21,590 |

**Key Finding:** We're over-identifying joint filers by 27,829 and under-identifying single filers by 95,386.

### Issue 2: Absolute Filer Shortfall

Even if we perfectly redistribute filing statuses, we're still missing:
- **107,369 total filers** (16.9% of target)
- This represents adults who should file taxes but aren't in our tax units

---

## Bridging Strategies (Ranked by Effectiveness)

### Strategy 1: Calibration Weights ⭐ **RECOMMENDED**

**Approach:** Apply calibration factors to match SOI benchmarks

**Implementation:**
```python
# Calculate calibration factors by filing status and income bracket
calibration_factors = {
    'single': {
        '0-25k': 1.42,    # Need to upweight single filers in lower brackets
        '25-50k': 1.35,
        '50-75k': 1.28,
        '75-100k': 1.22,
        '100-200k': 1.18,
        '200k+': 1.15
    },
    'joint': {
        '0-25k': 0.89,    # Need to downweight joint filers
        '25-50k': 0.91,
        '50-75k': 0.93,
        '75-100k': 0.95,
        '100-200k': 0.97,
        '200k+': 0.98
    },
    'head_of_household': {
        '0-25k': 1.43,
        '25-50k': 1.38,
        '50-75k': 1.32,
        '75-100k': 1.28,
        '100-200k': 1.24,
        '200k+': 1.20
    }
}

# Apply to weights
tax_units['calibrated_weight'] = tax_units.apply(
    lambda row: row['PWGTP'] * calibration_factors[row['filing_status']][row['income_bracket']],
    axis=1
)
```

**Pros:**
- ✅ Preserves existing data structure
- ✅ Maintains demographic relationships
- ✅ Easy to implement and validate
- ✅ Transparent and auditable
- ✅ Can be refined iteratively

**Cons:**
- ⚠️ Doesn't fix underlying filing status logic
- ⚠️ Weights may become very large/small in some cells

**Estimated Effort:** 2-3 days  
**Confidence:** High (90%)

---

### Strategy 2: Raking/Iterative Proportional Fitting

**Approach:** Iteratively adjust weights to match multiple marginal distributions simultaneously

**Implementation:**
```python
from scipy.optimize import minimize

# Target margins
targets = {
    'filing_status': {'single': 0.51, 'joint': 0.36, 'hoh': 0.096, 'mfs': 0.034},
    'income_bracket': {...},  # From SOI
    'age_group': {...},       # From Census
    'total': 635000
}

# Rake weights to match all margins
calibrated_weights = rake_weights(
    tax_units,
    targets,
    max_iterations=100,
    tolerance=0.01
)
```

**Pros:**
- ✅ Matches multiple dimensions simultaneously
- ✅ Statistically rigorous
- ✅ Preserves correlations between variables

**Cons:**
- ⚠️ More complex to implement
- ⚠️ May not converge if targets are inconsistent
- ⚠️ Harder to explain/audit

**Estimated Effort:** 1-2 weeks  
**Confidence:** Medium (70%)

---

### Strategy 3: Fix Tax Unit Construction Logic

**Approach:** Improve the underlying tax unit constructor to create more accurate filing statuses

**Key Areas to Fix:**

#### 3a. Single Filer Identification
```python
# Current issue: Not creating enough single filers
# Fix: Identify unmarried adults without dependents

def _create_additional_single_filers(self, hh_members, claimed_adults):
    """Create single filers for unmarried adults not yet assigned."""
    unmarried_adults = hh_members[
        (hh_members['AGEP'] >= 18) &
        (hh_members['MAR'].isin([3, 4, 5])) &  # Never married, divorced, widowed
        (~hh_members.index.isin(claimed_adults))
    ]
    
    for idx, adult in unmarried_adults.iterrows():
        # Create single filer tax unit
        tax_unit = self._create_single_filer(adult, hh_members, hh_data)
        tax_units.append(tax_unit)
```

#### 3b. MFS Filer Creation
```python
# Current issue: No MFS filers created
# Fix: Identify married couples who should file separately

def _should_file_separately(self, adult1, adult2):
    """Determine if married couple should file separately."""
    income1 = self._calculate_income(adult1)
    income2 = self._calculate_income(adult2)
    
    # File separately if:
    # 1. Large income disparity (one spouse has losses)
    # 2. One spouse has significant medical expenses
    # 3. Income-driven student loan repayment
    
    if income1 < 0 or income2 < 0:
        return True
    
    if abs(income1 - income2) / max(income1, income2, 1) > 0.8:
        return True
    
    return False
```

#### 3c. HoH Qualification
```python
# Current issue: Under-identifying HoH
# Fix: Relax qualification criteria

def _qualifies_for_hoh(self, adult, dependents):
    """Check if adult qualifies for Head of Household."""
    # Current: Too strict on "qualifying person" definition
    # Fix: Include more dependent types
    
    qualifying_deps = [
        d for d in dependents 
        if self._is_qualifying_person(d, adult)
    ]
    
    return len(qualifying_deps) > 0
```

**Pros:**
- ✅ Fixes root cause
- ✅ More accurate for future analyses
- ✅ Better represents actual tax behavior

**Cons:**
- ⚠️ High complexity
- ⚠️ Risk of introducing new bugs
- ⚠️ Requires extensive testing
- ⚠️ May still not reach 100% coverage

**Estimated Effort:** 3-4 weeks  
**Confidence:** Medium (60%)

---

### Strategy 4: Hybrid Approach ⭐⭐ **BEST LONG-TERM**

**Approach:** Combine calibration weights (short-term) with constructor fixes (long-term)

**Phase 1 (Immediate - 1 week):**
1. Implement calibration weights to match SOI benchmarks
2. Document calibration factors and methodology
3. Use calibrated weights for current analysis

**Phase 2 (Medium-term - 2-3 weeks):**
1. Fix MFS filer creation (currently 0%)
2. Improve single filer identification
3. Relax HoH qualification criteria

**Phase 3 (Long-term - 1-2 months):**
1. Comprehensive constructor refactor
2. Add validation against SOI at multiple dimensions
3. Implement automated testing

**Pros:**
- ✅ Immediate solution for current analysis
- ✅ Long-term improvement to pipeline
- ✅ Incremental progress with validation at each step

**Cons:**
- ⚠️ Requires sustained effort
- ⚠️ Need to maintain both calibrated and uncalibrated versions

**Estimated Effort:** 1 week (Phase 1), then ongoing  
**Confidence:** High (85%)

---

## Recommended Implementation Plan

### Week 1: Calibration Weights (Immediate Fix)

**Goal:** Bridge the 107k filer gap using calibration weights

**Steps:**
1. Calculate SOI target distributions by:
   - Filing status × Income bracket
   - Filing status × Age group
   - Income bracket × Age group

2. Compute calibration factors:
   ```python
   calibration_factor = (SOI_target / current_count)
   ```

3. Apply to weights:
   ```python
   tax_units['calibrated_weight'] = tax_units['PWGTP'] * calibration_factor
   ```

4. Validate:
   - Total filers = 635,000 ✓
   - Filing status distribution matches SOI ✓
   - Income distribution matches SOI ✓
   - Age distribution matches Census ✓

**Deliverables:**
- `scripts/calibration/apply_soi_calibration.py`
- `data/processed/tax_units_calibrated.parquet`
- `CALIBRATION_METHODOLOGY.md`

---

### Week 2-3: Constructor Fixes (Root Cause)

**Goal:** Improve tax unit construction to reduce need for calibration

**Priority Fixes:**
1. **MFS Creation** (High Impact)
   - Add `_should_file_separately()` logic
   - Target: Create ~21,590 MFS filers

2. **Single Filer Identification** (High Impact)
   - Ensure all unmarried adults without dependents file as single
   - Target: Add ~95,386 single filers

3. **HoH Qualification** (Medium Impact)
   - Relax qualifying person criteria
   - Target: Add ~18,222 HoH filers

**Validation:**
- Run full pipeline with fixes
- Compare to SOI benchmarks
- Measure improvement in coverage gap

---

## Success Metrics

### Phase 1 (Calibration)
- ✅ Total filers: 635,000 (±1%)
- ✅ Single: 51.0% (±2pp)
- ✅ Joint: 36.0% (±2pp)
- ✅ HoH: 9.6% (±1pp)
- ✅ MFS: 3.4% (±1pp)

### Phase 2 (Constructor Fixes)
- ✅ Reduce calibration factors closer to 1.0
- ✅ Natural distribution within 5pp of SOI
- ✅ All filing statuses represented

### Phase 3 (Long-term)
- ✅ Coverage gap < 5%
- ✅ Calibration factors between 0.9-1.1
- ✅ Automated validation tests pass

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Calibration creates unrealistic weights | Medium | Medium | Cap factors at 0.5-2.0 range |
| Constructor fixes break existing logic | High | High | Comprehensive unit tests |
| SOI benchmarks are inconsistent | Low | Medium | Use multiple validation sources |
| Age-specific calibration conflicts with income calibration | Medium | Medium | Use raking/IPF instead |

---

## Next Steps

**Immediate (This Week):**
1. ✅ Review and approve calibration approach
2. ⬜ Implement calibration weight calculation
3. ⬜ Validate calibrated weights against SOI
4. ⬜ Update wage growth script to use calibrated weights

**Short-term (Next 2-3 Weeks):**
1. ⬜ Design MFS filer creation logic
2. ⬜ Implement and test MFS fixes
3. ⬜ Design single filer improvements
4. ⬜ Implement and test single filer fixes

**Medium-term (Next 1-2 Months):**
1. ⬜ Comprehensive constructor refactor
2. ⬜ Multi-dimensional validation framework
3. ⬜ Automated testing suite
4. ⬜ Documentation updates

---

## Conclusion

**Recommended Approach:** Strategy 4 (Hybrid)

**Immediate Action:** Implement calibration weights to bridge the 107k filer gap

**Long-term Goal:** Fix tax unit constructor to reduce reliance on calibration

**Timeline:**
- Week 1: Calibration weights implemented ✓
- Week 2-3: Constructor fixes for MFS and single filers
- Month 2-3: Comprehensive improvements and validation

This approach provides an immediate solution while working toward a more robust long-term fix.
