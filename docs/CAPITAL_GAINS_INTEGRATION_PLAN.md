# Capital Gains Integration Plan for Synthetic Ultra-High-Income Units

**Date**: October 29, 2025  
**Status**: 📋 **PLAN - AWAITING APPROVAL**

---

## Executive Summary

Integrate capital gains modeling into synthetic ultra-high-income units ($1M+) using **Hawaii DOTAX SOI Table 21** data, which shows that **20.9% of taxable income** for $400K+ residents comes from net long-term capital gains.

### Key Data Point
**Hawaii DOTAX SOI 2022 Table 21**: $400K+ residents have **20.9% capital gains** share of total taxable income
- Ordinary income: 79.1%
- Capital gains: 20.9%

This is critical because:
1. **Current state**: Synthetic units assume 100% ordinary income
2. **Reality**: Ultra-high earners derive significant income from capital gains
3. **Tax impact**: Capital gains may be taxed at different rates than ordinary income

---

## Current State Analysis

### Synthetic Units (Current Implementation)

**File**: `src/tax/adjustments/ultra_high_income_synthesizer_v2.py` (lines 186-187)

```python
elif col in ['has_capital_gains', 'capital_gains', 'agi_adjustments']:
    synthetic_df[col] = 0  # ❌ No capital gains modeled
```

**Current synthetic units**:
- AGI: $5M, $10M, $25M, $50M
- Capital gains: $0 (0%)
- Ordinary income: 100% of AGI
- Tax calculation: Based entirely on ordinary income brackets

### Problem

Ultra-high-income filers have **significantly different income composition**:
- **Actual** (per DOTAX Table 21): 20.9% capital gains, 79.1% ordinary
- **Modeled** (current): 0% capital gains, 100% ordinary

This leads to:
1. **Overestimated tax liability**: Ordinary income taxed at higher rates
2. **Unrealistic income composition**: Missing major income source
3. **Inaccurate effective rates**: Not reflecting actual tax treatment

---

## Proposed Solution

### Phase 1: Parse Hawaii DOTAX Table 21 ✅ COMPLETE

**File**: `data/raw/Dotax Soi 2022 - 21.csv`

**Output**: `data/external/hawaii_capital_gains_by_agi.csv`

**Key findings**:
| AGI Class | Residents CG % | Nonresidents CG % |
|-----------|----------------|-------------------|
| $100K-$150K | 1.8% | 12.3% |
| $150K-$200K | 3.1% | 20.2% |
| $200K-$300K | 5.9% | 33.8% |
| $300K-$400K | 11.3% | 49.3% |
| **$400K+** | **20.9%** | **56.2%** |

### Phase 2: Define Capital Gains Shares by Synthetic Tier

Based on **National IRS SOI 2022 data** (Tables 14AR and 14ACG):

| Synthetic AGI | National CG Share | Recommended Hawaii Share | Ordinary Income Share |
|---------------|-------------------|-------------------------|----------------------|
| $5M | 31.6% | **30.0%** | 70.0% |
| $10M | 47.0% | **44.7%** | 55.3% |
| $25M | 49.4% | **47.0%** | 53.0% |
| $50M | 51.7% | **49.4%** | 50.6% |

**Rationale**:
- National $5M-$10M: 31.6% capital gains
- National $10M+: 47.0% capital gains
- Hawaii $400K+: 20.9% (DOTAX) vs National $500K-$1M: 12.4%
- **Hawaii has HIGHER CG share than national** (suggests more investment income)
- Use 95% of national shares for Hawaii (conservative adjustment)

### Phase 3: Update Synthetic Unit Creation

**File**: `src/tax/adjustments/ultra_high_income_synthesizer_v2.py`

**Changes needed**:

1. **Add capital gains share mapping** (based on national IRS SOI 2022):
```python
CAPITAL_GAINS_SHARES = {
    5_000_000: 0.30,   # 30% capital gains (national: 31.6%)
    10_000_000: 0.447, # 44.7% capital gains (national: 47.0%)
    25_000_000: 0.47,  # 47% capital gains (national: 49.4%)
    50_000_000: 0.494, # 49.4% capital gains (national: 51.7%)
}
```

2. **Update synthetic filer creation** (around line 126):
```python
synthetic_filers.append({
    'agi': spec['agi'],
    'filing_status': spec['filing_status'],
    'filing_status_hawaii': 'Joint_Surviving_Spouse',
    'num_dependents': 2,
    'num_adults': 2,
    'weight': weight_at_level,
    'is_synthetic_ultra_high': True,
    'total_deductions': 0,
    
    # NEW: Capital gains modeling
    'has_capital_gains': True,
    'capital_gains': spec['agi'] * CAPITAL_GAINS_SHARES[spec['agi']],
    'agi_without_cap_gains': spec['agi'] * (1 - CAPITAL_GAINS_SHARES[spec['agi']]),
    'agi_with_cap_gains': spec['agi'],
})
```

3. **Update field initialization** (around line 186):
```python
elif col in ['has_capital_gains']:
    synthetic_df[col] = True  # Changed from 0
elif col in ['capital_gains']:
    # Set based on AGI and capital gains share
    synthetic_df[col] = synthetic_df.apply(
        lambda row: row['agi'] * CAPITAL_GAINS_SHARES.get(row['agi'], 0.25),
        axis=1
    )
elif col in ['agi_without_cap_gains']:
    # Ordinary income = AGI - capital gains
    synthetic_df[col] = synthetic_df['agi'] - synthetic_df['capital_gains']
elif col in ['agi_with_cap_gains']:
    synthetic_df[col] = synthetic_df['agi']
```

### Phase 4: Update Tax Calculator (if needed)

**File**: `src/tax/brackets/hawaii_tax.py`

**Check if Hawaii tax calculator handles capital gains**:
1. Does Hawaii tax capital gains differently than ordinary income?
2. If yes, ensure calculator applies correct rates
3. If no, no changes needed (capital gains taxed as ordinary income)

**Research needed**: Hawaii tax treatment of long-term capital gains
- Federal: Preferential rates (0%, 15%, 20%)
- Hawaii: Need to verify if same treatment or taxed as ordinary income

### Phase 5: Validation

**Create validation script**: `scripts/validate_capital_gains_integration.py`

**Checks**:
1. Verify synthetic units have capital gains populated
2. Check capital gains shares match expected values
3. Compare tax liability before/after capital gains integration
4. Validate against DOTAX Table 21 aggregates

**Expected output**:
```
Synthetic Unit Capital Gains Validation:
AGI       CG Share    CG Amount    Ordinary Income    Tax (before)    Tax (after)
$5M       25%         $1.25M       $3.75M             $888,310        $TBD
$10M      30%         $3.00M       $7.00M             $1,811,075      $TBD
$25M      40%         $10.00M      $15.00M            $4,579,370      $TBD
$50M      50%         $25.00M      $25.00M            $9,193,194      $TBD
```

---

## Implementation Steps

### Step 1: Research Hawaii Capital Gains Tax Treatment ⏳ PENDING
**Action**: Determine if Hawaii taxes capital gains at preferential rates or as ordinary income
**Owner**: Research needed
**Deliverable**: Documentation of Hawaii CG tax treatment

### Step 2: Update UltraHighIncomeSynthesizerV2 ⏳ PENDING
**File**: `src/tax/adjustments/ultra_high_income_synthesizer_v2.py`
**Changes**:
- Add `CAPITAL_GAINS_SHARES` mapping
- Update synthetic filer creation to include CG fields
- Update field initialization logic

### Step 3: Update Tax Calculator (if needed) ⏳ PENDING
**File**: `src/tax/brackets/hawaii_tax.py`
**Changes**: TBD based on Step 1 research

### Step 4: Create Validation Script ⏳ PENDING
**File**: `scripts/validate_capital_gains_integration.py`
**Purpose**: Validate CG integration and measure impact

### Step 5: Run Full Pipeline and Compare ⏳ PENDING
**Action**: Run `scripts/regenerate_tax_units.py` and compare results
**Metrics**:
- $1M+ bracket tax (before vs after)
- Total tax gap (before vs after)
- Synthetic unit effective rates

---

## Expected Impact

### Tax Liability Changes

**Hypothesis**: If Hawaii taxes capital gains as ordinary income (no preferential treatment):
- **No change** in tax liability (CG + ordinary = same total AGI)
- **Better documentation** of income composition

**Hypothesis**: If Hawaii has preferential capital gains rates:
- **Lower tax liability** for synthetic units
- **Larger gap** in $1M+ bracket (may need to adjust)
- **More realistic** modeling of ultra-high-income taxation

### Example Calculation (if preferential rates exist)

**$5M synthetic unit (current)**:
- AGI: $5M (all ordinary)
- Tax: $888,310
- Effective rate: 17.77%

**$5M synthetic unit (with 25% CG at preferential rate)**:
- Ordinary income: $3.75M
- Capital gains: $1.25M
- Tax on ordinary: ~$666,000 (estimated)
- Tax on CG: ~$125,000 (if 10% preferential rate)
- Total tax: ~$791,000
- Effective rate: 15.82%

**Impact**: -$97,310 per unit, -$3.2M total for $5M tier

---

## Risk Assessment

### Low Risk
✅ **Data availability**: DOTAX Table 21 provides solid foundation  
✅ **Implementation complexity**: Straightforward field additions  
✅ **Reversibility**: Can easily revert if issues arise  

### Medium Risk
⚠️ **Tax calculator compatibility**: Need to verify CG handling  
⚠️ **Gap impact**: May increase $1M+ gap if preferential rates exist  

### Mitigation
- Research Hawaii CG tax treatment before implementation
- Run validation on small sample first
- Compare results with DOTAX aggregates
- Adjust synthetic unit weights if gap increases

---

## Decision Points

### Decision 1: Capital Gains Share by Tier
**Options**:
1. **Conservative**: Use 20.9% for all tiers (DOTAX $400K+ baseline)
2. **Graduated** (recommended): 25%, 30%, 40%, 50% by tier
3. **Aggressive**: Use national data (50-70% for ultra-wealthy)

**Recommendation**: Option 2 (Graduated) - balances realism with Hawaii data

### Decision 2: Tax Treatment
**Options**:
1. **Wait for research**: Delay until Hawaii CG treatment confirmed
2. **Assume ordinary**: Implement assuming CG taxed as ordinary income
3. **Assume preferential**: Implement with estimated preferential rates

**Recommendation**: Option 1 (Wait for research) - ensures accuracy

### Decision 3: Validation Threshold
**Options**:
1. **Strict**: Require <5% change in total gap
2. **Moderate**: Accept up to 10% change if more realistic
3. **Flexible**: Accept any change if income composition improves

**Recommendation**: Option 2 (Moderate) - prioritize realism over gap minimization

---

## Files to Create/Modify

### New Files
1. `scripts/validate_capital_gains_integration.py` - Validation script
2. `docs/HAWAII_CAPITAL_GAINS_TAX_RESEARCH.md` - Research findings

### Modified Files
1. `src/tax/adjustments/ultra_high_income_synthesizer_v2.py` - Add CG modeling
2. `src/tax/brackets/hawaii_tax.py` - Update if preferential rates exist
3. `scripts/regenerate_tax_units.py` - No changes needed (uses updated synthesizer)

### Data Files
1. ✅ `data/external/hawaii_capital_gains_by_agi.csv` - Parsed DOTAX Table 21

---

## Success Criteria

✅ **Synthetic units have capital gains populated** according to tier-specific shares  
✅ **Tax calculations reflect capital gains treatment** (ordinary or preferential)  
✅ **Validation shows realistic income composition** matching DOTAX patterns  
✅ **Total gap remains acceptable** (<15% ideally)  
✅ **Documentation complete** with research and rationale  

---

## Timeline Estimate

| Phase | Effort | Duration |
|-------|--------|----------|
| Research Hawaii CG tax | 1-2 hours | 0.5 days |
| Update synthesizer | 2-3 hours | 0.5 days |
| Update tax calculator | 1-4 hours | 0.5 days |
| Create validation | 2 hours | 0.5 days |
| Run and analyze | 1 hour | 0.5 days |
| **Total** | **7-12 hours** | **2-3 days** |

---

## Next Steps

### Immediate
1. **Research Hawaii capital gains tax treatment**
   - Check Hawaii Revised Statutes
   - Compare with federal treatment
   - Document findings

2. **Review plan with stakeholders**
   - Confirm capital gains shares by tier
   - Approve implementation approach
   - Set validation thresholds

### Upon Approval
1. Implement Phase 2-5 as outlined
2. Run validation and compare results
3. Document findings and update calibration

---

## Questions for Consideration

1. **Should we use different CG shares for residents vs nonresidents?**
   - DOTAX shows 20.9% (residents) vs 56.2% (nonresidents)
   - Current synthetic units don't distinguish residency
   - Recommendation: Use resident share (20.9% baseline) as most filers are residents

2. **Should we model short-term vs long-term capital gains separately?**
   - Table 21 only shows long-term CG
   - Short-term CG taxed as ordinary income
   - Recommendation: Model only long-term CG for simplicity

3. **Should we adjust synthetic unit weights after CG integration?**
   - If tax liability decreases, may need more synthetic units
   - Or increase tail multiplier
   - Recommendation: Evaluate after implementation

---

## Conclusion

Integrating capital gains into synthetic ultra-high-income units will:
- ✅ **Improve realism** of income composition
- ✅ **Align with DOTAX data** (20.9% CG share for $400K+)
- ✅ **Better model** ultra-high-income taxation
- ⚠️ **May impact gap** depending on Hawaii CG tax treatment

**Recommendation**: Proceed with implementation after confirming Hawaii capital gains tax treatment.

---

**Status**: 📋 **AWAITING APPROVAL TO PROCEED**

