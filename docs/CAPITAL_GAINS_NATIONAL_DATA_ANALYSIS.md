# Capital Gains Analysis - National IRS SOI 2022 Data

**Date**: October 29, 2025  
**Status**: ✅ **ANALYSIS COMPLETE**

---

## Executive Summary

Analyzed **National IRS SOI 2022 data** (Tables 14AR and 14ACG) to determine capital gains shares for ultra-high-income earners. Found that **capital gains represent 30-50% of total income** for $5M+ earners, significantly higher than previously estimated.

### Key Finding

**Ultra-high earners derive nearly HALF their income from capital gains:**
- $5M-$10M: **31.6%** capital gains
- $10M+: **47.0%** capital gains
- $50M+ (estimated): **~50%** capital gains

---

## Data Sources

### National IRS SOI 2022

**Files**:
1. `22in14ar.xls` - Table 1.4: All Returns by AGI (Total Income)
2. `22in14acg.xls` - Table 1.4A: Returns with Capital Gains

**Coverage**: All US returns, Tax Year 2022

### Hawaii DOTAX SOI 2022

**File**: `Dotax Soi 2022 - 21.csv` - Table 21: Net Long-Term Capital Gains by AGI

**Coverage**: Hawaii returns only, Tax Year 2022

---

## National Capital Gains Share by AGI

| AGI Bracket | Returns | Total Income ($B) | Net LT CG ($B) | **CG Share %** |
|-------------|---------|-------------------|----------------|----------------|
| $500K-$1M | 1,674,608 | $1,124.30 | $138.99 | **12.4%** |
| $1M-$1.5M | 360,882 | $435.18 | $75.70 | **17.4%** |
| $1.5M-$2M | 148,221 | $254.48 | $49.65 | **19.5%** |
| $2M-$5M | 208,129 | $621.04 | $156.53 | **25.2%** |
| **$5M-$10M** | **52,968** | **$362.82** | **$114.74** | **31.6%** |
| **$10M+** | **34,630** | **$1,051.52** | **$494.68** | **47.0%** |

### Key Observations

1. **Progressive increase**: CG share rises steadily with income
2. **Inflection at $5M**: CG share jumps from 25% to 32%
3. **Near parity at $10M+**: Almost half of income is capital gains
4. **Extreme wealth**: Likely 50%+ for $50M+ earners

---

## Hawaii vs National Comparison

| Income Level | Hawaii CG % | National CG % | Difference |
|--------------|-------------|---------------|------------|
| $400K+ | **20.9%** | 12.4% ($500K-$1M) | **+8.5 pp** |
| $400K+ | **20.9%** | 17.4% ($1M-$1.5M) | **+3.5 pp** |

### Analysis

**Hawaii has HIGHER capital gains share than national average!**

**Possible explanations**:
1. **Retiree population**: Hawaii has many wealthy retirees living on investment income
2. **Real estate gains**: High property appreciation generates CG
3. **Tourism industry**: Business sales and real estate transactions
4. **Tax haven**: Wealthy individuals relocating for tax benefits
5. **Sample composition**: Hawaii $400K+ may skew wealthier than national $500K-$1M

**Implication**: Using national CG shares for Hawaii synthetic units is **conservative** (may underestimate CG).

---

## Recommended Capital Gains Shares for Synthetic Units

Based on national data with 5% haircut for conservatism:

| Synthetic AGI | National CG % | **Recommended Hawaii %** | Ordinary Income % |
|---------------|---------------|-------------------------|-------------------|
| **$5M** | 31.6% | **30.0%** | 70.0% |
| **$10M** | 47.0% | **44.7%** | 55.3% |
| **$25M** | 49.4% (est.) | **47.0%** | 53.0% |
| **$50M** | 51.7% (est.) | **49.4%** | 50.6% |

### Rationale

1. **$5M**: Use $5M-$10M national bracket (31.6%) → 30.0% for Hawaii
2. **$10M**: Use $10M+ national bracket (47.0%) → 44.7% for Hawaii
3. **$25M**: Interpolate between $10M and $50M → 47.0% for Hawaii
4. **$50M**: Extrapolate from $10M+ trend → 49.4% for Hawaii

**Conservative adjustment**: Use 95% of national shares to account for:
- Uncertainty in Hawaii-specific patterns
- Potential differences in ultra-wealthy composition
- Data quality and sample size considerations

---

## Implementation Impact Analysis

### Current State (No Capital Gains)

**Synthetic units** ($5M, $10M, $25M, $50M):
- Capital gains: **0%**
- Ordinary income: **100%**
- Tax calculation: Based entirely on ordinary income brackets

**Example**: $10M synthetic unit
- AGI: $10M (all ordinary)
- Tax: $1,811,075
- Effective rate: 18.11%

### Proposed State (With Capital Gains)

**Synthetic units** with capital gains modeled:
- Capital gains: **30-50%** (tier-dependent)
- Ordinary income: **50-70%**
- Tax calculation: Mixed income sources

**Example**: $10M synthetic unit (44.7% CG)
- AGI: $10M
  - Ordinary income: $5.53M
  - Capital gains: $4.47M
- Tax: **TBD** (depends on Hawaii CG tax treatment)
- Effective rate: **TBD**

---

## Critical Question: Hawaii Capital Gains Tax Treatment

**Federal treatment**:
- Long-term capital gains: Preferential rates (0%, 15%, 20%)
- Short-term capital gains: Ordinary income rates

**Hawaii treatment**: **RESEARCH NEEDED**

### Scenario A: Hawaii Taxes CG as Ordinary Income

**Impact**: 
- ✅ No change in tax liability (CG + ordinary = same total AGI)
- ✅ Better documentation of income composition
- ✅ More realistic modeling

**Example** ($10M unit):
- Tax: $1,811,075 (same as before)
- Just better income breakdown

### Scenario B: Hawaii Has Preferential CG Rates

**Impact**:
- ⚠️ Lower tax liability for synthetic units
- ⚠️ Larger gap in $1M+ bracket
- ✅ More accurate modeling of actual taxation

**Example** ($10M unit, assuming 10% preferential rate):
- Ordinary tax: $5.53M × ~18% = ~$996,000
- CG tax: $4.47M × 10% = ~$447,000
- Total tax: ~$1,443,000 (-$368,000 vs current)

**Impact on $1M+ bracket**:
- Current synthetic contribution: $217.1M
- With preferential rates: ~$170M (-$47M)
- Would increase gap from -16.0% to -23.1%

---

## Data Files Created

1. **`data/external/irs_soi_national_2022_income.xls`** - National total income by AGI
2. **`data/external/irs_soi_national_2022_capgains.xls`** - National capital gains by AGI
3. **`data/external/national_capital_gains_share_by_agi.csv`** - Parsed CG shares
4. **`data/external/hawaii_capital_gains_by_agi.csv`** - Hawaii DOTAX Table 21 parsed
5. **`docs/CAPITAL_GAINS_INTEGRATION_PLAN.md`** - Implementation plan
6. **`docs/CAPITAL_GAINS_NATIONAL_DATA_ANALYSIS.md`** - This document

---

## Next Steps

### Immediate (Before Implementation)

1. **Research Hawaii capital gains tax treatment** ⏳ CRITICAL
   - Check Hawaii Revised Statutes
   - Compare with federal treatment
   - Determine if preferential rates exist
   - Document findings

2. **Review implementation plan** ⏳ PENDING
   - Confirm capital gains shares (30%, 44.7%, 47%, 49.4%)
   - Approve implementation approach
   - Set validation thresholds

### Upon Approval

1. **Update `UltraHighIncomeSynthesizerV2`** with CG shares
2. **Update tax calculator** (if preferential rates exist)
3. **Create validation script** to measure impact
4. **Run full pipeline** and compare results
5. **Document findings** and adjust if needed

---

## Risk Assessment

### Low Risk ✅
- Data quality: National IRS SOI is authoritative
- Implementation complexity: Straightforward field additions
- Reversibility: Can easily revert if issues arise

### Medium Risk ⚠️
- Hawaii CG tax treatment unknown (need research)
- Potential gap increase if preferential rates exist
- May need to adjust synthetic unit weights

### Mitigation Strategy
1. Research Hawaii CG tax treatment BEFORE implementation
2. Run validation on small sample first
3. Compare results with DOTAX aggregates
4. Adjust synthetic unit weights if gap increases significantly

---

## Validation Metrics

### Success Criteria

✅ **Synthetic units have realistic income composition** (30-50% CG)  
✅ **Tax calculations reflect actual CG treatment**  
✅ **Total gap remains acceptable** (<20% ideally)  
✅ **$1M+ bracket improves or stays similar**  
✅ **Effective rates match DOTAX patterns**  

### Warning Thresholds

⚠️ **Total gap increases by >5 percentage points**  
⚠️ **$1M+ gap increases by >10 percentage points**  
⚠️ **Synthetic effective rates <10% or >25%**  

---

## Conclusion

National IRS SOI 2022 data reveals that **ultra-high-income earners derive 30-50% of their income from capital gains**, far higher than previously estimated. This has significant implications for synthetic unit modeling:

1. **Income composition**: Must model CG to be realistic
2. **Tax treatment**: Critical to understand Hawaii's CG taxation
3. **Gap impact**: May increase gap if preferential rates exist
4. **Data quality**: National data provides solid foundation

**Recommendation**: 
1. ✅ **Proceed with CG integration** using national-calibrated shares
2. ⏳ **Research Hawaii CG tax treatment** before finalizing
3. ⏳ **Validate carefully** and adjust weights if needed

---

**Status**: ✅ **ANALYSIS COMPLETE - READY FOR IMPLEMENTATION DECISION**

