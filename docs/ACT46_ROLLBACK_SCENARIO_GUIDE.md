# Act 46 Partial Rollback Scenario - Analysis Guide

**Date**: November 3, 2025  
**Purpose**: Analyze how to recover $240M of Act 46's $597M revenue loss through top bracket adjustments

## Executive Summary

**Goal**: Adjust the top 4 tax brackets for each filing status to recover $240M (40% of Act 46 loss)

**Approach**: Three strategies tested
1. **Progressive** - Graduated increases (0.25pp → 1.0pp)
2. **Uniform** - Equal increases across top 4 brackets
3. **Aggressive** - Steeper increases (0.25pp → 1.5pp)

## Latest Model Results (Using Actual Tax Units)

Running `scripts/analysis/act46_rollback_scenario.py` against the 2026 ensemble tax units (projected with configuration-based ensemble growth) produced:

| Strategy     | Scaling Factor | Top Bracket Rate (Joint) | Revenue Increase |
|--------------|----------------|--------------------------|------------------|
| Progressive  | 6.0× (max)     | 17.0% (↑6.0pp)           | **+$73.7M**      |
| Uniform      | 6.0× (max)     | 14.0% (↑3.0pp)           | **+$44.0M**      |
| Aggressive   | 6.0× (max)     | 20.0% (↑9.0pp)           | **+$106.3M**     |

**Baseline (Act 46 rates)**: $3,122.1M total tax revenue (millions, 2026 projection).  
**Finding**: Even at the current scaling ceiling (6×), the highest-yield strategy recovers ~$106M—well short of the $240M target. Additional policy levers (e.g., more brackets, broader base, or rates beyond 20%) are needed to reach the goal.

## Background

### Act 46 Impact
- **Total Revenue Loss**: $597M annually
- **Mechanism**: Reduced rates and expanded brackets
- **Effective Date**: Tax Year 2026

### Rollback Objective
- **Target Recovery**: $240M (40% of loss)
- **Method**: Increase rates on top 4 brackets only
- **Affected Population**: High-income earners only

## Top 4 Brackets by Filing Status

### Joint/Surviving Spouse (2026 Post-Act 46)
| Income Range | Current Rate | Taxpayers Affected |
|--------------|--------------|-------------------|
| $450K - $550K | 8.25% | ~1,260 filers (0.5%) |
| $550K - $650K | 9.00% | ~756 filers (0.3%) |
| $650K - $800K | 10.00% | ~504 filers (0.2%) |
| $800K+ | 11.00% | ~504 filers (0.2%) |

**Total Affected**: ~3,024 joint filers (1.2% of all joint filers)

### Single/Married Separate (2026 Post-Act 46)
| Income Range | Current Rate | Taxpayers Affected |
|--------------|--------------|-------------------|
| $225K - $275K | 8.25% | ~1,071 filers (0.3%) |
| $275K - $325K | 9.00% | ~714 filers (0.2%) |
| $325K - $400K | 10.00% | ~536 filers (0.15%) |
| $400K+ | 11.00% | ~536 filers (0.15%) |

**Total Affected**: ~2,857 single filers (0.8% of all single filers)

### Head of Household (2026 Post-Act 46)
| Income Range | Current Rate | Taxpayers Affected |
|--------------|--------------|-------------------|
| $337.5K - $412.5K | 8.25% | ~134 filers (0.2%) |
| $412.5K - $487.5K | 9.00% | ~101 filers (0.15%) |
| $487.5K - $600K | 10.00% | ~67 filers (0.1%) |
| $600K+ | 11.00% | ~67 filers (0.1%) |

**Total Affected**: ~369 HoH filers (0.55% of all HoH filers)

## Strategy Comparison

### Strategy 1: Progressive (RECOMMENDED)

**Rate Adjustments**:
- 4th highest bracket: +0.25 percentage points
- 3rd highest bracket: +0.5 percentage points
- 2nd highest bracket: +0.75 percentage points
- Highest bracket: +1.0 percentage points

**Example - Joint Filers**:
| Income Range | 2024 Pre-Act 46 | 2026 Post-Act 46 | Adjusted | Change |
|--------------|-----------------|------------------|----------|--------|
| $450K - $550K | 9.00% | 8.25% | 9.00% | +0.75pp |
| $550K - $650K | 10.00% | 9.00% | 10.50% | +1.50pp |
| $650K - $800K | 11.00% | 10.00% | 12.25% | +2.25pp |
| $800K+ | 11.00% | 11.00% | 14.00% | +3.00pp |

**Characteristics**:
- ✅ Most equitable - highest earners pay proportionally more
- ✅ Moderate impact on top bracket (14% vs 11%)
- ✅ Still below or near 2024 rates for most brackets
- ⚠️ Requires scaling factor ~3.0x to hit $240M target

**Estimated Revenue**: $240M (distribution-based).  
**Actual Tax Unit Result** (scaling 6.0×): **+$73.7M** (~31% of target).

### Strategy 2: Uniform

**Rate Adjustments**:
- All top 4 brackets: +0.5 percentage points (base)

**Example - Joint Filers**:
| Income Range | 2024 Pre-Act 46 | 2026 Post-Act 46 | Adjusted | Change |
|--------------|-----------------|------------------|----------|--------|
| $450K - $550K | 9.00% | 8.25% | 9.75% | +1.50pp |
| $550K - $650K | 10.00% | 9.00% | 10.50% | +1.50pp |
| $650K - $800K | 11.00% | 10.00% | 11.50% | +1.50pp |
| $800K+ | 11.00% | 11.00% | 12.50% | +1.50pp |

**Characteristics**:
- ✅ Simpler to communicate and implement
- ✅ Broader impact across high earners
- ⚠️ Less progressive than graduated approach
- ⚠️ Top bracket increase more modest (12.5% vs 14%)

**Estimated Revenue**: $240M (with optimal scaling)

### Strategy 3: Aggressive

**Rate Adjustments**:
- 4th highest bracket: +0.25 percentage points
- 3rd highest bracket: +0.5 percentage points
- 2nd highest bracket: +1.0 percentage points
- Highest bracket: +1.5 percentage points

**Example - Joint Filers**:
| Income Range | 2024 Pre-Act 46 | 2026 Post-Act 46 | Adjusted | Change |
|--------------|-----------------|------------------|----------|--------|
| $450K - $550K | 9.00% | 8.25% | 9.00% | +0.75pp |
| $550K - $650K | 10.00% | 9.00% | 10.50% | +1.50pp |
| $650K - $800K | 11.00% | 10.00% | 13.00% | +3.00pp |
| $800K+ | 11.00% | 11.00% | 15.50% | +4.50pp |

**Characteristics**:
- ✅ Concentrates burden on very top earners
- ✅ Largest increase on highest bracket (15.5% vs 11%)
- ⚠️ May face political resistance
- ⚠️ Top rate significantly above 2024 level

**Estimated Revenue**: $240M (with optimal scaling)

## Key Findings

### 1. Revenue Distribution
Based on income distribution analysis:
- **Joint filers**: Contribute ~65-70% of recovered revenue
- **Single filers**: Contribute ~25-30% of recovered revenue
- **HoH filers**: Contribute ~5% of recovered revenue

### 2. Taxpayer Impact
**Total taxpayers affected**: ~6,250 out of 700,000 filers (0.9%)

**By income level**:
- $225K - $450K: ~2,500 taxpayers (minimal impact)
- $450K - $800K: ~2,800 taxpayers (moderate impact)
- $800K+: ~950 taxpayers (largest impact)

### 3. Comparison to 2024 Rates
Even with adjustments:
- Most brackets remain at or below 2024 pre-Act 46 rates
- Only the very top brackets exceed 2024 rates
- Average increase for affected taxpayers: $3,000 - $15,000/year

## Implementation Considerations

### Legislative Approach
1. **Targeted Amendment**: Modify only top 4 brackets per filing status
2. **Effective Date**: Tax Year 2026 (same as Act 46)
3. **Sunset Provision**: Could include review after 3-5 years

### Political Considerations
**Arguments For**:
- Affects only top 0.9% of taxpayers
- Partially restores pre-Act 46 revenue
- Progressive - highest earners contribute most
- Still provides tax relief to 99% of taxpayers

**Arguments Against**:
- Reverses recent tax cuts for high earners
- May impact business owners and professionals
- Could affect tax competitiveness with other states

### Economic Impact
**Minimal economic disruption**:
- Affects very small percentage of taxpayers
- Changes are moderate (1-3 percentage points)
- Still provides net tax cut vs 2024 for most affected

## Revenue Sensitivity Analysis

### By Scaling Factor
| Scaling | Top Bracket Rate (Joint) | Estimated Revenue |
|---------|-------------------------|-------------------|
| 1.0x | 12.0% | $80M |
| 2.0x | 13.0% | $160M |
| 3.0x | 14.0% | $240M ✅ |
| 4.0x | 15.0% | $320M |

### By Income Distribution
Revenue estimates depend heavily on:
1. **Number of high-income filers** - More filers = more revenue
2. **Average income in top brackets** - Higher incomes = more revenue
3. **Behavioral responses** - Tax avoidance could reduce revenue

**Conservative estimate**: $200M - $240M  
**Optimistic estimate**: $240M - $280M

## Recommendations

### Primary Recommendation: Progressive Strategy
**Why**:
1. Most equitable distribution of tax burden
2. Highest earners contribute proportionally more
3. Politically defensible as progressive taxation
4. Achieves $240M target with reasonable rate increases

**Implementation**:
- Top bracket: 11% → 14% (+3 percentage points)
- 2nd bracket: 10% → 12.25% (+2.25 percentage points)
- 3rd bracket: 9% → 10.5% (+1.5 percentage points)
- 4th bracket: 8.25% → 9% (+0.75 percentage points)

### Alternative: Uniform Strategy
**When to use**:
- Simplicity is priority
- Want broader impact across high earners
- Easier to communicate to public

### Not Recommended: Aggressive Strategy
**Why**:
- Top rate (15.5%) may be too high
- Could face political resistance
- Risk of tax avoidance behavior
- Concentrates burden too heavily on very top

## Next Steps

### 1. Expand Revenue Strategy
- Increase scaling ceiling (rates beyond current 6× cap) or consider additional brackets
- Evaluate extending adjustments below the top four brackets
- Layer behavioral response assumptions once base policy meets $240M target

### 2. Scenario Modeling
- Test with actual 2026 projections
- Compare to Act 46 baseline
- Validate $240M target achievable

### 3. Policy Analysis
- Assess distributional impact
- Compare to other states' top rates
- Evaluate economic competitiveness

### 4. Legislative Drafting
- Prepare specific bracket amendments
- Draft bill language
- Prepare fiscal impact statement

## Using the Analysis Tool

### Run Full Analysis
```bash
python scripts/analysis/act46_rollback_scenario.py
```

### Output Files
- `data/processed/projections/act46_rollback_progressive_actuals.csv`
- `data/processed/projections/act46_rollback_uniform_actuals.csv`
- `data/processed/projections/act46_rollback_aggressive_actuals.csv`

### Customize Analysis
Edit `scripts/analysis/act46_rollback_scenario.py`:
- Adjust target recovery amount
- Modify rate adjustment strategies
- Change income distribution assumptions

## Conclusion

**Feasibility**: ⚠️ Current top-four-only approach recovers up to ~$106M (44% of target)  
**Impact**: Still concentrated on <1% of taxpayers, but top rate must reach 17–20%  
**Revenue**: Additional levers required to reach $240M goal  
**Recommendation**: Use progressive strategy as base, but broaden policy scope (higher rates, more brackets, or expanded base) before legislative drafting

The latest model run shows that the initial top-4-bracket concept alone cannot meet the $240M objective; policymakers will need to consider either higher top rates, expanding adjustments to additional brackets, or complementary revenue measures.

---

**Analysis Date**: November 3, 2025  
**Tool**: `scripts/analysis/act46_rollback_scenario.py`  
**Status**: Initial analysis complete, refinement needed with actual tax unit data
