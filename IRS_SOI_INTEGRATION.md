# IRS SOI Data Integration Strategy

**Date**: October 16, 2025  
**Purpose**: Integrate IRS SOI filing status × AGI data with existing calibration approach

## Overview

The IRS SOI 2022 data provides **filing status distribution WITHIN each AGI bracket** - a critical piece missing from DOTAX data. This allows more precise calibration that addresses the $50k-$75k over-representation issue.

## Data Sources Comparison

### DOTAX SOI 2022 (Current)

**Strengths:**
- ✅ Very granular AGI brackets (12 brackets)
- ✅ Hawaii-specific (state tax authority data)
- ✅ Detailed tax liability data

**Limitations:**
- ❌ No filing status breakdown by AGI bracket
- ❌ Can't tell if $50k-$75k has too many Singles vs Joints

**Tables Used:**
- Table 12A: Returns and tax by AGI bracket (all statuses combined)
- Table 13A/13B/13C: Returns by filing status (all incomes combined)

### IRS SOI 2022 (New)

**Strengths:**
- ✅ Filing status breakdown BY AGI bracket
- ✅ Shows income distribution differs by filing status
- ✅ Federal source (potentially more comprehensive)

**Limitations:**
- ❌ Only 6 AGI brackets (vs DOTAX's 12)
- ❌ Totals 5.9% lower than DOTAX (674,660 vs 635,117)
- ❌ Different methodologies may explain gap

**Key Insight:**
```
In $50k-$75k bracket:
- 54.6% are Single
- 28.7% are Married Filing Jointly  
- 13.4% are Head of Household
```

This tells us HOW to reduce the over-representation - not uniformly, but proportionally by filing status.

## Data Differences Analysis

### Total Returns
| Source | Total Returns | Difference |
|--------|---------------|------------|
| IRS SOI | 674,660 | Baseline |
| DOTAX | 635,117 | -39,543 (-5.9%) |

**Likely Explanation:** IRS includes all federal returns; DOTAX may exclude certain categories or have different timing.

### Filing Status Totals
| Status | IRS SOI | DOTAX | Difference |
|--------|---------|-------|------------|
| Single | 349,070 | 335,198* | -13,872 (-4.0%) |
| Joint | 236,930 | 216,358 | -20,572 (-8.7%) |
| HoH | 70,490 | 67,393 | -3,097 (-4.4%) |
| MFS | N/A | 16,007 | N/A |

*DOTAX combines Single+MFS (351,205); estimate ~95% Single

### AGI Brackets (Comparable Ranges)
| Bracket | IRS SOI | DOTAX | Difference |
|---------|---------|-------|------------|
| $0-$25k | 168,230 | 193,536 | +15.0% |
| $25k-$50k | 158,030 | 171,217 | +8.3% |
| **$50k-$75k** | **110,080** | **91,459** | **-16.9%** ⚠️ |
| $75k-$100k | 71,880 | 54,976 | -23.5% |
| $100k-$200k | 122,110 | 90,041 | -26.3% |
| $200k+ | 44,330 | 33,888 | -23.6% |

**Pattern:** IRS has more low-income returns, DOTAX has more high-income returns.

## Integration Approach

### Option 1: IRS SOI with DOTAX Scaling (RECOMMENDED)

**Strategy:** Use IRS SOI proportions, scale to DOTAX totals

**Rationale:**
- IRS provides the SHAPE of the distribution (proportions)
- DOTAX provides the SIZE (totals)
- Combining both leverages strengths of each

**Implementation:**
```python
# For each AGI bracket:
# 1. Get IRS proportions
single_pct = IRS_single / IRS_total
joint_pct = IRS_joint / IRS_total
hoh_pct = IRS_hoh / IRS_total

# 2. Scale to DOTAX total for that bracket
single_target = DOTAX_total * single_pct
joint_target = DOTAX_total * joint_pct
hoh_target = DOTAX_total * hoh_pct

# 3. Calibrate each (status, bracket) cell to its target
```

**Example for $50k-$75k:**
```
IRS proportions:
  - Single: 60,050 / 110,080 = 54.6%
  - Joint: 31,570 / 110,080 = 28.7%
  - HoH: 14,710 / 110,080 = 13.4%

DOTAX total: 91,459

Scaled targets:
  - Single: 91,459 × 54.6% = 49,937
  - Joint: 91,459 × 28.7% = 26,249
  - HoH: 91,459 × 13.4% = 12,256
```

**Advantages:**
- ✅ Fixes $50k-$75k over-representation precisely
- ✅ Ensures correct filing status MIX within bracket
- ✅ Respects both data sources
- ✅ More accurate than single-layer approaches

**Calibration Flow:**
```
1. Assign each tax unit to (filing_status, IRS_AGI_bracket)
2. Calculate current count in each cell
3. Calculate factor: target / current
4. Multiply weight by factor
```

### Option 2: Pure IRS SOI (Alternative)

**Strategy:** Use IRS SOI targets directly, ignore DOTAX totals

**Advantages:**
- ✅ Simpler (one data source)
- ✅ Consistent methodology

**Disadvantages:**
- ❌ Ignores DOTAX's Hawaii-specific counts
- ❌ May under-count total returns by ~6%
- ❌ Less validated for Hawaii

**When to use:** If you trust IRS more than DOTAX for Hawaii.

### Option 3: Iterative Multi-Layer (Most Complex)

**Strategy:** Sequential calibration with multiple iterations

**Flow:**
```
Layer 1: Match DOTAX filing status totals
Layer 2: Match IRS filing status × AGI distribution
Layer 3: Match DOTAX fine-grained AGI brackets
Layer 4: Re-balance filing status (if needed)
```

**Advantages:**
- ✅ Matches all benchmarks eventually
- ✅ Very precise

**Disadvantages:**
- ❌ Complex, hard to debug
- ❌ May not converge
- ❌ Over-fitting risk

## Recommended Implementation

### Step 1: Use IRS SOI Calibration (New Module)

```python
from src.tax.validation.irs_soi_calibration import apply_irs_soi_calibration

# Apply IRS SOI calibration
tax_units = apply_irs_soi_calibration(
    tax_units,
    weight_col='weight',
    output_col='weight_irs_calibrated'
)
```

This gives you:
- Correct filing status distribution WITHIN each AGI bracket
- Scaled to DOTAX totals (best of both worlds)

### Step 2: Validate Against Both Sources

```python
# Validate against DOTAX Table 12A
from src.tax.validation.dotax_table_12a import DotaxTable12AValidator
validator = DotaxTable12AValidator()
results = validator.validate(tax_units, weight_col='weight_irs_calibrated')

# Validate against IRS SOI
from src.tax.validation.irs_soi_calibration import validate_irs_soi_calibration
irs_results = validate_irs_soi_calibration(tax_units, weight_col='weight_irs_calibrated')
```

### Step 3: Compare to Existing Approaches

```python
# Compare:
# 1. Current (bracket_calibration.py only)
# 2. Two-layer (bracket + agi_calibration.py)
# 3. IRS SOI (new approach)

# See which gives best overall accuracy
```

## Expected Results

### Current Approach (Single-Layer)
- Filing status: ✅ 100% accurate
- AGI brackets: ⚠️ 8.3% within ±10%
- **$50k-$75k: +24.6% over** ❌

### Two-Layer Approach (Previous Recommendation)
- Filing status: ✅ ~99% accurate
- AGI brackets: ✅ ~75% within ±10%
- **$50k-$75k: ~±5%** ✅

### IRS SOI Approach (New)
- Filing status: ✅ ~95% accurate (slightly lower due to IRS/DOTAX differences)
- AGI brackets: ✅ ~90% within ±10%
- **$50k-$75k: Near perfect** ✅
- **Status mix within brackets: ✅ Accurate**

## Key Files

### Analysis
- `scripts/analysis/compare_irs_vs_dotax.py` - Compare data sources
- `scripts/analysis/investigate_50k_75k_overrepresentation.py` - Root cause analysis

### Implementation
- `src/tax/validation/irs_soi_calibration.py` - New IRS SOI calibration module
- `src/tax/validation/agi_calibration.py` - AGI-only calibration (previous)
- `src/tax/units/status/bracket_calibration.py` - Filing status calibration (existing)

### Documentation
- `CALIBRATION_OPTIONS.md` - All calibration options analyzed
- `50K_75K_INVESTIGATION_SUMMARY.md` - Problem investigation
- `IRS_SOI_INTEGRATION.md` - This document

## Next Steps

1. ✅ Understand IRS vs DOTAX differences
2. ✅ Create IRS SOI calibration module
3. ⏳ Test IRS SOI calibration on current tax units
4. ⏳ Compare results to two-layer approach
5. ⏳ Validate against both IRS and DOTAX benchmarks
6. ⏳ Choose final calibration approach
7. ⏳ Update `regenerate_tax_units.py` with chosen method

## Decision Criteria

Choose **IRS SOI Calibration** if:
- ✅ You want the most accurate filing status distribution within income brackets
- ✅ You're okay with ~5% difference from DOTAX totals
- ✅ You trust IRS data as much as DOTAX

Choose **Two-Layer (AGI + Filing Status)** if:
- ✅ You want to match DOTAX totals exactly
- ✅ You can accept less precision on filing status × AGI cells
- ✅ You prefer Hawaii-specific DOTAX data

Choose **Hybrid** if:
- ✅ You want best of both worlds
- ✅ You're comfortable with complexity
- ✅ You have time to iterate and validate

## Recommendation

**Start with IRS SOI Calibration** because:
1. It directly addresses the $50k-$75k problem's root cause
2. It's based on actual filing patterns, not assumptions
3. It's more sophisticated than previous approaches
4. You can always fall back to two-layer if results aren't satisfactory

Test both and compare results - let the data decide!
