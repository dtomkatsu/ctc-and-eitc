# Revised Ensemble Weight Strategy - Using FY 2024 Actuals

## Critical Issue: FY 2025 is a Projection, Not Actual

### What We Know for Certain

| Fiscal Year | Revenue | Status | Notes |
|-------------|---------|--------|-------|
| **FY 2022** | $3,760M | ✅ ACTUAL | Peak collections (COVID recovery + capital gains) |
| **FY 2023** | $3,100M | ✅ ACTUAL | Includes -$311.7M constitutional refund |
| **FY 2024** | $3,280M | ✅ ACTUAL | **Most recent confirmed actual** |
| **FY 2025** | $3,288M | ⚠️ PROJECTION | Listed as "Actual" but is DOT estimate |
| **FY 2026** | $2,721M | ⚠️ PROJECTION | Post-Act 46 projection |

### The Problem with Using FY 2025

**FY 2025 ($3,288M) is DOT's own projection**, which means:
1. It's based on DOT's growth assumptions (which may differ from ours)
2. It's not yet validated by actual collections
3. Using it as a "target" creates circular reasoning (calibrating to another projection)
4. We'd be anchoring to someone else's model, not reality

## Recommended Approach: Anchor to FY 2024 Actuals

### Strategy 1: Conservative (Use FY 2024 as Base)

**Rationale**: FY 2024 is the last confirmed actual data point

```
Base Year: FY 2024 = $3,280M (total)
Resident portion: $3,280M × 0.912 = $2,991M

For 2026 projection (2 years forward):
- Conservative (0% growth): $2,991M
- Baseline (1% growth): $3,051M  
- Moderate (2% growth): $3,111M
- Optimistic (3% growth): $3,172M
```

**Pros**:
- Based on actual data, not projections
- Conservative approach reduces risk
- Clear baseline everyone can verify

**Cons**:
- Ignores FY 2025 estimate (may be too conservative)
- 2-year projection window instead of 1-year

### Strategy 2: Blend FY 2024 Actual + FY 2025 Estimate

**Rationale**: Use FY 2024 as anchor but incorporate FY 2025 trend

```
FY 2024 actual: $3,280M
FY 2025 estimate: $3,288M
Implied growth: +0.24% (very flat)

Resident FY 2024: $2,991M
Apply 0.24% growth: $2,998M (FY 2025 resident)
Apply 1-2% growth for 2026: $3,028M - $3,058M
```

**Pros**:
- Incorporates latest DOT thinking
- More moderate than ignoring FY 2025 entirely
- Still anchored to actual FY 2024

**Cons**:
- Partially reliant on DOT projection
- FY 2025 may be revised

### Strategy 3: Use FY 2022-2024 Trend (Post-Peak Normalization)

**Rationale**: Calculate growth rate from actual post-peak data

```
FY 2022 (peak): $3,760M
FY 2023 (adjusted): $3,412M (+$311.7M refund)
FY 2024 (actual): $3,280M

Post-peak trend:
- FY 2022 → FY 2023 (adj): -9.3%
- FY 2023 (adj) → FY 2024: -3.9%
- Average decline: -6.6% per year

This is normalization from COVID peak, not sustainable decline.

More realistic: Use FY 2023-2024 only
- FY 2023 (adj) → FY 2024: -3.9%
- Assume normalization complete, apply +1-2% growth
```

**Pros**:
- Based entirely on actual data
- Captures post-peak normalization
- No reliance on projections

**Cons**:
- Small sample size (2 years)
- Constitutional refund adjustment adds uncertainty

## Recommended Ensemble Weights (Revised)

### Option A: Conservative (FY 2024 Anchor)

```python
ensemble_weights = {
    'fy_actual_2022_2024': 0.40,  # Use actual post-peak trend
    'dotax_2018_2021': 0.15,      # Reduce pre-peak weight
    'bls_wage': 0.25,              # Keep wage data
    'acs_income': 0.10,            # Reduce optimistic ACS
    'demographics': 0.10           # Keep structural
}

# Expected result: ~1-2% CAGR
```

**Rationale**:
- Heavy weight on actual FY data (40%)
- Reduced weight on pre-peak DOTAX (15%)
- Moderate growth assumption

### Option B: Moderate (Blend FY 2024 + FY 2025 Estimate)

```python
ensemble_weights = {
    'fy_actual_2022_2024': 0.30,  # Actual data
    'fy_2025_estimate': 0.10,      # DOT FY 2025 projection
    'dotax_2018_2021': 0.20,       # Pre-peak historical
    'bls_wage': 0.25,              # Wage growth
    'acs_income': 0.10,            # Income trends
    'demographics': 0.05           # Demographics
}

# Expected result: ~2-3% CAGR
```

**Rationale**:
- Balanced approach
- Incorporates FY 2025 but doesn't over-rely on it
- More weight on actual data (40% total)

### Option C: Aggressive (Trust DOT FY 2025)

```python
ensemble_weights = {
    'fy_2025_estimate': 0.30,     # Trust DOT projection
    'fy_actual_2022_2024': 0.20,  # Recent actuals
    'dotax_2018_2021': 0.20,      # Historical
    'bls_wage': 0.20,              # Wage growth
    'acs_income': 0.05,            # Income trends
    'demographics': 0.05           # Demographics
}

# Expected result: ~2.5-3.5% CAGR
```

**Rationale**:
- Assumes DOT FY 2025 estimate is accurate
- Higher growth assumption
- More optimistic

## Recommended Target Revenue (2026 Residents)

### Based on Different Strategies

| Strategy | Base | Growth | 2026 Resident Target | 2026 Total Target |
|----------|------|--------|---------------------|-------------------|
| **Conservative (FY 2024)** | $2,991M | 1.5% | $3,036M | $3,329M |
| **Moderate (Blend)** | $2,998M | 2.0% | $3,058M | $3,353M |
| **Aggressive (FY 2025)** | $2,999M | 2.5% | $3,074M | $3,370M |

### Our Current Model

- **Current**: $3,298M residents (7.4% CAGR from CY 2022)
- **Overestimate**: +$222M to +$262M depending on strategy

## How to Handle FY Recent Data in Ensemble

### Recommended Implementation

```python
class FiscalYearComponent:
    """Fiscal year revenue component for ensemble."""
    
    def __init__(self, use_fy2025_estimate=False):
        """
        Initialize FY component.
        
        Args:
            use_fy2025_estimate: If True, include FY 2025 projection.
                                If False, use only confirmed actuals (FY 2024).
        """
        self.use_fy2025 = use_fy2025_estimate
        
        # Confirmed actuals
        self.fy_2022 = 3760  # Peak
        self.fy_2023_raw = 3100
        self.fy_2023_adjusted = 3412  # Add back $311.7M refund
        self.fy_2024 = 3280  # Most recent actual
        
        # Projection (use with caution)
        self.fy_2025_estimate = 3288
        
    def calculate_growth_rate(self):
        """Calculate growth rate from fiscal year data."""
        
        if self.use_fy2025:
            # Use FY 2024 → FY 2025 (estimate)
            # Very flat growth: +0.24%
            growth = (self.fy_2025_estimate / self.fy_2024) - 1
            years = 1
            note = "Using FY 2025 estimate (caution: projection)"
        else:
            # Use FY 2023 (adj) → FY 2024 (actual)
            # Post-peak normalization: -3.9%
            growth = (self.fy_2024 / self.fy_2023_adjusted) - 1
            years = 1
            note = "Using actual FY data only"
        
        return {
            'growth_rate': growth,
            'years': years,
            'note': note,
            'base_year': 2025 if self.use_fy2025 else 2024
        }
    
    def project_to_2026(self, forward_growth_assumption=0.02):
        """
        Project to 2026 from most recent data.
        
        Args:
            forward_growth_assumption: Growth rate to apply forward (default 2%)
        """
        if self.use_fy2025:
            base = self.fy_2025_estimate
            years_forward = 1
        else:
            base = self.fy_2024
            years_forward = 2
        
        # Convert to resident
        base_resident = base * 0.912
        
        # Project forward
        projected_resident = base_resident * (1 + forward_growth_assumption) ** years_forward
        projected_total = projected_resident / 0.912
        
        return {
            'base_year': 2025 if self.use_fy2025 else 2024,
            'base_total': base,
            'base_resident': base_resident,
            'years_forward': years_forward,
            'growth_assumption': forward_growth_assumption,
            'projected_resident_2026': projected_resident,
            'projected_total_2026': projected_total
        }
```

### Usage Example

```python
# Conservative approach (FY 2024 actuals only)
fy_component_conservative = FiscalYearComponent(use_fy2025_estimate=False)
projection_conservative = fy_component_conservative.project_to_2026(
    forward_growth_assumption=0.015  # 1.5% growth
)
# Result: $3,036M residents

# Moderate approach (include FY 2025 estimate)
fy_component_moderate = FiscalYearComponent(use_fy2025_estimate=True)
projection_moderate = fy_component_moderate.project_to_2026(
    forward_growth_assumption=0.020  # 2% growth
)
# Result: $3,058M residents

# For ensemble, weight these appropriately
ensemble_projection = (
    0.40 * projection_conservative['projected_resident_2026'] +
    0.20 * dotax_projection +
    0.25 * bls_projection +
    0.10 * acs_projection +
    0.05 * demographic_projection
)
```

## Validation Against DOT Projections

### DOT's Implicit Assumptions

If we reverse-engineer DOT's projections:

```
FY 2024 actual: $3,280M
FY 2025 estimate: $3,288M
Growth: +0.24% (essentially flat)

FY 2026 pre-policy: $3,288M (same as FY 2025)
Growth: 0.0% (completely flat)
```

**DOT is assuming ZERO growth from FY 2025 to FY 2026**

This suggests:
1. DOT expects revenue to plateau
2. Post-peak normalization is complete
3. No significant economic growth expected
4. Very conservative assumption

### Our Model Should Be

**More optimistic than DOT** (1-2% growth) because:
- Economic recovery continues
- Wage growth is positive
- Demographics support growth
- But not as optimistic as 7.4%!

## Final Recommendation

### Use Strategy: **Moderate (Option B)**

**Ensemble Weights**:
```python
{
    'fy_actual_2022_2024': 0.30,   # Confirmed actuals
    'fy_2025_estimate': 0.10,       # DOT estimate (use cautiously)
    'dotax_2018_2021': 0.20,        # Historical growth
    'bls_wage': 0.25,               # Wage trends
    'acs_income': 0.10,             # Income trends
    'demographics': 0.05            # Structural
}
```

**Target 2026 Resident Revenue**: **$3,058M** (±5%)

**Rationale**:
1. ✅ Anchored primarily to FY 2024 actual ($3,280M)
2. ✅ Incorporates FY 2025 estimate but doesn't over-rely (10% weight)
3. ✅ Assumes modest 2% growth (between DOT's 0% and our old 7.4%)
4. ✅ Balanced between conservative and optimistic
5. ✅ Defensible to stakeholders

**Income Adjustment**: Scale by **0.927** (-7.3%) to hit target

**Validation**:
- Conservative scenario: $3,036M (within range)
- Moderate scenario: $3,058M (target)
- Aggressive scenario: $3,074M (within range)

All scenarios are within ±5% of each other, providing robustness.

---

**Key Insight**: By recognizing FY 2025 as a projection, we avoid circular reasoning and anchor to actual FY 2024 data while still incorporating the latest DOT thinking with appropriate caution.

**Document Version**: 2.0  
**Date**: November 3, 2025  
**Status**: 🟢 **RECOMMENDED APPROACH**
