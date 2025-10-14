# Income Growth and Inflation Adjustment Methodology

## Overview

This document explains how we adjust income from base years (2022 for nonresidents, 2023 for residents) to the target year (2026) for tax policy modeling.

---

## Current Implementation

### 1. Resident Income Growth (2023 → 2026)

**Data Source:** Hawaii PUMS 2023

**Adjustment Factors:**
- **Base Year:** 2023 (PUMS survey year)
- **Target Year:** 2026 (policy analysis year)
- **Time Period:** 3 years
- **Nominal Wage Growth:** 15.0% total (4.77% annualized)
- **Inflation (CPI):** 8.84% total (2.86% annualized)
- **Real Growth Factor:** 1.056 (5.6% real growth)

**Calculation:**
```
Real Growth = Nominal Growth / Inflation
1.056 = 1.15 / 1.0884
```

**Application:**
```python
# For a resident with $75,000 income in 2023 PUMS data
income_2026 = 75000 * 1.056 = $79,200 (in 2026 real terms)
```

### 2. Nonresident Income Growth (2022 → 2026)

**Data Source:** DOTAX SOI Table 17A (2022)

**Adjustment Factors:**
- **Base Year:** 2022 (DOTAX data year)
- **Target Year:** 2026 (policy analysis year)
- **Time Period:** 4 years
- **Nominal Wage Growth:** 12.48% total (2.99% annualized)
- **Inflation (CPI):** 6.81% total (1.66% annualized)
- **Real Growth Factor:** 1.053 (5.3% real growth)

**Calculation:**
```
Real Growth = Nominal Growth / Inflation
1.053 = 1.1248 / 1.0681
```

**Application:**
```python
# For a nonresident with $100,000 AGI in 2022 DOTAX data
agi_2026 = 100000 * 1.053 = $105,300 (in 2026 real terms)
```

---

## Implementation Details

### File Structure

```
src/
├── config/
│   ├── __init__.py
│   └── income_growth.py          # Growth factor definitions
└── tax/
    └── units/
        ├── income.py              # Income calculation with growth
        └── nonresident_synthesizer.py  # Nonresident AGI projection
```

### Key Functions

**1. `apply_income_growth(income, is_resident)`**
- Applies appropriate growth factor based on residency
- Used throughout the codebase for consistent adjustments

**2. `calculate_person_income(person, apply_2026_growth=True, is_resident=True)`**
- Calculates person-level income from PUMS data
- Optionally applies 2026 growth projection
- Handles ADJINC (PUMS inflation adjustment) + 2026 growth

**3. `NonresidentSynthesizer._generate_agi_values()`**
- Generates synthetic nonresident AGI from 2022 brackets
- Automatically applies 1.053 growth factor to project to 2026

---

## Data Flow

### Resident Income (PUMS)

```
Raw PUMS Income (2023 nominal dollars)
    ↓
× ADJINC (PUMS adjustment to 2023 dollars)
    ↓
2023 Income (PUMS-adjusted)
    ↓
× 1.056 (Real growth 2023→2026)
    ↓
2026 Income (projected real terms)
```

### Nonresident Income (DOTAX)

```
DOTAX AGI Brackets (2022 dollars)
    ↓
Sample AGI within bracket
    ↓
× 1.053 (Real growth 2022→2026)
    ↓
2026 AGI (projected real terms)
```

---

## Proposed Improvements

### 1. **Income-Type-Specific Growth Rates**

**Current Limitation:** We apply the same growth rate to all income types (wages, self-employment, investment, retirement).

**Proposed Enhancement:**
```python
INCOME_TYPE_GROWTH = {
    'wages': {
        'resident': 1.056,      # Hawaii wage growth
        'nonresident': 1.053    # National wage growth
    },
    'self_employment': {
        'resident': 1.045,      # Lower growth for self-employment
        'nonresident': 1.040
    },
    'investment': {
        'resident': 1.070,      # Higher growth for investment income
        'nonresident': 1.070    # Same for both (national markets)
    },
    'retirement': {
        'resident': 1.025,      # COLA adjustments only
        'nonresident': 1.025
    }
}
```

**Rationale:**
- **Wages:** Tied to local/national labor markets
- **Self-Employment:** More volatile, historically lower growth
- **Investment Income:** Tied to stock/bond markets (higher recent growth)
- **Retirement:** Tied to COLA adjustments (lower growth)

**Implementation:**
```python
def calculate_person_income_enhanced(person, apply_2026_growth=True, is_resident=True):
    income_components = {
        'wages': float(person.get('WAGP', 0) or 0),
        'self_employment': float(person.get('SEMP', 0) or 0),
        'investment': float(person.get('INTP', 0) or 0) + float(person.get('DIV', 0) or 0),
        'retirement': float(person.get('RETP', 0) or 0) + float(person.get('SSP', 0) or 0) * 0.85,
        'other': float(person.get('OIP', 0) or 0)
    }
    
    total_income = 0.0
    for income_type, amount in income_components.items():
        if apply_2026_growth:
            growth_factor = INCOME_TYPE_GROWTH.get(income_type, {}).get(
                'resident' if is_resident else 'nonresident', 1.0
            )
            total_income += amount * growth_factor
        else:
            total_income += amount
    
    return total_income * float(person.get('ADJINC', 1.0) or 1.0)
```

### 2. **Income-Bracket-Specific Growth Rates**

**Current Limitation:** We apply the same growth rate regardless of income level.

**Proposed Enhancement:**
```python
BRACKET_GROWTH_MULTIPLIERS = {
    # Income bracket: multiplier to base growth rate
    (0, 25000): 0.95,           # Lower growth for low-income
    (25000, 75000): 1.00,       # Base growth for middle-income
    (75000, 200000): 1.05,      # Slightly higher for upper-middle
    (200000, float('inf')): 1.10  # Higher growth for high-income
}
```

**Rationale:**
- High-income earners historically see faster wage growth
- Investment income (concentrated among high earners) has grown faster
- Low-income earners face wage stagnation

**Implementation:**
```python
def get_bracket_multiplier(income):
    for (min_income, max_income), multiplier in BRACKET_GROWTH_MULTIPLIERS.items():
        if min_income <= income < max_income:
            return multiplier
    return 1.0

def apply_income_growth_enhanced(income, is_resident):
    base_growth = RESIDENT_GROWTH if is_resident else NONRESIDENT_GROWTH
    bracket_multiplier = get_bracket_multiplier(income)
    return income * base_growth.real_growth * bracket_multiplier
```

### 3. **Geographic Variation (for Residents)**

**Current Limitation:** We apply the same growth rate to all Hawaii residents.

**Proposed Enhancement:**
```python
PUMA_GROWTH_ADJUSTMENTS = {
    # Honolulu County (Oahu)
    '00100': 1.00,  # Urban Honolulu - base growth
    '00200': 0.98,  # Rural Oahu - slightly lower
    
    # Hawaii County (Big Island)
    '00300': 0.95,  # Hilo area - lower growth
    '00400': 0.93,  # Kona area - tourism-dependent
    
    # Maui County
    '00500': 0.94,  # Maui - tourism-dependent
    '00600': 0.92,  # Molokai/Lanai - lowest growth
    
    # Kauai County
    '00700': 0.95,  # Kauai - tourism-dependent
}
```

**Rationale:**
- Urban Honolulu has stronger wage growth (government, military, services)
- Neighbor islands more dependent on tourism (more volatile)
- Rural areas have lower wage growth

### 4. **Uncertainty Bounds**

**Current Limitation:** We use point estimates without uncertainty.

**Proposed Enhancement:**
```python
@dataclass
class GrowthFactorsWithUncertainty:
    base_year: int
    target_year: int
    nominal_growth: float
    nominal_growth_lower: float  # 95% CI lower bound
    nominal_growth_upper: float  # 95% CI upper bound
    inflation: float
    inflation_lower: float
    inflation_upper: float
    
    def get_real_growth_bounds(self):
        # Best case: high nominal growth, low inflation
        upper = self.nominal_growth_upper / self.inflation_lower
        # Worst case: low nominal growth, high inflation
        lower = self.nominal_growth_lower / self.inflation_upper
        # Central estimate
        central = self.nominal_growth / self.inflation
        return (lower, central, upper)

RESIDENT_GROWTH_UNCERTAIN = GrowthFactorsWithUncertainty(
    base_year=2023,
    target_year=2026,
    nominal_growth=1.15,
    nominal_growth_lower=1.10,   # Conservative estimate
    nominal_growth_upper=1.20,   # Optimistic estimate
    inflation=1.0884,
    inflation_lower=1.06,        # Low inflation scenario
    inflation_upper=1.12,        # High inflation scenario
)

# Example: (0.982, 1.056, 1.132)
# Real growth could range from -1.8% to +13.2%, central estimate +5.6%
```

**Use Case:**
- Sensitivity analysis
- Monte Carlo simulations
- Policy scenario modeling

### 5. **Historical Validation**

**Proposed Addition:** Validate growth assumptions against historical data.

```python
def validate_against_historical_data():
    """
    Compare projected growth rates to historical Hawaii wage growth.
    
    Data sources:
    - BLS State and Area Employment, Hours, and Earnings (SAE)
    - Hawaii DBEDT Economic Data
    - Census Bureau ACS historical data
    """
    historical_growth = {
        '2018-2019': 0.032,  # 3.2% nominal wage growth
        '2019-2020': 0.015,  # 1.5% (COVID impact)
        '2020-2021': 0.045,  # 4.5% (recovery)
        '2021-2022': 0.058,  # 5.8% (inflation spike)
        '2022-2023': 0.042,  # 4.2% (normalization)
    }
    
    avg_historical = sum(historical_growth.values()) / len(historical_growth)
    # avg_historical ≈ 3.84% per year
    
    projected_annual = RESIDENT_GROWTH.annual_nominal_rate
    # projected_annual ≈ 4.77% per year
    
    if abs(projected_annual - avg_historical) > 0.02:  # More than 2pp difference
        logger.warning(
            f"Projected growth ({projected_annual:.2%}) differs significantly "
            f"from historical average ({avg_historical:.2%})"
        )
```

---

## Recommended Implementation Priority

### Phase 1: Immediate (Current Implementation)
✅ **Completed:**
- Single growth factor per residency type
- Applied uniformly to all income
- Simple, transparent, defensible

### Phase 2: Near-term Enhancement
🔄 **Recommended:**
1. **Income-type-specific growth rates** (Improvement #1)
   - Most impactful improvement
   - Relatively easy to implement
   - Better reflects economic reality

2. **Historical validation** (Improvement #5)
   - Validates our assumptions
   - Builds credibility
   - Easy to add as documentation

### Phase 3: Medium-term Enhancement
📋 **Consider:**
3. **Income-bracket-specific growth** (Improvement #2)
   - More complex but more accurate
   - Important for distributional analysis
   - Requires careful calibration

4. **Uncertainty bounds** (Improvement #4)
   - Essential for sensitivity analysis
   - Useful for policy scenario modeling
   - Requires statistical expertise

### Phase 4: Long-term Enhancement
🔮 **Future:**
5. **Geographic variation** (Improvement #3)
   - Most complex implementation
   - Marginal benefit for state-level analysis
   - Consider only if doing county-level analysis

---

## Data Sources for Improvements

### Wage Growth Data
- **Hawaii:** [Hawaii DBEDT Economic Data](https://dbedt.hawaii.gov/economic/)
- **National:** [BLS Employment Cost Index](https://www.bls.gov/eci/)
- **By Industry:** [BLS State and Area Employment](https://www.bls.gov/sae/)

### Inflation Data
- **National CPI:** [BLS Consumer Price Index](https://www.bls.gov/cpi/)
- **Hawaii CPI:** [BLS CPI for Urban Hawaii](https://www.bls.gov/regions/west/news-release/consumerpriceindex_honolulu.htm)

### Income Distribution
- **By Type:** [IRS Statistics of Income](https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics)
- **By Bracket:** [Census Bureau ACS](https://data.census.gov/)

---

## Testing and Validation

### Unit Tests
```python
def test_resident_growth_factor():
    """Test resident income growth calculation."""
    base_income = 75000
    projected = apply_income_growth(base_income, is_resident=True)
    expected = 75000 * 1.056
    assert abs(projected - expected) < 0.01

def test_nonresident_growth_factor():
    """Test nonresident income growth calculation."""
    base_income = 100000
    projected = apply_income_growth(base_income, is_resident=False)
    expected = 100000 * 1.053
    assert abs(projected - expected) < 0.01

def test_growth_factor_consistency():
    """Test that growth factors are consistent across modules."""
    from config.income_growth import RESIDENT_GROWTH, NONRESIDENT_GROWTH
    
    # Check that real growth = nominal / inflation
    assert abs(RESIDENT_GROWTH.real_growth - 
               RESIDENT_GROWTH.nominal_growth / RESIDENT_GROWTH.inflation) < 0.001
    assert abs(NONRESIDENT_GROWTH.real_growth - 
               NONRESIDENT_GROWTH.nominal_growth / NONRESIDENT_GROWTH.inflation) < 0.001
```

### Integration Tests
```python
def test_full_pipeline_with_growth():
    """Test that growth factors are applied throughout the pipeline."""
    # Load PUMS data
    loader = PUMSDataLoader()
    households = loader.load_households_batch(batch_size=100)
    
    # Build tax units with growth
    constructor = TaxUnitConstructor()
    tax_units = constructor.process_households(households)
    
    # Verify income is projected to 2026
    for unit in tax_units:
        base_income = calculate_person_income(unit, apply_2026_growth=False)
        projected_income = calculate_person_income(unit, apply_2026_growth=True)
        
        # Projected should be ~5.6% higher for residents
        growth_ratio = projected_income / base_income if base_income > 0 else 1.0
        assert 1.05 < growth_ratio < 1.07  # Allow small variance
```

---

## Summary

### Current Implementation
✅ **Strengths:**
- Simple and transparent
- Defensible methodology
- Consistent across codebase
- Well-documented

⚠️ **Limitations:**
- Single growth rate for all income types
- No geographic variation
- No uncertainty bounds
- Point estimates only

### Recommended Next Steps

1. **Immediate:** Use current implementation (already done)
2. **Near-term:** Add income-type-specific growth rates
3. **Medium-term:** Add uncertainty bounds for sensitivity analysis
4. **Long-term:** Consider bracket-specific and geographic adjustments

### Key Takeaway

The current implementation provides a **solid foundation** for 2026 projections. The proposed improvements would increase accuracy but add complexity. Prioritize based on:
- **Policy questions:** What level of detail do we need?
- **Data availability:** Can we get reliable data for enhancements?
- **Time constraints:** What can we implement before the analysis deadline?

For most policy analyses, **income-type-specific growth rates** (Improvement #1) would provide the best return on investment.
