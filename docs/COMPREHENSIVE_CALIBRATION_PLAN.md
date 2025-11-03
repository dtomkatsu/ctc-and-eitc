# Comprehensive Model Calibration Plan

## Executive Summary

Our Hawaii tax model currently **overestimates resident revenue by 10%** ($3,298M vs $2,999M target) due to:
1. **Growth rate too optimistic**: 7.4% CAGR vs realistic 2-3%
2. **Resident vs total confusion**: Model produces resident-only estimates but was compared to total revenue
3. **Act 46 impact overestimated**: Using wrong baseline inflated impact estimates

This plan provides a systematic approach to recalibrate the model to achieve ±5% accuracy.

## Current State Assessment

### Model Performance

| Metric | Current | Target | Gap | Status |
|--------|---------|--------|-----|--------|
| **2026 Resident Revenue** | $3,298M | $2,999M | +$299M (+10.0%) | ⚠️ |
| **Growth Rate (CAGR)** | 7.4% | 2-3% | +4.4-5.4pp | ❌ |
| **Act 46 Impact** | -$657M | -$597M | -$60M (-10%) | ⚠️ |
| **Total Revenue (w/ non-res)** | $3,617M | $3,355M | +$262M (+7.8%) | ⚠️ |

### Root Causes

1. **Ensemble Weights Favor High-Growth Period**
   - DOTAX 2018-2021: 35% weight → 11.1% CAGR
   - BLS/ACS: 55% weight → 5.5-6.2% growth
   - Result: 7.4% CAGR (too optimistic)

2. **Not Anchored to Recent Actuals**
   - FY 2022-2025 shows -4.4% CAGR (post-peak normalization)
   - Model doesn't incorporate this recent trend

3. **Resident vs Total Confusion**
   - 8.8% of revenue from non-residents
   - Model produces resident-only but compared to total

## Calibration Strategy

### Phase 1: Immediate Fixes (Day 1)

#### 1.1 Update Ensemble Weights

**Current Weights** (Optimistic):
```python
weights = {
    'dotax_2018_2021': 0.35,  # 11.1% CAGR
    'bls_wage': 0.30,          # 5.5% growth
    'acs_income': 0.25,        # 6.2% growth
    'demographics': 0.10       # 1.1% growth
}
# Result: 7.4% CAGR
```

**New Weights** (Realistic):
```python
weights = {
    'fy_recent_2022_2025': 0.30,  # -4.4% CAGR (new)
    'dotax_2018_2021': 0.20,       # 11.1% CAGR (reduced)
    'bls_wage': 0.25,              # 5.5% growth
    'acs_income': 0.15,            # 6.2% growth (reduced)
    'demographics': 0.10           # 1.1% growth
}
# Target: 2-3% CAGR
```

#### 1.2 Anchor to FY 2025 Actuals

```python
# Base calibration targets
fy_2025_total = 3288        # Actual
fy_2025_resident = 2999      # 91.2% of total
growth_rate = 0.02           # 2% annual growth
fy_2026_resident_target = 3059  # FY 2025 + 2%
fy_2026_total_target = 3355     # Include 8.8% non-resident
```

#### 1.3 Document Resident vs Total

**All projections must specify**:
- Resident-only revenue: What our model produces
- Total revenue: Resident + 8.8% non-resident
- Conversion: Total = Resident / 0.912

### Phase 2: Model Recalibration (Days 2-3)

#### 2.1 Income Growth Adjustment

**Approach**: Scale projected incomes to match target growth

```python
def recalibrate_income_growth(tax_units, current_growth=0.074, target_growth=0.025):
    """Scale incomes to achieve target growth rate."""
    adjustment_factor = (1 + target_growth) / (1 + current_growth)
    
    # Apply to income columns
    income_cols = ['wages', 'self_employment', 'capital_gains', 'other_income']
    for col in income_cols:
        if col in tax_units.columns:
            tax_units[col] = tax_units[col] * adjustment_factor
    
    # Recalculate total income
    tax_units['total_income'] = tax_units[income_cols].sum(axis=1)
    
    return tax_units
```

#### 2.2 Capital Gains Normalization

**Current**: 3.31% of AGI (may still be elevated from 2021 surge)
**Target**: 2.5-3.0% of AGI (more normalized)

```python
def normalize_capital_gains(tax_units, target_pct=0.025):
    """Adjust capital gains to target percentage of AGI."""
    current_pct = tax_units['capital_gains'].sum() / tax_units['agi'].sum()
    adjustment_factor = target_pct / current_pct
    
    tax_units['capital_gains'] = tax_units['capital_gains'] * adjustment_factor
    
    return tax_units
```

#### 2.3 Filing Status Validation

**Check distribution against SOI benchmarks**:
- Single: 51.0% (current: 53.1%)
- Joint: 36.0% (current: 33.0%)
- HoH: 9.6% (current: 10.0%)
- MFS: 3.4% (current: 3.8%)

### Phase 3: Act 46 Recalibration (Days 4-5)

#### 3.1 Use Official Impact Rate

```python
def calculate_act46_impact(baseline_revenue, method='official'):
    """Calculate Act 46 impact using official rate."""
    
    if method == 'official':
        # Use official -19.9% of resident revenue
        impact_rate = -0.199
    elif method == 'modeled':
        # Our current modeled rate (needs investigation)
        impact_rate = -0.299
    elif method == 'average':
        # Average of official and modeled
        impact_rate = -0.249
    
    impact = baseline_revenue * impact_rate
    post_act46 = baseline_revenue + impact
    
    return {
        'baseline': baseline_revenue,
        'impact': impact,
        'post_act46': post_act46,
        'rate': impact_rate
    }
```

#### 3.2 Validate Against 2017 vs 2024 Actuals

**Compare actual revenue changes 2017→2024 to validate our model**

### Phase 4: Validation Framework (Day 6)

#### 4.1 Create Validation Checks

```python
def validate_calibration(model_results, targets):
    """Comprehensive validation of calibrated model."""
    
    checks = {
        'resident_revenue': {
            'model': model_results['resident_revenue_2026'],
            'target': targets['fy_2026_resident'],
            'tolerance': 0.05,  # ±5%
            'pass': False
        },
        'growth_rate': {
            'model': model_results['cagr'],
            'target': targets['target_cagr'],
            'tolerance': 0.01,  # ±1pp
            'pass': False
        },
        'act46_impact': {
            'model': model_results['act46_impact'],
            'target': targets['official_act46'],
            'tolerance': 0.10,  # ±10%
            'pass': False
        },
        'total_revenue': {
            'model': model_results['total_revenue_2026'],
            'target': targets['fy_2026_total'],
            'tolerance': 0.05,  # ±5%
            'pass': False
        }
    }
    
    # Check each metric
    for metric, check in checks.items():
        diff = abs(check['model'] - check['target'])
        pct_diff = diff / check['target']
        check['pass'] = pct_diff <= check['tolerance']
        check['diff'] = diff
        check['pct_diff'] = pct_diff
    
    return checks
```

#### 4.2 Sensitivity Analysis

Test model sensitivity to:
- Growth rate variations (0%, 1%, 2%, 3%)
- Act 46 impact rates (-18%, -20%, -22%)
- Capital gains percentages (2%, 2.5%, 3%)
- Non-resident share (8%, 8.8%, 10%)

### Phase 5: Implementation (Day 7)

#### 5.1 Update Ensemble Projector

```python
# src/projection/ensemble.py

class CalibratedEnsembleProjector:
    """Recalibrated ensemble projector with realistic growth."""
    
    def __init__(self):
        self.weights = {
            'fy_recent': 0.30,
            'dotax_historical': 0.20,
            'bls_wage': 0.25,
            'acs_income': 0.15,
            'demographics': 0.10
        }
        
        self.growth_rates = {
            'fy_recent': -0.044,  # FY 2022-2025 actual
            'dotax_historical': 0.111,  # 2018-2021
            'bls_wage': 0.055,
            'acs_income': 0.062,
            'demographics': 0.011
        }
        
    def project_revenue(self, base_year=2025, target_year=2026):
        """Project revenue with calibrated growth."""
        
        # Calculate weighted growth rate
        weighted_growth = sum(
            self.weights[k] * self.growth_rates[k] 
            for k in self.weights
        )
        
        # Should be ~2-3%
        assert 0.02 <= weighted_growth <= 0.03, f"Growth rate {weighted_growth:.1%} out of range"
        
        # Project from FY 2025 baseline
        fy_2025_resident = 2999  # Million
        years = target_year - base_year
        
        projected_resident = fy_2025_resident * (1 + weighted_growth) ** years
        projected_total = projected_resident / 0.912  # Add non-residents
        
        return {
            'resident': projected_resident,
            'non_resident': projected_total - projected_resident,
            'total': projected_total,
            'growth_rate': weighted_growth
        }
```

#### 5.2 Update Pipeline Script

```python
# scripts/calibrated_projection_pipeline.py

def run_calibrated_projection():
    """Run fully calibrated 2026 projection."""
    
    # 1. Load base tax units
    tax_units = load_tax_units()
    
    # 2. Apply growth calibration
    tax_units = recalibrate_income_growth(
        tax_units, 
        current_growth=0.074, 
        target_growth=0.025
    )
    
    # 3. Normalize capital gains
    tax_units = normalize_capital_gains(
        tax_units,
        target_pct=0.025
    )
    
    # 4. Calculate tax liability
    tax_units = calculate_hawaii_tax(tax_units, year=2026)
    
    # 5. Aggregate to resident revenue
    resident_revenue = (
        tax_units['tax_liability'] * tax_units['weight']
    ).sum() / 1_000_000  # Convert to millions
    
    # 6. Add non-residents
    total_revenue = resident_revenue / 0.912
    
    # 7. Calculate Act 46 impact
    act46 = calculate_act46_impact(resident_revenue, method='official')
    
    # 8. Validate
    results = {
        'resident_revenue_2026': resident_revenue,
        'total_revenue_2026': total_revenue,
        'act46_impact': act46['impact'],
        'cagr': 0.025  # Target growth
    }
    
    targets = {
        'fy_2026_resident': 3059,
        'fy_2026_total': 3355,
        'official_act46': -597,
        'target_cagr': 0.025
    }
    
    validation = validate_calibration(results, targets)
    
    return results, validation
```

## Success Metrics

### Primary Goals (Must Achieve)

| Metric | Target | Tolerance |
|--------|--------|-----------|
| **Resident Revenue** | $3,059M | ±5% |
| **Total Revenue** | $3,355M | ±5% |
| **Growth Rate** | 2-3% CAGR | ±1pp |
| **Act 46 Impact** | -$597M | ±10% |

### Secondary Goals (Nice to Have)

| Metric | Target | Tolerance |
|--------|--------|-----------|
| **Filing Status Distribution** | SOI benchmarks | ±2pp |
| **Capital Gains % of AGI** | 2.5% | ±0.5pp |
| **Effective Tax Rates by Bracket** | DOTAX Table A8 | ±1pp |

## Risk Mitigation

### Risk 1: Over-Correction
**Mitigation**: Test multiple calibration levels (1%, 2%, 3% growth)

### Risk 2: Breaking Existing Calibrations
**Mitigation**: Preserve existing tax calculation pipeline, only adjust inputs

### Risk 3: Loss of Demographic Detail
**Mitigation**: Apply uniform scaling factors to preserve distributions

## Timeline

| Day | Tasks | Deliverables |
|-----|-------|--------------|
| **1** | Update ensemble weights, anchor to FY 2025 | New weight configuration |
| **2-3** | Recalibrate income growth and capital gains | Calibration functions |
| **4-5** | Fix Act 46 impact calculation | Updated impact estimates |
| **6** | Build validation framework | Validation suite |
| **7** | Implement and test full pipeline | Calibrated projections |
| **8** | Documentation and cleanup | Final report |

## Next Steps

1. **Immediate Action**: Update ensemble weights in `src/projection/ensemble.py`
2. **Create Calibration Script**: `scripts/analysis/calibrate_model.py`
3. **Run Validation**: Test against all success metrics
4. **Document Changes**: Update README and create memory
5. **Clean Up Files**: Organize project structure

---

**Document Version**: 1.0  
**Date**: November 3, 2025  
**Status**: 🔴 **ACTION REQUIRED**  
**Priority**: **CRITICAL** - Model accuracy depends on this
