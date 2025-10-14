# Hawaii DOTAX SOI 2022 Data Integration Proposal

## Data Summary

**Source:** Hawaii Department of Taxation Statistics of Income 2022 (Table 5A)  
**Coverage:** 635,117 resident tax returns for tax year 2022

### Key Metrics from SOI Data

| Filing Status | Returns | % | Net AGI ($M) | Avg AGI | Tax After Credits ($M) | Avg Tax | Effective Rate |
|--------------|---------|---|--------------|---------|----------------------|---------|----------------|
| Married Filing Jointly | 216,358 | 34.1% | $26,129 | $120,767 | $1,551 | $7,169 | 5.94% |
| Single | 335,198 | 52.8% | $14,047 | $41,907 | $818 | $2,440 | 5.82% |
| Married Filing Separately | 16,007 | 2.5% | $3,122 | $195,040 | $249 | $15,556 | 7.98% |
| Head of Household | 67,393 | 10.6% | $3,725 | $55,273 | $177 | $2,626 | 4.75% |
| **TOTAL** | **635,117** | **100%** | **$47,023** | **$74,042** | **$2,795** | **$4,401** | **5.94%** |

### Credit Impact
- **Total credits:** $234M (7.7% reduction in tax liability)
- **By filing status:**
  - Joint: $123M (7.3% reduction)
  - Single: $46M (5.3% reduction)  
  - MFS: $40M (13.8% reduction)
  - HoH: $25M (12.4% reduction)

## Current Model Performance vs SOI

| Metric | PUMS Model | SOI 2022 | Gap | Status |
|--------|-----------|----------|-----|--------|
| **Filing Status Distribution** |
| Single | 50.5% | 52.8% | -2.3pp | ⚠️ Close |
| Married Filing Jointly | 41.2% | 34.1% | +7.1pp | ❌ Too high |
| Married Filing Separately | 2.8% | 2.5% | +0.3pp | ✅ Excellent |
| Head of Household | 5.4% | 10.6% | -5.2pp | ❌ Too low |
| **Total Returns** | 1,047,413 | 635,117 | +65% | ⚠️ PUMS overweight |

### Key Issues Identified

1. **Joint filers over-identified by 7.1pp** (should be 34.1%, we have 41.2%)
2. **HoH under-identified by 5.2pp** (should be 10.6%, we have 5.4%)
3. **PUMS weights produce 65% more returns** than actual SOI count
4. **Credit modeling incomplete** - SOI shows 7.7% tax reduction from credits

## Proposed Integration Strategies

### 1. **Filing Status Calibration** (High Priority)

**Problem:** Joint is 7.1pp too high, HoH is 5.2pp too low

**Solution A: Post-Processing Calibration**
```python
def calibrate_filing_status(tax_units: pd.DataFrame, target_distribution: dict) -> pd.DataFrame:
    """
    Calibrate filing status distribution to match SOI targets.
    
    Target distribution (from SOI 2022):
    - Single: 52.8%
    - Married Filing Jointly: 34.1%
    - Married Filing Separately: 2.5%
    - Head of Household: 10.6%
    """
    # Calculate current weighted distribution
    current = tax_units.groupby('filing_status')['weight'].sum()
    total = current.sum()
    current_pct = current / total
    
    # Calculate adjustment factors
    adjustments = {}
    for status in target_distribution:
        target_pct = target_distribution[status] / 100
        current_pct_val = current_pct.get(status, 0)
        if current_pct_val > 0:
            adjustments[status] = target_pct / current_pct_val
    
    # Apply adjustments to weights
    tax_units['calibrated_weight'] = tax_units.apply(
        lambda row: row['weight'] * adjustments.get(row['filing_status'], 1.0),
        axis=1
    )
    
    return tax_units
```

**Solution B: Improve HoH Logic** (Preferred)
- Allow married adults without spouse in household to file as HoH if they have children
- This addresses the "considered unmarried" provision in IRS rules
- Could add ~567 HoH filers based on our analysis

**Recommendation:** Implement Solution B first (improve logic), then use Solution A for fine-tuning

### 2. **Weight Calibration** (High Priority)

**Problem:** PUMS produces 1,047,413 returns vs SOI's 635,117 (65% overcount)

**Solution: Scale PUMS Weights**
```python
def calibrate_total_returns(tax_units: pd.DataFrame, target_total: int = 635117) -> pd.DataFrame:
    """
    Scale weights so total returns match SOI count.
    """
    current_total = tax_units['weight'].sum()
    scaling_factor = target_total / current_total
    
    tax_units['calibrated_weight'] = tax_units['weight'] * scaling_factor
    
    return tax_units
```

**Calibration Factor:** 0.606 (635,117 / 1,047,413)

**Impact:** This will make all revenue estimates more accurate

### 3. **Average AGI Benchmarking** (Medium Priority)

**Problem:** Need to validate our income distributions against SOI

**Solution: AGI Distribution Validation**
```python
def validate_agi_distribution(tax_units: pd.DataFrame, soi_benchmarks: dict):
    """
    Compare PUMS AGI distribution to SOI benchmarks by filing status.
    
    SOI Average AGI (2022):
    - Joint: $120,767
    - Single: $41,907
    - MFS: $195,040
    - HoH: $55,273
    """
    for status, target_agi in soi_benchmarks.items():
        status_units = tax_units[tax_units['filing_status'] == status]
        if len(status_units) > 0:
            # Weighted average AGI
            pums_avg_agi = (status_units['income'] * status_units['weight']).sum() / status_units['weight'].sum()
            ratio = pums_avg_agi / target_agi
            
            print(f"{status}: PUMS ${pums_avg_agi:,.0f} vs SOI ${target_agi:,.0f} (ratio: {ratio:.2f})")
            
            # Flag if ratio is outside acceptable range (0.9-1.1)
            if ratio < 0.9 or ratio > 1.1:
                print(f"  ⚠️ WARNING: AGI mismatch for {status}")
```

### 4. **Tax Credit Modeling** (High Priority)

**Problem:** SOI shows 7.7% tax reduction from credits, our model may underestimate

**Solution: Enhanced Credit Module**
```python
def apply_soi_calibrated_credits(tax_units: pd.DataFrame) -> pd.DataFrame:
    """
    Apply credits calibrated to match SOI 2022 credit impact.
    
    Target credit rates (% reduction in tax):
    - Joint: 7.3%
    - Single: 5.3%
    - MFS: 13.8%
    - HoH: 12.4%
    """
    credit_rates = {
        'married_filing_jointly': 0.073,
        'single': 0.053,
        'married_filing_separately': 0.138,
        'head_of_household': 0.124
    }
    
    for status, rate in credit_rates.items():
        mask = tax_units['filing_status'] == status
        # Apply credit as percentage of tax before credits
        tax_units.loc[mask, 'credits'] = tax_units.loc[mask, 'tax_before_credits'] * rate
        tax_units.loc[mask, 'tax_after_credits'] = tax_units.loc[mask, 'tax_before_credits'] * (1 - rate)
    
    return tax_units
```

**Note:** This should be integrated with existing credit modules in `src/tax/adjustments/hawaii_credits.py`

### 5. **Effective Tax Rate Validation** (Medium Priority)

**Problem:** Need to ensure our tax calculations produce realistic effective rates

**SOI Effective Tax Rates (After Credits / Net AGI):**
- Joint: 5.94%
- Single: 5.82%
- MFS: 7.98%
- HoH: 4.75%
- **Overall: 5.94%**

**Solution: Tax Rate Validation**
```python
def validate_effective_tax_rates(tax_units: pd.DataFrame, soi_rates: dict):
    """
    Validate that effective tax rates match SOI benchmarks.
    """
    for status, target_rate in soi_rates.items():
        status_units = tax_units[tax_units['filing_status'] == status]
        if len(status_units) > 0:
            total_agi = (status_units['income'] * status_units['weight']).sum()
            total_tax = (status_units['tax_after_credits'] * status_units['weight']).sum()
            
            if total_agi > 0:
                pums_rate = (total_tax / total_agi) * 100
                print(f"{status}: PUMS {pums_rate:.2f}% vs SOI {target_rate:.2f}%")
                
                if abs(pums_rate - target_rate) > 1.0:
                    print(f"  ⚠️ WARNING: Tax rate mismatch for {status}")
```

## Implementation Priority

### Phase 1: Critical Fixes (Immediate)
1. ✅ **Weight calibration** - Scale to match 635,117 total returns
2. ✅ **Filing status calibration** - Adjust to match SOI distribution
3. ✅ **Credit impact calibration** - Ensure 7.7% overall reduction

### Phase 2: Logic Improvements (Short-term)
4. ⚠️ **Improve HoH identification** - Allow "considered unmarried" married adults
5. ⚠️ **Validate AGI distributions** - Ensure income levels match SOI by filing status
6. ⚠️ **Validate effective tax rates** - Ensure tax calculations are realistic

### Phase 3: Validation & Refinement (Medium-term)
7. 📊 **Create validation dashboard** - Track all metrics against SOI benchmarks
8. 📊 **Document discrepancies** - Identify and explain remaining gaps
9. 📊 **Sensitivity analysis** - Test impact of calibration choices

## Expected Impact

### Before Calibration
- Filing status: Joint +7.1pp, HoH -5.2pp
- Total returns: 1,047,413 (65% too high)
- Tax revenue: Likely overestimated

### After Calibration
- Filing status: Within ±1pp of SOI targets
- Total returns: 635,117 (exact match)
- Tax revenue: Accurate to within 5% of actual
- Credit impact: 7.7% reduction (matching SOI)

## Code Integration

### New Module: `src/tax/calibration/soi_calibration.py`
```python
"""
SOI-based calibration for Hawaii tax model.

Calibrates PUMS-based tax units to match Hawaii DOTAX SOI 2022 benchmarks.
"""

class SOICalibrator:
    """Calibrate tax units to SOI benchmarks."""
    
    # SOI 2022 Benchmarks
    SOI_FILING_STATUS = {
        'single': 52.8,
        'married_filing_jointly': 34.1,
        'married_filing_separately': 2.5,
        'head_of_household': 10.6
    }
    
    SOI_TOTAL_RETURNS = 635117
    
    SOI_AVG_AGI = {
        'married_filing_jointly': 120767,
        'single': 41907,
        'married_filing_separately': 195040,
        'head_of_household': 55273
    }
    
    SOI_CREDIT_RATES = {
        'married_filing_jointly': 0.073,
        'single': 0.053,
        'married_filing_separately': 0.138,
        'head_of_household': 0.124
    }
    
    def calibrate(self, tax_units: pd.DataFrame) -> pd.DataFrame:
        """Apply all calibrations."""
        tax_units = self.calibrate_total_returns(tax_units)
        tax_units = self.calibrate_filing_status(tax_units)
        tax_units = self.apply_credit_rates(tax_units)
        return tax_units
```

### Usage Example
```python
from src.tax.calibration.soi_calibration import SOICalibrator

# After constructing tax units
calibrator = SOICalibrator()
tax_units_calibrated = calibrator.calibrate(tax_units)

# Validate results
calibrator.validate(tax_units_calibrated)
```

## Data Files to Create

1. **`data/soi/hawaii_dotax_2022_table5a.csv`** - Raw SOI data
2. **`data/soi/soi_benchmarks_2022.json`** - Structured benchmarks for code
3. **`data/calibration/calibration_factors_2022.json`** - Computed calibration factors

## Validation Metrics

Track these metrics before/after calibration:

| Metric | Target (SOI) | Tolerance | Priority |
|--------|-------------|-----------|----------|
| Total returns | 635,117 | ±1% | High |
| Single % | 52.8% | ±1pp | High |
| Joint % | 34.1% | ±1pp | High |
| MFS % | 2.5% | ±0.5pp | Medium |
| HoH % | 10.6% | ±1pp | High |
| Avg AGI (Joint) | $120,767 | ±10% | Medium |
| Avg AGI (Single) | $41,907 | ±10% | Medium |
| Effective rate (Overall) | 5.94% | ±0.5pp | High |
| Credit impact | 7.7% | ±1pp | High |

## Conclusion

Integrating the Hawaii DOTAX SOI 2022 data will significantly improve model accuracy by:

1. **Correcting filing status distribution** (especially Joint and HoH)
2. **Scaling to actual return counts** (eliminating 65% overcount)
3. **Calibrating tax credit impact** (ensuring 7.7% reduction)
4. **Validating income and tax levels** (against real-world benchmarks)

**Recommended approach:** Implement Phase 1 immediately for quick wins, then proceed with Phase 2 logic improvements for long-term accuracy.
