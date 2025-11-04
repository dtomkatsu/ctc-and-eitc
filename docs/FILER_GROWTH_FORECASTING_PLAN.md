# Filer Growth Forecasting Plan for 2026 Projection

## Current Situation
- **Base year**: 2022/2023 (PUMS data)
- **Current weighted filers**: 634,944
- **Target year**: 2026
- **Issue**: Currently inheriting static filer count, but should forecast growth

## Problem Statement
The projection model currently:
1. ✅ Projects income growth from 2023 → 2026 using ensemble model
2. ❌ Keeps filer count static at 634,944

This is inconsistent. If incomes grow, the tax-filing population should also grow due to:
- Population growth
- Demographic shifts (aging into filing age)
- Economic participation changes
- Marriage/household formation dynamics

## Proposed Approach

### Option 1: Ensemble Return Scaling (Recommended)
**Method**: Use the ensemble projection’s `projected_returns` by filing status as growth factors for microdata weights.

**Data Sources**:
- `scripts/projections/create_ensemble_2026_projections.py` (produces `2026_ensemble_projections_*.csv`)
- Baseline tax-unit weights from `tax_units_original.parquet`

**Implementation Outline**:
```python
# 1. Aggregate baseline returns (weights) by filing status
baseline_counts = (
    tax_units.groupby('filing_status')['weight']
    .sum()
    .rename('returns_2022')
)

# 2. Load ensemble projection outputs (already run during revenue analysis)
ensemble = pd.read_csv('data/processed/projections/2026_ensemble_projections_latest.csv')

# 3. Build status-specific scaling factors
status_factors = (
    ensemble[['filing_status', '2026_returns']]
    .merge(baseline_counts, on='filing_status')
)
status_factors['scaling_factor'] = (
    status_factors['2026_returns'] / status_factors['returns_2022']
)

# 4. Apply factors back to individual units
factor_map = status_factors['scaling_factor'].to_dict()
tax_units['weight_2026'] = tax_units['weight'] * tax_units['filing_status'].map(factor_map)
```

**Pros**:
- Directly aligned with the revenue projections already in use
- Preserves filing-status mix implied by ensemble (e.g., joint vs single growth rates)
- Easy to refresh when ensemble assumptions change

**Cons**:
- Requires ensemble outputs to stay current and versioned
- Does not capture within-status demographic shifts (e.g., age) on its own

#### Population Guardrail (DBEDT Outlook)
After scaling weights by filing status, reconcile the total resident filer count with DBEDT’s population forecast to keep filer/population ratios realistic.

1. **Load DBEDT outlook CSV** (e.g., `/Users/dtomkatsu/Downloads/population growth - Outlook.csv`).
   ```python
   outlook = pd.read_csv("/Users/dtomkatsu/Downloads/population growth - Outlook.csv",
                         skiprows=3)
   pop_2026 = outlook.loc[outlook['Economic Indicators'].str.contains('Total population'), '2026']
   pop_2026 = pd.to_numeric(pop_2026.astype(str).str.replace(',', ''), errors='coerce').iloc[0] * 1_000
   ```
2. **Establish 2022 filer-to-population ratio** using baseline weights and 2022 population (≈1,441,000 from the same table):
   ```python
   filer_to_pop_ratio_2022 = baseline_counts.sum() / 1_441_000
   ```
3. **Compute implied 2026 filer target**:
   ```python
   target_returns_guardrail = pop_2026 * filer_to_pop_ratio_2022
   ```
4. **Blend ensemble total with population guardrail**:
   ```python
   ensemble_total = status_factors['2026_returns'].sum()
   guardrail_factor = target_returns_guardrail / ensemble_total
   # Option A: if guardrail_factor within tolerance (e.g., 0.97–1.03), keep ensemble totals
   # Option B: otherwise scale all factors uniformly
   if guardrail_factor < 0.97 or guardrail_factor > 1.03:
       tax_units['weight_2026'] *= guardrail_factor
   ```
5. **Record adjustments** so downstream analyses know whether a population guardrail was applied.

This keeps the new filer weights consistent with both the ensemble projection and DBEDT’s statewide population outlook.

### Option 2: Simple Population Growth Scaling (Fallback)
**Method**: Scale weights proportionally to total Hawaii population growth when ensemble projections are unavailable.

**Data Sources**:
- Hawaii DBEDT population projections 2023-2026
- US Census Bureau population estimates
- Hawaii DOH vital statistics

**Implementation**:
```python
growth_factor = pop_2026 / pop_2023
tax_units['weight_2026'] = tax_units['weight'] * growth_factor
```

**Pros**:
- Simple, transparent fallback
- Uses official population projections

**Cons**:
- Assumes filer growth = population growth (may understate)
- Ignores filing-status specific dynamics

### Option 3: Age-Stratified Growth (Refinement)
**Method**: Apply different growth rates by age cohort

**Data Sources**:
- DBEDT age-stratified population projections
- IRS SOI filer data by age
- PUMS person-level age data

**Implementation**:
```python
# Age cohorts with different growth/filing rates
age_cohorts = {
    '18-24': {'pop_growth': 1.005, 'filing_rate': 0.45},
    '25-34': {'pop_growth': 1.015, 'filing_rate': 0.75},
    '35-54': {'pop_growth': 1.020, 'filing_rate': 0.85},
    '55-64': {'pop_growth': 1.025, 'filing_rate': 0.82},
    '65+':   {'pop_growth': 1.035, 'filing_rate': 0.78}
}

# Apply cohort-specific factors
for cohort, params in age_cohorts.items():
    mask = tax_units['age_cohort'] == cohort
    tax_units.loc[mask, 'weight_2026'] = (
        tax_units.loc[mask, 'weight_2023'] * 
        params['pop_growth'] *
        params['filing_rate_adj']
    )
```

**Pros**:
- Accounts for Hawaii's aging population
- More accurate demographic dynamics
- Can capture filing rate differences by age

**Cons**:
- More complex
- Requires age cohort assignment to tax units
- More data dependencies

### Option 4: Historical Filer Growth Trends (Validation-Focused)
**Method**: Use actual Hawaii DOTax filer counts 2018-2024 to extrapolate

**Data Sources**:
- DOTax Table A8 historical filer counts by year
- Compare actual vs population growth divergence

**Historical Analysis Needed**:
```
Year    Total Filers    Population    Filers/Pop Ratio
2018    628,000        1,420,000     44.2%
2019    630,000        1,425,000     44.2%
2020    615,000        1,430,000     43.0% (pandemic)
2021    627,000        1,435,000     43.7%
2022    635,117        1,440,196     44.1%
2023    ???            1,447,000     ???
2024    ???            1,454,000     ???
2026    ???            1,470,000     ???
```

**Pros**:
- Grounded in actual data
- Can identify structural trends
- Validates other approaches

**Cons**:
- Requires historical DOTax data compilation
- May have been affected by one-time shocks (COVID)

## Recommended Implementation Sequence

### Phase 1: Ensemble Alignment (Immediate)
1. Run `create_ensemble_2026_projections.py` and capture `2026_returns` by filing status
2. Calculate scaling factors vs 2022 baseline counts (weights)
3. Apply factors within `project_to_2026.py` before income projection
4. Apply DBEDT population guardrail to reconcile total filers with statewide outlook
5. Store factors and guardrail metadata alongside projection outputs for audit

### Phase 2: Validation & Fallback Readiness
1. Compile historical DOTax filer counts 2018-2024 to sanity-check ensemble growth
2. Compare ensemble-implied growth to population trends (identify large divergences)
3. Document population-scaling fallback parameters in config for contingency use

### Phase 3: Demographic Refinements (If Needed)
1. Introduce age-cohort adjustments when ensemble or validation indicates divergent growth
2. Assign age cohorts using PUMS person data and apply targeted factors
3. Reconcile results with historical filing rates and SOI age distributions

## Data Requirements

### Immediate (Phase 1)
- [ ] Latest ensemble projection output with `2026_returns` by filing status
- [ ] Baseline tax-unit filer counts (weights) by filing status
- [ ] DBEDT population outlook CSV (`population growth - Outlook.csv`) parsed into usable totals

### Near-term (Phase 2)
- [ ] DOTax Table A8 filer counts 2018-2024
- [ ] Compile into time series

### Future (Phase 3)
- [ ] DBEDT age-stratified population projections
- [ ] Link tax units to PUMS person age data
- [ ] IRS SOI filing rates by age (if available)

## Success Criteria
1. **Consistency**: Filer growth aligns with income/revenue growth expectations
2. **Realism**: 2026 filer count matches external forecasts (if available)
3. **Transparency**: Growth assumptions clearly documented and justified
4. **Validation**: Historical trends support chosen methodology

## Implementation Notes

### Where to Add Filer Growth
**File**: `scripts/projection/project_to_2026.py`
**Location**: After loading tax_units, before income projection

```python
def apply_filer_growth(
    tax_units: pd.DataFrame,
    base_year: int,
    target_year: int,
    method: str = 'population'
) -> pd.DataFrame:
    """
    Apply filer count growth from base_year to target_year.
    
    Args:
        tax_units: Base year tax units
        base_year: Base year (e.g., 2023)
        target_year: Target year (e.g., 2026)
        method: 'population', 'age_stratified', or 'historical'
    
    Returns:
        Tax units with adjusted weights
    """
    # Implementation here
    pass
```

### Configuration
Add to `config/model_config.py`:
```python
FILER_GROWTH = {
    'base_year': 2023,
    'base_population': 1_440_196,
    'projected_population_2026': 1_470_000,  # To be confirmed
    'method': 'population',  # or 'age_stratified'
}
```

## Next Steps
1. ✅ Disable filing status weight calibration
2. ✅ Document filer growth plan
3. [ ] Research Hawaii population projections for 2026
4. [ ] Implement Phase 1 (simple population growth)
5. [ ] Validate with historical DOTax data
6. [ ] Regenerate 2026 baseline with filer growth
7. [ ] Update Act 46 rollback analysis with new baseline
