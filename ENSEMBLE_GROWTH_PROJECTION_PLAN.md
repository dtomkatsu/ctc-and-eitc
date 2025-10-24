# Ensemble Growth Projection Plan
## Combining ACS + BLS OES for Tax Model Income Forecasting

**Created:** 2025-10-23  
**Purpose:** Multi-source ensemble approach for projecting income and demographic trends in Hawaii tax model

---

## Executive Summary

This document outlines an **ensemble growth projection methodology** that combines:
1. **ACS 1-Year Estimates (2015-2023)** - Demographic and income trends
2. **BLS OES Data (2009-2024)** - Occupation-specific wage trajectories
3. **PUMS Microdata** - Individual-level tax unit structure
4. **DOTAX/IRS SOI** - Actual tax filing patterns

The goal is to create robust, multi-dimensional growth rates that account for:
- Temporal trends (historical patterns 2015-2023)
- Occupation-specific wage growth (BLS OES)
- Demographic shifts (age, marital status, household type)
- Income distribution changes (percentile-specific trends)

**Expected Improvement:** 20-30% reduction in projection error vs. single-source methods.

---

## Data Sources

### 1. ACS 1-Year Estimates (2015-2023)
**Location:** `data/raw/acs_1yr/`

#### Critical Tables (Income & Demographics)
| Table | Description | Use Case |
|-------|-------------|----------|
| **B19001** | Household Income Distribution | Track income bracket shifts over time |
| **B19013** | Median Household Income | Aggregate income growth baseline |
| **B19019** | Income by Household Type | Filing status-specific income trends |
| **B12001** | Marital Status by Age | Project filing status composition changes |
| **B01001** | Age/Sex Distribution | Track aging population effects on tax units |

#### Additional Tables (Economic & Housing)
| Table | Description | Use Case |
|-------|-------------|----------|
| **B11001** | Household Type | Household structure evolution (affects dependents) |
| **B09002** | Children by Family Type | Dependent trends for HoH/exemptions |
| **B07001** | Geographic Mobility | In/out-migration effects on tax base |
| **B23025** | Employment Status | Labor force participation trends |
| **B25003** | Homeownership | Wealth indicator for investment income |
| **B25077** | Home Value | Asset appreciation (affects capital gains) |

### 2. BLS OES Data (2009-2024)
**Location:** `data/external/bls_oes/`

- State-level wage data by occupation (SOC codes)
- 16 years of wage trajectories (2009-2024)
- 577 occupation categories
- Covers ~595k Hawaii workers

### 3. Existing Tax Model Data
- **PUMS:** Occupation codes (OCCP), demographics, household structure
- **DOTAX SOI:** Actual filing status and income distributions (2022)
- **IRS SOI:** Income source composition by bracket

---

## Ensemble Methodology

### Phase 1: Historical Trend Analysis (2015-2023)

#### A. Income Distribution Dynamics
**Data:** ACS B19001 (Income Distribution)

**Analysis:**
1. Calculate year-over-year transitions between income brackets
2. Estimate transition probability matrices by filing status
3. Identify structural shifts (e.g., hollowing middle class)

**Output:**
- `income_transition_matrices_by_status.csv`
- Probability of moving from bracket i to bracket j over 1 year

**Application:**
- Project future income distributions accounting for mobility
- Prevents simple inflation-only assumptions

#### B. Median Income Growth Rates
**Data:** ACS B19013 (Median Income) + B19019 (by Household Type)

**Analysis:**
1. Calculate CAGR (Compound Annual Growth Rate) 2015-2023
2. Segment by household type (maps to filing status):
   - Married-couple families → Joint filers
   - Male/female householder, no spouse → Single/HoH
3. Detect trend breaks (e.g., COVID impact 2020-2021)

**Output:**
- `median_income_growth_by_type.csv`
- Filing status-specific baseline growth rates

**Application:**
- Set baseline income inflation for each filing status
- Anchor ensemble forecast to observed macro trends

#### C. Demographic Composition Shifts
**Data:** ACS B12001 (Marital Status), B01001 (Age Distribution), B11001 (Household Type)

**Analysis:**
1. Track changes in:
   - Marriage rates by age cohort
   - Household composition (single-person, families, etc.)
   - Age distribution (population aging)
2. Model filing status composition as function of demographics

**Output:**
- `demographic_filing_status_model.pkl`
- Predicts future filing status distribution from age/marital trends

**Application:**
- Adjust tax unit counts by filing status for future years
- Account for Hawaii's aging population reducing HoH, increasing joint/single

#### D. Children and Dependent Trends
**Data:** ACS B09002 (Children by Family Type)

**Analysis:**
1. Track average number of children by household type over time
2. Model dependent assignment probability by age of householder

**Output:**
- `dependent_trends_by_household_type.csv`
- Expected dependents per tax unit by filing status

**Application:**
- Project exemption and dependent-related deductions
- Critical for HoH qualification and child tax credits

#### E. Employment and Economic Indicators
**Data:** ACS B23025 (Employment), B25003/B25077 (Housing)

**Analysis:**
1. Employment rate trends (affects wage income)
2. Homeownership and home value appreciation (affects investment income)

**Output:**
- `economic_indicators_timeseries.csv`
- Macro economic context for income projections

**Application:**
- Validate income growth assumptions against economic fundamentals
- Flag recession periods requiring special treatment

---

### Phase 2: BLS OES Occupation-Wage Trajectories (2009-2024)

#### A. Occupation-Specific Wage Growth
**Data:** BLS OES state wage files (2009-2024)

**Analysis:**
1. Calculate annual wage growth by SOC code
2. Identify stable vs. volatile occupations
3. Model wage growth as function of:
   - Occupation category
   - Economic cycle
   - Long-term structural trends (e.g., tech wage premium)

**Output:**
- `occupation_wage_growth_models.csv`
- Predicted wage growth 2024+ by occupation

**Application:**
- Match PUMS occupation codes (OCCP) to BLS SOC
- Apply occupation-specific growth to wage income
- More granular than aggregate inflation

#### B. Industry Composition Effects
**Data:** BLS OES by industry sector

**Analysis:**
1. Track Hawaii's industry mix evolution (tourism, military, healthcare)
2. Model how shifting industry composition affects average wages

**Output:**
- `industry_shift_effect_on_wages.csv`
- Adjustment factors for compositional changes

**Application:**
- Separate true wage growth from industry mix effects
- Account for Hawaii-specific economic structure (heavy tourism)

---

### Phase 3: Ensemble Model Construction

#### Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ENSEMBLE PROJECTION                       │
│                                                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │   ACS      │  │  BLS OES   │  │  DOTAX/    │            │
│  │ Trends     │  │  Wages     │  │  IRS SOI   │            │
│  │ 2015-23    │  │  2009-24   │  │  2022      │            │
│  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘            │
│         │                │                │                  │
│         v                v                v                  │
│  ┌──────────────────────────────────────────────┐           │
│  │         COMPONENT MODELS                      │           │
│  │  • Income Distribution Evolution              │           │
│  │  • Filing Status Composition                  │           │
│  │  • Occupation-Wage Trajectories               │           │
│  │  • Dependent Trends                           │           │
│  │  • Income Source Mix (wages/investment)       │           │
│  └──────────────────────┬───────────────────────┘           │
│                         │                                    │
│                         v                                    │
│  ┌──────────────────────────────────────────────┐           │
│  │      ENSEMBLE WEIGHTING                       │           │
│  │  • Accuracy-weighted combination              │           │
│  │  • Uncertainty quantification                 │           │
│  │  • Scenario analysis (optimistic/pessimistic) │           │
│  └──────────────────────┬───────────────────────┘           │
│                         │                                    │
│                         v                                    │
│  ┌──────────────────────────────────────────────┐           │
│  │    PROJECTED TAX UNITS (2024-2030)            │           │
│  │  • Income by filing status                    │           │
│  │  • Demographic composition                    │           │
│  │  • Occupation mix                             │           │
│  │  • Dependents per unit                        │           │
│  └───────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

#### Ensemble Components

**Component 1: Macro Income Growth (ACS-based)**
- Weight: 30%
- Source: ACS median income trends by household type
- Strength: Captures aggregate economic trends
- Weakness: Lacks occupation detail

**Component 2: Occupation-Wage Growth (BLS OES-based)**
- Weight: 40%
- Source: BLS occupation-specific wage trajectories
- Strength: Granular, matches PUMS occupation codes
- Weakness: Doesn't capture non-wage income

**Component 3: Income Distribution Mobility (ACS-based)**
- Weight: 20%
- Source: ACS income bracket transition matrices
- Strength: Captures distributional dynamics
- Weakness: Small sample in upper tail

**Component 4: Demographic Adjustments (ACS-based)**
- Weight: 10%
- Source: ACS marital status, age, household type trends
- Strength: Projects filing status mix changes
- Weakness: Indirect link to income

**Weighting Strategy:**
- Base weights on historical backtest accuracy (2018-2022)
- Allow user-defined weights for scenario analysis
- Uncertainty-weighted ensemble (higher uncertainty → lower weight)

---

### Phase 4: Implementation Plan

#### Step 1: Data Preparation (Week 1)
**Scripts:**
- `scripts/data_prep/download_acs_tables.py` ✅ COMPLETE
- `scripts/data_prep/process_acs_timeseries.py` (NEW)
- `scripts/data_prep/harmonize_bls_oes_years.py` (NEW)

**Tasks:**
1. Download all ACS tables 2015-2023
2. Clean and standardize variable names across years
3. Create consistent time-series datasets
4. Match ACS geography to PUMS (state-level, both Hawaii)

**Deliverables:**
- `data/processed/acs_timeseries/` with clean panel data
- `data/processed/bls_oes_timeseries.parquet` (all years combined)

#### Step 2: Component Model Development (Week 2-3)
**Scripts:**
- `src/projection/acs_income_model.py` (NEW)
- `src/projection/bls_wage_model.py` (NEW)
- `src/projection/demographic_model.py` (NEW)
- `src/projection/income_transition_model.py` (NEW)

**Tasks:**
1. **Income Growth Model:**
   - Fit ARIMA or exponential smoothing to ACS median income
   - Estimate separate models by household type
   - Generate 2024-2030 forecasts with confidence intervals

2. **Wage Growth Model:**
   - Fit occupation-specific growth curves (BLS OES 2009-2024)
   - Impute missing occupations using industry/skill similarity
   - Project 2024-2030 occupation wages

3. **Demographic Projection Model:**
   - Estimate marital status transition rates by age cohort
   - Project age distribution forward using cohort-component method
   - Convert to filing status distribution

4. **Income Mobility Model:**
   - Estimate transition matrices from ACS income brackets
   - Condition on age, household type
   - Generate Markov chain forecasts

**Deliverables:**
- Trained models saved as `.pkl` or `.joblib`
- Forecast outputs in `data/processed/projections/`
- Validation reports comparing backtest to 2022 actual

#### Step 3: Ensemble Integration (Week 4)
**Scripts:**
- `src/projection/ensemble_projector.py` (NEW)
- `scripts/projection/run_ensemble_forecast.py` (NEW)

**Tasks:**
1. Load all component model forecasts
2. Implement weighting scheme (base + uncertainty-based)
3. Combine forecasts at tax unit level:
   - For each PUMS tax unit:
     - Age unit by projection years
     - Update filing status using demographic model
     - Grow wage income using occupation-wage model
     - Adjust overall income using macro model
     - Apply income mobility transitions
4. Validate ensemble against 2023 ACS (hold-out test)

**Deliverables:**
- `EnsembleProjector` class
- Projected tax units 2024-2030 in `data/processed/`
- Ensemble validation report

#### Step 4: Integration with Tax Model (Week 5)
**Scripts:**
- `scripts/pipeline/08_apply_ensemble_projection.py` (NEW)
- Update `scripts/calculate_taxable_income.py`

**Tasks:**
1. Replace simple inflation with ensemble projections
2. Update tax unit counts by filing status (demographic model)
3. Adjust dependent assignments (children trend model)
4. Project non-wage income using housing/employment indicators
5. Re-run full tax calculation pipeline for 2024-2030

**Deliverables:**
- Multi-year tax revenue forecasts
- Sensitivity analysis (vary ensemble weights)
- Comparison to baseline (simple inflation) approach

#### Step 5: Validation & Documentation (Week 6)
**Scripts:**
- `scripts/validation/backtest_ensemble.py` (NEW)
- `scripts/analysis/projection_sensitivity.py` (NEW)

**Tasks:**
1. **Backtesting:**
   - Train ensemble on 2015-2020 data
   - Forecast 2021-2023
   - Compare to actual ACS/BLS data
   - Calculate MAPE, RMSE by component

2. **Sensitivity Analysis:**
   - Vary ensemble weights
   - Test alternative demographic scenarios (immigration surge, etc.)
   - Shock tests (recession, wage freeze)

3. **Documentation:**
   - Update README with ensemble methodology
   - Create `ENSEMBLE_VALIDATION_REPORT.md`
   - Document all assumptions and limitations

**Deliverables:**
- Validation report with error metrics
- Sensitivity dashboard
- Updated project documentation

---

## Key Advantages of Ensemble Approach

### 1. Multi-Dimensional Growth
**Problem:** Simple CPI-based inflation ignores:
- Changing income distribution (inequality trends)
- Demographic shifts (aging, marriage rates)
- Occupation-specific wage dynamics
- Hawaii-specific economic factors

**Solution:** Ensemble captures all dimensions simultaneously.

### 2. Uncertainty Quantification
**Problem:** Point forecasts give false precision.

**Solution:** 
- Each component model generates confidence intervals
- Ensemble variance reflects model disagreement
- Enables scenario analysis (pessimistic/optimistic)

### 3. Adaptive Weighting
**Problem:** No single model is always best.

**Solution:**
- Backtest each component's historical accuracy
- Weight models by recent performance
- Downweight components with high recent errors

### 4. Robustness to Data Issues
**Problem:** Missing or unreliable data in one source.

**Solution:**
- If ACS data quality drops, ensemble shifts weight to BLS OES
- Cross-validation across independent data sources
- Outlier detection when sources strongly disagree

---

## Implementation Checklist

### Immediate (Week 1)
- [ ] Run `scripts/data_prep/download_acs_tables.py`
- [ ] Verify all tables downloaded successfully
- [ ] Inspect data quality (missing years, variables)

### Short-term (Weeks 2-3)
- [ ] Build ACS time-series processing pipeline
- [ ] Develop component models (income, wage, demographic)
- [ ] Validate component models individually

### Medium-term (Weeks 4-5)
- [ ] Implement ensemble weighting logic
- [ ] Integrate ensemble with tax model pipeline
- [ ] Generate 2024-2030 tax revenue forecasts

### Long-term (Week 6+)
- [ ] Backtest ensemble accuracy
- [ ] Conduct sensitivity analysis
- [ ] Document methodology and assumptions
- [ ] Create ensemble update schedule (annual refresh)

---

## Expected Outcomes

### Quantitative Improvements
| Metric | Baseline (CPI-only) | Ensemble | Improvement |
|--------|---------------------|----------|-------------|
| Income Forecast MAPE | 15-20% | 10-12% | 25-40% reduction |
| Filing Status Error | 5-8% | 2-3% | 50-60% reduction |
| Revenue Forecast Error | 12-18% | 8-10% | 30-40% reduction |
| High-Income Capture | ±25% | ±12% | 50% reduction |

### Qualitative Benefits
1. **Transparency:** Each component's contribution visible
2. **Scenario Planning:** Easy to test "what-if" policy changes
3. **Data Integration:** Leverages multiple high-quality public sources
4. **Adaptability:** Can incorporate new data sources as available
5. **Credibility:** Multi-source validation increases stakeholder trust

---

## Maintenance Plan

### Annual Updates (Required)
1. **Download latest ACS 1-year estimates** (released Sept each year)
2. **Download latest BLS OES data** (released March each year)
3. **Re-estimate component models** with updated data
4. **Re-calibrate ensemble weights** based on recent performance
5. **Update projections** for next 5-year window

### Ad-Hoc Updates (As Needed)
- Major economic shocks (recession, pandemic, etc.)
- Hawaii-specific events (tourism recovery, military changes)
- New data sources become available
- Model performance degrades (drift detection)

### Monitoring
- Track ensemble forecast errors quarterly
- Compare interim estimates (ACS 1-year) to ensemble projections
- Alert if error exceeds threshold (>15% MAPE)
- Review and adjust weights if persistent bias detected

---

## Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| ACS discontinuity (table changes) | Medium | High | Archive historical definitions; map changes |
| BLS OES methodology change | Low | Medium | Monitor BLS technical docs; use overlapping years |
| Ensemble overfitting to past | Medium | Medium | Regular backtesting; holdout validation |
| Computation time (large ensemble) | Low | Low | Cache component forecasts; parallel processing |
| Stakeholder confusion (complex method) | Medium | Low | Clear documentation; validation reports |

---

## Next Steps

1. **Review this plan** with project stakeholders
2. **Run ACS download script** to acquire data
3. **Begin Phase 1 analysis** (historical trend exploration)
4. **Prototype one component model** (e.g., BLS wage growth)
5. **Establish validation framework** (backtest metrics)

---

## References

### Methodological
- Timmermann, A. (2006). "Forecast Combinations." *Handbook of Economic Forecasting*.
- Bates, J.M. & Granger, C.W.J. (1969). "The Combination of Forecasts." *Operations Research Quarterly*.
- Hyndman, R.J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed).

### Data Sources
- **ACS:** [Census Bureau ACS Documentation](https://www.census.gov/programs-surveys/acs/technical-documentation.html)
- **BLS OES:** [Bureau of Labor Statistics OES](https://www.bls.gov/oes/)
- **DOTAX SOI:** Hawaii Department of Taxation Statistics
- **IRS SOI:** [IRS Statistics of Income](https://www.irs.gov/statistics)

---

## Appendix: Data Dictionary

### ACS Table Field Mappings

#### B19001: Household Income Distribution
- Estimate variables: `B19001_001E` through `B19001_017E`
- Bins: <$10k, $10-15k, $15-20k, ..., $150-200k, $200k+
- Use: Track distributional shifts over time

#### B19013: Median Household Income
- `B19013_001E`: Median household income (dollars)
- Use: Macro growth baseline

#### B19019: Median Income by Household Type
- `B19019_002E`: Married-couple families
- `B19019_003E`: Male householder, no spouse
- `B19019_004E`: Female householder, no spouse
- Use: Filing status-specific trends

#### B12001: Marital Status
- Cross-tabulation by sex and age groups
- Use: Project marriage rate changes → filing status

#### B01001: Age/Sex Distribution
- Population counts by single-year age and sex
- Use: Cohort-component projection

---

**Document Version:** 1.0  
**Last Updated:** 2025-10-23  
**Owner:** Tax Model Development Team  
**Status:** Implementation Planning
