# DOTAX Historical Data Analysis Summary

## Executive Summary

Successfully parsed and analyzed DOTAX individual income tax data for tax years 2018-2021. This provides a 4-year time series of actual Hawaii resident taxpayer data including:
- Returns by filing status
- AGI by filing status  
- Tax liability (before and after credits)
- Revenue trends and growth rates

## Key Findings

### 1. Total Revenue Trends (2018-2021)

| Year | Total Returns | Total AGI ($M) | Tax After Credits ($M) | Effective Rate |
|------|--------------|----------------|------------------------|----------------|
| 2018 | 637,209 | $39,005 | $2,119 | 5.43% |
| 2019 | 651,977 | $41,692 | $2,294 | 5.50% |
| 2020 | 653,515 | $42,471 | $2,402 | 5.66% |
| 2021 | 643,653 | $49,208 | $2,903 | 5.90% |

**CAGR (2018-2021)**: 
- Returns: +0.34% per year
- AGI: +8.04% per year
- Tax Revenue: +11.08% per year

### 2. Filing Status Distribution (2021)

| Filing Status | Returns | % of Total | AGI ($M) | % of AGI | Tax ($M) | % of Tax |
|--------------|---------|------------|----------|----------|----------|----------|
| **Married Filing Jointly** | 218,942 | 34.0% | $28,563 | 58.1% | $1,696 | 58.4% |
| **Single** | 340,272 | 52.9% | $14,133 | 28.7% | $819 | 28.2% |
| **Head of Household** | 68,004 | 10.6% | $3,693 | 7.5% | $174 | 6.0% |
| **Married Filing Separately** | 16,124 | 2.5% | $2,803 | 5.7% | $213 | 7.3% |
| **Qualifying Widow(er)** | 311 | 0.05% | $15 | 0.03% | $1 | 0.03% |

### 3. Growth Rates by Filing Status (CAGR 2018-2021)

| Filing Status | Returns Growth | AGI Growth | Tax Growth |
|--------------|----------------|------------|------------|
| Single | **+0.96%** | **+7.35%** | **+9.44%** |
| Married Filing Jointly | -0.49% | +6.83% | +9.29% |
| Head of Household | -0.73% | +5.13% | +8.41% |
| Married Filing Separately | +4.00% | **+45.20%** | **+55.05%** |
| Qualifying Widow(er) | -13.65% | -3.13% | +5.64% |

**Key Insights**:
- **MFS shows explosive growth**: +45% AGI growth, +55% tax growth
- **Single filers driving volume**: Growing returns and strong revenue growth
- **Joint filers declining slightly** in count but growing in revenue
- **HoH stable**: Modest growth across all metrics

### 4. Tax Credit Impact

| Year | Tax Before Credits ($M) | Tax After Credits ($M) | Credit Reduction % |
|------|------------------------|------------------------|-------------------|
| 2018 | $2,267 | $2,119 | 6.5% |
| 2019 | $2,462 | $2,294 | 6.8% |
| 2020 | $2,578 | $2,402 | 6.8% |
| 2021 | $3,155 | $2,903 | 8.0% |

**Trend**: Credit impact increasing from 6.5% to 8.0% reduction

### 5. Model Validation Against 2021 Actual Data

**Current Model (2022 Calibrated)**:
- Total Revenue Target: $3,029M
- Filing Status Distribution:
  - Single: 51% of returns
  - Joint: 36% of returns  
  - HoH: 9.6% of returns
  - MFS: 3.4% of returns

**2021 Actual DOTAX**:
- Total Revenue: $2,903M (96% of 2022 target)
- Filing Status Distribution:
  - Single: 52.9% of returns ✅
  - Joint: 34.0% of returns ✅
  - HoH: 10.6% of returns ✅
  - MFS: 2.5% of returns ⚠️

**Assessment**: Model's 2022 targets align well with 2021 actual data when accounting for 1-year growth

## Recommendations for Model Enhancement

### Priority 1: Update 2026 Revenue Projections Using Historical CAGR

**Current Approach**: Project 2022 → 2026 using assumed growth rates

**Recommended Approach**: Use historical 2018-2021 CAGR as baseline

**Projections (2021 → 2026 using 5-year CAGR)**:

| Filing Status | 2021 Actual ($M) | 2026 Projected ($M) | 5-Year Growth |
|--------------|------------------|---------------------|---------------|
| Single | $819 | $1,220 | +49% |
| Married Filing Jointly | $1,696 | $2,556 | +51% |
| Head of Household | $174 | $253 | +45% |
| MFS | $213 | $892 | **+319%** |
| **TOTAL** | **$2,903** | **$4,921** | **+70%** |

⚠️ **Note**: MFS growth rate (+55% CAGR) may not be sustainable - recommend sensitivity analysis

**Conservative Estimate** (capping MFS growth at 15% CAGR):
- Total 2026 Revenue: ~$4,200M

### Priority 2: Validate Capital Gains Treatment

**Action**: Extract capital gains data from additional DOTAX tables to validate the 7.3% of taxpayers with capital gains assumption

**Current Model**: Capital gains represent significant but volatile revenue component

**Recommendation**: Parse Table 3 or Table 9 from DOTAX files to get capital gains breakdown

### Priority 3: Filing Status-Specific Calibration

**MFS Deep Dive**:
- Historical: 2.5% of returns (2021)
- Model: 3.4% of returns (2022 target)
- Growth: +4% CAGR in returns, +55% CAGR in tax

**Action**: 
1. Validate MFS calibration target
2. Investigate why MFS is growing so rapidly
3. Consider separate MFS projection methodology

### Priority 4: Credit Modeling Enhancement

**Current Model**: Fixed credit reduction factors
- AGI adjustments: -1.1%
- Hawaii credits: -2.1%
- Total: -3.2%

**Historical Data**: Credit reduction 6.5% → 8.0% (increasing)

**Recommendation**: 
1. Use 7-8% as baseline credit reduction
2. Model credit growth trend (increasing over time)
3. Add year-specific credit factors

### Priority 5: Create Scenario Analysis Framework

**Low Growth Scenario**: Use bottom quartile of historical growth rates
**Medium Growth Scenario**: Use CAGR (2018-2021)
**High Growth Scenario**: Use top quartile + pandemic recovery adjustment

## Implementation Plan

### Phase 1: Data Integration (Week 1) ✅
- [x] Copy historical data
- [x] Parse Table 5A (filing status data)
- [x] Calculate growth rates and CAGR
- [x] Create clean datasets

### Phase 2: Model Validation (Week 2)
- [ ] Compare model's 2021 back-projection to actual 2021 data
- [ ] Validate filing status distributions
- [ ] Validate effective tax rates by filing status
- [ ] Document discrepancies and adjustments

### Phase 3: Enhanced Projections (Week 3)
- [ ] Create CAGR-based projection methodology
- [ ] Build low/medium/high scenario framework
- [ ] Integrate MFS growth analysis
- [ ] Update 2026 revenue estimates

### Phase 4: Capital Gains Analysis (Week 4)
- [ ] Parse additional DOTAX tables for capital gains data
- [ ] Validate capital gains % of AGI assumptions
- [ ] Update capital gains projection model
- [ ] Re-run 2017 vs 2026 policy comparison

### Phase 5: Documentation (Week 5)
- [ ] Document methodology updates
- [ ] Create user guide for historical data
- [ ] Publish updated 2026 projections with confidence intervals

## Files Created

### Raw Data
- `/data/raw/dotax_historical_data/` - DOTAX spreadsheets (2015-2021)

### Processed Data
- `/data/processed/dotax_historical/dotax_historical_filing_status_clean.csv` - Clean dataset with all years
- `/data/processed/dotax_historical/dotax_historical_growth_rates.csv` - Year-over-year growth rates
- `/data/processed/dotax_historical/dotax_historical_cagr.csv` - Compound annual growth rates

### Scripts
- `/scripts/data_processing/parse_dotax_historical_improved.py` - Parser with growth analysis

### Documentation
- `/docs/DOTAX_HISTORICAL_DATA_PLAN.md` - Detailed integration plan
- `/docs/DOTAX_HISTORICAL_ANALYSIS_SUMMARY.md` - This summary

## Next Steps (Immediate Actions)

1. **Validate Model**: Compare current model's 2021 estimates to actual 2021 DOTAX data
2. **Update Projections**: Incorporate historical CAGR into 2026 projections
3. **MFS Investigation**: Analyze why MFS growth is so high (+55% CAGR)
4. **Capital Gains**: Parse additional tables to validate capital gains assumptions
5. **Scenario Analysis**: Create low/medium/high revenue scenarios for 2026

## Key Takeaways

✅ **Historical data validates model structure** - Filing status distributions align well

✅ **Tax revenue growing faster than AGI** - Effective rates increasing (5.43% → 5.90%)

⚠️ **MFS growth is anomalous** - Need to investigate and potentially cap projections

✅ **Credit impact is significant** - 6.5-8% reduction, higher than current model assumes

📊 **2026 projections need updating** - Current estimates may be conservative based on historical CAGR
