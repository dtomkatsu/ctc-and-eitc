# DOTAX Historical Data Integration Plan

## Data Overview

We now have DOTAX individual income tax tables for tax years 2018-2021 (2015-2017 have different format).

### Available Data

**Table 5A - Returns by Filing Status** (Successfully Parsed)
- Number of returns by filing status
- Hawaii AGI (positive and negative) 
- % distribution by filing status
- Tax liability before credits
- Tax liability after credits
- Available years: 2018, 2019, 2020, 2021

### Key Metrics from Table 5A (2021 example)

| Filing Status | Returns | Hawaii AGI+ ($M) | Tax Before Credits ($M) | Tax After Credits ($M) |
|--------------|---------|------------------|-------------------------|------------------------|
| Married Filing Jointly | 218,942 | $28,977 | $1,826 | $1,696 |
| Single | 340,272 | $14,351 | $867 | $819 |
| Married Filing Separately | 16,124 | $2,834 | $260 | $213 |
| Head of Household | 68,004 | $3,711 | $201 | $174 |
| Qualifying Widow(er) | 311 | $17 | $1 | $1 |
| **TOTAL** | **643,653** | **$49,890** | **$3,155** | **$2,903** |

## Integration Plan

### Phase 1: Data Cleaning & Standardization ✅

**Objective**: Parse and clean historical DOTAX data into usable format

**Tasks**:
1. ✅ Parse Table 5A (filing status data) for all years
2. ⏳ Fix column names to be meaningful
3. ⏳ Create standardized filing status names
4. ⏳ Parse additional tables (AGI brackets, deductions, etc.)
5. ⏳ Create time series dataset with all years

**Output**: Clean CSV files with standardized columns

### Phase 2: Historical Trend Analysis

**Objective**: Understand revenue trends and filing patterns over time

**Analyses**:
1. **Filing Status Trends (2018-2021)**
   - How have filing status distributions changed?
   - Which filing statuses are growing/declining?
   
2. **Revenue Trends**
   - Year-over-year tax revenue growth by filing status
   - AGI growth rates by filing status
   - Impact of tax credits over time
   
3. **Effective Tax Rate Analysis**
   - Calculate effective rates: (Tax / AGI) by filing status
   - Track changes in effective rates over time
   - Identify which groups saw tax increases/decreases

**Output**: 
- Time series charts
- Trend analysis report
- Growth rate calculations

### Phase 3: Model Validation & Calibration

**Objective**: Use historical data to validate and improve the tax model

**Key Validations**:

1. **Filing Status Distribution (2022 Target)**
   - Current model: Based on 2022 DOTAX SOI
   - Historical comparison: Do 2018-2021 trends support 2022 distribution?
   - Action: Validate or adjust filing status calibration factors

2. **Revenue Projections (2026)**
   - Historical growth rates: Calculate CAGR 2018-2021
   - Apply growth rates to project 2022 → 2026
   - Compare to current model projections
   - Action: Adjust growth assumptions if needed

3. **Tax Credit Impact**
   - Historical: (Tax After Credits / Tax Before Credits)
   - Years 2018-2021: Credit reduction ~5-8%
   - Current model: Credit reduction ~2.1% + itemized deductions ~1.2%
   - Action: Validate credit assumption or adjust

4. **Filing Status Revenue Mix**
   - Historical: % of total revenue by filing status
   - Example (2021):
     - Joint: 58.4% of revenue
     - Single: 28.2% of revenue
     - MFS: 7.3% of revenue
     - HoH: 6.0% of revenue
   - Compare to current model's calibrated distribution
   - Action: Fine-tune revenue calibration if needed

### Phase 4: Enhanced Revenue Projections

**Objective**: Create more robust 2026 revenue projections using historical trends

**Methods**:

1. **Trend-Based Projections**
   ```
   2026 Revenue = 2021 Revenue × (1 + CAGR)^5
   where CAGR = Compound Annual Growth Rate from 2018-2021
   ```

2. **Filing Status Growth Rates**
   - Calculate separate growth rates by filing status
   - Project each filing status to 2026
   - Sum to get total revenue

3. **Economic Adjustment Factors**
   - Adjust for known policy changes (standard deduction changes)
   - Adjust for economic factors (inflation, population growth)
   - Create low/medium/high scenarios

**Output**:
- Updated 2026 revenue projections with confidence intervals
- Filing status-specific projections
- Sensitivity analysis

### Phase 5: Capital Gains Analysis Enhancement

**Objective**: Better model capital gains revenue using historical data

**Analyses**:
1. Extract capital gains data from additional tables (if available)
2. Calculate historical capital gains as % of total AGI
3. Analyze capital gains volatility
4. Improve capital gains projections for 2026

### Phase 6: Policy Scenario Testing

**Objective**: Use historical data to test policy scenarios

**Scenarios**:
1. **2017 vs 2026 Deductions** (already implemented)
   - Validate using historical effective rate trends
   
2. **Tax Bracket Adjustments**
   - Model impact of bracket changes using historical rate distributions
   
3. **Credit Policy Changes**
   - Model impact of credit expansions/reductions

## Success Metrics

### Data Quality
- ✅ Successfully parsed 2018-2021 data
- ⏳ Clean, standardized dataset with all key metrics
- ⏳ Complete time series without gaps

### Model Validation
- ⏳ Model's 2021 projections within ±5% of actual 2021 DOTAX data
- ⏳ Filing status distribution matches historical trends
- ⏳ Revenue growth rates align with historical CAGR

### Improved Projections
- ⏳ 2026 projections have confidence intervals
- ⏳ Multiple scenarios (low/medium/high) based on historical volatility
- ⏳ Filing status-specific projections match historical patterns

## Implementation Timeline

### Week 1: Data Preparation
- [x] Copy historical data files
- [x] Create initial parser
- [ ] Fix column names and standardization
- [ ] Create clean master dataset

### Week 2: Analysis
- [ ] Calculate historical trends
- [ ] Create visualization dashboard
- [ ] Validate model against historical data

### Week 3: Model Enhancement
- [ ] Integrate historical growth rates
- [ ] Update 2026 projections
- [ ] Create sensitivity analysis

### Week 4: Documentation
- [ ] Document findings
- [ ] Update model documentation
- [ ] Create user guide for historical data

## Next Steps (Immediate)

1. **Fix Parser** - Clean up column names in Table 5A parser
2. **Extract Table 2** - AGI bracket distribution data
3. **Calculate Trends** - Year-over-year growth rates
4. **Validate 2021** - Compare model output to actual 2021 data
5. **Update Projections** - Incorporate historical growth rates into 2026 model

## Files Created

- `/data/raw/dotax_historical_data/` - Raw DOTAX spreadsheets (2015-2021)
- `/data/processed/dotax_historical/` - Parsed CSV files
- `/scripts/data_processing/parse_dotax_historical_data.py` - Parser script
- `/docs/DOTAX_HISTORICAL_DATA_PLAN.md` - This plan document
