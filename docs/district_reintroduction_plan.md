# Hawaii Legislative District-Level CTC Analysis: Reintroduction Strategy

## Executive Summary

This plan outlines a comprehensive strategy to reintroduce state legislative district-level Child Tax Credit (CTC) analysis for Hawaii, leveraging the newly integrated IRS SOI ZIP code data and enhanced geographic crosswalks. The approach addresses previous limitations while ensuring accuracy and validation through real tax filing data.

## Current State Assessment

### ✅ Completed Infrastructure
- **ZIP-to-PUMA Crosswalk**: 100% coverage of IRS SOI ZIP codes (58 ZIPs mapped)
- **Enhanced Geographic Mapping**: ZIP-PUMA-District crosswalk with 34,301 records
- **IRS SOI Data Integration**: 406 Hawaii ZIP codes with tax filing patterns
- **County-Level Analysis**: Fully functional with 3 counties mapped
- **Tax Unit Construction**: Production-ready with 85.8% joint filer capture rate

### ❌ Previous District-Level Challenges
- **Incomplete Coverage**: Only 9 House and 5 Senate districts in original analysis
- **Geographic Gaps**: Missing PUMA-to-district mappings for outer islands
- **Validation Issues**: No real-world benchmarking against actual tax data
- **Data Quality**: Inconsistent crosswalk coverage across legislative boundaries

## Strategic Approach: IRS SOI-Guided District Analysis

### Phase 1: IRS SOI District Aggregation (Weeks 1-2)

**Objective**: Create district-level IRS benchmarks from ZIP code data

**Key Activities**:
1. **Aggregate IRS SOI Data by District**
   - Use enhanced ZIP-PUMA-District crosswalk
   - Calculate filing status distributions by House/Senate district
   - Extract income patterns and return counts by district
   - Identify CTC-relevant metrics (dependents, income thresholds)

2. **District Coverage Analysis**
   - Map all 51 House districts and 25 Senate districts
   - Identify districts with sufficient IRS data coverage
   - Flag districts requiring imputation or special handling
   - Create data quality scores by district

3. **Validation Framework Setup**
   - Establish IRS SOI as ground truth for district patterns
   - Define acceptable variance thresholds for PUMS estimates
   - Create district-level benchmarking metrics

**Deliverables**:
- `data/irs_soi/district_aggregated_soi.csv`
- `scripts/aggregate_soi_by_district.py`
- District coverage quality report

### Phase 2: District-Specific Calibration (Weeks 3-4)

**Objective**: Implement district-level filing status and income calibration

**Key Activities**:
1. **Filing Status Calibration by District**
   - Replace state-level SOI targets with district-specific targets
   - Implement district-aware weight raking in tax unit construction
   - Handle districts with insufficient data through neighboring district pooling
   - Validate calibrated results against IRS district totals

2. **Income Distribution Adjustment**
   - Compare PUMS vs IRS income distributions by district
   - Apply district-specific income scaling factors if needed
   - Ensure CTC eligibility thresholds are properly calibrated
   - Validate adjusted income against IRS AGI totals

3. **Nearest-Neighbor Imputation for Incomplete Districts**
   - **Similarity Metrics**:
     - Demographic composition (age, income, household size)
     - Urban/rural classification
     - Economic indicators (median income, poverty rates)
     - Geographic proximity
   - **Implementation**:
     - Create similarity scores between districts
     - Identify 3-5 most similar districts with SOI data
     - Weighted average of neighbors' income distributions
     - Fall back to county or state averages if insufficient neighbors
   - **Data Quality Tracking**:
     - Flag all imputed values with source references
     - Calculate confidence scores based on:
       - Number of similar districts used
       - Strength of similarity metrics
       - Population coverage of source data
     - Document all fallback mechanisms used

4. **Geographic Precision Enhancement**
   - Use ZIP-level granularity to improve district boundary accuracy
   - Handle ZIP codes that cross district boundaries with population weighting
   - Implement fractional allocation for multi-district ZIP codes

**Deliverables**:
- Enhanced `TaxUnitConstructor` with district-level calibration
- `scripts/calibrate_districts_with_soi.py`
- District-specific filing status targets

### Phase 3: District CTC Pipeline Implementation (Weeks 5-6)

**Objective**: Build production district-level CTC estimation pipeline

**Key Activities**:
1. **Enhanced District Mapping**
   - Update `src/analysis/district_ctc.py` with new crosswalks
   - Implement robust district assignment for all tax units
   - Add validation checks for complete district coverage
   - Handle edge cases and boundary issues

2. **District CTC Calculation**
   - Modify `scripts/generate_district_ctc_estimates.py` for full coverage
   - Implement district-specific CTC calculations with IRS validation
   - Add confidence intervals based on data quality scores
   - Create district-level summary statistics

3. **Quality Assurance Framework**
   - Compare district CTC totals to IRS benchmarks where available
   - Implement outlier detection for unrealistic district estimates
   - Create data quality flags and confidence scores
   - Add automated validation checks

**Deliverables**:
- Updated district CTC analysis pipeline
- All 51 House + 25 Senate district estimates
- Quality assurance reports

### Phase 4: Comprehensive Reporting & Validation (Weeks 7-8)

**Objective**: Create comprehensive district analysis with validation

**Key Activities**:
1. **District Profile Generation**
   - Individual district CTC impact summaries
   - Demographic and economic context for each district
   - Comparison to state and county averages
   - Policy impact scenarios by district

2. **Cross-District Analysis**
   - Identify highest/lowest CTC impact districts
   - Analyze urban vs rural patterns
   - Compare Oahu vs neighbor island districts
   - Create district ranking and clustering analysis

3. **IRS Validation & Benchmarking**
   - Compare PUMS-based estimates to IRS actuals by district
   - Calculate validation metrics and confidence intervals
   - Document methodology and limitations
   - Create accuracy assessment reports

4. **Interactive Visualization**
   - District-level maps with CTC impact data
   - Interactive dashboards for policy analysis
   - Comparative visualization tools
   - Export capabilities for legislative use

**Deliverables**:
- Complete district CTC analysis reports
- Interactive visualization dashboard
- Validation and accuracy assessment
- Policy impact analysis tools

## Technical Implementation Strategy

### Data Architecture Enhancements

```
IRS SOI ZIP Data → ZIP-PUMA-District Crosswalk → District Aggregation
                                                        ↓
PUMS Tax Units → District-Calibrated Construction → District CTC Estimates
                                                        ↓
                      Validation & Quality Assurance → Final Reports
```

### Key Technical Components

1. **Enhanced Crosswalk System**
   - Multi-level geographic mapping (ZIP→PUMA→District→County)
   - Population-weighted allocation for boundary crossings
   - Data quality scoring and validation

2. **District-Aware Tax Unit Construction**
   - District-specific SOI calibration targets
   - Geographic filing status patterns
   - Income distribution adjustments

3. **Robust Validation Framework**
   - IRS SOI benchmarking at multiple geographic levels
   - Confidence interval calculation
   - Outlier detection and quality flagging

4. **Comprehensive Reporting System**
   - Automated district profile generation
   - Interactive visualization capabilities
   - Policy scenario analysis tools

## Risk Mitigation Strategies

### Data Quality Risks
- **Insufficient ZIP Coverage**: Use neighboring district pooling for sparse areas
- **Boundary Crossing Issues**: Implement population-weighted fractional allocation
- **IRS Data Limitations**: Create confidence scores based on data availability

### Technical Risks
- **Performance Issues**: Implement efficient batch processing and caching
- **Memory Constraints**: Use chunked processing for large datasets
- **Validation Failures**: Build robust error handling and fallback methods

### Methodological Risks
- **Over-Calibration**: Maintain transparency about adjustments and limitations
- **Geographic Precision**: Document assumptions about ZIP-district relationships
- **Temporal Misalignment**: Account for differences between PUMS and IRS data years

## Success Metrics

### Coverage Metrics
- **District Coverage**: 100% of 51 House + 25 Senate districts
- **Population Coverage**: >95% of Hawaii population mapped to districts
- **Data Quality**: >80% of districts with high-confidence estimates

### Accuracy Metrics
- **IRS Validation**: <10% variance from IRS totals where comparable
- **Filing Status Accuracy**: District-level filing patterns within 5% of IRS SOI
- **Income Calibration**: District AGI totals within 15% of IRS benchmarks

### Usability Metrics
- **Report Generation**: Automated district profiles for all districts
- **Visualization**: Interactive maps and dashboards functional
- **Policy Analysis**: Scenario analysis tools operational

## Timeline Summary

| Phase | Duration | Key Deliverable |
|-------|----------|----------------|
| 1 | Weeks 1-2 | IRS SOI district aggregation |
| 2 | Weeks 3-4 | District-specific calibration |
| 3 | Weeks 5-6 | District CTC pipeline |
| 4 | Weeks 7-8 | Comprehensive reporting |

**Total Timeline**: 8 weeks to full district-level analysis capability

## Next Immediate Actions

1. **Start Phase 1**: Create `scripts/aggregate_soi_by_district.py`
2. **Validate Crosswalk**: Ensure all districts have adequate ZIP coverage
3. **Design Calibration**: Plan district-specific SOI target methodology
4. **Update Pipeline**: Modify existing district analysis scripts

This strategy leverages our successful IRS SOI integration to overcome previous district-level limitations while ensuring accuracy through real-world tax data validation.
