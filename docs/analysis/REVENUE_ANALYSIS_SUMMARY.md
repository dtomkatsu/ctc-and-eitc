# Hawaii State Tax Revenue Analysis: 2017 vs 2024

## Executive Summary

**Total Revenue Impact: -$772.1 Million (-18.06%)**

The 2024 tax brackets and standard deductions result in a **significant revenue decrease** of $772.1 million compared to the 2017 tax system, representing an 18.06% reduction in state income tax revenue.

---

## Key Findings

### Overall Revenue Impact

| Metric | 2017 | 2024 | Change | % Change |
|--------|------|------|--------|----------|
| **Total Revenue** | $4,275.3M | $3,503.2M | **-$772.1M** | **-18.06%** |
| Total Taxable Income | $56,385.9M | $54,743.9M | -$1,642.0M | -2.91% |
| Total Deductions | $1,768.3M | $3,536.7M | +$1,768.3M | +100.00% |
| Avg Effective Rate | 5.77% | 4.28% | -1.49pp | -25.82% |

### Key Drivers of Revenue Loss

1. **Doubled Standard Deductions**
   - Standard deductions increased by 100% (doubled) from 2017 to 2024
   - This alone reduced taxable income by $1.77 billion
   - Lower-income filers benefited most proportionally

2. **Expanded Tax Brackets**
   - Income thresholds increased, pushing more income into lower tax brackets
   - Reduced effective tax rates across all income levels

3. **Combined Effect**
   - Average effective tax rate dropped from 5.77% to 4.28%
   - This 1.49 percentage point reduction applied across $58.2B in gross income

---

## Impact by Filing Status

### Single Filers
- **Tax Units:** 228,459 (weighted)
- **Revenue Loss:** -$217.1M (-17.31%)
- **Impact:** Moderate revenue loss, slightly better than average

### Married Filing Jointly
- **Tax Units:** 256,555 (weighted)
- **Revenue Loss:** -$512.0M (-18.24%)
- **Impact:** Largest absolute revenue loss due to population size

### Head of Household
- **Tax Units:** 42,617 (weighted)
- **Revenue Loss:** -$43.0M (-20.06%)
- **Impact:** Highest percentage loss among filing statuses

---

## Impact by Income Level

The revenue loss is **progressive** - lower-income taxpayers saw larger percentage reductions:

| Income Quintile | Income Range | Revenue Loss | % Change |
|-----------------|--------------|--------------|----------|
| **Q1 (Lowest)** | $0 - $22K | -$16.0M | **-67.41%** |
| **Q2** | $22K - $57K | -$87.7M | **-40.15%** |
| **Q3** | $57K - $102K | -$146.6M | **-24.74%** |
| **Q4** | $102K - $170K | -$185.7M | **-18.38%** |
| **Q5 (Highest)** | $170K+ | -$336.0M | **-13.83%** |

### Distributional Analysis

- **Lowest quintile** saw the largest percentage reduction (67.41%)
  - Doubled standard deduction had outsized impact on low incomes
  - Many moved from taxable to non-taxable status

- **Highest quintile** saw the smallest percentage reduction (13.83%)
  - Still contributed the largest absolute revenue loss ($336M)
  - Standard deduction less impactful for high earners

- **Middle quintiles** experienced moderate reductions (18-40%)
  - Benefited from both expanded brackets and higher deductions

---

## Policy Implications

### Revenue Considerations

1. **Significant Revenue Loss**
   - $772M annual revenue reduction is substantial
   - Represents 18% of total income tax revenue
   - May require offsetting revenue sources or spending adjustments

2. **Progressive Tax Relief**
   - Lower-income taxpayers received proportionally larger tax cuts
   - Aligns with progressive tax policy goals
   - May improve income inequality measures

3. **Economic Stimulus**
   - $772M returned to taxpayers annually
   - Potential multiplier effects on local economy
   - Increased disposable income across all income levels

### Fiscal Planning

- **Budget Impact:** State must account for $772M annual revenue shortfall
- **Sustainability:** Consider long-term fiscal sustainability
- **Trade-offs:** Evaluate against spending priorities and service levels

---

## Technical Details

### Data Source
- **Tax Units:** 29,060 sample units (527,631 weighted)
- **Source:** PUMS (Public Use Microdata Sample) 2023 5-Year ACS
- **Geographic Coverage:** Hawaii statewide

### Methodology
1. Applied 2017 tax brackets and standard deductions to current population
2. Applied 2024 tax brackets and standard deductions to same population
3. Calculated weighted revenue totals using PUMS household weights
4. Analyzed differences by filing status and income quintile

### Assumptions
- Same population and income distribution for both scenarios
- No behavioral responses (e.g., changes in work effort, tax planning)
- No consideration of federal tax interactions
- Standard deductions only (no itemized deductions modeled)

---

## Files Generated

1. **Detailed Comparison:** `data/processed/revenue_comparison_2017_vs_2024.parquet`
   - Individual tax unit calculations for both years
   - Includes tax liability, deductions, and changes for each unit

2. **Summary Statistics:** `data/processed/revenue_summary_2017_vs_2024.csv`
   - Aggregate metrics and totals
   - Easy to import into spreadsheets

3. **Analysis Script:** `scripts/analyze_revenue_impact_2017_vs_2024.py`
   - Reproducible analysis
   - Can be adapted for other year comparisons

---

## Next Steps

### Further Analysis Opportunities

1. **Compare Additional Years**
   - Analyze 2026 and 2028 projected brackets
   - Identify long-term revenue trends

2. **Behavioral Modeling**
   - Estimate labor supply responses to tax changes
   - Model tax planning and avoidance behavior

3. **Federal Interaction**
   - Consider federal tax deductibility of state taxes
   - Analyze combined federal-state tax burden

4. **Alternative Scenarios**
   - Model partial adoption of 2024 changes
   - Explore revenue-neutral alternatives

5. **Geographic Analysis**
   - Break down by county or legislative district
   - Identify regional impacts

---

**Analysis Date:** October 7, 2025  
**Analyst:** Hawaii Income Tax Model  
**Contact:** See project documentation for details
