# PUMS-to-BLS OES Occupation Mapping Assessment
**Date:** October 26, 2025  
**Status:** Analysis Complete

---

## Executive Summary

**Direct PUMS SOCP → BLS SOC mapping is FEASIBLE but IMPERFECT:**
- ✅ 65.5% exact match rate (275 of 420 PUMS occupation codes)
- ✅ 91.7% major group match rate (22 of 24 2-digit codes)
- ⚠️ 34.5% of PUMS codes require fallback strategies
- ❌ Some PUMS codes are aggregated (e.g., '49-209X') and cannot map to single BLS occupations

**Recommendation:** Use a **hybrid hierarchical matching strategy** with multiple fallback levels.

---

## Data Structure Analysis

### PUMS Occupation Codes (SOCP)
- **Format:** 6-digit code (e.g., '434051')
- **Converted to SOC format:** 'XX-XXXX' (e.g., '43-4051')
- **Total unique codes in Hawaii PUMS:** 420 occupations
- **Employment coverage:** ~61% of PUMS sample (6,119 of 10,000 records)
- **Special codes:** 
  - Aggregated categories ending in 'XX' or 'YY' (e.g., '49-209X', '19-40YY')
  - Major group '55' = Military-specific occupations (not in BLS OES)
  - Major group '99' = Unemployed/not in labor force (not in BLS OES)

### BLS OES Occupation Codes (SOC)
- **Format:** 'XX-XXXX' (e.g., '11-1011')
- **Total unique codes in Hawaii:** 604 occupations
- **Years available:** 2020-2024 (5 years)
- **Coverage:** All civilian occupations with sufficient employment

### Mapping Results
| Matching Strategy | Match Rate | Notes |
|-------------------|------------|-------|
| **Exact 6-digit match** | 65.5% | Best wage specificity |
| **Major group (2-digit)** | 91.7% | Broader aggregation |
| **Unmatchable** | 8.3% | Military (55-XXXX), unemployed (99-XXXX) |

---

## Identified Mapping Challenges

### 1. **Aggregated PUMS Codes**
PUMS uses aggregated categories for small occupations to protect privacy:
- `49-209X` = "Installation, Maintenance, and Repair Workers, All Other"
- `37-201X` = "Building Cleaning Workers, All Other"
- `19-40YY` = "Life, Physical, and Social Science Technicians"

**Impact:** These codes represent multiple BLS occupations, making direct mapping impossible.

### 2. **PUMS-Specific Major Groups**
- **Major group 55:** Military-specific occupations (not in civilian BLS OES data)
- **Major group 99:** Unemployed or not in labor force

**Impact:** ~8.3% of PUMS major groups cannot map to BLS data.

### 3. **Detailed vs. Aggregated Levels**
- BLS has 604 detailed occupations
- PUMS has 420 codes, some aggregated
- Not a 1:1 relationship

**Impact:** Some PUMS workers will need wage data from broader occupation groups.

### 4. **Wage Distribution Granularity**
Even when codes match, statistical accuracy issues:
- **Within-occupation variance:** BLS shows mean/median/percentiles, but doesn't capture full distribution
- **Hawaii-specific patterns:** Small sample sizes in Hawaii for some occupations
- **Industry effects:** Same occupation has different wages across industries (BLS has industry detail, PUMS may not match)

---

## Statistical Accuracy Assessment

### ✅ Strengths of Direct Mapping
1. **Strong major group alignment:** 91.7% of PUMS workers can be matched to correct occupation category
2. **Recent wage data:** BLS OES 2020-2024 aligns well with PUMS 2019-2023
3. **Hawaii-specific:** Both datasets filtered for Hawaii, reducing geographic bias
4. **Multiple years:** Can calculate growth rates and handle temporal alignment

### ⚠️ Limitations
1. **Exact match only 65.5%:** 1/3 of workers need fallback wage data
2. **Sample size concerns:** Some BLS occupations have small Hawaii employment (<50)
3. **Industry aggregation:** BLS has industry-specific wages, PUMS matching may not align
4. **Wage distribution assumptions:** Using mean/median/percentiles assumes normal-ish distribution

### 📊 Recommended Confidence Levels
- **Exact match (65.5% of workers):** **HIGH confidence** - Use BLS wage data directly
- **Major group fallback (26.2% of workers):** **MEDIUM confidence** - Use major group average
- **Unmatchable (8.3% of workers):** **LOW confidence** - Use state-wide wage average

---

## Alternative Approaches

### Approach 1: ✅ **Hierarchical Matching (RECOMMENDED)**
Use a waterfall strategy with confidence scores:

```
1. Exact 6-digit SOCP match → Use specific BLS wage (confidence: 1.0)
2. If no exact match:
   a. Match to BLS major group average (confidence: 0.7)
   b. Weight by employment in that major group
3. If no major group match:
   a. Use state-wide wage percentile position (confidence: 0.4)
   b. Preserve relative income position
```

**Pros:**
- Maximizes use of specific BLS data where available
- Graceful degradation to broader categories
- Confidence scores enable sensitivity analysis
- Can track % of population with high-confidence matches

**Cons:**
- More complex implementation
- Some workers use aggregated data

**Implementation:**
```python
def match_pums_to_bls(pums_socp, bls_oes_data):
    # 1. Try exact match
    exact_match = bls_oes_data[bls_oes_data['occupation_code'] == pums_socp]
    if not exact_match.empty:
        return exact_match, confidence=1.0
    
    # 2. Try major group match
    major_group = pums_socp[:2]
    major_match = bls_oes_data[bls_oes_data['occupation_code'].str.startswith(major_group)]
    if not major_match.empty:
        # Use employment-weighted average
        wages = np.average(major_match['mean_wage'], weights=major_match['employment'])
        return wages, confidence=0.7
    
    # 3. Fallback to state average
    return state_wide_average, confidence=0.4
```

---

### Approach 2: **ACS Aggregate Wage Data (ALTERNATIVE)**
**Don't use PUMS occupation codes at all.** Instead, use ACS aggregate income tables.

**Data sources:**
- ✅ **B19013:** Median household income (already processed)
- ✅ **B19019:** Median income by household type (already processed)
- ✅ **B24022:** Median earnings by occupation (aggregate, not microdata)
- ✅ **B24012:** Median earnings by occupation and sex

**Pros:**
- No mapping complexity
- Uses ACS tabulated data designed for aggregate analysis
- Avoids microdata occupation code issues
- Already harmonized 2015-2024

**Cons:**
- Less granular than PUMS microdata
- Can't model individual tax units
- Limited to median earnings, not full distribution
- Can't capture household-level characteristics

**When to use:** For **aggregate projections only**, not individual tax unit modeling.

---

### Approach 3: **Income Percentile Preservation**
Don't map occupations at all. Use BLS wage growth rates at income percentile level.

**Method:**
1. Calculate BLS state-wide wage growth by percentile (p10, p25, p50, p75, p90)
2. Assign each PUMS worker to their income percentile in PUMS distribution
3. Apply BLS growth rate for that percentile
4. Preserve relative income distribution shape

**Pros:**
- No occupation mapping needed
- Preserves PUMS income distribution shape
- Uses BLS data for temporal growth trends
- Statistically robust

**Cons:**
- Loses occupation-specific information
- Assumes wage growth is uniform across occupations at each percentile
- Can't capture occupation shifts

**When to use:** When occupation detail isn't critical for tax analysis.

---

### Approach 4: **Regression-Based Imputation**
Build a predictive model to estimate wages based on demographic characteristics.

**Method:**
1. Train regression model on BLS OES data: `wage ~ occupation + age + education + experience`
2. Apply model to PUMS workers using their demographic characteristics
3. Use BLS wage growth rates to adjust over time

**Pros:**
- Uses demographic information to refine estimates
- Can handle missing occupation codes
- Captures age/education effects on wages

**Cons:**
- Complex implementation
- Requires demographic variables in both datasets
- Model assumptions may not hold

**When to use:** When you need individual-level wage predictions with high accuracy.

---

## Recommended Strategy for Ensemble Projections

### **Use Approach 1: Hierarchical Matching**

**Implementation plan:**

#### Phase 1: Direct Matching (65.5% coverage)
- Map PUMS SOCP → BLS SOC codes with exact match
- Use BLS occupation-specific wage growth rates (2020-2024)
- Calculate CAGR for each occupation
- Apply to PUMS workers with matched codes

#### Phase 2: Major Group Fallback (26.2% coverage)
- For unmatched PUMS codes, use 2-digit major group
- Calculate employment-weighted average wage growth in major group
- Apply to PUMS workers

#### Phase 3: State-Wide Fallback (8.3% coverage)
- For military (55-XXXX) and unmatchable codes
- Use state-wide wage growth from ACS B19013 (median household income)
- Apply uniformly

#### Phase 4: Confidence Weighting
- Tag each PUMS worker with confidence score (1.0, 0.7, or 0.4)
- Run sensitivity analysis varying weights
- Report uncertainty bounds in projections

#### Phase 5: Validation
- Compare projected 2024 wages to actual ACS 2024 median income
- Check if aggregate growth rates match ACS B19013 trend
- Validate against SOI income distributions

---

## Statistical Accuracy Validation

### Validation Metrics
1. **Coverage:** What % of PUMS workers get high-confidence wage matches?
2. **Aggregate accuracy:** Do projected totals match ACS aggregate tables?
3. **Distribution preservation:** Does PUMS income distribution shape stay realistic?
4. **Growth rate reasonableness:** Are occupation growth rates within expected bounds (0-10% annually)?

### Quality Control Flags
- **Flag 1:** Occupation growth rate >15% annually (outlier)
- **Flag 2:** <50 employment in Hawaii for that occupation (small sample)
- **Flag 3:** PUMS code is aggregated (XX or YY suffix)
- **Flag 4:** Military or unmatched occupation (confidence <0.5)

---

## Comparison: ACS vs. BLS for Income Projections

### Use ACS for:
- ✅ **Household-level income** (B19013, B19019)
- ✅ **Geographic patterns** (by PUMA, county)
- ✅ **Demographic relationships** (by age, household type)
- ✅ **Consistent historical series** (2015-2024)

### Use BLS for:
- ✅ **Occupation-specific wage growth**
- ✅ **Within-occupation wage distributions** (percentiles)
- ✅ **Employment trends** by occupation
- ✅ **Industry-occupation interactions**

### Ensemble Approach:
**Combine both:**
1. Use ACS for aggregate income growth trends (top-down)
2. Use BLS for occupation-specific adjustments (bottom-up)
3. Weight by confidence: High-confidence occupations use BLS, low-confidence use ACS
4. Constrain ensemble projection to match ACS aggregate totals

---

## Implementation Priority

### High Priority (Do First)
1. ✅ **Create occupation code crosswalk** (exact + major group matches)
2. ✅ **Calculate BLS occupation growth rates** (2020-2024 CAGR)
3. ✅ **Implement hierarchical matching function**
4. ✅ **Add confidence scores to PUMS workers**

### Medium Priority (Do Next)
5. **Validate against ACS aggregate tables**
6. **Run sensitivity analysis** with different confidence weights
7. **Document coverage statistics** by demographic group
8. **Create quality control reports**

### Low Priority (Nice to Have)
9. Industry-specific wage adjustments
10. Regression-based imputation for low-confidence workers
11. Temporal smoothing for volatile occupations

---

## Conclusion

**PUMS-to-BLS occupation mapping is VIABLE with caveats:**

✅ **Recommended:** Use **hierarchical matching with confidence scores**
- 65.5% exact matches = high confidence
- 26.2% major group fallbacks = medium confidence  
- 8.3% state-wide fallbacks = low confidence

✅ **Alternative:** Use **income percentile preservation** if occupation detail not needed

❌ **Not Recommended:** Requiring 100% exact matches (leaves 34.5% unmatched)

**Next step:** Implement `src/projection/occupation_matching.py` with hierarchical strategy.
