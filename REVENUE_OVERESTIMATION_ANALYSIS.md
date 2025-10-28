# Revenue Overestimation Analysis: Why Model is Higher Than Benchmark

## The Paradox
**Expected**: Model revenue should be LOWER than $3.029B benchmark (missing capital gains)  
**Actual**: Model revenue is $3.047B (+$18M higher than benchmark)

---

## Root Cause Analysis

### ✅ **Capital Gains Theory is CORRECT**
- High earners ($400k+): Model underestimates by -$560M (-56%)
- This confirms we're missing capital gains income (concentrated in high earners)

### ⚠️ **But We Have a Bigger Problem: Middle-Income Overestimation**
- Middle/upper-middle income: Model overestimates by +$578M
- Net effect: +$578M - $560M = +$18M higher than benchmark

---

## Overestimation Pattern by Income Bracket

| Income Bracket | DOTax Revenue | Model Revenue | Overestimate | % Too High |
|---------------|---------------|---------------|--------------|------------|
| **$50k-$75k** | $293M | $419M | **+$126M** | **+43.0%** |
| **$100k-$150k** | $438M | $594M | **+$156M** | **+35.6%** |
| **$75k-$100k** | $261M | $358M | **+$97M** | **+37.2%** |
| **$40k-$50k** | $116M | $149M | **+$33M** | **+28.4%** |
| **$150k-$200k** | $294M | $361M | **+$67M** | **+22.8%** |
| **$200k-$300k** | $310M | $361M | **+$51M** | **+16.6%** |

**Total Middle-Income Overestimation: +$530M**

---

## Potential Causes of Overestimation

### 1. **Income Inflation in PUMS Data**
- PUMS 2022 data may reflect higher incomes than actual 2022 tax returns
- Economic growth between data collection periods
- ADJINC adjustment factors may be inflating incomes too much

### 2. **Population Scaling Issues**
- Scaling factor (~20x) may be too high
- PUMS sample may not represent actual tax filer population
- Some PUMS individuals may not file taxes (students, retirees, etc.)

### 3. **Tax Calculation Methodology Differences**
- Our 2017 bracket application may differ from DOTax methodology
- Standard deduction application differences
- Filing status distribution impacts

### 4. **Insufficient Deductions/Adjustments**
- DOTax filers may use more aggressive deduction strategies
- Our AGI adjustments (-1.1%) may be underestimated
- Missing itemized deductions that reduce taxable income
- Hawaii-specific deductions we're not capturing

### 5. **Data Quality Issues**
- PUMS income may include non-taxable sources
- Survey reporting vs actual tax return differences
- Timing mismatches between survey and tax years

---

## The Two-Part Problem

### Part 1: Missing Capital Gains (Expected)
- **Impact**: -$560M in high-earner revenue
- **Cause**: PUMS doesn't capture capital gains income
- **Status**: ✅ Expected and understood

### Part 2: Middle-Income Overestimation (Unexpected)
- **Impact**: +$578M in middle-income revenue  
- **Cause**: Unknown - needs investigation
- **Status**: ⚠️ Problematic and needs fixing

---

## Investigation Priorities

### 🔴 **High Priority**
1. **Income Distribution Validation**
   - Compare PUMS income vs DOTax AGI by bracket
   - Check if ADJINC factors are inflating incomes
   - Validate population scaling factors

2. **Deduction Application Rates**
   - Check if our deduction rates are too low
   - Compare itemization rates by income level
   - Validate AGI adjustment percentages

### 🟡 **Medium Priority**
3. **Tax Calculation Methodology**
   - Compare effective tax rates by bracket
   - Validate 2017 bracket application
   - Check filing status distribution impacts

4. **Data Quality Checks**
   - Identify non-filer population in PUMS
   - Check for non-taxable income inclusion
   - Validate survey vs tax return differences

---

## Expected Outcome After Fixes

If we fix the middle-income overestimation:
- **Current**: $3.047B (+$18M vs benchmark)
- **After fixes**: ~$2.5B (-$530M vs benchmark)
- **Result**: Model would be properly LOWER than benchmark (as expected without capital gains)

---

## Conclusion

The revenue discrepancy has **two offsetting components**:
1. ✅ **Missing capital gains** (-$560M) - expected and correct
2. ⚠️ **Middle-income overestimation** (+$578M) - unexpected and problematic

**The model should be $530M+ lower than the benchmark once the overestimation issues are resolved.**
