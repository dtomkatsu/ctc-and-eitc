# Investigation Complete: Middle-Income Overestimation Root Cause

## Executive Summary

**Your Initial Question**: Why is the model revenue ($3.047B) HIGHER than the benchmark ($3.029B) when it should be LOWER due to missing capital gains?

**Answer Found**: The model revenue is higher due to **systematic overestimation** in middle-income brackets (+$578M) that offsets the expected capital gains gap (-$560M).

---

## Critical Discovery: What We're Actually Comparing

### The $3.029B Benchmark
- **Source**: Hawaii DOTax SOI Table 12A for 2022
- **What it is**: Actual 2022 tax returns filed by Hawaii residents
- **Tax law used**: 2022 brackets, 2022 deductions, 2022 exemptions

### The $3.047B Model  
- **Source**: Our model using 2022 PUMS data
- **What it is**: Calculated tax on 2022 incomes
- **Tax law used**: 2022 brackets, 2022 deductions, 2022 exemptions

**Key Finding**: Both use the SAME tax law (2022). This is an apples-to-apples comparison.

---

## The Two-Part Overestimation Problem

### Part 1: Population Distribution (+$253M)

We have TOO MANY people in middle-income brackets:

| Bracket | DOTax % | Model % | Difference |
|---------|---------|---------|------------|
| $0k-$10k | 20.4% | 12.1% | -8.3pp (missing low-income) |
| $50k-$75k | 14.4% | 18.0% | +3.6pp (too many) |
| $100k-$150k | 9.8% | 11.4% | +1.6pp (too many) |

**Root Causes:**
1. **Income vs AGI Gap**: PUMS measures total income, DOTax measures AGI (after "above the line" deductions like IRA contributions)
2. **Non-Filer Inclusion**: PUMS includes people who don't file taxes (students, low-income, etc.)
3. **Scaling Issues**: PUMS weights may not match actual tax filer distribution

### Part 2: Tax Calculation (+$333M)

We calculate 14-17% MORE tax per person than actual returns:

| Bracket | DOTax Avg Tax | Model Avg Tax | Overestimate |
|---------|---------------|---------------|--------------|
| $50k-$75k | $3,201 | $3,676 | +$475 (+14.8%) |
| $75k-$100k | $4,751 | $5,551 | +$800 (+16.8%) |
| $100k-$150k | $7,055 | $8,239 | +$1,184 (+16.8%) |

**Root Causes:**
1. **Insufficient Deductions**: Our -4.4% adjustment is too low
   - Itemized deductions: -1.2%
   - AGI adjustments: -1.1%
   - Hawaii credits: -2.1%
   - **Total: -4.4%**
   
2. **Real Taxpayers Take More Deductions**: 
   - More aggressive itemization
   - Additional Hawaii-specific deductions
   - Tax preparation strategies we don't model

3. **Possible Filing Status Mix**: If we have wrong proportion of Single vs Joint vs HoH

---

## Personal Exemptions: Verified as Applied

✅ **Good News**: Personal exemptions ARE being applied correctly
- $1,144 per person (adults + dependents)
- Applied before bracket calculation
- This is NOT the source of overestimation

---

## Why Not Capital Gains?

You were correct that the model SHOULD be lower due to missing capital gains:

- **High earners ($400k+)**: Model underestimates by -$560M (-56%)
  - This IS the missing capital gains effect
  
- **Middle income ($20k-$200k)**: Model overestimates by +$578M
  - This is the systematic overestimation problem

**Net Effect**: +$578M - $560M = +$18M higher than benchmark

---

## The Real Issues

### Issue 1: Income → AGI Conversion Missing

**PUMS income ≠ Tax return AGI**

Example: $50,000 PUMS income
- Minus IRA contribution: -$3,000
- Minus self-employed health insurance: -$2,000
- **= $45,000 AGI reported on tax return**

**Impact**: We're placing people in higher AGI brackets than they actually file in.

### Issue 2: Non-Filers Included

PUMS includes:
- College students with part-time income below filing threshold
- Retirees with only Social Security (not taxable in Hawaii)
- Low-income individuals below standard deduction
- Part-year residents

**Impact**: We're creating tax units for people who don't actually file, inflating filer counts in certain brackets.

### Issue 3: Deduction Protocol Insufficient

Real 2022 taxpayers achieved much more aggressive tax reduction:

- Our model: -4.4% total adjustments
- Actual taxpayers: Likely -15% to -20% effective reduction
  - Itemized deductions (SALT, mortgage, charitable)
  - Above-the-line adjustments (IRA, HSA, etc.)
  - Hawaii credits (food/excise, renewable energy, etc.)
  - Tax preparation strategies

**Impact**: We're calculating higher tax per person than real taxpayers actually paid.

---

## Recommended Fixes

### Priority 1: Income → AGI Adjustment
Apply estimated "above the line" deductions to convert PUMS income to estimated AGI:
- IRA/401k contributions: -3% to -5% of income
- Self-employed deductions: -5% to -8% for self-employed
- HSA contributions: -0.5% average
- Student loan interest: -0.5% for eligible filers

**Expected impact**: Shift distribution down 1-2 brackets, reducing revenue by ~$150M-$200M

### Priority 2: Non-Filer Identification
Create logic to identify and exclude likely non-filers:
- Income below filing threshold + no dependents
- Full-time students with only part-time wages
- Recipients of only Social Security/SSI
- Part-year residents (unless we can identify their Hawaii income portion)

**Expected impact**: Remove ~50,000-80,000 phantom filers, reducing revenue by ~$50M-$100M

### Priority 3: Enhanced Deduction Modeling
Increase adjustment factors to match actual taxpayer behavior:
- Itemized deductions: -1.2% → -3.5% (increase for middle/upper income)
- AGI adjustments: -1.1% → -2.5% (capture more retirement savings, etc.)
- Hawaii credits: -2.1% → -3.0% (better model food/excise usage)
- **New total: -9.0% (vs current -4.4%)**

**Expected impact**: Reduce revenue by ~$150M-$180M

### Priority 4: Filing Status Validation
Verify our Single/Joint/HoH distribution matches DOTax:
- Our model may have too many single filers
- Joint filers have wider brackets → lower tax on same income
- Adjust constructor logic if significant mismatch

**Expected impact**: TBD, depends on actual distribution mismatch

---

## Expected Outcome After All Fixes

| Component | Current | After Fixes | Change |
|-----------|---------|-------------|--------|
| Population distribution | +$253M | ~$50M | -$200M |
| Tax calculation | +$333M | ~$50M | -$280M |
| **Total overestimation** | +$586M | ~$100M | **-$486M** |

### Final Expected Result
- **Current model**: $3.047B (+$18M vs benchmark)
- **After fixes**: ~$2.56B (-$470M vs benchmark)
- **Result**: Model would be properly LOWER than benchmark

This -$470M gap would be reasonable given:
- Missing capital gains income (~$300M-$400M)
- Remaining modeling limitations
- Data quality differences (PUMS vs actual tax returns)

---

## Conclusion

The middle-income overestimation has **three root causes**:

1. **PUMS Income ≠ AGI** (+~$200M)
   - Need to apply "above the line" deductions

2. **Non-Filers Included** (+~$100M)
   - Need to identify and exclude

3. **Insufficient Deductions** (+~$280M)
   - Need to increase adjustment factors from -4.4% to -9.0%

**None of these relate to missing 2017 brackets or exemptions.**  
**This is purely about modeling 2022 tax law more accurately.**

The capital gains effect (-$560M in high earners) is real and expected. The problem is that middle-income overestimation is masking it.

---

## Update: Capital Gains Now Included (Oct 27, 2025)

**Added Capital Gains Modeling** for apples-to-apples comparison with DOTax benchmarks:

- Created `src/tax/adjustments/capital_gains.py` using DOTax Table 21 percentages
- Capital gains added to AGI based on income bracket (0.5% for low income → 20.9% for $400k+)
- Participation rates vary by bracket (0.03% at $0k-$10k → 47.2% at $400k+)
- **Results**: ~$558M in capital gains added to model (vs $2,995M in Table 21 benchmark)
- **Toggle capability**: `include_capital_gains` parameter in `regenerate_tax_units.py` allows easy scenario modeling with/without cap gains

**Impact on Model Accuracy**:
- Total tax: $2.657B vs $3.029B benchmark (-12.3% gap)
- Improved from -14.2% without capital gains
- High-income gap ($400k+): -57.6% (still underestimated due to concentration of cap gains)
- **4/12 brackets within ±10%** of DOTax targets

**Key Fields Added**:
- `capital_gains`: Estimated capital gains amount
- `agi_with_cap_gains`: AGI including capital gains (used for tax calculation)
- `agi_without_cap_gains`: Original AGI for scenario comparisons
