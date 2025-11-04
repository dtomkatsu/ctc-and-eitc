# Middle-Income Overestimation: Root Cause Analysis

## Executive Summary

The model overestimates tax revenue by **+$578M** in middle-income brackets, which offsets the expected **-$560M** underestimation from missing capital gains. This results in the model being **+$18M higher** than the benchmark instead of properly lower.

**Two Root Causes Identified:**
1. **Population Distribution Issues**: +$253M
2. **Tax Calculation Overestimation**: +$333M

---

## Finding 1: Population Distribution Issues (+$253M)

### The Problem
We're **underrepresenting low-income** and **overrepresenting middle-income** filers:

| Bracket | DOTax % | Model % | Difference |
|---------|---------|---------|------------|
| **$0k-$10k** | 20.4% | 12.1% | **-8.3pp** ⚠️ |
| $10k-$20k | 10.1% | 9.0% | -1.1pp |
| $20k-$30k | 9.1% | 10.8% | +1.7pp |
| **$50k-$75k** | 14.4% | 18.0% | **+3.6pp** ⚠️ |
| $75k-$100k | 8.7% | 10.2% | +1.5pp |
| $100k-$150k | 9.8% | 11.4% | +1.6pp |

### Root Causes

#### A. **Income vs AGI Gap**
- PUMS measures total income (wages, self-employment, interest, etc.)
- Tax returns measure Adjusted Gross Income (AGI)
- AGI = Income - "above the line" deductions (IRA, health insurance, etc.)
- **Impact**: PUMS $45k income → $40k AGI on tax return → shifts bracket down

#### B. **Deduction Modeling Fixes** *(Updated Oct 27, 2025)*
- Removed SALT/income-tax deductions from the Hawaii itemized estimator in `src/tax/adjustments/itemized_deductions.py`
- Hawaii taxable income should not deduct Hawaii income tax payments; we now retain only mortgage interest, charitable, medical, and property tax components
- Re-ran `scripts/regenerate_tax_units.py` after the change; middle-income tax gaps improved while keeping brackets within ±10% progress tracking intact

#### B. **Non-Filer Inclusion**
- PUMS includes ALL individuals with income
- Not everyone files taxes (students with part-time jobs, retirees below threshold, etc.)
- Low-income individuals are less likely to file
- **Impact**: We create tax units for people who don't actually file

#### C. **Population Scaling Issues**
- We scale PUMS sample (~35K tax units) by ~20x to full population
- PUMS weights may not accurately represent actual tax filer distribution
- Overweighting certain demographics shifts distribution

### Impact
- **+$253M** from having more filers in higher brackets
- Shifts average income upward artificially

---

## Finding 2: Tax Calculation Overestimation (+$333M)

### The Problem
Average tax per filer is **14-17% higher** than DOTax benchmark across all middle-income brackets:

| Bracket | DOTax Avg | Model Avg | Difference |
|---------|-----------|-----------|------------|
| $20k-$30k | $889 | $1,018 | +$129 (+14.6%) |
| $30k-$40k | $1,533 | $1,741 | +$208 (+13.6%) |
| $40k-$50k | $2,166 | $2,479 | +$313 (+14.4%) |
| $50k-$75k | $3,201 | $3,676 | +$475 (+14.8%) |
| $75k-$100k | $4,751 | $5,551 | +$800 (+16.8%) |
| $100k-$150k | $7,055 | $8,239 | +$1,184 (+16.8%) |

### Example: $75k Joint Filer Analysis

**Theoretical 2017 Calculation:**
- Income: $75,000
- 2017 standard deduction: $4,400
- Taxable income: $70,600
- Applying 2017 brackets: ~$4,425 tax
- **Theoretical effective rate: 5.9%**

**DOTax Benchmark:**
- Average tax: $4,751
- **Actual effective rate: 6.3%**

**Our Model:**
- Average tax: $5,551
- **Model effective rate: 7.4%**

**Gap: Our model is 16.8% higher than DOTax, 25.4% higher than theoretical**

### Root Causes

#### A. **Insufficient Deduction Modeling**
Current adjustments:
- Itemized deductions: -1.2%
- AGI adjustments: -1.1%  
- Hawaii credits: -2.1%
- **Total: -4.4%**

**Issue**: Real taxpayers likely take more aggressive deductions
- Personal exemptions (existed in 2017, eliminated 2018+)
- More aggressive itemization for middle-income filers
- Additional Hawaii-specific deductions we're missing

#### B. **Personal Exemptions Missing?**
- 2017 tax law included **personal exemptions**
- Hawaii: ~$1,144 per exemption in 2017
- Family of 4: $4,576 additional deduction
- **Critical**: Need to verify if we're applying personal exemptions!

#### C. **Standard vs Itemized Deduction Application**
- 2017 had MUCH lower standard deductions ($4,400 joint)
- More people itemized in 2017 vs today
- If we're only applying standard deductions, we're missing itemizers
- Itemizers would have significantly lower tax liability

#### D. **Filing Status Distribution**
- Joint filers have wider brackets → lower tax on same income
- Single filers have narrower brackets → higher tax
- If we have too many single filers, increases average tax
- Need to verify our filing status distribution matches DOTax

### Impact
- **+$333M** from calculating too much tax per person
- Compounds across entire middle-income population

---

## Likely Missing Component: Personal Exemptions

### 2017 Tax Law
**Personal exemptions reduced taxable income by ~$1,144 per person:**
- Single filer: -$1,144
- Joint filer with 2 kids: -$4,576

### Impact if Missing
For a $75k joint filer with 2 kids:
- Without exemptions: $70,600 taxable → $4,425 tax
- **With exemptions: $66,024 taxable → $3,896 tax**
- Difference: $529 (12% reduction)

**If we're missing personal exemptions across the board:**
- Expected reduction: ~10-12% of tax liability
- Would account for ~$300M of the $333M gap!

---

## Recommended Investigation Steps

### Priority 1: Verify Personal Exemptions
1. **Check if 2017 personal exemptions are being applied**
   - Load tax calculation code
   - Verify exemption amounts per filing status
   - Confirm exemptions reduce taxable income before bracket application

### Priority 2: Income vs AGI Analysis
2. **Compare PUMS income to DOTax AGI distributions**
   - Load actual PUMS income data
   - Compare to DOTax AGI by bracket
   - Quantify the income → AGI gap

### Priority 3: Non-Filer Identification
3. **Identify non-filers in PUMS sample**
   - Students with income below filing threshold
   - Retirees with only Social Security
   - Part-time workers below standard deduction
   - Remove or down-weight these individuals

### Priority 4: Manual Tax Calculations
4. **Run sample tax calculations with 2017 rules**
   - Pick 10-20 sample tax units from different brackets
   - Manually calculate their 2017 tax liability
   - Compare to model output
   - Identify specific calculation errors

### Priority 5: Filing Status Validation
5. **Verify filing status distribution**
   - Compare our Single/Joint/HoH mix to DOTax
   - Adjust if significantly different
   - Recalculate revenue impact

---

## Expected Outcome After Fixes

### If Personal Exemptions Missing (Most Likely)
- **Current**: +$333M tax overestimation
- **After fix**: ~$30M overestimation (10% of current)
- **Net effect**: Model would be ~$300M lower

### If All Issues Fixed
- Fix population distribution: -$253M
- Fix tax calculation: -$333M
- **Total correction: -$586M**

### Final Expected Result
- Current model: $3.047B (+$18M vs benchmark)
- After corrections: ~$2.46B (-$570M vs benchmark)
- **Result**: Model would be properly LOWER than benchmark (missing capital gains)

---

## Conclusion

The middle-income overestimation has **two main components**:

1. **Population Distribution** (+$253M)
   - PUMS income vs AGI gap
   - Non-filer inclusion
   - Scaling issues

2. **Tax Calculation** (+$333M)
   - **Most likely: Missing personal exemptions**
   - Insufficient deductions
   - Possible filing status issues

**Critical Next Step**: Verify if 2017 personal exemptions are being applied in the tax calculation. This single issue could explain the majority of the $333M overestimation.
