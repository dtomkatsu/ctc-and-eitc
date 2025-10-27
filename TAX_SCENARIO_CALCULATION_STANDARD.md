# Hawaii Tax Scenario Calculation Standard
**Mandatory Protocol for All Tax Revenue Projections and Policy Analysis**

**Effective Date:** October 27, 2025  
**Status:** ✅ **REQUIRED FOR ALL SCENARIOS**

---

## 🎯 **Overview**

All Hawaii tax revenue scenario calculations must follow this comprehensive protocol to ensure realistic taxpayer behavior modeling and accurate policy impact assessments. Raw tax calculations overestimate revenue by ~4.4% and provide misleading policy comparisons.

---

## 📋 **Mandatory Components**

### **1. Tax Liability Calculation** ✅
- Apply appropriate year's tax brackets and standard deductions
- Use correct filing status mapping (PUMS → Hawaii format)
- Handle missing filing statuses appropriately

### **2. Itemized Deductions** ✅ **REQUIRED**
**Components:**
- **SALT deductions:** Hawaii income tax + property taxes + GET (capped at $10,000)
- **Mortgage interest:** Estimated from PUMS housing data (VALP, MRGP)
- **Charitable contributions:** Income-based statistical model from SOI patterns
- **Medical expenses:** Age/income-based model (above 7.5% AGI threshold)

**Logic:** Choose higher of standard vs itemized deduction for each tax unit

**Expected Impact:** -1.2% revenue reduction

### **3. AGI Adjustments** ✅ **REQUIRED**
**Components:**
- **IRA contributions:** 0.1-1.5% of income (income-dependent)
- **Self-employed health insurance:** 0.2% base, 3× for self-employed
- **Self-employed retirement:** 0.07% base, 5× for self-employed  
- **Student loan interest:** 0.02% (age-based, $2,500 cap)
- **Educator expenses:** $300 flat (5% of filers)

**Expected Impact:** -1.1% revenue reduction

### **4. Hawaii Tax Credits** ✅ **REQUIRED**
**Components:**
- **Food/Excise Tax Credit:** $110/exemption with income phase-out
- **Renewable Energy Credit:** ~2% of filers, average $2,000
- **Child & Dependent Care Credit:** $200-$800 based on income/dependents
- **Low-Income Renters Credit:** $50-$150 for qualifying renters

**Expected Impact:** -2.1% revenue reduction

### **5. Population Scaling** ✅ **REQUIRED**
- Scale from PUMS sample (~35K units) to full Hawaii population (~700K filers)
- **Scaling factor:** ~20.1×

---

## 🔧 **Implementation Requirements**

### **Code Integration Pattern:**
```python
# Step 1: Base tax calculation
results = tax_calculator.calculate_tax_for_dataframe(
    tax_units, year=target_year, 
    income_col='income', filing_status_col='hi_filing_status'
)

# Step 2: Apply itemized deductions
results = apply_itemized_deductions(results, tax_units)

# Step 3: Apply AGI adjustments  
results = apply_agi_adjustments(results, tax_units)

# Step 4: Apply Hawaii tax credits
results = apply_hawaii_tax_credits(results, tax_units)

# Step 5: Scale to full population
final_revenue = results['net_tax_after_all_adjustments'].sum() * scaling_factor
```

### **Required Modules:**
- `src/tax/brackets/hawaii_tax.py` - Base tax calculation
- `src/tax/deductions/itemized_deductions.py` - Itemized deduction modeling
- `src/tax/adjustments/agi_adjustments.py` - AGI adjustment calculations
- `src/tax/adjustments/hawaii_credits.py` - Hawaii state tax credits

---

## 📊 **Validation Benchmarks**

### **Expected Patterns:**
| Component | Expected Impact | Validation Check |
|-----------|----------------|------------------|
| **Itemized Deductions** | -1.2% revenue | 8% to 55% itemization by income |
| **AGI Adjustments** | -1.1% revenue | ~1.0% of total income |
| **Hawaii Credits** | -2.1% revenue | Match Hawaii DOT utilization data |
| **Total Adjustments** | -4.4% revenue | Final revenue $3-4B range |

### **Quality Checks:**
- Itemization rates by income level (Under $50K: 8%, Over $200K: 55%)
- AGI adjustment patterns by demographics and employment type
- Credit utilization rates consistent with Hawaii Department of Taxation data
- Final scaled revenue within expected Hawaii range ($3-4 billion)

---

## ✅ **Validation: Enhanced vs Standard Protocol**

### **Recent Validation Results (October 27, 2025):**

| Approach | 2026 Revenue | Notes |
|----------|--------------|-------|
| **Enhanced Protocol (2026 deductions)** | **$3.41B** | ✅ **Standard compliant** |
| **Enhanced Protocol (2017 deductions)** | **$3.74B** | ✅ **Standard compliant** |
| **Raw Tax Calculation (2024 policy)** | $4.16B | ❌ **Non-compliant** (+22% overestimate) |
| **Raw Tax Calculation (2017 policy)** | $5.09B | ❌ **Non-compliant** (+49% overestimate) |

**Key Finding:** Enhanced protocol results are consistent with ensemble model ($3.41B), while raw calculations significantly overestimate revenue.

---

## 🎯 **Policy Comparison Results**

### **Corrected 2017 vs 2026 Standard Deduction Impact:**

| Metric | 2017-Level Deductions | 2026-Level Deductions | Impact |
|--------|----------------------|----------------------|--------|
| **Total Revenue** | $3.74B | $3.41B | **-$322M (-8.6%)** |
| **Average Tax per Filer** | $5,337 | $4,877 | **-$460 (-8.6%)** |
| **Average Effective Rate** | 3.90% | 3.25% | **-0.66pp (-16.9%)** |

**Corrected Finding:** Standard deduction increases reduced revenue by $322M (-8.6%), not the previously calculated $937M (-18.4%) from flawed raw comparison.

---

## 📚 **Reference Implementation**

### **Example Script:**
`scripts/analysis/compare_2017_vs_2026_policy_enhanced.py`

**Features:**
- ✅ Follows complete standard protocol
- ✅ Fair comparison using same tax brackets (2026) for both scenarios
- ✅ Includes all required adjustments (itemized, AGI, credits)
- ✅ Proper population scaling
- ✅ Validation against ensemble model
- ✅ Comprehensive documentation

### **Output Validation:**
```
Enhanced ensemble model: $3410M
Closest scenario result: $3414M  
Difference: $4M (0.1%)
✅ Results are consistent with enhanced ensemble model
```

---

## 🚨 **Compliance Requirements**

### **MANDATORY for All Scenarios:**
1. ✅ **Use this protocol** for any Hawaii tax revenue projection
2. ✅ **Include all four adjustment types** (itemized, AGI, credits, scaling)
3. ✅ **Document methodology** clearly in all reports
4. ✅ **Validate results** against expected ranges and patterns
5. ✅ **Flag non-compliant** calculations as "raw estimates only"

### **Prohibited Practices:**
- ❌ **Raw tax calculations** without adjustments for policy analysis
- ❌ **Inconsistent methodologies** between scenarios being compared
- ❌ **Missing population scaling** for full Hawaii estimates
- ❌ **Ignoring taxpayer behavior** (itemization, credits, etc.)

---

## 📈 **Expected Benefits**

### **Accuracy Improvements:**
- **Revenue estimates:** Within 5% of actual taxpayer behavior
- **Policy comparisons:** Fair apples-to-apples analysis
- **Stakeholder confidence:** Realistic, defensible projections
- **Decision support:** Accurate impact assessments for policy makers

### **Consistency Benefits:**
- **Standardized methodology** across all Hawaii tax analysis
- **Reproducible results** with documented assumptions
- **Quality assurance** through validation benchmarks
- **Professional credibility** with comprehensive modeling

---

## 🔄 **Update Protocol**

### **When to Update This Standard:**
- New Hawaii tax policy changes (brackets, deductions, credits)
- Updated SOI data with different adjustment patterns
- Improved modeling techniques or data sources
- Validation reveals systematic biases

### **Update Process:**
1. Document proposed changes with rationale
2. Test against historical data for validation
3. Update all reference implementations
4. Communicate changes to all users
5. Archive previous version for reference

---

## 📞 **Support and Questions**

### **Implementation Support:**
- **Reference scripts:** `scripts/analysis/compare_*_enhanced.py`
- **Module documentation:** Individual module docstrings
- **Validation tools:** Built-in quality checks and benchmarks

### **Quality Assurance:**
- **Automated validation:** Results must pass consistency checks
- **Peer review:** Complex scenarios require secondary validation
- **Documentation:** All assumptions and limitations clearly stated

---

## 🎯 **Summary**

**This standard ensures all Hawaii tax revenue scenarios provide realistic, accurate, and comparable results by incorporating comprehensive taxpayer behavior modeling.**

**Key Requirements:**
1. ✅ **Complete adjustment protocol** (itemized + AGI + credits + scaling)
2. ✅ **Consistent methodology** across all scenarios
3. ✅ **Validation against benchmarks** ($3-4B range, adjustment patterns)
4. ✅ **Clear documentation** of all assumptions and limitations

**Result:** Accurate, defensible tax revenue projections that support informed policy decision-making for Hawaii.
