# Itemized Deduction Modeling for Ensemble Projections
**Integration with Hawaii Tax Revenue Forecasting**

---

## Overview

Itemized deductions significantly impact tax liability calculations and must be incorporated into the ensemble projection system. Currently, the model only uses standard deductions, which underestimates the tax benefits available to higher-income filers and overestimates tax revenue.

---

## Current Gap Analysis

### Standard Deduction Only Model
- **Current approach:** All tax units use standard deduction
- **2026 standard deductions:** Joint $16,000, HoH $12,000, Single $8,000
- **Problem:** Misses ~10-15% of filers who benefit from itemizing
- **Impact:** Overestimates tax revenue, especially for higher-income groups

### Itemization Patterns (National SOI Data)
- **Overall itemization rate:** ~13% of filers nationally
- **By income level:**
  - <$50K: ~5% itemize
  - $50K-$100K: ~15% itemize  
  - $100K-$200K: ~35% itemize
  - >$200K: >50% itemize
- **Hawaii specifics:** Moderate SALT, high housing costs → moderate itemization rates

---

## Itemized Deduction Components

### 1. State and Local Taxes (SALT)
**Components:**
- Hawaii state income tax (calculated by our model)
- Property taxes (from housing costs)
- Sales taxes (Hawaii GET - General Excise Tax)

**PUMS Data Sources:**
- `TAXP`: Property taxes paid
- `GRNTP`: Gross rent (includes property tax component)
- `SMOCP`: Selected monthly owner costs (includes property tax)

**Modeling Approach:**
```python
def calculate_salt_deduction(tax_unit, hawaii_income_tax):
    property_tax = estimate_property_tax(tax_unit['housing_costs'])
    sales_tax = estimate_get_tax(tax_unit['income'])  # Hawaii GET
    salt_total = hawaii_income_tax + property_tax + sales_tax
    return min(salt_total, 10000)  # Federal SALT cap
```

### 2. Mortgage Interest
**Data Sources:**
- `VALP`: Property value
- `MHP`: Mobile home costs
- `MRGP`: First mortgage payment (includes interest)

**Modeling Approach:**
```python
def estimate_mortgage_interest(tax_unit):
    if tax_unit['owns_home'] and tax_unit['has_mortgage']:
        home_value = tax_unit['VALP']
        mortgage_payment = tax_unit['MRGP'] * 12
        # Estimate interest portion (varies by loan age, rates)
        interest_rate = get_prevailing_rate(year)
        estimated_interest = estimate_interest_portion(
            home_value, mortgage_payment, interest_rate
        )
        return estimated_interest
    return 0
```

### 3. Charitable Contributions
**Modeling Approach:**
- Income-based statistical model from SOI data
- Higher-income filers give higher percentages
- Hawaii-specific adjustment factors

```python
def estimate_charitable_deduction(income, age, filing_status):
    # Base rate by income level (from SOI patterns)
    base_rate = get_charitable_rate_by_income(income)
    
    # Adjust for demographics
    age_factor = 1.2 if age > 65 else 1.0
    state_factor = 0.95  # Hawaii slightly below national average
    
    return income * base_rate * age_factor * state_factor
```

### 4. Medical Expenses
**Threshold:** >7.5% of AGI
**Modeling Approach:**
- Age-based medical expense model
- Only deductible portion above 7.5% AGI threshold

```python
def estimate_medical_deduction(income, age, num_dependents):
    base_medical = estimate_medical_expenses(age, num_dependents)
    threshold = income * 0.075
    return max(0, base_medical - threshold)
```

---

## Implementation Plan

### Phase 1: Data Preparation
**Scripts to Create:**
- `src/tax/deductions/itemized_deductions.py` - Main deduction calculator
- `src/tax/deductions/salt_calculator.py` - State and local tax estimator
- `src/tax/deductions/mortgage_interest.py` - Mortgage interest estimator
- `src/tax/deductions/charitable_giving.py` - Charitable contribution model

**Data Requirements:**
- SOI itemized deduction patterns by income level
- Hawaii property tax rates by county/area
- Mortgage interest rate time series
- Hawaii GET (sales tax) rates and patterns

### Phase 2: Itemization Decision Model
**Probability Model:**
```python
def itemization_probability(income, age, filing_status, has_mortgage, state_tax):
    # Logistic regression based on SOI patterns
    features = [
        log(income),
        age / 100,
        1 if filing_status == 'married_filing_jointly' else 0,
        1 if has_mortgage else 0,
        state_tax / income
    ]
    
    # Coefficients from SOI analysis
    coefficients = [2.1, 0.8, 0.3, 1.2, 15.0]
    logit = sum(f * c for f, c in zip(features, coefficients))
    probability = 1 / (1 + exp(-logit))
    
    return probability
```

**Decision Logic:**
```python
def choose_deduction_method(tax_unit, standard_deduction):
    # Calculate itemized deduction components
    salt = calculate_salt_deduction(tax_unit)
    mortgage_interest = estimate_mortgage_interest(tax_unit)
    charitable = estimate_charitable_deduction(tax_unit)
    medical = estimate_medical_deduction(tax_unit)
    
    total_itemized = salt + mortgage_interest + charitable + medical
    
    # Choose higher benefit
    if total_itemized > standard_deduction:
        return {
            'deduction_type': 'itemized',
            'deduction_amount': total_itemized,
            'salt': salt,
            'mortgage_interest': mortgage_interest,
            'charitable': charitable,
            'medical': medical
        }
    else:
        return {
            'deduction_type': 'standard',
            'deduction_amount': standard_deduction
        }
```

### Phase 3: Ensemble Integration
**Project Deduction Components Forward:**
```python
def project_itemized_deductions(tax_unit, target_year, income_growth_factor):
    base_year_deductions = calculate_base_year_deductions(tax_unit)
    
    projected_deductions = {}
    
    # SALT: grows with income but capped at $10,000
    salt_growth = min(
        base_year_deductions['salt'] * income_growth_factor,
        10000
    )
    projected_deductions['salt'] = salt_growth
    
    # Mortgage interest: stable or declining over time
    mortgage_decline_factor = 0.98 ** (target_year - base_year)
    projected_deductions['mortgage_interest'] = (
        base_year_deductions['mortgage_interest'] * mortgage_decline_factor
    )
    
    # Charitable: grows with income
    projected_deductions['charitable'] = (
        base_year_deductions['charitable'] * income_growth_factor
    )
    
    # Medical: grows with income and age
    age_factor = 1.02 ** (target_year - base_year)  # Medical inflation
    projected_deductions['medical'] = (
        base_year_deductions['medical'] * income_growth_factor * age_factor
    )
    
    return projected_deductions
```

### Phase 4: Tax Calculation Integration
**Modified Tax Calculator:**
```python
def calculate_tax_with_itemized_deductions(tax_unit, year):
    # Calculate standard deduction
    standard_deduction = get_standard_deduction(
        year, tax_unit['filing_status']
    )
    
    # Calculate itemized deductions
    itemized_result = choose_deduction_method(tax_unit, standard_deduction)
    
    # Use higher deduction
    deduction_amount = itemized_result['deduction_amount']
    
    # Calculate taxable income
    taxable_income = max(0, tax_unit['income'] - deduction_amount)
    
    # Apply tax brackets
    tax_liability = calculate_tax_from_brackets(
        taxable_income, year, tax_unit['filing_status']
    )
    
    return {
        'tax_liability': tax_liability,
        'deduction_type': itemized_result['deduction_type'],
        'deduction_amount': deduction_amount,
        'taxable_income': taxable_income,
        **itemized_result  # Include component details
    }
```

---

## Expected Impact on Revenue Projections

### Itemization Rate Estimates
**By Income Level (Hawaii-adjusted):**
- <$50K: ~8% itemize (higher than national due to housing costs)
- $50K-$100K: ~18% itemize
- $100K-$200K: ~40% itemize
- >$200K: >55% itemize

### Revenue Impact
**Expected reduction in tax revenue:**
- **Low-income (<$50K):** -2% (few itemizers, small deductions)
- **Middle-income ($50K-$100K):** -5% (moderate itemization)
- **High-income (>$100K):** -8% to -12% (high itemization rates, large deductions)
- **Overall:** -4% to -6% reduction in total tax revenue

### Sensitivity to Economic Conditions
**2026 Projection Adjustments:**
- **Housing market:** Higher home values → higher property taxes → more SALT deductions
- **Interest rates:** Higher rates → higher mortgage interest deductions
- **Income growth:** Higher incomes → more charitable giving, higher SALT (up to cap)

---

## Validation and Quality Control

### Data Sources for Validation
1. **Hawaii Department of Taxation:** State-specific itemization rates
2. **IRS SOI Hawaii data:** Federal itemization patterns
3. **Census ACS housing data:** Property values and mortgage patterns
4. **Bureau of Economic Analysis:** Hawaii charitable giving patterns

### Quality Checks
1. **Itemization rates by income:** Compare to SOI benchmarks
2. **Average deduction amounts:** Validate against national patterns
3. **SALT cap impact:** Ensure $10,000 federal cap is properly applied
4. **Total revenue impact:** Compare to known Hawaii tax revenue

### Sensitivity Analysis
1. **Vary itemization probability model parameters**
2. **Test different mortgage interest rate assumptions**
3. **Adjust charitable giving rates**
4. **Sensitivity to property tax rate changes**

---

## Implementation Timeline

### Week 1: Foundation
- Create deduction calculation modules
- Implement SALT, mortgage interest, charitable, medical estimators
- Build itemization probability model

### Week 2: Integration
- Integrate with existing tax calculator
- Add deduction projection logic to ensemble system
- Update tax liability calculations

### Week 3: Validation
- Compare results to SOI benchmarks
- Validate itemization rates by income level
- Adjust model parameters based on Hawaii-specific data

### Week 4: Production
- Run full 2026 projection with itemized deductions
- Generate sensitivity analysis
- Document methodology and assumptions

---

## Expected Results

### 2026 Revenue Projection (with itemized deductions)
**Current projection (standard deduction only):** $3.6B (scaled)
**With itemized deductions:** ~$3.4B (scaled) - 5-6% reduction
**More accurate representation** of actual taxpayer behavior and tax benefits

### Policy Analysis Capabilities
- **SALT cap impact:** Model effect of raising/lowering $10,000 cap
- **Standard deduction changes:** Compare itemization vs standard deduction usage
- **Mortgage interest deduction:** Model impact of potential federal changes
- **Charitable incentives:** Model impact of enhanced charitable deduction policies

This itemized deduction modeling will significantly improve the accuracy and policy relevance of the Hawaii tax revenue projections.
