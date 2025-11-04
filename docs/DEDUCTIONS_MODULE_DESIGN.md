# Deductions & Exemptions Module Design

## Overview
Module to calculate **taxable income** from AGI by applying deductions (itemized vs standard) and personal exemptions. Designed for policy flexibility to model revenue impacts of changes.

---

## Data Sources (Table A-4, A-5, A-6)

### **A4-1: Itemized Deductions by Type**
- Medical and dental expenses
- Taxes paid
- Interest expense
- Charitable contributions
- Casualty losses & misc deductions
- **By AGI bracket and filing status**

### **A4-2: Itemized vs Standard Deductions**
- Total allowable itemized deductions
- Standard deductions (number and amount)
- **Key insight**: Shows propensity to itemize by income bracket

### **A5: Personal Exemptions**
- Regular exemptions (self + spouse + qualified dependents)
- Dependent exemptions
- Age exemptions (65+)
- **Total exemption amounts by AGI bracket**

### **A6: Exemption Distribution**
- Number of returns by exemption count (1-6+)
- By AGI bracket
- **Useful for validation and modeling family size**

---

## Architecture Design

```
src/tax/deductions/
├── __init__.py
├── parsers.py              # Parse A4-1, A4-2, A5, A6 tables
├── itemized.py             # Itemized deduction logic
├── standard.py             # Standard deduction logic
├── exemptions.py           # Personal exemption logic
├── calculator.py           # Main taxable income calculator
└── policy.py               # Policy parameters (for modeling changes)
```

---

## Key Design Principles

### 1. **Separation of Data and Policy**
```python
# DON'T hardcode policy values
taxable_income = agi - 2400  # ❌ What is 2400?

# DO use configurable policy parameters
from src.tax.deductions.policy import DeductionPolicy
policy = DeductionPolicy(year=2022)
taxable_income = agi - policy.standard_deduction['single']  # ✅
```

### 2. **Itemize vs Standard Decision**
```python
# Compare and take the greater deduction
itemized_amount = calculate_itemized_deductions(tax_unit, policy)
standard_amount = policy.get_standard_deduction(filing_status, age_65_plus)

deduction = max(itemized_amount, standard_amount)
deduction_type = 'itemized' if itemized_amount > standard_amount else 'standard'
```

### 3. **Benchmark-Based Estimation**
Since we don't have individual-level deduction details in PUMS:
- Use **AGI bracket averages** from A4-2 and A5
- Apply **propensity to itemize** based on income (from A4-2 ratios)
- Adjust for **filing status** and **family size**

---

## Implementation Approach

### **Option 1: Benchmark Assignment (Recommended for Initial Implementation)**

**Pros:**
- Simple and fast
- Guaranteed to match SOI totals
- No assumptions about individual behavior

**Cons:**
- Less granular
- Can't model individual-level changes easily

**How it works:**
```python
def assign_deduction_from_benchmark(agi, filing_status, benchmarks):
    """Assign deduction based on AGI bracket average."""
    bracket = find_agi_bracket(agi, benchmarks)
    
    # Get propensity to itemize from benchmark
    itemize_pct = bracket['itemized_returns'] / bracket['total_returns']
    
    # Randomly assign itemized vs standard based on propensity
    if random.random() < itemize_pct:
        return bracket['avg_itemized_deduction'], 'itemized'
    else:
        return bracket['avg_standard_deduction'], 'standard'
```

### **Option 2: Synthetic Itemized Deduction Generation**

**Pros:**
- More granular individual-level variation
- Can model specific deduction type changes (e.g., SALT cap)
- Better for counterfactual policy analysis

**Cons:**
- More complex
- Requires distributional assumptions
- May not exactly match SOI totals (needs calibration)

**How it works:**
```python
def generate_itemized_deductions(tax_unit, benchmarks, policy):
    """Generate synthetic itemized deductions based on correlates."""
    agi = tax_unit['agi']
    filing_status = tax_unit['filing_status']
    
    # Use benchmark distributions as guides
    bracket = find_agi_bracket(agi, benchmarks)
    
    # Generate correlated deduction components
    deductions = {}
    deductions['taxes'] = sample_from_distribution(
        bracket['taxes_avg'], 
        bracket['taxes_std'],
        correlation_with_agi=0.7
    )
    deductions['mortgage_interest'] = sample_from_distribution(
        bracket['interest_avg'],
        bracket['interest_std'],
        correlation_with_agi=0.6
    )
    # ... other deduction types
    
    total_itemized = sum(deductions.values())
    
    # Apply policy limits (e.g., SALT cap)
    total_itemized = policy.apply_deduction_limits(deductions, filing_status)
    
    return total_itemized, deductions
```

### **Option 3: Hybrid Approach (Recommended for Production)**

**Combine both approaches:**
1. Use **Option 1** to ensure aggregate accuracy
2. Add **Option 2** for policy modeling flexibility
3. Use **calibration weights** to reconcile differences

```python
def calculate_deductions(tax_unit, benchmarks, policy, mode='benchmark'):
    """
    Calculate deductions with multiple modes.
    
    Args:
        mode: 'benchmark' | 'synthetic' | 'hybrid'
    """
    if mode == 'benchmark':
        return assign_deduction_from_benchmark(tax_unit, benchmarks)
    
    elif mode == 'synthetic':
        deduction, breakdown = generate_itemized_deductions(tax_unit, benchmarks, policy)
        return apply_calibration_factor(deduction, breakdown)
    
    elif mode == 'hybrid':
        # Use synthetic for policy changes, benchmark for baseline
        if policy.is_baseline():
            return assign_deduction_from_benchmark(tax_unit, benchmarks)
        else:
            return generate_itemized_deductions(tax_unit, benchmarks, policy)
```

---

## Policy Parameters Structure

```python
class DeductionPolicy:
    """
    Configurable policy parameters for deductions and exemptions.
    Enables revenue modeling of policy changes.
    """
    
    def __init__(self, year=2022):
        self.year = year
        self._set_baseline_parameters()
    
    def _set_baseline_parameters(self):
        """Set baseline 2022 Hawaii tax parameters."""
        
        # Standard deductions (2022 Hawaii values)
        self.standard_deduction = {
            'single': 2200,
            'joint': 4400,
            'hoh': 3212,
            'mfs': 2200
        }
        
        # Additional standard deduction for 65+
        self.standard_deduction_additional_age = {
            'single': 1300,
            'joint': 1050,  # per person
            'hoh': 1300,
            'mfs': 1300
        }
        
        # Personal exemption amount per person
        self.personal_exemption = 1144  # 2022 value
        
        # Itemized deduction limits (if any)
        self.itemized_deduction_limits = {
            'salt_cap': None,  # Hawaii may not have SALT cap
            'medical_agi_threshold': 0.075,  # 7.5% AGI floor
            'charitable_agi_limit': 0.5,  # 50% AGI limit
            'misc_agi_threshold': 0.02  # 2% AGI floor
        }
    
    def set_policy_change(self, parameter, value):
        """
        Modify policy parameter for scenario analysis.
        
        Examples:
            policy.set_policy_change('personal_exemption', 1500)  # Increase exemption
            policy.set_policy_change('standard_deduction.single', 2500)  # Increase std ded
        """
        if '.' in parameter:
            # Handle nested parameters
            parts = parameter.split('.')
            obj = self
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
        else:
            setattr(self, parameter, value)
    
    def get_standard_deduction(self, filing_status, age_65_plus_count=0):
        """Calculate standard deduction including age adjustments."""
        base = self.standard_deduction[filing_status]
        
        if age_65_plus_count > 0:
            additional = self.standard_deduction_additional_age[filing_status]
            if filing_status == 'joint':
                # Joint filers get additional per person
                base += additional * age_65_plus_count
            else:
                # Single/HOH/MFS get one additional
                base += additional
        
        return base
    
    def get_total_exemptions(self, num_exemptions):
        """Calculate total personal exemption amount."""
        return self.personal_exemption * num_exemptions
```

---

## Taxable Income Calculator

```python
class TaxableIncomeCalculator:
    """
    Calculate taxable income from AGI.
    
    Formula: Taxable Income = AGI - Deductions - Personal Exemptions
    """
    
    def __init__(self, deduction_benchmarks, exemption_benchmarks, policy=None):
        self.deduction_benchmarks = deduction_benchmarks
        self.exemption_benchmarks = exemption_benchmarks
        self.policy = policy or DeductionPolicy()
    
    def calculate(self, tax_unit, mode='benchmark'):
        """
        Calculate taxable income for a tax unit.
        
        Args:
            tax_unit: Dict or DataFrame row with AGI, filing_status, etc.
            mode: 'benchmark' | 'synthetic' | 'hybrid'
        
        Returns:
            Dict with taxable_income and breakdown
        """
        agi = tax_unit['agi']
        filing_status = tax_unit['filing_status']
        
        # Calculate deductions
        deduction, deduction_type = self.calculate_deduction(
            tax_unit, mode=mode
        )
        
        # Calculate exemptions
        exemption_amount = self.calculate_exemptions(tax_unit)
        
        # Calculate taxable income (cannot be negative for Hawaii)
        taxable_income = max(0, agi - deduction - exemption_amount)
        
        return {
            'taxable_income': taxable_income,
            'agi': agi,
            'deduction': deduction,
            'deduction_type': deduction_type,
            'exemption_amount': exemption_amount,
            'num_exemptions': tax_unit.get('num_exemptions', 1)
        }
    
    def calculate_batch(self, tax_units_df, mode='benchmark'):
        """Calculate taxable income for all tax units."""
        results = []
        
        for _, tax_unit in tax_units_df.iterrows():
            result = self.calculate(tax_unit, mode=mode)
            results.append(result)
        
        return pd.DataFrame(results)
```

---

## Data Parsing Strategy

### **Parse A4-2 (Deductions by AGI)**
```python
def parse_deductions_table():
    """
    Parse Table A-4 Part 2 for deduction benchmarks.
    
    Returns:
        DataFrame with columns:
        - filing_status
        - agi_min, agi_max
        - total_returns
        - itemized_returns
        - itemized_amount_millions
        - standard_returns
        - standard_amount_millions
        - avg_itemized
        - avg_standard
        - itemize_pct
    """
    # Parse separately for taxable and nontaxable
    # Merge similar to how we handled AGI brackets
```

### **Parse A5 (Exemptions by AGI)**
```python
def parse_exemptions_table():
    """
    Parse Table A-5 for exemption benchmarks.
    
    Returns:
        DataFrame with columns:
        - agi_min, agi_max
        - total_returns
        - total_exemptions
        - avg_exemptions_per_return
        - exemption_amount_millions
        - avg_exemption_amount_per_return
    """
```

---

## Policy Modeling Examples

### **Example 1: Increase Standard Deduction**
```python
# Baseline
baseline_policy = DeductionPolicy(year=2022)
baseline_results = calculator.calculate_batch(tax_units, policy=baseline_policy)
baseline_revenue = calculate_total_revenue(baseline_results)

# Scenario: Increase standard deduction by 20%
scenario_policy = DeductionPolicy(year=2022)
scenario_policy.set_policy_change('standard_deduction.single', 2640)  # 2200 * 1.2
scenario_policy.set_policy_change('standard_deduction.joint', 5280)   # 4400 * 1.2
scenario_results = calculator.calculate_batch(tax_units, policy=scenario_policy)
scenario_revenue = calculate_total_revenue(scenario_results)

revenue_impact = scenario_revenue - baseline_revenue
print(f"Revenue impact: ${revenue_impact/1e6:.1f}M")
```

### **Example 2: Eliminate Personal Exemptions**
```python
# Scenario: Eliminate personal exemptions (like federal TCJA)
scenario_policy = DeductionPolicy(year=2022)
scenario_policy.set_policy_change('personal_exemption', 0)

# May want to increase standard deduction to compensate
scenario_policy.set_policy_change('standard_deduction.single', 4400)
scenario_policy.set_policy_change('standard_deduction.joint', 8800)
```

### **Example 3: Cap Itemized Deductions**
```python
# Scenario: Cap total itemized deductions at $20k (Pease limitation style)
scenario_policy = DeductionPolicy(year=2022)
scenario_policy.itemized_deduction_cap = 20000
scenario_policy.itemized_deduction_cap_threshold = 300000  # Only applies above $300k AGI
```

---

## Validation Strategy

1. **Aggregate Totals**: Compare to Table A-4 and A-5 totals
   - Total deductions by filing status
   - Total exemptions by AGI bracket
   - Itemized vs standard split

2. **Distribution Checks**: Validate patterns
   - Itemization rate increases with income ✓
   - Average deduction amounts reasonable ✓
   - Exemptions align with family size ✓

3. **Revenue Impact**: Compare calculated tax to benchmarks
   - Use calculated taxable income in Hawaii tax calculator
   - Compare total revenue to Table A-2 totals
   - Iterate and calibrate if needed

---

## Implementation Roadmap

### **Phase 1: Data Parsing & Benchmarks** (Week 1)
- [ ] Parse Table A-4-2 (deductions)
- [ ] Parse Table A-5 (exemptions)
- [ ] Parse Table A-6 (exemption distributions)
- [ ] Create merged benchmark datasets
- [ ] Validate totals against source tables

### **Phase 2: Policy Parameters** (Week 1)
- [ ] Create `DeductionPolicy` class
- [ ] Document 2022 baseline parameters
- [ ] Add policy change methods
- [ ] Create policy scenario templates

### **Phase 3: Taxable Income Calculator** (Week 2)
- [ ] Implement benchmark assignment mode
- [ ] Add deduction type selection logic
- [ ] Add exemption calculation
- [ ] Batch processing for all tax units
- [ ] Validation against benchmarks

### **Phase 4: Synthetic Deductions (Optional)** (Week 3)
- [ ] Analyze deduction correlations with AGI
- [ ] Implement synthetic generation with distributions
- [ ] Add calibration weights
- [ ] Validate against benchmarks

### **Phase 5: Policy Modeling** (Week 4)
- [ ] Create scenario templates
- [ ] Build revenue impact calculator
- [ ] Create policy comparison reports
- [ ] Add sensitivity analysis tools

---

## Recommended Starting Approach

**Start with Option 1 (Benchmark Assignment):**

1. Parse A4-2 and A5 to get average deductions/exemptions by AGI bracket
2. Assign to each tax unit based on their AGI bracket and filing status
3. Use propensity to itemize from benchmarks (randomized but matches aggregate)
4. Validate that totals match Table A-4 and A-5
5. Integrate with Hawaii tax calculator to compute tax liability
6. Compare total revenue to benchmarks

**This gets you:**
- ✅ Accurate baseline revenue estimates
- ✅ Correct distribution of deductions/exemptions
- ✅ Foundation for policy modeling
- ✅ Fast implementation (1-2 weeks)

**Later enhancement:**
- Add synthetic deduction generation for granular policy analysis
- Add detailed itemized deduction breakdowns (SALT, mortgage, etc.)
- Calibration for specific policy scenarios

---

## Questions to Consider

1. **Does Hawaii have a SALT deduction cap?** (Check state tax code)
2. **Are there AGI phaseouts for exemptions?** (Like federal Pease limitation)
3. **Do we need separate aged 65+ exemptions?** (Table A-5 shows age exemptions)
4. **Should we model AMT?** (Alternative Minimum Tax - may be in other tables)
5. **Blind exemptions?** (Check if Hawaii has additional exemptions for blind taxpayers)

---

## File Structure

```
data/
├── raw/
│   └── Selected Resident Return Data/
│       ├── Dotax Soi 2022 - A4-1.csv  ✅ Copied
│       ├── Dotax Soi 2022 - A4-2.csv  ✅ Copied
│       ├── Dotax Soi 2022 - A5.csv    ✅ Copied
│       └── Dotax Soi 2022 - A6.csv    ✅ Copied
│
├── processed/
│   ├── deduction_benchmarks.csv       (← To create)
│   ├── exemption_benchmarks.csv       (← To create)
│   └── tax_units_with_taxable_income.parquet  (← Final output)
│
└── policy/
    ├── baseline_2022_policy.json      (← Policy parameters)
    └── scenarios/
        ├── increased_std_deduction.json
        ├── eliminated_exemptions.json
        └── itemized_cap.json
```

---

## Next Steps

1. **Review this design document** - Confirm the approach aligns with your goals
2. **Start with Phase 1** - Parse the deduction and exemption tables
3. **Validate policy parameters** - Confirm 2022 Hawaii values
4. **Implement benchmark assignment** - Quick baseline implementation
5. **Integrate with tax calculator** - Connect to existing Hawaii tax module
6. **Build policy scenarios** - Start modeling revenue impacts

Ready to proceed? I recommend starting with parsing Table A-4-2 to get the deduction benchmarks!
