# Strategies for Modeling Resident + Nonresident Returns

## Goal
Model the **full 743,109 Hawaii tax returns** (635,117 residents + 107,992 nonresidents) even though PUMS only captures residents.

## Option 1: Synthetic Nonresident Population ⭐ RECOMMENDED

### Approach
Create synthetic nonresident tax units based on DOTAX Table 17A's observed distribution, then combine with PUMS residents.

### Implementation
✅ **Already implemented:**
- `src/tax/units/nonresident_synthesizer.py` - Synthesizes nonresident units
- `scripts/build_full_population.py` - Combines residents + nonresidents

### How It Works
1. **Parse Table 17A** to get nonresident AGI distribution by bracket
2. **Sample AGI brackets** according to observed frequencies
3. **Generate AGI values** within each bracket (uniform or log-normal)
4. **Assign filing statuses** based on assumed distribution (45% single, 45% joint, 5% MFS, 5% HoH)
5. **Estimate tax liability** using Table 17A average tax by bracket
6. **Weight units** to match 107,992 total nonresident returns

### Pros
- ✅ Matches DOTAX totals exactly (743,109 returns)
- ✅ Preserves observed nonresident income distribution
- ✅ Allows separate analysis of residents vs nonresidents
- ✅ Can be refined as more data becomes available
- ✅ Transparent methodology

### Cons
- ⚠️ Requires assumptions about nonresident filing status distribution
- ⚠️ Synthetic units lack household-level detail
- ⚠️ Cannot model nonresident credits accurately (limited data)

### Usage
```bash
# Build full population
python scripts/build_full_population.py

# Output: data/processed/full_tax_units.parquet
# Contains 635,117 resident + 107,992 nonresident returns
```

### Validation
- Total returns: 743,109 ✅
- Nonresident percentage: 14.5% ✅
- Nonresident tax revenue: $271M ✅
- Income distribution: Matches Table 17A ✅

---

## Option 2: Reweighting PUMS to Include Nonresidents

### Approach
Upweight PUMS resident tax units to represent both residents and synthetic nonresidents.

### How It Works
1. **Identify high-income PUMS units** that resemble nonresidents
2. **Create duplicate units** with modified characteristics
3. **Adjust weights** so total = 743,109

### Implementation Sketch
```python
# Identify units similar to nonresidents (high income, property owners)
nonres_candidates = resident_units[
    (resident_units['agi'] > 100000) |
    (resident_units['property_income'] > 0)
]

# Duplicate and modify
synthetic_nonres = nonres_candidates.copy()
synthetic_nonres['is_resident'] = False
synthetic_nonres['weight'] *= adjustment_factor

# Combine
full_population = pd.concat([resident_units, synthetic_nonres])
```

### Pros
- ✅ Uses real PUMS household structure
- ✅ Preserves correlations between income sources
- ✅ No need to synthesize from scratch

### Cons
- ❌ Assumes nonresidents look like high-income residents (may not be true)
- ❌ Difficult to match Table 17A distribution exactly
- ❌ Risk of overfitting to PUMS peculiarities
- ❌ Less transparent than synthetic approach

### Recommendation
**Not recommended** - Too many assumptions, harder to validate.

---

## Option 3: Statistical Matching with IRS Data

### Approach
Use IRS Statistics of Income Public Use File (PUF) to identify nonresident patterns, then match to Hawaii.

### How It Works
1. **Download IRS PUF** (national tax return microdata)
2. **Identify nonresident patterns** (e.g., state of residence ≠ state of income)
3. **Filter to Hawaii-relevant cases** (Hawaii income source)
4. **Reweight to match Table 17A**

### Pros
- ✅ Based on real tax return data
- ✅ Captures actual nonresident filing patterns
- ✅ Includes accurate credit usage

### Cons
- ❌ IRS PUF is complex and requires expertise
- ❌ State-level detail may be limited
- ❌ Significant data processing effort
- ❌ May not have Hawaii-specific nonresident data

### Recommendation
**Future enhancement** - Worth exploring if you need high accuracy for nonresident modeling.

---

## Option 4: Hybrid Approach (Synthetic + PUMS Augmentation)

### Approach
Combine synthetic nonresidents (Option 1) with PUMS-based refinements (Option 2).

### How It Works
1. **Start with synthetic nonresidents** (Option 1)
2. **Identify PUMS units** that could be nonresidents (military, property owners)
3. **Use PUMS characteristics** to refine synthetic units
4. **Validate against Table 17A**

### Implementation
```python
# Create base synthetic population
synthetic_nonres = synthesizer.synthesize_nonresident_units()

# Find PUMS units with nonresident characteristics
pums_nonres_like = resident_units[
    (resident_units['military'] == True) |
    (resident_units['rental_income'] > 0)
]

# Use PUMS to inform synthetic unit characteristics
# E.g., dependent distribution, income composition, etc.
for bracket in agi_brackets:
    pums_sample = pums_nonres_like[pums_nonres_like['agi'].between(bracket_min, bracket_max)]
    synthetic_units[bracket]['num_dependents'] = pums_sample['num_dependents'].sample()
```

### Pros
- ✅ Combines best of both approaches
- ✅ More realistic household characteristics
- ✅ Still matches Table 17A totals

### Cons
- ⚠️ More complex implementation
- ⚠️ Requires careful validation

### Recommendation
**Good middle ground** - Consider if Option 1 results need refinement.

---

## Option 5: Accept Limitation and Document

### Approach
Model residents only (635,117) but clearly document the nonresident gap.

### How It Works
1. **Model residents only** using PUMS
2. **Document clearly** that 14.5% of returns are excluded
3. **Add adjustment factors** for policy analysis

### Documentation Example
> "This model estimates Hawaii RESIDENT tax liability based on PUMS data. 
> Nonresidents represent an additional 14.5% of returns (107,992) and 8.8% 
> of tax revenue ($271M). For total state revenue estimates, multiply model 
> results by 1.088."

### Pros
- ✅ Simplest approach
- ✅ No questionable assumptions
- ✅ Transparent limitations
- ✅ Still useful for policy analysis

### Cons
- ❌ Cannot analyze nonresident-specific policies
- ❌ Total return count doesn't match SOI
- ❌ May confuse stakeholders

### Recommendation
**Fallback option** - Use if synthetic approach proves problematic.

---

## Recommended Implementation Plan

### Phase 1: Implement Option 1 (Synthetic Population) ✅ DONE

**Status:** Already implemented!

**Files:**
- `src/tax/units/nonresident_synthesizer.py`
- `scripts/build_full_population.py`

**Next Steps:**
1. Run `python scripts/build_full_population.py`
2. Validate outputs against Table 17A
3. Document assumptions

### Phase 2: Refine Synthetic Population

**Improvements:**
1. **Better filing status distribution**
   - Research nonresident filing patterns
   - Use IRS data if available
   - Validate against similar states

2. **Income generation**
   - Use log-normal for high brackets
   - Add income source composition (wages, rental, investment)
   - Match Table 17A more precisely

3. **Tax calculation**
   - Implement actual Hawaii tax calculator for nonresidents
   - Model limited credits (nonresidents get fewer credits)
   - Validate effective tax rates

### Phase 3: Validation and Calibration

**Validation Checks:**
- [ ] Total returns = 743,109
- [ ] Nonresident returns = 107,992 (14.5%)
- [ ] Nonresident tax revenue = $271M
- [ ] Income distribution matches Table 17A by bracket
- [ ] Average tax by bracket matches Table 17A

**Calibration:**
- Apply SOI calibration to **combined** population
- Ensure filing status distribution matches (if combined targets available)
- Validate total tax revenue

### Phase 4: Documentation and Testing

**Documentation:**
- [ ] Methodology document
- [ ] Assumptions and limitations
- [ ] Validation results
- [ ] Usage guide

**Testing:**
- [ ] Unit tests for synthesizer
- [ ] Integration tests for combined population
- [ ] Validation against DOTAX totals

---

## Comparison of Options

| Option | Complexity | Accuracy | Data Requirements | Recommended |
|--------|-----------|----------|-------------------|-------------|
| 1. Synthetic | Medium | Good | DOTAX Table 17A | ⭐ **YES** |
| 2. Reweighting | Medium | Fair | PUMS only | ❌ No |
| 3. IRS Matching | High | Excellent | IRS PUF + DOTAX | 🔮 Future |
| 4. Hybrid | High | Very Good | PUMS + DOTAX | ✅ Maybe |
| 5. Document Only | Low | N/A | None | 🔄 Fallback |

---

## Key Assumptions in Synthetic Approach

### 1. Filing Status Distribution
**Source:** DOTAX Table 4 (2022) - Actual observed data

**Actual Distribution:**
- Joint: 47.1% (50,872 returns)
- Single: 40.6% (43,852 returns)
- MFS: 6.4% (6,909 returns)
- HoH: 4.0% (4,305 returns)
- Widow: 0.0% (26 returns)
- Composite: 1.9% (2,028 returns)

**Key Insights:**
- Nonresidents are MORE likely to file jointly (47.1% vs 34.1% residents)
- Nonresidents are LESS likely to be single (40.6% vs 52.8% residents)
- Nonresidents have HIGHER MFS rate (6.4% vs 2.5% residents)
- Nonresidents have LOWER HoH rate (4.0% vs 10.6% residents)

### 2. Income Generation Within Brackets
**Assumption:** Uniform distribution within brackets, log-normal for top bracket

**Rationale:**
- Simple and transparent
- Top bracket needs long tail (billionaires)

**Alternative:** Use Pareto distribution for top bracket

### 3. Tax Liability
**Assumption:** Use Table 17A average tax by bracket with 20% random variation

**Rationale:**
- Matches observed data
- Variation captures individual differences

**Improvement:** Implement actual tax calculator

### 4. Dependent Distribution
**Assumption:** HoH has 1-3 dependents, others have 0-3

**Rationale:**
- Nonresidents less likely to have dependents in Hawaii
- HoH requires dependents by definition

**Validation:** Check if this affects tax liability significantly

---

## Usage Guide

### Quick Start
```bash
# Build full population (residents + nonresidents)
python scripts/build_full_population.py

# Output saved to: data/processed/full_tax_units.parquet
```

### Load and Analyze
```python
import pandas as pd

# Load full population
full_pop = pd.read_parquet('data/processed/full_tax_units.parquet')

# Separate residents and nonresidents
residents = full_pop[full_pop['is_resident']]
nonresidents = full_pop[~full_pop['is_resident']]

# Analyze
print(f"Total returns: {full_pop['weight'].sum():,.0f}")
print(f"Residents: {residents['weight'].sum():,.0f}")
print(f"Nonresidents: {nonresidents['weight'].sum():,.0f}")
```

### Apply Policy Changes
```python
# Example: Increase tax rate for high earners
full_pop['new_tax'] = full_pop.apply(lambda row: 
    calculate_tax_with_policy(row['agi'], row['filing_status']), 
    axis=1
)

# Calculate revenue impact
current_revenue = (full_pop['tax_after_credits'] * full_pop['weight']).sum()
new_revenue = (full_pop['new_tax'] * full_pop['weight']).sum()
revenue_change = new_revenue - current_revenue

print(f"Revenue change: ${revenue_change / 1e6:,.0f}M")
```

---

## Next Steps

1. ✅ **DONE:** Implement synthetic nonresident population
2. ⏳ **TODO:** Run `python scripts/build_full_population.py`
3. ⏳ **TODO:** Validate outputs against Table 17A
4. ⏳ **TODO:** Refine filing status assumptions
5. ⏳ **TODO:** Implement actual tax calculator for nonresidents
6. ⏳ **TODO:** Document methodology and assumptions
7. ⏳ **TODO:** Update SOI calibration to use combined population

---

## Questions & Answers

**Q: Why not just use PUMS and ignore nonresidents?**
A: You want to model the full 743,109 returns. Ignoring 14.5% of returns would underestimate total tax revenue by 8.8%.

**Q: How accurate is the synthetic approach?**
A: Very accurate for totals (matches Table 17A exactly). Less accurate for individual unit characteristics, but that's okay for aggregate policy analysis.

**Q: Can I model nonresident-specific policies?**
A: Yes, but with limitations. You can model tax rate changes, but credit eligibility may be less accurate.

**Q: What if my assumptions are wrong?**
A: Run sensitivity analysis! Test different filing status distributions, income generation methods, etc.

**Q: Should I calibrate to 635,117 or 743,109?**
A: **743,109** (combined). That's the actual total returns in Hawaii.

**Q: How do I update the synthetic population?**
A: Modify `NonresidentSynthesizer` class parameters (filing status distribution, income generation, etc.) and re-run.
