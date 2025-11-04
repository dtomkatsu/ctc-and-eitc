# Income Bracket Misalignment - Root Cause Analysis

## Problems Identified

### 1. **Income Inflation Applied**
- **Current**: `calculate_tax_unit_income()` applies 2026 growth projection by default (`apply_2026_growth=True`)
- **Effect**: Income is inflated by **5.6% real growth** (2023→2026)
- **Problem**: We're comparing inflated 2026-projected income to 2022 SOI data

### 2. **Data Year Mismatch** 
- **PUMS Data**: 2018-2023 (based on SERIALNO "2018..." and ADJINC values)
- **SOI Data**: 2022
- **Problem**: Even without growth projection, there's a year mismatch

### 3. **Systematic Distribution Issues**

**Single/MFS:**
- ✅ Total match: -63 (-0.0%)
- ❌ Too many low-income: $0-$2.4K is +94.2%
- ❌ Missing high-income: $48K+ has -100%

**Married Filing Jointly:**
- ✅ Total match: -1 (-0.0%)  
- ❌ Too few low-income: $0-$28.8K is -50% to -99%
- ❌ Too many middle-income: $38.4K-$96K is +26% to +135%

**Head of Household:**
- ✅ Total match: +226 (+0.3%)
- ❌ Too few low-income: $0-$14.4K is -93% to -100%
- ❌ Too many high-income: $72K+ is +50% to +311%

## Root Causes

### Cause 1: Income Growth Projection
From `/src/tax/units/income.py`:
```python
def calculate_tax_unit_income(
    tax_unit: pd.DataFrame,
    apply_2026_growth: bool = True,  # ← DEFAULT IS TRUE!
    is_resident: bool = True
) -> float:
```

This applies 5.6% real growth which shifts all incomes upward, explaining why we have:
- Fewer filers in low brackets
- More filers in middle/high brackets

### Cause 2: Filing Status-Income Correlation
The income distribution issues correlate with filing status:
- **Single filers**: Tend to have lower incomes → growth pushes them out of lowest brackets
- **MFJ filers**: Tend to have higher incomes → growth pushes them into upper-middle brackets  
- **HoH filers**: Middle income → growth pushes them into high brackets

### Cause 3: Zero/Low Income Filers
Many tax units have very low income after standard deduction:
- Single: 149,491 filers with taxable income <$2,400 (vs SOI 82,072)
- These are likely:
  - Part-time workers
  - Students
  - Retirees with minimal income
  - People filing for refundable credits

The growth projection doesn't help low-income filers proportionally.

## Solutions

### Solution 1: Disable Growth Projection for SOI Comparison ⭐⭐⭐ RECOMMENDED

**Change**: Set `apply_2026_growth=False` when generating tax units for SOI comparison

**Implementation**:
```python
# In constructor.py, when calling calculate_tax_unit_income:
income = calculate_tax_unit_income(income_df, apply_2026_growth=False)
```

**Expected Impact**:
- Reduces all incomes by ~5.6%
- Shifts distributions back toward lower brackets
- Should improve bracket alignment significantly

**Pros**:
- Simple one-line change
- No data downloads needed
- Maintains data integrity
- Reversible (can enable for 2026 projections)

**Cons**:
- Stil has year mismatch (2018-2023 PUMS vs 2022 SOI)
- May not perfectly align due to data year differences

### Solution 2: Apply Deflation to Match 2022 ⭐⭐

**Change**: Apply deflation factor to bring PUMS income to 2022 levels

**Implementation**:
```python
# Calculate deflation factor (2023 → 2022)
# Assuming 3-4% annual inflation
deflation_factor = 1.0 / 1.03  # ~-3%

income = calculate_tax_unit_income(income_df, apply_2026_growth=False) * deflation_factor
```

**Expected Impact**:
- Further reduces incomes to match 2022 levels
- Better bracket alignment

**Pros**:
- More accurate 2022 match
- Accounts for inflation between PUMS and SOI years

**Cons**:
- Requires estimating inflation rate
- More complex
- May need calibration

### Solution 3: Download 2022 PUMS Data ⭐

**Change**: Replace current PUMS data with 2022 5-Year ACS PUMS

**Implementation**:
- Download 2022 PUMS from Census Bureau
- Replace current data files
- Regenerate tax units

**Expected Impact**:
- Perfect year match with SOI 2022
- Best possible bracket alignment

**Pros**:
- Exact year match
- Most accurate comparison
- No income adjustments needed

**Cons**:
- Requires data download and validation
- Time-consuming
- May have other data differences

## Recommended Action Plan

### Step 1: Quick Fix (5 minutes)
Disable growth projection in constructor:

```python
# src/tax/units/constructor.py
# Find all calls to calculate_tax_unit_income and add parameter:
income = calculate_tax_unit_income(income_df, apply_2026_growth=False)
```

### Step 2: Regenerate Tax Units (5 minutes)
```bash
python scripts/regenerate_tax_units.py
```

### Step 3: Re-run SOI Comparison (2 minutes)
```bash
python scripts/compare_to_soi_tables.py
```

### Step 4: Assess Results
- Check if bracket alignment improved
- Look for remaining systematic issues
- Determine if further adjustments needed

### Step 5: If Needed - Apply Deflation
If brackets still don't align:
- Calculate appropriate deflation factor
- Apply to income calculation
- Regenerate and compare again

## Expected Outcomes

With growth projection disabled:
- **Low income brackets**: Should see +50-100% increase in filers
- **Middle income brackets**: Should see -20-40% decrease in filers
- **High income brackets**: Should see -30-50% decrease in filers
- **Overall alignment**: Should improve from 8-17% to 60-80% brackets within ±10%

## Technical Details

### Current Income Calculation Flow
1. Load PUMS person data (2018-2023)
2. Sum income components (WAGP, SEMP, etc.)
3. Apply ADJINC (adjusts to PUMS survey year)
4. **Apply 2026 growth projection (+5.6%)** ← PROBLEM
5. Calculate taxable income (subtract standard deduction)
6. Map to SOI brackets

### Proposed Income Calculation Flow
1. Load PUMS person data (2018-2023)
2. Sum income components (WAGP, SEMP, etc.)
3. Apply ADJINC (adjusts to PUMS survey year)
4. **Skip growth projection** ← FIX
5. *Optional*: Apply deflation to match 2022
6. Calculate taxable income (subtract standard deduction)
7. Map to SOI brackets

## Code Locations

Files to modify:
1. `/src/tax/units/constructor.py` - Lines ~1105, ~1235, ~1372
2. `/src/tax/units/income.py` - Default parameter on line 73

Search for:
```python
calculate_tax_unit_income(income_df)
```

Replace with:
```python
calculate_tax_unit_income(income_df, apply_2026_growth=False)
```
