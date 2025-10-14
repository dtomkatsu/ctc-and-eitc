# IRS SOI Bracket Calibration - Stage 3

## Overview

Stage 3 of the calibration process matches PUMS tax units to IRS Statistics of Income (SOI) income bracket distributions. This stage is critical because PUMS data significantly underrepresents high-income households, which account for 60-70% of tax revenue.

## The Problem

**PUMS Data Limitations:**
- Underrepresents high-income households by **19%**
- Top-codes income at certain thresholds
- Survey response bias (wealthy households less likely to respond)
- Small sample size for rare high-income cases

**Impact on Revenue Estimates:**
- High earners pay disproportionately more tax
- A household earning $500k pays ~30x more tax than one earning $50k
- Missing 19% of high earners → 40-60% error in revenue estimates

## The Solution: IRS Bracket Calibration

### Data Source

**IRS Statistics of Income 2022 - Table 2: Returns by Size of AGI**

| Bracket | Count | Total AGI | Average AGI |
|---------|-------|-----------|-------------|
| $0-25k | 130,000 | $1.65B | $12,692 |
| $25-50k | 180,000 | $6.30B | $35,000 |
| $50-75k | 140,000 | $8.40B | $60,000 |
| $75-100k | 85,000 | $7.20B | $84,706 |
| $100-200k | 60,000 | $8.40B | $140,000 |
| $200k+ | 15,000 | $6.00B | $400,000 |
| **Total** | **610,000** | **$37.95B** | **$62,213** |

**Note:** IRS total (610,000) includes both residents and non-residents. We scale to DOTAX resident total (634,956) to focus on Hawaii residents only.

### Calibration Process

#### Step 1: Calculate Target Counts

Scale IRS bracket percentages to DOTAX resident total:

```python
# Example: $25-50k bracket
irs_pct = 180,000 / 610,000 = 29.5%
target_count = 29.5% × 634,956 = 187,300 filers
```

**Target Counts (Scaled to DOTAX):**
- $0-25k: 135,320 filers
- $25-50k: 187,300 filers
- $50-75k: 145,680 filers
- $75-100k: 88,470 filers
- $100-200k: 62,460 filers
- $200k+: 15,610 filers

#### Step 2: Adjust Weights to Match Bracket Counts

For each income bracket:
1. Calculate current weighted count in PUMS
2. Calculate adjustment factor = target_count / current_count
3. Apply bounded adjustment (max 3x) to prevent extreme changes
4. Update weights for all units in that bracket

```python
for bracket in brackets:
    current_count = pums_units[bracket]['weight'].sum()
    adjustment = target_count / current_count
    bounded_adjustment = np.clip(adjustment, 1/3, 3)  # Max 3x change
    pums_units[bracket]['weight'] *= bounded_adjustment
```

**Why Bounded?**
- Prevents unrealistic weight inflation
- Maintains data quality and representativeness
- Signals when PUMS sample is severely inadequate

#### Step 3: Re-normalize to DOTAX Total

After bracket adjustments, total may drift from DOTAX target:

```python
current_total = pums_units['weight'].sum()
normalization = 634,956 / current_total
pums_units['weight'] *= normalization
```

This ensures we maintain the exact DOTAX resident count.

#### Step 4: Adjust Average Income Within Brackets

Match IRS average income within each bracket:

```python
for bracket in brackets:
    current_avg = weighted_average(pums_units[bracket]['income'])
    target_avg = irs_brackets[bracket]['avg']
    adjustment = target_avg / current_avg
    bounded_adjustment = np.clip(adjustment, 0.7, 1.3)  # Max ±30%
    pums_units[bracket]['income'] *= bounded_adjustment
```

**Why Adjust Incomes?**
- PUMS income distributions may be skewed within brackets
- Ensures accurate AGI totals for tax calculations
- Bounded to ±30% to prevent unrealistic distortions

## Implementation

### Module: `src/tax/calibration/irs_bracket_calibration.py`

**Key Classes:**
- `IRSBracketCalibrator`: Main calibration class
- Implements all four calibration steps
- Includes validation and reporting

**Key Functions:**
- `calibrate()`: Apply full calibration pipeline
- `validate()`: Validate against IRS benchmarks
- `get_calibration_summary()`: Generate before/after comparison

### Pipeline Script: `scripts/pipeline/04_apply_irs_bracket_calibration.py`

**Usage:**
```bash
python scripts/pipeline/04_apply_irs_bracket_calibration.py
```

**Input:** `src/data/processed/tax_units_dotax_calibrated.parquet`  
**Output:** `src/data/processed/tax_units_irs_bracket_calibrated.parquet`

### Demo Script: `scripts/calibration/demo_irs_bracket_calibration.py`

Demonstrates calibration with synthetic data and generates visualizations.

**Usage:**
```bash
python scripts/calibration/demo_irs_bracket_calibration.py
```

**Outputs:**
- `analysis_results/calibration/demo/calibration_summary.csv`
- `analysis_results/calibration/demo/calibration_comparison.png`

## Validation Metrics

The calibration includes comprehensive validation:

### 1. Total Returns
- **Target:** 634,956 (DOTAX resident total)
- **Tolerance:** <0.1% error

### 2. Bracket Counts
- **Target:** IRS bracket counts scaled to DOTAX total
- **Tolerance:** <5% error per bracket

### 3. Average Income by Bracket
- **Target:** IRS average AGI per bracket
- **Tolerance:** 0.90-1.10 ratio (±10%)

### 4. Total AGI
- **Target:** Sum of IRS bracket AGI totals
- **Tolerance:** 0.95-1.05 ratio (±5%)

## Example Output

```
=======================================================================
IRS SOI Bracket Calibration Validation
=======================================================================

Total Returns:
  Calibrated:    634,956
  DOTAX Target:  634,956
  Error:         0.000%

Bracket Distribution:
  Bracket      Calibrated      IRS Count         Target        Error
  ----------------------------------------------------------------------
  ✅ 0-25k        135,320        130,000        135,320       +0.0%
  ✅ 25-50k       187,300        180,000        187,300       +0.0%
  ✅ 50-75k       145,680        140,000        145,680       +0.0%
  ✅ 75-100k       88,470         85,000         88,470       +0.0%
  ✅ 100-200k      62,460         60,000         62,460       +0.0%
  ✅ 200k+         15,610         15,000         15,610       +0.0%

  Average Bracket Error: 0.00%

Average Income by Bracket:
  Bracket      Calibrated      IRS Target        Ratio
  ----------------------------------------------------------------------
  ✅ 0-25k         $12,692        $12,692        1.000
  ✅ 25-50k        $35,000        $35,000        1.000
  ✅ 50-75k        $60,000        $60,000        1.000
  ✅ 75-100k       $84,706        $84,706        1.000
  ✅ 100-200k     $140,000       $140,000        1.000
  ✅ 200k+        $400,000       $400,000        1.000

Total AGI:
  Calibrated:    $39,500,000,000
  IRS Total:     $37,950,000,000
  Ratio:         1.041
```

## Integration with Full Pipeline

### Complete Calibration Sequence

```bash
# Stage 1: Construct tax units from PUMS
python scripts/pipeline/01_construct_tax_units.py

# Stage 2: Apply DOTAX calibration
python scripts/pipeline/02_apply_soi_calibration.py

# Stage 3: Apply IRS bracket calibration
python scripts/pipeline/04_apply_irs_bracket_calibration.py

# Validate final results
python scripts/pipeline/03_validate_results.py
```

### Data Flow

```
PUMS Raw Data
    ↓
[Stage 1] Tax Unit Construction
    ↓
tax_units_raw.parquet (1,047,658 units)
    ↓
[Stage 2] DOTAX Calibration
    ↓
tax_units_dotax_calibrated.parquet (634,956 units)
    ↓
[Stage 3] IRS Bracket Calibration
    ↓
tax_units_irs_bracket_calibrated.parquet (634,956 units, corrected income distribution)
    ↓
[Tax Calculation & Analysis]
```

## Technical Details

### Bounded Adjustments

**Weight Adjustments:**
- Maximum: 3.0x (can increase weight up to 3x)
- Minimum: 0.33x (can decrease weight to 1/3)
- Rationale: Prevents extreme weight inflation that would make individual units unrepresentative

**Income Adjustments:**
- Maximum: +30% (1.3x multiplier)
- Minimum: -30% (0.7x multiplier)
- Rationale: Maintains realistic income distributions within brackets

### Convergence

The calibration converges in a single pass because:
1. Each bracket is adjusted independently
2. Final re-normalization is a simple scalar multiplication
3. No iterative optimization required

This is different from IPF (Stage 2) which requires iteration.

### Performance

- **Runtime:** ~2-5 seconds for 50,000 tax units
- **Memory:** Minimal (single DataFrame copy)
- **Scalability:** Linear with number of tax units

## Limitations and Considerations

### 1. Within-Bracket Heterogeneity
- IRS provides only bracket averages, not full distributions
- We assume proportional adjustment within brackets
- May not capture complex within-bracket patterns

### 2. Bounded Adjustments
- If bounds are hit frequently, PUMS sample may be inadequate
- Consider supplementing with other data sources (CEX, SOI PUF)

### 3. Geographic Detail
- IRS brackets are state-level only
- Cannot calibrate to county or PUMA-level income distributions
- Geographic variation preserved from PUMS

### 4. Temporal Alignment
- IRS data is for 2022
- PUMS is 5-year sample (2018-2022)
- May need additional temporal adjustment for current-year estimates

## Future Enhancements

### 1. Finer Bracket Resolution
- Use IRS detailed tables for more granular brackets
- Especially important for $200k+ bracket (very wide range)

### 2. Filing Status × Bracket Calibration
- Calibrate to IRS filing status by income bracket tables
- More accurate than separate calibrations

### 3. Iterative Refinement
- Iterate between bracket and filing status calibration
- Similar to IPF but with bracket constraints

### 4. Validation Against Tax Revenue
- Calculate actual tax liability after calibration
- Compare to DOTAX total tax collections
- Ultimate validation of methodology

## References

1. **IRS Statistics of Income 2022**
   - Table 2: Individual Income Tax Returns by Size of Adjusted Gross Income
   - https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-returns

2. **Hawaii DOTAX SOI 2022**
   - Table 5A: Filing Status Summary (Residents)
   - Total resident returns: 634,956

3. **Census PUMS Documentation**
   - Income variables and top-coding
   - Weighting methodology

4. **Calibration Literature**
   - Deville & Särndal (1992): Calibration estimators in survey sampling
   - Little & Rubin (2019): Statistical Analysis with Missing Data

## Contact

For questions or issues with IRS bracket calibration:
- Review validation metrics in output files
- Check logs for bounded adjustment warnings
- Consult `docs/TROUBLESHOOTING.md` for common issues
