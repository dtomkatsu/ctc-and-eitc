# Hawaii State-Wide Tax Estimation

This project builds a Hawaii **RESIDENT-ONLY** income tax model that aligns closely with Department of Taxation (DOTAX) Statistics of Income (SOI) benchmarks. We combine:
- **DOTAX SOI administrative tables** as the source of truth for counts, income, deductions, and liabilities.
- **ACS PUMS microdata** to construct detailed tax units (filers, spouses, dependents) and to retain demographic structure.

⚠️ **CRITICAL**: This model estimates **RESIDENT-ONLY** revenue (~91.2% of total). Non-resident revenue (~8.8%) is not captured by PUMS data. All projections must specify resident vs total revenue.

Recent work focuses on calibrating the Hawaii tax model to match DOTAX Table A8 (2022) tax liability benchmarks through systematic calibrations of deductions, income distributions, and filer weights.

## Key Features

- **Resident-focused calibration** to DOTAX 2022 tables (A2, A4, A9) for filing status counts, deductions, and liabilities.
- **PUMS-derived tax unit construction** with repaired filing status logic, dependent assignment, and income calculation.
- **Hawaii tax calculator** (`src/tax/hawaii_calculator.py`) supporting 2022 tax brackets with standard deductions and personal exemptions.
- **Systematic tax calibration pipeline** with six sequential components plus an optional IPF fine-tuning stage:
  1. **Itemized Deduction Reduction** – Corrects deduction overshoot (40-60% reduction)
  2. **Comprehensive Bracket Calibration** – Matches DOTAX filer counts across all brackets using Pareto-based scaling (`src/tax/adjustments/pareto_calibration.py`)
  3. **Income Distribution Calibration** – Adjusts bracket-level income distributions to hit effective rate targets (`src/tax/adjustments/income_distribution_calibrator.py`)
  4. **Weight Calibration (Low/Middle Income)** – Fine-tunes $0-$200k brackets to eliminate filer count residuals (`src/tax/adjustments/comprehensive_weight_calibrator.py`)
  5. **Ultra-High-Income Synthesis** – Redistributes $1M+ filers and adds synthetic $5M-$50M units to fill the Pareto tail (`src/tax/adjustments/ultra_high_income_synthesizer.py`)
  6. **Final Gap Closer** – Applies targeted weight, deduction, and tax adjustments to close the residual gap (`src/tax/adjustments/final_gap_closer.py`)
  
    **Optional:** `apply_ipf_calibration()` fine-tunes filing status weights once structural corrections are in place. In practice we often skip IPF because the structural stages now hit a tighter target (-13% vs -40% for pure IPF), but the tooling remains available for experiments that focus specifically on filing-status margins.
- **Validation tooling** to compare modeled revenue against DOTAX Table A8 and highlight remaining gaps.
- **Model accuracy**: 
  - **Resident Revenue**: $3,298M vs $2,999M target (+10% overestimate, needs recalibration)
  - **Growth Rate**: 7.4% CAGR (too optimistic, target 2-3%)
  - **Act 46 Impact**: -$657M vs -$597M official (-10% error)
  - **Effective Tax Rates**: 93% of brackets within ±1.0pp
  - **Post-Calibration Gap**: -13.0% (after full sequential calibration)

## Current Calibration Status (November 2025)

✅ **Model calibrated using three-scenario approach anchored to FY 2024 actual**:

| Scenario | 2026 Resident | 2026 Total | Growth | Income Adj | Status |
|----------|---------------|------------|--------|------------|--------|
| Conservative | $3,082M | $3,379M | 1.5% | ×0.934 | ✅ |
| **Moderate** | **$3,085M** | **$3,382M** | **2.0%** | **×0.935** | **✅ RECOMMENDED** |
| Aggressive | $3,074M | $3,370M | 2.5% | ×0.932 | ✅ |

**Key Insight**: FY 2025 ($3,288M) is a DOT projection, not actual. Calibration anchored to FY 2024 actual ($3,280M).

See `docs/FINAL_CALIBRATION_APPROACH.md` for complete methodology.

## Hawaii Tax Calibration Pipeline

The system implements a comprehensive four-stage tax calibration pipeline to align modeled tax liability with DOTAX Table A8 benchmarks:

### Stage 1: Itemized Deduction Reduction
**Purpose**: Correct systematic overestimation of itemized deductions

**Implementation** (`src/tax/deductions/itemized_estimator.py`):
- Reduced deduction rates by 40-60% across all income levels
- Base rates: 2-8% of AGI (vs original 5-15%)
- Prevents deduction overshoot that was causing tax under-collection
- Impact: +$150M in tax liability

### Stage 2: Comprehensive Bracket Calibration
**Purpose**: Match DOTAX filer counts across all AGI brackets

**Implementation** (`src/tax/adjustments/pareto_calibration.py`):
- Applies Pareto-based scaling factors to every bracket ($0-$10k through $1M+$)
- Preserves overall filer totals while aligning each bracket to DOTAX targets
- Supports optional synthetic filer creation for extreme high-income cases
- Impact: Exact filer count match across brackets, prerequisite for later stages

### Stage 3: Income Distribution Calibration
**Purpose**: Adjust income distributions within brackets to match target effective rates

**Implementation** (`src/tax/adjustments/income_distribution_calibrator.py`):
- Percentile-based redistribution within AGI brackets
- Systematically shifts incomes to match target effective rates
- Applied to $100k+ brackets
- Achieves 93.3% of brackets within ±1.0pp on effective rates
- Impact: Structural accuracy improved

### Stage 4: Weight Calibration (Low/Middle Income)
**Purpose**: Eliminate filer count residuals in low- and middle-income brackets

**Implementation** (`src/tax/adjustments/comprehensive_weight_calibrator.py`):
- Direct bracket-level weight adjustment for $0-$200k brackets
- Recalculates taxes after each adjustment to keep liabilities synchronized
- Impact: Tight control over filer counts without disturbing high-income calibration

### Stage 5: Ultra-High-Income Synthesis
**Purpose**: Restore the missing Pareto tail for $1M+ filers

**Implementation** (`src/tax/adjustments/ultra_high_income_synthesizer.py`):
- Redistributes a portion of $1M-$5M filer weight to $5M, $10M, $25M, and $50M income tiers
- Preserves the original 1,824 filer count while boosting total $1M+ tax liability
- Impact: Adds ~$400M in modeled taxes, reducing the total gap by ~11 percentage points
- **Current configuration (as of 2025-11-05)**: Manually tuned “Iteration 9 – Final Balance” allocation (MFJ 35%/28%/22%/15%, Single 20%/15%/10%, HoH 4%/3%, MFS 10%/6%). This calibration must be re-run manually whenever new data or updated DOTAX/SOI targets are introduced.

### Stage 6: Final Gap-Closing Adjustments (Hybrid Solution C)
**Purpose**: Apply targeted tweaks after structural corrections

**Implementation** (`src/tax/adjustments/final_gap_closer.py`):
- Step 1: Reduce middle-income weights by 8-10% ($10k-$75k)
- Step 2: Reduce high-income deductions by an additional 15% ($200k+)
- Step 3: Apply calibrated tax multipliers where residual gaps remain
- Impact: Finalizes sequential calibration and prepares for optional IPF fine-tuning (typically skipped)

### Optional: IPF Fine-Tuning (Filing Status)
**Purpose**: Nudge filing-status distributions to DOTAX targets without undoing structural adjustments

**Implementation** (`src/tax/calibration/apply_ipf_calibration`):
- Iterative proportional fitting limited to filing-status margins (`calibrate_filing_status=True`, others False)
- Uses a 2% tolerance and max 20 iterations; max weight change is constrained for stability
- Frequently omitted because the structural pipeline already matches filer counts and improves tax accuracy, but remains available for experiments focused on filing-status precision

### Pipeline Execution

The complete pipeline is implemented in `scripts/regenerate_tax_units.py`:

```python
# 1. Construct tax units from PUMS
tax_units = constructor.create_tax_units()

# 2. Apply itemized deduction reduction
tax_units['total_deductions'] = apply_itemized_deduction_reduction(tax_units)

# 3. Calculate initial taxes
tax_units = calculator.calculate_tax_units_batch(tax_units)

# 4. Apply comprehensive bracket calibration (Pareto)
tax_units = apply_pareto_calibration(
    tax_units,
    threshold=200000,
    calibrate_all_brackets=True,
    add_synthetic=False,
)

# 5. Apply income distribution calibration
tax_units = apply_income_distribution_calibration(
    tax_units,
    threshold=100000,
    recalculate_tax=False,
    method='percentile',
)

# 6. Apply low/middle-income weight calibration
tax_units = apply_comprehensive_weight_calibration(
    tax_units,
    calibrate_all_brackets=False,
)

# 7. Apply ultra-high-income synthesis
tax_units = apply_ultra_high_income_synthesis(tax_units, target_tax_m=663.0)

# 8. Apply final gap-closing adjustments
tax_units = apply_final_gap_closer(tax_units)

# 9. (Optional) Run filing-status-only IPF
# tax_units = apply_ipf_calibration(
#     tax_units,
#     max_iterations=20,
#     tolerance=0.02,
#     calibrate_filer_counts=False,
#     calibrate_tax_totals=False,
#     calibrate_filing_status=True,
# )
```

### Current Model Accuracy

**Effective Tax Rates (vs DOTAX Table A8):**
- 93% of brackets within ±1.0pp (14 of 15 brackets)
- 60% within ±0.5pp (9 of 15 brackets)
- All brackets $0-$500k within ±10% on total tax

**Total Tax Liability:**
- Model (sequential calibration): $2,635M
- DOTAX Target: $3,029M
- Gap: **-13.0%** (down from -40.4% with pure IPF and -23.9% without ultra-high-income synthesis)

**Gap Analysis (Post-Synthesis):**
- $1M+ bracket: -$206M (primary residual gap, driven by missing $50M+ filers)
- $200k-$750k brackets: -$98M (structural deductions and rate gaps)
- Middle-income surplus: +$25M (over-collection partially offsets high-income deficit)

**Root Cause of Remaining Gap:**
PUMS microdata is top-coded around $2M AGI, missing the Pareto tail of ultra-wealthy earners ($10M, $50M, $100M+) who pay disproportionate share of taxes. This is a data limitation, not a model deficiency.

### Validation Scripts

- `scripts/compare_table_a8.py` - Compare model vs DOTax Table A8 by bracket
- `scripts/analyze_remaining_gap.py` - Decompose remaining gap by bracket
- `scripts/regenerate_tax_units.py` - Run complete calibration pipeline

## Methodology: SOI-Primary Hybrid Approach

### Why This Approach?

**Data Quality Hierarchy:**
```
DOTAX SOI (Residents) ⭐⭐⭐⭐⭐  →  Primary source for counts and income
IRS SOI (All Filers)   ⭐⭐⭐⭐⭐  →  Validation and non-resident data
PUMS (Survey Sample)   ⭐⭐      →  Demographics and geography only
```

**Critical Issues with PUMS-Only Approach:**
- Overcounts tax units by **65%** (1,047,658 vs 634,956 actual)
- Underrepresents high-income households by **19%**
- Would cause **40-60% error** in revenue estimates

### How It Works

1. **Tax Unit Construction** (PUMS-based)
   - Identify household relationships and filing status
   - Calculate income from PUMS fields
   - Determine dependents and family structure

2. **SOI Calibration** (DOTAX/IRS-based)
   - Apply filing status-specific weight adjustments
   - Calibrate income distributions to match SOI benchmarks
   - Preserve demographic/geographic detail from PUMS

3. **Result**
   - Accurate tax unit counts matching official data
   - Correct income distributions including high earners
   - Rich demographic detail for policy analysis

### Three-Stage Calibration Process

The system implements a comprehensive three-stage calibration pipeline:

#### **Stage 1: Tax Unit Construction**
- Construct tax units from PUMS household data
- Identify filing status (Single, Joint, HoH, MFS)
- Calculate income from PUMS fields
- Determine dependents and family structure

#### **Stage 2: DOTAX Calibration**
- Calibrate to DOTAX resident totals (634,956 returns)
- Apply filing status-specific weight adjustments
- Use Iterative Proportional Fitting (IPF) for multi-dimensional calibration
- Validate against DOTAX filing status distribution

**Key Adjustments:**
- Single: 0.6634, Joint: 0.5009, HoH: 1.1932, MFS: 0.6094
- Ensures exact match to DOTAX resident counts
- Preserves demographic/geographic detail from PUMS

#### **Stage 3: IRS SOI Bracket Calibration**
- Match IRS SOI income bracket **counts** (scaled to DOTAX total)
- Adjust **average income** within each bracket to match IRS targets
- Bounded adjustments to prevent extreme distortions

**IRS Brackets:**
- $0-25k: 130,000 returns, avg $12,692
- $25-50k: 180,000 returns, avg $35,000
- $50-75k: 140,000 returns, avg $60,000
- $75-100k: 85,000 returns, avg $84,706
- $100-200k: 60,000 returns, avg $140,000
- $200k+: 15,000 returns, avg $400,000

**Process:**
1. Adjust weights to match bracket counts (max 3x adjustment)
2. Re-normalize to maintain DOTAX total
3. Adjust incomes within brackets (max ±30% adjustment)
4. Validate against IRS benchmarks

#### **Stage 4: High-Income Enhancement**
- Generate synthetic high-income records to fill PUMS gap
- Use Pareto distribution fitted to IRS targets
- Match IRS $200k+ bracket count and average income
- Match IRS percentile floors (top 1% starts at $650k)

**The Problem:**
- PUMS undersamples high-income households by 19%
- Survey response bias (wealthy less likely to respond)
- Top-coding and small sample size issues

**The Solution:**
1. Calculate gap between PUMS and IRS high-income counts
2. Generate synthetic records using Pareto distribution
3. Fit to match: avg=$400k, top 1% floor=$650k
4. Re-calibrate to maintain DOTAX total

**Why This Matters:**
- High earners account for 60-70% of tax revenue
- A $500k household pays ~30x more tax than $50k household
- Missing 19% of high earners → 40-60% error in revenue estimates

#### **Stage 5: Income Source Split**
- Split total AGI into component sources by income bracket
- Use IRS SOI Table 2 income source distributions
- Maintain total income consistency

**Income Sources:**
- Wages and salaries
- Dividends
- Interest
- Business income
- Capital gains
- Pensions
- Other income

**Process:**
1. Assign each tax unit to income bracket
2. Apply IRS source percentages for that bracket
3. Split total income proportionally
4. Validate sum equals total AGI

**Why This Matters:**
- Different income sources have different tax treatment
- Capital gains taxed at lower rates than wages
- Investment income affects tax liability calculations
- Essential for accurate revenue estimates

#### **Stage 6: Wage Growth Adjustment (Phase 1)** ⭐ **NEW**
- Adjust 2022 PUMS wage data to 2024 using BLS OES growth rates
- Apply occupation-specific adjustments where available
- Update total income to reflect wage changes

**BLS OES Data Available:**
- 20092024 state-level wage tables (`data/external/bls_oes/state_YYYY*.xls[x]`)
- 577 occupation-level records (latest tables)
- Covers 594,880 workers (2024 release)

**State Wage File Coverage:**
| Year | Filename |
|------|----------|
| 2009 | `state_2009.xls` |
| 2010 | `state_M2010_dl.xls` |
| 2011 | `state_M2011_dl.xls` |
| 2012 | `state_M2012_dl.xls` |
| 2013 | `state_M2013_dl.xls` |
| 2014 | `state_M2014_dl.xlsx` |
| 2015 | `state_M2015_dl.xlsx` |
| 2016 | `state_M2016_dl.xlsx` |
| 2017 | `state_M2017_dl.xlsx` |
| 2018 | `state_M2018_dl.xlsx` |
| 2019 | `state_M2019_dl.xlsx` |
| 2020 | `state_M2020_dl.xlsx` |
| 2021 | `state_M2021_dl.xlsx` |
| 2022 | `state_M2022_dl.xlsx` |
| 2023 | `state_M2023_dl.xlsx` |
| 2024 | `state_M2024_dl.xlsx` |

**Growth Rates (2022 → 2024):**
- Overall: 11.38% (employment-weighted)
- Food service: 25.3%
- Healthcare support: 22.7%
- Arts & entertainment: 49.0%
- Range: -36% to +115% (occupation-specific)

**Process:**
1. Load BLS OES data for 20092024
2. Calculate occupation-specific growth rates (e.g., 20222024)
3. Match PUMS occupation codes to BLS SOC codes
4. Apply adjustments with fallbacks
5. Update total income

**Why This Matters:**
- PUMS data is from 2022, need 2024 values
- Wages grew significantly 2022-2024
- Occupation-specific trends vary widely
- Critical for accurate 2024 tax calculations

## Statistical Matching with Multiple Public Data Sources

To further improve income estimation accuracy (15-25% error reduction), the system integrates multiple high-quality public datasets through statistical matching:

### Data Sources Integrated

1. **BLS Occupational Employment Statistics (OES)**
   - State-level wage distributions by occupation and industry
   - Critical for temporal alignment (adjusting PUMS to current year)
   - Occupation-specific wage growth rates
   - **Use Case**: Adjust 2022 PUMS wages to 2023/2024 using actual wage growth

2. **Consumer Expenditure Survey (CEX)**
   - Income source distributions (wages, investment, business, etc.)
   - Addresses PUMS undercounting of non-wage income
   - Demographic-specific income patterns
   - **Use Case**: Impute investment and business income that PUMS misses

3. **National SOI Public Use File (PUF)**
   - Detailed income relationships from actual tax returns
   - High-income household characteristics
   - Tax pattern templates
   - **Use Case**: Model complex income sources using real tax return patterns

### Statistical Matching Methods

The system implements multiple matching algorithms:

- **Propensity Score Matching**: Matches units with similar probability of treatment
- **Hot Deck Imputation**: Matches within demographic "decks" 
- **Nearest Neighbor**: Distance-based matching on key variables
- **Mahalanobis Distance**: Accounts for correlation structure

### How It Works

```python
from src.tax.units.income_enhancement import IncomeEnhancer, EnhancementConfig

# Configure enhancement
config = EnhancementConfig(
    use_bls_wage_adjustment=True,      # Temporal alignment
    use_cex_income_imputation=True,    # Non-wage income
    use_soi_puf_matching=True,         # Complex income patterns
    pums_year=2022,
    target_year=2023
)

# Apply enhancement
enhancer = IncomeEnhancer(config)
enhanced_units = enhancer.enhance_tax_units(tax_units)
```

### Expected Impact

- **Wage Accuracy**: ±5% improvement through BLS temporal alignment
- **Non-Wage Income**: 15-20% improvement through CEX imputation
- **High-Income Modeling**: 10-15% improvement through SOI PUF matching
- **Overall Error Reduction**: 15-25% compared to PUMS-only approach

### Quality Tracking

Each enhancement includes quality metrics:
- Match quality scores (0-100)
- Confidence levels (Excellent, Good, Fair, Poor)
- Source attribution for imputed values
- Balance statistics to validate matching quality

## Validation System

The project includes comprehensive validation to ensure accurate tax unit construction:

- Validates filing status assignments
- Checks income calculations
- Verifies dependent relationships
- Compares distributions to SOI benchmarks

## Project Structure

```
hawaii-tax-estimation/
├── src/
│   ├── tax/                       # Core tax calculation logic
│   │   ├── units/                 # Tax unit construction
│   │   │   ├── constructor.py     # Main tax unit constructor
│   │   │   ├── dependencies.py    # Dependency determination
│   │   │   ├── income.py          # Income calculations
│   │   │   ├── relationships.py   # Relationship mapping
│   │   │   ├── status/            # Filing status determination
│   │   │   ├── soi_calibration.py # SOI weight calibration
│   │   │   ├── income_enhancement.py # Statistical matching
│   │   │   └── validation/        # Validation logic
│   │   ├── calibration/           # IPF and weight calibration
│   │   ├── brackets/              # Tax bracket definitions
│   │   └── adjustments/           # Tax adjustments
│   ├── data/                      # Data processing and loaders
│   │   ├── pums_loader.py         # PUMS data loader
│   │   └── dotax_soi_parser.py    # DOTAX SOI parser
│   └── analysis/                  # Analysis modules
├── scripts/                       # Organized by function
│   ├── pipeline/                  # Production pipeline (numbered)
│   │   ├── 01_construct_tax_units.py
│   │   ├── 02_apply_soi_calibration.py
│   │   └── 03_validate_results.py
│   ├── analysis/                  # Analysis scripts
│   │   ├── filing_status/         # Filing status analysis
│   │   ├── income/                # Income distribution analysis
│   │   └── validation/            # SOI comparison and validation
│   ├── calibration/               # Calibration testing
│   │   ├── demo_ipf_calibration.py
│   │   └── test_ipf_calibration.py
│   ├── data_prep/                 # Data preparation
│   │   └── download_pums.py
│   └── archived/                  # Old exploratory scripts
├── data/                          # Data storage
│   ├── raw/                       # Raw PUMS and SOI data
│   ├── processed/                 # Processed tax units
│   └── irs_soi/                   # IRS SOI benchmarks
├── tests/                         # Test suite
│   ├── unit/                      # Unit tests
│   ├── integration/               # Integration tests
│   └── validation/                # Validation tests
├── archived/                      # Archived CTC/EITC/district files
├── docs/                          # Documentation
│   ├── IPF_CALIBRATION_GUIDE.md
│   └── PROJECT_REORGANIZATION_PLAN.md
└── config/                        # Configuration files
    └── income_growth.py
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` and add your Census API key:
     ```
     CENSUS_API_KEY=your_api_key_here
     ```
   - You can get a free API key from: https://api.census.gov/data/key_signup.html

4. Download PUMS data:
   ```bash
   python scripts/data_prep/download_pums.py
   ```
   This will download the data to `data/raw/pums/`.

   Alternatively, you can specify options:
   ```bash
   python scripts/data_prep/download_pums.py --year 2022 --state 15 --data-dir data/raw/pums
   ```

## Main Pipeline Scripts

The project is now organized into functional subdirectories for better maintainability:

### Production Pipeline (`scripts/pipeline/`)
Core production scripts numbered for execution order:
1. **01_construct_tax_units.py** - Construct tax units from PUMS data
2. **02_apply_soi_calibration.py** - Apply DOTAX SOI weight calibration (Stage 2)
3. **03_validate_results.py** - Validate results against benchmarks
4. **04_apply_irs_bracket_calibration.py** - Apply IRS SOI bracket calibration (Stage 3)
5. **05_apply_high_income_enhancement.py** - Apply high-income enhancement (Stage 4)
6. **06_apply_income_source_split.py** - Apply income source split (Stage 5)
7. **07_apply_wage_growth_adjustment.py** - Apply wage growth adjustment (Stage 6)

### Analysis Scripts (`scripts/analysis/`)
Organized by topic:
- **filing_status/** - Filing status analysis and validation
- **income/** - Income distribution and high-income analysis
- **validation/** - SOI comparison and validation scripts

### Calibration Scripts (`scripts/calibration/`)
- **demo_ipf_calibration.py** - IPF calibration demonstration
- **test_ipf_calibration.py** - IPF testing on real data
- **demo_irs_bracket_calibration.py** - IRS bracket calibration demonstration
- **demo_high_income_enhancement.py** - High-income enhancement demonstration
- **demo_wage_growth_options.py** - Wage growth adjustment options demonstration

### Data Preparation (`scripts/data_prep/`)
- **download_pums.py** - Download PUMS data from Census Bureau
- **create_official_crosswalk.py** - Create geographic crosswalks

### Running the Pipeline

To run the tax unit construction pipeline:

```bash
# 1. Download the data (if not already done)
python scripts/data_prep/download_pums.py

# 2. Construct tax units (Stage 1)
python scripts/pipeline/01_construct_tax_units.py

# 3. Apply DOTAX SOI calibration (Stage 2)
python scripts/pipeline/02_apply_soi_calibration.py

# 4. Apply IRS bracket calibration (Stage 3)
python scripts/pipeline/04_apply_irs_bracket_calibration.py

# 5. Apply high-income enhancement (Stage 4)
python scripts/pipeline/05_apply_high_income_enhancement.py

# 6. Apply income source split (Stage 5)
python scripts/pipeline/06_apply_income_source_split.py

# 7. Apply wage growth adjustment (Stage 6)
python scripts/pipeline/07_apply_wage_growth_adjustment.py

# 8. Validate results
python scripts/pipeline/03_validate_results.py
```

### Running Analysis Scripts

```bash
# Compare to SOI benchmarks
python scripts/analysis/validation/compare_to_soi.py

# Analyze filing status distribution
python scripts/analysis/filing_status/analyze_filing_status_gaps.py

# Analyze income distribution
python scripts/analysis/income/analyze_income_distribution.py

# Test IPF calibration
python scripts/calibration/demo_ipf_calibration.py

# Demo IRS bracket calibration
python scripts/calibration/demo_irs_bracket_calibration.py

# Demo high-income enhancement
python scripts/calibration/demo_high_income_enhancement.py
```

The results will be saved in the `data/processed/` directory.

## Key Features

- **Tax Unit Construction**: Robust construction of tax units from PUMS data with SOI calibration
- **Data Reconciliation**: Automatic alignment of PUMS data with DOTAX and IRS SOI benchmarks
- **Multiple Filing Statuses**: Support for Single, Head of Household, Married Filing Jointly, and Married Filing Separately with status-specific adjustments
- **Hawaii Income Tax Calculator**: Full implementation of Hawaii state tax brackets (2017-2031)
- **Income Growth Modeling**: Project future tax years with configurable growth rates
- **Tax Scenario Analysis**: Compare tax impacts across different years and bracket structures
- **Hawaii-Specific Rules**: Custom logic for Hawaii's unique family structures and tax laws
- **Data Quality Validation**: Comprehensive checks against official benchmarks
- **Comprehensive Testing**: Extensive test suite including unit, integration, and validation tests

## Quick Start: Tax Calculations

```python
from src.tax.brackets import load_tax_data

# Load Hawaii tax calculator
calculator = load_tax_data()

# Calculate tax for a single filer with $50,000 income in 2024
result = calculator.calculate_tax(50000, 2024, 'single')

print(f"Tax Liability: ${result['tax_liability']:,.2f}")
print(f"Effective Rate: {result['effective_rate']:.2f}%")
```

See [Tax Calculation Guide](docs/TAX_CALCULATION_GUIDE.md) for detailed documentation.

## Data Sources and Quality

### Primary Data Sources

#### 1. DOTAX SOI 2022 (Primary - Residents)
- **Source**: Hawaii Department of Taxation
- **Coverage**: 634,956 tax returns (2022 residents)
- **Quality**: ⭐⭐⭐⭐⭐ Administrative data (actual tax returns)
- **Use**: Primary source for tax unit counts and income distributions
- **Files**: `data/raw/Dotax Soi 2022 - 13A.csv` (Single), `13B.csv` (Joint), `13C.csv` (HoH)

#### 2. IRS SOI 2022 (Validation - All Filers)
- **Source**: Internal Revenue Service
- **Coverage**: 674,660 tax returns (2022, all Hawaii filers)
- **Quality**: ⭐⭐⭐⭐⭐ Administrative data
- **Use**: Validation and non-resident data
- **Difference from DOTAX**: +5.89% (includes non-residents, part-year, military)

#### 3. PUMS Microdata (Supplementary - Demographics)
- **Source**: U.S. Census Bureau (2023 5-Year ACS)
- **Coverage**: ~1.04M weighted tax units (before calibration)
- **Quality**: ⭐⭐ Survey data with sampling issues
- **Use**: Geographic distribution (PUMA/district), demographics, household composition
- **Limitations**: 
  - Overcounts tax units by 65%
  - Underrepresents high-income households by 19%
  - Requires SOI calibration for accuracy

### Data Quality and Reconciliation

**Why DOTAX/IRS SOI is Primary:**
- Administrative data from actual tax returns (not survey estimates)
- Complete coverage of all tax filers
- Accurate income distributions including high earners
- Official source for Hawaii tax statistics

**Why PUMS is Supplementary:**
- Survey data with significant sampling error
- Systematically undercounts high-income households (who pay 60-70% of taxes)
- Overcounts total tax units by 65%
- **However**: Provides rich demographic and geographic detail unavailable in SOI

**Reconciliation Strategy:**
1. Use DOTAX/IRS SOI for total counts and income distributions
2. Use PUMS for demographic/geographic distribution patterns
3. Apply SOI calibration to PUMS weights to align totals
4. Validate results against both DOTAX and IRS SOI benchmarks

## Running Tests

To run the test suite:

```bash
# Run all tests
pytest

# Run smoke tests only
pytest tests/test_smoke.py -v

# Run unit tests for tax unit construction
pytest tests/units/ -v

# Validate against DOTAX benchmarks
pytest tests/validation/test_dotax_comparison.py -v

# Check data reconciliation
pytest tests/validation/test_data_reconciliation.py -v
```

## Limitations and Known Issues

### 1. Data Quality Limitations

**PUMS Data Issues (Addressed by SOI Calibration):**
- ✅ Overcounting by 65% → Fixed with 0.6061 adjustment factor
- ✅ High-income undercount → Addressed with income-bracket-specific calibration
- ⚠️ May still miss some Hawaii-specific filing patterns

**Remaining Uncertainties:**
- Year mismatch: PUMS (2023) vs DOTAX/IRS (2022)
- Non-filers not fully represented in any data source
- Survey sampling error in PUMS demographic data

### 2. Calibration Limitations

**Adjustment Factor Precision:**
- Based on 2022 DOTAX data (634,956 returns)
- May need recalibration as new data becomes available
- Filing status-specific factors are estimates

**High-Income Estimation:**
- Even with calibration, top 1% may have ±15-20% uncertainty
- PUMS fundamentally undersamples very high earners
- Consider using DOTAX detailed records for top earners

### 3. Recommended Improvements (Priority #1)

**To Achieve 30-50% Error Reduction:**
1. **Integrate DOTAX Administrative Records**
   - Use detailed records for top 1% of earners
   - Direct income data instead of survey estimates
   - Expected improvement: 20-30% error reduction

2. **Add DOL Wage Data**
   - Model non-filers more accurately
   - Cross-validate PUMS income estimates
   - Expected improvement: 10-15% error reduction

3. **Enhance High-Income Modeling**
   - Use Pareto distribution for top earners
   - Validate against actual DOTAX millionaire counts
   - Expected improvement: 15-25% error reduction

### 4. Current Error Estimates

**With SOI Calibration (Current):**
- Total tax units: ±5% (aligned to DOTAX)
- Low/middle income (<$100K): ±10-15%
- High income ($100K-$500K): ±15-20%
- Very high income (>$500K): ±20-30%

**Without SOI Calibration (PUMS-only):**
- Total tax units: +65% (severe overcount)
- Revenue estimates: +40-60% (severe overestimate)
- Not recommended for policy analysis

## Usage

### Basic Example (SOI-Calibrated)

```python
from src.tax.units.constructor import TaxUnitConstructor
from src.data.pums_loader import PUMSDataLoader
import pandas as pd

# Load PUMS data
pums_loader = PUMSDataLoader()
person_df, hh_df = pums_loader.load_data()

# Create tax units with SOI calibration (RECOMMENDED)
constructor = TaxUnitConstructor(
    person_df, 
    hh_df,
    use_soi_calibration=True,              # Enable SOI calibration
    soi_calibration_method='filing_status' # Use filing status-specific factors
)
tax_units = constructor.create_rule_based_units()

# Analyze the results
print(f"Created {len(tax_units):,} tax units")
print(f"Weighted total (calibrated): {tax_units['weight'].sum():,.0f} tax units")
print(f"Target (DOTAX): 634,956 tax units")

# View filing status distribution
status_dist = tax_units.groupby('filing_status')['weight'].sum()
print("\nFiling Status Distribution:")
for status, count in status_dist.items():
    pct = (count / status_dist.sum()) * 100
    print(f"  {status}: {count:,.0f} ({pct:.1f}%)")

# Check calibration quality
if 'weight_original' in tax_units.columns:
    original_total = tax_units['weight_original'].sum()
    calibrated_total = tax_units['weight'].sum()
    print(f"\nCalibration Impact:")
    print(f"  Original PUMS total: {original_total:,.0f}")
    print(f"  Calibrated total: {calibrated_total:,.0f}")
    print(f"  Adjustment: {(calibrated_total/original_total - 1)*100:+.1f}%")
```

### Choosing Calibration Method

```python
# Method 1: Overall adjustment (simplest)
constructor = TaxUnitConstructor(
    person_df, hh_df,
    use_soi_calibration=True,
    soi_calibration_method='overall'  # 0.6061 factor for all
)

# Method 2: Filing status-specific (recommended)
constructor = TaxUnitConstructor(
    person_df, hh_df,
    use_soi_calibration=True,
    soi_calibration_method='filing_status'  # Different factor per status
)

# Method 3: Income bracket-specific (most accurate for revenue)
constructor = TaxUnitConstructor(
    person_df, hh_df,
    use_soi_calibration=True,
    soi_calibration_method='income_bracket'  # Addresses high-income undercount
)

# Disable calibration (not recommended - for analysis only)
constructor = TaxUnitConstructor(
    person_df, hh_df,
    use_soi_calibration=False  # Use raw PUMS weights
)
```

### Manual Calibration (Advanced)

```python
from src.tax.units.soi_calibration import calibrate_to_soi_benchmarks, load_dotax_benchmarks

# Load benchmarks
dotax_benchmarks = load_dotax_benchmarks()

# Create tax units without automatic calibration
constructor = TaxUnitConstructor(person_df, hh_df, use_soi_calibration=False)
tax_units = constructor.create_rule_based_units()

# Apply custom calibration
calibrated_units = calibrate_to_soi_benchmarks(
    tax_units,
    dotax_benchmarks=dotax_benchmarks,
    method='income_bracket'
)

print(f"Calibrated to DOTAX target: {dotax_benchmarks['total_returns']:,} returns")
```

## Data Reconciliation and Adjustments

### PUMS Data Adjustments

Our analysis revealed significant discrepancies between PUMS data and official tax filing data that require adjustments:

1. **Overall Adjustment Factor**: 0.6061
   - PUMS raw data overcounts tax units by approximately 65% compared to DOTAX data
   - **All PUMS weights should be multiplied by 0.6061** to align with official tax return counts

2. **Filing Status-Specific Adjustments**:
   | Filing Status | Adjustment Factor | Notes |
   |---------------|-------------------|-------|
   | Single | 0.6634 | PUMS overcounts single filers |
   | Married Filing Jointly | 0.5009 | PUMS significantly overcounts joint filers |
   | Head of Household | 1.1932 | PUMS undercounts HoH filers |
   | Married Filing Separately | 0.6094 | PUMS overcounts MFS filers |

3. **Income Adjustments**:
   - PUMS average income is 19% lower than IRS SOI data
   - Total PUMS income is 43% higher than IRS SOI due to overcounting
   - Income distributions are calibrated to match SOI percentiles

### Data Source Reconciliation

#### DOTAX vs IRS SOI
- **DOTAX**: 634,956 returns (2022, residents only)
- **IRS SOI**: 674,660 returns (2022, includes all filers)
- **Difference**: 5.89% fewer returns in DOTAX (expected due to non-resident filers in IRS data)

#### PUMS vs Official Data
| Metric | PUMS (Raw) | DOTAX (2022) | IRS SOI (2022) |
|--------|------------|---------------|----------------|
| Total Returns | 1,047,658 | 634,956 | 674,660 |
| Single Filers | 52.9% | 55.3% | 51.7% |
| Joint Filers | 41.2% | 34.1% | 35.1% |
| Head of Household | 5.4% | 10.6% | 10.4% |
| MFS | 2.8% | 2.9% | 2.7% |

### Income Growth Methodology

1. **Base Year**: 2022 (DOTAX/IRS SOI data)
2. **Projection Method**:
   - Apply annual growth rates by income bracket
   - Different growth rates for different income percentiles
   - Adjust for inflation and real income growth
3. **Validation**:
   - Compare with BLS wage growth data
   - Validate against IRS SOI historical trends
   - Check against Hawaii-specific economic indicators

## Understanding PUMS Weights and Sample Scaling

PUMS (Public Use Microdata Sample) data represents a sample of the population, not the complete count. Each record in the sample has an associated weight that indicates how many people/households in the population that record represents.

### Key Points About Weights:
- **Household Weights (WGTP)**: Each household in the sample has a weight indicating how many similar households it represents in the population.
- **Sample vs. Population**: The raw count of tax units in our sample (e.g., ~47,000) is much smaller than the actual population count because it's just a sample.
- **Weighted Analysis**: To get population-level estimates, always use the weight column when performing calculations.
- **Adjustment Factors**: In addition to standard PUMS weights, apply the 0.6061 adjustment factor to align with official tax return counts.

### Example: Calculating Total Tax Units
To get the estimated total number of tax units in the population:

```python
import pandas as pd

# Load the tax units data
tax_units = pd.read_parquet('data/processed/tax_units_rule_based.parquet')

# Calculate weighted total
total_weighted_units = tax_units['weight'].sum()
print(f"Estimated total tax units in population: {total_weighted_units:,.0f}")
```

## Tax Unit Construction

This project constructs tax units from PUMS (Public Use Microdata Sample) data to model Hawaii's tax filing units. The tax unit construction process involves several key steps:

### 1. Data Preprocessing
- Person and household data is loaded from PUMS
- Each person is assigned a unique ID based on household (SERIALNO) and person number (SPORDER)
- Adults (age 18+) are identified and flagged
- Household data is merged with person records

### 2. Household Processing
For each household, the system identifies and creates different types of tax units in this priority order:

1. **Married Filing Jointly**
   - Identifies married couples living together
   - Combines their incomes and dependents into a single tax unit
   - Applies joint filing rules and income thresholds

2. **Married Filing Separately**
   - Some married couples may choose to file separately
   - Determined based on income differences, self-employment status, and other factors
   - Each spouse gets their own tax unit with their own dependents

3. **Head of Household**
   - Single adults with qualifying dependents
   - Must provide more than half the cost of maintaining the household
   - More favorable tax rates than single filers

4. **Single Filers**
   - Adults not in any other category
   - May or may not have dependents

### 3. Dependent Assignment
- Dependents are assigned to the most appropriate tax unit
- Qualifying dependents must be:
  - Under age 19 (or 24 if a student)
  - Related to the filer
  - Not filing their own return
- Special rules apply for qualifying relatives and children of divorced/separated parents

### 4. Income Calculation
- Combines all income sources for each tax unit
- Adjusts for inflation using ADJINC factor
- Handles various income types (wages, self-employment, investments, etc.)
- Calculates adjusted gross income (AGI)

### 5. Validation
- Ensures all household members are properly assigned
- Validates income calculations
- Checks for duplicate or missing information
- Verifies dependent relationships

### Output
- The final output is a DataFrame containing all tax units with their attributes:
  - Filing status
  - Income
  - Number of dependents
  - Dependents list
  - Household ID
  - Weights for population scaling

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
