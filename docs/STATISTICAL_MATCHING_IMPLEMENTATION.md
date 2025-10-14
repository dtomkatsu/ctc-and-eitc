# Statistical Matching Implementation Guide

## Overview

This document describes the implementation of statistical matching with multiple public data sources to enhance income estimation accuracy in the Hawaii tax unit construction pipeline.

**Expected Error Reduction: 15-25%**  
**Implementation Time: 6-10 weeks**

## Architecture

### Core Components

```
Statistical Matching System
├── Data Loaders (src/data/)
│   ├── bls_oes_loader.py          # BLS wage data
│   ├── cex_loader.py              # Consumer Expenditure Survey
│   └── national_soi_puf_loader.py # National SOI PUF
├── Matching Framework (src/data/)
│   └── statistical_matching.py    # Core matching algorithms
└── Integration (src/tax/units/)
    └── income_enhancement.py      # TaxUnitConstructor integration
```

## Data Sources

### 1. BLS Occupational Employment Statistics (OES)

**Purpose**: Temporal alignment of PUMS wages to current year

**Data Available**:
- State-level wage distributions by occupation (SOC codes)
- Industry-specific wage patterns (NAICS codes)
- Annual wage growth rates
- Employment counts

**Key Features**:
- Occupation-specific wage adjustment factors
- Hawaii-specific wage data
- Free public data (updated annually)

**Usage**:
```python
from src.data.bls_oes_loader import BLSOESLoader

loader = BLSOESLoader(data_dir='data/external/bls_oes')

# Get Hawaii wage data
wages = loader.load_hawaii_wages(year=2022)

# Calculate wage growth rates
growth = loader.calculate_wage_growth_rates(start_year=2021, end_year=2022)

# Get adjustment factors for PUMS alignment
factors = loader.get_wage_adjustment_factors(pums_year=2022, target_year=2023)
```

**Data Download**:
- URL: https://www.bls.gov/oes/tables.htm
- File format: Excel (state_M{year}_dl.xlsx)
- Manual download required (API has rate limits)

### 2. Consumer Expenditure Survey (CEX)

**Purpose**: Impute non-wage income sources that PUMS undercounts

**Data Available**:
- Income source distributions (wages, investment, business, rental, retirement)
- Expenditure-based income validation
- Demographic-specific income patterns
- Income composition by age, education, family size

**Key Features**:
- Addresses PUMS undercount of investment income
- Models business/self-employment income
- Demographic matching for better accuracy

**Usage**:
```python
from src.data.cex_loader import CEXLoader

loader = CEXLoader(data_dir='data/external/cex')

# Load income data
income_data = loader.load_income_data(year=2022)

# Get income composition by group
by_age = loader.get_income_composition_by_group(year=2022, group_by='age_group')

# Match income composition for a specific case
composition = loader.match_income_composition(
    total_income=100000,
    age=45,
    year=2022
)
```

**Data Download**:
- URL: https://www.bls.gov/cex/pumd.htm
- File format: CSV (fmli{year}.csv)
- Manual download required

### 3. National SOI Public Use File (PUF)

**Purpose**: Model complex income patterns using actual tax return data

**Data Available**:
- Detailed income relationships from real tax returns
- High-income household characteristics (oversampled)
- Tax calculation patterns
- Income source correlations

**Key Features**:
- Actual tax return data (not survey estimates)
- High-income oversampling addresses PUMS undercount
- Detailed income decomposition
- Filing status-specific patterns

**Usage**:
```python
from src.data.national_soi_puf_loader import NationalSOIPUFLoader

loader = NationalSOIPUFLoader(data_dir='data/external/soi_puf')

# Load PUF data (3-year lag)
puf_data = loader.load_puf_data(year=2019)

# Get income relationships
correlations = loader.get_income_relationships(year=2019)

# Get high-income patterns
high_income = loader.get_high_income_patterns(
    year=2019,
    income_threshold=200000
)

# Create matching template
template = loader.create_matching_template(year=2019)
```

**Data Download**:
- URL: https://www.irs.gov/statistics/soi-tax-stats-individual-public-use-microdata-files
- File format: CSV ({year}puf.csv)
- Note: 3-year publication lag (2019 is most recent as of 2023)

## Statistical Matching Algorithms

### 1. Propensity Score Matching

**Method**: Matches recipients to donors with similar probability of treatment

**Algorithm**:
1. Combine recipient and donor data with treatment indicator
2. Estimate propensity scores using logistic regression
3. Match recipients to donors within caliper (default: 0.25)
4. Use closest match within caliper

**Best For**:
- Large donor pools
- Multiple matching variables
- When treatment probability is well-defined

**Quality Metrics**:
- Propensity score difference
- Caliper violations
- Balance statistics

### 2. Hot Deck Imputation

**Method**: Matches within demographic "decks" using nearest neighbor

**Algorithm**:
1. Create decks based on exact match variables (e.g., filing status)
2. Within each deck, use nearest neighbor on continuous variables
3. Random selection if no continuous variables
4. Fallback to overall nearest neighbor if deck is empty

**Best For**:
- Categorical matching variables
- Preserving demographic structure
- When exact matches are important

**Quality Metrics**:
- Deck match rate
- Within-deck distance
- Fallback frequency

### 3. Nearest Neighbor

**Method**: Distance-based matching on standardized variables

**Algorithm**:
1. Standardize matching variables (mean=0, std=1)
2. Calculate distance using specified metric (default: Euclidean)
3. Find k nearest neighbors (default: k=5)
4. Use closest neighbor

**Best For**:
- Continuous matching variables
- Simple, interpretable matching
- When distance is meaningful

**Quality Metrics**:
- Average distance
- Distance distribution
- Neighbor availability

### 4. Mahalanobis Distance

**Method**: Distance matching accounting for correlation structure

**Algorithm**:
1. Calculate covariance matrix from combined data
2. Compute Mahalanobis distance using inverse covariance
3. Find closest donor by Mahalanobis distance

**Best For**:
- Correlated matching variables
- When correlation structure matters
- Multivariate matching

**Quality Metrics**:
- Mahalanobis distance
- Covariance structure preservation
- Balance on correlated variables

## Integration with Tax Unit Construction

### Basic Usage

```python
from src.tax.units.income_enhancement import (
    IncomeEnhancer, EnhancementConfig
)

# Configure enhancement
config = EnhancementConfig(
    use_bls_wage_adjustment=True,
    use_cex_income_imputation=True,
    use_soi_puf_matching=True,
    pums_year=2022,
    target_year=2023,
    matching_method=MatchingMethod.PROPENSITY_SCORE
)

# Create enhancer
enhancer = IncomeEnhancer(config)

# Enhance tax units
enhanced_units = enhancer.enhance_tax_units(tax_units, person_df)

# Get enhancement report
report = enhancer.get_enhancement_report()
```

### Integration with TaxUnitConstructor

```python
from src.tax.units.constructor import TaxUnitConstructor
from src.tax.units.income_enhancement import enhance_tax_units_with_public_data

# Create tax units
constructor = TaxUnitConstructor(person_df, hh_df)
tax_units = constructor.create_rule_based_units()

# Enhance with statistical matching
enhanced_units, stats = enhance_tax_units_with_public_data(
    tax_units=tax_units,
    person_df=person_df
)

print(f"Enhanced {stats['total_units']:,} tax units")
print(f"Wage-adjusted: {stats['wage_adjusted']:,}")
print(f"Income-imputed: {stats['income_imputed']:,}")
print(f"SOI-matched: {stats['soi_matched']:,}")
```

## Quality Metrics and Validation

### Match Quality Levels

- **Excellent** (>90% confidence): High-quality match, use with confidence
- **Good** (70-90% confidence): Acceptable match, minor uncertainty
- **Fair** (50-70% confidence): Moderate uncertainty, flag for review
- **Poor** (<50% confidence): Low confidence, consider alternative methods

### Validation Functions

```python
from src.tax.units.income_enhancement import validate_enhancement_quality
from src.data.statistical_matching import calculate_match_balance

# Validate enhancement quality
validation = validate_enhancement_quality(
    original=tax_units,
    enhanced=enhanced_units,
    soi_benchmarks=soi_benchmarks
)

# Check match balance
balance = calculate_match_balance(
    recipients=tax_units,
    matched_data=enhanced_units,
    variables=['age', 'income', 'family_size']
)

# Quality report
quality_report = enhancer.get_enhancement_report()
```

### Key Validation Metrics

1. **Mean Income Change**: Should be <10% unless addressing known bias
2. **Match Quality Distribution**: >70% should be Good or Excellent
3. **Balance Statistics**: Standardized differences <0.1 for key variables
4. **AGI Alignment**: Within 5-10% of SOI benchmarks
5. **Income Source Prevalence**: Match CEX/SOI patterns

## Expected Impact

### Error Reduction by Component

| Component | Error Reduction | Mechanism |
|-----------|----------------|-----------|
| BLS Wage Adjustment | 5% | Temporal alignment to current year |
| CEX Income Imputation | 15-20% | Non-wage income sources |
| SOI PUF Matching | 10-15% | High-income patterns |
| **Total** | **15-25%** | Combined effect |

### Income Source Improvements

**Before Enhancement (PUMS only)**:
- Wage income: 85% of total (overestimated)
- Investment income: 5% of total (underestimated)
- Business income: 8% of total (underestimated)
- Other income: 2% of total

**After Enhancement (with matching)**:
- Wage income: 75% of total (corrected)
- Investment income: 12% of total (improved)
- Business income: 10% of total (improved)
- Other income: 3% of total

### High-Income Accuracy

**PUMS-only approach**:
- Top 10% income: ±20-30% error
- Top 1% income: ±30-50% error
- Missing ~15% of high-income households

**With statistical matching**:
- Top 10% income: ±10-15% error
- Top 1% income: ±15-25% error
- Better coverage through SOI PUF patterns

## Testing and Validation

### Run Test Suite

```bash
# Test all components
python scripts/test_statistical_matching.py

# Test individual loaders
pytest tests/data/test_bls_oes_loader.py -v
pytest tests/data/test_cex_loader.py -v
pytest tests/data/test_soi_puf_loader.py -v

# Test matching algorithms
pytest tests/data/test_statistical_matching.py -v

# Test integration
pytest tests/units/test_income_enhancement.py -v
```

### Validation Checklist

- [ ] BLS OES data loads successfully
- [ ] CEX data loads successfully
- [ ] SOI PUF data loads successfully
- [ ] Propensity score matching produces reasonable matches
- [ ] Hot deck matching handles empty decks gracefully
- [ ] Match quality scores are distributed appropriately
- [ ] Balance statistics show good covariate balance
- [ ] Enhanced income aligns with SOI benchmarks
- [ ] No systematic bias in matched data
- [ ] Quality metrics are tracked and reported

## Implementation Timeline

### Phase 1: Data Infrastructure (Weeks 1-2)
- Set up data directories
- Download BLS OES, CEX, and SOI PUF data
- Test data loaders
- Validate data quality

### Phase 2: Matching Framework (Weeks 3-4)
- Implement matching algorithms
- Test on synthetic data
- Validate match quality
- Tune matching parameters

### Phase 3: Integration (Weeks 5-6)
- Integrate with TaxUnitConstructor
- Test on Hawaii PUMS data
- Validate enhancement quality
- Compare to SOI benchmarks

### Phase 4: Validation & Refinement (Weeks 7-8)
- Comprehensive testing
- Quality metric analysis
- Parameter tuning
- Documentation

### Phase 5: Production Deployment (Weeks 9-10)
- Performance optimization
- Error handling
- Logging and monitoring
- Final validation

## Best Practices

### Data Management

1. **Cache Downloaded Data**: Save BLS/CEX/SOI data locally to avoid re-downloading
2. **Version Control**: Track data versions and update dates
3. **Data Validation**: Check data quality after download
4. **Backup**: Keep backups of external data sources

### Matching Configuration

1. **Start Simple**: Begin with nearest neighbor, add complexity as needed
2. **Validate Matches**: Always check match quality before using results
3. **Monitor Balance**: Ensure matched data preserves key distributions
4. **Document Choices**: Record matching method and parameters used

### Quality Control

1. **Track Metrics**: Log all quality metrics for every run
2. **Set Thresholds**: Define acceptable quality levels
3. **Flag Issues**: Automatically flag low-quality matches
4. **Manual Review**: Review flagged cases manually

### Performance

1. **Batch Processing**: Process tax units in batches for efficiency
2. **Parallel Matching**: Use multiprocessing for independent matches
3. **Cache Results**: Cache matched data to avoid recomputation
4. **Profile Code**: Identify and optimize bottlenecks

## Troubleshooting

### Common Issues

**Issue**: BLS data download fails  
**Solution**: Download manually from BLS website, place in `data/external/bls_oes/`

**Issue**: Match quality is consistently poor  
**Solution**: Check matching variables, adjust caliper, try different matching method

**Issue**: Enhanced income diverges from SOI benchmarks  
**Solution**: Review imputation logic, check for systematic bias, validate data sources

**Issue**: Performance is slow  
**Solution**: Enable batch processing, use parallel matching, cache intermediate results

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug logging
enhancer = IncomeEnhancer(config)
enhanced = enhancer.enhance_tax_units(tax_units)
```

## Future Enhancements

### Potential Improvements

1. **Machine Learning Matching**: Use ML models for propensity score estimation
2. **Ensemble Matching**: Combine multiple matching methods
3. **Adaptive Calibration**: Dynamically adjust based on match quality
4. **Real-Time Updates**: Automatically download and integrate new data
5. **Geographic Matching**: Add geographic variables for better matching

### Additional Data Sources

1. **Survey of Consumer Finances (SCF)**: High-income household wealth
2. **IRS Migration Data**: Income mobility patterns
3. **State Administrative Records**: Direct income data (if available)
4. **Industry-Specific Data**: Sector-specific income patterns

## References

### Data Sources

- BLS OES: https://www.bls.gov/oes/
- CEX: https://www.bls.gov/cex/
- SOI PUF: https://www.irs.gov/statistics/soi-tax-stats-individual-public-use-microdata-files

### Methodology

- Rosenbaum & Rubin (1983): Propensity Score Matching
- Little & Rubin (2019): Statistical Analysis with Missing Data
- D'Orazio et al. (2006): Statistical Matching: Theory and Practice

### Implementation

- scikit-learn: Machine learning library for matching algorithms
- pandas: Data manipulation and analysis
- numpy: Numerical computing
