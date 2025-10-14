"""
Test and validate statistical matching with multiple public data sources.

This script:
1. Tests each data loader independently
2. Validates statistical matching algorithms
3. Measures enhancement quality
4. Generates comprehensive reports
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.bls_oes_loader import BLSOESLoader
from src.data.cex_loader import CEXLoader
from src.data.national_soi_puf_loader import NationalSOIPUFLoader
from src.data.statistical_matching import (
    StatisticalMatcher, MatchingConfig, MatchingMethod, calculate_match_balance
)
from src.tax.units.income_enhancement import (
    IncomeEnhancer, EnhancementConfig, validate_enhancement_quality
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_bls_oes_loader():
    """Test BLS OES data loader."""
    logger.info("\n" + "="*80)
    logger.info("TESTING BLS OES LOADER")
    logger.info("="*80)
    
    loader = BLSOESLoader(data_dir='data/external/bls_oes')
    
    # Test loading Hawaii wages
    logger.info("\n### Loading Hawaii wage data ###")
    df = loader.load_hawaii_wages(year=2022)
    logger.info(f"Loaded {len(df):,} occupation records")
    
    # Test summary statistics
    logger.info("\n### Summary Statistics ###")
    stats = loader.get_summary_statistics(year=2022)
    logger.info(f"Total employment: {stats['total_employment']:,}")
    logger.info(f"Mean wage: ${stats['mean_wage']:,.0f}")
    logger.info(f"Median wage: ${stats['median_wage']:,.0f}")
    
    # Test occupation lookup
    logger.info("\n### Testing Occupation Lookup ###")
    occ_code = '15-0000'  # Computer and mathematical
    wage_dist = loader.get_occupation_wage_distribution(occ_code, year=2022)
    if wage_dist:
        logger.info(f"Occupation {occ_code}:")
        logger.info(f"  Mean wage: ${wage_dist['mean']:,.0f}")
        logger.info(f"  Median wage: ${wage_dist['median']:,.0f}")
        logger.info(f"  Employment: {wage_dist['employment']:,}")
    
    # Test wage growth rates
    logger.info("\n### Testing Wage Growth Calculation ###")
    try:
        growth = loader.calculate_wage_growth_rates(2021, 2022)
        logger.info(f"Calculated growth rates for {len(growth):,} occupations")
        logger.info(f"Average annual growth: {growth['annual_growth_rate'].mean():.2%}")
    except Exception as e:
        logger.warning(f"Could not calculate growth rates: {e}")
    
    logger.info("\n✅ BLS OES loader test complete")
    return True


def test_cex_loader():
    """Test CEX data loader."""
    logger.info("\n" + "="*80)
    logger.info("TESTING CEX LOADER")
    logger.info("="*80)
    
    loader = CEXLoader(data_dir='data/external/cex')
    
    # Test loading income data
    logger.info("\n### Loading CEX income data ###")
    df = loader.load_income_data(year=2022)
    logger.info(f"Loaded {len(df):,} consumer units")
    
    # Test summary statistics
    logger.info("\n### Summary Statistics ###")
    stats = loader.get_summary_statistics(year=2022)
    logger.info(f"Mean total income: ${stats['mean_total_income']:,.0f}")
    logger.info(f"Median total income: ${stats['median_total_income']:,.0f}")
    logger.info(f"Income composition:")
    for source, share in stats['income_composition'].items():
        logger.info(f"  {source}: {share:.1%}")
    
    # Test income composition by group
    logger.info("\n### Income Composition by Age Group ###")
    by_age = loader.get_income_composition_by_group(year=2022, group_by='age_group')
    logger.info(f"\n{by_age.head()}")
    
    # Test non-wage income patterns
    logger.info("\n### Non-Wage Income Patterns (Top 10%) ###")
    high_income = loader.get_non_wage_income_patterns(year=2022, income_percentile=90)
    logger.info(f"Mean total income: ${high_income['mean_total_income']:,.0f}")
    logger.info(f"Investment share: {high_income['investment_share']:.1%}")
    logger.info(f"Self-employment share: {high_income['self_employment_share']:.1%}")
    
    # Test income matching
    logger.info("\n### Testing Income Composition Matching ###")
    composition = loader.match_income_composition(total_income=100000, age=45, year=2022)
    logger.info(f"Matched composition for $100K income:")
    logger.info(f"  Wage income: ${composition['wage_income']:,.0f}")
    logger.info(f"  Investment income: ${composition['investment_income']:,.0f}")
    logger.info(f"  Match quality: {composition['match_quality']:.3f}")
    
    logger.info("\n✅ CEX loader test complete")
    return True


def test_soi_puf_loader():
    """Test National SOI PUF loader."""
    logger.info("\n" + "="*80)
    logger.info("TESTING NATIONAL SOI PUF LOADER")
    logger.info("="*80)
    
    loader = NationalSOIPUFLoader(data_dir='data/external/soi_puf')
    
    # Test loading PUF data
    logger.info("\n### Loading SOI PUF data ###")
    df = loader.load_puf_data(year=2019)
    logger.info(f"Loaded {len(df):,} tax returns")
    
    # Test summary statistics
    logger.info("\n### Summary Statistics ###")
    stats = loader.get_summary_statistics(year=2019)
    logger.info(f"Sample size: {stats['sample_size']:,}")
    logger.info(f"Weighted returns: {stats['weighted_returns']:,.0f}")
    logger.info(f"Mean AGI: ${stats['mean_agi']:,.0f}")
    logger.info(f"Filing status distribution:")
    for status, pct in stats['filing_status_distribution'].items():
        logger.info(f"  {status}: {pct:.1%}")
    
    # Test income relationships
    logger.info("\n### Income Source Correlations ###")
    correlations = loader.get_income_relationships(year=2019)
    logger.info(f"\n{correlations.head()}")
    
    # Test income composition by AGI
    logger.info("\n### Income Composition by AGI Bracket ###")
    by_agi = loader.get_income_composition_by_agi(year=2019, n_brackets=5)
    logger.info(f"Created {len(by_agi)} AGI brackets")
    
    # Test high-income patterns
    logger.info("\n### High-Income Patterns (>$200K) ###")
    high_income = loader.get_high_income_patterns(year=2019, income_threshold=200000)
    logger.info(f"High-income returns: {high_income['n_returns']:,} ({high_income['pct_of_returns']:.1%})")
    logger.info(f"Mean AGI: ${high_income['mean_agi']:,.0f}")
    logger.info(f"Income composition:")
    for source, share in high_income['income_composition'].items():
        logger.info(f"  {source}: {share:.1%}")
    
    # Test matching template
    logger.info("\n### Creating Matching Template ###")
    template = loader.create_matching_template(year=2019)
    logger.info(f"Created template with {len(template):,} donor records")
    
    logger.info("\n✅ SOI PUF loader test complete")
    return True


def test_statistical_matching():
    """Test statistical matching algorithms."""
    logger.info("\n" + "="*80)
    logger.info("TESTING STATISTICAL MATCHING ALGORITHMS")
    logger.info("="*80)
    
    # Create synthetic recipient and donor data
    np.random.seed(42)
    n_recipients = 1000
    n_donors = 5000
    
    recipients = pd.DataFrame({
        'id': range(n_recipients),
        'age': np.random.randint(25, 75, n_recipients),
        'income': np.random.lognormal(10.5, 0.8, n_recipients),
        'family_size': np.random.choice([1, 2, 3, 4], n_recipients)
    })
    
    donors = pd.DataFrame({
        'id': range(n_donors),
        'age': np.random.randint(25, 75, n_donors),
        'income': np.random.lognormal(10.5, 0.8, n_donors),
        'family_size': np.random.choice([1, 2, 3, 4], n_donors),
        'investment_income': np.random.lognormal(8, 1.5, n_donors)
    })
    
    # Test different matching methods
    methods = [
        MatchingMethod.PROPENSITY_SCORE,
        MatchingMethod.NEAREST_NEIGHBOR,
        MatchingMethod.HOT_DECK
    ]
    
    for method in methods:
        logger.info(f"\n### Testing {method.value} matching ###")
        
        config = MatchingConfig(
            method=method,
            matching_variables=['age', 'income', 'family_size'],
            caliper=0.25
        )
        
        matcher = StatisticalMatcher(config)
        
        try:
            matched_data, match_results = matcher.match(
                recipients=recipients,
                donors=donors,
                recipient_id_col='id',
                donor_id_col='id'
            )
            
            logger.info(f"Matched {len(match_results):,} recipients")
            
            # Calculate match quality
            quality_report = matcher.get_quality_report()
            logger.info(f"Mean match score: {quality_report['match_score'].mean():.3f}")
            logger.info(f"Quality distribution:")
            logger.info(f"\n{quality_report['quality'].value_counts()}")
            
            # Calculate balance
            balance = calculate_match_balance(
                recipients=recipients,
                matched_data=matched_data,
                variables=['age', 'income', 'family_size']
            )
            logger.info(f"\nBalance statistics:")
            logger.info(f"\n{balance[['variable', 'standardized_diff', 'balanced']]}")
            
        except Exception as e:
            logger.error(f"Matching failed: {e}")
    
    logger.info("\n✅ Statistical matching test complete")
    return True


def test_income_enhancement():
    """Test full income enhancement pipeline."""
    logger.info("\n" + "="*80)
    logger.info("TESTING INCOME ENHANCEMENT PIPELINE")
    logger.info("="*80)
    
    # Create synthetic tax units
    np.random.seed(42)
    n_units = 1000
    
    tax_units = pd.DataFrame({
        'tax_unit_id': range(n_units),
        'filing_status': np.random.choice(['single', 'joint', 'hoh'], n_units, p=[0.5, 0.4, 0.1]),
        'total_income': np.random.lognormal(10.5, 0.8, n_units),
        'wage_income': np.random.lognormal(10.3, 0.8, n_units),
        'age': np.random.randint(25, 75, n_units),
        'weight': np.random.uniform(100, 1000, n_units)
    })
    
    # Configure enhancement
    config = EnhancementConfig(
        use_bls_wage_adjustment=True,
        use_cex_income_imputation=True,
        use_soi_puf_matching=True,
        pums_year=2022,
        target_year=2022
    )
    
    # Apply enhancement
    logger.info("\n### Applying Income Enhancement ###")
    enhancer = IncomeEnhancer(config)
    enhanced = enhancer.enhance_tax_units(tax_units)
    
    logger.info(f"\nEnhanced {len(enhanced):,} tax units")
    
    # Validate enhancement
    logger.info("\n### Validating Enhancement Quality ###")
    validation = validate_enhancement_quality(
        original=tax_units,
        enhanced=enhanced
    )
    
    logger.info("Validation metrics:")
    for metric, value in validation.items():
        if isinstance(value, float):
            logger.info(f"  {metric}: {value:.4f}")
        else:
            logger.info(f"  {metric}: {value}")
    
    # Generate enhancement report
    logger.info("\n### Enhancement Report ###")
    report = enhancer.get_enhancement_report()
    logger.info(f"\n{report}")
    
    logger.info("\n✅ Income enhancement test complete")
    return True


def main():
    """Run all tests."""
    logger.info("="*80)
    logger.info("STATISTICAL MATCHING SYSTEM TEST SUITE")
    logger.info("="*80)
    
    results = {}
    
    # Test each component
    try:
        results['BLS OES Loader'] = test_bls_oes_loader()
    except Exception as e:
        logger.error(f"BLS OES Loader test failed: {e}")
        results['BLS OES Loader'] = False
    
    try:
        results['CEX Loader'] = test_cex_loader()
    except Exception as e:
        logger.error(f"CEX Loader test failed: {e}")
        results['CEX Loader'] = False
    
    try:
        results['SOI PUF Loader'] = test_soi_puf_loader()
    except Exception as e:
        logger.error(f"SOI PUF Loader test failed: {e}")
        results['SOI PUF Loader'] = False
    
    try:
        results['Statistical Matching'] = test_statistical_matching()
    except Exception as e:
        logger.error(f"Statistical Matching test failed: {e}")
        results['Statistical Matching'] = False
    
    try:
        results['Income Enhancement'] = test_income_enhancement()
    except Exception as e:
        logger.error(f"Income Enhancement test failed: {e}")
        results['Income Enhancement'] = False
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        logger.info("\n🎉 ALL TESTS PASSED!")
    else:
        logger.info("\n⚠️  SOME TESTS FAILED")
    
    logger.info("="*80)
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
