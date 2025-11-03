#!/usr/bin/env python3
"""
Validate Refined Model Against Historical Data and Review Capital Gains Treatment

This script:
1. Compares refined model against 2018-2021 historical DOTAX data
2. Analyzes capital gains trends and realism
3. Validates capital gains as % of AGI against historical patterns
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_historical_dotax_data() -> pd.DataFrame:
    """Load 2018-2021 historical DOTAX data."""
    project_root = Path(__file__).parent.parent.parent
    file_path = project_root / 'data' / 'processed' / 'dotax_historical' / 'dotax_historical_filing_status_clean.csv'
    
    df = pd.read_csv(file_path)
    logger.info(f"Loaded historical DOTAX data for years: {sorted(df['year'].unique())}")
    return df


def load_refined_model() -> pd.DataFrame:
    """Load the refined model data."""
    project_root = Path(__file__).parent.parent.parent
    refined_files = sorted((project_root / 'data' / 'processed').glob('tax_units_refined_hoh_*.parquet'))
    
    if not refined_files:
        raise FileNotFoundError("No refined model files found")
    
    file_path = refined_files[-1]
    logger.info(f"Loading refined model: {file_path.name}")
    
    df = pd.read_parquet(file_path)
    return df


def analyze_historical_trends(historical_df: pd.DataFrame) -> dict:
    """Analyze historical trends from DOTAX data."""
    
    # Calculate year-over-year growth
    trends = {}
    
    for filing_status in historical_df['filing_status'].unique():
        fs_data = historical_df[historical_df['filing_status'] == filing_status].sort_values('year')
        
        if len(fs_data) < 2:
            continue
        
        # Calculate CAGR for 2018-2021
        first = fs_data.iloc[0]
        last = fs_data.iloc[-1]
        years = last['year'] - first['year']
        
        if years > 0:
            trends[filing_status] = {
                'returns_cagr': ((last['num_returns'] / first['num_returns']) ** (1/years) - 1) * 100,
                'agi_cagr': ((last['agi_net_millions'] / first['agi_net_millions']) ** (1/years) - 1) * 100,
                'tax_cagr': ((last['tax_after_credits_millions'] / first['tax_after_credits_millions']) ** (1/years) - 1) * 100,
                'effective_rate_2021': last['effective_rate_after_credits'],
                'avg_agi_2021': last['avg_agi_per_return'],
            }
    
    # Overall trends
    overall_first = historical_df[historical_df['year'] == historical_df['year'].min()].groupby('year').sum().iloc[0]
    overall_last = historical_df[historical_df['year'] == historical_df['year'].max()].groupby('year').sum().iloc[0]
    years = historical_df['year'].max() - historical_df['year'].min()
    
    trends['OVERALL'] = {
        'returns_cagr': ((overall_last['num_returns'] / overall_first['num_returns']) ** (1/years) - 1) * 100,
        'agi_cagr': ((overall_last['agi_net_millions'] / overall_first['agi_net_millions']) ** (1/years) - 1) * 100,
        'tax_cagr': ((overall_last['tax_after_credits_millions'] / overall_first['tax_after_credits_millions']) ** (1/years) - 1) * 100,
    }
    
    return trends


def analyze_capital_gains(model_df: pd.DataFrame) -> dict:
    """Analyze capital gains in the model."""
    
    # Check if capital gains columns exist
    if 'capital_gains' not in model_df.columns:
        logger.warning("No capital_gains column found in model")
        return {}
    
    # Calculate capital gains statistics
    total_cap_gains = (model_df['capital_gains'] * model_df['weight']).sum()
    total_agi = (model_df['agi'] * model_df['weight']).sum()
    
    # By filing status
    cap_gains_by_status = {}
    
    for status in model_df['filing_status'].unique():
        status_data = model_df[model_df['filing_status'] == status]
        
        status_cap_gains = (status_data['capital_gains'] * status_data['weight']).sum()
        status_agi = (status_data['agi'] * status_data['weight']).sum()
        status_returns_with_gains = status_data[status_data['capital_gains'] > 0]['weight'].sum()
        status_total_returns = status_data['weight'].sum()
        
        cap_gains_by_status[status] = {
            'total_capital_gains_millions': status_cap_gains / 1_000_000,
            'cap_gains_pct_of_agi': (status_cap_gains / status_agi * 100) if status_agi > 0 else 0,
            'pct_with_cap_gains': (status_returns_with_gains / status_total_returns * 100) if status_total_returns > 0 else 0,
            'avg_cap_gains_if_any': status_cap_gains / status_returns_with_gains if status_returns_with_gains > 0 else 0,
        }
    
    # Overall statistics
    overall_stats = {
        'total_capital_gains_millions': total_cap_gains / 1_000_000,
        'cap_gains_pct_of_agi': (total_cap_gains / total_agi * 100) if total_agi > 0 else 0,
        'pct_with_cap_gains': (model_df[model_df['capital_gains'] > 0]['weight'].sum() / model_df['weight'].sum() * 100),
    }
    
    # Income distribution of capital gains
    income_brackets = [0, 50000, 100000, 200000, 500000, 1000000, float('inf')]
    cap_gains_by_income = []
    
    for i in range(len(income_brackets) - 1):
        bracket_mask = (model_df['agi'] >= income_brackets[i]) & (model_df['agi'] < income_brackets[i+1])
        bracket_data = model_df[bracket_mask]
        
        if len(bracket_data) > 0:
            bracket_cap_gains = (bracket_data['capital_gains'] * bracket_data['weight']).sum()
            bracket_agi = (bracket_data['agi'] * bracket_data['weight']).sum()
            
            cap_gains_by_income.append({
                'income_bracket': f"${income_brackets[i]:,.0f}-${income_brackets[i+1]:,.0f}" if income_brackets[i+1] != float('inf') else f">${income_brackets[i]:,.0f}",
                'capital_gains_millions': bracket_cap_gains / 1_000_000,
                'cap_gains_pct_of_agi': (bracket_cap_gains / bracket_agi * 100) if bracket_agi > 0 else 0,
                'pct_of_total_cap_gains': (bracket_cap_gains / total_cap_gains * 100) if total_cap_gains > 0 else 0,
            })
    
    return {
        'overall': overall_stats,
        'by_filing_status': cap_gains_by_status,
        'by_income_bracket': cap_gains_by_income
    }


def project_with_historical_cagr(model_df: pd.DataFrame, trends: dict, years_forward: int = 4) -> pd.DataFrame:
    """Project model forward using historical CAGR."""
    
    projections = []
    
    for status in model_df['filing_status'].unique():
        status_data = model_df[model_df['filing_status'] == status]
        
        # Map filing status names
        trend_status = {
            'single': 'Single',
            'married_filing_jointly': 'Married Filing Jointly',
            'married_filing_separately': 'Married Fil. Separately',
            'head_of_household': 'Head of Household',
        }.get(status, status)
        
        if trend_status in trends:
            trend = trends[trend_status]
            
            current_returns = status_data['weight'].sum()
            current_agi = (status_data['agi'] * status_data['weight']).sum() / 1_000_000
            current_tax = (status_data['hi_state_tax'] * status_data['weight']).sum() / 1_000_000
            
            # Project forward
            projected_returns = current_returns * ((1 + trend['returns_cagr']/100) ** years_forward)
            projected_agi = current_agi * ((1 + trend['agi_cagr']/100) ** years_forward)
            projected_tax = current_tax * ((1 + trend['tax_cagr']/100) ** years_forward)
            
            projections.append({
                'filing_status': status,
                'current_returns': current_returns,
                'projected_returns_2026': projected_returns,
                'current_agi_millions': current_agi,
                'projected_agi_millions_2026': projected_agi,
                'current_tax_millions': current_tax,
                'projected_tax_millions_2026': projected_tax,
                'returns_cagr': trend['returns_cagr'],
                'agi_cagr': trend['agi_cagr'],
                'tax_cagr': trend['tax_cagr'],
            })
    
    return pd.DataFrame(projections)


def validate_capital_gains_trends(cap_gains_stats: dict) -> dict:
    """Validate capital gains against known benchmarks."""
    
    validations = {}
    
    # Known benchmarks (from IRS and research)
    benchmarks = {
        'cap_gains_pct_of_agi': {
            'typical_range': (3, 8),  # Capital gains typically 3-8% of AGI
            'recession': 2,  # Can drop to 2% in recessions
            'boom': 12,  # Can exceed 10% in market booms
        },
        'pct_with_cap_gains': {
            'typical_range': (5, 15),  # 5-15% of filers have capital gains
            'high_income': 50,  # 50%+ for high income filers
        },
        'by_income': {
            '<$50k': (0, 2),  # Very little capital gains
            '$50k-$100k': (1, 4),  # Some capital gains
            '$100k-$200k': (2, 8),  # More capital gains
            '$200k-$500k': (5, 15),  # Significant capital gains
            '$500k-$1M': (10, 25),  # Large capital gains
            '>$1M': (20, 50),  # Very large capital gains
        }
    }
    
    # Validate overall percentage
    overall_pct = cap_gains_stats['overall']['cap_gains_pct_of_agi']
    if benchmarks['cap_gains_pct_of_agi']['typical_range'][0] <= overall_pct <= benchmarks['cap_gains_pct_of_agi']['typical_range'][1]:
        validations['overall_pct'] = f"✅ {overall_pct:.1f}% is within typical range (3-8%)"
    else:
        validations['overall_pct'] = f"⚠️ {overall_pct:.1f}% is outside typical range (3-8%)"
    
    # Validate percentage with gains
    pct_with = cap_gains_stats['overall']['pct_with_cap_gains']
    if benchmarks['pct_with_cap_gains']['typical_range'][0] <= pct_with <= benchmarks['pct_with_cap_gains']['typical_range'][1]:
        validations['pct_with_gains'] = f"✅ {pct_with:.1f}% of filers with gains is within typical range (5-15%)"
    else:
        validations['pct_with_gains'] = f"⚠️ {pct_with:.1f}% of filers with gains is outside typical range (5-15%)"
    
    return validations


def main():
    """Main validation workflow."""
    
    logger.info("="*100)
    logger.info("REFINED MODEL VALIDATION: HISTORICAL TRENDS & CAPITAL GAINS ANALYSIS")
    logger.info("="*100)
    
    # Load data
    historical_df = load_historical_dotax_data()
    model_df = load_refined_model()
    
    # Analyze historical trends
    logger.info("\n" + "="*100)
    logger.info("HISTORICAL TREND ANALYSIS (2018-2021)")
    logger.info("="*100)
    
    trends = analyze_historical_trends(historical_df)
    
    print("\n### Historical Growth Rates (CAGR 2018-2021)")
    print("-"*80)
    print(f"{'Filing Status':<30} {'Returns':>10} {'AGI':>10} {'Tax':>10}")
    print("-"*80)
    
    for status, trend in trends.items():
        if status != 'OVERALL':
            print(f"{status:<30} {trend['returns_cagr']:>9.1f}% {trend['agi_cagr']:>9.1f}% {trend['tax_cagr']:>9.1f}%")
    
    print("-"*80)
    overall = trends.get('OVERALL', {})
    print(f"{'OVERALL':<30} {overall.get('returns_cagr', 0):>9.1f}% {overall.get('agi_cagr', 0):>9.1f}% {overall.get('tax_cagr', 0):>9.1f}%")
    
    # Project model with historical CAGR
    logger.info("\n" + "="*100)
    logger.info("MODEL PROJECTIONS USING HISTORICAL CAGR")
    logger.info("="*100)
    
    projections_df = project_with_historical_cagr(model_df, trends, years_forward=4)
    
    print("\n### 2026 Projections Using Historical Growth Rates")
    print("-"*100)
    print(f"{'Filing Status':<30} {'2022 Tax ($M)':>15} {'2026 Tax ($M)':>15} {'Growth':>10}")
    print("-"*80)
    
    for _, row in projections_df.iterrows():
        growth = (row['projected_tax_millions_2026'] / row['current_tax_millions'] - 1) * 100
        print(f"{row['filing_status']:<30} ${row['current_tax_millions']:>14,.1f} ${row['projected_tax_millions_2026']:>14,.1f} {growth:>9.1f}%")
    
    print("-"*80)
    total_current = projections_df['current_tax_millions'].sum()
    total_projected = projections_df['projected_tax_millions_2026'].sum()
    total_growth = (total_projected / total_current - 1) * 100
    print(f"{'TOTAL':<30} ${total_current:>14,.1f} ${total_projected:>14,.1f} {total_growth:>9.1f}%")
    
    # Analyze capital gains
    logger.info("\n" + "="*100)
    logger.info("CAPITAL GAINS ANALYSIS")
    logger.info("="*100)
    
    cap_gains_stats = analyze_capital_gains(model_df)
    
    if cap_gains_stats:
        print("\n### Capital Gains Overview")
        print("-"*80)
        overall = cap_gains_stats['overall']
        print(f"Total Capital Gains: ${overall['total_capital_gains_millions']:,.1f}M")
        print(f"Capital Gains as % of AGI: {overall['cap_gains_pct_of_agi']:.2f}%")
        print(f"% of Filers with Capital Gains: {overall['pct_with_cap_gains']:.1f}%")
        
        print("\n### Capital Gains by Filing Status")
        print("-"*80)
        print(f"{'Filing Status':<30} {'Cap Gains ($M)':>15} {'% of AGI':>10} {'% w/Gains':>12}")
        print("-"*80)
        
        for status, stats in cap_gains_stats['by_filing_status'].items():
            print(f"{status:<30} ${stats['total_capital_gains_millions']:>14,.1f} "
                  f"{stats['cap_gains_pct_of_agi']:>9.2f}% {stats['pct_with_cap_gains']:>11.1f}%")
        
        print("\n### Capital Gains by Income Bracket")
        print("-"*80)
        print(f"{'Income Bracket':<25} {'Cap Gains ($M)':>15} {'% of AGI':>10} {'% of Total':>12}")
        print("-"*80)
        
        for bracket_stats in cap_gains_stats['by_income_bracket']:
            print(f"{bracket_stats['income_bracket']:<25} ${bracket_stats['capital_gains_millions']:>14,.1f} "
                  f"{bracket_stats['cap_gains_pct_of_agi']:>9.2f}% {bracket_stats['pct_of_total_cap_gains']:>11.1f}%")
        
        # Validate capital gains
        logger.info("\n" + "="*100)
        logger.info("CAPITAL GAINS VALIDATION")
        logger.info("="*100)
        
        validations = validate_capital_gains_trends(cap_gains_stats)
        
        print("\n### Capital Gains Realism Check")
        print("-"*80)
        for key, validation in validations.items():
            print(f"• {validation}")
        
        # Historical comparison
        print("\n### Historical Capital Gains Context")
        print("-"*80)
        print("Typical Ranges (from IRS/research):")
        print("• Capital gains as % of AGI: 3-8% (normal), 2% (recession), 10-12% (boom)")
        print("• % of filers with gains: 5-15% (overall), 50%+ (high income)")
        print("• By income level:")
        print("  - <$50k: 0-2% of AGI")
        print("  - $50-100k: 1-4% of AGI")
        print("  - $100-200k: 2-8% of AGI")
        print("  - $200-500k: 5-15% of AGI")
        print("  - $500k-1M: 10-25% of AGI")
        print("  - >$1M: 20-50% of AGI")
    
    # Summary assessment
    logger.info("\n" + "="*100)
    logger.info("SUMMARY ASSESSMENT")
    logger.info("="*100)
    
    print("\n### Model Validation Against Historical Trends")
    print("-"*80)
    
    # Check if model's effective rates align with historical
    historical_2021 = historical_df[historical_df['year'] == 2021]
    model_effective_rate = (model_df['hi_state_tax'] * model_df['weight']).sum() / (model_df['agi'] * model_df['weight']).sum() * 100
    historical_2021_rate = historical_2021['tax_after_credits_millions'].sum() / historical_2021['agi_net_millions'].sum() * 100
    
    print(f"Model Effective Rate (2022): {model_effective_rate:.2f}%")
    print(f"Historical Rate (2021): {historical_2021_rate:.2f}%")
    print(f"Difference: {model_effective_rate - historical_2021_rate:+.2f}pp")
    
    if abs(model_effective_rate - historical_2021_rate) < 1.0:
        print("✅ Model effective rate aligns well with historical data")
    else:
        print("⚠️ Model effective rate differs from historical trend")
    
    print("\n### 2026 Revenue Projection Assessment")
    print("-"*80)
    print(f"Using Historical CAGR: ${total_projected:,.0f}M")
    print(f"Growth from 2022: {total_growth:.1f}%")
    print(f"Implied Annual Growth: {total_growth/4:.1f}%")
    
    if 5 <= total_growth/4 <= 15:
        print("✅ Projected growth rate is realistic based on historical trends")
    else:
        print("⚠️ Projected growth rate may be unrealistic")
    
    print("\n### Capital Gains Treatment Assessment")
    print("-"*80)
    if cap_gains_stats:
        cap_gains_pct = cap_gains_stats['overall']['cap_gains_pct_of_agi']
        if 3 <= cap_gains_pct <= 8:
            print(f"✅ Capital gains ({cap_gains_pct:.1f}% of AGI) within normal range")
        else:
            print(f"⚠️ Capital gains ({cap_gains_pct:.1f}% of AGI) outside normal range")
    else:
        print("❌ No capital gains data found in model")


if __name__ == "__main__":
    main()
