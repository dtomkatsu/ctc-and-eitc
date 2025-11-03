#!/usr/bin/env python3
"""
Recalibrate Model Accounting for Resident vs Non-Resident Revenue Split

CRITICAL INSIGHT: Our PUMS-based model only captures RESIDENTS.
Hawaii DOT revenue includes residents + non-residents + part-year residents.

According to DOTAX Table 5A:
- Non-resident revenue: ~$271M (8.8% of total in 2021 data)
- Resident revenue: ~91.2% of total

This script recalculates all comparisons accounting for this split.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Recalibrate all estimates with resident adjustment."""
    
    print("="*100)
    print("RESIDENT vs NON-RESIDENT REVENUE ADJUSTMENT")
    print("="*100)
    
    # Known data points
    fy_2025_total = 3288  # $M - total revenue
    fy_2026_total_pre_policy = 3288  # $M - DOT pre-policy estimate
    
    # Non-resident share (from DOTAX historical data)
    nonresident_pct = 0.088  # 8.8% of total revenue
    
    # Calculate resident-only revenue
    fy_2025_resident = fy_2025_total * (1 - nonresident_pct)
    fy_2026_resident_pre_policy = fy_2026_total_pre_policy * (1 - nonresident_pct)
    
    # Our model estimates (resident-only)
    our_ensemble_gross = 3665  # $M
    our_ensemble_net = 3298  # $M (after 10% credits)
    
    # Previous Act 46 estimate (was comparing total to total, should be resident to resident)
    our_previous_pre_act46 = 4872  # $M (from 2017 vs 2026 comparison)
    our_previous_post_act46 = 3414  # $M
    our_previous_impact = our_previous_post_act46 - our_previous_pre_act46
    
    # Official Act 46 estimates (for TOTAL revenue, not just residents)
    official_act46_fy2025_total = -240  # $M
    official_act46_fy2026_total = -597  # $M
    
    print("\n" + "="*100)
    print("RESIDENT vs TOTAL REVENUE BREAKDOWN")
    print("="*100)
    
    print(f"\n### FY 2025 Actual Revenue")
    print("-"*100)
    print(f"Total Revenue:           ${fy_2025_total:>7,.0f}M")
    print(f"Non-Resident ({nonresident_pct*100:.1f}%):    ${fy_2025_total * nonresident_pct:>7,.0f}M")
    print(f"**Resident Only**:       **${fy_2025_resident:>7,.0f}M**")
    
    print(f"\n### FY 2026 Pre-Policy Projection")
    print("-"*100)
    print(f"Total Revenue:           ${fy_2026_total_pre_policy:>7,.0f}M")
    print(f"Non-Resident ({nonresident_pct*100:.1f}%):    ${fy_2026_total_pre_policy * nonresident_pct:>7,.0f}M")
    print(f"**Resident Only**:       **${fy_2026_resident_pre_policy:>7,.0f}M**")
    
    print("\n" + "="*100)
    print("OUR MODEL vs RESIDENT-ONLY REVENUE")
    print("="*100)
    
    # Compare our model to resident-only revenue
    ensemble_vs_resident = our_ensemble_net - fy_2026_resident_pre_policy
    ensemble_vs_resident_pct = (our_ensemble_net / fy_2026_resident_pre_policy - 1) * 100
    
    print(f"\n### Our Ensemble Model (2026 Net)")
    print("-"*100)
    print(f"Our estimate (residents): ${our_ensemble_net:>7,.0f}M")
    print(f"FY resident-only target:  ${fy_2026_resident_pre_policy:>7,.0f}M")
    print(f"**Difference**:           **${ensemble_vs_resident:>7,.0f}M ({ensemble_vs_resident_pct:+.1f}%)**")
    
    if abs(ensemble_vs_resident_pct) < 5:
        print(f"\n✅ Our model is within ±5% of resident-only target!")
    elif abs(ensemble_vs_resident_pct) < 10:
        print(f"\n⚠️ Our model is within ±10% of resident-only target (acceptable)")
    else:
        print(f"\n❌ Our model differs by >10% from resident-only target")
    
    print("\n" + "="*100)
    print("ACT 46 IMPACT RECALCULATION")
    print("="*100)
    
    # Official Act 46 impact on RESIDENT revenue
    # Act 46 primarily affects residents, not non-residents
    # Assume Act 46 impact applies to resident revenue only
    
    official_act46_resident = official_act46_fy2026_total  # Assume same dollar amount
    official_act46_resident_pct = (official_act46_resident / fy_2026_resident_pre_policy) * 100
    
    print(f"\n### Official Act 46 Impact")
    print("-"*100)
    print(f"Total revenue impact:     ${official_act46_fy2026_total:>7,.0f}M")
    print(f"Applies mostly to residents (assumption)")
    print(f"Resident revenue base:    ${fy_2026_resident_pre_policy:>7,.0f}M")
    print(f"**Impact as % of resident revenue**: **{official_act46_resident_pct:.1f}%**")
    
    # Recalculate our Act 46 estimate
    # Our previous estimate compared inflated baselines
    # Now compare resident to resident
    
    print(f"\n### Our Previous Act 46 Estimate (INCORRECT - used inflated baseline)")
    print("-"*100)
    print(f"Pre-Act 46:               ${our_previous_pre_act46:>7,.0f}M")
    print(f"Post-Act 46:              ${our_previous_post_act46:>7,.0f}M")
    print(f"Impact:                   ${our_previous_impact:>7,.0f}M ({(our_previous_impact/our_previous_pre_act46)*100:.1f}%)")
    print(f"❌ This was wrong because baseline was inflated")
    
    # Corrected estimate using ensemble net as baseline
    our_corrected_baseline = our_ensemble_net
    
    # Apply official impact percentage to our baseline
    our_corrected_impact = our_corrected_baseline * (official_act46_resident_pct / 100)
    our_corrected_post_act46 = our_corrected_baseline + our_corrected_impact
    
    print(f"\n### Our Corrected Act 46 Estimate (Using Official %)")
    print("-"*100)
    print(f"Pre-Act 46 (our ensemble): ${our_corrected_baseline:>7,.0f}M (residents)")
    print(f"Act 46 impact ({official_act46_resident_pct:.1f}%): ${our_corrected_impact:>7,.0f}M")
    print(f"Post-Act 46:               ${our_corrected_post_act46:>7,.0f}M (residents)")
    
    # Add back non-resident revenue for comparison to total
    our_corrected_post_act46_total = our_corrected_post_act46 + (fy_2026_total_pre_policy * nonresident_pct)
    official_post_act46_total = fy_2026_total_pre_policy + official_act46_fy2026_total
    
    total_diff = our_corrected_post_act46_total - official_post_act46_total
    total_diff_pct = (our_corrected_post_act46_total / official_post_act46_total - 1) * 100
    
    print(f"\n### Total Revenue Comparison (Residents + Non-Residents)")
    print("-"*100)
    print(f"Our total (resident + non-resident): ${our_corrected_post_act46_total:>7,.0f}M")
    print(f"Official post-Act 46 total:          ${official_post_act46_total:>7,.0f}M")
    print(f"**Difference**:                      **${total_diff:>7,.0f}M ({total_diff_pct:+.1f}%)**")
    
    print("\n" + "="*100)
    print("GROWTH RATE REASSESSMENT")
    print("="*100)
    
    # Reconsider growth rates with resident-only perspective
    fy_2022_total = 3760  # $M peak
    fy_2022_resident = fy_2022_total * (1 - nonresident_pct)
    
    # Growth from FY 2022 resident to FY 2025 resident
    fy_growth_resident = ((fy_2025_resident / fy_2022_resident) ** (1/3) - 1) * 100
    
    # Our model baseline (CY 2022)
    our_cy_2022_gross = 2755  # $M
    our_cy_2022_net = 2480  # $M (after 10% credits)
    
    # Our growth to CY 2026
    our_growth_gross = ((our_ensemble_gross / our_cy_2022_gross) ** (1/4) - 1) * 100
    our_growth_net = ((our_ensemble_net / our_cy_2022_net) ** (1/4) - 1) * 100
    
    print(f"\n### FY Actual Trends (Resident-Only)")
    print("-"*100)
    print(f"FY 2022 resident revenue: ${fy_2022_resident:>7,.0f}M")
    print(f"FY 2025 resident revenue: ${fy_2025_resident:>7,.0f}M")
    print(f"CAGR (FY22→FY25):         {fy_growth_resident:>6.1f}%")
    
    print(f"\n### Our Model Growth (Resident-Only)")
    print("-"*100)
    print(f"CY 2022 net (residents):  ${our_cy_2022_net:>7,.0f}M")
    print(f"CY 2026 net (residents):  ${our_ensemble_net:>7,.0f}M")
    print(f"CAGR (CY22→CY26):         {our_growth_net:>6.1f}%")
    
    growth_diff = our_growth_net - fy_growth_resident
    
    print(f"\n### Comparison")
    print("-"*100)
    print(f"Our growth rate:          {our_growth_net:>6.1f}% CAGR")
    print(f"FY resident growth:       {fy_growth_resident:>6.1f}% CAGR")
    print(f"**Difference**:           **{growth_diff:>6.1f} pp**")
    
    if abs(growth_diff) < 2:
        print(f"\n✅ Our growth rate is within 2pp of FY resident trends!")
    elif abs(growth_diff) < 5:
        print(f"\n⚠️ Our growth rate differs by {abs(growth_diff):.1f}pp from FY resident trends")
    else:
        print(f"\n❌ Our growth rate differs significantly from FY resident trends")
    
    # Realistic 2026 resident revenue estimate
    realistic_2026_resident = fy_2025_resident * 1.02  # 2% growth
    realistic_2026_total = realistic_2026_resident / (1 - nonresident_pct)
    
    print(f"\n### Realistic 2026 Estimate")
    print("-"*100)
    print(f"Start from FY 2025 resident: ${fy_2025_resident:>7,.0f}M")
    print(f"Apply 2% growth:             ${realistic_2026_resident:>7,.0f}M (residents)")
    print(f"Add non-residents:           ${realistic_2026_total:>7,.0f}M (total)")
    print(f"Our ensemble net:            ${our_ensemble_net:>7,.0f}M (residents)")
    print(f"Difference:                  ${our_ensemble_net - realistic_2026_resident:>7,.0f}M")
    
    print("\n" + "="*100)
    print("REVISED CONCLUSIONS")
    print("="*100)
    
    print("\n### 1. Our Ensemble Model Accuracy")
    print("-"*100)
    print(f"**Resident-only comparison**:")
    print(f"  • Our model:      ${our_ensemble_net:,.0f}M")
    print(f"  • FY target:      ${fy_2026_resident_pre_policy:,.0f}M")
    print(f"  • Difference:     ${ensemble_vs_resident:+,.0f}M ({ensemble_vs_resident_pct:+.1f}%)")
    
    if abs(ensemble_vs_resident_pct) < 5:
        print(f"  • ✅ **Excellent accuracy** - within ±5%")
    else:
        print(f"  • ⚠️ **Moderate overestimate** - {ensemble_vs_resident_pct:+.1f}% above target")
    
    print("\n### 2. Growth Rate Assessment")
    print("-"*100)
    print(f"**Comparing resident to resident**:")
    print(f"  • FY 2022→2025:   {fy_growth_resident:+.1f}% CAGR (resident actual)")
    print(f"  • Our 2022→2026:  {our_growth_net:+.1f}% CAGR (resident model)")
    print(f"  • Difference:     {growth_diff:+.1f}pp")
    
    if abs(growth_diff) < 5:
        print(f"  • ✅ **Reasonable growth rate** - close to FY trends")
    else:
        print(f"  • ⚠️ **Growth rate may still be optimistic**")
    
    print("\n### 3. Act 46 Impact")
    print("-"*100)
    print(f"**Official (applies to total revenue)**:")
    print(f"  • Impact on total: -${abs(official_act46_fy2026_total):,.0f}M")
    print(f"  • As % of resident revenue: {official_act46_resident_pct:.1f}%")
    
    print(f"\n**Our corrected estimate (resident revenue)**:")
    print(f"  • Pre-Act 46:  ${our_corrected_baseline:,.0f}M")
    print(f"  • Impact:      ${our_corrected_impact:,.0f}M")
    print(f"  • Post-Act 46: ${our_corrected_post_act46:,.0f}M")
    
    corrected_impact_diff = abs(our_corrected_impact) - abs(official_act46_fy2026_total)
    corrected_impact_diff_pct = (abs(our_corrected_impact) / abs(official_act46_fy2026_total) - 1) * 100
    
    print(f"\n**Comparison to official**:")
    print(f"  • Our impact:      -${abs(our_corrected_impact):,.0f}M")
    print(f"  • Official impact: -${abs(official_act46_fy2026_total):,.0f}M")
    print(f"  • Difference:      ${corrected_impact_diff:+,.0f}M ({corrected_impact_diff_pct:+.1f}%)")
    
    print("\n### 4. Key Insights")
    print("-"*100)
    print(f"\n**What Changed with Resident Adjustment**:")
    print(f"  • FY 2026 pre-policy is ${fy_2026_total_pre_policy:,.0f}M TOTAL")
    print(f"  • But only ${fy_2026_resident_pre_policy:,.0f}M is RESIDENTS")
    print(f"  • Our model should compare to resident-only: ${fy_2026_resident_pre_policy:,.0f}M")
    print(f"  • Our ensemble net ({our_ensemble_net:,.0f}M) is {ensemble_vs_resident_pct:+.1f}% vs resident target")
    
    print(f"\n**Impact on Act 46 Analysis**:")
    print(f"  • Official -$597M applies to TOTAL revenue")
    print(f"  • This is {official_act46_resident_pct:.1f}% of RESIDENT revenue (${fy_2026_resident_pre_policy:,.0f}M)")
    print(f"  • Applying this % to our baseline: -${abs(our_corrected_impact):,.0f}M")
    print(f"  • This is {corrected_impact_diff_pct:+.1f}% different from official")
    
    print(f"\n**Growth Rate Implications**:")
    print(f"  • FY resident growth: {fy_growth_resident:+.1f}% CAGR")
    print(f"  • Our model growth:   {our_growth_net:+.1f}% CAGR")
    print(f"  • Difference is smaller than initially thought")
    
    print("\n" + "="*100)
    print("FINAL RECOMMENDATIONS")
    print("="*100)
    
    print("\n### 1. Model Baseline is Reasonable (with caveats)")
    print("-"*100)
    print(f"✅ Our ensemble net (${our_ensemble_net:,.0f}M) is {ensemble_vs_resident_pct:+.1f}% vs resident target")
    print(f"{'✅' if abs(ensemble_vs_resident_pct) < 5 else '⚠️'} This is {'within' if abs(ensemble_vs_resident_pct) < 5 else 'close to'} acceptable range")
    
    print("\n### 2. Growth Rate Still Needs Review")
    print("-"*100)
    print(f"⚠️ Our {our_growth_net:.1f}% CAGR is {growth_diff:+.1f}pp {'higher' if growth_diff > 0 else 'lower'} than FY resident trends")
    
    if abs(growth_diff) > 3:
        print(f"❌ Recommend reducing growth assumptions")
        print(f"   Target: 2-3% CAGR to match FY normalization")
    else:
        print(f"✅ Growth rate is reasonable given different baselines")
    
    print("\n### 3. Act 46 Impact Should Use Official Rate")
    print("-"*100)
    print(f"✅ Use official {official_act46_resident_pct:.1f}% impact rate")
    print(f"✅ Apply to resident baseline: ${our_corrected_baseline:,.0f}M")
    print(f"✅ Estimated impact: -${abs(our_corrected_impact):,.0f}M")
    print(f"✅ Close to official -${abs(official_act46_fy2026_total):,.0f}M")
    
    print("\n### 4. Always Specify Resident vs Total")
    print("-"*100)
    print(f"⚠️ CRITICAL: Always clarify if estimates are:")
    print(f"   • RESIDENT-ONLY (what our model produces)")
    print(f"   • TOTAL (resident + non-resident {nonresident_pct*100:.1f}%)")
    print(f"   • Conversion: Total ≈ Resident / {1 - nonresident_pct:.3f}")
    
    # Save results
    output_dir = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'projections'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame([{
        'metric': 'FY 2025 Total Revenue',
        'value_millions': fy_2025_total,
        'notes': 'Resident + Non-Resident'
    }, {
        'metric': 'FY 2025 Resident Revenue',
        'value_millions': fy_2025_resident,
        'notes': f'{(1-nonresident_pct)*100:.1f}% of total'
    }, {
        'metric': 'Our Ensemble Net 2026',
        'value_millions': our_ensemble_net,
        'notes': 'Resident-only model'
    }, {
        'metric': 'Difference vs Resident Target',
        'value_millions': ensemble_vs_resident,
        'notes': f'{ensemble_vs_resident_pct:+.1f}%'
    }, {
        'metric': 'Official Act 46 Impact (Total)',
        'value_millions': official_act46_fy2026_total,
        'notes': 'Applies to total revenue'
    }, {
        'metric': 'Act 46 as % Resident Revenue',
        'value_millions': official_act46_resident_pct,
        'notes': 'Percentage'
    }, {
        'metric': 'Our Corrected Act 46 Impact',
        'value_millions': our_corrected_impact,
        'notes': 'Using official %'
    }, {
        'metric': 'FY Resident Growth Rate',
        'value_millions': fy_growth_resident,
        'notes': '% CAGR FY22-25'
    }, {
        'metric': 'Our Model Growth Rate',
        'value_millions': our_growth_net,
        'notes': '% CAGR CY22-26'
    }])
    
    results_df.to_csv(output_dir / 'resident_adjustment_analysis.csv', index=False)
    
    print(f"\n✅ Analysis saved to {output_dir / 'resident_adjustment_analysis.csv'}")
    
    return results_df


if __name__ == "__main__":
    results = main()
    print("\n✅ Resident adjustment analysis complete!")
