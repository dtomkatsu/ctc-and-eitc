"""
Compare IRS SOI vs DOTAX Data Sources

This script compares the IRS SOI and DOTAX benchmarks to understand:
1. How they align (or differ)
2. Which is more appropriate for calibration
3. How to integrate them into the existing calibration approach
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# IRS SOI 2022 Hawaii State Totals (from row 31 "00000,Total")
IRS_SOI_TOTAL = {
    'total_returns': 674660,
    'single': 349070,
    'joint': 236930,
    'head_of_household': 70490,
}

# IRS SOI 2022 by AGI Bracket (rows 32-37 from "00000" section)
# Format: (min_agi, max_agi, total, single, joint, hoh)
IRS_SOI_AGI_BRACKETS = [
    (0, 25000, 168230, 126860, 20870, 16550),
    (25000, 50000, 158030, 99630, 27890, 24670),
    (50000, 75000, 110080, 60050, 31570, 14710),
    (75000, 100000, 71880, 29120, 33740, 7040),
    (100000, 200000, 122110, 26770, 87030, 6580),
    (200000, float('inf'), 44330, 6640, 35830, 940),
]

# DOTAX SOI 2022 Table 12A (AGI brackets, ALL filing statuses combined)
DOTAX_TABLE_12A = [
    (0, 10000, 129376),
    (10000, 20000, 64160),
    (20000, 30000, 57835),
    (30000, 40000, 59827),
    (40000, 50000, 53555),
    (50000, 75000, 91459),
    (75000, 100000, 54976),
    (100000, 150000, 62065),
    (150000, 200000, 27976),
    (200000, 300000, 18937),
    (300000, 400000, 6076),
    (400000, float('inf'), 8875),
]

# DOTAX SOI 2022 Table 13A/13B/13C (Filing status totals)
DOTAX_FILING_STATUS = {
    'single_and_mfs': 351205,  # Combined in Table 13A
    'married_filing_jointly': 216358,
    'head_of_household': 67393,
}


def compare_totals():
    """Compare total returns across data sources."""
    
    logger.info("="*80)
    logger.info("TOTAL RETURNS COMPARISON")
    logger.info("="*80)
    
    irs_total = IRS_SOI_TOTAL['total_returns']
    dotax_total = sum(count for _, _, count in DOTAX_TABLE_12A)
    
    logger.info(f"\nIRS SOI 2022:    {irs_total:>10,} total returns")
    logger.info(f"DOTAX Table 12A: {dotax_total:>10,} total returns")
    logger.info(f"Difference:      {dotax_total - irs_total:>+10,} ({(dotax_total - irs_total)/irs_total*100:+.2f}%)")
    
    if abs(dotax_total - irs_total) / irs_total > 0.01:
        logger.info("\n⚠️  WARNING: IRS and DOTAX totals differ by >1%!")
        logger.info("   This suggests different methodologies or coverage.")
    else:
        logger.info("\n✅ IRS and DOTAX totals align closely.")


def compare_filing_status():
    """Compare filing status totals."""
    
    logger.info("\n" + "="*80)
    logger.info("FILING STATUS COMPARISON")
    logger.info("="*80)
    
    logger.info(f"\n{'Filing Status':<30} {'IRS SOI':>12} {'DOTAX':>12} {'Difference':>12}")
    logger.info("-"*80)
    
    # Single
    irs_single = IRS_SOI_TOTAL['single']
    dotax_single = DOTAX_FILING_STATUS['single_and_mfs']  # Combined with MFS
    logger.info(f"{'Single':<30} {irs_single:>12,} {dotax_single:>12,} {'(incl. MFS)':>12}")
    
    # Joint
    irs_joint = IRS_SOI_TOTAL['joint']
    dotax_joint = DOTAX_FILING_STATUS['married_filing_jointly']
    diff_joint = dotax_joint - irs_joint
    pct_joint = (diff_joint / irs_joint * 100) if irs_joint > 0 else 0
    logger.info(f"{'Married Filing Jointly':<30} {irs_joint:>12,} {dotax_joint:>12,} {diff_joint:>+11,} ({pct_joint:+.1f}%)")
    
    # Head of Household
    irs_hoh = IRS_SOI_TOTAL['head_of_household']
    dotax_hoh = DOTAX_FILING_STATUS['head_of_household']
    diff_hoh = dotax_hoh - irs_hoh
    pct_hoh = (diff_hoh / irs_hoh * 100) if irs_hoh > 0 else 0
    logger.info(f"{'Head of Household':<30} {irs_hoh:>12,} {dotax_hoh:>12,} {diff_hoh:>+11,} ({pct_hoh:+.1f}%)")
    
    logger.info("\n📊 Key Observation:")
    if abs(pct_joint) < 5 and abs(pct_hoh) < 5:
        logger.info("   Filing status totals align well between IRS and DOTAX")
    else:
        logger.info("   Filing status totals show some differences")


def compare_agi_brackets():
    """Compare AGI bracket distributions."""
    
    logger.info("\n" + "="*80)
    logger.info("AGI BRACKET COMPARISON")
    logger.info("="*80)
    logger.info("\nNote: IRS uses broader brackets than DOTAX")
    
    # Map IRS brackets to DOTAX equivalents
    comparisons = [
        ("$0-$25k (IRS) vs $0-$10k + $10k-$20k (DOTAX)", 
         168230, 
         129376 + 64160),
        ("$25k-$50k (IRS) vs $20k-$30k + $30k-$40k + $40k-$50k (DOTAX)", 
         158030, 
         57835 + 59827 + 53555),
        ("$50k-$75k (both)", 
         110080, 
         91459),
        ("$75k-$100k (both)", 
         71880, 
         54976),
        ("$100k-$200k (IRS) vs $100k-$150k + $150k-$200k (DOTAX)", 
         122110, 
         62065 + 27976),
        ("$200k+ (IRS) vs $200k+ all (DOTAX)", 
         44330, 
         18937 + 6076 + 8875),
    ]
    
    logger.info(f"\n{'Bracket':<60} {'IRS SOI':>12} {'DOTAX':>12} {'Diff':>10}")
    logger.info("-"*100)
    
    for label, irs_count, dotax_count in comparisons:
        diff = dotax_count - irs_count
        pct = (diff / irs_count * 100) if irs_count > 0 else 0
        marker = " ⚠️" if abs(pct) > 10 else ""
        logger.info(f"{label:<60} {irs_count:>12,} {dotax_count:>12,} {pct:>+9.1f}%{marker}")


def analyze_irs_filing_status_by_bracket():
    """Analyze IRS filing status distribution within each AGI bracket."""
    
    logger.info("\n" + "="*80)
    logger.info("IRS SOI: FILING STATUS DISTRIBUTION BY AGI BRACKET")
    logger.info("="*80)
    logger.info("\nThis shows what % of each income bracket is Single/Joint/HoH")
    
    logger.info(f"\n{'AGI Bracket':<20} {'Total':>10} {'Single':>10} {'%':>6} {'Joint':>10} {'%':>6} {'HoH':>10} {'%':>6}")
    logger.info("-"*90)
    
    for min_agi, max_agi, total, single, joint, hoh in IRS_SOI_AGI_BRACKETS:
        if max_agi == float('inf'):
            label = f"${min_agi//1000}k+"
        else:
            label = f"${min_agi//1000}k-${max_agi//1000}k"
        
        single_pct = (single / total * 100) if total > 0 else 0
        joint_pct = (joint / total * 100) if total > 0 else 0
        hoh_pct = (hoh / total * 100) if total > 0 else 0
        
        logger.info(f"{label:<20} {total:>10,} {single:>10,} {single_pct:>5.1f}% {joint:>10,} {joint_pct:>5.1f}% {hoh:>10,} {hoh_pct:>5.1f}%")
    
    logger.info("\n📊 Key Patterns:")
    logger.info("   - Lower income (<$25k): 75.4% Single, 12.4% Joint, 9.8% HoH")
    logger.info("   - Middle income ($50k-$75k): 54.6% Single, 28.7% Joint, 13.4% HoH")
    logger.info("   - High income ($200k+): 15.0% Single, 80.8% Joint, 2.1% HoH")
    logger.info("   → Joint filers dominate high income, Singles dominate low income")


def integration_recommendations():
    """Recommend how to integrate IRS SOI with existing calibration."""
    
    logger.info("\n" + "="*80)
    logger.info("INTEGRATION RECOMMENDATIONS")
    logger.info("="*80)
    
    logger.info("\n🔍 DATA SOURCE ANALYSIS:")
    logger.info("\n1. DOTAX (Current Primary Source):")
    logger.info("   ✅ More granular AGI brackets (12 vs 6)")
    logger.info("   ✅ Hawaii-specific (state tax authority)")
    logger.info("   ✅ Includes very detailed breakdowns")
    logger.info("   ❌ Doesn't break down filing status by AGI bracket")
    
    logger.info("\n2. IRS SOI (Newly Available):")
    logger.info("   ✅ Filing status breakdown by AGI bracket")
    logger.info("   ✅ Federal source (potentially more reliable)")
    logger.info("   ❌ Broader AGI brackets (less granular)")
    logger.info("   ❌ Totals differ slightly from DOTAX (-5.6%)")
    
    logger.info("\n" + "-"*80)
    logger.info("RECOMMENDED APPROACH: Three-Layer Calibration")
    logger.info("-"*80)
    
    logger.info("\n📋 Layer 1: Filing Status Totals (DOTAX Tables 13A/13B/13C)")
    logger.info("   - Match overall Single, MFJ, HoH, MFS counts")
    logger.info("   - Use existing bracket_calibration.py logic")
    logger.info("   - Target: 351,205 Single+MFS, 216,358 MFJ, 67,393 HoH")
    
    logger.info("\n📋 Layer 2: Filing Status × AGI Distribution (IRS SOI - NEW)")
    logger.info("   - Match filing status distribution WITHIN each AGI bracket")
    logger.info("   - Example: In $50k-$75k bracket, ensure:")
    logger.info("     • 54.6% are Single")
    logger.info("     • 28.7% are Joint")
    logger.info("     • 13.4% are HoH")
    logger.info("   - This fixes the over-representation issue more precisely")
    
    logger.info("\n📋 Layer 3: Overall AGI Distribution (DOTAX Table 12A)")
    logger.info("   - Ensure total returns in each AGI bracket match")
    logger.info("   - Use existing agi_calibration.py")
    logger.info("   - Target: 91,459 in $50k-$75k bracket")
    
    logger.info("\n" + "-"*80)
    logger.info("ALTERNATIVE: Two-Layer with IRS SOI Priority")
    logger.info("-"*80)
    
    logger.info("\n📋 Layer 1: Filing Status × AGI (IRS SOI)")
    logger.info("   - Directly calibrate to IRS (filing_status, agi_bracket) cells")
    logger.info("   - Example: 60,050 Single returns in $50k-$75k")
    logger.info("   - More granular than current approach")
    
    logger.info("\n📋 Layer 2: Fine-tune to DOTAX granular brackets")
    logger.info("   - Within IRS $50k-$75k, match DOTAX's single $50k-$75k target")
    logger.info("   - Addresses DOTAX's finer granularity")
    
    logger.info("\n" + "="*80)
    logger.info("NEXT STEPS")
    logger.info("="*80)
    
    logger.info("\n1. ✅ Analyze IRS vs DOTAX differences (this script)")
    logger.info("2. ⏳ Decide: Three-layer or Two-layer with IRS priority")
    logger.info("3. ⏳ Implement chosen approach")
    logger.info("4. ⏳ Validate against both IRS and DOTAX benchmarks")
    logger.info("5. ⏳ Compare results to current calibration")


def main():
    """Run all comparisons."""
    
    logger.info("IRS SOI vs DOTAX DATA SOURCE COMPARISON")
    logger.info("Understanding how to integrate both sources\n")
    
    compare_totals()
    compare_filing_status()
    compare_agi_brackets()
    analyze_irs_filing_status_by_bracket()
    integration_recommendations()
    
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)


if __name__ == '__main__':
    main()
