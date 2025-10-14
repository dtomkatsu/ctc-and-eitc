"""
Analyze IRS SOI data to extract filing status distribution.
"""

import pandas as pd
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_soi_filing_status():
    """Analyze IRS SOI data to extract filing status distribution."""
    
    # Load the processed SOI benchmarks
    soi_file = Path('data/processed/soi_benchmarks_2022.csv')
    if not soi_file.exists():
        logger.error(f"SOI benchmarks file not found: {soi_file}")
        return
    
    # Load the data
    df = pd.read_csv(soi_file)
    
    # Extract filing status counts and percentages
    total_returns = df['total_returns'].iloc[0]
    single_returns = df['single_returns'].iloc[0]
    joint_returns = df['joint_returns'].iloc[0]
    hoh_returns = df['hoh_returns'].iloc[0]
    mfs_returns = df['mfs_returns'].iloc[0]
    
    # Calculate percentages
    single_pct = (single_returns / total_returns) * 100
    joint_pct = (joint_returns / total_returns) * 100
    hoh_pct = (hoh_returns / total_returns) * 100
    mfs_pct = (mfs_returns / total_returns) * 100
    
    # Print results
    logger.info("="*80)
    logger.info("IRS SOI FILING STATUS DISTRIBUTION (2022)")
    logger.info("="*80)
    logger.info(f"Total returns: {total_returns:,}")
    logger.info(f"Single returns: {single_returns:,} ({single_pct:.1f}%)")
    logger.info(f"Joint returns: {joint_returns:,} ({joint_pct:.1f}%)")
    logger.info(f"Head of Household returns: {hoh_returns:,} ({hoh_pct:.1f}%)")
    logger.info(f"Married Filing Separately returns: {mfs_returns:,} ({mfs_pct:.1f}%)")
    
    # Verify that the percentages add up to ~100%
    total_pct = single_pct + joint_pct + hoh_pct + mfs_pct
    logger.info(f"\nTotal percentage: {total_pct:.1f}%")
    
    # Calculate the actual percentage of married returns (joint + MFS)
    total_married = joint_returns + mfs_returns
    total_married_pct = (total_married / total_returns) * 100
    logger.info(f"\nTotal married returns (joint + MFS): {total_married:,} ({total_married_pct:.1f}%)")
    
    # Compare with our current model's distribution
    logger.info("\n" + "="*80)
    logger.info("COMPARISON WITH CURRENT MODEL")
    logger.info("="*80)
    
    # Current model's distribution (from our previous run)
    model_total = 45093  # From the last run
    model_single = 25592  # 56.8%
    model_joint = 14026  # 31.2%
    model_hoh = 4528     # 10.1%
    model_mfs = 873      # 1.9%
    
    logger.info(f"\n{'Status':<25} {'IRS SOI %':>10} {'Model %':>10} {'Difference (pp)':>15}")
    logger.info("-" * 60)
    logger.info(f"{'Single':<25} {single_pct:>9.1f}% {model_single/model_total*100:>9.1f}% {model_single/model_total*100 - single_pct:>14.1f}pp")
    logger.info(f"{'Joint':<25} {joint_pct:>9.1f}% {model_joint/model_total*100:>9.1f}% {model_joint/model_total*100 - joint_pct:>14.1f}pp")
    logger.info(f"{'Head of Household':<25} {hoh_pct:>9.1f}% {model_hoh/model_total*100:>9.1f}% {model_hoh/model_total*100 - hoh_pct:>14.1f}pp")
    logger.info(f"{'Married Filing Sep':<25} {mfs_pct:>9.1f}% {model_mfs/model_total*100:>9.1f}% {model_mfs/model_total*100 - mfs_pct:>14.1f}pp")
    
    # Calculate the married percentage difference
    model_married_pct = (model_joint + model_mfs) / model_total * 100
    logger.info(f"\n{'Married (Joint+MFS)':<25} {total_married_pct:>9.1f}% {model_married_pct:>9.1f}% {model_married_pct - total_married_pct:>14.1f}pp")

if __name__ == '__main__':
    analyze_soi_filing_status()
