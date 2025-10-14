"""
[NOT CURRENTLY USED] Synthesize nonresident tax units for Hawaii based on DOTAX SOI Table 17A.

This module is not currently used in the main tax unit construction pipeline.
It's kept for future reference when non-resident analysis is needed.

Since PUMS only captures Hawaii residents, this module can create synthetic
nonresident tax units to match the observed distribution in DOTAX data.

Approach:
1. Parse Table 17A income distribution
2. Create synthetic tax units matching AGI brackets
3. Assign filing statuses based on national patterns
4. Calculate tax liability using simplified rules
5. Weight to match 107,992 nonresident returns
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from config.income_growth import NONRESIDENT_GROWTH


logger = logging.getLogger(__name__)


class NonresidentSynthesizer:
    """
    Synthesize nonresident tax units based on DOTAX SOI Table 17A.
    
    Since PUMS only captures Hawaii residents, we need to create synthetic
    nonresident tax units to model the full 743,109 Hawaii tax returns.
    """
    
    # Actual filing status distribution for nonresidents from DOTAX Table 4 (2022)
    # Source: Dotax Soi 2022 - 4.csv
    NONRES_FILING_STATUS = {
        'joint': 0.471,      # 47.1% - Married Filing Jointly (50,872 returns)
        'single': 0.406,     # 40.6% - Single (43,852 returns)
        'mfs': 0.064,        # 6.4% - Married Filing Separately (6,909 returns)
        'hoh': 0.040,        # 4.0% - Head of Household (4,305 returns)
        'widow': 0.000,      # 0.0% - Qualifying Widow(er) (26 returns)
        'composite': 0.019   # 1.9% - Composite (2,028 returns)
    }
    
    def __init__(self, data_dir: str = "data/raw"):
        """Initialize synthesizer with DOTAX data."""
        self.parser = DOTAXSOIParser(data_dir=data_dir)
        self.nonres_data = self.parser.parse_nonresident_data()
        self.nonres_totals = self.parser.get_nonresident_totals()
        
    def synthesize_nonresident_units(self, 
                                     num_samples: int = 10000,
                                     random_seed: int = 42) -> pd.DataFrame:
        """
        Create synthetic nonresident tax units.
        
        Args:
            num_samples: Number of synthetic units to create (before weighting)
            random_seed: Random seed for reproducibility
            
        Returns:
            DataFrame with synthetic nonresident tax units
        """
        np.random.seed(random_seed)
        
        logger.info(f"Synthesizing {num_samples} nonresident tax units...")
        
        # Step 1: Sample AGI brackets based on Table 17A distribution
        agi_brackets = self._sample_agi_brackets(num_samples)
        
        # Step 2: Generate AGI within each bracket
        agi_values = self._generate_agi_values(agi_brackets)
        
        # Step 3: Assign filing statuses
        filing_statuses = self._assign_filing_statuses(num_samples)
        
        # Step 4: Calculate weights to match 107,992 total returns
        base_weight = self.nonres_totals['total_returns'] / num_samples
        
        # Step 5: Estimate tax liability based on Table 17A averages
        tax_before, tax_after = self._estimate_tax_liability(agi_brackets, agi_values)
        
        # Step 6: Create synthetic tax units DataFrame
        synthetic_units = pd.DataFrame({
            'tax_unit_id': [f'NONRES_{i:06d}' for i in range(num_samples)],
            'household_id': [f'NONRES_HH_{i:06d}' for i in range(num_samples)],
            'filing_status': filing_statuses,
            'agi': agi_values,
            'agi_bracket': agi_brackets,
            'num_dependents': self._assign_dependents(filing_statuses),
            'tax_before_credits': tax_before,
            'tax_after_credits': tax_after,
            'weight': base_weight,
            'is_resident': False,
            'data_source': 'synthetic_nonresident'
        })
        
        # Validate totals
        self._validate_synthetic_units(synthetic_units)
        
        logger.info(f"Created {len(synthetic_units)} synthetic nonresident units")
        logger.info(f"Weighted total: {synthetic_units['weight'].sum():,.0f}")
        logger.info(f"Target: {self.nonres_totals['total_returns']:,}")
        
        return synthetic_units
    
    def _sample_agi_brackets(self, num_samples: int) -> np.ndarray:
        """Sample AGI brackets based on Table 17A distribution."""
        # Get distribution from Table 17A
        brackets = []
        probs = []
        
        for _, row in self.nonres_data.iterrows():
            bracket_label = f"{row['agi_bracket_min']:.0f}-{row['agi_bracket_max']:.0f}"
            brackets.append(bracket_label)
            probs.append(row['pct_returns'])
        
        # Normalize probabilities
        probs = np.array(probs) / sum(probs)
        
        # Sample brackets
        sampled_brackets = np.random.choice(brackets, size=num_samples, p=probs)
        
        return sampled_brackets
    
    def _generate_agi_values(self, agi_brackets: np.ndarray) -> np.ndarray:
        """
        Generate AGI values within each bracket.
        
        Note: AGI values from DOTAX Table 17A are in 2022 dollars.
        We apply the nonresident growth factor (1.053) to project to 2026.
        """
        agi_values = np.zeros(len(agi_brackets))
        
        for i, bracket in enumerate(agi_brackets):
            # Parse bracket min/max
            parts = bracket.split('-')
            min_agi = float(parts[0])
            max_agi = float(parts[1])
            
            # Handle infinity
            if max_agi == np.inf:
                # For top bracket, use log-normal distribution
                # Mean around $500k, high variance
                base_agi = np.random.lognormal(mean=13, sigma=0.8)
            else:
                # Uniform distribution within bracket
                # (Could use more sophisticated distribution)
                base_agi = np.random.uniform(min_agi, max_agi)
            
            # Apply 2026 growth projection (2022 → 2026)
            agi_values[i] = NONRESIDENT_GROWTH.apply_to_income(base_agi)
        
        return agi_values
    
    def _assign_filing_statuses(self, num_samples: int) -> np.ndarray:
        """Assign filing statuses based on actual DOTAX Table 4 distribution."""
        statuses = list(self.NONRES_FILING_STATUS.keys())
        probs = list(self.NONRES_FILING_STATUS.values())
        
        # Normalize probabilities (in case they don't sum to exactly 1.0)
        probs = np.array(probs) / sum(probs)
        
        return np.random.choice(statuses, size=num_samples, p=probs)
    
    def _assign_dependents(self, filing_statuses: np.ndarray) -> np.ndarray:
        """Assign number of dependents based on filing status."""
        num_dependents = np.zeros(len(filing_statuses), dtype=int)
        
        for i, status in enumerate(filing_statuses):
            if status == 'hoh':
                # HoH must have at least 1 dependent
                num_dependents[i] = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
            elif status == 'widow':
                # Qualifying widow(er) must have at least 1 dependent child
                num_dependents[i] = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
            elif status == 'joint':
                # Joint filers may have dependents
                num_dependents[i] = np.random.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])
            elif status == 'composite':
                # Composite returns - typically complex, assume some dependents
                num_dependents[i] = np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2])
            else:
                # Single/MFS less likely to have dependents
                num_dependents[i] = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
        
        return num_dependents
    
    def _estimate_tax_liability(self, 
                                agi_brackets: np.ndarray,
                                agi_values: np.ndarray) -> tuple:
        """
        Estimate tax liability based on Table 17A averages.
        
        Returns:
            (tax_before_credits, tax_after_credits)
        """
        tax_before = np.zeros(len(agi_brackets))
        tax_after = np.zeros(len(agi_brackets))
        
        # Create lookup dict from Table 17A
        bracket_tax = {}
        for _, row in self.nonres_data.iterrows():
            bracket_label = f"{row['agi_bracket_min']:.0f}-{row['agi_bracket_max']:.0f}"
            bracket_tax[bracket_label] = {
                'avg_before': row['avg_tax_before'],
                'avg_after': row['avg_tax_after']
            }
        
        # Assign tax based on bracket averages
        for i, bracket in enumerate(agi_brackets):
            if bracket in bracket_tax:
                # Use average from Table 17A with some random variation
                avg_before = bracket_tax[bracket]['avg_before']
                avg_after = bracket_tax[bracket]['avg_after']
                
                # Add 20% random variation
                tax_before[i] = avg_before * np.random.uniform(0.8, 1.2)
                tax_after[i] = avg_after * np.random.uniform(0.8, 1.2)
            else:
                # Fallback: estimate as 8% of AGI
                tax_before[i] = agi_values[i] * 0.08
                tax_after[i] = agi_values[i] * 0.07
        
        return tax_before, tax_after
    
    def _validate_synthetic_units(self, synthetic_units: pd.DataFrame):
        """Validate synthetic units match DOTAX totals."""
        weighted_total = synthetic_units['weight'].sum()
        target_total = self.nonres_totals['total_returns']
        
        pct_diff = abs(weighted_total - target_total) / target_total
        
        if pct_diff > 0.01:  # More than 1% difference
            logger.warning(
                f"Synthetic units total ({weighted_total:,.0f}) differs from "
                f"target ({target_total:,}) by {pct_diff:.1%}"
            )
        else:
            logger.info(f"✅ Synthetic units total matches target within 1%")
        
        # Validate tax revenue
        weighted_tax = (synthetic_units['tax_after_credits'] * synthetic_units['weight']).sum()
        target_tax = self.nonres_totals['total_tax_after'] * 1_000_000  # Convert from millions
        
        tax_diff = abs(weighted_tax - target_tax) / target_tax
        
        if tax_diff > 0.1:  # More than 10% difference
            logger.warning(
                f"Synthetic tax revenue (${weighted_tax/1e6:.0f}M) differs from "
                f"target (${target_tax/1e6:.0f}M) by {tax_diff:.1%}"
            )
        else:
            logger.info(f"✅ Synthetic tax revenue matches target within 10%")


def combine_resident_nonresident_units(resident_units: pd.DataFrame,
                                       nonresident_units: pd.DataFrame) -> pd.DataFrame:
    """
    Combine resident and nonresident tax units into single dataset.
    
    Args:
        resident_units: Tax units from PUMS (residents)
        nonresident_units: Synthetic nonresident tax units
        
    Returns:
        Combined DataFrame with all tax units
    """
    # Ensure consistent columns
    common_cols = [
        'tax_unit_id', 'household_id', 'filing_status', 'agi',
        'num_dependents', 'tax_before_credits', 'tax_after_credits',
        'weight', 'is_resident', 'data_source'
    ]
    
    # Add is_resident flag to resident units if not present
    if 'is_resident' not in resident_units.columns:
        resident_units['is_resident'] = True
    
    if 'data_source' not in resident_units.columns:
        resident_units['data_source'] = 'pums_resident'
    
    # Select common columns
    resident_subset = resident_units[common_cols].copy()
    nonresident_subset = nonresident_units[common_cols].copy()
    
    # Combine
    combined = pd.concat([resident_subset, nonresident_subset], ignore_index=True)
    
    logger.info(f"Combined {len(resident_subset):,} resident + {len(nonresident_subset):,} nonresident units")
    logger.info(f"Total weighted returns: {combined['weight'].sum():,.0f}")
    logger.info(f"  Residents: {resident_subset['weight'].sum():,.0f}")
    logger.info(f"  Nonresidents: {nonresident_subset['weight'].sum():,.0f}")
    
    return combined


def main():
    """Demo usage of nonresident synthesizer."""
    logging.basicConfig(level=logging.INFO)
    
    # Create synthesizer
    synthesizer = NonresidentSynthesizer(data_dir='data/raw')
    
    # Generate synthetic nonresident units
    nonres_units = synthesizer.synthesize_nonresident_units(num_samples=10000)
    
    print("\n" + "="*80)
    print("SYNTHETIC NONRESIDENT TAX UNITS")
    print("="*80)
    
    # Summary statistics
    print(f"\nTotal synthetic units: {len(nonres_units):,}")
    print(f"Weighted total: {nonres_units['weight'].sum():,.0f}")
    print(f"Target: 107,992")
    
    print("\n### Filing Status Distribution ###")
    filing_dist = nonres_units.groupby('filing_status')['weight'].sum()
    for status, count in filing_dist.items():
        pct = count / filing_dist.sum()
        print(f"{status}: {count:,.0f} ({pct:.1%})")
    
    print("\n### Income Statistics ###")
    print(f"Mean AGI: ${(nonres_units['agi'] * nonres_units['weight']).sum() / nonres_units['weight'].sum():,.0f}")
    print(f"Median AGI: ${nonres_units['agi'].median():,.0f}")
    
    print("\n### Tax Revenue ###")
    total_tax = (nonres_units['tax_after_credits'] * nonres_units['weight']).sum()
    print(f"Total tax (after credits): ${total_tax / 1e6:.0f}M")
    print(f"Target: $271M")
    print(f"Difference: {(total_tax / 1e6 - 271) / 271 * 100:+.1f}%")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
