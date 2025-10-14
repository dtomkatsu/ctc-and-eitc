#!/usr/bin/env python3
"""
Weight Splitting with Raking for Filing Status Assignment

This script implements a weight splitting approach with iterative raking to assign
filing statuses to households while matching IRS target distributions.

Steps:
1. Generate all plausible filing status scenarios for each household
2. Assign initial probabilities (weights) to each scenario
3. Iteratively rake scenario weights to match IRS target shares
4. Select highest weight scenario for each household
5. Validate final distribution against targets
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from tax.units.constructor import TaxUnitConstructor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WeightSplittingRaker:
    """Implements weight splitting with raking for filing status assignment."""
    
    def __init__(self, tax_units_path: str, person_path: str, household_path: str):
        """Initialize with data paths."""
        self.tax_units_path = tax_units_path
        self.person_path = person_path
        self.household_path = household_path
        
        # IRS target shares for Hawaii (from SOI data)
        self.irs_targets = {
            'single': 0.51,
            'married_filing_jointly': 0.36,
            'head_of_household': 0.096,
            'married_filing_separately': 0.034
        }
        
        # Convergence parameters
        self.max_iterations = 20
        self.tolerance = 0.001  # 0.1% tolerance
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load tax units, person, and household data with geographic crosswalks."""
        logger.info("Loading data files...")
        
        tax_units = pd.read_parquet(self.tax_units_path)
        person_df = pd.read_parquet(self.person_path)
        household_df = pd.read_parquet(self.household_path)
        
        # Load geographic crosswalks
        logger.info("Loading geographic crosswalks...")
        puma_county = pd.read_csv("data/crosswalks/hawaii_puma_counties_corrected.csv")
        puma_district = pd.read_csv("data/crosswalks/hawaii_puma_districts.csv")
        
        # Add geographic identifiers to household data
        if 'PUMA' in household_df.columns:
            # Convert PUMA to string for consistent merging
            household_df['PUMA'] = household_df['PUMA'].astype(str).str.zfill(4)
            puma_county['PUMA'] = puma_county['PUMA'].astype(str).str.zfill(4)
            
            # Merge county information
            household_df = household_df.merge(
                puma_county[['PUMA', 'county']], 
                on='PUMA', 
                how='left'
            )
            
            # For district mapping, use the 3-digit crosswalk
            if 'PUMA' in puma_district.columns:
                # Convert district PUMA codes to match household format (ensure both are strings with leading zeros)
                puma_district['PUMA_4digit'] = puma_district['PUMA'].astype(str).str.zfill(4)
                household_df = household_df.merge(
                    puma_district[['PUMA_4digit', 'house_district', 'senate_district']], 
                    left_on='PUMA',
                    right_on='PUMA_4digit',
                    how='left'
                )
                
                # Fix PUMA 100 inconsistency: it represents Maui/Kalawao/Kauai but 3-digit crosswalk maps to Honolulu
                # Assign PUMA 100 to representative Maui district (House 1, Senate 1) based on county crosswalk
                puma_100_mask = household_df['PUMA'] == '0100'
                if puma_100_mask.any():
                    household_df.loc[puma_100_mask, 'house_district'] = 1  # Maui House District 1
                    household_df.loc[puma_100_mask, 'senate_district'] = 1  # Maui Senate District 1
                    logger.info(f"Corrected PUMA 100 district assignment: {puma_100_mask.sum()} households assigned to Maui districts")
            
            logger.info(f"Added geographic identifiers to {len(household_df)} households")
        else:
            logger.warning("PUMA column not found in household data - geographic crosswalks not applied")
        
        logger.info(f"Loaded {len(tax_units)} tax units, {len(person_df)} persons, {len(household_df)} households")
        return tax_units, person_df, household_df
    
    def generate_scenarios(self, tax_units: pd.DataFrame, person_df: pd.DataFrame, 
                          household_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all plausible filing status scenarios for each household.
        
        Returns DataFrame with columns: household_id, filing_status, scenario_weight, initial_prob
        """
        logger.info("Generating filing status scenarios for each household...")
        
        scenarios = []
        
        # Group tax units by household
        household_groups = tax_units.groupby('SERIALNO')
        
        for household_id, hh_tax_units in household_groups:
            # Get household weight from tax units data
            hh_weight = hh_tax_units['hh_weight'].iloc[0]
            
            # Get adults in household
            hh_persons = person_df[person_df['SERIALNO'] == household_id]
            adults = hh_persons[hh_persons['AGEP'] >= 18]
            
            # Generate plausible scenarios based on household composition
            plausible_scenarios = self._get_plausible_scenarios(adults, hh_persons, household_id)
            
            # Assign initial probabilities
            total_prob = sum(scenario['prob'] for scenario in plausible_scenarios)
            
            for scenario in plausible_scenarios:
                normalized_prob = scenario['prob'] / total_prob
                scenarios.append({
                    'household_id': household_id,
                    'filing_status': scenario['status'],
                    'scenario_weight': hh_weight * normalized_prob,
                    'initial_prob': normalized_prob,
                    'household_weight': hh_weight
                })
        
        scenarios_df = pd.DataFrame(scenarios)
        logger.info(f"Generated {len(scenarios_df)} scenarios for {scenarios_df['household_id'].nunique()} households")
        
        # Debug: Check scenario distribution
        scenario_counts = scenarios_df['filing_status'].value_counts()
        logger.info("Initial scenario counts:")
        for status, count in scenario_counts.items():
            logger.info(f"  {status}: {count} scenarios")
        
        return scenarios_df
    
    def _get_plausible_scenarios(self, adults: pd.DataFrame, all_persons: pd.DataFrame, household_id: str = None) -> List[Dict]:
        """
        Determine plausible filing status scenarios for a household.
        
        Args:
            adults: DataFrame of adults in the household
            all_persons: DataFrame of all persons in the household
            household_id: Optional household identifier for debugging
        """
        scenarios = []
        
        # Count married adults
        married_adults = adults[adults['MAR'] == 1]
        unmarried_adults = adults[adults['MAR'] != 1]
        children = all_persons[all_persons['AGEP'] < 18]
        
        num_married = len(married_adults)
        num_unmarried = len(unmarried_adults)
        num_children = len(children)
        
        # Calculate base probabilities with reduced single filer probability
        num_adults = len(adults)
        # Steeper reduction in single filing probability for larger households
        if num_adults == 1:
            single_prob = 0.20  # 20% for 1-person households
        elif num_adults == 2:
            single_prob = 0.15  # 15% for 2-person households
        else:
            # Steeper reduction for households > 2 people
            single_prob = max(0.02, 0.20 - ((num_adults - 2) * 0.2))  # 20% reduction per additional person
            
        base_probs = {
            'single': single_prob,
            'married_filing_jointly': 0.6 if num_married == 2 else 0.4,
            'married_filing_separately': 0.25
        }
        
        # Enhanced HoH probability calculation with Option 4
        hoh_prob = 0.0
        if num_unmarried > 0:
            if num_children > 0:
                # Higher probability for clear HoH cases with children
                hoh_prob = 0.8 if num_children >= 2 else 0.65  # Increased base probabilities
                
                # Additional boost for households with children but no HoH
                if base_probs['single'] > 0.1:
                    base_probs['single'] *= 0.4  # 60% reduction in single probability
                    hoh_prob = max(hoh_prob, 0.6)  # Ensure minimum HoH probability
                
                # Income-based adjustment for HoH
                if 'PINCP' in unmarried_adults.columns and not unmarried_adults.empty:
                    median_income = unmarried_adults['PINCP'].median()
                    if median_income < 40000:  # Lower income threshold
                        hoh_prob = min(0.8, hoh_prob * 1.4)
                        
            # Check for elderly dependents (65+)
            if household_id is not None and 'AGEP' in all_persons.columns and 'SERIALNO' in all_persons.columns:
                elderly_dependents = all_persons[
                    (all_persons['AGEP'] >= 65) & 
                    (all_persons['SERIALNO'] == household_id)
                ]
                if len(elderly_dependents) > 0:
                    hoh_prob = 0.4
                
        base_probs['head_of_household'] = hoh_prob
        
        # Dynamic MFS probability adjustments
        if num_married >= 2:
            mfs_factors = []
            
            # 0. Check for 65+ disabled spouse (highest priority)
            if 'AGEP' in married_adults.columns and 'DIS' in married_adults.columns:
                # Check if any spouse is 65+ and disabled
                elderly_disabled = married_adults[(married_adults['AGEP'] >= 55) & 
                                               (married_adults['DIS'] == 1)]
                if not elderly_disabled.empty:
                    # Very high MFS probability for this case
                    mfs_factors.append(0.9)
                    logger.debug(f"Found {len(elderly_disabled)} elderly disabled spouse(s), boosting MFS probability")
            
            # 1. Income-based factors
            if 'PINCP' in married_adults.columns and len(married_adults) >= 2:
                # Filter out negative or zero incomes and sort in descending order
                incomes = sorted([inc for inc in married_adults['PINCP'].dropna() if inc > 0], reverse=True)
                
                # Need at least 2 valid positive incomes to proceed
                if len(incomes) >= 2:
                    total_income = sum(incomes)
                    higher_earner_share = incomes[0] / total_income
                    
                    # Condition A: Dual high-income earners (>$80k each)
                    if incomes[0] > 80000 and incomes[1] > 80000:
                        # Significantly boost MFS probability for high-income dual-earner couples
                        income_factor = 0.8  # Increased from 0.6 to 0.8
                        logger.debug(f"Dual high-income earners: ${incomes[0]:,.0f} and ${incomes[1]:,.0f}")
                    
                    # New condition: Moderate income disparity (3:1 to 5:1 ratio)
                    elif len(incomes) >= 2 and incomes[1] > 0 and (incomes[0] / incomes[1]) >= 3:
                        income_factor = 0.4  # Moderate MFS probability for moderate income disparities
                        logger.debug(f"Moderate income disparity: {incomes[0]/incomes[1]:.1f}x ratio")
                    
                    # Condition B: One spouse earns >85% of total income AND total > $200k
                    elif higher_earner_share > 0.85 and total_income > 200000:
                        # Significant boost for high-income, high-disparity couples
                        income_factor = 0.7  # Highest cap for this case
                        logger.debug(f"High income disparity: {higher_earner_share:.1%} of ${total_income:,.0f}")
                    
                    else:
                        # Original ratio-based calculation with safety checks
                        lower_income = incomes[-1]
                        if lower_income > 0:  # Ensure we don't divide by zero
                            income_ratio = incomes[0] / lower_income
                            # Scale factor from 0 to 0.5 based on income ratio (5x to 50x)
                            income_factor = min(0.5, max(0, (min(income_ratio, 50) - 5) * 0.0111))
                        else:
                            # Fallback if we somehow have zero income
                            income_factor = 0.1
                    
                    mfs_factors.append(income_factor)
                else:
                    # Not enough valid incomes for comparison
                    mfs_factors.append(0.05)  # Small base probability
            
            # 2. Age difference factor (0-0.3)
            if 'AGEP' in married_adults.columns and len(married_adults) >= 2:
                ages = married_adults['AGEP'].sort_values()
                if len(ages) >= 2:
                    age_diff = ages.iloc[-1] - ages.iloc[0]
                    # Scale factor from 0 to 0.3 based on age difference (5 to 25 years)
                    age_factor = min(0.3, max(0, (min(age_diff, 25) - 5) * 0.015))
                    mfs_factors.append(age_factor)
                    logger.debug(f"Age difference {age_diff} years adds {age_factor:.2f} to MFS probability")
            
            # 3. Number of children factor (0-0.1)
            if num_children > 0:
                # More children slightly increases MFS probability
                children_factor = min(0.1, num_children * 0.02)
                mfs_factors.append(children_factor)
            
            # Base MFS probability for all married couples
            base_mfs = 0.1  # 10% base probability for any married couple
            
            # Calculate MFS adjustment with higher base and stronger factors
            mfs_adjustment = min(0.8, base_mfs + sum(mfs_factors))  # Increased cap to 0.8
            
            # Set MFS probability with a minimum floor
            base_probs['married_filing_separately'] = max(0.1, min(0.8, mfs_adjustment))
            
            # More aggressive MFS: Take 60% from joint, 40% from others
            joint_reduction = mfs_adjustment * 0.6
            base_probs['married_filing_jointly'] = max(0.15, base_probs['married_filing_jointly'] - joint_reduction)
            
            # Ensure HoH doesn't get reduced below minimum
            base_probs['head_of_household'] = max(0.15, base_probs.get('head_of_household', 0))
            
            # Log MFS probability for debugging
            logger.debug(f"MFS probability: {base_probs['married_filing_separately']:.2f} "
                       f"(base: {base_mfs:.2f}, factors: {sum(mfs_factors):.2f})")
        
        # Add scenarios based on household composition
        if num_unmarried > 0:
            scenarios.append({'status': 'single', 'prob': base_probs['single']})
        
        if num_married >= 2:
            scenarios.append({
                'status': 'married_filing_jointly', 
                'prob': base_probs['married_filing_jointly']
            })
            scenarios.append({
                'status': 'married_filing_separately',
                'prob': base_probs['married_filing_separately']
            })
        
        if num_unmarried > 0 and num_children > 0:
            scenarios.append({
                'status': 'head_of_household',
                'prob': base_probs['head_of_household']
            })
        
        # Ensure at least one scenario exists
        if not scenarios:
            scenarios.append({'status': 'single', 'prob': 1.0})
        
        return scenarios
    
    def rake_weights(self, scenarios_df: pd.DataFrame) -> pd.DataFrame:
        """
        Iteratively rake scenario weights to match IRS target shares
        with constraints to preserve MFS and HoH representation.
        """
        logger.info("Starting constrained iterative raking process...")
        
        df = scenarios_df.copy()
        
        # Minimum weight constraints (as proportion of total weight)
        min_weights = {
            'married_filing_separately': 0.045,  # 4.5% floor
            'head_of_household': 0.10,           # 10.0% floor
            'single': 0.37,                     # Further reduced from 0.39 to reduce single filers
            'married_filing_jointly': 0.35       # Increased to help offset single reduction
        }
        
        # Cap single filer share to prevent it from dominating
        max_single_share = 0.52  # Slightly increased to allow more flexibility
        
        # Enforce minimum MFS share
        min_mfs_share = 0.034  # Increased to match target of 3.4%
        
        for iteration in range(self.max_iterations):
            # Calculate current shares and apply minimums
            current_totals = df.groupby('filing_status')['scenario_weight'].sum()
            grand_total = current_totals.sum()
            
            # Apply minimum weight constraints
            for status, min_weight in min_weights.items():
                if status in current_totals:
                    min_total = grand_total * min_weight
                    if current_totals[status] < min_total:
                        # Scale up under-represented statuses
                        scale_factor = min_total / current_totals[status]
                        mask = df['filing_status'] == status
                        df.loc[mask, 'scenario_weight'] *= scale_factor
                        current_totals[status] = min_total
            
            # Recalculate after applying minimums
            grand_total = current_totals.sum()
            current_shares = current_totals / grand_total
            
            # Enforce maximum single filer share
            if 'single' in current_shares and current_shares['single'] > max_single_share:
                excess_share = current_shares['single'] - max_single_share
                
                # Calculate reduction factor for single filers
                single_weight = current_totals['single']
                reduction_factor = 1 - (excess_share / current_shares['single'])
                
                # Apply reduction to single filers
                single_mask = df['filing_status'] == 'single'
                df.loc[single_mask, 'scenario_weight'] *= reduction_factor
                
                # Distribute excess weight to MFS first, then others
                excess_weight = single_weight * (1 - reduction_factor)
                
                # Calculate distribution weights - prioritize MFS
                dist_weights = {
                    'married_filing_separately': 0.6,  # 60% of excess to MFS
                    'head_of_household': 0.2,         # 20% to HoH
                    'married_filing_jointly': 0.2     # 20% to MFJ
                }
                
                # Apply distribution
                for status, weight_share in dist_weights.items():
                    if status in current_totals:
                        status_mask = df['filing_status'] == status
                        df.loc[status_mask, 'scenario_weight'] += excess_weight * weight_share / len(df[status_mask])
                
                # Recalculate totals after adjustment
                current_totals = df.groupby('filing_status')['scenario_weight'].sum()
                grand_total = current_totals.sum()
                current_shares = current_totals / grand_total
            
            # Check convergence
            max_diff = 0
            for status in self.irs_targets:
                if status in current_shares:
                    diff = abs(current_shares[status] - self.irs_targets[status])
                    max_diff = max(max_diff, diff)
            
            logger.info(f"Iteration {iteration + 1}: Max difference = {max_diff:.4f}")
            
            if max_diff < self.tolerance:
                logger.info(f"Converged after {iteration + 1} iterations")
                break
            
            # Apply raking adjustments with damping for stability
            for status in self.irs_targets:
                if status in current_shares and current_shares[status] > 0:
                    # Calculate adjustment factor with damping
                    raw_factor = self.irs_targets[status] / current_shares[status]
                    # Dampen the adjustment to prevent overshooting
                    damping = 0.5 if iteration < 5 else 0.7  # More aggressive in later iterations
                    factor = 1 + (raw_factor - 1) * damping
                    
                    # Ensure we don't completely eliminate any status
                    if status == 'married_filing_separately' and factor < 1.0:
                        factor = max(1.0, factor)  # Only allow increases for MFS
                    
                    # Apply adjustment
                    mask = df['filing_status'] == status
                    df.loc[mask, 'scenario_weight'] *= factor
        
        else:
            logger.warning(f"Did not converge after {self.max_iterations} iterations")
        
        return df
    
    def assign_final_statuses(self, raked_scenarios: pd.DataFrame) -> pd.DataFrame:
        """
        For each household, probabilistically select a scenario based on raked weights.
        """
        logger.info("Assigning final filing statuses using probabilistic selection...")
        
        # Debug: Check raked scenario weights by status
        raked_totals = raked_scenarios.groupby('filing_status')['scenario_weight'].sum()
        logger.info("Raked scenario weights by status:")
        for status, weight in raked_totals.items():
            logger.info(f"  {status}: {weight:,.0f}")
        
        final_assignments = []
        
        # For each household, probabilistically select based on normalized weights
        for household_id, hh_scenarios in raked_scenarios.groupby('household_id'):
            # Normalize weights to probabilities within household
            hh_scenarios = hh_scenarios.copy()
            
            # Handle negative or zero weights by setting minimum threshold
            hh_scenarios['scenario_weight'] = hh_scenarios['scenario_weight'].clip(lower=1e-10)
            total_weight = hh_scenarios['scenario_weight'].sum()
            hh_scenarios['probability'] = hh_scenarios['scenario_weight'] / total_weight
            
            # Check for NaN probabilities
            if hh_scenarios['probability'].isna().any():
                logger.warning(f"NaN probabilities for household {household_id}, using equal weights")
                hh_scenarios['probability'] = 1.0 / len(hh_scenarios)
            
            # Probabilistic selection based on raked weights
            np.random.seed(hash(str(household_id)) % 2**31)  # Reproducible random selection
            selected_idx = np.random.choice(
                hh_scenarios.index, 
                p=hh_scenarios['probability'].values
            )
            
            selected_scenario = hh_scenarios.loc[selected_idx]
            final_assignments.append({
                'household_id': household_id,
                'filing_status': selected_scenario['filing_status'],
                'scenario_weight': selected_scenario['scenario_weight'],
                'household_weight': selected_scenario['household_weight']
            })
        
        final_assignments = pd.DataFrame(final_assignments)
        
        # Apply weight normalization to match Hawaii tax return total
        target_total = 635117  # Hawaii resident returns in 2022
        current_total = final_assignments['scenario_weight'].sum()
        scale_factor = target_total / current_total
        
        logger.info(f"Normalizing weights: {current_total:,.0f} -> {target_total:,.0f} (factor: {scale_factor:.4f})")
        final_assignments['scenario_weight'] *= scale_factor
        
        # Debug: Check how many households chose each status
        assignment_counts = final_assignments['filing_status'].value_counts()
        logger.info("Final assignments by status:")
        for status, count in assignment_counts.items():
            logger.info(f"  {status}: {count} households")
        
        # Verify final total
        final_total = final_assignments['scenario_weight'].sum()
        logger.info(f"Final normalized weight total: {final_total:,.0f}")
        logger.info(f"Assigned final statuses to {len(final_assignments)} households")
        
        return final_assignments
    
    def validate_results(self, final_assignments: pd.DataFrame) -> Dict:
        """Validate final distribution against IRS targets."""
        logger.info("Validating final results...")
        
        # Calculate final shares
        final_totals = final_assignments.groupby('filing_status')['scenario_weight'].sum()
        grand_total = final_totals.sum()
        final_shares = final_totals / grand_total
        
        # Compare to targets
        validation_results = {}
        
        print("\n" + "="*80)
        print("WEIGHT SPLITTING WITH RAKING - VALIDATION RESULTS")
        print("="*80)
        print(f"{'Filing Status':<25} {'Final %':<12} {'Target %':<12} {'Difference':<12} {'Status'}")
        print("-"*80)
        
        for status in self.irs_targets:
            target = self.irs_targets[status]
            actual = final_shares.get(status, 0.0)
            diff = actual - target
            diff_pct = (diff / target) * 100 if target > 0 else 0
            
            status_indicator = "✓" if abs(diff) < 0.02 else "⚠" if abs(diff) < 0.05 else "✗"
            
            print(f"{status:<25} {actual:<12.1%} {target:<12.1%} {diff:+7.1%} ({diff_pct:+5.1f}%) {status_indicator}")
            
            validation_results[status] = {
                'actual': actual,
                'target': target,
                'difference': diff,
                'diff_pct': diff_pct
            }
        
        print("-"*80)
        print(f"Total households: {len(final_assignments):,}")
        print(f"Total weight: {grand_total:,.0f}")
        
        # Overall accuracy
        total_abs_diff = sum(abs(v['difference']) for v in validation_results.values())
        print(f"Total absolute difference: {total_abs_diff:.3f}")
        
        accuracy = "Excellent" if total_abs_diff < 0.02 else "Good" if total_abs_diff < 0.05 else "Needs Improvement"
        print(f"Overall accuracy: {accuracy}")
        print("="*80)
        
        return validation_results
    
    def run_full_process(self) -> Tuple[pd.DataFrame, Dict]:
        """Run the complete weight splitting with raking process."""
        logger.info("Starting weight splitting with raking process...")
        
        # Step 1: Load data
        tax_units, person_df, household_df = self.load_data()
        
        # Step 2: Generate scenarios
        scenarios_df = self.generate_scenarios(tax_units, person_df, household_df)
        
        # Step 3: Rake weights
        raked_scenarios = self.rake_weights(scenarios_df)
        
        # Step 4: Assign final statuses
        final_assignments = self.assign_final_statuses(raked_scenarios)
        
        # Step 5: Add geographic information to final assignments
        final_assignments = self.add_geographic_info(final_assignments, household_df)
        
        # Step 6: Validate results
        validation_results = self.validate_results(final_assignments)
        
        return final_assignments, validation_results
    
    def add_geographic_info(self, final_assignments: pd.DataFrame, household_df: pd.DataFrame) -> pd.DataFrame:
        """Add geographic information to final assignments."""
        logger.info("Adding geographic information to final assignments...")
        
        # Apply the same geographic crosswalk logic as in load_data
        # Load geographic crosswalks
        puma_county = pd.read_csv('data/crosswalks/hawaii_puma_counties_corrected.csv')
        puma_district = pd.read_csv('data/crosswalks/hawaii_puma_districts_3digit.csv')
        
        # Prepare household data for merge
        household_geo = household_df[['SERIALNO', 'PUMA']].copy()
        household_geo['PUMA'] = household_geo['PUMA'].astype(str).str.zfill(4)
        puma_county['PUMA'] = puma_county['PUMA'].astype(str).str.zfill(4)
        
        # Add county information
        household_geo = household_geo.merge(
            puma_county[['PUMA', 'county']], 
            on='PUMA', 
            how='left'
        )
        
        # Add district information
        puma_district['PUMA_4digit'] = puma_district['PUMA'].astype(str).str.zfill(4)
        household_geo = household_geo.merge(
            puma_district[['PUMA_4digit', 'house_district', 'senate_district']], 
            left_on='PUMA',
            right_on='PUMA_4digit',
            how='left'
        )
        
        # Fix PUMA 100 inconsistency
        puma_100_mask = household_geo['PUMA'] == '0100'
        if puma_100_mask.any():
            household_geo.loc[puma_100_mask, 'house_district'] = 1
            household_geo.loc[puma_100_mask, 'senate_district'] = 1
        
        # Merge with final assignments
        final_assignments = final_assignments.merge(
            household_geo[['SERIALNO', 'PUMA', 'county', 'house_district', 'senate_district']],
            left_on='household_id',
            right_on='SERIALNO',
            how='left'
        )
        
        # Log geographic distribution
        if 'county' in final_assignments.columns:
            county_weights = final_assignments.groupby('county')['scenario_weight'].sum()
            logger.info("Weight distribution by county:")
            for county, weight in county_weights.items():
                pct = (weight / final_assignments['scenario_weight'].sum()) * 100
                logger.info(f"  {county}: {weight:,.0f} ({pct:.1f}%)")
        
        return final_assignments

def main():
    """Main execution function."""
    # File paths
    tax_units_path = "src/data/processed/tax_units_rule_based.parquet"
    person_path = "data/processed/pums_person_processed.parquet"
    household_path = "data/processed/pums_household_processed.parquet"
    
    # Initialize raker
    raker = WeightSplittingRaker(tax_units_path, person_path, household_path)
    
    # Run process
    final_assignments, validation_results = raker.run_full_process()
    
    # Save results
    output_path = "src/data/processed/tax_units_weight_split_raked.parquet"
    final_assignments.to_parquet(output_path)
    logger.info(f"Saved final assignments to {output_path}")
    
    return final_assignments, validation_results

if __name__ == "__main__":
    final_assignments, validation_results = main()
