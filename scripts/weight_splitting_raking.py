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
        """Load tax units, person, and household data."""
        logger.info("Loading data files...")
        
        tax_units = pd.read_parquet(self.tax_units_path)
        person_df = pd.read_parquet(self.person_path)
        household_df = pd.read_parquet(self.household_path)
        
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
            plausible_scenarios = self._get_plausible_scenarios(adults, hh_persons)
            
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
    
    def _get_plausible_scenarios(self, adults: pd.DataFrame, all_persons: pd.DataFrame) -> List[Dict]:
        """Determine plausible filing status scenarios for a household."""
        scenarios = []
        
        # Count married adults
        married_adults = adults[adults['MAR'] == 1]
        unmarried_adults = adults[adults['MAR'] != 1]
        children = all_persons[all_persons['AGEP'] < 18]
        
        num_married = len(married_adults)
        num_unmarried = len(unmarried_adults)
        num_children = len(children)
        
        # Scenario 1: All single filers
        if num_unmarried > 0:
            # Base probability for single filing
            single_prob = 0.4 if num_married == 0 else 0.2
            scenarios.append({'status': 'single', 'prob': single_prob})
        
        # Scenario 2: Joint filing (if married couples exist)
        if num_married >= 2:
            # Higher probability if clear married couple
            joint_prob = 0.6 if num_married == 2 else 0.4
            scenarios.append({'status': 'married_filing_jointly', 'prob': joint_prob})
        
        # Scenario 3: Head of Household (if unmarried adults with children)
        if num_unmarried > 0 and num_children > 0:
            # Check if potential HoH qualifiers exist
            hoh_prob = 0.3 if num_children > 0 else 0.1
            scenarios.append({'status': 'head_of_household', 'prob': hoh_prob})
        
        # Scenario 4: Married Filing Separately (if married couples)
        if num_married >= 2:
            # Higher probability for MFS to ensure it's represented
            mfs_prob = 0.15  # Increased from 0.05 to 0.15
            scenarios.append({'status': 'married_filing_separately', 'prob': mfs_prob})
        
        # Ensure at least one scenario exists
        if not scenarios:
            scenarios.append({'status': 'single', 'prob': 1.0})
        
        return scenarios
    
    def rake_weights(self, scenarios_df: pd.DataFrame) -> pd.DataFrame:
        """
        Iteratively rake scenario weights to match IRS target shares.
        """
        logger.info("Starting iterative raking process...")
        
        df = scenarios_df.copy()
        
        for iteration in range(self.max_iterations):
            # Calculate current shares
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
            
            # Apply raking adjustments
            for status in self.irs_targets:
                if status in current_shares and current_shares[status] > 0:
                    factor = self.irs_targets[status] / current_shares[status]
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
        
        # Debug: Check how many households chose each status
        assignment_counts = final_assignments['filing_status'].value_counts()
        logger.info("Final assignments by status:")
        for status, count in assignment_counts.items():
            logger.info(f"  {status}: {count} households")
        
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
        
        # Step 5: Validate results
        validation_results = self.validate_results(final_assignments)
        
        return final_assignments, validation_results

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
