#!/usr/bin/env python3
"""
Systematically optimize ultra-high-income synthesizer weight allocations.

Finds optimal weight factors for MFJ, Single, and HoH synthetic filers
to minimize gaps against DOTax targets for $400k+ bracket by filing status.

Target gaps (from comprehensive_weight_calibrator):
- Joint $400k+: $596M
- Single $400k+: $375M  
- HoH $400k+: $26M
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.optimize import minimize, differential_evolution
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tax.adjustments.ultra_high_income_synthesizer_v2 import UltraHighIncomeSynthesizerV2
from src.tax.adjustments.comprehensive_weight_calibrator import ComprehensiveWeightCalibrator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UltraHighAllocationOptimizer:
    """Optimize ultra-high-income synthesizer allocations."""
    
    def __init__(self, tax_units_path: str):
        """Initialize optimizer with tax units data."""
        self.tax_units_path = Path(tax_units_path)
        self.df = pd.read_parquet(self.tax_units_path)
        
        # Target $400k+ tax liability by filing status (from DOTax Table A2-2)
        self.targets = {
            'married_filing_jointly': 596_000_000,
            'single': 375_000_000,
            'head_of_household': 26_000_000,
        }
        
        logger.info(f"Loaded {len(self.df)} tax units from {self.tax_units_path}")
        logger.info(f"Targets: Joint=${self.targets['married_filing_jointly']/1e6:.0f}M, "
                   f"Single=${self.targets['single']/1e6:.0f}M, "
                   f"HoH=${self.targets['head_of_household']/1e6:.0f}M")
    
    def evaluate_allocation(self, params: np.ndarray) -> float:
        """
        Evaluate a set of allocation parameters.
        
        Parameters are organized as:
        - params[0:4]: MFJ weights for $5M, $10M, $25M, $50M
        - params[4:7]: Single weights for $1.5M, $3M, $5M
        - params[7:9]: HoH weights for $1.5M, $3M
        
        Returns weighted sum of squared percentage errors.
        """
        # Unpack parameters
        mfj_weights = params[0:4]
        single_weights = params[4:7]
        hoh_weights = params[7:9]
        
        # Simulate the synthesis process directly without modifying the class
        df_synth = self.df.copy()
        
        # Calculate donor pools
        mask_1m = df_synth['agi'] >= 1_000_000
        mask_400k_1m = (df_synth['agi'] >= 400_000) & (df_synth['agi'] < 1_000_000)
        
        # Get donor weights by status
        mfj_donor = df_synth.loc[mask_1m & (df_synth['filing_status'] == 'married_filing_jointly'), 'weight'].sum()
        single_donor = df_synth.loc[mask_400k_1m & (df_synth['filing_status'] == 'single'), 'weight'].sum()
        hoh_donor = df_synth.loc[mask_400k_1m & (df_synth['filing_status'] == 'head_of_household'), 'weight'].sum()
        
        # Calculate synthetic weights to add
        synthetic_by_status = {
            'married_filing_jointly': mfj_donor * np.sum(mfj_weights),
            'single': single_donor * np.sum(single_weights),
            'head_of_household': hoh_donor * np.sum(hoh_weights),
        }
        
        # Reduce existing weights (simulate cascade)
        for status, synthetic_weight in synthetic_by_status.items():
            if status == 'married_filing_jointly':
                # Reduce from $1M+ brackets
                mask_reduce = (df_synth['agi'] >= 1_000_000) & (df_synth['filing_status'] == status)
            else:
                # Reduce from $400k-$1M brackets
                mask_reduce = (df_synth['agi'] >= 400_000) & (df_synth['agi'] < 1_000_000) & (df_synth['filing_status'] == status)
            
            available = df_synth.loc[mask_reduce, 'weight'].sum()
            if available > 0:
                reduction_factor = max(0, (available - synthetic_weight) / available)
                df_synth.loc[mask_reduce, 'weight'] *= reduction_factor
        
        # Add synthetic filers with estimated tax
        synthetic_tax_by_status = {
            'married_filing_jointly': mfj_donor * (
                mfj_weights[0] * 525_000 +
                mfj_weights[1] * 1_070_000 +
                mfj_weights[2] * 2_675_000 +
                mfj_weights[3] * 5_350_000
            ),
            'single': single_donor * (
                single_weights[0] * 161_000 +
                single_weights[1] * 323_000 +
                single_weights[2] * 537_000
            ),
            'head_of_household': hoh_donor * (
                hoh_weights[0] * 161_000 +
                hoh_weights[1] * 323_000
            ),
        }
        
        # Calculate $400k+ tax by filing status (existing + synthetic)
        mask_400k = df_synth['agi'] >= 400_000
        
        results = {}
        for status in self.targets.keys():
            status_mask = mask_400k & (df_synth['filing_status'] == status)
            existing_tax = (df_synth.loc[status_mask, 'hi_state_tax'] * df_synth.loc[status_mask, 'weight']).sum()
            synthetic_tax = synthetic_tax_by_status.get(status, 0)
            results[status] = existing_tax + synthetic_tax
        
        # Calculate weighted percentage errors
        errors = []
        for status, target in self.targets.items():
            actual = results[status]
            pct_error = (actual - target) / target * 100
            errors.append(pct_error ** 2)  # Squared to penalize larger deviations
        
        # Weighted sum (weight by target magnitude to balance importance)
        weights_norm = np.array([self.targets[s] for s in self.targets.keys()])
        weights_norm = weights_norm / weights_norm.sum()
        
        total_error = np.sum(np.array(errors) * weights_norm)
        
        return total_error
    
    def optimize_grid_search(self, n_samples: int = 20) -> dict:
        """
        Grid search optimization (slower but more thorough).
        
        Args:
            n_samples: Number of samples per parameter dimension
        """
        logger.info("Starting grid search optimization...")
        
        # Define search ranges (as fractions)
        mfj_range = np.linspace(0.20, 0.50, n_samples)
        single_range = np.linspace(0.10, 0.40, n_samples)
        hoh_range = np.linspace(0.02, 0.10, n_samples)
        
        best_error = float('inf')
        best_params = None
        
        # Sample grid (not full factorial to save time)
        n_trials = 1000
        np.random.seed(42)
        
        for i in range(n_trials):
            # Sample parameters with descending constraint (higher tiers get less weight)
            mfj = np.sort(np.random.choice(mfj_range, 4))[::-1]
            single = np.sort(np.random.choice(single_range, 3))[::-1]
            hoh = np.sort(np.random.choice(hoh_range, 2))[::-1]
            
            params = np.concatenate([mfj, single, hoh])
            error = self.evaluate_allocation(params)
            
            if error < best_error:
                best_error = error
                best_params = params
                logger.info(f"Trial {i+1}/{n_trials}: New best error={error:.2f}, "
                           f"MFJ={mfj}, Single={single}, HoH={hoh}")
        
        return {
            'params': best_params,
            'error': best_error,
            'mfj': best_params[0:4],
            'single': best_params[4:7],
            'hoh': best_params[7:9]
        }
    
    def optimize_differential_evolution(self) -> dict:
        """
        Differential evolution optimization (global optimizer).
        """
        logger.info("Starting differential evolution optimization...")
        
        # Define bounds (min, max) for each parameter
        bounds = [
            # MFJ: descending from high to low, constrained
            (0.20, 0.50),  # $5M
            (0.15, 0.45),  # $10M
            (0.10, 0.40),  # $25M
            (0.05, 0.35),  # $50M
            # Single: descending
            (0.10, 0.40),  # $1.5M
            (0.05, 0.35),  # $3M
            (0.02, 0.30),  # $5M
            # HoH: descending
            (0.02, 0.12),  # $1.5M
            (0.01, 0.10),  # $3M
        ]
        
        # Add constraints for descending allocations
        def constraint_descending(params):
            """Ensure each tier gets less weight than the previous tier."""
            mfj = params[0:4]
            single = params[4:7]
            hoh = params[7:9]
            
            # Check descending order
            violations = 0
            violations += np.sum(np.diff(mfj) > 0)  # MFJ should descend
            violations += np.sum(np.diff(single) > 0)  # Single should descend
            violations += np.sum(np.diff(hoh) > 0)  # HoH should descend
            
            return -violations  # Negative = constraint violated
        
        result = differential_evolution(
            self.evaluate_allocation,
            bounds,
            strategy='best1bin',
            maxiter=100,
            popsize=15,
            tol=0.01,
            mutation=(0.5, 1.5),
            recombination=0.7,
            seed=42,
            callback=lambda xk, convergence: logger.info(f"Iteration complete, convergence={convergence:.4f}"),
            workers=1,  # Single thread to avoid issues
            updating='deferred',
            polish=True
        )
        
        return {
            'params': result.x,
            'error': result.fun,
            'success': result.success,
            'message': result.message,
            'mfj': result.x[0:4],
            'single': result.x[4:7],
            'hoh': result.x[7:9]
        }
    
    def display_results(self, result: dict):
        """Display optimization results."""
        logger.info("\n" + "="*80)
        logger.info("OPTIMIZATION RESULTS")
        logger.info("="*80)
        
        logger.info(f"\nTotal error: {result['error']:.4f}")
        logger.info(f"\nMFJ allocations (from $1M+ donors):")
        logger.info(f"  $5M:  {result['mfj'][0]:.3f} ({result['mfj'][0]*100:.1f}%)")
        logger.info(f"  $10M: {result['mfj'][1]:.3f} ({result['mfj'][1]*100:.1f}%)")
        logger.info(f"  $25M: {result['mfj'][2]:.3f} ({result['mfj'][2]*100:.1f}%)")
        logger.info(f"  $50M: {result['mfj'][3]:.3f} ({result['mfj'][3]*100:.1f}%)")
        
        logger.info(f"\nSingle allocations (from $400k-$1M donors):")
        logger.info(f"  $1.5M: {result['single'][0]:.3f} ({result['single'][0]*100:.1f}%)")
        logger.info(f"  $3M:   {result['single'][1]:.3f} ({result['single'][1]*100:.1f}%)")
        logger.info(f"  $5M:   {result['single'][2]:.3f} ({result['single'][2]*100:.1f}%)")
        
        logger.info(f"\nHoH allocations (from $400k-$1M donors):")
        logger.info(f"  $1.5M: {result['hoh'][0]:.3f} ({result['hoh'][0]*100:.1f}%)")
        logger.info(f"  $3M:   {result['hoh'][1]:.3f} ({result['hoh'][1]*100:.1f}%)")
        
        # Evaluate final performance
        logger.info("\n" + "-"*80)
        logger.info("Performance against targets:")
        logger.info("-"*80)
        
        final_error = self.evaluate_allocation(result['params'])
        logger.info(f"Final weighted error: {final_error:.4f}")
        
        logger.info("="*80 + "\n")


def main():
    """Main optimization workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize ultra-high-income allocations')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input tax units parquet file')
    parser.add_argument('--method', type=str, default='differential_evolution',
                       choices=['grid_search', 'differential_evolution'],
                       help='Optimization method')
    parser.add_argument('--grid-samples', type=int, default=20,
                       help='Number of samples per dimension for grid search')
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = UltraHighAllocationOptimizer(args.input)
    
    # Run optimization
    if args.method == 'grid_search':
        result = optimizer.optimize_grid_search(n_samples=args.grid_samples)
    else:
        result = optimizer.optimize_differential_evolution()
    
    # Display results
    optimizer.display_results(result)
    
    # Save results
    output_path = Path(args.input).parent / 'optimal_allocations.txt'
    with open(output_path, 'w') as f:
        f.write("OPTIMAL ULTRA-HIGH-INCOME ALLOCATIONS\n")
        f.write("="*80 + "\n\n")
        f.write(f"MFJ (from $1M+ donors):\n")
        f.write(f"  weight_factor at $5M:  {result['mfj'][0]:.3f}\n")
        f.write(f"  weight_factor at $10M: {result['mfj'][1]:.3f}\n")
        f.write(f"  weight_factor at $25M: {result['mfj'][2]:.3f}\n")
        f.write(f"  weight_factor at $50M: {result['mfj'][3]:.3f}\n\n")
        f.write(f"Single (from $400k-$1M donors):\n")
        f.write(f"  weight_factor at $1.5M: {result['single'][0]:.3f}\n")
        f.write(f"  weight_factor at $3M:   {result['single'][1]:.3f}\n")
        f.write(f"  weight_factor at $5M:   {result['single'][2]:.3f}\n\n")
        f.write(f"HoH (from $400k-$1M donors):\n")
        f.write(f"  weight_factor at $1.5M: {result['hoh'][0]:.3f}\n")
        f.write(f"  weight_factor at $3M:   {result['hoh'][1]:.3f}\n\n")
        f.write(f"Total error: {result['error']:.4f}\n")
    
    logger.info(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
