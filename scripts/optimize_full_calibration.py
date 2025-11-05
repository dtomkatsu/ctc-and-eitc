#!/usr/bin/env python3
"""
Comprehensive Tax Calibration Optimizer

Systematically optimizes all calibration stages to align filer counts, AGI, 
and tax liability across all brackets with DOTAX Table A8 targets.

KEY DIFFERENCE FROM PREVIOUS OPTIMIZER:
- Runs ENTIRE regenerate_tax_units.py pipeline for each evaluation
- Evaluates against FINAL output after all calibration stages
- Avoids the pre-calibration data evaluation issue

Usage:
    python scripts/optimize_full_calibration.py --iterations 30 --method differential_evolution
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution, minimize
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import all pipeline components
from src.tax.units.constructor import TaxUnitConstructor
from src.tax.calculator import HawaiiTaxCalculator
from src.tax.deductions.itemized_estimator import apply_itemized_deduction_reduction
from src.tax.adjustments.pareto_calibration import apply_pareto_calibration
from src.tax.adjustments.income_distribution_calibrator import apply_income_distribution_calibration
from src.tax.adjustments.comprehensive_weight_calibrator import apply_comprehensive_weight_calibration
from src.tax.adjustments.ultra_high_income_synthesizer_v2 import UltraHighIncomeSynthesizerV2
from src.tax.adjustments.final_gap_closer import apply_final_gap_closer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FullCalibrationOptimizer:
    """Optimize all calibration parameters across the full pipeline."""
    
    def __init__(self):
        """Initialize optimizer with DOTAX targets."""
        self.dotax_targets = self._load_dotax_targets()
        self.evaluation_count = 0
        self.best_error = float('inf')
        self.best_params = None
        
        logger.info("Initialized FullCalibrationOptimizer")
        logger.info(f"Loaded targets for {len(self.dotax_targets)} brackets")
    
    def _load_dotax_targets(self):
        """Load DOTAX Table A8 targets by bracket."""
        # DOTAX 2022 Table A8 targets (in millions for AGI and tax)
        targets = [
            {'bracket': '$0-$10k', 'min': 0, 'max': 10_000, 
             'filers': 129_257, 'agi_m': 0.618, 'tax_m': 3.0, 'weight': 0.5},
            {'bracket': '$10-$20k', 'min': 10_000, 'max': 20_000, 
             'filers': 64_037, 'agi_m': 0.945, 'tax_m': 21.0, 'weight': 0.5},
            {'bracket': '$20-$30k', 'min': 20_000, 'max': 30_000, 
             'filers': 57_821, 'agi_m': 1.444, 'tax_m': 51.0, 'weight': 0.7},
            {'bracket': '$30-$40k', 'min': 30_000, 'max': 40_000, 
             'filers': 59_992, 'agi_m': 2.093, 'tax_m': 92.0, 'weight': 0.8},
            {'bracket': '$40-$50k', 'min': 40_000, 'max': 50_000, 
             'filers': 53_533, 'agi_m': 2.388, 'tax_m': 116.0, 'weight': 0.9},
            {'bracket': '$50-$75k', 'min': 50_000, 'max': 75_000, 
             'filers': 91_537, 'agi_m': 5.666, 'tax_m': 293.0, 'weight': 1.0},
            {'bracket': '$75-$100k', 'min': 75_000, 'max': 100_000, 
             'filers': 54_945, 'agi_m': 4.729, 'tax_m': 261.0, 'weight': 1.0},
            {'bracket': '$100-$150k', 'min': 100_000, 'max': 150_000, 
             'filers': 62_066, 'agi_m': 7.474, 'tax_m': 438.0, 'weight': 1.2},
            {'bracket': '$150-$200k', 'min': 150_000, 'max': 200_000, 
             'filers': 28_013, 'agi_m': 4.823, 'tax_m': 294.0, 'weight': 1.3},
            {'bracket': '$200-$300k', 'min': 200_000, 'max': 300_000, 
             'filers': 18_915, 'agi_m': 4.566, 'tax_m': 310.0, 'weight': 1.5},
            {'bracket': '$300-$400k', 'min': 300_000, 'max': 400_000, 
             'filers': 6_076, 'agi_m': 2.085, 'tax_m': 153.0, 'weight': 1.7},
            {'bracket': '$400k+', 'min': 400_000, 'max': float('inf'), 
             'filers': 8_875, 'agi_m': 8.800, 'tax_m': 998.0, 'weight': 2.0},
        ]
        return targets
    
    def run_full_pipeline(self, params):
        """
        Run complete regenerate_tax_units.py pipeline with custom parameters.
        
        This is the KEY method - it runs the ENTIRE pipeline to avoid
        the pre-calibration evaluation issue.
        """
        try:
            # 1. Construct tax units from PUMS
            logger.debug("Step 1: Constructing tax units from PUMS...")
            pums_path = project_root / 'data' / 'raw' / 'hawaii_pums_2022.csv'
            constructor = TaxUnitConstructor(str(pums_path))
            tax_units = constructor.create_tax_units()
            
            # 2. Apply itemized deduction reduction (parameterized)
            logger.debug("Step 2: Applying itemized deduction reduction...")
            # For now, use fixed reduction (can be parameterized later)
            tax_units['total_deductions'] = apply_itemized_deduction_reduction(tax_units)
            
            # 3. Calculate initial taxes
            logger.debug("Step 3: Calculating initial taxes...")
            calculator = HawaiiTaxCalculator(tax_year=2022)
            tax_units = calculator.calculate_tax_units_batch(tax_units)
            
            # 4. Apply Pareto calibration (fixed targets from DOTAX)
            logger.debug("Step 4: Applying Pareto calibration...")
            tax_units = apply_pareto_calibration(
                tax_units,
                threshold=200_000,
                calibrate_all_brackets=True,
                add_synthetic=False,
            )
            
            # 5. Apply income distribution calibration
            logger.debug("Step 5: Applying income distribution calibration...")
            tax_units = apply_income_distribution_calibration(
                tax_units,
                threshold=100_000,
                recalculate_tax=False,
                method='percentile',
            )
            
            # 6. Apply weight calibration (can be parameterized)
            logger.debug("Step 6: Applying weight calibration...")
            tax_units = apply_comprehensive_weight_calibration(
                tax_units,
                calibrate_all_brackets=False,
            )
            
            # 7. Apply ultra-high-income synthesis (keep current optimal config)
            logger.debug("Step 7: Applying ultra-high-income synthesis...")
            synthesizer = UltraHighIncomeSynthesizerV2(
                pareto_alpha=1.064,
                tail_multiplier=0.58
            )
            tax_units = synthesizer.redistribute_within_million_plus(tax_units, target_tax_m=663.0)
            
            # 8. Recalculate taxes after all adjustments
            logger.debug("Step 8: Recalculating taxes...")
            tax_units = calculator.calculate_tax_units_batch(tax_units)
            
            # 9. Apply final gap closer (can be parameterized)
            logger.debug("Step 9: Applying final gap closer...")
            tax_units = apply_final_gap_closer(tax_units)
            
            # 10. Final tax calculation
            logger.debug("Step 10: Final tax calculation...")
            tax_units = calculator.calculate_tax_units_batch(tax_units)
            
            return tax_units
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise
    
    def evaluate(self, params_array):
        """
        Evaluate parameter set against DOTAX targets.
        
        Returns weighted sum of squared percentage errors across all brackets
        for filer counts, AGI, and tax liability.
        """
        self.evaluation_count += 1
        
        # Convert params array to dict (for now, we'll start with a simple approach)
        # TODO: Expand this to parameterize more stages
        params = {
            'placeholder': params_array[0] if len(params_array) > 0 else 1.0
        }
        
        try:
            # Run full pipeline
            logger.info(f"Evaluation {self.evaluation_count}: Running full pipeline...")
            tax_units = self.run_full_pipeline(params)
            
            # Calculate errors for each bracket
            total_error = 0
            bracket_errors = []
            
            for target in self.dotax_targets:
                bracket_name = target['bracket']
                
                # Create bracket mask
                if target['max'] == float('inf'):
                    mask = tax_units['agi'] >= target['min']
                else:
                    mask = (tax_units['agi'] >= target['min']) & (tax_units['agi'] < target['max'])
                
                # Calculate model values
                model_filers = tax_units.loc[mask, 'weight'].sum()
                model_agi_m = (tax_units.loc[mask, 'agi'] * tax_units.loc[mask, 'weight']).sum() / 1_000_000
                model_tax_m = (tax_units.loc[mask, 'hi_state_tax'] * tax_units.loc[mask, 'weight']).sum() / 1_000_000
                
                # Calculate percentage errors
                filer_error = ((model_filers - target['filers']) / target['filers']) ** 2 if target['filers'] > 0 else 0
                agi_error = ((model_agi_m - target['agi_m']) / target['agi_m']) ** 2 if target['agi_m'] > 0 else 0
                tax_error = ((model_tax_m - target['tax_m']) / target['tax_m']) ** 2 if target['tax_m'] > 0 else 0
                
                # Weighted combination (tax liability most important)
                bracket_error = (0.3 * filer_error + 0.3 * agi_error + 0.4 * tax_error) * target['weight']
                
                total_error += bracket_error
                bracket_errors.append({
                    'bracket': bracket_name,
                    'error': bracket_error,
                    'filer_pct': (model_filers - target['filers']) / target['filers'] * 100,
                    'agi_pct': (model_agi_m - target['agi_m']) / target['agi_m'] * 100,
                    'tax_pct': (model_tax_m - target['tax_m']) / target['tax_m'] * 100,
                })
            
            # Log progress
            if total_error < self.best_error:
                self.best_error = total_error
                self.best_params = params
                logger.info(f"  NEW BEST! Total error: {total_error:.4f}")
                logger.info(f"  Top 3 worst brackets:")
                sorted_errors = sorted(bracket_errors, key=lambda x: x['error'], reverse=True)[:3]
                for be in sorted_errors:
                    logger.info(f"    {be['bracket']}: tax={be['tax_pct']:+.1f}%, agi={be['agi_pct']:+.1f}%, filers={be['filer_pct']:+.1f}%")
            else:
                logger.info(f"  Total error: {total_error:.4f} (best: {self.best_error:.4f})")
            
            return total_error
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return 1e10  # Large penalty for failed evaluations
    
    def optimize(self, method='differential_evolution', max_iterations=30):
        """
        Run optimization to find best parameter set.
        
        Args:
            method: Optimization method ('differential_evolution' or 'minimize')
            max_iterations: Maximum iterations for optimization
        """
        logger.info(f"Starting optimization with method={method}, max_iterations={max_iterations}")
        logger.info("="*80)
        
        # For initial version, use a simple 1D parameter space
        # TODO: Expand to multi-dimensional parameter space
        bounds = [(0.5, 1.5)]  # Placeholder bounds
        
        if method == 'differential_evolution':
            result = differential_evolution(
                self.evaluate,
                bounds,
                strategy='best1bin',
                maxiter=max_iterations,
                popsize=15,
                tol=0.01,
                mutation=(0.5, 1.5),
                recombination=0.7,
                seed=42,
                workers=1,  # Must be 1 to avoid conflicts
                updating='deferred',
                polish=True,
                callback=lambda xk, convergence: logger.info(f"Iteration complete, convergence={convergence:.4f}")
            )
            
            return {
                'success': result.success,
                'error': result.fun,
                'params': result.x,
                'message': result.message,
                'evaluations': self.evaluation_count
            }
        
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def display_results(self, result):
        """Display optimization results."""
        logger.info("\n" + "="*80)
        logger.info("OPTIMIZATION RESULTS")
        logger.info("="*80)
        logger.info(f"\nSuccess: {result['success']}")
        logger.info(f"Final error: {result['error']:.4f}")
        logger.info(f"Total evaluations: {result['evaluations']}")
        logger.info(f"Message: {result['message']}")
        logger.info("\nBest parameters:")
        logger.info(f"  {result['params']}")
        logger.info("="*80 + "\n")


def main():
    """Main optimization workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize full tax calibration pipeline')
    parser.add_argument('--iterations', type=int, default=30,
                       help='Maximum optimization iterations')
    parser.add_argument('--method', type=str, default='differential_evolution',
                       choices=['differential_evolution', 'minimize'],
                       help='Optimization method')
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = FullCalibrationOptimizer()
    
    # Run optimization
    result = optimizer.optimize(
        method=args.method,
        max_iterations=args.iterations
    )
    
    # Display results
    optimizer.display_results(result)
    
    # Save results
    output_path = project_root / 'data' / 'processed' / 'full_calibration_optimization_results.json'
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'result': {
                'success': result['success'],
                'error': float(result['error']),
                'params': result['params'].tolist() if hasattr(result['params'], 'tolist') else result['params'],
                'message': result['message'],
                'evaluations': result['evaluations']
            }
        }, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
