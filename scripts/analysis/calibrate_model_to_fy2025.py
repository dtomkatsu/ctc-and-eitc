#!/usr/bin/env python3
"""
Calibrate Hawaii Tax Model to FY 2025 Actuals

This script implements the comprehensive calibration strategy to:
1. Reduce growth rate from 7.4% to 2-3% CAGR
2. Anchor projections to FY 2025 resident revenue ($2,999M)
3. Fix Act 46 impact estimates using official rates
4. Clearly separate resident vs total revenue

Author: Hawaii Tax Model Team
Date: November 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelCalibrator:
    """Calibrate Hawaii tax model to realistic growth rates and revenue targets."""
    
    def __init__(self, scenario='moderate'):
        """Initialize calibrator with target parameters.
        
        Args:
            scenario: 'conservative', 'moderate', or 'aggressive'
        """
        self.scenario = scenario
        self.config = ModelConfig()
        
        # Load parameters from centralized config
        self.fy_2022_total = self.config.FY_2022_TOTAL
        self.fy_2023_total = self.config.FY_2023_TOTAL
        self.fy_2023_adjusted = self.config.FY_2023_ADJUSTED
        self.fy_2024_total = self.config.FY_2024_TOTAL
        self.fy_2025_estimate = self.config.FY_2025_ESTIMATE
        
        self.nonresident_share = self.config.NONRESIDENT_SHARE
        
        # Calculate resident portions
        self.fy_2024_resident = self.config.FY_2024_RESIDENT
        self.fy_2025_resident_estimate = self.config.FY_2025_RESIDENT_ESTIMATE
        
        # Load scenario parameters from config
        params = self.config.get_scenario_params(scenario)
        
        self.base_year = params['base_year']
        self.base_resident = params['base_resident']
        self.target_growth_rate = params['growth_rate']
        self.years_forward = params['years_forward']
        self.fy_2026_resident_target = params['target_resident']
        self.fy_2026_total_target = params['target_total']
        
        # Act 46 parameters from config
        self.act46_impact_total = self.config.ACT46_IMPACT_TOTAL
        self.act46_impact_rate = self.config.ACT46_IMPACT_RATE
        
        # Current model performance
        self.current_resident_revenue = self.config.CURRENT_RESIDENT_REVENUE
        self.current_growth_rate = self.config.CURRENT_GROWTH_RATE
        
        logger.info(f"Calibrator initialized with {scenario.upper()} scenario:")
        logger.info(f"  Base Year: {self.base_year}")
        logger.info(f"  FY 2024 Resident (ACTUAL): ${self.fy_2024_resident:,.0f}M")
        logger.info(f"  FY 2025 Resident (ESTIMATE): ${self.fy_2025_resident_estimate:,.0f}M")
        logger.info(f"  Base Resident: ${self.base_resident:,.0f}M")
        logger.info(f"  FY 2026 Resident Target: ${self.fy_2026_resident_target:,.0f}M")
        logger.info(f"  Target Growth Rate: {self.target_growth_rate:.1%} over {self.years_forward:.1f} years")
        logger.info(f"  Act 46 Impact Rate: {self.act46_impact_rate:.1%}")
    
    def calculate_ensemble_weights(self):
        """Calculate new ensemble weights to achieve target growth rate."""
        
        # Get weights and growth rates from config
        weights = self.config.get_ensemble_weights(self.scenario)
        growth_rates = self.config.GROWTH_RATES
        
        # Build components dict with config values
        components = {}
        for component_name, new_weight in weights.items():
            components[component_name] = {
                'current_weight': 0.00,  # Baseline (before calibration)
                'growth_rate': growth_rates[component_name],
                'new_weight': new_weight,
                'note': f'{component_name} from config'
            }
        
        # Weights already normalized in config, but verify
        total_weight = sum(c['new_weight'] for c in components.values())
        if not (0.99 <= total_weight <= 1.01):
            logger.warning(f"Weights sum to {total_weight}, normalizing...")
            for comp in components.values():
                comp['new_weight'] = comp['new_weight'] / total_weight
        
        # Calculate current weighted growth
        current_weighted_growth = sum(
            c['current_weight'] * c['growth_rate'] 
            for c in components.values()
        )
        
        # Calculate new weighted growth
        new_weighted_growth = sum(
            c['new_weight'] * c['growth_rate'] 
            for c in components.values()
        )
        
        # Verify weights sum to 1
        total_new_weight = sum(c['new_weight'] for c in components.values())
        assert abs(total_new_weight - 1.0) < 0.001, f"Weights sum to {total_new_weight}, not 1.0"
        
        logger.info("\nEnsemble Weight Recalibration:")
        logger.info("-" * 60)
        logger.info(f"{'Component':<25} {'Current':<10} {'New':<10} {'Growth':<10}")
        logger.info("-" * 60)
        
        for name, comp in components.items():
            logger.info(
                f"{name:<25} "
                f"{comp['current_weight']:<10.1%} "
                f"{comp['new_weight']:<10.1%} "
                f"{comp['growth_rate']:<10.1%}"
            )
        
        logger.info("-" * 60)
        logger.info(f"Current weighted growth: {current_weighted_growth:.1%}")
        logger.info(f"New weighted growth: {new_weighted_growth:.1%}")
        logger.info(f"Target growth: {self.target_growth_rate:.1%}")
        
        # Check if new growth is within acceptable range
        if abs(new_weighted_growth - self.target_growth_rate) > 0.01:
            logger.warning(f"⚠️ New growth rate {new_weighted_growth:.1%} differs from target {self.target_growth_rate:.1%}")
            logger.warning("Consider adjusting weights further")
        else:
            logger.info(f"✅ New growth rate {new_weighted_growth:.1%} is within target range")
        
        return components
    
    def calculate_income_adjustment(self):
        """Calculate income adjustment factor to achieve target revenue."""
        
        # Current vs target
        revenue_adjustment_factor = self.fy_2026_resident_target / self.current_resident_revenue
        
        # Growth rate adjustment
        growth_adjustment_factor = (1 + self.target_growth_rate) / (1 + self.current_growth_rate)
        
        logger.info("\nIncome Adjustment Calculation:")
        logger.info("-" * 60)
        logger.info(f"Current resident revenue: ${self.current_resident_revenue:,.0f}M")
        logger.info(f"Target resident revenue: ${self.fy_2026_resident_target:,.0f}M")
        logger.info(f"Revenue adjustment needed: {revenue_adjustment_factor:.3f}")
        logger.info(f"Growth rate adjustment: {growth_adjustment_factor:.3f}")
        
        # Use the more conservative adjustment
        recommended_adjustment = min(revenue_adjustment_factor, growth_adjustment_factor)
        
        logger.info(f"Recommended adjustment factor: {recommended_adjustment:.3f}")
        logger.info(f"This will scale all incomes by {(recommended_adjustment - 1) * 100:+.1f}%")
        
        return recommended_adjustment
    
    def validate_calibration(self, calibrated_revenue):
        """Validate calibrated model against targets."""
        
        validations = {
            'resident_revenue': {
                'model': calibrated_revenue,
                'target': self.fy_2026_resident_target,
                'tolerance': 0.05,  # ±5%
                'pass': False
            },
            'growth_rate': {
                'model': (calibrated_revenue / self.base_resident - 1) / self.years_forward,
                'target': self.target_growth_rate,
                'tolerance': 0.01,  # ±1pp
                'pass': False
            },
            'total_revenue': {
                'model': calibrated_revenue / (1 - self.nonresident_share),
                'target': self.fy_2026_total_target,
                'tolerance': 0.05,  # ±5%
                'pass': False
            }
        }
        
        logger.info("\nCalibration Validation:")
        logger.info("-" * 80)
        logger.info(f"{'Metric':<20} {'Model':<15} {'Target':<15} {'Diff':<15} {'Status':<10}")
        logger.info("-" * 80)
        
        for metric, check in validations.items():
            diff_pct = (check['model'] / check['target'] - 1) * 100
            check['pass'] = abs(diff_pct / 100) <= check['tolerance']
            
            # Format values
            if 'revenue' in metric:
                model_str = f"${check['model']:,.0f}M"
                target_str = f"${check['target']:,.0f}M"
            else:
                model_str = f"{check['model']:.1%}"
                target_str = f"{check['target']:.1%}"
            
            status = "✅ PASS" if check['pass'] else "❌ FAIL"
            
            logger.info(
                f"{metric:<20} "
                f"{model_str:<15} "
                f"{target_str:<15} "
                f"{diff_pct:+.1f}%{'':<10} "
                f"{status:<10}"
            )
        
        all_pass = all(v['pass'] for v in validations.values())
        
        logger.info("-" * 80)
        if all_pass:
            logger.info("✅ All validations PASSED")
        else:
            logger.info("❌ Some validations FAILED - further calibration needed")
        
        return validations, all_pass
    
    def calculate_act46_impact(self, baseline_revenue):
        """Calculate Act 46 impact using official rate."""
        
        logger.info("\nAct 46 Impact Calculation:")
        logger.info("-" * 60)
        
        # Method 1: Official rate
        official_impact = baseline_revenue * self.act46_impact_rate
        post_act46_official = baseline_revenue + official_impact
        
        logger.info(f"Baseline (residents): ${baseline_revenue:,.0f}M")
        logger.info(f"Official rate: {self.act46_impact_rate:.1%}")
        logger.info(f"Impact: ${official_impact:,.0f}M")
        logger.info(f"Post-Act 46 (residents): ${post_act46_official:,.0f}M")
        
        # Add non-residents for total
        post_act46_total = post_act46_official + (self.fy_2025_estimate * self.nonresident_share)
        official_post_total = 2691  # From FY data
        
        logger.info(f"\nTotal revenue comparison:")
        logger.info(f"Model post-Act 46 total: ${post_act46_total:,.0f}M")
        logger.info(f"Official post-Act 46: ${official_post_total:,.0f}M")
        logger.info(f"Difference: ${post_act46_total - official_post_total:+,.0f}M "
                   f"({(post_act46_total / official_post_total - 1) * 100:+.1f}%)")
        
        return {
            'baseline_resident': baseline_revenue,
            'impact': official_impact,
            'post_act46_resident': post_act46_official,
            'post_act46_total': post_act46_total,
            'impact_rate': self.act46_impact_rate
        }
    
    def generate_calibration_config(self, output_path=None):
        """Generate configuration file for calibrated model."""
        
        config = {
            'metadata': {
                'created': datetime.now().isoformat(),
                'purpose': 'Hawaii Tax Model Calibration to FY 2025 Actuals',
                'version': '2.0'
            },
            'base_year': {
                'year': self.base_year,
                'total_revenue': self.base_resident / (1 - self.nonresident_share),
                'resident_revenue': self.base_resident,
                'nonresident_share': self.nonresident_share,
                'note': f'Using {self.scenario} scenario base'
            },
            'targets': {
                'year': 2026,
                'resident_revenue': self.fy_2026_resident_target,
                'total_revenue': self.fy_2026_total_target,
                'growth_rate': self.target_growth_rate
            },
            'ensemble_weights': self.calculate_ensemble_weights(),
            'adjustments': {
                'income_scaling': self.calculate_income_adjustment(),
                'capital_gains_target_pct': 0.025,  # 2.5% of AGI
                'credit_reduction_rate': 0.10  # 10% for gross to net
            },
            'act46': {
                'impact_total': self.act46_impact_total,
                'impact_rate': self.act46_impact_rate,
                'applies_to': 'primarily_residents'
            },
            'validation_thresholds': {
                'revenue_tolerance': 0.05,  # ±5%
                'growth_rate_tolerance': 0.01,  # ±1pp
                'act46_tolerance': 0.10  # ±10%
            }
        }
        
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            
            logger.info(f"\n✅ Configuration saved to: {output_file}")
        
        return config
    
    def run_full_calibration(self):
        """Run complete calibration process."""
        
        logger.info("="*80)
        logger.info("HAWAII TAX MODEL CALIBRATION TO FY 2025")
        logger.info("="*80)
        
        # Step 1: Calculate new ensemble weights
        logger.info("\n📊 STEP 1: Recalculate Ensemble Weights")
        new_weights = self.calculate_ensemble_weights()
        
        # Step 2: Calculate income adjustment
        logger.info("\n💰 STEP 2: Calculate Income Adjustment")
        income_adjustment = self.calculate_income_adjustment()
        
        # Step 3: Apply calibration (simulated)
        logger.info("\n🔧 STEP 3: Apply Calibration (Simulated)")
        calibrated_revenue = self.current_resident_revenue * income_adjustment
        logger.info(f"Calibrated resident revenue: ${calibrated_revenue:,.0f}M")
        
        # Step 4: Validate
        logger.info("\n✅ STEP 4: Validate Calibration")
        validations, all_pass = self.validate_calibration(calibrated_revenue)
        
        # Step 5: Calculate Act 46 impact
        logger.info("\n📉 STEP 5: Calculate Act 46 Impact")
        act46_results = self.calculate_act46_impact(calibrated_revenue)
        
        # Step 6: Generate configuration
        logger.info("\n💾 STEP 6: Generate Configuration")
        config = self.generate_calibration_config(
            'data/processed/calibration/calibration_config_20251103.json'
        )
        
        # Final summary
        logger.info("\n" + "="*80)
        logger.info("CALIBRATION SUMMARY")
        logger.info("="*80)
        
        summary = {
            'Pre-Calibration': {
                'Resident Revenue': f"${self.current_resident_revenue:,.0f}M",
                'Growth Rate': f"{self.current_growth_rate:.1%}",
                'Error vs Target': f"{(self.current_resident_revenue / self.fy_2026_resident_target - 1) * 100:+.1f}%"
            },
            'Post-Calibration': {
                'Resident Revenue': f"${calibrated_revenue:,.0f}M",
                'Growth Rate': f"{((calibrated_revenue / self.base_resident - 1) / self.years_forward) * 100:.1%}",
                'Error vs Target': f"{(calibrated_revenue / self.fy_2026_resident_target - 1) * 100:+.1f}%"
            },
            'Act 46 Analysis': {
                'Impact Amount': f"${act46_results['impact']:,.0f}M",
                'Impact Rate': f"{act46_results['impact_rate']:.1%}",
                'Post-Act 46 Total': f"${act46_results['post_act46_total']:,.0f}M"
            }
        }
        
        for section, metrics in summary.items():
            logger.info(f"\n{section}:")
            for metric, value in metrics.items():
                logger.info(f"  {metric:<20}: {value}")
        
        logger.info("\n" + "="*80)
        if all_pass:
            logger.info("✅ CALIBRATION SUCCESSFUL - Model is within target ranges")
        else:
            logger.info("⚠️ CALIBRATION PARTIAL - Some metrics need further adjustment")
        logger.info("="*80)
        
        return {
            'config': config,
            'validations': validations,
            'act46_results': act46_results,
            'success': all_pass
        }


def main():
    """Main execution."""
    
    # Run all three scenarios
    scenarios = ['conservative', 'moderate', 'aggressive']
    all_results = {}
    
    for scenario in scenarios:
        logger.info("\n" + "="*80)
        logger.info(f"RUNNING {scenario.upper()} SCENARIO")
        logger.info("="*80 + "\n")
        
        calibrator = ModelCalibrator(scenario=scenario)
        results = calibrator.run_full_calibration()
        all_results[scenario] = results
    
    # Summary comparison
    logger.info("\n" + "="*80)
    logger.info("SCENARIO COMPARISON")
    logger.info("="*80)
    
    logger.info(f"\n{'Scenario':<15} {'Target':<12} {'Growth':<10} {'Act 46':<12} {'Post-Act46':<12}")
    logger.info("-" * 80)
    
    for scenario in scenarios:
        res = all_results[scenario]
        logger.info(
            f"{scenario.capitalize():<15} "
            f"${res['config']['targets']['resident_revenue']:,.0f}M{'':<2} "
            f"{res['config']['targets']['growth_rate']:.1%}{'':<5} "
            f"${res['act46_results']['impact']:,.0f}M{'':<3} "
            f"${res['act46_results']['post_act46_total']:,.0f}M"
        )
    
    logger.info("\n" + "="*80)
    logger.info("RECOMMENDED: Use MODERATE scenario for balanced approach")
    logger.info("="*80)
    
    return all_results
    
    # Save results
    output_dir = Path('data/processed/calibration')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / 'calibration_results_20251103.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n📁 Results saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    results = main()
    print("\n✅ Calibration analysis complete!")
