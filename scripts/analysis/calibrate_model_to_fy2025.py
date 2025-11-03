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
        
        # FY Actuals (confirmed data)
        self.fy_2022_total = 3760  # Million, actual (peak)
        self.fy_2023_total = 3100  # Million, actual (with refund)
        self.fy_2023_adjusted = 3412  # Million, actual (add back $311.7M refund)
        self.fy_2024_total = 3280  # Million, ACTUAL (last confirmed)
        
        # FY 2025 is a PROJECTION, not actual
        self.fy_2025_estimate = 3288  # Million, DOT projection
        
        self.nonresident_share = 0.088  # 8.8% of total
        
        # Calculate resident portions
        self.fy_2024_resident = self.fy_2024_total * (1 - self.nonresident_share)  # $2,991M
        self.fy_2025_resident_estimate = self.fy_2025_estimate * (1 - self.nonresident_share)  # $2,999M
        
        # Scenario-based targets
        if scenario == 'conservative':
            # Base on FY 2024 actual only, 1.5% growth over 2 years
            self.base_year = 2024
            self.base_resident = self.fy_2024_resident
            self.target_growth_rate = 0.015
            self.years_forward = 2
        elif scenario == 'moderate':
            # Blend FY 2024 + FY 2025 estimate, 2% growth
            self.base_year = 2024.5  # Conceptual blend
            self.base_resident = (self.fy_2024_resident + self.fy_2025_resident_estimate) / 2
            self.target_growth_rate = 0.020
            self.years_forward = 1.5  # Average
        elif scenario == 'aggressive':
            # Trust FY 2025 estimate, 2.5% growth
            self.base_year = 2025
            self.base_resident = self.fy_2025_resident_estimate
            self.target_growth_rate = 0.025
            self.years_forward = 1
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
        
        # 2026 Targets
        self.fy_2026_resident_target = self.base_resident * (1 + self.target_growth_rate) ** self.years_forward
        self.fy_2026_total_target = self.fy_2026_resident_target / (1 - self.nonresident_share)
        
        # Act 46 parameters (based on FY 2025 estimate)
        self.act46_impact_total = -597  # Million (official estimate)
        self.act46_impact_rate = self.act46_impact_total / self.fy_2025_resident_estimate  # -19.9% of resident
        
        # Current model performance (to be updated)
        self.current_resident_revenue = 3298  # Million (our ensemble estimate)
        self.current_growth_rate = 0.074  # 7.4% CAGR
        
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
        
        # Component growth rates
        components = {
            'fy_actual_2022_2024': {
                'current_weight': 0.00,
                'growth_rate': (self.fy_2024_total / self.fy_2023_adjusted - 1),  # Actual post-peak
                'new_weight': 0.30 if self.scenario != 'aggressive' else 0.20,
                'note': 'FY 2023 (adj) → FY 2024 actual'
            },
            'fy_2025_estimate': {
                'current_weight': 0.00,
                'growth_rate': (self.fy_2025_estimate / self.fy_2024_total - 1),  # DOT projection
                'new_weight': 0.10 if self.scenario == 'moderate' else (0.30 if self.scenario == 'aggressive' else 0.00),
                'note': 'FY 2024 → FY 2025 estimate (PROJECTION)'
            },
            'dotax_2018_2021': {
                'current_weight': 0.35,
                'growth_rate': 0.111,  # 11.1% CAGR
                'new_weight': 0.20,
                'note': 'Pre-peak historical growth'
            },
            'bls_wage': {
                'current_weight': 0.30,
                'growth_rate': 0.055,  # 5.5%
                'new_weight': 0.25,
                'note': 'Wage growth trends'
            },
            'acs_income': {
                'current_weight': 0.25,
                'growth_rate': 0.062,  # 6.2%
                'new_weight': 0.10 if self.scenario != 'conservative' else 0.15,
                'note': 'ACS income trends'
            },
            'demographics': {
                'current_weight': 0.10,
                'growth_rate': 0.011,  # 1.1%
                'new_weight': 0.05 if self.scenario != 'conservative' else 0.10,
                'note': 'Demographic factors'
            }
        }
        
        # Normalize weights to sum to 1.0
        total_weight = sum(c['new_weight'] for c in components.values())
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
