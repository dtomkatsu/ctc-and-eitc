"""
Centralized configuration for Hawaii Tax Model.

This module consolidates all model parameters, paths, and settings
into a single source of truth. All scripts should import from here
rather than hardcoding values.

Usage:
    from config import ModelConfig
    
    # Get current scenario parameters
    config = ModelConfig()
    weights = config.ENSEMBLE_WEIGHTS
    
    # Get specific scenario
    params = ModelConfig.get_scenario_params('conservative')
"""

from pathlib import Path
from typing import Dict, Any


class ModelConfig:
    """Centralized configuration for Hawaii tax model."""
    
    # =========================================================================
    # PROJECT PATHS
    # =========================================================================
    
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # Data directories
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    EXTERNAL_DATA_DIR = DATA_DIR / "external"
    
    # PUMS data
    PUMS_DIR = RAW_DATA_DIR / "pums"
    PUMS_PERSON_FILE = PUMS_DIR / "csv_phi.zip"
    PUMS_HOUSEHOLD_FILE = PUMS_DIR / "csv_hhi.zip"
    
    # Output directories
    CALIBRATION_OUTPUT_DIR = PROCESSED_DATA_DIR / "calibration"
    PROJECTION_OUTPUT_DIR = PROCESSED_DATA_DIR / "projections"
    TAX_UNITS_OUTPUT_DIR = PROCESSED_DATA_DIR / "tax_units"
    
    # Documentation
    DOCS_DIR = PROJECT_ROOT / "docs"
    
    # Logs
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    # =========================================================================
    # MODEL SCENARIO (Change this to switch scenarios)
    # =========================================================================
    
    ACTIVE_SCENARIO = 'moderate'  # 'conservative', 'moderate', or 'aggressive'
    
    # =========================================================================
    # FISCAL YEAR DATA (Confirmed Actuals)
    # =========================================================================
    
    # Historical actuals
    FY_2022_TOTAL = 3760  # Million (peak - COVID recovery)
    FY_2023_TOTAL = 3100  # Million (with constitutional refund)
    FY_2023_ADJUSTED = 3412  # Million (add back $311.7M refund)
    FY_2024_TOTAL = 3280  # Million - LAST CONFIRMED ACTUAL
    
    # FY 2025 is a PROJECTION, not actual
    FY_2025_ESTIMATE = 3288  # Million (DOT projection)
    
    # =========================================================================
    # REVENUE SHARES
    # =========================================================================
    
    NONRESIDENT_SHARE = 0.088  # 8.8% of total revenue
    RESIDENT_SHARE = 0.912     # 91.2% of total revenue
    
    # Calculate resident portions
    FY_2024_RESIDENT = FY_2024_TOTAL * RESIDENT_SHARE  # $2,991M
    FY_2025_RESIDENT_ESTIMATE = FY_2025_ESTIMATE * RESIDENT_SHARE  # $2,999M
    
    # =========================================================================
    # ENSEMBLE WEIGHTS (Scenario-Specific)
    # =========================================================================
    
    # Moderate scenario (RECOMMENDED)
    ENSEMBLE_WEIGHTS_MODERATE = {
        'fy_actual_2022_2024': 0.30,   # Confirmed actuals (PRIMARY)
        'fy_2025_estimate': 0.10,       # DOT estimate (use cautiously)
        'dotax_2018_2021': 0.20,        # Historical growth period
        'bls_wage': 0.25,               # Wage trends
        'acs_income': 0.10,             # Income trends
        'demographics': 0.05            # Structural factors
    }
    
    # Conservative scenario
    ENSEMBLE_WEIGHTS_CONSERVATIVE = {
        'fy_actual_2022_2024': 0.30,   # Confirmed actuals (PRIMARY)
        'fy_2025_estimate': 0.00,       # Don't use projection
        'dotax_2018_2021': 0.20,        # Historical growth
        'bls_wage': 0.25,               # Wage trends
        'acs_income': 0.15,             # Income trends
        'demographics': 0.10            # Structural
    }
    
    # Aggressive scenario
    ENSEMBLE_WEIGHTS_AGGRESSIVE = {
        'fy_actual_2022_2024': 0.20,   # Confirmed actuals
        'fy_2025_estimate': 0.30,       # Trust DOT projection
        'dotax_2018_2021': 0.20,        # Historical growth
        'bls_wage': 0.20,               # Wage trends
        'acs_income': 0.05,             # Income trends
        'demographics': 0.05            # Structural
    }
    
    # Active weights (based on ACTIVE_SCENARIO)
    @property
    def ENSEMBLE_WEIGHTS(self) -> Dict[str, float]:
        """Get ensemble weights for active scenario."""
        if self.ACTIVE_SCENARIO == 'conservative':
            return self.ENSEMBLE_WEIGHTS_CONSERVATIVE
        elif self.ACTIVE_SCENARIO == 'moderate':
            return self.ENSEMBLE_WEIGHTS_MODERATE
        elif self.ACTIVE_SCENARIO == 'aggressive':
            return self.ENSEMBLE_WEIGHTS_AGGRESSIVE
        else:
            raise ValueError(f"Unknown scenario: {self.ACTIVE_SCENARIO}")
    
    # =========================================================================
    # CALIBRATION PARAMETERS (Scenario-Specific)
    # =========================================================================
    
    # Conservative scenario
    CONSERVATIVE_PARAMS = {
        'base_year': 2024,
        'base_resident': FY_2024_RESIDENT,
        'growth_rate': 0.015,           # 1.5% annual
        'years_forward': 2,             # Project 2 years
        'target_resident': 3082,        # Million
        'target_total': 3379,           # Million
        'income_scaling': 0.934,        # -6.6% adjustment
    }
    
    # Moderate scenario (RECOMMENDED)
    MODERATE_PARAMS = {
        'base_year': 2024.5,  # Conceptual blend
        'base_resident': (FY_2024_RESIDENT + FY_2025_RESIDENT_ESTIMATE) / 2,
        'growth_rate': 0.020,           # 2.0% annual
        'years_forward': 1.5,           # Average
        'target_resident': 3085,        # Million
        'target_total': 3383,           # Million (with non-residents)
        'income_scaling': 0.935,        # -6.5% adjustment
    }
    
    # Aggressive scenario
    AGGRESSIVE_PARAMS = {
        'base_year': 2025,
        'base_resident': FY_2025_RESIDENT_ESTIMATE,
        'growth_rate': 0.025,           # 2.5% annual
        'years_forward': 1,
        'target_resident': 3074,        # Million
        'target_total': 3371,           # Million
        'income_scaling': 0.932,        # -6.8% adjustment
    }
    
    @classmethod
    def get_scenario_params(cls, scenario: str = None) -> Dict[str, Any]:
        """
        Get parameters for specific scenario.
        
        Args:
            scenario: 'conservative', 'moderate', or 'aggressive'
                     If None, uses ACTIVE_SCENARIO
        
        Returns:
            Dictionary of scenario parameters
        """
        if scenario is None:
            scenario = cls.ACTIVE_SCENARIO
        
        scenarios = {
            'conservative': cls.CONSERVATIVE_PARAMS,
            'moderate': cls.MODERATE_PARAMS,
            'aggressive': cls.AGGRESSIVE_PARAMS
        }
        
        if scenario not in scenarios:
            raise ValueError(f"Unknown scenario: {scenario}")
        
        return scenarios[scenario]
    
    # =========================================================================
    # ACT 46 PARAMETERS
    # =========================================================================
    
    ACT46_IMPACT_RATE = -0.199  # -19.9% of resident revenue (official)
    ACT46_IMPACT_TOTAL = -597   # Million (official estimate)
    
    # Official post-Act 46 target
    FY_2026_POST_ACT46_OFFICIAL = 2691  # Million (total)
    
    # =========================================================================
    # GROWTH RATE COMPONENTS (for ensemble calculation)
    # =========================================================================
    
    GROWTH_RATES = {
        'fy_actual_2022_2024': (FY_2024_TOTAL / FY_2023_ADJUSTED - 1),  # -3.9%
        'fy_2025_estimate': (FY_2025_ESTIMATE / FY_2024_TOTAL - 1),     # +0.2%
        'dotax_2018_2021': 0.111,   # 11.1% CAGR (pre-peak)
        'bls_wage': 0.055,          # 5.5% wage growth
        'acs_income': 0.062,        # 6.2% income growth
        'demographics': 0.011       # 1.1% population growth
    }
    
    # =========================================================================
    # FILING STATUS TARGETS (from SOI)
    # =========================================================================
    
    SOI_FILING_STATUS_TARGETS = {
        'single': 0.510,    # 51.0%
        'joint': 0.360,     # 36.0%
        'hoh': 0.096,       # 9.6%
        'mfs': 0.034        # 3.4%
    }
    
    # =========================================================================
    # TAX CALCULATION PARAMETERS
    # =========================================================================
    
    # Current model performance (before calibration)
    CURRENT_RESIDENT_REVENUE = 3298  # Million (ensemble estimate)
    CURRENT_GROWTH_RATE = 0.074      # 7.4% CAGR (too high)
    
    # Validation tolerances
    REVENUE_TOLERANCE = 0.05  # ±5%
    GROWTH_RATE_TOLERANCE = 0.01  # ±1 percentage point
    ACT46_TOLERANCE = 0.10  # ±10%
    
    # =========================================================================
    # SYNTHETIC HIGH-INCOME PARAMETERS
    # =========================================================================
    
    SYNTHETIC_HIGH_INCOME_ENABLED = False  # Set to True to activate
    PARETO_ALPHA = 1.3  # Tail distribution parameter
    TAIL_MULTIPLIER = 0.45  # Adjustment factor
    
    SYNTHETIC_INCOME_TIERS = [
        5_000_000,   # $5M
        10_000_000,  # $10M
        25_000_000,  # $25M
        50_000_000   # $50M
    ]
    
    # =========================================================================
    # POPULATION SCALING
    # =========================================================================
    
    PUMS_SAMPLE_SIZE = 35000  # Approximate tax units in PUMS
    HAWAII_TOTAL_FILERS = 700000  # Approximate total filers
    SCALING_FACTOR = HAWAII_TOTAL_FILERS / PUMS_SAMPLE_SIZE  # ~20.1
    
    # =========================================================================
    # ITEMIZED DEDUCTION PARAMETERS
    # =========================================================================
    
    ITEMIZED_DEDUCTION_RATES = {
        'salt_cap': 10000,  # Federal SALT cap
        'mortgage_interest_rate': 0.70,  # 70% of payment is interest
        'charitable_base_rate': 0.025,   # 2.5% of income
        'charitable_senior_multiplier': 1.3,
        'charitable_high_income_multiplier': 1.5,
        'medical_base_rate': 0.05,       # 5% of income
        'medical_senior_multiplier': 2.0,
        'medical_agi_threshold': 0.075   # 7.5% AGI threshold
    }
    
    # =========================================================================
    # CREDIT PARAMETERS
    # =========================================================================
    
    HAWAII_CREDITS = {
        'food_excise_per_exemption': 110,  # $110 per exemption
        'renewable_energy_avg': 2000,      # $2,000 average
        'renewable_energy_rate': 0.02,     # 2% of filers
        'child_care_min': 200,
        'child_care_max': 800,
        'renter_credit_min': 50,
        'renter_credit_max': 150
    }
    
    # =========================================================================
    # OUTPUT SETTINGS
    # =========================================================================
    
    # File naming
    OUTPUT_DATE_FORMAT = "%Y%m%d_%H%M%S"
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # =========================================================================
    # VALIDATION SETTINGS
    # =========================================================================
    
    # Expected ranges for validation
    VALIDATION_RANGES = {
        'resident_revenue_2026': (3000, 3150),  # Million
        'total_revenue_2026': (3300, 3450),     # Million
        'growth_rate': (0.010, 0.030),          # 1-3% annual
        'act46_impact': (-650, -550),           # Million
        'post_act46_total': (2650, 2800)        # Million
    }
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    @classmethod
    def get_ensemble_weights(cls, scenario: str = None) -> Dict[str, float]:
        """Get ensemble weights for specified scenario."""
        if scenario is None:
            scenario = cls.ACTIVE_SCENARIO
        
        if scenario == 'conservative':
            return cls.ENSEMBLE_WEIGHTS_CONSERVATIVE
        elif scenario == 'moderate':
            return cls.ENSEMBLE_WEIGHTS_MODERATE
        elif scenario == 'aggressive':
            return cls.ENSEMBLE_WEIGHTS_AGGRESSIVE
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
    
    @classmethod
    def calculate_weighted_growth(cls, scenario: str = None) -> float:
        """Calculate weighted growth rate for scenario."""
        weights = cls.get_ensemble_weights(scenario)
        
        weighted_growth = sum(
            weights[component] * cls.GROWTH_RATES[component]
            for component in weights.keys()
        )
        
        return weighted_growth
    
    @classmethod
    def get_target_revenue(cls, scenario: str = None) -> Dict[str, float]:
        """Get target revenues for scenario."""
        params = cls.get_scenario_params(scenario)
        
        return {
            'resident': params['target_resident'],
            'total': params['target_total'],
            'nonresident': params['target_total'] - params['target_resident']
        }
    
    @classmethod
    def validate_scenario(cls, scenario: str) -> bool:
        """Validate that scenario is properly configured."""
        try:
            weights = cls.get_ensemble_weights(scenario)
            params = cls.get_scenario_params(scenario)
            
            # Check weights sum to 1.0
            weight_sum = sum(weights.values())
            if not (0.99 <= weight_sum <= 1.01):
                raise ValueError(f"Weights sum to {weight_sum}, not 1.0")
            
            # Check required parameters exist
            required = ['growth_rate', 'target_resident', 'income_scaling']
            for param in required:
                if param not in params:
                    raise ValueError(f"Missing parameter: {param}")
            
            return True
            
        except Exception as e:
            print(f"Scenario validation failed: {e}")
            return False
    
    @classmethod
    def print_summary(cls, scenario: str = None):
        """Print configuration summary."""
        if scenario is None:
            scenario = cls.ACTIVE_SCENARIO
        
        print(f"\n{'='*80}")
        print(f"HAWAII TAX MODEL CONFIGURATION - {scenario.upper()} SCENARIO")
        print(f"{'='*80}\n")
        
        params = cls.get_scenario_params(scenario)
        weights = cls.get_ensemble_weights(scenario)
        
        print("Scenario Parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        print("\nEnsemble Weights:")
        for component, weight in weights.items():
            print(f"  {component}: {weight:.1%}")
        
        print(f"\nWeighted Growth Rate: {cls.calculate_weighted_growth(scenario):.1%}")
        
        targets = cls.get_target_revenue(scenario)
        print("\nTarget Revenues (2026):")
        for key, value in targets.items():
            print(f"  {key}: ${value:,.0f}M")
        
        print(f"\n{'='*80}\n")


# Create default instance
config = ModelConfig()


if __name__ == "__main__":
    # Print configuration summary
    ModelConfig.print_summary()
    
    # Validate all scenarios
    print("\nValidating scenarios...")
    for scenario in ['conservative', 'moderate', 'aggressive']:
        valid = ModelConfig.validate_scenario(scenario)
        status = "✅ VALID" if valid else "❌ INVALID"
        print(f"  {scenario.capitalize()}: {status}")
