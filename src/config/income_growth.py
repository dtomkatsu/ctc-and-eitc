"""
Income Growth and Inflation Adjustments for 2026 Projections.

This module defines the growth factors used to project income from base years
(2022 for nonresidents, 2023 for residents) to the target year (2026).

Key Concepts:
- Nominal Growth: Total percentage increase in dollar amounts
- Inflation: General price level increases (CPI)
- Real Growth: Nominal growth adjusted for inflation (purchasing power)

Data Sources:
- Hawaii wage growth: Hawaii Department of Labor & Industrial Relations
- National wage growth: Bureau of Labor Statistics
- CPI: Bureau of Labor Statistics Consumer Price Index
"""

from dataclasses import dataclass
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class GrowthFactors:
    """
    Growth factors for income projection.
    
    Attributes:
        base_year: Starting year for the projection
        target_year: Ending year for the projection
        nominal_growth: Total nominal growth factor (e.g., 1.15 = 15% growth)
        inflation: Total inflation factor (e.g., 1.0884 = 8.84% inflation)
        real_growth: Real growth factor (nominal / inflation)
        description: Human-readable description of the adjustment
    """
    base_year: int
    target_year: int
    nominal_growth: float
    inflation: float
    real_growth: float
    description: str
    
    @property
    def years(self) -> int:
        """Number of years in the projection period."""
        return self.target_year - self.base_year
    
    @property
    def annual_nominal_rate(self) -> float:
        """Annualized nominal growth rate."""
        return self.nominal_growth ** (1 / self.years) - 1
    
    @property
    def annual_inflation_rate(self) -> float:
        """Annualized inflation rate."""
        return self.inflation ** (1 / self.years) - 1
    
    @property
    def annual_real_rate(self) -> float:
        """Annualized real growth rate."""
        return self.real_growth ** (1 / self.years) - 1
    
    def apply_to_income(self, income: float) -> float:
        """
        Apply real growth factor to income.
        
        Args:
            income: Base year income
            
        Returns:
            Projected income in target year (real terms)
        """
        return income * self.real_growth
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            'base_year': self.base_year,
            'target_year': self.target_year,
            'years': self.years,
            'nominal_growth': self.nominal_growth,
            'inflation': self.inflation,
            'real_growth': self.real_growth,
            'annual_nominal_rate': f"{self.annual_nominal_rate:.2%}",
            'annual_inflation_rate': f"{self.annual_inflation_rate:.2%}",
            'annual_real_rate': f"{self.annual_real_rate:.2%}",
            'description': self.description
        }
    
    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"{self.description}\n"
            f"  Period: {self.base_year} → {self.target_year} ({self.years} years)\n"
            f"  Nominal Growth: {(self.nominal_growth - 1):.2%} "
            f"({self.annual_nominal_rate:.2%}/year)\n"
            f"  Inflation: {(self.inflation - 1):.2%} "
            f"({self.annual_inflation_rate:.2%}/year)\n"
            f"  Real Growth: {(self.real_growth - 1):.2%} "
            f"({self.annual_real_rate:.2%}/year)"
        )


# ============================================================================
# RESIDENT INCOME GROWTH (2023 PUMS → 2026)
# ============================================================================

RESIDENT_GROWTH = GrowthFactors(
    base_year=2023,
    target_year=2026,
    nominal_growth=1.15,      # 15.0% total wage growth (Hawaii-specific)
    inflation=1.0884,         # 8.84% total CPI increase
    real_growth=1.056,        # 5.6% real growth (1.15 / 1.0884)
    description="Hawaii Resident Income Growth (2023 PUMS → 2026)"
)

# ============================================================================
# NONRESIDENT INCOME GROWTH (2022 DOTAX → 2026)
# ============================================================================

NONRESIDENT_GROWTH = GrowthFactors(
    base_year=2022,
    target_year=2026,
    nominal_growth=1.1248,    # 12.48% total wage growth (national average)
    inflation=1.0681,         # 6.81% total CPI increase
    real_growth=1.053,        # 5.3% real growth (1.1248 / 1.0681)
    description="Nonresident Income Growth (2022 DOTAX → 2026)"
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_growth_factors(project_root=None):
    """Build growth factors using latest CPI data, with hardcoded fallbacks.

    Returns (resident_growth, nonresident_growth) tuple of GrowthFactors.
    If project_root is provided and FRED/BLS CPI data is available,
    the inflation component is updated to reflect actual data.
    """
    if project_root is None:
        return RESIDENT_GROWTH, NONRESIDENT_GROWTH

    try:
        from src.projection.growth_rate_loader import load_all_rates
        from pathlib import Path

        rates = load_all_rates(Path(project_root))

        resident = GrowthFactors(
            base_year=RESIDENT_GROWTH.base_year,
            target_year=RESIDENT_GROWTH.target_year,
            nominal_growth=RESIDENT_GROWTH.nominal_growth,
            inflation=rates["inflation_resident"],
            real_growth=RESIDENT_GROWTH.nominal_growth / rates["inflation_resident"],
            description=f"Hawaii Resident Income Growth ({RESIDENT_GROWTH.base_year} PUMS → {RESIDENT_GROWTH.target_year}) [data-driven CPI]",
        )

        nonresident = GrowthFactors(
            base_year=NONRESIDENT_GROWTH.base_year,
            target_year=NONRESIDENT_GROWTH.target_year,
            nominal_growth=NONRESIDENT_GROWTH.nominal_growth,
            inflation=rates["inflation_nonresident"],
            real_growth=NONRESIDENT_GROWTH.nominal_growth / rates["inflation_nonresident"],
            description=f"Nonresident Income Growth ({NONRESIDENT_GROWTH.base_year} DOTAX → {NONRESIDENT_GROWTH.target_year}) [data-driven CPI]",
        )

        logger.info(f"Built data-driven growth factors: resident inflation={rates['inflation_resident']:.4f}, nonresident inflation={rates['inflation_nonresident']:.4f}")
        return resident, nonresident

    except Exception as e:
        logger.warning(f"Failed to build data-driven growth factors: {e}. Using hardcoded defaults.")
        return RESIDENT_GROWTH, NONRESIDENT_GROWTH


def get_growth_factors(is_resident: bool) -> GrowthFactors:
    """
    Get appropriate growth factors based on residency status.

    Args:
        is_resident: True for Hawaii residents, False for nonresidents

    Returns:
        GrowthFactors object with appropriate adjustments
    """
    return RESIDENT_GROWTH if is_resident else NONRESIDENT_GROWTH


def apply_income_growth(income: float, is_resident: bool) -> float:
    """
    Apply appropriate income growth to project to 2026.
    
    Args:
        income: Base year income (2023 for residents, 2022 for nonresidents)
        is_resident: True for Hawaii residents, False for nonresidents
        
    Returns:
        Projected 2026 income (real terms)
    """
    factors = get_growth_factors(is_resident)
    return factors.apply_to_income(income)


def log_growth_factors():
    """Log all growth factors for documentation."""
    logger.info("="*80)
    logger.info("INCOME GROWTH FACTORS FOR 2026 PROJECTIONS")
    logger.info("="*80)
    logger.info("\n" + str(RESIDENT_GROWTH))
    logger.info("\n" + str(NONRESIDENT_GROWTH))
    logger.info("="*80)


# ============================================================================
# VALIDATION AND DIAGNOSTICS
# ============================================================================

def validate_growth_factors():
    """Validate that growth factors are reasonable."""
    issues = []
    
    # Check resident growth
    if RESIDENT_GROWTH.annual_nominal_rate > 0.10:  # >10% annual growth
        issues.append(
            f"Resident annual nominal growth ({RESIDENT_GROWTH.annual_nominal_rate:.2%}) "
            "seems unusually high"
        )
    
    if RESIDENT_GROWTH.annual_inflation_rate > 0.05:  # >5% annual inflation
        issues.append(
            f"Resident annual inflation ({RESIDENT_GROWTH.annual_inflation_rate:.2%}) "
            "seems unusually high"
        )
    
    # Check nonresident growth
    if NONRESIDENT_GROWTH.annual_nominal_rate > 0.10:
        issues.append(
            f"Nonresident annual nominal growth ({NONRESIDENT_GROWTH.annual_nominal_rate:.2%}) "
            "seems unusually high"
        )
    
    if NONRESIDENT_GROWTH.annual_inflation_rate > 0.05:
        issues.append(
            f"Nonresident annual inflation ({NONRESIDENT_GROWTH.annual_inflation_rate:.2%}) "
            "seems unusually high"
        )
    
    # Check consistency
    real_growth_diff = abs(RESIDENT_GROWTH.real_growth - NONRESIDENT_GROWTH.real_growth)
    if real_growth_diff > 0.02:  # More than 2 percentage points difference
        logger.warning(
            f"Real growth rates differ significantly: "
            f"Residents {(RESIDENT_GROWTH.real_growth - 1):.2%} vs "
            f"Nonresidents {(NONRESIDENT_GROWTH.real_growth - 1):.2%}"
        )
    
    if issues:
        for issue in issues:
            logger.warning(f"⚠️  {issue}")
        return False
    else:
        logger.info("✅ All growth factors validated successfully")
        return True


if __name__ == '__main__':
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    log_growth_factors()
    print("\n")
    validate_growth_factors()
    
    # Example calculations
    print("\n" + "="*80)
    print("EXAMPLE CALCULATIONS")
    print("="*80)
    
    resident_income = 75000
    nonresident_income = 100000
    
    print(f"\nResident with ${resident_income:,} in 2023:")
    print(f"  → ${apply_income_growth(resident_income, True):,.0f} in 2026 (real terms)")
    
    print(f"\nNonresident with ${nonresident_income:,} in 2022:")
    print(f"  → ${apply_income_growth(nonresident_income, False):,.0f} in 2026 (real terms)")
