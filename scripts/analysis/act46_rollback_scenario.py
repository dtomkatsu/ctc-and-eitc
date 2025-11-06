#!/usr/bin/env python3
"""Act 46 partial rollback analysis using actual tax units."""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ModelConfig
from src.tax.brackets.hawaii_tax import HawaiiTaxCalculator


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class RevenueImpact:
    """Container for revenue impact results."""

    total_increase: float
    achievement_rate: float
    by_status: Dict[str, float]
    baseline_revenue: float
    adjusted_revenue: float


class Act46RollbackAnalyzer:
    """Analyze partial rollback of Act 46 using actual tax unit data."""

    def __init__(
        self,
        scenario: str = "moderate",
        tax_units_path: Optional[Path] = None,
        projection_year: int = 2026,
        top_bracket_count: int = 5,
    ) -> None:
        self.config = ModelConfig()
        self.scenario = scenario
        self.projection_year = projection_year
        self.top_bracket_count = top_bracket_count

        # Act 46 targets
        self.act46_total_loss = 597.0  # Million (official)
        self.target_recovery = 240.0   # Million (40% recovery)
        self.recovery_percentage = self.target_recovery / self.act46_total_loss

        # Load tax brackets and deductions
        self.data_dir = self.config.DATA_DIR
        self.brackets_df = pd.read_csv(self.data_dir / "raw" / "hawaii_tax_brackets_master_all.csv")
        self.deductions_df = pd.read_csv(self.data_dir / "raw" / "hawaii_standard_deductions_by_year.csv")
        self.brackets_2024 = self._filter_brackets(2024)
        self.brackets_2026 = self._filter_brackets(self.projection_year)

        # Load tax units
        default_tax_units = self.config.PROJECTION_OUTPUT_DIR / "tax_units_2026_baseline.parquet"
        self.tax_units_path = Path(tax_units_path) if tax_units_path else default_tax_units
        if not self.tax_units_path.exists():
            raise FileNotFoundError(f"Tax unit file not found: {self.tax_units_path}")

        self.tax_units_raw = pd.read_parquet(self.tax_units_path)
        self.tax_units = self._prepare_tax_units(self.tax_units_raw.copy())

        # Baseline calculator (post-Act46 rates)
        self.calculator_baseline = HawaiiTaxCalculator(self.brackets_2026, self.deductions_df)

        logger.info("Act 46 Rollback Analyzer initialized")
        logger.info("  Scenario: %s", scenario)
        logger.info("  Tax units loaded: %s", len(self.tax_units))
        logger.info("  Target recovery: $%.1fM (%.1f%% of Act 46 loss)", self.target_recovery, self.recovery_percentage * 100)

    # ------------------------------------------------------------------
    # Data preparation helpers
    # ------------------------------------------------------------------
    def _filter_brackets(self, year: int) -> pd.DataFrame:
        df = self.brackets_df[self.brackets_df["year"] == year].copy()
        if df.empty:
            raise ValueError(f"No brackets found for year {year}")
        return df

    def _prepare_tax_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """Project tax unit incomes to projection year using config growth."""

        required_cols = {"income", "filing_status", "weight"}
        missing = required_cols - set(df.columns)
        if missing:
            raise KeyError(f"Missing required columns in tax units: {missing}")

        # Use already-projected income from ensemble file
        df["income_projected"] = df["income"]
        df["growth_factor_config"] = df.get("income_growth_factor", 1.0)
        
        # Normalize filing statuses to align with calculator expectations
        status_map = {
            "married_filing_separately": "married_filing_separate",
            "married filing separately": "married_filing_separate",
            "married filing jointly": "married_filing_jointly",
            "head_of_household": "head_of_household",
            "head of household": "head_of_household",
            "qualifying_widow(er)": "qualifying_widow",
            "widow(er)": "qualifying_widow",
        }
        df["filing_status_clean"] = (
            df["filing_status"].str.lower().map(status_map).fillna(df["filing_status"].str.lower())
        )
        
        logger.info(
            "Using ensemble-projected incomes (%.0f weighted filers, mean growth factor %.3f)",
            df["weight"].sum(),
            df["growth_factor_config"].mean(),
        )

        return df

    # ------------------------------------------------------------------
    # Bracket adjustment helpers
    # ------------------------------------------------------------------
    def identify_top_brackets(self, filing_status: str) -> pd.DataFrame:
        status_brackets = self.brackets_2026[self.brackets_2026["filing_status"] == filing_status].copy()
        if status_brackets.empty:
            raise ValueError(f"No brackets found for filing status {filing_status}")
        status_brackets = status_brackets.sort_values("income_min", ascending=False)
        return status_brackets.head(self.top_bracket_count).sort_values("income_min")

    def calculate_rate_adjustments(self, adjustment_strategy: str = "progressive") -> Dict[int, float]:
        if adjustment_strategy == "uniform":
            return {i: 0.5 for i in range(self.top_bracket_count)}
        if adjustment_strategy == "progressive":
            base_step = 0.25
            return {i: base_step * (i + 1) for i in range(self.top_bracket_count)}
        if adjustment_strategy == "aggressive":
            increments = [0.25, 0.5]
            while len(increments) < self.top_bracket_count:
                if len(increments) == 2:
                    increments.append(1.0)
                else:
                    increments.append(increments[-1] + 0.5)
            return {i: increments[i] for i in range(self.top_bracket_count)}
        if adjustment_strategy == "targeted":
            # Fixed increases: Top bracket +1.0pp, 2nd bracket +0.5pp, next 3 brackets +0.25pp
            # No scaling - these are the actual increases we want
            return {
                0: 0.25,  # 3rd highest bracket: +0.25pp
                1: 0.25,  # 4th highest bracket: +0.25pp  
                2: 0.25,  # 5th highest bracket: +0.25pp
                3: 0.5,   # 2nd highest bracket: +0.5pp
                4: 1.0    # Highest bracket: +1.0pp
            }
        raise ValueError(f"Unknown adjustment strategy: {adjustment_strategy}")

    def create_adjusted_brackets(self, adjustment_strategy: str, scaling_factor: float) -> pd.DataFrame:
        adjusted = self.brackets_2026.copy()
        adjustments = self.calculate_rate_adjustments(adjustment_strategy)

        for status in adjusted["filing_status"].unique():
            top_brackets = self.identify_top_brackets(status)
            for i, (_, row) in enumerate(top_brackets.iterrows()):
                idx = row.name
                if adjustment_strategy == "targeted":
                    # For targeted strategy, use fixed increases (no scaling)
                    increase = adjustments[i]
                else:
                    # For other strategies, apply scaling
                    increase = adjustments[i] * scaling_factor
                adjusted.loc[idx, "rate"] = adjusted.loc[idx, "rate"] + increase
                adjusted.loc[idx, "rate_increase"] = increase
                adjusted.loc[idx, "adjusted"] = True

        return adjusted

    # ------------------------------------------------------------------
    # Revenue calculation
    # ------------------------------------------------------------------
    def _calculate_revenue(self, calculator: HawaiiTaxCalculator, tax_units: pd.DataFrame, use_corrected_baseline: bool = False) -> float:
        # Use corrected Act 46 calculation for baseline only
        if use_corrected_baseline:
            from pathlib import Path
            act46_path = Path("data/processed/projections/tax_units_2026_act46_corrected.parquet")
            if act46_path.exists():
                df_act46 = pd.read_parquet(act46_path)
                revenue = (df_act46['tax_liability_act46'] * df_act46['weight']).sum() / 1e6
                return revenue
        
        # Standard calculation for adjusted scenarios
        liabilities = []
        filing_status_col = "filing_status_clean"
        weights = tax_units["weight"].values
        
        # Use taxable income if available (accounts for itemized deductions)
        # Otherwise fall back to income_projected
        if "taxable_income" in tax_units.columns:
            incomes = tax_units["taxable_income"].values
            use_taxable = True
        else:
            incomes = tax_units["income_projected"].values
            use_taxable = False

        for income, status in zip(incomes, tax_units[filing_status_col]):
            tax_info = calculator.calculate_tax(income, self.projection_year, status, use_taxable_income=use_taxable)
            liabilities.append(tax_info["tax_liability"])

        liabilities = np.array(liabilities)
        revenue = float(np.sum(liabilities * weights)) / 1e6  # Convert to millions
        return revenue

    def estimate_revenue(self, adjusted_brackets: pd.DataFrame) -> RevenueImpact:
        calculator_adjusted = HawaiiTaxCalculator(adjusted_brackets, self.deductions_df)

        baseline_revenue = self._calculate_revenue(self.calculator_baseline, self.tax_units, use_corrected_baseline=True)
        adjusted_revenue = self._calculate_revenue(calculator_adjusted, self.tax_units)

        total_increase = adjusted_revenue - baseline_revenue
        achievement_rate = total_increase / self.target_recovery if self.target_recovery else 0.0

        # Revenue by filing status
        by_status = {}
        for status in self.brackets_2026["filing_status"].unique():
            subset = self.tax_units[self.tax_units["filing_status"] == status]
            if subset.empty:
                continue
            base_rev = self._calculate_revenue(self.calculator_baseline, subset)
            adj_rev = self._calculate_revenue(calculator_adjusted, subset)
            by_status[status] = adj_rev - base_rev

        return RevenueImpact(
            total_increase=total_increase,
            achievement_rate=achievement_rate,
            by_status=by_status,
            baseline_revenue=baseline_revenue,
            adjusted_revenue=adjusted_revenue,
        )

    # ------------------------------------------------------------------
    # Scaling search and reporting
    # ------------------------------------------------------------------
    def find_optimal_scaling(self, adjustment_strategy: str, tolerance: float = 0.02) -> Tuple[float, RevenueImpact]:
        logger.info("Finding scaling factor for strategy '%s'", adjustment_strategy)
        low, high = 0.25, 6.0
        best_scaling, best_impact = 1.0, None

        # Expand search range if high bound insufficient
        max_expansions = 6
        for _ in range(max_expansions):
            adjusted_high = self.create_adjusted_brackets(adjustment_strategy, high)
            impact_high = self.estimate_revenue(adjusted_high)
            if impact_high.total_increase >= self.target_recovery:
                best_scaling, best_impact = high, impact_high
                break
            low = high
            best_scaling, best_impact = high, impact_high
            high *= 2
        else:
            logger.warning(
                "Unable to reach $%.1fM target even at scaling %.2f (increase=$%.1fM)",
                self.target_recovery,
                high,
                best_impact.total_increase if best_impact else float("nan"),
            )
            return best_scaling, best_impact

        for _ in range(30):
            mid = (low + high) / 2
            adjusted = self.create_adjusted_brackets(adjustment_strategy, mid)
            impact = self.estimate_revenue(adjusted)

            logger.debug("  scaling=%.3f -> increase=$%.1fM (%.1f%% of target)", mid, impact.total_increase, impact.achievement_rate * 100)

            if abs(impact.total_increase - self.target_recovery) / self.target_recovery <= tolerance:
                logger.info("  ✅ scaling %.3f meets target (increase $%.1fM)", mid, impact.total_increase)
                return mid, impact

            if impact.total_increase < self.target_recovery:
                low = mid
            else:
                high = mid

            best_scaling, best_impact = mid, impact

        logger.warning("Did not converge within tolerance; returning best attempt")
        return best_scaling, best_impact

    def generate_report(self, adjustment_strategy: str, output_file: Optional[Path] = None) -> pd.DataFrame:
        if adjustment_strategy == "targeted":
            # For targeted strategy, use fixed scaling of 1.0 (no optimization needed)
            scaling = 1.0
            adjusted = self.create_adjusted_brackets(adjustment_strategy, scaling)
            impact = self.estimate_revenue(adjusted)
        else:
            # For other strategies, find optimal scaling
            scaling, impact = self.find_optimal_scaling(adjustment_strategy)
            adjusted = self.create_adjusted_brackets(adjustment_strategy, scaling)

        logger.info("Revenue Impact Summary")
        logger.info("  Baseline revenue : $%.1fM", impact.baseline_revenue)
        logger.info("  Adjusted revenue : $%.1fM", impact.adjusted_revenue)
        logger.info("  Increase          : $%.1fM (%.1f%% of $240M target)", impact.total_increase, impact.achievement_rate * 100)
        logger.info("  Scaling factor    : %.3f", scaling)

        for status, value in impact.by_status.items():
            logger.info("    %s: $%.1fM", status, value)

        report_rows = []
        for status in adjusted["filing_status"].unique():
            status_brackets = adjusted[(adjusted["filing_status"] == status) & (adjusted.get("adjusted", False))]
            for _, row in status_brackets.iterrows():
                orig = self.brackets_2026[(self.brackets_2026["filing_status"] == status) & (self.brackets_2026["income_min"] == row["income_min"])]
                rate_2024 = self._get_rate_for_income(self.brackets_2024, status, row["income_min"])
                report_rows.append(
                    {
                        "filing_status": status,
                        "income_min": row["income_min"],
                        "income_max": row["income_max"],
                        "rate_2024_pre_act46": rate_2024,
                        "rate_2026_post_act46": float(orig["rate"].iloc[0]) if not orig.empty else np.nan,
                        "rate_2026_adjusted": row["rate"],
                        "rate_increase_from_2026": row.get("rate_increase", np.nan),
                        "still_below_2024": row["rate"] < rate_2024 if rate_2024 is not None else np.nan,
                        "scaling_factor": scaling,
                        "total_revenue_increase": impact.total_increase,
                    }
                )

        report_df = pd.DataFrame(report_rows)
        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            report_df.to_csv(output_file, index=False)
            logger.info("Report saved to %s", output_file)

        return report_df

    def _get_rate_for_income(self, brackets: pd.DataFrame, status: str, income_min: float) -> Optional[float]:
        status_brackets = brackets[brackets["filing_status"] == status]
        if status_brackets.empty:
            return None
        # Find bracket with matching income_min or closest lower bound
        candidates = status_brackets[status_brackets["income_min"] <= income_min]
        if candidates.empty:
            return float(status_brackets["rate"].min())
        return float(candidates.sort_values("income_min", ascending=False).iloc[0]["rate"])


def main() -> None:
    analyzer = Act46RollbackAnalyzer()
    strategies = ["targeted"]  # Run only the targeted strategy with fixed increases

    for strategy in strategies:
        logger.info("\n%s\nStrategy: %s\n%s", "=" * 80, strategy.upper(), "=" * 80)
        output_path = Path("data/processed/projections") / f"act46_rollback_{strategy}_actuals.csv"
        analyzer.generate_report(strategy, output_path)

    logger.info("\n%s\nAnalysis complete. Reports available in data/processed/projections\n%s", "=" * 80, "=" * 80)


if __name__ == "__main__":
    main()
