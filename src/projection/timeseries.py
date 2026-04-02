"""
Time-Series Forecasting for Income Growth Projections

Uses Holt's Damped Trend Exponential Smoothing on ACS time-series data
(2015-2024, 9 data points with 2020 missing) to produce statistically
grounded growth rate forecasts with confidence intervals.

Also provides BLS OES log-linear trend analysis for wage growth validation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ETSForecaster:
    """
    Holt's Damped Trend Exponential Smoothing for ACS income series.

    Handles missing years (e.g., 2020) by interpolation before fitting.
    Produces point forecasts and approximate prediction intervals.
    """

    def __init__(self, damped_trend: bool = True):
        self.damped_trend = damped_trend
        self._model = None
        self._fit_result = None
        self._fitted_years = None

    def fit(self, values: np.ndarray, years: np.ndarray) -> 'ETSForecaster':
        """
        Fit ETS model to the series.

        Args:
            values: Observed values (e.g., median household income)
            years: Corresponding years
        """
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        # Create a complete annual series, interpolating any missing years
        all_years = np.arange(years.min(), years.max() + 1)
        complete_values = np.interp(all_years, years, values)

        # Create pandas series with annual frequency
        idx = pd.date_range(start=f'{int(all_years[0])}-01-01', periods=len(all_years), freq='YS')
        series = pd.Series(complete_values, index=idx)

        self._model = ExponentialSmoothing(
            series,
            trend='add',
            damped_trend=self.damped_trend,
            seasonal=None,
        )
        self._fit_result = self._model.fit(optimized=True)
        self._fitted_years = all_years
        self._residual_std = np.std(self._fit_result.resid.dropna())

        return self

    def forecast(self, horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forecast future values with approximate 80% prediction intervals.

        Returns:
            (point_forecast, lower_80_ci, upper_80_ci) each of length `horizon`
        """
        if self._fit_result is None:
            raise RuntimeError("Must call fit() before forecast()")

        fcast = self._fit_result.forecast(horizon)
        point = fcast.values

        # Approximate prediction intervals (widen with sqrt of horizon step)
        z_80 = 1.28  # 80% CI z-score
        steps = np.arange(1, horizon + 1)
        interval_width = z_80 * self._residual_std * np.sqrt(steps)

        lower = point - interval_width
        upper = point + interval_width

        return point, lower, upper

    def get_annual_growth_rate(self, horizon: int) -> Tuple[float, float, float]:
        """
        Extract implied compound annual growth rate from the forecast.

        Returns:
            (central_cagr, lower_cagr, upper_cagr) as decimals
        """
        point, lower, upper = self.forecast(horizon)
        last_observed = self._fit_result.fittedvalues.iloc[-1]

        if last_observed <= 0:
            return 0.0, 0.0, 0.0

        central_cagr = (point[-1] / last_observed) ** (1 / horizon) - 1
        lower_cagr = (lower[-1] / last_observed) ** (1 / horizon) - 1
        upper_cagr = (upper[-1] / last_observed) ** (1 / horizon) - 1

        return central_cagr, lower_cagr, upper_cagr


class BLSTrendForecaster:
    """
    Log-linear trend analysis on BLS OES wage data (2020-2024).

    Fits log(wage) ~ year across occupations within an income bracket,
    weighted by employment. Provides bracket-specific growth rates
    with standard errors.
    """

    def __init__(self, project_root: Optional[Path] = None):
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        self.project_root = project_root
        self._bracket_rates = None

    def fit(self) -> 'BLSTrendForecaster':
        """Fit log-linear trends from BLS OES timeseries."""
        bls_path = self.project_root / "data/processed/bls_oes_timeseries.csv"
        if not bls_path.exists():
            logger.warning(f"BLS OES file not found: {bls_path}")
            self._bracket_rates = {}
            return self

        df = pd.read_csv(bls_path)

        # Use mean_wage_annual_cagr already computed in the data
        # Group by wage percentile ranges to get bracket-level rates
        latest = df[df['year'] == df['year'].max()].copy()
        latest = latest.dropna(subset=['mean_wage_annual', 'mean_wage_annual_cagr', 'employment'])

        if latest.empty:
            self._bracket_rates = {}
            return self

        # Classify occupations into income brackets by mean wage
        brackets = [
            (0, 25_000), (25_000, 50_000), (50_000, 75_000),
            (75_000, 100_000), (100_000, 200_000), (200_000, float('inf'))
        ]

        self._bracket_rates = {}
        for lo, hi in brackets:
            mask = (latest['mean_wage_annual'] >= lo) & (latest['mean_wage_annual'] < hi)
            subset = latest[mask]
            if len(subset) < 3:
                continue

            # Employment-weighted CAGR
            weights = subset['employment'].values
            cagrs = subset['mean_wage_annual_cagr'].values
            valid = np.isfinite(cagrs) & np.isfinite(weights) & (weights > 0)
            if valid.sum() < 3:
                continue

            weighted_cagr = np.average(cagrs[valid], weights=weights[valid])
            # Standard error via weighted std
            weighted_var = np.average((cagrs[valid] - weighted_cagr) ** 2, weights=weights[valid])
            se = np.sqrt(weighted_var / valid.sum())

            self._bracket_rates[(lo, hi)] = {
                'cagr': weighted_cagr,
                'se': se,
                'n_occupations': int(valid.sum()),
            }

            label = f"${lo/1000:.0f}k–${hi/1000:.0f}k" if hi != float('inf') else f"${lo/1000:.0f}k+"
            logger.info(f"  BLS bracket {label}: CAGR={weighted_cagr:.1%} ± {se:.1%} "
                        f"({valid.sum()} occupations)")

        return self

    def get_bracket_rate(self, agi: float) -> Optional[Dict[str, float]]:
        """Get growth rate info for a given AGI bracket."""
        if not self._bracket_rates:
            return None
        for (lo, hi), info in self._bracket_rates.items():
            if lo <= agi < hi:
                return info
        return None

    def get_all_bracket_rates(self) -> Dict[tuple, float]:
        """Return {(lo, hi): cagr} for use in SourceSpecificGrowthProjector."""
        if not self._bracket_rates:
            return {}
        return {bracket: info['cagr'] for bracket, info in self._bracket_rates.items()}


class ACSIncomeForecaster:
    """
    Orchestrates ETS forecasting on ACS income time series.

    Fits models on:
    - B19013: Median household income (aggregate trend)
    - B19019: Income by household type (filing-status-specific)

    Produces growth rate forecasts with confidence intervals.
    """

    def __init__(self, project_root: Optional[Path] = None):
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        self.project_root = project_root
        self._forecasters: Dict[str, ETSForecaster] = {}
        self._results: Dict[str, Dict] = {}

    def fit_all(self) -> 'ACSIncomeForecaster':
        """Fit ETS models on all relevant ACS series."""
        acs_dir = self.project_root / "data/processed/acs_timeseries/wide"

        # Median household income (aggregate)
        b19013 = pd.read_csv(acs_dir / "B19013_wide.csv")
        self._fit_series('median_hh_income', b19013, 'median_household_income')

        # Income by household type
        b19019_path = acs_dir / "B19019_wide.csv"
        if b19019_path.exists():
            b19019 = pd.read_csv(b19019_path)
            for col in b19019.columns:
                if col.startswith('median_income_'):
                    self._fit_series(col, b19019, col)

        return self

    def _fit_series(self, name: str, df: pd.DataFrame, value_col: str):
        """Fit a single ETS model."""
        valid = df.dropna(subset=[value_col])
        if len(valid) < 5:
            logger.warning(f"Skipping {name}: only {len(valid)} data points")
            return

        years = valid['year'].values
        values = valid[value_col].values.astype(float)

        forecaster = ETSForecaster(damped_trend=True)
        forecaster.fit(values, years)
        self._forecasters[name] = forecaster

        # Get 3-year forecast (2025-2027 from 2024 base)
        central, lower, upper = forecaster.get_annual_growth_rate(3)
        self._results[name] = {
            'central_cagr': central,
            'lower_cagr': lower,
            'upper_cagr': upper,
        }

        logger.info(f"  {name}: ETS CAGR = {central:.1%} "
                     f"(80% CI: {lower:.1%} – {upper:.1%})")

    def get_results(self) -> Dict[str, Dict]:
        """Return all fitted forecasts."""
        return dict(self._results)

    def get_aggregate_growth(self) -> Tuple[float, float, float]:
        """Get aggregate median household income growth rate (central, lower, upper)."""
        if 'median_hh_income' not in self._results:
            return 0.04, 0.02, 0.06  # fallback defaults
        r = self._results['median_hh_income']
        return r['central_cagr'], r['lower_cagr'], r['upper_cagr']


class DOTAXCollectionsForecaster:
    """
    ETS forecasting on DOTAX annual individual income tax collections (2016-2024).

    Collections vs. liability distinction:
        - Collections in calendar year X reflect primarily TY(X-1) filing (large April spike)
          plus withholding for TY(X).
        - The lag-adjustment ratio converts CY collections to TY liability.
        - Ratio calibrated to TY2022 SOI liability ($3,029M) vs. CY2023 collections
          ($3,385M, which mainly reflect TY2022 April filings): ratio ≈ 0.895.

    Usage: fit on CY2016-2024, forecast CY2025-2028, apply ratio to estimate TY2027 liability.
    """

    # TY2022 SOI liability / CY2023 collections (CY(X+1) ≈ TY(X) returns)
    COLLECTIONS_TO_LIABILITY_RATIO = 3_029 / 3_385  # ≈ 0.895

    def __init__(self, project_root: Optional[Path] = None):
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        self.project_root = project_root
        self._annual_collections: Optional[pd.Series] = None
        self._forecaster: Optional[ETSForecaster] = None

    def load_and_aggregate(self) -> 'DOTAXCollectionsForecaster':
        """Load DOTAX collections CSV and compute annual individual income tax totals."""
        csv_path = self.project_root / "data/raw/hawaii_tax_collections_2016_2025.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"DOTAX collections file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        df['year'] = df['Year Month'].str.strip().str.split().str[-1].astype(int)
        df['tax_type'] = df['Tax Type'].str.strip()

        inc = df[df['tax_type'] == 'Individual Income'].copy()

        # 2025 is partial (Jan-Sep only) — exclude it to avoid downward bias
        annual = inc[inc['year'] < 2025].groupby('year')['Amount'].sum() / 1e6

        # Treat 2020-2022 as COVID-distorted before fitting ETS.
        # CY2020: -3.2% (pandemic disruption). CY2021: +41% spike (federal stimulus,
        # deferred TY2020 payments, accelerated capital gains). CY2022: elevated
        # post-pandemic recovery. ETS interprets this volatile period as a trend
        # inflection, causing systematic overprediction (MAPE ~9.4% on 2023-2024).
        # Fix: linearly interpolate 2020-2022 between 2019 and 2023, removing the
        # COVID disruption while preserving the structural growth trend.
        if 2019 in annual.index and 2023 in annual.index:
            y2019 = annual[2019]
            y2023 = annual[2023]
            covid_years = [yr for yr in [2020, 2021, 2022] if yr in annual.index]
            for yr in covid_years:
                frac = (yr - 2019) / (2023 - 2019)
                smoothed = y2019 + frac * (y2023 - y2019)
                logger.info(f"  COVID outlier treatment {yr}: "
                            f"${annual[yr]:,.1f}M → ${smoothed:,.1f}M")
                annual[yr] = smoothed

        self._annual_collections = annual

        logger.info("  DOTAX Individual Income Tax Collections ($M):")
        for yr, amt in annual.items():
            logger.info(f"    {yr}: ${amt:,.1f}M")

        return self

    def fit(self) -> 'DOTAXCollectionsForecaster':
        """Fit ETS on annual collections 2016-2024."""
        if self._annual_collections is None:
            self.load_and_aggregate()

        years = self._annual_collections.index.values.astype(float)
        values = self._annual_collections.values.astype(float)

        self._forecaster = ETSForecaster(damped_trend=True)
        self._forecaster.fit(values, years)
        return self

    def forecast_ty_liability(
        self,
        target_ty: int = 2027,
    ) -> Tuple[float, float, float]:
        """
        Forecast TY liability for a given target tax year.

        Since CY(X+1) collections ≈ TY(X) liability, we forecast CY(target_ty + 1)
        and multiply by the collections-to-liability ratio.

        Returns:
            (central_estimate_M, lower_80_M, upper_80_M)
        """
        if self._forecaster is None:
            self.fit()

        last_year = int(self._annual_collections.index.max())
        horizon = (target_ty + 1) - last_year  # e.g., 2027+1=2028; 2028-2024=4 steps

        point, lower, upper = self._forecaster.forecast(horizon)

        # The last forecast step corresponds to CY(target_ty + 1) collections
        cy_central = point[-1]
        cy_lower = lower[-1]
        cy_upper = upper[-1]

        r = self.COLLECTIONS_TO_LIABILITY_RATIO
        ty_central = cy_central * r
        ty_lower = cy_lower * r
        ty_upper = cy_upper * r

        # Also report the CAGR from last observed year to target CY
        last_obs = self._annual_collections.iloc[-1]
        cagr = (cy_central / last_obs) ** (1 / horizon) - 1

        logger.info(f"\n  DOTAX Collections ETS forecast:")
        logger.info(f"    Last observed (CY{last_year}): ${last_obs:,.1f}M")
        logger.info(f"    Projected CY{target_ty + 1}: ${cy_central:,.1f}M "
                     f"(80% CI: ${cy_lower:,.1f}M – ${cy_upper:,.1f}M)")
        logger.info(f"    Implied CAGR ({last_year}→CY{target_ty+1}): {cagr:.1%}")
        logger.info(f"    Collections-to-liability ratio: {r:.3f}")
        logger.info(f"    → TY{target_ty} liability estimate: ${ty_central:,.1f}M "
                     f"(80% CI: ${ty_lower:,.1f}M – ${ty_upper:,.1f}M)")

        return ty_central, ty_lower, ty_upper

    def backtest(self, holdout_years: int = 2) -> Dict[str, float]:
        """Backtest ETS on held-out collections years."""
        from .backtester import ForecastBacktester
        if self._annual_collections is None:
            self.load_and_aggregate()

        values = self._annual_collections.values.astype(float)
        years = self._annual_collections.index.values.astype(float)

        backtester = ForecastBacktester(holdout_years=holdout_years)
        return backtester.evaluate(values, years, series_name="DOTAX_collections")
