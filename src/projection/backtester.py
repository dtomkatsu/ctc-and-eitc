"""
Forecast Backtester

Hold-out validation for time-series forecasts. Fits on earlier years,
forecasts the held-out years, and computes accuracy metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict
import logging

from .timeseries import ETSForecaster

logger = logging.getLogger(__name__)


class ForecastBacktester:
    """
    Hold-out validation: fit on 2015-2022, forecast 2023-2024, compare to actuals.
    """

    def __init__(self, holdout_years: int = 2):
        self.holdout_years = holdout_years

    def evaluate(
        self,
        values: np.ndarray,
        years: np.ndarray,
        series_name: str = "series",
    ) -> Dict[str, float]:
        """
        Run hold-out evaluation on a time series.

        Args:
            values: Full observed series
            years: Corresponding years
            series_name: Label for logging

        Returns:
            Dict with MAPE, RMSE, and prediction interval coverage
        """
        if len(values) <= self.holdout_years + 3:
            logger.warning(f"Not enough data for backtesting {series_name} "
                           f"({len(values)} points, need >{self.holdout_years + 3})")
            return {'mape': float('nan'), 'rmse': float('nan'), 'coverage_80': float('nan')}

        # Split
        train_values = values[:-self.holdout_years]
        train_years = years[:-self.holdout_years]
        test_values = values[-self.holdout_years:]

        # Fit and forecast
        forecaster = ETSForecaster(damped_trend=True)
        forecaster.fit(train_values, train_years)
        point, lower, upper = forecaster.forecast(self.holdout_years)

        # Metrics
        errors = test_values - point
        abs_pct_errors = np.abs(errors / test_values) * 100
        mape = float(np.mean(abs_pct_errors))
        rmse = float(np.sqrt(np.mean(errors ** 2)))

        # Coverage: how many actuals fall within the 80% CI
        within_ci = (test_values >= lower) & (test_values <= upper)
        coverage = float(np.mean(within_ci))

        logger.info(f"  Backtest {series_name}:")
        logger.info(f"    Train: {int(train_years[0])}-{int(train_years[-1])}, "
                     f"Test: {int(years[-self.holdout_years])}-{int(years[-1])}")
        for i in range(self.holdout_years):
            yr = int(years[-(self.holdout_years - i)])
            logger.info(f"    {yr}: actual=${test_values[i]:,.0f}  "
                         f"forecast=${point[i]:,.0f}  "
                         f"error={abs_pct_errors[i]:.1f}%  "
                         f"CI=[${lower[i]:,.0f}, ${upper[i]:,.0f}]  "
                         f"{'✅' if within_ci[i] else '❌'}")
        logger.info(f"    MAPE: {mape:.1f}%  RMSE: ${rmse:,.0f}  "
                     f"80% CI coverage: {coverage:.0%}")

        return {
            'mape': mape,
            'rmse': rmse,
            'coverage_80': coverage,
            'point_forecasts': point.tolist(),
            'actuals': test_values.tolist(),
        }
