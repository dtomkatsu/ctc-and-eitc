#!/usr/bin/env python3
"""
Fetch economic time-series data from FRED (Federal Reserve Economic Data) API.

Pulls Hawaii-specific series used to replace hardcoded growth assumptions in the
tax projection model.

Series fetched:
    CUUSA426SA0          — CPI-U All Items, Urban Hawaii (monthly)
    HISTHPI              — All-Transactions House Price Index, Hawaii (quarterly)
    HIURN                — Unemployment Rate, Hawaii (monthly)
    QTAXTOTALQTAXCAT3HINO — Total State Tax Collections, Hawaii (quarterly)
    STTMINWGHI           — State Minimum Wage Rate, Hawaii (annual)

Output: data/external/fred/{series_id}_{label}.csv  (columns: date, value)

Usage:
    python scripts/data_prep/fetch_fred.py
    python scripts/data_prep/fetch_fred.py --series hawaii_cpi hawaii_hpi
    python scripts/data_prep/fetch_fred.py --start 2015-01-01
    python scripts/data_prep/fetch_fred.py --dry-run
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"

# Map of friendly names to FRED series IDs
SERIES = {
    "hawaii_cpi": {
        "series_id": "CUUSA426SA0",
        "description": "CPI-U All Items, Urban Hawaii (monthly)",
    },
    "hawaii_hpi": {
        "series_id": "HISTHPI",
        "description": "All-Transactions House Price Index, Hawaii (quarterly)",
    },
    "hawaii_unemployment": {
        "series_id": "HIURN",
        "description": "Unemployment Rate, Hawaii (monthly)",
    },
    "hawaii_tax_collections": {
        "series_id": "QTAXTOTALQTAXCAT3HINO",
        "description": "Total State Tax Collections, Hawaii (quarterly, thousands $)",
    },
    "hawaii_min_wage": {
        "series_id": "STTMINWGHI",
        "description": "State Minimum Wage Rate, Hawaii (annual)",
    },
}

# Series whose absence degrades model accuracy (used by growth_rate_loader)
CRITICAL_SERIES = {"hawaii_cpi", "hawaii_hpi"}


class FREDFetcher:
    """Fetch economic time-series from the FRED API."""

    def __init__(self, api_key: str, output_dir: Path):
        self.api_key = api_key
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch_series(
        self,
        series_id: str,
        start_date: str = "2010-01-01",
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch a single FRED series.

        Returns DataFrame with columns: date, value
        """
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date,
            "sort_order": "asc",
        }
        if end_date:
            params["observation_end"] = end_date

        try:
            resp = requests.get(FRED_API_URL, params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"FRED API request failed for {series_id}: {e}")
            return pd.DataFrame()

        data = resp.json()

        # Check for API errors
        if "error_code" in data:
            logger.error(f"FRED API error for {series_id}: {data.get('error_message', 'Unknown')}")
            return pd.DataFrame()

        observations = data.get("observations", [])
        if not observations:
            logger.warning(f"No observations returned for {series_id}")
            return pd.DataFrame()

        rows = []
        for obs in observations:
            value_str = obs.get("value", ".")
            if value_str == ".":  # FRED uses "." for missing values
                continue
            try:
                rows.append({
                    "date": obs["date"],
                    "value": float(value_str),
                })
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping observation in {series_id}: {e}")
                continue

        df = pd.DataFrame(rows)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        logger.info(f"Fetched {len(df)} observations for {series_id}")
        return df

    def fetch_all(
        self,
        series_names: Optional[list[str]] = None,
        start_date: str = "2010-01-01",
    ) -> tuple[dict[str, pd.DataFrame], list[str]]:
        """Fetch multiple FRED series.

        Args:
            series_names: List of friendly names (keys in SERIES dict).
                          If None, fetches all.
            start_date: Start date for all series.

        Returns:
            Tuple of (dict mapping friendly name to DataFrame,
                      list of failed series names).
        """
        if series_names is None:
            series_names = list(SERIES.keys())

        results = {}
        failed = []
        for name in series_names:
            if name not in SERIES:
                logger.warning(f"Unknown series name: {name}. Skipping.")
                continue

            info = SERIES[name]
            logger.info(f"Fetching {name} ({info['series_id']}): {info['description']}")
            df = self.fetch_series(info["series_id"], start_date=start_date)
            if not df.empty:
                results[name] = df
            else:
                failed.append(name)
                logger.warning(f"Failed to fetch {name} ({info['series_id']})")

        return results, failed

    def save(self, dataframes: dict[str, pd.DataFrame]) -> list[Path]:
        """Save all fetched series to CSV files.

        Returns list of paths written.
        """
        paths = []
        for name, df in dataframes.items():
            if df.empty:
                continue

            series_id = SERIES[name]["series_id"]
            filename = f"{series_id}_{name}.csv"
            output_path = self.output_dir / filename
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(df)} rows to {output_path}")
            paths.append(output_path)

        return paths


def main():
    parser = argparse.ArgumentParser(
        description="Fetch FRED economic data for Hawaii tax projections"
    )
    parser.add_argument(
        "--series",
        nargs="+",
        choices=list(SERIES.keys()),
        default=None,
        help="Specific series to fetch (default: all)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2010-01-01",
        help="Start date (YYYY-MM-DD, default: 2010-01-01)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("FRED_API_KEY", ""),
        help="FRED API key (default: from FRED_API_KEY env var)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be fetched without making API calls",
    )
    args = parser.parse_args()

    if args.dry_run:
        names = args.series or list(SERIES.keys())
        for name in names:
            info = SERIES[name]
            print(f"  {name}: {info['series_id']} — {info['description']}")
        print(f"\nWould fetch {len(names)} series from {args.start}")
        return

    if not args.api_key:
        logger.error(
            "No FRED API key. Set FRED_API_KEY env var or pass --api-key. "
            "Register at https://fred.stlouisfed.org/docs/api/api_key.html"
        )
        sys.exit(1)

    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "data" / "external" / "fred"

    fetcher = FREDFetcher(api_key=args.api_key, output_dir=output_dir)
    dataframes, failed = fetcher.fetch_all(series_names=args.series, start_date=args.start)

    if failed:
        critical_failed = [s for s in failed if s in CRITICAL_SERIES]
        if critical_failed:
            logger.error(f"Critical series failed: {critical_failed}")
        else:
            logger.warning(f"Optional series failed: {failed}")

    if dataframes:
        paths = fetcher.save(dataframes)
        print(f"\nSaved {len(paths)} series to {output_dir}/")
        for p in paths:
            print(f"  {p.name}")
    else:
        logger.warning("No data fetched")


if __name__ == "__main__":
    main()
