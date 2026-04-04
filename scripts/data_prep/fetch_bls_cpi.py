#!/usr/bin/env python3
"""
Fetch BLS CPI-U data for Honolulu metro area via the BLS Public Data API.

Series: CUURS49FSA0 — CPI-U All Items, Honolulu (semi-annual: January and July)

This provides a cross-validation source for the FRED Hawaii CPI series
(CUUSA426SA0) with finer Honolulu metro geography.

Output: data/external/bls_cpi/honolulu_cpi.csv  (columns: year, period, value, date)

Usage:
    python scripts/data_prep/fetch_bls_cpi.py
    python scripts/data_prep/fetch_bls_cpi.py --start-year 2015
    python scripts/data_prep/fetch_bls_cpi.py --api-key YOUR_KEY  # v2 for higher rate limits
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

# BLS API endpoints
BLS_API_V1 = "https://api.bls.gov/publicAPI/v1/timeseries/data/"
BLS_API_V2 = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

# Honolulu CPI-U All Items (Not Seasonally Adjusted)
SERIES_ID = "CUURS49FSA0"


class BLSCPIFetcher:
    """Fetch CPI-U for Honolulu from BLS API."""

    def __init__(self, output_dir: Path, api_key: Optional[str] = None):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key
        # v2 with key: 500 requests/day, 10-year range
        # v1 without key: 25 requests/day, 10-year range
        self.api_url = BLS_API_V2 if api_key else BLS_API_V1

    def fetch(
        self,
        start_year: int = 2010,
        end_year: int = 2026,
    ) -> pd.DataFrame:
        """Fetch CPI-U for Honolulu.

        BLS API limits to 20-year range per request.
        Returns DataFrame with columns: year, period, value, date
        """
        # BLS API allows max 20 year span
        if end_year - start_year > 20:
            end_year = start_year + 20

        payload = {
            "seriesid": [SERIES_ID],
            "startyear": str(start_year),
            "endyear": str(end_year),
        }
        if self.api_key:
            payload["registrationkey"] = self.api_key

        headers = {"Content-type": "application/json"}

        try:
            resp = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"BLS API request failed: {e}")
            return pd.DataFrame()

        data = resp.json()

        if data.get("status") != "REQUEST_SUCCEEDED":
            logger.error(f"BLS API error: {data.get('message', 'Unknown error')}")
            return pd.DataFrame()

        results = data.get("Results", {}).get("series", [])
        if not results:
            logger.warning("No series data returned from BLS")
            return pd.DataFrame()

        series_data = results[0].get("data", [])
        if not series_data:
            logger.warning(f"No observations for {SERIES_ID}")
            return pd.DataFrame()

        rows = []
        for obs in series_data:
            try:
                year = int(obs["year"])
                period = obs["periodName"]  # e.g., "January", "July", "Annual"
                period_code = obs["period"]  # e.g., "M01", "M07", "M13"
                value = float(obs["value"])

                # Convert to date for consistency
                month_map = {
                    "M01": "01", "M02": "02", "M03": "03", "M04": "04",
                    "M05": "05", "M06": "06", "M07": "07", "M08": "08",
                    "M09": "09", "M10": "10", "M11": "11", "M12": "12",
                    "M13": "01",  # Annual average — map to Jan
                    "S01": "01", "S02": "07",  # Semi-annual
                }
                month = month_map.get(period_code, "01")
                date = f"{year}-{month}-01"

                rows.append({
                    "year": year,
                    "period": period,
                    "value": value,
                    "date": date,
                })
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping BLS observation: {e}")
                continue

        df = pd.DataFrame(rows)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

        logger.info(f"Fetched {len(df)} CPI observations for Honolulu ({start_year}-{end_year})")
        return df

    def save(self, df: pd.DataFrame) -> Path:
        """Save to CSV."""
        if df.empty:
            logger.warning("No CPI data to save")
            return Path()

        output_path = self.output_dir / "honolulu_cpi.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved BLS CPI data to {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Fetch BLS CPI-U for Honolulu")
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("BLS_API_KEY", ""),
        help="BLS API v2 key (optional, increases rate limits)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "data" / "external" / "bls_cpi"

    fetcher = BLSCPIFetcher(
        output_dir=output_dir,
        api_key=args.api_key if args.api_key else None,
    )
    df = fetcher.fetch(start_year=args.start_year, end_year=args.end_year)
    if not df.empty:
        fetcher.save(df)
        print(f"\nSaved {len(df)} observations to {output_dir}/honolulu_cpi.csv")
    else:
        print("No data fetched")


if __name__ == "__main__":
    main()
