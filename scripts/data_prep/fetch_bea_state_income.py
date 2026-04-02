#!/usr/bin/env python3
"""
Fetch BEA State Annual Income (SAINC5N) for Hawaii via the BEA REST API.

Produces data/raw/bea_hawaii_sainc5n.csv with columns:
    line_code, description, year, value_thousands

Line codes fetched:
    10 — Personal income
    46 — Dividends, interest, and rent
    47 — Personal current transfer receipts
    50 — Wages and salaries
    70 — Proprietors' income

Usage:
    python scripts/data_prep/fetch_bea_state_income.py
    python scripts/data_prep/fetch_bea_state_income.py --start-year 2018 --end-year 2024
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BEA_API_URL = "https://apps.bea.gov/api/data/"

LINE_CODES = {
    10: "Personal income",
    46: "Dividends interest and rent",
    47: "Personal current transfer receipts",
    50: "Wages and salaries",
    70: "Proprietors income",
}


class BEAFetcher:
    """Fetch SAINC5N data for Hawaii from the BEA API."""

    def __init__(self, api_key: str, output_dir: Path):
        self.api_key = api_key
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch_sainc5n(
        self, start_year: int = 2019, end_year: int = 2024
    ) -> pd.DataFrame:
        """Fetch SAINC5N table for Hawaii.

        Returns DataFrame matching the existing CSV schema:
            line_code, description, year, value_thousands
        """
        years_str = ",".join(str(y) for y in range(start_year, end_year + 1))

        rows = []
        logger.info(f"Fetching BEA SAINC5N for Hawaii, years {start_year}-{end_year}")

        # BEA API requires separate calls for each line code (no comma-separated lists)
        for line_code in LINE_CODES:
            params = {
                "UserID": self.api_key,
                "method": "GetData",
                "datasetname": "Regional",
                "TableName": "SAINC5N",
                "GeoFips": "15000",  # Hawaii
                "LineCode": str(line_code),
                "Year": years_str,
                "ResultFormat": "JSON",
            }

            try:
                resp = requests.get(BEA_API_URL, params=params, timeout=30)
                resp.raise_for_status()
            except requests.RequestException as e:
                logger.error(f"BEA API request failed for line code {line_code}: {e}")
                continue

            data = resp.json()

            # BEA wraps results in BEAAPI -> Results -> Data
            try:
                records = data["BEAAPI"]["Results"]["Data"]
            except (KeyError, TypeError):
                error_msg = data.get("BEAAPI", {}).get("Results", {}).get("Error", "Unknown error")
                logger.error(f"BEA API error for line code {line_code}: {error_msg}")
                continue

            for rec in records:
                try:
                    year = int(rec["TimePeriod"])
                    # BEA returns values as strings, sometimes with commas
                    value_str = rec.get("DataValue", "0").replace(",", "")
                    value = int(value_str)

                    rows.append(
                        {
                            "line_code": line_code,
                            "description": LINE_CODES.get(line_code, ""),
                            "year": year,
                            "value_thousands": value,
                        }
                    )
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping record for line {line_code}: {e}")
                    continue

        df = pd.DataFrame(rows).sort_values(["line_code", "year"]).reset_index(drop=True)
        logger.info(f"Fetched {len(df)} records from BEA API")
        return df

    def save(self, df: pd.DataFrame) -> Path:
        """Save to CSV matching existing schema."""
        if df.empty:
            logger.warning("No data to save")
            return Path()

        output_path = self.output_dir / "bea_hawaii_sainc5n.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved BEA data to {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Fetch BEA SAINC5N for Hawaii")
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--api-key", type=str, default=os.getenv("BEA_API_KEY", ""))
    args = parser.parse_args()

    if not args.api_key:
        logger.error(
            "No BEA API key. Set BEA_API_KEY env var or pass --api-key. "
            "Register at https://apps.bea.gov/API/signup/"
        )
        sys.exit(1)

    project_root = Path(__file__).parent.parent.parent
    fetcher = BEAFetcher(api_key=args.api_key, output_dir=project_root / "data" / "raw")
    df = fetcher.fetch_sainc5n(start_year=args.start_year, end_year=args.end_year)
    if not df.empty:
        fetcher.save(df)


if __name__ == "__main__":
    main()
