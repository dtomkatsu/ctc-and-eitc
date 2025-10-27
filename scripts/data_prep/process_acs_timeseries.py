#!/usr/bin/env python3
"""Process raw ACS 1-year tables into harmonized time-series panels."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)

# Table definitions (mirrors download_acs_tables.py)
CRITICAL_TABLES: Dict[str, str] = {
    "B19001": "Income Distribution",
    "B19013": "Median Income",
    "B19019": "Income by Household Type",
    "B12001": "Marital Status",
    "B01001": "Age/Sex Distribution",
}

ADDITIONAL_TABLES: Dict[str, str] = {
    "B11001": "Household Type",
    "B09002": "Children by Family Type",
    "B07001": "Migration",
    "B23025": "Employment",
    "B25003": "Homeownership",
    "B25077": "Home Value",
}

# Column rename map by table_code
COLUMN_RENAMES: Dict[str, Dict[str, str]] = {
    "B19013": {
        "B19013_001E": "median_household_income",
    },
    "B19019": {
        "B19019_002E": "median_income_married_couple",
        "B19019_003E": "median_income_male_no_spouse",
        "B19019_004E": "median_income_female_no_spouse",
    },
    "B23025": {
        "B23025_002E": "population_16_over",
        "B23025_004E": "civilian_labor_force",
        "B23025_005E": "employed",
        "B23025_006E": "unemployed",
        "B23025_007E": "armed_forces",
    },
    "B25003": {
        "B25003_002E": "owner_occupied",
        "B25003_003E": "renter_occupied",
    },
    "B25077": {
        "B25077_001E": "median_home_value",
    },
}

# Preferred column ordering additions
FRONT_COLUMNS = ["year", "year_weight", "NAME", "state", "table_code"]

# Year weight overrides (2020 experimental down-weighted if ever added)
YEAR_WEIGHTS = {
    2020: 0.35,
}


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_and_clean_table(table_code: str, category_dir: Path) -> Optional[pd.DataFrame]:
    """Load per-year CSVs for a table and return cleaned DataFrame."""

    per_year_files: List[Path] = sorted(
        f for f in category_dir.glob(f"{table_code}_*.csv") if "combined" not in f.name
    )

    if not per_year_files:
        LOGGER.warning("No raw files found for %s in %s", table_code, category_dir)
        return None

    frames: List[pd.DataFrame] = []

    for csv_path in per_year_files:
        LOGGER.debug("Loading %s", csv_path)
        df = pd.read_csv(csv_path)

        # Drop margin of error columns
        df = df.loc[:, ~df.columns.str.endswith("M")]

        # Coerce numeric columns (excluding metadata columns)
        for column in df.columns:
            if column in {"NAME", "state", "table_code"}:
                continue
            df[column] = pd.to_numeric(df[column], errors="coerce")

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("year").reset_index(drop=True)
    combined["year"] = combined["year"].astype(int)

    # Apply renames if specified
    rename_map = COLUMN_RENAMES.get(table_code, {})
    if rename_map:
        combined = combined.rename(columns=rename_map)

    # Assign year weights (default 1.0)
    combined["year_weight"] = combined["year"].map(YEAR_WEIGHTS).fillna(1.0)

    # Reorder columns so metadata comes first
    ordered_cols = [col for col in FRONT_COLUMNS if col in combined.columns]
    remaining_cols = [col for col in combined.columns if col not in ordered_cols]
    combined = combined[ordered_cols + remaining_cols]

    return combined


def create_long_format(df: pd.DataFrame, id_columns: List[str]) -> pd.DataFrame:
    """Convert wide ACS table to long format for category/value analysis."""

    value_columns = [col for col in df.columns if col not in id_columns]
    long_df = df.melt(
        id_vars=id_columns,
        value_vars=value_columns,
        var_name="variable",
        value_name="value",
    )
    return long_df


def process_tables(input_dir: Path, output_dir: Path) -> None:
    """Process all ACS tables into harmonized outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    wide_dir = output_dir / "wide"
    long_dir = output_dir / "long"
    wide_dir.mkdir(exist_ok=True)
    long_dir.mkdir(exist_ok=True)

    tables = {**CRITICAL_TABLES, **ADDITIONAL_TABLES}

    for table_code in tables:
        category = "critical" if table_code in CRITICAL_TABLES else "additional"
        category_dir = input_dir / category

        df = load_and_clean_table(table_code, category_dir)
        if df is None:
            continue

        wide_path = wide_dir / f"{table_code}_wide.parquet"
        LOGGER.info("Writing wide table: %s", wide_path)
        df.to_parquet(wide_path, index=False)

        # Save CSV snapshot for easier inspection
        csv_path = wide_dir / f"{table_code}_wide.csv"
        df.to_csv(csv_path, index=False)

        # Long-format outputs for distribution-style tables
        if table_code in {"B19001", "B12001", "B11001", "B09002"}:
            id_columns = [col for col in ["year", "year_weight", "NAME", "state", "table_code"] if col in df.columns]
            long_df = create_long_format(df, id_columns=id_columns)
            long_path = long_dir / f"{table_code}_long.parquet"
            LOGGER.info("Writing long table: %s", long_path)
            long_df.to_parquet(long_path, index=False)
            long_csv_path = long_dir / f"{table_code}_long.csv"
            long_df.to_csv(long_csv_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process ACS raw CSV tables into harmonized time-series panels",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/acs_1yr"),
        help="Directory containing raw ACS downloads",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/acs_timeseries"),
        help="Directory to store processed outputs",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)

    LOGGER.info("Processing ACS tables from %s", args.input_dir)
    process_tables(args.input_dir, args.output_dir)
    LOGGER.info("Processing complete. Outputs written to %s", args.output_dir)


if __name__ == "__main__":
    main()
