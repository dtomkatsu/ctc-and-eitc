#!/usr/bin/env python3
"""
Fetch BLS OES state-level Excel files for Hawaii wage growth analysis.

Downloads annual OES state zip files from BLS and extracts the state-level
Excel sheet, saving as state_M{year}_dl.xlsx in data/external/bls_oes/.

BLS OES zip URL pattern: https://www.bls.gov/oes/special.requests/oesm{YY}st.zip
  e.g., 2024 → oesm24st.zip, 2014 → oesm14st.zip

Usage:
    python scripts/data_prep/fetch_bls_oes_historical.py
    python scripts/data_prep/fetch_bls_oes_historical.py --start 2014 --end 2024
"""

import argparse
import io
import logging
import os
import sys
import zipfile
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# BLS OES state zip URL template
BLS_OES_URL = "https://www.bls.gov/oes/special.requests/oesm{yy}st.zip"

# Expected file name inside the zip (state-level flat file)
# BLS naming varies by year:
# 2024: state_M2024_dl.xlsx
# 2020+: state_M{year}_dl.xlsx
# 2014-2019: state_M{year}_dl.xls
EXPECTED_NAMES = [
    "state_M{year}_dl.xlsx",
    "state_M{year}_dl.xls",
    "state_{year}.xlsx",
    "state_{year}.xls",
]


def _year_to_yy(year: int) -> str:
    """Convert 4-digit year to 2-digit zero-padded string."""
    return str(year)[-2:].zfill(2)


def fetch_year(year: int, output_dir: Path, session: requests.Session) -> bool:
    """Download BLS OES state zip for a given year and extract the state file."""
    yy = _year_to_yy(year)
    url = BLS_OES_URL.format(yy=yy)
    logger.info(f"  [{year}] Fetching {url}")

    try:
        resp = session.get(url, timeout=120, stream=True)
        resp.raise_for_status()
    except requests.HTTPError as e:
        logger.error(f"  [{year}] HTTP error: {e}")
        return False
    except requests.RequestException as e:
        logger.error(f"  [{year}] Download failed: {e}")
        return False

    content = b""
    for chunk in resp.iter_content(chunk_size=1024 * 1024):
        content += chunk
    logger.info(f"  [{year}] Downloaded {len(content) / 1e6:.1f} MB")

    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile:
        logger.error(f"  [{year}] Bad zip file")
        return False

    names_in_zip = zf.namelist()
    logger.info(f"  [{year}] Zip contains: {len(names_in_zip)} files")

    # Find the state-level Excel file
    found = None
    for pattern in EXPECTED_NAMES:
        candidate = pattern.format(year=year)
        # Check exact match first, then case-insensitive
        if candidate in names_in_zip:
            found = candidate
            break
        matches = [n for n in names_in_zip if n.lower() == candidate.lower()]
        if matches:
            found = matches[0]
            break

    if found is None:
        # Try to find any state-level file (exclude temp/lock files starting with ~$)
        state_files = [n for n in names_in_zip
                       if 'state' in n.lower()
                       and (n.endswith('.xls') or n.endswith('.xlsx'))
                       and not os.path.basename(n).startswith('~')]
        if state_files:
            found = state_files[0]
            logger.warning(f"  [{year}] Using fallback file: {found}")
        else:
            logger.error(f"  [{year}] No state Excel file found in zip. Contents: {names_in_zip[:10]}")
            return False

    ext = Path(found).suffix
    out_name = f"state_M{year}_dl{ext}"
    out_path = output_dir / out_name

    with zf.open(found) as src, open(out_path, 'wb') as dst:
        dst.write(src.read())

    logger.info(f"  [{year}] Saved → {out_name}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Fetch BLS OES historical state files")
    parser.add_argument("--start", type=int, default=2014, help="Start year (default: 2014)")
    parser.add_argument("--end", type=int, default=2024, help="End year (default: 2024)")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "data" / "external" / "bls_oes"
    output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (research; ctc-eitc Hawaii tax model)"
    })

    success = []
    failed = []

    for year in range(args.start, args.end + 1):
        # Skip if already downloaded
        existing = list(output_dir.glob(f"state*{year}*"))
        if existing:
            logger.info(f"  [{year}] Already exists: {existing[0].name} — skipping")
            success.append(year)
            continue

        ok = fetch_year(year, output_dir, session)
        (success if ok else failed).append(year)

    logger.info(f"\nDone: {len(success)} succeeded, {len(failed)} failed")
    if failed:
        logger.warning(f"Failed years: {failed}")
    else:
        logger.info("All years downloaded successfully")
        logger.info(f"\nNext step: run scripts/data_prep/harmonize_bls_oes_years.py")


if __name__ == "__main__":
    main()
