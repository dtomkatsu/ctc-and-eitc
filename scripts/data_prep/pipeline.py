#!/usr/bin/env python3
"""
Master data pipeline for the Hawaii tax projection model.

Orchestrates fetching, processing, and (optionally) tax unit regeneration
across all automated data sources. Tracks data freshness via a manifest
file and supports scheduled execution with macOS notifications.

Usage:
    python scripts/data_prep/pipeline.py                    # Fetch all, skip if fresh
    python scripts/data_prep/pipeline.py --force             # Force re-download everything
    python scripts/data_prep/pipeline.py --sources bls bea   # Only specific sources
    python scripts/data_prep/pipeline.py --check-only        # Report freshness, don't download
    python scripts/data_prep/pipeline.py --regenerate        # Fetch + rebuild tax units
    python scripts/data_prep/pipeline.py --scheduled         # Scheduled mode: fetch + notify
    python scripts/data_prep/pipeline.py --manual-status     # Show status of manual sources
"""

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env before anything else
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

import pandas as pd

from config.pipeline_config import PipelineConfig, SourceConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    success: bool
    message: str
    files_updated: list[Path] = field(default_factory=list)
    skipped: bool = False  # True if skipped due to freshness


@dataclass
class PipelineReport:
    timestamp: datetime
    results: dict[str, StepResult]
    manual_warnings: list[str] = field(default_factory=list)

    @property
    def any_updated(self) -> bool:
        return any(
            r.success and not r.skipped and r.files_updated
            for r in self.results.values()
        )

    def to_text(self) -> str:
        lines = [
            f"Pipeline Report — {self.timestamp:%Y-%m-%d %H:%M}",
            "=" * 60,
        ]
        for source, result in self.results.items():
            status = "SKIP" if result.skipped else ("OK" if result.success else "FAIL")
            lines.append(f"  [{status:>4}] {source}: {result.message}")
            for f in result.files_updated:
                lines.append(f"         → {f}")
        if self.manual_warnings:
            lines.append("")
            lines.append("Manual source warnings:")
            for w in self.manual_warnings:
                lines.append(f"  ! {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Manual source checker
# ---------------------------------------------------------------------------

MANUAL_SOURCES = {
    "dotax_soi": {
        "description": "Hawaii Dept. of Taxation Statistics of Income (2022)",
        "files": ["data/raw/Dotax Soi 2022 - 5A.csv"],
        "update_instructions": (
            "Request updated SOI tables from DOTAX or download from "
            "https://tax.hawaii.gov/stats/ when available."
        ),
    },
    "irs_soi_zip": {
        "description": "IRS SOI ZIP Code Data",
        "files": ["data/irs_soi/22zp12hi.xlsx"],
        "update_instructions": (
            "Download from https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics-zip-code-data-soi"
        ),
    },
    "dotax_collections": {
        "description": "Hawaii monthly tax collections (2016-present)",
        "files": ["data/raw/hawaii_tax_collections_2016_2025.csv"],
        "update_instructions": (
            "Update manually from DOTAX monthly revenue reports."
        ),
    },
}


def check_manual_sources(project_root: Path) -> list[str]:
    """Return warnings for stale or missing manual data files."""
    warnings = []
    now = datetime.now()
    for name, info in MANUAL_SOURCES.items():
        for rel_path in info["files"]:
            full = project_root / rel_path
            if not full.exists():
                warnings.append(
                    f"{info['description']}: file missing ({rel_path}). "
                    f"{info['update_instructions']}"
                )
            else:
                age_days = (now - datetime.fromtimestamp(full.stat().st_mtime)).days
                if age_days > 365:
                    warnings.append(
                        f"{info['description']}: {age_days} days old ({rel_path}). "
                        f"{info['update_instructions']}"
                    )
    return warnings


# ---------------------------------------------------------------------------
# Manifest (freshness tracking)
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


class Manifest:
    """Tracks when each data source was last fetched."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict = {"version": 1, "sources": {}}
        if self.path.exists():
            with open(self.path) as f:
                self.data = json.load(f)

    def last_fetched(self, source: str) -> Optional[datetime]:
        entry = self.data.get("sources", {}).get(source, {})
        ts = entry.get("last_fetched")
        if ts:
            return datetime.fromisoformat(ts)
        return None

    def is_fresh(self, source: str, max_age_days: int) -> bool:
        last = self.last_fetched(source)
        if last is None:
            return False
        return (datetime.now() - last) < timedelta(days=max_age_days)

    def record(self, source: str, files: list[Path]):
        self.data.setdefault("sources", {})[source] = {
            "last_fetched": datetime.now().isoformat(),
            "files": [str(f) for f in files],
        }
        self.data["last_run"] = datetime.now().isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def summary(self) -> dict[str, Optional[str]]:
        """Return {source_name: last_fetched_iso_or_None}."""
        out = {}
        for name in self.data.get("sources", {}):
            out[name] = self.data["sources"][name].get("last_fetched")
        return out


# ---------------------------------------------------------------------------
# DataPipeline
# ---------------------------------------------------------------------------

class DataPipeline:
    """Orchestrates all data fetching, processing, and regeneration."""

    def __init__(self, project_root: Path, config: Optional[PipelineConfig] = None):
        self.root = project_root
        self.config = config or PipelineConfig.default()
        self.manifest = Manifest(self.root / self.config.manifest_path)

    # ------------------------------------------------------------------
    # Individual source fetchers
    # ------------------------------------------------------------------

    def fetch_census_pums(self, force: bool = False) -> StepResult:
        """Download Census PUMS person + household files for Hawaii via FTP.

        Uses the Census FTP bulk download (all ~290 columns) by default.
        The column guard only blocks API-method downloads from overwriting
        full FTP data; FTP downloads always proceed since they provide
        the complete column set.
        """
        source = "census_pums"
        cfg = self.config.sources.get(source)
        if cfg and not cfg.enabled:
            return StepResult(True, "Disabled in config", skipped=True)
        if not force and self.manifest.is_fresh(source, cfg.freshness_days if cfg else 180):
            return StepResult(True, "Data is fresh, skipping", skipped=True)

        data_dir = self.root / "data" / "raw" / "pums"

        # Determine the PUMS year: use config if set, otherwise latest available
        pums_year = int(os.getenv("PUMS_YEAR", "2024"))

        try:
            from scripts.data_prep.download_pums import PUMSDownloader

            downloader = PUMSDownloader(
                year=pums_year, state="15", data_dir=data_dir,
                api_key=os.getenv("CENSUS_API_KEY", ""),
                method="ftp",
            )
            ok = downloader.download_all()
            if ok:
                files = list(data_dir.glob("psam_*.csv"))
                self.manifest.record(source, files)
                return StepResult(True, f"Downloaded PUMS {pums_year} via FTP", files)
            return StepResult(False, f"PUMS {pums_year} FTP download returned errors")
        except Exception as e:
            return StepResult(False, f"PUMS download failed: {e}")

    def fetch_acs_tables(self, force: bool = False) -> StepResult:
        """Download ACS 1-year aggregate tables."""
        source = "acs_tables"
        cfg = self.config.sources.get(source)
        if cfg and not cfg.enabled:
            return StepResult(True, "Disabled in config", skipped=True)
        if not force and self.manifest.is_fresh(source, cfg.freshness_days if cfg else 180):
            return StepResult(True, "Data is fresh, skipping", skipped=True)

        api_key = os.getenv("CENSUS_API_KEY", "")
        if not api_key or api_key == "YOUR_CENSUS_API_KEY":
            return StepResult(False, "CENSUS_API_KEY not set in .env")

        try:
            from scripts.data_prep.download_acs_tables import ACSTableDownloader

            output_dir = self.root / "data" / "raw" / "acs_1yr"
            downloader = ACSTableDownloader(api_key=api_key, output_dir=output_dir)
            results = downloader.download_all_tables(start_year=2015, end_year=2024)
            n_tables = sum(len(years) for years in results.values())
            files = list(output_dir.rglob("*.csv"))
            self.manifest.record(source, files)
            return StepResult(True, f"Downloaded {n_tables} table-year combos", files)
        except Exception as e:
            return StepResult(False, f"ACS download failed: {e}")

    def fetch_bls_oes(self, force: bool = False) -> StepResult:
        """Download BLS OES state-level wage files."""
        source = "bls_oes"
        cfg = self.config.sources.get(source)
        if cfg and not cfg.enabled:
            return StepResult(True, "Disabled in config", skipped=True)
        if not force and self.manifest.is_fresh(source, cfg.freshness_days if cfg else 180):
            return StepResult(True, "Data is fresh, skipping", skipped=True)

        try:
            import requests as req
            from scripts.data_prep.fetch_bls_oes_historical import fetch_year

            output_dir = self.root / "data" / "external" / "bls_oes"
            output_dir.mkdir(parents=True, exist_ok=True)

            session = req.Session()
            session.headers.update(
                {"User-Agent": "Mozilla/5.0 (research; ctc-eitc Hawaii tax model)"}
            )

            fetched = []
            for year in range(2014, datetime.now().year + 1):
                existing = list(output_dir.glob(f"state*{year}*"))
                if existing and not force:
                    continue
                ok = fetch_year(year, output_dir, session)
                if ok:
                    fetched.append(year)

            files = list(output_dir.glob("state_M*"))
            self.manifest.record(source, files)
            msg = f"BLS OES up to date. New years fetched: {fetched or 'none'}"
            return StepResult(True, msg, files)
        except Exception as e:
            return StepResult(False, f"BLS OES download failed: {e}")

    def fetch_bea_state_income(self, force: bool = False) -> StepResult:
        """Fetch BEA SAINC5N for Hawaii via API."""
        source = "bea_state_income"
        cfg = self.config.sources.get(source)
        if cfg and not cfg.enabled:
            return StepResult(True, "Disabled in config", skipped=True)
        if not force and self.manifest.is_fresh(source, cfg.freshness_days if cfg else 90):
            return StepResult(True, "Data is fresh, skipping", skipped=True)

        api_key = os.getenv("BEA_API_KEY", "")
        if not api_key:
            return StepResult(
                False,
                "BEA_API_KEY not set. Register at https://apps.bea.gov/API/signup/ "
                "and add to .env",
            )

        try:
            from scripts.data_prep.fetch_bea_state_income import BEAFetcher

            fetcher = BEAFetcher(api_key=api_key, output_dir=self.root / "data" / "raw")
            df = fetcher.fetch_sainc5n(start_year=2019, end_year=2024)
            if df.empty:
                return StepResult(False, "BEA API returned no data")
            path = fetcher.save(df)
            self.manifest.record(source, [path])
            return StepResult(True, f"Fetched {len(df)} BEA records", [path])
        except Exception as e:
            return StepResult(False, f"BEA fetch failed: {e}")

    def fetch_fred(self, force: bool = False) -> StepResult:
        """Fetch FRED economic time-series for Hawaii (CPI, HPI, unemployment, etc.)."""
        source = "fred"
        cfg = self.config.sources.get(source)
        if cfg and not cfg.enabled:
            return StepResult(True, "Disabled in config", skipped=True)
        if not force and self.manifest.is_fresh(source, cfg.freshness_days if cfg else 30):
            return StepResult(True, "Data is fresh, skipping", skipped=True)

        api_key = os.getenv("FRED_API_KEY", "")
        if not api_key:
            return StepResult(
                False,
                "FRED_API_KEY not set. Register at https://fred.stlouisfed.org/docs/api/api_key.html "
                "and add to .env",
            )

        try:
            from scripts.data_prep.fetch_fred import FREDFetcher, CRITICAL_SERIES

            fetcher = FREDFetcher(
                api_key=api_key,
                output_dir=self.root / "data" / "external" / "fred",
            )
            dataframes, failed = fetcher.fetch_all(start_date="2010-01-01")
            if not dataframes:
                return StepResult(False, "FRED: all series failed")

            paths = fetcher.save(dataframes)
            self.manifest.record(source, paths)
            series_names = list(dataframes.keys())

            # Check if any critical series failed
            critical_failed = [s for s in failed if s in CRITICAL_SERIES]
            optional_failed = [s for s in failed if s not in CRITICAL_SERIES]

            if critical_failed:
                return StepResult(
                    False,
                    f"Critical FRED series failed: {', '.join(critical_failed)}. "
                    f"Succeeded: {', '.join(series_names)}",
                    paths,
                )

            msg = f"Fetched {len(dataframes)} FRED series: {', '.join(series_names)}"
            if optional_failed:
                msg += f" (optional unavailable: {', '.join(optional_failed)})"
            return StepResult(True, msg, paths)
        except Exception as e:
            return StepResult(False, f"FRED fetch failed: {e}")

    def fetch_bls_cpi(self, force: bool = False) -> StepResult:
        """Fetch BLS CPI-U for Honolulu metro area."""
        source = "bls_cpi"
        cfg = self.config.sources.get(source)
        if cfg and not cfg.enabled:
            return StepResult(True, "Disabled in config", skipped=True)
        if not force and self.manifest.is_fresh(source, cfg.freshness_days if cfg else 30):
            return StepResult(True, "Data is fresh, skipping", skipped=True)

        try:
            from scripts.data_prep.fetch_bls_cpi import BLSCPIFetcher

            fetcher = BLSCPIFetcher(
                output_dir=self.root / "data" / "external" / "bls_cpi",
            )
            df = fetcher.fetch(start_year=2015)
            if df.empty:
                return StepResult(False, "BLS CPI API returned no data")
            path = fetcher.save(df)
            self.manifest.record(source, [path])
            return StepResult(True, f"Fetched {len(df)} BLS CPI observations", [path])
        except Exception as e:
            return StepResult(False, f"BLS CPI fetch failed: {e}")

    # ------------------------------------------------------------------
    # Post-fetch processing
    # ------------------------------------------------------------------

    def harmonize_bls_oes(self) -> StepResult:
        """Harmonize raw BLS OES Excel files into timeseries."""
        try:
            from scripts.data_prep.harmonize_bls_oes_years import BLSOESHarmonizer

            input_dir = self.root / "data" / "external" / "bls_oes"
            output_dir = self.root / "data" / "processed"

            harmonizer = BLSOESHarmonizer(
                data_dir=input_dir, output_dir=output_dir
            )
            ts_df, occ_df = harmonizer.process_all_years(start_year=2009, end_year=2024)

            files = [
                output_dir / "bls_oes_timeseries.parquet",
                output_dir / "bls_oes_occupation_summary.parquet",
            ]
            return StepResult(
                True,
                f"Harmonized {len(ts_df)} BLS OES records across {ts_df['year'].nunique()} years",
                [f for f in files if f.exists()],
            )
        except Exception as e:
            return StepResult(False, f"BLS harmonization failed: {e}")

    def regenerate_tax_units(self) -> StepResult:
        """Run regenerate_tax_units.py to rebuild calibrated tax units."""
        try:
            script = self.root / "scripts" / "regenerate_tax_units.py"
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode == 0:
                parquets = sorted(
                    (self.root / "data" / "processed").glob(
                        "tax_units_filing_status_calibrated_*.parquet"
                    )
                )
                latest = [parquets[-1]] if parquets else []
                return StepResult(True, "Tax units regenerated", latest)
            return StepResult(False, f"regenerate_tax_units.py failed:\n{result.stderr[-500:]}")
        except Exception as e:
            return StepResult(False, f"Tax unit regeneration failed: {e}")

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(
        self,
        sources: Optional[list[str]] = None,
        force: bool = False,
        regenerate: bool = False,
        check_only: bool = False,
    ) -> PipelineReport:
        """Run the pipeline.

        Args:
            sources: Which sources to fetch (None = all enabled).
            force: Ignore freshness, re-download everything.
            regenerate: After fetching, rebuild tax units.
            check_only: Just report freshness, don't download.
        """
        results: dict[str, StepResult] = {}

        # Map source names to fetch methods
        fetch_map = {
            "census_pums": self.fetch_census_pums,
            "acs_tables": self.fetch_acs_tables,
            "bls_oes": self.fetch_bls_oes,
            "bea_state_income": self.fetch_bea_state_income,
            "fred": self.fetch_fred,
            "bls_cpi": self.fetch_bls_cpi,
        }

        if sources:
            fetch_map = {k: v for k, v in fetch_map.items() if k in sources}

        # Check-only mode
        if check_only:
            for name in fetch_map:
                cfg = self.config.sources.get(name)
                fresh = self.manifest.is_fresh(name, cfg.freshness_days if cfg else 180)
                last = self.manifest.last_fetched(name)
                if last:
                    age = (datetime.now() - last).days
                    status = f"Last fetched {age} days ago ({last:%Y-%m-%d})"
                    status += " [FRESH]" if fresh else " [STALE]"
                else:
                    status = "Never fetched [STALE]"
                results[name] = StepResult(True, status, skipped=True)

            manual_warnings = check_manual_sources(self.root)
            return PipelineReport(datetime.now(), results, manual_warnings)

        # Parallel fetch phase (independent sources)
        logger.info("=" * 60)
        logger.info("DATA PIPELINE — Fetch Phase")
        logger.info("=" * 60)

        if self.config.parallel_downloads and len(fetch_map) > 1:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
                futures = {
                    pool.submit(fn, force): name
                    for name, fn in fetch_map.items()
                }
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        results[name] = future.result()
                    except Exception as e:
                        results[name] = StepResult(False, f"Unexpected error: {e}")
                    logger.info(
                        f"  [{name}] {'OK' if results[name].success else 'FAIL'}: "
                        f"{results[name].message}"
                    )
        else:
            for name, fn in fetch_map.items():
                try:
                    results[name] = fn(force)
                except Exception as e:
                    results[name] = StepResult(False, f"Unexpected error: {e}")
                logger.info(
                    f"  [{name}] {'OK' if results[name].success else 'FAIL'}: "
                    f"{results[name].message}"
                )

        # Post-fetch processing (sequential, with dependency checks)
        logger.info("")
        logger.info("Post-fetch processing:")

        bls_ok = results.get("bls_oes", StepResult(False, "")).success
        if bls_ok:
            results["bls_harmonize"] = self.harmonize_bls_oes()
            logger.info(f"  [bls_harmonize] {results['bls_harmonize'].message}")
        else:
            results["bls_harmonize"] = StepResult(
                False, "Skipped — BLS OES fetch did not succeed", skipped=True
            )

        # Regenerate tax units if requested.
        # Only proceed if PUMS data is known-good: either a fresh fetch
        # succeeded this run, or PUMS wasn't in the fetch list (meaning
        # existing on-disk data is being used as-is).
        if regenerate or self.config.auto_regenerate:
            pums_was_fetched = "census_pums" in (sources or [])
            pums_ok = results.get("census_pums", StepResult(False, "")).success
            if not pums_was_fetched or pums_ok:
                logger.info("\nRegenerating tax units...")
                results["regenerate"] = self.regenerate_tax_units()
                logger.info(f"  [regenerate] {results['regenerate'].message}")
            else:
                results["regenerate"] = StepResult(
                    False, "Skipped — PUMS fetch failed", skipped=True
                )

        manual_warnings = check_manual_sources(self.root)
        return PipelineReport(datetime.now(), results, manual_warnings)


# ---------------------------------------------------------------------------
# macOS notification
# ---------------------------------------------------------------------------

def notify_macos(title: str, message: str):
    """Send a macOS notification via osascript."""
    try:
        subprocess.run(
            [
                "osascript",
                "-e",
                f'display notification "{message}" with title "{title}"',
            ],
            check=False,
            timeout=5,
        )
    except Exception:
        pass


def git_commit_if_updated(project_root: Path, report: PipelineReport) -> bool:
    """Auto-commit to git if any data was updated. Returns True if commit succeeded."""
    if not report.any_updated:
        return False

    try:
        # List updated files
        updated_sources = [
            name
            for name, r in report.results.items()
            if r.success and not r.skipped and r.files_updated
        ]
        if not updated_sources:
            return False

        # Stage the manifest (always updated if any source succeeded)
        manifest_path = project_root / "data" / "pipeline_manifest.json"
        subprocess.run(
            ["git", "add", str(manifest_path)],
            cwd=str(project_root),
            check=True,
            capture_output=True,
        )

        # Stage any updated data files
        for source_name, result in report.results.items():
            for file_path in result.files_updated:
                if file_path.exists():
                    subprocess.run(
                        ["git", "add", str(file_path)],
                        cwd=str(project_root),
                        check=True,
                        capture_output=True,
                    )

        # Build commit message
        msg_lines = [
            f"Auto-update data pipeline — {datetime.now():%Y-%m-%d %H:%M}",
            "",
            "Sources updated:",
        ]
        for source in updated_sources:
            msg_lines.append(f"  • {source}")
        msg_lines.extend(
            [
                "",
                "Automated commit from monthly pipeline run.",
                "",
                "Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>",
            ]
        )
        commit_msg = "\n".join(msg_lines)

        # Commit
        result = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=str(project_root),
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            logger.info("✓ Auto-committed updated data to git")
            return True
        elif "nothing to commit" in result.stdout or "nothing to commit" in result.stderr:
            logger.info("No git changes to commit (staging area empty)")
            return False
        else:
            logger.error(f"Git commit failed: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"Git auto-commit failed: {e}")
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Hawaii tax model data pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["census_pums", "acs_tables", "bls_oes", "bea_state_income", "fred", "bls_cpi"],
        help="Only fetch specific sources",
    )
    parser.add_argument("--force", action="store_true", help="Ignore freshness, re-download")
    parser.add_argument("--check-only", action="store_true", help="Report freshness only")
    parser.add_argument("--regenerate", action="store_true", help="Rebuild tax units after fetch")
    parser.add_argument("--scheduled", action="store_true", help="Scheduled mode: fetch + notify")
    parser.add_argument("--manual-status", action="store_true", help="Show manual source status")
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    config = PipelineConfig.default()

    # Set up log file for scheduled runs
    if args.scheduled:
        config.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = project_root / config.log_dir / f"{datetime.now():%Y-%m-%d}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(file_handler)

    pipeline = DataPipeline(project_root, config)

    # Manual status mode
    if args.manual_status:
        warnings = check_manual_sources(project_root)
        if warnings:
            print("Manual source status:")
            for w in warnings:
                print(f"  ! {w}")
        else:
            print("All manual source files present and < 1 year old.")
        return

    # Run pipeline
    report = pipeline.run(
        sources=args.sources,
        force=args.force,
        regenerate=args.regenerate,
        check_only=args.check_only,
    )

    # Print report
    print("\n" + report.to_text())

    # Scheduled mode: auto-commit and notify
    if args.scheduled:
        if report.any_updated:
            # Auto-commit updated data files and manifest
            git_commit_if_updated(project_root, report)

            # Notify user
            updated_sources = [
                name
                for name, r in report.results.items()
                if r.success and not r.skipped and r.files_updated
            ]
            notify_macos(
                "Hawaii Tax Model",
                f"New data available: {', '.join(updated_sources)}. "
                f"Run pipeline with --regenerate to update tax units.",
            )

    # Exit with error if any non-skipped step failed
    if any(not r.success and not r.skipped for r in report.results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
