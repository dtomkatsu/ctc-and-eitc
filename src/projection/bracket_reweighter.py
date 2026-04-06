"""
Bracket-Level Filer Count Scaling (Priority 2 — IRS Historic Table 2 calibration)

After income growth shifts filers between brackets via nominal AGI increases,
this module reweights records per AGI bracket so the total weighted filer count
matches projected 2027 targets derived from IRS Historical Table 2 (actual Hawaii
filer counts by AGI class, 2018–2022), with a linear trend projected to 2027.

Approach mirrors TPC/ITEP microsimulation calibration methodology.

IRS Historic Table 2 — per-state XLSX files:
  URL: https://www.irs.gov/pub/irs-soi/{YY}in12hi.xlsx  (YY = 18..22)
  Hawaii is always state #12 alphabetically, file code "hi".

  XLSX layout (wide format, one state per file):
    Row "Number of returns":
      Col 0 : row label
      Col 1 : All returns total
      Col 2 : Under $1          ← "no positive AGI"
      Col 3 : $1 under $10,000
      Col 4 : $10,000 under $25,000
      Col 5 : $25,000 under $50,000
      Col 6 : $50,000 under $75,000
      Col 7 : $75,000 under $100,000
      Col 8 : $100,000 under $200,000
      Col 9 : $200,000 under $500,000
      Col 10: $500,000 under $1,000,000
      Col 11: $1,000,000 or more

  Our AGI_STUB mapping (IRS cols → internal stub):
    Stub 1 ($0–$25k)      = col 3 + col 4  ($1–$10k + $10k–$25k; col 2 excluded)
    Stub 2 ($25k–$50k)    = col 5
    Stub 3 ($50k–$75k)    = col 6
    Stub 4 ($75k–$100k)   = col 7
    Stub 5 ($100k–$200k)  = col 8
    Stub 6 ($200k–$500k)  = col 9
    Stub 7 ($500k–$1M)    = col 10
    Stub 8 ($1M+)         = col 11  ← excluded from reweighting

  Note: the "Under $1" column (negative/zero AGI) is intentionally excluded from
  stub 1 — those filers have AGI ≤ 0 and fall outside our model's AGI-positive
  brackets.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import requests

    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

try:
    import openpyxl  # noqa: F401

    _HAS_OPENPYXL = True
except ImportError:
    _HAS_OPENPYXL = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# IRS Table 2 URL — Hawaii is always "in12hi" across all available years
_IRS_URL_PATTERN = "https://www.irs.gov/pub/irs-soi/{yy}in12hi.xlsx"

# XLSX column index (0-based) → (lo, hi) bracket boundaries in dollars
# Col 0 is the row label; col 1 is "All returns" total.
# Cols 2–11 are AGI size classes in the Table 2 wide format.
_COL_TO_BRACKET: dict[int, tuple[float, float]] = {
    2: (float("-inf"), 1),        # "Under $1" — excluded (zero/negative AGI)
    3: (1, 10_000),               # $1 under $10,000
    4: (10_000, 25_000),          # $10,000 under $25,000
    5: (25_000, 50_000),          # $25,000 under $50,000
    6: (50_000, 75_000),          # $50,000 under $75,000
    7: (75_000, 100_000),         # $75,000 under $100,000
    8: (100_000, 200_000),        # $100,000 under $200,000
    9: (200_000, 500_000),        # $200,000 under $500,000
    10: (500_000, 1_000_000),     # $500,000 under $1,000,000
    11: (1_000_000, float("inf")),# $1,000,000 or more
}

# Map IRS xlsx column indices → internal AGI_STUB numbers.
# Stub 1 spans cols 3+4 combined (aggregated below).
_COL_TO_STUB: dict[int, int] = {
    # col 2: "Under $1" — intentionally excluded from stub 1
    5:  2,
    6:  3,
    7:  4,
    8:  5,
    9:  6,
    10: 7,
    11: 8,
}
_STUB1_COLS = [3, 4]  # $1–$10k + $10k–$25k → stub 1 ($0–$25k)

# Internal bracket definitions — what our model's AGI field uses
STUB_TO_BRACKET: dict[int, tuple[float, float]] = {
    1: (0, 25_000),
    2: (25_000, 50_000),
    3: (50_000, 75_000),
    4: (75_000, 100_000),
    5: (100_000, 200_000),
    6: (200_000, 500_000),
    7: (500_000, 1_000_000),
    8: (1_000_000, float("inf")),  # excluded from reweighting
}

# Stubs reweighted (sub-$1M; Pareto synthesis handles $1M+)
REWEIGHT_STUBS = [1, 2, 3, 4, 5, 6, 7]

# Guardrails
PROJECTION_FLOOR_FACTOR = 0.80   # projected value must be ≥ 80% of 2022 baseline
PROJECTION_CEIL_FACTOR  = 1.50   # projected value must be ≤ 150% of 2022 baseline
SCALE_MIN = 0.5                  # per-bracket weight scale floor
SCALE_MAX = 2.0                  # per-bracket weight scale ceiling


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class IrsHistoric2Loader:
    """Downloads and caches IRS Historic Table 2 Hawaii XLSX files.

    Cached to ``cache_dir/irs_hist2_{YY}.xlsx``.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/external/irs_hist2")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, year: int) -> Path:
        yy = str(year)[2:]
        return self.cache_dir / f"irs_hist2_{yy}.xlsx"

    def _fetch_bytes(self, year: int) -> Optional[bytes]:
        """Return raw XLSX bytes for *year*, using cache when available."""
        cache_path = self._cache_path(year)

        if cache_path.exists():
            logger.info(f"  {year}: loading from cache ({cache_path.name})")
            return cache_path.read_bytes()

        if not _HAS_REQUESTS:
            logger.warning("  'requests' not installed — cannot download IRS Table 2")
            return None

        yy = str(year)[2:]
        url = _IRS_URL_PATTERN.format(yy=yy)
        logger.info(f"  {year}: downloading {url}")
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            cache_path.write_bytes(resp.content)
            logger.info(f"  {year}: cached to {cache_path.name}")
            return resp.content
        except Exception as exc:
            logger.warning(f"  {year}: download failed — {exc}")
            return None

    @staticmethod
    def _parse_xlsx(raw_bytes: bytes) -> Optional[dict[int, float]]:
        """Parse XLSX bytes into {agi_stub: n_returns}.

        Returns None if parsing fails or "Number of returns" row is not found.
        """
        if not _HAS_OPENPYXL:
            logger.warning("  openpyxl not installed — cannot parse XLSX")
            return None

        try:
            import openpyxl

            wb = openpyxl.load_workbook(io.BytesIO(raw_bytes), data_only=True, read_only=True)
            ws = wb.active

            for row in ws.iter_rows(values_only=True):
                label = str(row[0]).strip() if row[0] is not None else ""
                if "number of returns" in label.lower() and "single" not in label.lower():
                    # Found the target row
                    stub_counts: dict[int, float] = {}

                    # Stub 1: aggregate cols 3 and 4
                    s1_total = 0.0
                    for c in _STUB1_COLS:
                        val = row[c] if c < len(row) else None
                        if val is not None:
                            try:
                                s1_total += float(val)
                            except (TypeError, ValueError):
                                pass
                    stub_counts[1] = s1_total

                    # Remaining stubs: one col each
                    for col_idx, stub in _COL_TO_STUB.items():
                        val = row[col_idx] if col_idx < len(row) else None
                        if val is not None:
                            try:
                                stub_counts[stub] = float(val)
                            except (TypeError, ValueError):
                                pass

                    wb.close()
                    return stub_counts

            wb.close()
            logger.warning("  'Number of returns' row not found in XLSX")
            return None

        except Exception as exc:
            logger.warning(f"  XLSX parse error: {exc}")
            return None

    def load(self, years: list[int] | None = None) -> pd.DataFrame:
        """Return DataFrame: index=year, columns=AGI_STUB (int 1–8), values=filer count.

        Missing years are silently omitted.
        """
        if years is None:
            years = [2018, 2019, 2020, 2021, 2022]

        logger.info("Loading IRS Historic Table 2 for Hawaii ...")
        rows: dict[int, dict[int, float]] = {}

        for year in years:
            raw = self._fetch_bytes(year)
            if raw is None:
                logger.warning(f"  {year}: skipped (no data)")
                continue

            parsed = self._parse_xlsx(raw)
            if parsed is None:
                logger.warning(f"  {year}: parse failed — skipping")
                continue

            rows[year] = parsed
            total = sum(v for k, v in parsed.items() if k in REWEIGHT_STUBS)
            logger.info(
                f"  {year}: loaded {len(parsed)} stubs, "
                f"sub-$1M total={total:,.0f}, "
                f"$1M+={parsed.get(8, 0):,.0f}"
            )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index.name = "year"
        df = df.sort_index()
        df.columns = [int(c) for c in df.columns]
        df = df[sorted(df.columns)]
        return df


# ---------------------------------------------------------------------------
# Projector
# ---------------------------------------------------------------------------


class BracketTargetProjector:
    """Fits a per-bracket linear trend and projects to a target year.

    Guardrails clamp the projection to [80%, 150%] of the 2022 baseline to
    prevent COVID-era dips or one-off spikes from distorting the extrapolation.
    """

    def __init__(self, historical: pd.DataFrame) -> None:
        self.historical = historical

    def project(self, target_year: int = 2027) -> dict[int, float]:
        """Return {agi_stub: projected_filer_count} for *target_year*."""
        if self.historical.empty:
            logger.warning("BracketTargetProjector: no historical data — empty targets")
            return {}

        years = self.historical.index.values.astype(float)
        baseline_year = int(years.max())
        targets: dict[int, float] = {}

        logger.info(
            f"Projecting bracket targets → {target_year} "
            f"(baseline={baseline_year}, fit over {[int(y) for y in years]})"
        )
        hdr = f"  {'Stub':<5} {'Bracket':<22} {'Baseline':>10} {'Trend':>12} {'Clamped':>10}"
        logger.info(hdr)
        logger.info("  " + "-" * 63)

        for stub in sorted(self.historical.columns):
            series = self.historical[stub].dropna()
            if len(series) < 2:
                logger.warning(f"  Stub {stub}: only {len(series)} data point(s) — skipping")
                continue

            x = series.index.values.astype(float)
            y = series.values.astype(float)

            slope, intercept = np.polyfit(x, y, 1)
            trend_value = slope * target_year + intercept

            baseline = float(series.loc[baseline_year]) if baseline_year in series.index else float(series.iloc[-1])
            floor = PROJECTION_FLOOR_FACTOR * baseline
            ceil  = PROJECTION_CEIL_FACTOR  * baseline
            clamped = float(np.clip(trend_value, floor, ceil))
            clamped = max(0.0, clamped)

            lo, hi = STUB_TO_BRACKET.get(stub, (0, 0))
            hi_str = f"${hi/1e3:.0f}k" if hi < 1e9 else "$1M+"
            lo_str = f"${lo/1e3:.0f}k"
            bracket_str = f"{lo_str}–{hi_str}"
            clamp_note = " [clamped]" if abs(clamped - trend_value) > 1 else ""

            logger.info(
                f"  {stub:<5} {bracket_str:<22} {baseline:>10,.0f} "
                f"{trend_value:>12,.0f} {clamped:>10,.0f}{clamp_note}"
            )
            targets[stub] = clamped

        return targets


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def apply_bracket_reweighting(
    tax_units: pd.DataFrame,
    target_year: int = 2027,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Reweight tax_units per AGI bracket to match IRS Historic Table 2 projections.

    Sub-$1M brackets (stubs 1–7) are reweighted; stub 8 ($1M+) is excluded
    because Pareto synthesis replaces those filers immediately after this step.

    Degrades gracefully: if IRS data cannot be loaded, returns tax_units unchanged.

    Parameters
    ----------
    tax_units   : post-income-growth DataFrame with 'agi' and 'weight' columns.
    target_year : year to project bracket targets to (default 2027).
    cache_dir   : directory for caching downloaded XLSX files.

    Returns
    -------
    DataFrame with adjusted 'weight' column (copy of input on failure).
    """
    logger.info("=" * 80)
    logger.info("STEP: BRACKET-LEVEL FILER COUNT REWEIGHTING (IRS Historic Table 2)")
    logger.info("=" * 80)

    result = tax_units.copy()

    try:
        # 1. Load IRS historical series
        loader = IrsHistoric2Loader(cache_dir=cache_dir)
        historical = loader.load()

        if historical.empty:
            logger.warning(
                "  IRS Historic Table 2: no data loaded — skipping bracket reweighting"
            )
            return result

        # 2. Log the historical series
        logger.info("\nHistorical Hawaii filer counts by AGI bracket (IRS Table 2):")
        logger.info(historical.to_string())
        logger.info("")

        # 3. Project targets
        projector = BracketTargetProjector(historical)
        targets = projector.project(target_year=target_year)

        if not targets:
            logger.warning("  No targets projected — skipping reweighting")
            return result

        # 4. Apply per-bracket weight scaling
        agi     = result["agi"].values.astype(float)
        weights = result["weight"].values.copy().astype(float)

        logger.info(
            f"\nApplying bracket reweighting "
            f"(scale clamped to [{SCALE_MIN}×, {SCALE_MAX}×]):"
        )
        hdr = (
            f"  {'Stub':<5} {'Bracket':<22} {'Before':>12} "
            f"{'Target':>12} {'Scale':>8} {'After':>12}"
        )
        logger.info(hdr)
        logger.info("  " + "-" * 75)

        for stub in REWEIGHT_STUBS:
            if stub not in targets:
                continue
            lo, hi = STUB_TO_BRACKET[stub]
            target_count = targets[stub]

            mask = (agi >= lo) & (agi < hi)
            actual_weighted = weights[mask].sum()

            if actual_weighted <= 0:
                lo_k = f"${lo/1e3:.0f}k" if lo > 0 else "$0"
                hi_k = f"${hi/1e3:.0f}k"
                logger.warning(
                    f"  Stub {stub} [{lo_k}–{hi_k}): no filers — skipping"
                )
                continue
            if target_count <= 0:
                logger.warning(f"  Stub {stub}: target={target_count:.0f} — skipping")
                continue

            raw_scale = target_count / actual_weighted
            scale = float(np.clip(raw_scale, SCALE_MIN, SCALE_MAX))
            weights[mask] *= scale

            lo_str = f"${lo/1e3:.0f}k" if lo > 0 else "$0"
            hi_str = f"${hi/1e3:.0f}k" if hi < 1e9 else "$1M+"
            bracket_str = f"{lo_str}–{hi_str}"
            clamp_note = " [clamped]" if abs(scale - raw_scale) > 0.001 else ""

            logger.info(
                f"  {stub:<5} {bracket_str:<22} {actual_weighted:>12,.0f} "
                f"{target_count:>12,.0f} {scale:>8.4f} {weights[mask].sum():>12,.0f}"
                + clamp_note
            )

        result["weight"] = weights

        before_total = tax_units["weight"].sum()
        after_total  = result["weight"].sum()
        logger.info(f"\n  Total weighted filers: {before_total:,.0f} → {after_total:,.0f}")
        logger.info("=" * 80)

    except Exception as exc:
        logger.warning(
            f"Bracket reweighting failed ({type(exc).__name__}: {exc}) — "
            "continuing without reweighting"
        )
        return tax_units

    return result
