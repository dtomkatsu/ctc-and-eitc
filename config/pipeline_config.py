"""Pipeline configuration for automated data fetching."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SourceConfig:
    """Configuration for a single data source."""
    name: str
    enabled: bool = True
    freshness_days: int = 180          # Skip if data newer than this many days
    typical_release_month: Optional[int] = None  # Month new data usually appears
    requires_api_key: Optional[str] = None       # Env var name for API key


@dataclass
class PipelineConfig:
    """Master pipeline configuration."""
    sources: dict[str, SourceConfig] = field(default_factory=dict)
    parallel_downloads: bool = True
    max_workers: int = 3
    auto_regenerate: bool = False   # If True, rebuild tax units after new data
    manifest_path: Path = field(default_factory=lambda: Path("data/pipeline_manifest.json"))
    log_dir: Path = field(default_factory=lambda: Path("logs/pipeline"))

    @classmethod
    def default(cls) -> "PipelineConfig":
        """Return default configuration with all sources."""
        return cls(
            sources={
                "census_pums": SourceConfig(
                    name="Census PUMS",
                    freshness_days=180,
                    typical_release_month=9,
                    requires_api_key="CENSUS_API_KEY",
                ),
                "acs_tables": SourceConfig(
                    name="ACS 1-Year Tables",
                    freshness_days=180,
                    typical_release_month=9,
                    requires_api_key="CENSUS_API_KEY",
                ),
                "bls_oes": SourceConfig(
                    name="BLS OES Wages",
                    freshness_days=180,
                    typical_release_month=4,
                ),
                "bea_state_income": SourceConfig(
                    name="BEA State Income",
                    freshness_days=90,
                    typical_release_month=9,
                    requires_api_key="BEA_API_KEY",
                ),
                "fred": SourceConfig(
                    name="FRED Economic Data",
                    freshness_days=30,
                    typical_release_month=None,  # Continuous updates
                    requires_api_key="FRED_API_KEY",
                ),
                "bls_cpi": SourceConfig(
                    name="BLS CPI-U Honolulu",
                    freshness_days=30,
                    typical_release_month=None,  # Semi-annual releases
                    requires_api_key=None,  # BLS API v1 requires no key
                ),
            }
        )
