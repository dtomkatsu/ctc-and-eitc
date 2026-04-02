# Data Pipeline Setup Guide

## Overview

The automated data pipeline (`scripts/data_prep/pipeline.py`) pulls new data from 4 automatable sources:

1. **Census PUMS** — Person/household microdata (annual, September)
2. **ACS 1-Year Tables** — Demographic/economic aggregates (annual, September)
3. **BLS OES** — Occupational wage data (annual, March/April)
4. **BEA State Income** — State annual income by line code (annual, September)

Plus monitoring of 4 manual-only sources (DOTAX SOI, IRS SOI, CEX, DOTAX collections).

---

## One-Time Setup

### 1. Create `.env` file with API keys

```bash
cp .env.example .env
```

Add your API keys:
```
CENSUS_API_KEY=2104852dd7bfd83fbc9e320d650eb57decc11817
BEA_API_KEY=0F2130E5-5E89-44DA-8759-2E7A8AB4C67A
```

**Note:** `.env` is gitignored and will not be committed.

### 2. Install python-dotenv (if not already)

```bash
pip install python-dotenv
```

### 3. Launchd scheduler (already loaded)

A launchd agent at `~/Library/LaunchAgents/com.ctc-eitc.datapipeline.plist` automatically runs the pipeline on the **1st of each month at 9:00 AM**.

To verify it's loaded:
```bash
launchctl list | grep ctc-eitc
```

To manually trigger it:
```bash
launchctl start com.ctc-eitc.datapipeline
```

---

## Usage

### Check what data is stale

```bash
python scripts/data_prep/pipeline.py --check-only
```

Shows last fetch date for each source and whether it's fresh.

### Fetch only stale data

```bash
python scripts/data_prep/pipeline.py
```

Downloads any source not fetched in the past N days (see config/pipeline_config.py for thresholds).

### Force fresh download of a source

```bash
python scripts/data_prep/pipeline.py --sources bls_oes --force
```

### Fetch all + rebuild tax units

```bash
python scripts/data_prep/pipeline.py --regenerate
```

This runs the full pipeline through `scripts/regenerate_tax_units.py`.

### Check manual source status

```bash
python scripts/data_prep/pipeline.py --manual-status
```

Warns about missing or stale DOTAX SOI, IRS SOI, or CEX files.

---

## Freshness Thresholds

| Source | Days | Typical Release |
|--------|------|-----------------|
| PUMS | 180 | September |
| ACS | 180 | September |
| BLS OES | 180 | March/April |
| BEA | 90 | September |

If data is fresher than these thresholds, the pipeline skips re-download. Use `--force` to override.

---

## Manifest Tracking

The pipeline maintains `data/pipeline_manifest.json` which records:
- Last fetch timestamp per source
- Files updated in each run
- Overall last-run timestamp

This file is committed to git for reproducibility.

---

## Scheduled Run Behavior

When `--scheduled` mode runs (monthly via launchd):

1. Checks freshness of all sources
2. Downloads anything stale
3. Harmonizes BLS data if downloaded
4. **Auto-commits to git** if any data was updated (manifest + data files)
   - Commit message includes timestamp and list of updated sources
   - Only commits if there were actual changes (skips if all sources fresh)
5. **Sends macOS notification** if new data found (does NOT auto-regenerate tax units)
6. Logs to `logs/pipeline/{YYYY-MM-DD}.log`

Example auto-commit message:
```
Auto-update data pipeline — 2026-04-02 13:53

Sources updated:
  • bea_state_income

Automated commit from monthly pipeline run.
```

To regenerate tax units after new data arrives:
```bash
python scripts/data_prep/pipeline.py --regenerate
```

---

## API Keys

### Census API Key
- Register: https://api.census.gov/data/key_signup.html
- Used for: PUMS, ACS tables
- Already configured

### BEA API Key
- Register: https://apps.bea.gov/API/signup/
- Used for: State income (SAINC5N) data
- Already configured

If you need to refresh or update either key, edit `.env` and restart the pipeline.

---

## Logs

- **Scheduled run logs:** `logs/pipeline/{YYYY-MM-DD}.log`
- **Launchd logs:** `logs/pipeline/launchd_stdout.log`, `launchd_stderr.log`

View recent logs:
```bash
tail -100 logs/pipeline/launchd_stderr.log
```

---

## Next Steps

1. **Monitor the first scheduled run** — Should trigger on the 1st of next month at 9 AM
2. **Check logs** after the run to confirm data was fetched
3. **When notified of new data**, run `--regenerate` to rebuild tax units
4. **Test on demand** using `--check-only` to verify the system is working

---

## Troubleshooting

**"BEA_API_KEY not set"**
- Check `.env` file exists in project root
- Run `source .venv/bin/activate` before running pipeline

**"CENSUS_API_KEY not set"**
- Same as above — verify `.env` and venv activation

**Launchd agent not running**
- Check if it's loaded: `launchctl list | grep ctc-eitc`
- Reload if needed: `launchctl load ~/Library/LaunchAgents/com.ctc-eitc.datapipeline.plist`
- Verify plist syntax: `plutil -lint ~/Library/LaunchAgents/com.ctc-eitc.datapipeline.plist`

**Manual sources stale**
- DOTAX SOI — Request updated tables from DOTAX
- IRS SOI — Download from https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics-zip-code-data-soi
- CEX — Download from https://www.bls.gov/cex/pumd.htm
- DOTAX Collections — Update from DOTAX monthly reports
