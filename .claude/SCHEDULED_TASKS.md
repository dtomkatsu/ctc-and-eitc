# Scheduled Tasks & Cron Jobs

This project uses automated scheduling for routine data pipeline work. All scheduled tasks are documented here so OpenClaw and other tooling can be aware of what's running.

## Task Registry

See `.claude/scheduled_tasks_registry.json` for machine-readable task definitions.

Human-readable summary below.

---

## Active Scheduled Tasks

### 1. Data Pipeline (Monthly)

**What it does:**
- Fetches fresh data from Census PUMS, ACS tables, BLS OES, and BEA state income
- Harmonizes BLS data into timeseries
- Auto-commits manifest + updated files to git
- Sends macOS notification when new data available

**Schedule:**
- **When:** 1st of each month at 9:00 AM
- **Frequency:** Monthly
- **Type:** launchd agent

**Launcher:**
- **Plist:** `~/Library/LaunchAgents/com.ctc-eitc.datapipeline.plist`
- **Command:** `python scripts/data_prep/pipeline.py --scheduled`
- **Working Directory:** `/Users/devinthomas/ctc-and-eitc`

**Data Sources:**
- Census PUMS — Person/household microdata (API)
- ACS 1-Year Tables — Demographic aggregates (API)
- BLS OES — Occupational wages (HTTP download)
- BEA State Income — State income by line code (API)

**Outputs:**
- `data/raw/pums/*.csv`
- `data/raw/acs_1yr/**/*.csv`
- `data/external/bls_oes/*.xlsx`
- `data/raw/bea_hawaii_sainc5n.csv`
- `data/pipeline_manifest.json` (updated with timestamps)

**Git Integration:**
- Auto-commits if any data was fetched
- Commit message includes timestamp + list of updated sources
- Example: `Auto-update data pipeline — 2026-04-02 13:53`

**Manual Sources Monitored:**
- DOTAX SOI (Hawaii tax data) — request from DOTAX
- IRS SOI ZIP Code — download from IRS website
- CEX (Consumer Expenditure Survey) — download from BLS
- DOTAX Collections — manual monthly updates

**Logs:**
- Daily log: `logs/pipeline/{YYYY-MM-DD}.log`
- Launchd stdout: `logs/pipeline/launchd_stdout.log`
- Launchd stderr: `logs/pipeline/launchd_stderr.log`
- System logs: `~/Library/Logs/com.ctc-eitc.datapipeline.log` (if created by launchd)

**Documentation:**
- Full setup & usage guide: `PIPELINE_SETUP.md`
- Pipeline code: `scripts/data_prep/pipeline.py`
- Configuration: `config/pipeline_config.py`

---

## Freshness Thresholds

The pipeline checks data freshness before downloading. These thresholds determine whether to re-fetch:

| Source | Threshold | Typical Release |
|--------|-----------|-----------------|
| PUMS | 180 days | September |
| ACS | 180 days | September |
| BLS OES | 180 days | March/April |
| BEA | 90 days | September |

---

## Manual Workflow

After the scheduled run:

1. **Check notification** (if sent) → New data is available
2. **Verify logs** → `tail logs/pipeline/$(date +%Y-%m-%d).log`
3. **Review manifest** → `cat data/pipeline_manifest.json`
4. **Regenerate tax units** (when ready) → `python scripts/data_prep/pipeline.py --regenerate`

---

## OpenClaw Awareness

This registry allows OpenClaw to:
- **Discover** all automated work in the project
- **Track** when scheduled tasks should run
- **Monitor** logs and outcomes
- **Coordinate** with other scheduled work (e.g., OpenClaw heartbeat)
- **Alert** if critical tasks fail

To integrate with OpenClaw:
1. Read `.claude/scheduled_tasks_registry.json` at startup
2. Monitor `logs/pipeline/` directory for new entries
3. Check git log for auto-commits from the pipeline
4. Optionally create session notes after each scheduled run

---

## How to Verify Launchd Is Running

```bash
# List all scheduled tasks in OpenClaw agents
launchctl list | grep ctc-eitc

# Load the agent (if not already loaded)
launchctl load ~/Library/LaunchAgents/com.ctc-eitc.datapipeline.plist

# Unload the agent
launchctl unload ~/Library/LaunchAgents/com.ctc-eitc.datapipeline.plist

# View launchd logs for this job
log stream --predicate 'process == "com.ctc-eitc.datapipeline"'
```

---

## Future Tasks

This registry is extensible. Future scheduled tasks (e.g., auto-regenerate tax units, weekly validation checks) can be added by:

1. Creating the launchd plist in `~/Library/LaunchAgents/`
2. Adding an entry to `.claude/scheduled_tasks_registry.json`
3. Documenting in this file
