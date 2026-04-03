#!/Users/devinthomas/ctc-and-eitc/.venv/bin/python3
"""
Download and process "State Tax Collections and Distribution" XLSX files from Hawaii Department of Taxation.
Only downloads files for months we don't already have.
Source: https://tax.hawaii.gov/stats/a5_3txcolrpt/
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Import pandas for CSV handling
try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not installed.")
    sys.exit(1)

def get_existing_months(output_path):
    """Get set of months already in CSV."""
    if not output_path.exists():
        print(f"[{datetime.now()}] No existing CSV found")
        return set()

    try:
        df = pd.read_csv(output_path)
        if 'Year Month' in df.columns:
            existing_months = set(df['Year Month'].unique())
            print(f"[{datetime.now()}] CSV has data for: {sorted(existing_months)}")
            return existing_months
        else:
            print(f"[{datetime.now()}] CSV has unexpected format")
            return set()
    except Exception as e:
        print(f"[{datetime.now()}] Error reading CSV: {e}")
        return set()

def extract_month_from_filename(filename):
    """Extract month from filename like '202509collec.xlsx' -> 'Sep 2025'."""
    # Format: YYYYMM (e.g., 202509 = Sept 2025)
    if len(filename) >= 6 and filename[:6].isdigit():
        year_month_str = filename[:6]
        year = year_month_str[:4]
        month_num = int(year_month_str[4:6])

        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        if 1 <= month_num <= 12:
            month_name = months[month_num - 1]
            return f"{month_name} {year}"

    return None

def scrape_hawaii_tax_xlsx():
    """Download State Tax Collections and Distribution XLSX files for missing months."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: Playwright not installed.")
        return False

    url = "https://tax.hawaii.gov/stats/a5_3txcolrpt/"
    output_path = Path("/Users/devinthomas/ctc-and-eitc/data/raw/hawaii_tax_collections_looker_studio.csv")
    log_path = Path("/Users/devinthomas/ctc-and-eitc/logs/pipeline/dotax_scrape.log")

    # Get existing months in CSV
    existing_months = get_existing_months(output_path)

    # Load existing data to append to
    existing_df = None
    if output_path.exists():
        try:
            existing_df = pd.read_csv(output_path)
        except Exception as e:
            print(f"[{datetime.now()}] Error loading existing CSV: {e}")
            existing_df = None

    try:
        with sync_playwright() as p:
            print(f"[{datetime.now()}] Opening browser...")
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            print(f"[{datetime.now()}] Navigating to {url}")
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            time.sleep(5)

            print(f"[{datetime.now()}] Looking for 'State Tax Collections and Distribution' files...")

            rows = page.locator("tr").all()
            print(f"[{datetime.now()}] Inspecting {len(rows)} table rows...")

            xlsx_urls_to_download = []

            for i, row in enumerate(rows[:50]):
                try:
                    row_text = row.text_content()
                    if "State Tax Collections" in row_text or "Collections and Distribution" in row_text:
                        # Find XLSX link in this row
                        link = row.locator('a[href$=".xlsx"]').first
                        if link:
                            href = link.get_attribute("href")

                            # Handle relative URLs
                            if href.startswith("/"):
                                full_url = "https://tax.hawaii.gov" + href
                            elif not href.startswith("http"):
                                full_url = "https://tax.hawaii.gov/stats/a5_3txcolrpt/" + href
                            else:
                                full_url = href

                            # Extract month from filename
                            filename = href.split("/")[-1]
                            month = extract_month_from_filename(filename)

                            if month:
                                if month in existing_months:
                                    print(f"[{datetime.now()}] ✓ Already have {month}")
                                else:
                                    print(f"[{datetime.now()}] ✗ Missing {month} - will download")
                                    xlsx_urls_to_download.append((month, full_url, filename))
                            else:
                                print(f"[{datetime.now()}] ? Could not parse month from {filename}")
                except Exception as e:
                    pass

            if not xlsx_urls_to_download:
                print(f"[{datetime.now()}] No missing 'State Tax Collections and Distribution' files")
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, 'a') as f:
                    f.write(f"[{datetime.now()}] SKIPPED: No missing State Tax Collections files\n")
                browser.close()
                return True

            browser.close()

            # Download and process missing files
            import requests
            all_new_rows = []

            for month, xlsx_file_url, filename in xlsx_urls_to_download:
                print(f"\n[{datetime.now()}] Downloading {month}: {xlsx_file_url}")

                try:
                    response = requests.get(xlsx_file_url, timeout=30)
                    response.raise_for_status()

                    downloads_dir = Path.home() / "Downloads"
                    xlsx_file = downloads_dir / filename
                    xlsx_file.write_bytes(response.content)
                    print(f"[{datetime.now()}] Downloaded to: {xlsx_file}")

                    # Read and extract data
                    print(f"[{datetime.now()}] Extracting data from {month}...")
                    df_raw = pd.read_excel(xlsx_file, sheet_name=0, header=None)

                    # Extract data from rows (skip header rows 0-3)
                    month_rows = 0
                    for idx in range(4, len(df_raw)):
                        tax_type = df_raw.iloc[idx, 0]
                        amount = df_raw.iloc[idx, 2]

                        # Skip empty or section headers
                        if pd.isna(tax_type) or pd.isna(amount):
                            continue

                        tax_type_str = str(tax_type).strip()
                        if tax_type_str.endswith(':') or tax_type_str == '':
                            continue

                        try:
                            amount_float = float(amount)
                            all_new_rows.append({
                                'Year Month': month,
                                'Tax Type': tax_type_str,
                                'Amount': amount_float
                            })
                            month_rows += 1
                        except ValueError:
                            pass

                    print(f"[{datetime.now()}] Extracted {month_rows} rows for {month}")

                except Exception as e:
                    print(f"[{datetime.now()}] ✗ Error downloading {month}: {e}")
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(log_path, 'a') as f:
                        f.write(f"[{datetime.now()}] ERROR downloading {month}: {e}\n")

            if all_new_rows:
                # Combine with existing data
                output_path.parent.mkdir(parents=True, exist_ok=True)

                if existing_df is not None:
                    new_df = pd.DataFrame(all_new_rows)
                    result_df = pd.concat([existing_df, new_df], ignore_index=True)
                    print(f"[{datetime.now()}] Combined with existing data: {len(result_df)} total rows")
                else:
                    result_df = pd.DataFrame(all_new_rows)
                    print(f"[{datetime.now()}] Created new CSV: {len(result_df)} rows")

                result_df.to_csv(output_path, index=False)
                print(f"[{datetime.now()}] Saved to: {output_path}")

                log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, 'a') as f:
                    f.write(f"[{datetime.now()}] SUCCESS: Added {len(all_new_rows)} new rows\n")

                return True
            else:
                print(f"[{datetime.now()}] No new data extracted")
                return False

    except Exception as e:
        print(f"[{datetime.now()}] FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = scrape_hawaii_tax_xlsx()
    sys.exit(0 if success else 1)
