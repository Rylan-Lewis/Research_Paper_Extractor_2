# incremental.py
"""
Sheet-only incremental ingestion driver (patched for Sheets quotas + dynamic tuning).

Usage:
  - Ensure my_pipeline.py implements:
      fetch_openalex_for_journals() -> List[dict]
      process_paper_by_meta(meta) -> dict (must include keys used in FIELDNAMES)
  - Set env vars SHEET_ID and GCP_SA_JSON (the service account JSON contents).
  - Run: python incremental.py
"""

import os
import json
import time
import random
from typing import List
from tqdm import tqdm

import gspread
from oauth2client.service_account import ServiceAccountCredentials

# import pipeline functions (from your pipeline implementation)
from pipeline import fetch_openalex_for_journals, process_paper_by_meta

# ---------------- CONFIG ----------------
SHEET_ID_ENV = "SHEET_ID"
GCP_SA_JSON_ENV = "GCP_SA_JSON"
WORKSHEET_NAME = None   # None -> first worksheet/tab
ID_HEADER = "paper_id"  # header name of the ID column in the sheet

# Default output columns (must match keys returned by process_paper_by_meta)
FIELDNAMES = [
    "paper_id", "title", "authors", "year", "venue",
    "abstract_full", "keywords", "abstract_summary", "paper_type", "link"
]

# Safety and tuning parameters
DEFAULT_APPEND_BATCH_SIZE = 50     # soft cap for rows per append call (starting point)
APPEND_RETRIES = 5
INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 30.0

# Sheets quota safety
# Google Sheets per-minute per-user write quota ~60. Leave margin.
APPEND_CALLS_PER_MINUTE = 50

# Max recommended payload (2MB doc); keep margin.
MAX_PAYLOAD_BYTES = 1_800_000

# Dynamic tuning sample size: how many processed rows to sample to estimate avg row size.
DYNAMIC_SAMPLE_SIZE = 8

# ---------------- Helpers ----------------
_last_append_time = 0.0

def normalize_openalex_id(raw_id: str) -> str | None:
    if not raw_id:
        return None
    raw = str(raw_id).strip()
    if raw.startswith("https://openalex.org/") or raw.startswith("http://openalex.org/"):
        return raw.split("/")[-1]
    if raw.startswith("openalex:"):
        return raw.split(":", 1)[1]
    return raw

def gs_client_from_env(sa_env=GCP_SA_JSON_ENV):
    sa_json_str = os.environ.get(sa_env)
    if not sa_json_str:
        raise ValueError(f"{sa_env} not found in environment (set GCP_SA_JSON secret).")
    sa_json = json.loads(sa_json_str)
    scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(sa_json, scope)
    client = gspread.authorize(creds)
    return client

def open_worksheet(client, sheet_id: str, worksheet_name=None):
    sh = client.open_by_key(sheet_id)
    if worksheet_name:
        ws = sh.worksheet(worksheet_name)
    else:
        ws = sh.get_worksheet(0)
    return ws

def read_existing_ids_from_sheet(ws, id_header=ID_HEADER) -> set:
    # Try to read header row to find ID column; otherwise assume column A
    try:
        header = ws.row_values(1)
    except Exception:
        header = []
    col_index = None
    for i, h in enumerate(header):
        if h.strip().lower() == id_header.strip().lower():
            col_index = i + 1
            break
    if col_index is None:
        col_index = 1
    col_vals = ws.col_values(col_index)
    # drop header if present
    if col_vals and header and col_vals[0].strip().lower() == id_header.strip().lower():
        col_vals = col_vals[1:]
    existing = set()
    for v in col_vals:
        n = normalize_openalex_id(v)
        if n:
            existing.add(n)
    return existing

# ----- payload estimation helpers -----
def estimate_row_bytes(row: List[str]) -> int:
    # approximate utf-8 length
    s = "\t".join([("" if v is None else str(v)) for v in row]) + "\n"
    return len(s.encode("utf-8"))

def estimate_rows_bytes(rows: List[List[str]]) -> int:
    return sum(estimate_row_bytes(r) for r in rows)

def chunk_rows_for_append(rows: List[List[str]], max_rows=DEFAULT_APPEND_BATCH_SIZE, max_bytes=MAX_PAYLOAD_BYTES):
    """
    Yield chunks that respect both max_rows and max_bytes.
    """
    buf = []
    buf_bytes = 0
    for row in rows:
        rbytes = estimate_row_bytes(row)
        # if adding this row would exceed either limit, yield current buf first
        if buf and (len(buf) >= max_rows or (buf_bytes + rbytes) > max_bytes):
            yield buf
            buf = []
            buf_bytes = 0
        buf.append(row)
        buf_bytes += rbytes
        if len(buf) >= max_rows:
            yield buf
            buf = []
            buf_bytes = 0
    if buf:
        yield buf

def _ensure_rate_limit(min_interval_seconds: float):
    """
    Ensure at least min_interval_seconds has passed since last append.
    """
    global _last_append_time
    now = time.time()
    elapsed = now - _last_append_time
    if elapsed < min_interval_seconds:
        to_sleep = min_interval_seconds - elapsed
        jitter = min(0.2, to_sleep * 0.05)
        sleep_time = to_sleep + (random.random() * jitter)
        print(f"[throttle] sleeping {sleep_time:.2f}s to respect per-minute quota")
        time.sleep(sleep_time)

def append_rows_with_retries(ws, rows: List[List[str]], retries=APPEND_RETRIES):
    """
    Append rows with exponential backoff and quota-aware pacing.
    """
    global _last_append_time
    min_interval = 60.0 / float(APPEND_CALLS_PER_MINUTE)
    attempt = 0
    while True:
        try:
            _ensure_rate_limit(min_interval)
            ws.append_rows(rows, value_input_option="USER_ENTERED")
            _last_append_time = time.time()
            return
        except Exception as e:
            attempt += 1
            if attempt > retries:
                print(f"[ERROR] append_rows failed after {retries} attempts: {e}")
                raise
            backoff = min(INITIAL_BACKOFF * (2 ** (attempt - 1)) * (1 + random.random() * 0.1), MAX_BACKOFF)
            print(f"[WARN] append_rows error (attempt {attempt}/{retries}): {e}. Backing off {backoff:.1f}s")
            time.sleep(backoff)

# ---------------- Main incremental flow ----------------
def incremental_run_sheet_only():
    sheet_id = os.environ.get(SHEET_ID_ENV)
    if not sheet_id:
        raise ValueError("SHEET_ID environment variable not set.")
    client = gs_client_from_env()
    ws = open_worksheet(client, sheet_id, WORKSHEET_NAME)
    print("Opened worksheet:", ws.title)

    # 1) read existing ids
    existing_ids = read_existing_ids_from_sheet(ws)
    print(f"Existing IDs in sheet: {len(existing_ids)}")

    # 2) fetch remote papers via pipeline fetcher
    fetched = fetch_openalex_for_journals()
    print(f"Fetched {len(fetched)} records from remote source.")

    # 3) normalize and index fetched by id
    fetched_by_id = {}
    for r in fetched:
        raw = r.get("paperId") or r.get("paper_id") or r.get("id") or r.get("openalex_id")
        pid = normalize_openalex_id(raw)
        if not pid:
            continue
        fetched_by_id[pid] = r

    print(f"Normalized fetched ids: {len(fetched_by_id)}")

    # 4) compute new ids
    new_ids = [pid for pid in fetched_by_id.keys() if pid not in existing_ids]
    print(f"New papers to process: {len(new_ids)}")
    if not new_ids:
        print("No new items found. Exiting.")
        return

    # 5) process new items and append in safe chunks
    processed = 0
    batch_rows = []
    sample_row_sizes = []
    dynamic_max_rows = DEFAULT_APPEND_BATCH_SIZE

    for pid in tqdm(new_ids, desc="Processing new papers"):
        meta = fetched_by_id[pid]
        try:
            out = process_paper_by_meta(meta)
        except Exception as e:
            print(f"[WARN] Error processing {pid}: {e}")
            continue

        # ensure normalized paper_id is used
        out["paper_id"] = pid

        # create ordered row
        row = [str(out.get(col, "")) for col in FIELDNAMES]
        batch_rows.append(row)
        processed += 1

        # collect sample sizes for dynamic tuning
        if len(sample_row_sizes) < DYNAMIC_SAMPLE_SIZE:
            sample_row_sizes.append(estimate_row_bytes(row))
            if len(sample_row_sizes) == DYNAMIC_SAMPLE_SIZE:
                avg_row_bytes = max(1, sum(sample_row_sizes) / len(sample_row_sizes))
                # compute rows that would fit under MAX_PAYLOAD_BYTES
                rows_by_size = max(1, int(MAX_PAYLOAD_BYTES // avg_row_bytes))
                # cap rows to DEFAULT_APPEND_BATCH_SIZE to avoid too-large call counts
                dynamic_max_rows = min(DEFAULT_APPEND_BATCH_SIZE, rows_by_size)
                print(f"[tune] avg_row_bytes={avg_row_bytes:.1f}, dynamic_max_rows={dynamic_max_rows}")

        # flush when we reach the dynamic capacity (or hard cap if tuning not ready)
        effective_cap = dynamic_max_rows if len(sample_row_sizes) >= DYNAMIC_SAMPLE_SIZE else DEFAULT_APPEND_BATCH_SIZE
        if len(batch_rows) >= effective_cap:
            # chunk defensively (this will respect bytes and max_rows)
            for chunk in chunk_rows_for_append(batch_rows, max_rows=effective_cap, max_bytes=MAX_PAYLOAD_BYTES):
                append_rows_with_retries(ws, chunk)
                print(f"[append] appended {len(chunk)} rows")
            batch_rows = []

    # final flush
    if batch_rows:
        for chunk in chunk_rows_for_append(batch_rows, max_rows=dynamic_max_rows, max_bytes=MAX_PAYLOAD_BYTES):
            append_rows_with_retries(ws, chunk)
            print(f"[append] appended {len(chunk)} rows (final)")

    print(f"Done. Processed and appended {processed} new papers.")

if __name__ == "__main__":
    start = time.time()
    incremental_run_sheet_only()

    print("Elapsed:", time.time() - start)
