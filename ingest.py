import os
import json
import time
import random
from typing import List
from tqdm import tqdm

import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pipeline import fetch_openalex_for_journals, process_paper_by_meta

# ---------------- CONFIG ----------------
SHEET_ID_ENV = "SHEET_ID"
GCP_SA_JSON_ENV = "GCP_SA_JSON"
WORKSHEET_NAME = None  
ID_HEADER = "paper_id" 

FIELDNAMES = [
    "paper_id", "title", "authors", "year", "venue",
    "abstract_full", "keywords", "abstract_summary", "paper_type", "link"
]

TAB_NAMES = ["Main", "Secondary", "Entrepreneurship / startup", "Career / labor market"]
KEYWORD_TAB_RULES = {
    "Entrepreneurship / startup": ["entrepreneurship", "startup"],
    "Career / labor market": ["career", "labor market"],
}

append_ORDER = ["Entrepreneurship / startup", "Career / labor market", "Main", "Secondary"]
PAPERS_PER_append = 50

def append_all_batches(worksheets, batch_rows):
    for name in append_ORDER:
        rows = batch_rows.get(name)
        if not rows:
            continue
        for chunk in chunk_rows_for_append(rows, max_rows=DEFAULT_APPEND_BATCH_SIZE, max_bytes=MAX_PAYLOAD_BYTES):
            append_rows_with_retries(worksheets[name], chunk)
            print(f"[append] appended {len(chunk)} rows to '{name}'")
        batch_rows[name] = []

def matches_keyword_group(keywords_str: str, terms: List[str]) -> bool:
    if not keywords_str:
        return False
    text = keywords_str.lower()
    return any(t.lower() in text for t in terms)

# Safety and tuning parameters
DEFAULT_APPEND_BATCH_SIZE = 50 
APPEND_RETRIES = 5
INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 30.0

# Google Sheets per-minute per-user write quota
APPEND_CALLS_PER_MINUTE = 50

# Max recommended payload (2MB doc)
MAX_PAYLOAD_BYTES = 1_800_000

# Dynamic tuning sample size: how many processed rows to sample to estimate avg row size.
DYNAMIC_SAMPLE_SIZE = 8

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

def open_all_worksheets(client, sheet_id: str):
    sh = client.open_by_key(sheet_id)
    return {name: sh.worksheet(name) for name in TAB_NAMES}


def read_existing_ids_from_sheet(ws, id_header=ID_HEADER) -> set:
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

#Ensure at least min_interval_seconds has passed since last append.
def _ensure_rate_limit(min_interval_seconds: float):
    global _last_append_time
    now = time.time()
    elapsed = now - _last_append_time
    if elapsed < min_interval_seconds:
        to_sleep = min_interval_seconds - elapsed
        jitter = min(0.2, to_sleep * 0.05)
        sleep_time = to_sleep + (random.random() * jitter)
        print(f"[throttle] sleeping {sleep_time:.2f}s to respect per-minute quota")
        time.sleep(sleep_time)

# Append rows with exponential backoff and quota-aware pacing.
def append_rows_with_retries(ws, rows: List[List[str]], retries=APPEND_RETRIES):
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

# Main incremental flow
def incremental_run_sheet_only():
    sheet_id = os.environ.get(SHEET_ID_ENV)
    if not sheet_id:
        raise ValueError("SHEET_ID environment variable not set.")
    client = gs_client_from_env()
    worksheets = open_all_worksheets(client, sheet_id)
    print("Opened tabs:", list(worksheets.keys()))

    # 1) read existing ids, per tab
    existing_ids = {name: read_existing_ids_from_sheet(ws) for name, ws in worksheets.items()}
    for name, ids in existing_ids.items():
        print(f"Existing IDs in '{name}': {len(ids)}")

    # 2) fetch remote papers via pipeline fetcher (covers Main + Secondary journals)
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

    # 4) gate processing on the paper's primary tab (Main or Secondary)
    new_ids = [
        pid for pid, meta in fetched_by_id.items()
        if pid not in existing_ids[meta.get("journal_group", "Main")]
    ]
    print(f"New papers to process: {len(new_ids)}")
    if not new_ids:
        print("No new items found. Exiting.")
        return

    # 5) process new items and route into per-tab batches
    processed = 0
    batch_rows = {name: [] for name in TAB_NAMES}
    since_last_append = 0
    loop_start = time.time()

    for i, pid in enumerate(tqdm(new_ids, desc="Processing new papers"), start=1):
        meta = fetched_by_id[pid]
        try:
            paper_start = time.time()
            out = process_paper_by_meta(meta)
            paper_elapsed = time.time() - paper_start
        except Exception as e:
            print(f"[WARN] Error processing {pid}: {e}")
            continue

        # timing readout: every 10 papers, print avg + projected total
        if i % 10 == 0:
            avg = (time.time() - loop_start) / i
            remaining = (len(new_ids) - i) * avg
            print(f"[timing] {i}/{len(new_ids)} | last paper {paper_elapsed:.1f}s | "
                  f"avg {avg:.1f}s/paper | est. {remaining/3600:.1f}h remaining")

        # ensure normalized paper_id is used
        out["paper_id"] = pid

        # create ordered row
        row = [str(out.get(col, "")) for col in FIELDNAMES]

        # primary tab: Main or Secondary, based on which journal it came from
        primary_tab = meta.get("journal_group", "Main")
        batch_rows[primary_tab].append(row)

        # keyword tabs: additive, not exclusive — a paper can also land here
        kw_str = out.get("keywords", "")
        for tab_name, terms in KEYWORD_TAB_RULES.items():
            if matches_keyword_group(kw_str, terms) and pid not in existing_ids[tab_name]:
                batch_rows[tab_name].append(row)

        processed += 1
        since_last_append += 1

        # Append all tabs together, keyword tabs first, every PAPERS_PER_append papers
        if since_last_append >= PAPERS_PER_append:
            append_all_batches(worksheets, batch_rows)
            since_last_append = 0

    # 6) final append same grouped order
    append_all_batches(worksheets, batch_rows)

    print(f"Done. Processed {processed} new papers.")

if __name__ == "__main__":
    start = time.time()
    incremental_run_sheet_only()

    print("Elapsed:", time.time() - start)
