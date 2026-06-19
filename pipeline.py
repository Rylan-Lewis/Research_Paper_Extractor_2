
# 1) Imports & model init
import time
import re
from datetime import datetime
from typing import List, Dict, Any

import requests
import pandas as pd
from tqdm import tqdm
from collections import Counter

# keyword & embedding libs
import yake
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# initialize embeddings model (keep as in your code)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Qwen model (per your original snippet). Note: heavy, may not run on CPU-only runner.
QWEN_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    QWEN_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 2) OpenAlex fetching code (adapted names to match driver expectations)
OPENALEX_WORKS = "https://api.openalex.org/works"
OPENALEX_PER_PAGE = 200
OPENALEX_RATE_SLEEP = 0.5

ALLOWED_JOURNALS = {
    "Management Science",
    "Administrative Science Quarterly",
    "Academy of Management Journal",
    "Strategic Management Journal",
    "Organization Science"
}

ISSN_MAP = {
    "Management Science": "0025-1909",
    "Administrative Science Quarterly": "0001-8392",
    "Academy of Management Journal": "0001-4273",
    "Strategic Management Journal": "0143-2095",
    "Organization Science": "1047-7039",
}

YEARS_BACK = 5

def _reconstruct_abstract_from_inverted_index(inv_idx: Dict[str, List[int]]) -> str:
    if not inv_idx:
        return ""
    try:
        max_pos = 0
        for token_positions in inv_idx.values():
            if token_positions:
                max_pos = max(max_pos, max(token_positions))
        words = [""] * (max_pos + 1)
        for token, positions in inv_idx.items():
            for pos in positions:
                if 0 <= pos <= max_pos:
                    words[pos] = token
        abstract = " ".join([w for w in words if w]).strip()
        return abstract
    except Exception:
        return ""

def _openalex_fetch_page(filter_str: str, per_page: int = OPENALEX_PER_PAGE, cursor: str = "*") -> Dict[str, Any]:
    params = {
        "filter": filter_str,
        "per_page": per_page,
        "cursor": cursor
    }
    resp = requests.get(OPENALEX_WORKS, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()

def fetch_openalex_for_journal(journal_name: str, years_back: int = YEARS_BACK, per_page: int = OPENALEX_PER_PAGE) -> List[Dict[str, Any]]:
    now = datetime.now().year
    start_year = now - years_back
    date_filter = f"from_publication_date:{start_year}-01-01"
    issn = ISSN_MAP.get(journal_name)
    if issn:
        venue_filter = f"primary_location.source.issn:{issn}"
    else:
        venue_filter = f'primary_location.source.display_name.search:"{journal_name}"'
    filter_str = f"{venue_filter},{date_filter}"

    all_papers = []
    cursor = "*"
    page_count = 0
    while True:
        try:
            data = _openalex_fetch_page(filter_str, per_page=per_page, cursor=cursor)
        except Exception as e:
            print(f"[OpenAlex] error fetching {journal_name} (cursor={cursor}): {e}")
            break
        results = data.get("results", [])
        if not results:
            break
        for item in results:
            paper = {}
            # OpenAlex 'id' is like "https://openalex.org/W12345"
            paper["paperId"] = item.get("id")
            paper["title"] = item.get("title") or ""
            authors = []
            for a in item.get("authorships", []) or []:
                au = a.get("author") or {}
                name = au.get("display_name") or au.get("id") or ""
                if name:
                    authors.append({"name": name})
            paper["authors"] = authors
            pub_date = item.get("publication_date")
            if pub_date:
                try:
                    paper["year"] = int(pub_date.split("-")[0])
                except Exception:
                    paper["year"] = item.get("publication_year") or None
            else:
                paper["year"] = item.get("publication_year") or None
            abstract = item.get("abstract") or ""
            if not abstract:
                inv = item.get("abstract_inverted_index")
                abstract = _reconstruct_abstract_from_inverted_index(inv) if inv else ""
            paper["abstract"] = abstract or ""
            host = item.get("host_venue") or {}
            if host.get("display_name"):
                paper["venue"] = host.get("display_name")
            else:
                pl = item.get("primary_location") or {}
                src = pl.get("source") or {}
                paper["venue"] = src.get("display_name") or journal_name
            if item.get("doi"):
                paper["doi"] = item.get("doi")
            all_papers.append(paper)
        page_count += 1
        meta = data.get("meta", {})
        next_cursor = meta.get("next_cursor")
        if not next_cursor:
            break
        cursor = next_cursor
        time.sleep(OPENALEX_RATE_SLEEP)
    print(f"[OpenAlex] {journal_name}: fetched {len(all_papers)} papers (pages={page_count})")
    return all_papers

def fetch_openalex_for_journals(years_back: int = YEARS_BACK) -> List[Dict[str, Any]]:
    overall = {}
    print(f"Fetching papers for journals (last {years_back} years) via OpenAlex across {len(ALLOWED_JOURNALS)} journals...")
    for journal in ALLOWED_JOURNALS:
        print(f" -> querying journal (OpenAlex): {journal}")
        try:
            papers = fetch_openalex_for_journal(journal, years_back=years_back, per_page=OPENALEX_PER_PAGE)
        except Exception as e:
            print(f"[OpenAlex] error fetching {journal}: {e}")
            papers = []
        for p in papers:
            pid = p.get("paperId") or (p.get("title") or "").strip().lower()
            if not pid:
                continue
            if pid not in overall:
                overall[pid] = p
    collected = list(overall.values())
    print(f"Fetched {len(collected)} unique papers across journals via OpenAlex.")
    return collected

# 3) Processing helpers (keywords, summarization, paper type)
def extract_keywords(paper, top_k=5):
    kws = paper.get("keywords")
    if kws:
        if isinstance(kws, list):
            if len(kws) > 0 and isinstance(kws[0], dict):
                vals = [k.get("name", "").strip() for k in kws if k.get("name")]
            else:
                vals = [str(k).strip() for k in kws if k]
        else:
            vals = [str(kws).strip()]
        vals = [v for v in vals if v]
        if vals:
            vals = vals[:top_k]
            return vals, "; ".join(vals)

    abstract = (paper.get("abstract") or "").strip()
    if not abstract:
        return [], ""

    kw_extractor = yake.KeywordExtractor(
        lan="en",
        n=3,
        top=40,
        dedupLim=0.9,
        features=None
    )
    yake_res = kw_extractor.extract_keywords(abstract)
    if not yake_res:
        return [], ""

    candidates = [kw for kw, score in sorted(yake_res, key=lambda x: x[1])[:40]]

    cand_clean = []
    for c in candidates:
        c_str = c.strip()
        if not c_str:
            continue
        if len(c_str) <= 2:
            continue
        if c_str.isdigit():
            continue
        if re.fullmatch(r'[\W_]+', c_str):
            continue
        cand_clean.append(c_str)
    if not cand_clean:
        return [], ""

    # embedding rerank
    doc_emb = embedder.encode(abstract, convert_to_tensor=True)
    cand_embs = embedder.encode(cand_clean, convert_to_tensor=True)
    sims_doc = util.cos_sim(doc_emb, cand_embs)[0].cpu().numpy()
    MIN_SIM = 0.25
    scored = []
    for i, c in enumerate(cand_clean):
        score = float(sims_doc[i])
        if score >= MIN_SIM:
            scored.append((c, score))
    if not scored:
        fallback = cand_clean[:6]
        final = fallback[:top_k]
        return final, "; ".join(final)
    scored.sort(key=lambda x: x[1], reverse=True)
    final_keywords = [c for c, s in scored[:top_k]]
    return final_keywords, "; ".join(final_keywords)

def summarize_abstract_with_qwen(abstract, max_new_tokens=64):
    if not abstract or not abstract.strip():
        return ""
    system_prompt = (
        "You are an academic assistant. Given an academic abstract, "
        "write TWO powerful sentences (max three) capturing: "
        "1) Core research QUESTION, 2) KEY METHOD, 3) MAIN CONTRIBUTION/RESULTS. "
        "Synthesize entire abstract. Avoid generic opening lines. "
        "Use precise academic language. No verbatim copying."
    )
    user_prompt = f"Abstract:\n{abstract}\n\nTLDR:"
    full_prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    try:
        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if "<|im_start|>assistant" in decoded:
            summary = decoded.split("<|im_start|>assistant")[-1].strip()
        elif "TLDR:" in decoded:
            summary = decoded.split("TLDR:")[-1].strip()
        else:
            summary = decoded.strip()
        summary = re.sub(r'<\|[^>]+>', '', summary).strip()
        summary = re.sub(r'\b(assistant|user|system)\b', '', summary, flags=re.IGNORECASE).strip()
        summary = re.sub(r'TLDR\s*:?\s*', '', summary, flags=re.IGNORECASE).strip()
        sentences = re.split(r'(?<=[.!?])\s+', summary)
        summary_trimmed = " ".join(sentences[:2]).strip()
        if summary_trimmed and summary_trimmed[-1] not in ".!?":
            summary_trimmed += "."
        return summary_trimmed
    except Exception as e:
        print(f"[WARN] summarize_abstract_with_qwen failed: {e}")
        return ""

def infer_paper_type(title, abstract):
    text = f"Title: {title}\nAbstract: {abstract}"
    if len(text) < 50:
        return "unclear"
    prompt = f"""Classify this academic paper into ONE category:

                Options: review, qualitative, quantitative, case_study, essay, research, unclear

                Paper:
                { text }

                Type:"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                temperature=0.0,
                eos_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    except Exception as e:
        print(f"[WARN] infer_paper_type generation failed: {e}")
        return "unclear"

    if "Type:" in response:
        classification = response.split("Type:")[-1].strip().split()[0].lower()
    else:
        classification = response.strip().lower()
    type_map = {
        "review": "review", "survey": "review",
        "qualitative": "qualitative", "quantitative": "quantitative",
        "quant": "quantitative", "case": "case_study", "study": "case_study",
        "essay": "essay", "perspective": "essay", "opinion": "essay",
        "research": "research", "empirical": "research", "experimental": "research"
    }
    return type_map.get(classification, "unclear")

# 4) Top-level API: fetch_openalex_for_journals and process_paper_by_meta
def fetch_openalex_for_journals(years_back: int = YEARS_BACK) -> List[Dict[str, Any]]:
    return fetch_openalex_for_journals_impl(years_back=years_back)

def fetch_openalex_for_journals_impl(years_back: int = YEARS_BACK) -> List[Dict[str, Any]]:
    overall = {}
    print(f"Fetching papers for journals (last {years_back} years) via OpenAlex across {len(ALLOWED_JOURNALS)} journals...")
    for journal in ALLOWED_JOURNALS:
        try:
            papers = fetch_openalex_for_journal(journal, years_back=years_back, per_page=OPENALEX_PER_PAGE)
        except Exception as e:
            print(f"[OpenAlex] error fetching {journal}: {e}")
            papers = []
        for p in papers:
            pid = p.get("paperId") or (p.get("title") or "").strip().lower()
            if not pid:
                continue
            if pid not in overall:
                overall[pid] = p
    collected = list(overall.values())
    print(f"Fetched {len(collected)} unique papers across journals via OpenAlex.")
    return collected

def process_paper_by_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given a single paper metadata dict from OpenAlex fetch, return a dict
    with keys:
      paper_id, title, authors, year, venue, abstract_full,
      keywords, abstract_summary, paper_type, link
    """
    out = {}
    raw_pid = meta.get("paperId") or meta.get("paper_id") or meta.get("id") or meta.get("openalex_id")
    # keep normalized id (strip url/prefix)
    if raw_pid and isinstance(raw_pid, str) and raw_pid.startswith("https://"):
        pid = raw_pid.split("/")[-1]
    else:
        pid = raw_pid
    out["paper_id"] = pid

    title = (meta.get("title") or "").strip()
    out["title"] = title

    # authors to comma-separated string
    raw_authors = meta.get("authors") or []
    authors_list = []
    for a in raw_authors:
        if isinstance(a, dict):
            name = a.get("name") or a.get("display_name") or a.get("author") or a.get("id") or ""
        else:
            name = str(a or "")
        if name:
            authors_list.append(name)
    out["authors"] = ", ".join(authors_list)

    out["year"] = meta.get("year") or ""
    out["venue"] = meta.get("venue") or ""

    abstract = (meta.get("abstract") or "") or ""
    out["abstract_full"] = abstract

    # keywords
    try:
        kws_list, kws_str = extract_keywords(meta, top_k=6)
    except Exception as e:
        print(f"[WARN] extract_keywords error for {pid}: {e}")
        kws_list, kws_str = [], ""
    out["keywords"] = kws_str

    # summary (Qwen)
    try:
        summary = summarize_abstract_with_qwen(abstract, max_new_tokens=64)
    except Exception as e:
        print(f"[WARN] summarize error for {pid}: {e}")
        summary = ""
    out["abstract_summary"] = summary

    # paper type classification
    try:
        ptype = infer_paper_type(title, abstract)
    except Exception as e:
        print(f"[WARN] paper type infer error for {pid}: {e}")
        ptype = "unclear"
    out["paper_type"] = ptype

    # link
    out["link"] = meta.get("doi") or meta.get("url") or meta.get("openalex_url") or meta.get("paperId") or ""

    return out

# Export the two main functions with stable names:
def fetch_openalex_for_journals_wrapper():
    return fetch_openalex_for_journals_impl()

# alias for the incremental driver
fetch_openalex_for_journals = fetch_openalex_for_journals_impl
