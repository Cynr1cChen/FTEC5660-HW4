"""FTEC5660 Homework 4: the retrieval half of a RAG assistant over company reports.

Fill in the five functions marked ### YOUR CODE HERE. You may add imports, constants and helper
functions above them, but do not change the provided code below them (the models, the generator,
the loaders, the runner and the scoring).

    python3 hw4.py --data-folder public_test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
import traceback
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek

load_dotenv()


from bs4 import BeautifulSoup, Tag
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

# =============================================================================
# Your code: add imports, constants and helper functions here.
# =============================================================================

_LEGAL_SUFFIXES = {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co"}
_CORPUS_COMPANIES: dict[str, str] = {}
_CORPUS_DOC_CATALOG: dict[str, list[tuple[int, str, str]]] = {}
_CORPUS_BROKERS: set[str] = set()
_CORPUS_AUDITORS: set[str] = set()
_PARSE_CACHE: dict[str, dict] = {}
_RERANK_LLM: ChatDeepSeek | None = None

# In the controlled ablation each row enables just the next component.
ABLATION_STEPS = (
    ("table_text", "Preserve table rows, columns and captions"),
    ("section_chunking", "Split on sections and table rows"),
    ("context_header", "Add contextual chunk headers"),
    ("metadata_routing", "Parse queries and route by company, year and document type"),
    ("company_coverage", "Reserve retrieval slots for both companies"),
    ("local_bm25", "Compute BM25 within each metadata-filtered pool"),
    ("financial_tokens", "Use financial tokenization and inflection handling"),
    ("bm25_parameters", "Tune BM25 k1 and b"),
    ("query_expansion", "Expand financial queries with related terms"),
    ("ranking_priors", "Boost exact tables and financial phrases"),
    ("rerank", "Rerank candidate passages with the LLM"),
    ("metadata_gate", "Abstain for unsupported company, document type or year"),
    ("evidence_gate", "Abstain when the LLM finds insufficient evidence"),
    ("widen_on_rejection", "Recheck an initial rejection on a larger pool"),
)


def _features() -> frozenset[str]:
    variant = os.environ.get("HW4_VARIANT", "final")
    if variant in ("final", "V7"):
        return frozenset(key for key, _ in ABLATION_STEPS)
    if variant == "B0":
        return frozenset()

    # Keep the former variant names usable for the existing report.
    legacy = {"V1": 1, "V2": 3, "V3": 6, "V4": 6, "V5": 10, "V6": 11}
    if variant in legacy:
        features = frozenset(key for key, _ in ABLATION_STEPS[:legacy[variant]])
        return features - {"company_coverage"} if variant == "V3" else features
    if re.fullmatch(r"A\d{2}", variant) and 1 <= int(variant[1:]) <= len(ABLATION_STEPS):
        return frozenset(key for key, _ in ABLATION_STEPS[:int(variant[1:])])
    raise ValueError(f"Unknown HW4_VARIANT: {variant!r}")


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for",
    "from", "had", "has", "have", "how", "in", "is", "it", "its", "many", "more",
    "much", "of", "on", "or", "the", "their", "to", "was", "were", "what", "which",
    "who", "with", "usd", "million", "millions", "billion", "percent", "december",
    "31", "end", "ended", "year", "fiscal", "company", "companies", "higher", "more",
    "hold", "held", "report", "reported", "generate", "generated", "set", "give",
    "announce", "announced", "audited", "spend", "spent", "earn", "earned", "shares",
}
_PLURAL_STEMS = {
    "inventories": "inventory",
    "employees": "employee",
    "borrowings": "borrowing",
    "receivables": "receivable",
    "payables": "payable",
    "activities": "activity",
    "expenses": "expense",
    "assets": "asset",
    "liabilities": "liability",
    "segments": "segment",
    "regions": "region",
    "estimates": "estimate",
    "ratings": "rating",
    "revenues": "revenue",
    "profits": "profit",
    "earnings": "earning",
    "purchases": "purchase",
    "repurchases": "repurchase",
    "dividends": "dividend",
    "directors": "director",
    "officers": "officer",
    "executives": "executive",
}


def _clean_ws(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text or "")).strip()


def _normalize_company_key(name: str) -> str:
    text = unicodedata.normalize("NFKC", str(name)).lower().replace("_", " ")
    words = [w for w in re.sub(r"[^a-z0-9]+", " ", text).split() if w not in _LEGAL_SUFFIXES]
    return " ".join(words)


def _strip_company_name(raw_name: str) -> str:
    name = _clean_ws(raw_name)
    # Remove trailing parenthetical exchange/ticker such as "(TSX: BWE)" or "(SBWR)"
    name = re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()
    # Strip trailing legal suffixes while preserving core company words
    tokens = name.split()
    while tokens and re.sub(r"[^a-z0-9]", "", tokens[-1].lower()) in _LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens) if tokens else name


def _extract_cell_text(cell: Tag) -> str:
    return _clean_ws(cell.get_text(" ", strip=True))


def _expand_row_cells(tr: Tag) -> list[str]:
    vals: list[str] = []
    for cell in tr.find_all(["th", "td"], recursive=False):
        txt = _extract_cell_text(cell)
        try:
            span = max(1, int(cell.get("colspan", 1)))
        except (TypeError, ValueError):
            span = 1
        vals.extend([txt] * span)
    return vals


def table_to_text(table_html: str) -> str:
    """Task 1.1: turn one HTML <table> into text that keeps every value with its row and its column.

    Put each value on a line (at most 400 characters) together with its row label and its column
    header, e.g. "Revenue | 2025: 2,710.4 | 2024: 2,629.1", one table row per line, keep the text of
    every cell unchanged, and include the caption (it holds the unit) in the output. Do not call the
    LLM here.
    """
    soup = BeautifulSoup(table_html, "lxml")
    if "table_text" not in _features():
        return soup.get_text(separator="\n", strip=True)
    table = soup.find("table")
    if table is None:
        return ""

    lines: list[str] = []
    caption_tag = table.find("caption")
    if caption_tag is not None:
        cap_text = _clean_ws(caption_tag.get_text(" ", strip=True))
        if cap_text:
            lines.append(cap_text)

    thead = table.find("thead")
    header_rows: list[list[str]] = []
    body_trs: list[Tag] = []

    if thead is not None:
        for tr in thead.find_all("tr", recursive=False):
            header_rows.append(_expand_row_cells(tr))
        for tbody in table.find_all("tbody", recursive=False):
            body_trs.extend(tbody.find_all("tr", recursive=False))
        # Handle any direct tr outside thead/tbody
        for tr in table.find_all("tr", recursive=False):
            if tr.parent is table:
                body_trs.append(tr)
    else:
        all_trs = table.find_all("tr")
        if all_trs:
            first_cells = all_trs[0].find_all(["th", "td"], recursive=False)
            if first_cells and all(c.name == "th" for c in first_cells):
                header_rows.append(_expand_row_cells(all_trs[0]))
                body_trs = all_trs[1:]
            else:
                body_trs = all_trs

    col_headers: list[str] = []
    if header_rows:
        num_cols = max(len(r) for r in header_rows)
        for col_idx in range(num_cols):
            parts = []
            for hr in header_rows:
                if col_idx < len(hr) and hr[col_idx]:
                    if not parts or parts[-1] != hr[col_idx]:
                        parts.append(hr[col_idx])
            col_headers.append(" ".join(parts))

    for tr in body_trs:
        raw_cells = tr.find_all(["th", "td"], recursive=False)
        if not raw_cells:
            continue
        tr_classes = tr.get("class") or []
        is_section_row = ("section" in tr_classes) or (
            len(raw_cells) == 1 and int(raw_cells[0].get("colspan", 1) or 1) > 1
        )
        if is_section_row:
            sec_text = _clean_ws(tr.get_text(" ", strip=True))
            if sec_text:
                lines.append(f"[{sec_text}]")
            continue

        cells = [_extract_cell_text(c) for c in raw_cells]
        if not any(cells):
            continue
        row_label = cells[0]
        if len(cells) == 1:
            lines.append(row_label)
            continue

        pairs: list[str] = []
        for idx in range(1, len(cells)):
            val = cells[idx]
            col_h = col_headers[idx] if idx < len(col_headers) and col_headers[idx] else f"Col {idx}"
            pairs.append(f"{col_h}: {val}")

        full_line = f"{row_label} | " + " | ".join(pairs)
        if len(full_line) <= MAX_LINE_CHARS:
            lines.append(full_line)
        else:
            # Split wide rows across multiple lines while keeping row_label and <= MAX_LINE_CHARS
            cur_pairs: list[str] = []
            for pair in pairs:
                candidate = f"{row_label} | " + " | ".join(cur_pairs + [pair])
                if cur_pairs and len(candidate) > MAX_LINE_CHARS:
                    lines.append(f"{row_label} | " + " | ".join(cur_pairs))
                    cur_pairs = [pair]
                else:
                    cur_pairs.append(pair)
            if cur_pairs:
                lines.append(f"{row_label} | " + " | ".join(cur_pairs))

    return "\n".join(lines)


def _extract_doc_metadata(file_name: str, soup: BeautifulSoup) -> dict[str, Any]:
    title_text = _clean_ws(soup.title.get_text(" ", strip=True)) if soup.title else ""
    broker_div = soup.find("div", class_="broker")
    h1 = soup.find("h1")
    h2 = soup.find("h2")
    subtitle_p = soup.find("p", class_="subtitle")

    h1_text = _clean_ws(h1.get_text(" ", strip=True)) if h1 else ""
    h2_text = _clean_ws(h2.get_text(" ", strip=True)) if h2 else ""
    broker_text = _clean_ws(broker_div.get_text(" ", strip=True)) if broker_div else ""
    sub_text = _clean_ws(subtitle_p.get_text(" ", strip=True)) if subtitle_p else ""

    if broker_div is not None or "results review" in h2_text.lower() or "results review" in title_text.lower():
        doc_type = "research_report"
    elif "earnings conference call" in h2_text.lower() or "earnings call" in title_text.lower() or "earnings call" in h2_text.lower():
        doc_type = "earnings_call"
    else:
        doc_type = "annual_report"

    company = _strip_company_name(h1_text)
    if not company and title_text:
        company = _strip_company_name(title_text.split("|")[0])

    fiscal_year: int | None = None
    if doc_type == "annual_report":
        for candidate in (sub_text, title_text, _clean_ws(soup.find("div", class_="cover").get_text(" ", strip=True)) if soup.find("div", class_="cover") else ""):
            m = re.search(r"\b(20\d{2})\b", candidate)
            if m:
                fiscal_year = int(m.group(1))
                break
    else:
        for candidate in (h2_text, title_text):
            m = re.search(r"\b(?:FY\s*)?(20\d{2})\b", candidate, flags=re.I)
            if m:
                fiscal_year = int(m.group(1))
                break

    if fiscal_year is None:
        m = re.search(r"\b(20\d{2})\b", _clean_ws(soup.get_text(" ", strip=True)[:600]))
        fiscal_year = int(m.group(1)) if m else 2025

    return {
        "source": file_name,
        "company": company,
        "fiscal_year": int(fiscal_year),
        "doc_type": doc_type,
        "broker": broker_text.split("|")[0].strip() if broker_text else "",
    }


def _block_text(tag: Tag) -> str:
    # Convert <br> to newlines cleanly while preserving inline <b>/<a> spacing
    clone = BeautifulSoup(str(tag), "lxml")
    root = clone.find(tag.name)
    if root is None:
        return ""
    for br in root.find_all("br"):
        br.replace_with("\n")
    raw = root.get_text(" ", strip=False)
    lines = [_clean_ws(ln) for ln in raw.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def _split_long_block(text: str, max_chars: int, is_table: bool = False, overlap: int = 120) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    lines = text.splitlines()
    if is_table and lines:
        caption_line = lines[0]
        data_lines = lines[1:]
        pieces: list[str] = []
        cur = caption_line
        active_section: str | None = None
        pending_section: str | None = None
        for ln in data_lines:
            if ln.startswith("[") and ln.endswith("]"):
                active_section = ln
                pending_section = ln
                continue
            add_block = f"{pending_section}\n{ln}" if pending_section else ln
            candidate = f"{cur}\n{add_block}" if cur else add_block
            if len(candidate) <= max_chars:
                cur = candidate
                pending_section = None
            else:
                if cur:
                    pieces.append(cur)
                prefix = f"{caption_line}\n{active_section}" if active_section else caption_line
                if len(f"{prefix}\n{ln}") <= max_chars:
                    cur = f"{prefix}\n{ln}"
                elif len(f"{caption_line}\n{ln}") <= max_chars:
                    cur = f"{caption_line}\n{ln}"
                else:
                    cur = ln[:max_chars]
                pending_section = None
        if cur:
            pieces.append(cur)
        return pieces

    # Narrative block splitting by lines, then sentences/words with overlap
    pieces = []
    if len(lines) > 1:
        cur = ""
        for ln in lines:
            if len(ln) > max_chars:
                if cur:
                    pieces.append(cur)
                    cur = ""
                pieces.extend(_split_long_block(ln, max_chars, is_table=False, overlap=overlap))
            elif not cur:
                cur = ln
            elif len(cur) + 1 + len(ln) <= max_chars:
                cur = f"{cur}\n{ln}"
            else:
                pieces.append(cur)
                cur = ln
        if cur:
            pieces.append(cur)
        return pieces

    # Single long paragraph: split by sentence boundaries, fallback to words
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) == 1:
        words = text.split()
        cur = ""
        for w in words:
            if not cur:
                cur = w[:max_chars]
            elif len(cur) + 1 + len(w) <= max_chars:
                cur = f"{cur} {w}"
            else:
                pieces.append(cur)
                tail = cur[-overlap:] if overlap > 0 else ""
                if " " in tail:
                    tail = tail[tail.find(" ") + 1:]
                cur = f"{tail} {w}".strip() if tail and len(tail) + 1 + len(w) <= max_chars else w[:max_chars]
        if cur:
            pieces.append(cur)
        return pieces

    cur = ""
    for sent in sentences:
        if len(sent) > max_chars:
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.extend(_split_long_block(sent, max_chars, is_table=False, overlap=overlap))
        elif not cur:
            cur = sent
        elif len(cur) + 1 + len(sent) <= max_chars:
            cur = f"{cur} {sent}"
        else:
            pieces.append(cur)
            tail = cur[-overlap:] if overlap > 0 else ""
            if " " in tail:
                tail = tail[tail.find(" ") + 1:]
            cur = f"{tail} {sent}".strip() if tail and len(tail) + 1 + len(sent) <= max_chars else sent
    if cur:
        pieces.append(cur)
    return pieces


def _register_corpus_entities(soup: BeautifulSoup, meta: dict[str, Any]) -> None:
    comp_key = _normalize_company_key(meta["company"])
    if comp_key:
        _CORPUS_COMPANIES[comp_key] = meta["company"]
        entries = _CORPUS_DOC_CATALOG.setdefault(comp_key, [])
        entry = (meta["fiscal_year"], meta["doc_type"], meta["source"])
        if entry not in entries:
            entries.append(entry)
    if meta.get("broker"):
        bk = _normalize_company_key(meta["broker"])
        if bk:
            _CORPUS_BROKERS.add(bk)
    if meta["doc_type"] == "earnings_call":
        for li in soup.find_all("li"):
            txt = _clean_ws(li.get_text(" ", strip=True))
            if " - " in txt:
                aff = txt.split(" - ", 1)[1].strip()
                if aff and not any(t in aff.lower() for t in ("chief ", "head of", "officer")):
                    ak = _normalize_company_key(aff)
                    if ak:
                        _CORPUS_BROKERS.add(ak)
    elif meta["doc_type"] == "annual_report":
        for h3 in soup.find_all("h3"):
            if "auditor" in _clean_ws(h3.get_text(" ", strip=True)).lower():
                sib = h3.find_next_sibling()
                while sib is not None and sib.name == "p":
                    ptxt = _clean_ws(sib.get_text(" ", strip=True))
                    for m in re.findall(
                        r"\b([A-Z][A-Za-z&.\-]+(?:\s+[A-Z][A-Za-z&.\-]+){0,4}\s+(?:LLP|Partners|Audit|Associates))\b",
                        ptxt,
                    ):
                        ak = _normalize_company_key(m)
                        if ak:
                            _CORPUS_AUDITORS.add(ak)
                    sib = sib.find_next_sibling()


def chunk_document(file_name: str, html: str) -> list[Document]:
    """Task 1.2: split one document into chunks of at most MAX_CHUNK_CHARS characters.

    Every chunk needs the metadata "source" (= file_name), "company", "fiscal_year" (an int) and
    "doc_type" (one of DOC_TYPES). Do not call the LLM here.
    """
    features = _features()
    soup = BeautifulSoup(html, "lxml")
    meta = _extract_doc_metadata(file_name, soup)
    _register_corpus_entities(soup, meta)

    base_meta = {
        "source": meta["source"],
        "company": meta["company"],
        "fiscal_year": meta["fiscal_year"],
        "doc_type": meta["doc_type"],
    }

    if "section_chunking" not in features:
        if "table_text" in features:
            for tbl in soup.find_all("table"):
                tbl_txt = table_to_text(str(tbl))
                new_tag = soup.new_tag("pre")
                new_tag.string = f"\n{tbl_txt}\n"
                tbl.replace_with(new_tag)
        raw_text = soup.get_text(separator="\n", strip=True)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return [Document(page_content=t[:MAX_CHUNK_CHARS], metadata=dict(base_meta)) for t in splitter.split_text(raw_text) if t.strip()]

    body = soup.body or soup
    for tag_name in ("nav", "script", "style"):
        for t in body.find_all(tag_name):
            t.decompose()

    # Extract ordered content blocks without emitting orphan headings
    blocks: list[tuple[str, str, bool, str]] = []
    cur_h2 = ""
    cur_h3 = ""

    def walk(node: Tag) -> None:
        nonlocal cur_h2, cur_h3
        for child in node.children:
            if not isinstance(child, Tag):
                continue
            name = child.name.lower()
            if name == "table":
                t_text = table_to_text(str(child))
                if t_text:
                    blocks.append((cur_h2, cur_h3, True, t_text))
            elif name == "h1":
                txt = _block_text(child)
                if txt:
                    blocks.append((cur_h2, cur_h3, False, txt))
            elif name == "h2":
                txt = _block_text(child)
                if txt:
                    cur_h2 = txt
                    cur_h3 = ""
            elif name == "h3":
                txt = _block_text(child)
                if txt:
                    cur_h3 = txt
            elif name in ("ul", "ol"):
                items = [f"- {_block_text(li)}" for li in child.find_all("li", recursive=False)]
                items = [it for it in items if it != "- "]
                if items:
                    blocks.append((cur_h2, cur_h3, False, "\n".join(items)))
            elif name == "p":
                txt = _block_text(child)
                if txt:
                    blocks.append((cur_h2, cur_h3, False, txt))
            elif name == "div" and "broker" in (child.get("class") or []):
                txt = _block_text(child)
                if txt:
                    blocks.append((cur_h2, cur_h3, False, txt))
            else:
                walk(child)

    walk(body)

    def make_header(h2: str, h3: str) -> str:
        if "context_header" not in features:
            return ""
        sec_parts = [p for p in (h2, h3) if p]
        sec = " > ".join(sec_parts) if sec_parts else "Overview"
        return f"[{meta['company']} | FY{meta['fiscal_year']} {meta['doc_type']} | {sec}]\n"

    # Pack blocks into section-aware chunks <= MAX_CHUNK_CHARS
    chunks: list[Document] = []
    cur_h2_grp = ""
    cur_h3_grp = ""
    cur_h3_last_in_chunk = ""
    cur_header = make_header("", "")
    cur_parts: list[str] = []
    cur_has_table = False

    def flush_current() -> None:
        nonlocal cur_parts, cur_has_table, cur_h3_last_in_chunk
        if not cur_parts:
            return
        body_text = "\n\n".join(cur_parts)
        full_text = cur_header + body_text
        sec_label = " > ".join(p for p in (cur_h2_grp, cur_h3_grp) if p) or "Overview"
        chunk_meta = {**base_meta, "section": sec_label, "is_table": cur_has_table}
        if len(full_text) <= MAX_CHUNK_CHARS:
            chunks.append(Document(page_content=full_text, metadata=chunk_meta))
        else:
            avail = max(200, MAX_CHUNK_CHARS - len(cur_header))
            for piece in _split_long_block(body_text, avail, is_table=cur_has_table):
                chunks.append(Document(page_content=cur_header + piece, metadata=chunk_meta))
        cur_parts = []
        cur_has_table = False
        cur_h3_last_in_chunk = ""

    for b_h2, b_h3, is_table, text in blocks:
        if cur_parts and b_h2 != cur_h2_grp:
            flush_current()
        elif cur_parts and b_h3 != cur_h3_last_in_chunk:
            cur_len = len(cur_header) + sum(len(p) + 2 for p in cur_parts)
            if cur_has_table or is_table or cur_len >= 600:
                flush_current()

        if not cur_parts:
            cur_h2_grp = b_h2
            cur_h3_grp = b_h3
            cur_h3_last_in_chunk = b_h3
            cur_header = make_header(b_h2, b_h3)
            # Prepend h2/h3 inline label if helpful and not already inside text
            heading_prefix_parts = []
            if b_h2 and b_h2.lower() not in text.lower()[:120]:
                heading_prefix_parts.append(b_h2)
            if b_h3 and b_h3.lower() not in text.lower()[:120]:
                heading_prefix_parts.append(b_h3)
            block_str = ("\n".join(heading_prefix_parts) + "\n" + text) if heading_prefix_parts else text
        else:
            # Continuing inside same chunk; if entering a new h3, drop h3 from chunk header and prefix inline
            if b_h3 != cur_h3_last_in_chunk:
                cur_h3_grp = ""
                cur_header = make_header(cur_h2_grp, "")
                cur_h3_last_in_chunk = b_h3
                block_str = f"{b_h3}\n{text}" if (b_h3 and b_h3.lower() not in text.lower()[:120]) else text
            else:
                block_str = text

        candidate_len = (
            len(cur_header) + sum(len(p) + 2 for p in cur_parts) + (2 if cur_parts else 0) + len(block_str)
        )
        if cur_parts and candidate_len > MAX_CHUNK_CHARS:
            flush_current()
            cur_h2_grp = b_h2
            cur_h3_grp = b_h3
            cur_h3_last_in_chunk = b_h3
            cur_header = make_header(b_h2, b_h3)
            heading_prefix_parts = []
            if b_h3 and b_h3.lower() not in text.lower()[:120]:
                heading_prefix_parts.append(b_h3)
            block_str = ("\n".join(heading_prefix_parts) + "\n" + text) if heading_prefix_parts else text

        avail = max(200, MAX_CHUNK_CHARS - len(cur_header))
        if len(block_str) > avail:
            flush_current()
            cur_h2_grp = b_h2
            cur_h3_grp = b_h3
            cur_header = make_header(b_h2, b_h3)
            sec_label = " > ".join(p for p in (b_h2, b_h3) if p) or "Overview"
            chunk_meta = {**base_meta, "section": sec_label, "is_table": is_table}
            for piece in _split_long_block(text, avail, is_table=is_table):
                chunks.append(Document(page_content=cur_header + piece, metadata=chunk_meta))
        else:
            cur_parts.append(block_str)
            if is_table:
                cur_has_table = True

    flush_current()
    return chunks


def _is_non_target_org(name: str) -> bool:
    k = _normalize_company_key(name)
    if not k:
        return True
    if k in _CORPUS_COMPANIES:
        return False
    if k in _CORPUS_BROKERS or k in _CORPUS_AUDITORS:
        return True
    bad_prefixes = (
        "chief ", "board ", "chair ", "audit ", "independent ", "general ",
        "fourth ", "first ", "second ", "third ", "january", "february", "march",
        "april", "may", "june", "july", "august", "september", "october", "november", "december",
    )
    if k.startswith(bad_prefixes):
        return True
    return False


def _extract_json_obj(raw: str) -> dict | None:
    text = (raw or "").strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _comparison_names(question: str) -> tuple[bool, list[str]]:
    """Extract two option spans without absorbing the metric's date or trailing clause."""
    between = re.search(r"\bbetween\s+(.+)", question, flags=re.I)
    prefix, colon, options = question.partition(":")
    if between:
        options = between[1]
        separator = r"\band\b"
    elif colon and re.search(r"\b(?:which|compare)\b", prefix, flags=re.I):
        separator = r"(?:\bor\b|\bversus\b|\bvs\b\.?)"
        if not re.search(separator, options, flags=re.I):
            return False, []
    else:
        is_comparison = bool(re.search(
            r"\b(?:compare|versus|vs|compared (?:with|to))\b|\bwhich\b.+\bor\b", question, flags=re.I,
        ))
        return is_comparison, []

    # Preserve the full name, even if it resembles a company in the corpus.
    # Only strip recognized timing, unit or follow-up clauses from the options.
    year_or_date = (
        r"(?:the\s+end\s+of\s+)?(?:the\s+)?(?:FY\s*|fiscal\s+year\s+)?"
        r"(?:\d{1,2}\s+[A-Za-z]+\s+|[A-Za-z]+\s+\d{1,2},?\s+)?20\d{2}\b"
    )
    options = re.split(
        rf",?\s+\(?(?:in|for|during|at|as\s+of|on)\s+{year_or_date}|"
        r"\s+(?:at|by)\s+year[- ]end\b|\s+in\s+(?:USD|dollars|millions)\b|"
        r",\s*(?=(?:which|who|and\s+(?:why|what|which)|respectively)\b)",
        options, maxsplit=1, flags=re.I,
    )[0]
    candidates = []
    for match in re.finditer(rf"\s+{separator}\s+", options, flags=re.I):
        pair = [_strip_company_name(part.strip(" .,:;?!\"'")) for part in (options[:match.start()], options[match.end():])]
        if all(pair) and not any(re.search(r"[?!;:]", name) for name in pair):
            known = sum(_normalize_company_key(name) in _CORPUS_COMPANIES for name in pair)
            candidates.append((known, pair))
    if not candidates:
        return True, []
    candidates.sort(key=lambda item: item[0], reverse=True)
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        return True, []  # Ambiguous conjunction inside an organization name.
    return True, candidates[0][1]


def parse_query(question: str) -> dict:
    """Task 2.2: {"companies": [names], "fiscal_year": int or None, "doc_type": one of DOC_TYPES}.

    The rules for each field are under Task 2.2 in the handout. You may use make_llm() here.
    """
    if "metadata_routing" not in _features():
        return {}
    question = _clean_ws(question).translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'}))
    if question in _PARSE_CACHE:
        return dict(_PARSE_CACHE[question])

    q_low = question.lower()
    if "earnings call" in q_low or "conference call" in q_low or "fourth-quarter earnings" in q_low:
        doc_type = "earnings_call"
    elif (
        any(kw in q_low for kw in ("results review", "target price", "investment rating", "equity research", "forecast"))
        or re.search(r"\b(?:broker|analyst|rating|estimate|estimates)\b", q_low)
    ):
        doc_type = "research_report"
    else:
        doc_type = "annual_report"

    # Parse fiscal_year (metric year, stripping report-title year markers if multiple years present)
    all_years = [int(y) for y in re.findall(r"\b(?:FY\s*)?(20\d{2})\b", question, flags=re.I)]
    if len(all_years) == 1:
        fiscal_year: int | None = all_years[0]
    elif len(all_years) > 1:
        cleaned_q = re.sub(r"\b(?:Q\d\s+)?FY\s*20\d{2}\s+(?:results\s+review|earnings(?:\s+conference)?\s+call)\b", "", question, flags=re.I)
        rem_years = [int(y) for y in re.findall(r"\b(?:FY\s*)?(20\d{2})\b", cleaned_q, flags=re.I)]
        fiscal_year = rem_years[0] if rem_years else all_years[-1]
    else:
        fiscal_year = None

    companies: list[str] = []
    seen_keys: set[str] = set()

    def add_company(cand: str) -> None:
        cleaned = _strip_company_name(cand.strip(" .,:;?\"'"))
        if not cleaned or _is_non_target_org(cleaned):
            return
        k = _normalize_company_key(cleaned)
        if k and k not in seen_keys:
            seen_keys.add(k)
            companies.append(cleaned)

    # Rule A: Prefer explicit company options for comparisons.
    is_comparison, comparison_names = _comparison_names(question)
    for name in comparison_names:
        add_company(name)

    # Rule B: Research report target company ("give X", "set for X", "forecast for X")
    if not companies:
        for pat in (
            r"\b(?:give|set\s+for|forecast\s+for|forecasts?\s+for|rate|cover)\s+([A-Z][A-Za-z0-9&.\-]+(?:\s+(?:[A-Z][A-Za-z0-9&.\-]+|and|&)){0,4}?)(?:'s\b|'(?=\s|$|\?)|\s+in\s+its\b|\s+shares\b|\s*,|\?|$)",
        ):
            for m in re.finditer(pat, question):
                add_company(m.group(1))

    # Rule C: Possessive "<Company>'s" or "<Company>'"
    if not companies:
        for m in re.finditer(
            r"\b([A-Z][A-Za-z0-9&.\-]+(?:\s+(?:[A-Z][A-Za-z0-9&.\-]+|and|&)){0,4})(?:'s\b|'(?=\s|[.,?]|$))",
            question,
        ):
            add_company(m.group(1))

    # Rule D: Action verb / headquarters / audit frames
    if not companies:
        for pat in (
            r"\bdid\s+([A-Z][A-Za-z0-9&.\-]+(?:\s+(?:[A-Z][A-Za-z0-9&.\-]+|and|&)){0,4}?)\s+(?:hold|report|generate|earn|spend|announce|employ|guide|pay|invest|record|achieve|make|have)\b",
            r"\bis\s+([A-Z][A-Za-z0-9&.\-]+(?:\s+(?:[A-Z][A-Za-z0-9&.\-]+|and|&)){0,4}?)\s+(?:headquartered|based|located)\b",
            r"\baudited\s+([A-Z][A-Za-z0-9&.\-]+(?:\s+(?:[A-Z][A-Za-z0-9&.\-]+|and|&)){0,4}?)(?:'s\b|'(?=\s|$|\?)|\s+financial\b)",
        ):
            for m in re.finditer(pat, question):
                add_company(m.group(1))

    # Rule E: Known corpus company scan with maximal-span check
    if not companies and _CORPUS_COMPANIES:
        for cname in sorted(_CORPUS_COMPANIES.values(), key=len, reverse=True):
            m = re.search(rf"\b{re.escape(cname)}\b", question, flags=re.I)
            # Do not count a known name that is merely a prefix of another name.
            if m and not re.match(r"\s+[A-Z][a-z]+", question[m.end():]):
                add_company(cname)

    # Rule F: Resolve absent or incomplete comparison entities.
    if not companies or (is_comparison and len(companies) < 2):
        try:
            llm = make_llm(temperature=0.0, thinking=False, max_tokens=120)
            prompt = (
                "Extract structured query fields from the financial question as JSON only.\n"
                'Schema: {"companies": ["..."], "fiscal_year": int_or_null, "doc_type": "annual_report"|"earnings_call"|"research_report"}\n'
                "Rules:\n"
                "- companies: list every target company asked about (even if fictional/unknown); EXCLUDE brokers/research firms and audit firms.\n"
                "- fiscal_year: the fiscal year the question asks about as int, or null if none.\n"
                "- doc_type: earnings_call if asking about earnings call; research_report if asking about broker rating/target price/forecast/results review; else annual_report.\n"
                f"Question: {question}\nJSON:"
            )
            resp = llm.invoke(prompt)
            obj = _extract_json_obj(str(resp.content))
            if obj and isinstance(obj.get("companies"), list):
                normalized_question = f" {_normalize_company_key(question)} "
                for c in obj["companies"]:
                    if isinstance(c, str) and f" {_normalize_company_key(c)} " in normalized_question:
                        add_company(c)
                if fiscal_year is None and obj.get("fiscal_year") is not None:
                    fiscal_year = as_year(obj["fiscal_year"])
                if obj.get("doc_type") in DOC_TYPES:
                    doc_type = obj["doc_type"]
        except Exception:
            pass

    result = {"companies": companies, "fiscal_year": fiscal_year, "doc_type": doc_type}
    _PARSE_CACHE[question] = dict(result)
    return result


def _tokenize_financial(text: str, strip_years_and_units: bool = False) -> list[str]:
    raw_tokens = re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?%?|[a-z0-9]+(?:-[a-z0-9]+)*", unicodedata.normalize("NFKC", text).lower())
    out: list[str] = []
    for tok in raw_tokens:
        if tok in _STOPWORDS:
            continue
        if strip_years_and_units and re.fullmatch(r"(?:fy)?20\d{2}", tok):
            continue
        out.append(tok)
        if tok in _PLURAL_STEMS:
            out.append(_PLURAL_STEMS[tok])
        if "-" in tok:
            for sub in tok.split("-"):
                if sub and sub not in _STOPWORDS:
                    out.append(sub)
                    if sub in _PLURAL_STEMS:
                        out.append(_PLURAL_STEMS[sub])
    return out


def _query_tokens(question: str, companies: list[str], features: frozenset[str]) -> list[str]:
    if "financial_tokens" not in features:
        return question.lower().split()
    q_clean = question
    for company in companies:
        q_clean = re.sub(re.escape(company), " ", q_clean, flags=re.I)
    tokens = _tokenize_financial(q_clean, strip_years_and_units=True)
    if "query_expansion" not in features:
        return tokens or _tokenize_financial(question)
    ql = question.lower()
    expansions: list[str] = []
    rules = [
        (("cash and cash equivalents", "cash"), ["consolidated", "balance", "sheet", "cash", "equivalents"]),
        (("total assets",), ["consolidated", "balance", "sheet", "total", "assets"]),
        (("total equity", "shareholders equity", "stockholders equity"), ["consolidated", "balance", "sheet", "equity", "total", "equity"]),
        (("accounts receivable", "receivables"), ["accounts", "receivable", "net", "note", "balance", "sheet"]),
        (("inventories", "inventory"), ["inventories", "consolidated", "balance", "sheet"]),
        (("long-term borrowing", "long-term borrowings", "long term borrowing"), ["long-term", "borrowings", "note", "balance", "sheet"]),
        (("short-term borrowing", "short-term borrowings"), ["short-term", "borrowings", "note", "balance", "sheet"]),
        (("net cash from operating activities", "operating cash flow"), ["consolidated", "cash", "flow", "statement", "operating", "activities"]),
        (("gross profit",), ["consolidated", "income", "statement", "gross", "profit"]),
        (("operating income",), ["consolidated", "income", "statement", "operating", "income"]),
        (("net income",), ["consolidated", "income", "statement", "net", "income"]),
        (("revenue",), ["revenue", "consolidated", "income", "statement"]),
        (("segment",), ["revenue", "by", "segment", "note"]),
        (("asia pacific", "europe", "americas", "emea", "region"), ["revenue", "by", "region", "note"]),
        (("employees", "employee", "headcount", "workforce"), ["employees", "total", "employees", "function", "note"]),
        (("chief financial officer", "cfo"), ["executive", "officers", "chief", "financial", "officer", "appointed", "succeeding"]),
        (("chief executive officer", "ceo"), ["executive", "officers", "chief", "executive", "officer"]),
        (("audit firm", "audited", "auditor"), ["independent", "auditor", "audited", "opinion"]),
        (("headquartered", "headquarters", "city"), ["headquarters", "business", "overview"]),
        (("target price",), ["12-month", "target", "price", "valuation"]),
        (("rating",), ["rating", "buy", "hold", "reduce"]),
        (("forecast", "estimate"), ["estimate", "changes", "financial", "summary"]),
        (("share repurchase", "buyback"), ["share", "repurchase", "programme", "capital", "allocation"]),
        (("guide", "guidance", "revenue growth"), ["outlook", "expect", "revenue", "growth", "approximately"]),
    ]
    for triggers, extra in rules:
        if any(tr in ql for tr in triggers):
            expansions.extend(extra)
    return tokens + expansions or _tokenize_financial(question)


def build_index(chunks: list[Document]) -> Any:
    """Task 2: index the chunks of the whole corpus (the output of chunk_document), once per run.

    Return anything retrieve() needs: a BM25 or TF-IDF index, the chunks grouped by metadata, ...
    """
    features = _features()
    companies: dict[str, str] = {}
    by_comp_doctype: dict[tuple[str, str], list[int]] = {}
    years_by_comp_doctype: dict[tuple[str, str], set[int]] = {}
    tokenized_chunks: list[list[str]] = []

    use_whitespace = "financial_tokens" not in features
    for idx, ch in enumerate(chunks):
        meta = ch.metadata or {}
        comp = meta.get("company", "")
        ck = _normalize_company_key(comp)
        if ck:
            companies.setdefault(ck, comp)
        dt = meta.get("doc_type", "annual_report")
        fy = as_year(meta.get("fiscal_year"))
        if ck:
            by_comp_doctype.setdefault((ck, dt), []).append(idx)
            if fy is not None:
                years_by_comp_doctype.setdefault((ck, dt), set()).add(fy)
        toks = ch.page_content.lower().split() if use_whitespace else _tokenize_financial(ch.page_content)
        tokenized_chunks.append(toks if toks else ["empty"])

    covered_years: dict[tuple[str, str], set[int]] = {}
    for (ck, dt), ys in years_by_comp_doctype.items():
        cov: set[int] = set()
        for y in ys:
            if dt == "annual_report":
                cov.update((y, y - 1))
            elif dt == "earnings_call":
                cov.update((y, y + 1))
            elif dt == "research_report":
                cov.update((y - 1, y, y + 1, y + 2))
            else:
                cov.add(y)
        covered_years[(ck, dt)] = cov

    bm25_parameters = {"k1": 1.4, "b": 0.55} if "bm25_parameters" in features else {}
    bm25_global = BM25Okapi(tokenized_chunks, **bm25_parameters) if chunks else None
    return {
        "chunks": chunks,
        "tokenized_chunks": tokenized_chunks,
        "bm25_global": bm25_global,
        "bm25_parameters": bm25_parameters,
        "companies": companies,
        "by_comp_doctype": by_comp_doctype,
        "years_by_comp_doctype": years_by_comp_doctype,
        "covered_years": covered_years,
        "chunks_lower": [c.page_content.lower() for c in chunks],
    }


def _select_target_indices(index: dict, ck: str, dt: str, fy: int | None) -> list[int]:
    cand_indices = index["by_comp_doctype"].get((ck, dt), [])
    if not cand_indices:
        return []
    if dt == "annual_report" and fy is not None:
        avail_years = index["years_by_comp_doctype"].get((ck, dt), set())
        target_doc_year: int | None = None
        if fy in avail_years:
            target_doc_year = fy
        elif (fy + 1) in avail_years:
            target_doc_year = fy + 1
        elif avail_years:
            # Prefer nearest future year or latest available
            ge = [y for y in avail_years if y >= fy]
            target_doc_year = min(ge) if ge else max(avail_years)
        if target_doc_year is not None:
            filtered = [
                i for i in cand_indices
                if as_year((index["chunks"][i].metadata or {}).get("fiscal_year")) == target_doc_year
            ]
            if filtered:
                return filtered
    return cand_indices


def _score_candidate_indices(
    index: dict,
    cand_indices: list[int],
    question: str,
    q_toks: list[str],
    features: frozenset[str],
) -> list[int]:
    if not cand_indices:
        return []
    if "local_bm25" in features:
        sub_corpus = [index["tokenized_chunks"][i] for i in cand_indices]
        local_bm25 = BM25Okapi(sub_corpus, **index["bm25_parameters"])
        scores = list(local_bm25.get_scores(q_toks))
    else:
        scores = list(index["bm25_global"].get_batch_scores(q_toks, cand_indices))

    if "ranking_priors" in features:
        ql = question.lower()
        wants_exact_num = any(
            kw in ql for kw in (
                "how much", "what was", "what were", "which company had", "which company held",
                "revenue", "income", "profit", "assets", "equity", "cash", "receivable",
                "inventories", "borrowing", "employees",
            )
        )
        # Key phrase bonuses to prevent balance-sheet split collisions or rounded-prose overshadowing
        key_phrases = [
            "cash and cash equivalents", "accounts receivable, net", "total assets", "total equity",
            "long-term borrowings", "short-term borrowings", "net cash from operating activities",
            "gross profit", "operating income", "net income", "total employees", "total revenue",
            "chief financial officer", "chief executive officer", "independent auditor",
            "headquarters", "12-month target price", "share repurchase",
        ]
        matched_phrases = [p for p in key_phrases if p in ql or p.rstrip("s") in ql]

        for pos, idx in enumerate(cand_indices):
            ch = index["chunks"][idx]
            meta = ch.metadata or {}
            txt_low = index["chunks_lower"][idx]
            sec_low = (meta.get("section") or "").lower()
            is_tbl = bool(meta.get("is_table"))

            if wants_exact_num:
                if is_tbl:
                    scores[pos] += 1.5
                if "letter to shareholders" in sec_low or "contents" in sec_low or "risk factors" in sec_low:
                    scores[pos] -= 1.5
                if "business overview" in sec_low and "employees" in ql:
                    scores[pos] -= 1.0
            for phr in matched_phrases:
                if phr in txt_low:
                    scores[pos] += 2.5
            # Boost specific region or segment name match
            for tok in _tokenize_financial(question, strip_years_and_units=True):
                if len(tok) >= 5 and tok not in _STOPWORDS and tok in txt_low:
                    scores[pos] += 0.25

    ranked = sorted(range(len(cand_indices)), key=lambda p: scores[p], reverse=True)
    return [cand_indices[p] for p in ranked]


def _llm_rerank_and_verify(
    question: str,
    candidate_chunks: list[Document],
    check_answerable: bool,
    is_comparison: bool,
) -> tuple[bool | None, list[int]]:
    if not candidate_chunks:
        return (False, [])
    passages = []
    for i, ch in enumerate(candidate_chunks, 1):
        meta = ch.metadata or {}
        passages.append(
            f"[{i}] company={meta.get('company')} | fy={meta.get('fiscal_year')} | doc_type={meta.get('doc_type')}\n{ch.page_content}"
        )
    context_str = "\n\n".join(passages)
    comp_note = (
        "- For a comparison, identify the requested fact for each company separately. Include passage indices supporting both sides when available.\n"
        if is_comparison
        else ""
    )
    ans_instr = (
        '- "answerable": true if these passages support every fact needed to answer for the requested company and year. '
        'Accept equivalent wording and prior-year columns, but do not substitute a different metric or period. '
        'Return false if any required fact is unsupported by these passages.\n'
        if check_answerable
        else '- "answerable": true\n'
    )
    prompt = (
        "You are a financial RAG reranker and evidence verifier.\n"
        f"Question: {question}\n\n"
        f"Candidate passages:\n{context_str}\n\n"
        "Respond with JSON ONLY:\n"
        '{"answerable": true/false, "order": [best_idx_1, best_idx_2, ...]}\n'
        "Rules:\n"
        f"{ans_instr}"
        f"{comp_note}"
        "- \"order\": up to 5 1-based passage indices ranked from best to worst.\n"
        "- Strongly prefer authoritative financial statements or note tables with exact figures over rounded prose summaries (e.g. Letter to shareholders), unless the question asks about an earnings call or research report.\n"
        "JSON:"
    )
    try:
        global _RERANK_LLM
        if _RERANK_LLM is None:
            _RERANK_LLM = make_llm(temperature=0.0, thinking=False, max_tokens=120)
        resp = _RERANK_LLM.invoke(prompt)
        obj = _extract_json_obj(str(resp.content))
        if obj is None:
            return None, []
        answerable = obj.get("answerable")
        if type(answerable) is not bool:
            answerable = None
        raw_order = obj.get("order")
        if not isinstance(raw_order, list):
            return answerable, []
        order = []
        seen = set()
        for item in raw_order:
            if type(item) is int and 1 <= item <= len(candidate_chunks) and item not in seen:
                seen.add(item)
                order.append(item)
        return (answerable, order)
    except Exception:
        # A transport or response-format failure is not evidence of absence.
        return None, []


def _select_retrieved(
    candidates: list[Document], order: list[int], covered_companies: list[str],
) -> list[Document]:
    """Keep the reranker's preferences, with at least one passage per comparison company."""
    preferred = list(dict.fromkeys([*(position - 1 for position in order), *range(len(candidates))]))
    selected = set(preferred[:MAX_RETRIEVED])
    company_keys = [_normalize_company_key(chunk.metadata.get("company", "")) for chunk in candidates]
    reserved = {
        next(i for i in preferred if company_keys[i] == key)
        for key in covered_companies
    }
    for missing in reserved.difference(selected):
        remove = next(i for i in reversed(preferred) if i in selected and i not in reserved)
        selected.remove(remove)
        selected.add(missing)
    return [candidates[i] for i in preferred if i in selected]


def retrieve(index: Any, question: str) -> list[Document]:
    """Return at most five original chunks, or [] when the corpus cannot answer."""
    features = _features()
    chunks: list[Document] = index["chunks"]
    if not chunks:
        return []

    parsed = parse_query(question)
    companies = parsed.get("companies") or []
    year = as_year(parsed.get("fiscal_year"))
    doc_type = parsed.get("doc_type") or "annual_report"
    company_keys = [_normalize_company_key(name) for name in companies]

    if "metadata_gate" in features:
        if not company_keys or any(
            not index["by_comp_doctype"].get((key, doc_type))
            or (year is not None and year not in index["covered_years"].get((key, doc_type), set()))
            for key in company_keys
        ):
            return []

    pools = {
        key: pool for key in company_keys
        if (pool := _select_target_indices(index, key, doc_type, year))
    }
    q_tokens = _query_tokens(question, companies, features)
    if not pools:
        # The naive baseline and non-abstaining ablations can rank all chunks.
        scores = index["bm25_global"].get_scores(q_tokens)
        indices = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
        return [chunks[i] for i in indices[:MAX_RETRIEVED]]

    is_comparison = len(pools) > 1 and "company_coverage" in features
    if is_comparison:
        rankings = [
            _score_candidate_indices(index, pool, question, q_tokens, features)
            for pool in pools.values()
        ]
        ranked = [i for row in zip_longest(*rankings) for i in row if i is not None]
    else:
        pool = [i for indices in pools.values() for i in indices]
        ranked = _score_candidate_indices(index, pool, question, q_tokens, features)

    if "rerank" not in features:
        return [chunks[i] for i in ranked[:MAX_RETRIEVED]]

    if is_comparison:
        initial = [i for row in zip_longest(*(r[:5] for r in rankings)) for i in row if i is not None]
        expanded = [i for row in zip_longest(*(r[:15] for r in rankings)) for i in row if i is not None]
    else:
        initial, expanded = ranked[:10], ranked[:24]

    candidates = [chunks[i] for i in initial]
    verify = "evidence_gate" in features
    answerable, order = _llm_rerank_and_verify(question, candidates, verify, is_comparison)
    if verify and answerable is False:
        if "widen_on_rejection" not in features or len(expanded) == len(initial):
            return []
        candidates = [chunks[i] for i in expanded]
        answerable, order = _llm_rerank_and_verify(question, candidates, True, is_comparison)
        if answerable is False:
            return []

    return _select_retrieved(candidates, order, list(pools)[:MAX_RETRIEVED] if is_comparison else [])


# =============================================================================
# Provided code below. Do not change it.
# =============================================================================

LLM_MODEL = "deepseek-v4-flash"
DOC_TYPES = ("annual_report", "earnings_call", "research_report")

MAX_CHUNK_CHARS = 1500  # characters per chunk
MAX_RETRIEVED = 5       # chunks per question
TIME_LIMIT_SECONDS = 30 * 60

# USD per 1M tokens, only for the cost estimate that the runner prints.
PRICE_LLM_INPUT, PRICE_LLM_OUTPUT = 0.15, 0.60

# Points (Section 2 of the handout).
POINTS = {"1.1": 8, "1.2": 7, "2.1": 5, "2.2": 10, "2.3": 15, "3": 10, "4": 20}
RECALL_ZERO, RECALL_FULL = 0.50, 0.90  # Task 2.3: 0 points at or below RECALL_ZERO, full points from RECALL_FULL
MAX_LINE_CHARS = 400                   # Task 1.1
MIN_PROBES_FOUND = 0.9                 # Task 1.2


class UsageTracker(BaseCallbackHandler):
    """Counts LLM tokens, for the cost estimate. There is no budget."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.llm_calls = self.llm_input_tokens = self.llm_output_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        tokens_in = tokens_out = 0
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                tokens_in += usage.get("input_tokens", 0)
                tokens_out += usage.get("output_tokens", 0)
        with self._lock:
            self.llm_calls += 1
            self.llm_input_tokens += tokens_in
            self.llm_output_tokens += tokens_out

    def cost(self) -> float:
        return (self.llm_input_tokens * PRICE_LLM_INPUT + self.llm_output_tokens * PRICE_LLM_OUTPUT) / 1e6

    def summary(self) -> dict:
        return {"llm_calls": self.llm_calls, "llm_input_tokens": self.llm_input_tokens,
                "llm_output_tokens": self.llm_output_tokens, "cost_usd": round(self.cost(), 4)}


USAGE = UsageTracker()            # your calls
GENERATOR_USAGE = UsageTracker()  # the generator's calls


def _chat(usage: UsageTracker, temperature: float, thinking: bool, max_tokens: int | None) -> ChatDeepSeek:
    return ChatDeepSeek(
        model=LLM_MODEL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=3,
        # Thinking mode ignores `temperature` and its tokens count as output.
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
        callbacks=[usage],
    )


def make_llm(temperature: float = 0.0, thinking: bool = False, max_tokens: int | None = None) -> ChatDeepSeek:
    """deepseek-v4-flash. Use it for every LLM call you make (only in parse_query, build_index and retrieve)."""
    return _chat(USAGE, temperature, thinking, max_tokens)


# ----------------------------------------------------------------------------- the generator

GENERATOR_PROMPT = """You answer questions about company reports, using only the context below.

Context:
{context}

Question: {question}

Rules:
- Use only the context. The header of each passage says which document it comes from.
- Check that the passage you use is about the company and the fiscal year that the question asks about.
- If the context does not contain the answer, reply exactly NOT_FOUND.
- For a number, reply with that one number only, in the unit the question asks for, without units, words or other numbers (for example: 1913.4).
- For a name (a person, a company, a city, an audit firm or a rating), reply with the name only.

Answer:"""

_generator: ChatDeepSeek | None = None


def format_context(chunks: list[Document]) -> str:
    """The retrieved chunks as the generator sees them: a header from the metadata, then the text."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata or {}
        header = " | ".join(f"{key}: {str(meta[key])[:100]}"
                            for key in ("source", "company", "fiscal_year", "doc_type") if meta.get(key) not in (None, ""))
        parts.append(f"[{i}] {header}\n{chunk.page_content}")
    return "\n\n".join(parts)


def generate(question: str, chunks: list[Document]) -> str:
    """The fixed generator: deepseek-v4-flash answers from the retrieved chunks only."""
    global _generator
    if not chunks:
        return "NOT_FOUND"
    if _generator is None:
        _generator = _chat(GENERATOR_USAGE, 0.0, False, 64)
    prompt = GENERATOR_PROMPT.format(context=format_context(chunks), question=question)
    return str(_generator.invoke(prompt).content).strip()


# ----------------------------------------------------------------------------- loaders

def load_corpus(folder: str | Path) -> dict[str, str]:
    """{file name: raw HTML} for every .html file in the folder."""
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(folder).glob("*.html"))}


def load_json(path: str | Path) -> Any:
    """The parsed file, or None if it does not exist (the private set has no answers)."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- scoring

# One number: optional minus sign, optional thousands separators, optional decimals.
NUMBER_RE = re.compile(r"(?<![\d.,])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
TITLES = {"mr", "mrs", "ms", "dr", "prof", "professor", "sir"}
LEGAL_SUFFIXES = {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co"}


def parse_number(text: str) -> float | None:
    """The single number in the text, or None if there is not exactly one."""
    found = NUMBER_RE.findall(text.replace("\u2212", "-"))
    if len(found) != 1:
        return None
    return float(found[0].replace(",", ""))


def normalize(text: str) -> str:
    """Lower case, no punctuation, no titles, padded with spaces for whole-word matching."""
    text = unicodedata.normalize("NFKC", str(text)).lower().replace("_", " ")
    words = [w for w in re.sub(r"[^a-z0-9]+", " ", text).split() if w not in TITLES]
    return " " + " ".join(words) + " "


def company_key(name: Any) -> str:
    """A company name without case, punctuation or legal suffix: 'Northgate Robotics Inc.' -> 'northgate robotics'."""
    return " ".join(w for w in normalize(name).split() if w not in LEGAL_SUFFIXES)


def squash(text: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().casefold()


def as_year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def answer_correct(answer: str, gold: dict) -> bool:
    answer = str(answer or "")
    said_not_found = " not found " in normalize(answer)
    if gold["type"] == "not_found":
        return said_not_found
    if said_not_found:
        return False
    if gold["type"] == "number":
        value = parse_number(answer)
        if value is None:
            return False
        return abs(value - gold["answer"]) <= max(0.01 * abs(gold["answer"]), 0.05)
    names = [gold["answer"], *gold.get("aliases", [])]
    if not any(normalize(name) in normalize(answer) for name in names):
        return False
    return not any(normalize(other) in normalize(answer) for other in gold.get("reject", []))


def table_score(output: Any, table: dict) -> float:
    """Task 1.1, one table: the share of cells found on a line with their row and column labels and
    without the label of another row, halved if the caption is missing."""
    if not isinstance(output, str) or not table["cells"]:
        return 0.0
    lines = [squash(line) for line in output.splitlines()]
    lines = [line for line in lines if line and len(line) <= MAX_LINE_CHARS]
    labels = {squash(row) for row, _, _ in table["cells"]}

    def kept(row: str, col: str, value: str) -> bool:
        row, col, value = squash(row), squash(col), squash(value)
        others = [o for o in labels if o not in row]  # e.g. "total liabilities" may be in "total liabilities and equity"
        return any(row in line and col in line and value in line and not any(o in line for o in others)
                   for line in lines)

    share = sum(kept(*cell) for cell in table["cells"]) / len(table["cells"])
    return share if squash(table["caption"]) in squash(output) else share / 2


def document_problems(chunks: Any, error: str | None, meta: dict, file_name: str) -> list[str]:
    """Task 1.2, one document: why its chunks fail the checks (an empty list means they pass)."""
    if error:
        return [f"error: {error}"]
    if not isinstance(chunks, list) or not chunks:
        return ["no chunks"]
    if not all(isinstance(c, Document) for c in chunks):
        return ["not a list of Document"]
    problems = []
    longest = max(len(c.page_content) for c in chunks)
    if longest > MAX_CHUNK_CHARS:
        problems.append(f"a chunk has {longest} characters")
    for key, ok in (("source", lambda v: v == file_name),
                    ("company", lambda v: company_key(v) == company_key(meta["company"])),
                    ("fiscal_year", lambda v: as_year(v) == meta["fiscal_year"]),
                    ("doc_type", lambda v: v == meta["doc_type"])):
        wrong = [c.metadata.get(key) for c in chunks if not ok(c.metadata.get(key))]
        if wrong:
            problems.append(f"{key} {wrong[0]!r} (expected {meta.get(key, file_name)!r}) on {len(wrong)} chunk(s)")
    text = squash("\n".join(c.page_content for c in chunks))
    missing = [p for p in meta["probes"] if squash(p) not in text]
    if len(missing) > (1 - MIN_PROBES_FOUND) * len(meta["probes"]):
        problems.append(f"text missing from the chunks: {missing[:3]}")
    return problems


def parse_score(parsed: Any, gold: dict) -> float:
    """Task 2.2, one question: the share of the three fields that are right."""
    if not isinstance(parsed, dict):
        return 0.0
    expected = gold["query"]
    companies = parsed.get("companies")
    if isinstance(companies, str):
        companies = [companies]
    right = 0
    if isinstance(companies, (list, tuple)) and {company_key(c) for c in companies} == {company_key(c) for c in expected["companies"]}:
        right += 1
    if "fiscal_year" in parsed and as_year(parsed["fiscal_year"]) == expected["fiscal_year"] and \
            (parsed["fiscal_year"] is None) == (expected["fiscal_year"] is None):
        right += 1
    if parsed.get("doc_type") == expected["doc_type"]:
        right += 1
    return right / 3


def contains(chunk_text: str, printed: str) -> bool:
    """The fact as printed in the document is in the chunk (a number only as a whole number)."""
    text, printed = squash(chunk_text), squash(printed)
    if re.fullmatch(r"[\d,.%-]+", printed):
        return re.search(rf"(?<![\d.,]){re.escape(printed)}(?![\d]|[.,]\d)", text) is not None
    return printed in text


def retrieval_hit(chunks: list[Document], gold: dict) -> bool:
    """For every fact the question needs, a retrieved chunk of a document that states it contains it."""
    return bool(gold["evidence"]) and all(
        any(c.metadata.get("source") == file and contains(c.page_content, printed)
            for c in chunks for file, printed in group)
        for group in gold["evidence"])


def clip(x: float) -> float:
    return max(0.0, min(1.0, x))


# ----------------------------------------------------------------------------- runner

def main() -> None:
    parser = argparse.ArgumentParser(description="FTEC5660 HW4: retrieval for RAG over company reports")
    parser.add_argument("--data-folder", default="public_test",
                        help="folder with corpus/, questions.json and (optionally) the answer files")
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    corpus = load_corpus(folder / "corpus")
    questions = load_json(folder / "questions.json")
    ground_truth = load_json(folder / "ground_truth.json")
    docs_meta = load_json(folder / "docs.json")
    tables = load_json(folder / "tables.json")
    print(f"{len(corpus)} documents, {len(questions)} questions")
    start = time.perf_counter()
    points: dict[str, float] = {}

    # Task 1.1: tables (no model calls allowed)
    if tables:
        before = USAGE.llm_calls
        scores = []
        for table in tables:
            try:
                scores.append(table_score(table_to_text(table["html"]), table))
            except Exception as e:
                print(f"table_to_text failed on a table of {table['source']}: {e!r}")
                scores.append(0.0)
        if USAGE.llm_calls != before:
            print("WARNING: table_to_text called the LLM, which is not allowed (Task 1.1 = 0).")
            scores = [0.0]
        points["1.1"] = POINTS["1.1"] * sum(scores) / len(scores)
        print(f"\nTask 1.1 table_to_text: {sum(scores) / len(scores):.1%} of the table cells kept with their labels")

    # Task 1.2: chunks (no model calls allowed)
    before = USAGE.llm_calls
    chunks_by_file: dict[str, list] = {}
    errors: dict[str, str] = {}
    for file_name, html in corpus.items():
        try:
            chunks_by_file[file_name] = list(chunk_document(file_name, html) or [])
        except Exception as e:
            traceback.print_exc()
            chunks_by_file[file_name], errors[file_name] = [], repr(e)
    called_models = USAGE.llm_calls != before
    if called_models:
        print("WARNING: chunk_document called the LLM, which is not allowed (Task 1.2 = 0).")
    if docs_meta:
        problems = {f: document_problems(chunks_by_file.get(f), errors.get(f), m, f) for f, m in docs_meta.items()}
        passed = sum(not p for p in problems.values()) / len(problems)
        points["1.2"] = 0.0 if called_models else POINTS["1.2"] * passed
        print(f"Task 1.2 chunk_document: {passed:.1%} of the documents pass")
        for f, p in [(f, p) for f, p in problems.items() if p][:5]:
            print(f"  {f}: {'; '.join(p)}")

    # the chunks that retrieve() may return
    all_chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]
                  if isinstance(c, Document) and len(c.page_content) <= MAX_CHUNK_CHARS]
    by_text: dict[str, list[Document]] = {}
    for c in all_chunks:
        by_text.setdefault(c.page_content, []).append(c)
    print(f"{len(all_chunks)} chunks to index")

    # Task 2.2: query parsing
    parsed: dict[str, Any] = {}
    for qid, question in questions.items():
        try:
            parsed[qid] = parse_query(question)
        except Exception as e:
            parsed[qid] = f"error: {e!r}"

    # Tasks 2.1, 2.3 and 3: retrieval
    index = build_index(all_chunks)
    index_seconds = time.perf_counter() - start
    print(f"Index built in {index_seconds:.1f}s | {USAGE.summary()}")
    retrieved: dict[str, list[Document]] = {}
    valid: dict[str, bool] = {}
    for qid, question in questions.items():
        try:
            result = retrieve(index, question)
            ok = isinstance(result, (list, tuple)) and len(result) <= MAX_RETRIEVED
            kept = []
            for chunk in list(result or []) if isinstance(result, (list, tuple)) else []:
                matches = by_text.get(getattr(chunk, "page_content", None), [])
                if not matches:
                    ok = False
                    continue
                source = (getattr(chunk, "metadata", None) or {}).get("source")
                kept.append(next((m for m in matches if m.metadata.get("source") == source), matches[0]))
            retrieved[qid], valid[qid] = kept[:MAX_RETRIEVED], ok
        except Exception:
            traceback.print_exc()
            retrieved[qid], valid[qid] = [], False
    retrieval_seconds = time.perf_counter() - start - index_seconds

    # Task 4: the fixed generator
    def answer(qid: str) -> str:
        try:
            return generate(questions[qid], retrieved[qid])
        except Exception as e:
            print(f"Generator failed on {qid}: {e!r}")
            return ""

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = dict(zip(questions, pool.map(answer, questions)))
    total_seconds = time.perf_counter() - start

    rows = []
    for qid, question in questions.items():
        sources = sorted({c.metadata.get("source", "") for c in retrieved[qid]})
        row = {"id": qid, "category": "", "question": question, "parsed": json.dumps(parsed[qid], default=str),
               "parse_score": "", "retrieved": ";".join(sources), "valid": valid[qid], "retrieval_hit": "",
               "answer": answers[qid], "correct": "", "expected": ""}
        if ground_truth and qid in ground_truth:
            gold = ground_truth[qid]
            row["category"] = gold["category"]
            row["parse_score"] = round(parse_score(parsed[qid], gold), 3)
            row["retrieval_hit"] = retrieval_hit(retrieved[qid], gold) if gold["evidence"] else ""
            row["correct"] = answer_correct(answers[qid], gold)
            if not row["correct"]:
                row["expected"] = gold["answer"]
        rows.append(row)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")

    if ground_truth:
        graded = [r for r in rows if r["id"] in ground_truth]
        answerable = [r for r in graded if ground_truth[r["id"]]["evidence"]]
        unanswerable = [r for r in graded if not ground_truth[r["id"]]["evidence"]]
        parse_mean = sum(r["parse_score"] for r in graded) / len(graded)
        non_empty = sum(r["valid"] and bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        recall = sum(r["retrieval_hit"] for r in answerable) / max(len(answerable), 1)
        kept = sum(bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        rejected = sum(not r["retrieved"] for r in unanswerable) / max(len(unanswerable), 1)
        accuracy = sum(r["correct"] for r in graded) / len(graded)
        points["2.1"] = POINTS["2.1"] * non_empty
        points["2.2"] = POINTS["2.2"] * parse_mean
        points["2.3"] = POINTS["2.3"] * clip((recall - RECALL_ZERO) / (RECALL_FULL - RECALL_ZERO))
        points["3"] = POINTS["3"] * clip(kept + rejected - 1)
        points["4"] = POINTS["4"] * accuracy
        print(f"Task 2.1 valid, non-empty retrieval on answerable questions: {non_empty:.1%}")
        print(f"Task 2.2 parse_query fields right: {parse_mean:.1%}")
        print(f"Task 2.3 retrieval recall@{MAX_RETRIEVED} on answerable questions: {recall:.1%}")
        print(f"Task 3   answerable questions with chunks: {kept:.1%} | unanswerable questions with []: {rejected:.1%}")
        print(f"Task 4   answer accuracy: {accuracy:.1%}")
        by_category: dict[str, list[tuple]] = {}
        for r in graded:
            by_category.setdefault(r["category"], []).append((r["correct"], r["retrieval_hit"]))
        print("\nBy category       accuracy   recall")
        for category, results in sorted(by_category.items()):
            hits = [h for _, h in results if h != ""]
            recall_text = f"{sum(hits) / len(hits):.1%}" if hits else "-"
            print(f"  {category:<14}{sum(c for c, _ in results) / len(results):>9.1%}{recall_text:>9}")

    if points:
        print("\nEstimated points (the grade uses the private set):")
        for task in POINTS:
            if task in points:
                print(f"  Task {task:<4}{points[task]:6.2f} / {POINTS[task]}")
        print(f"  Total    {sum(points.values()):6.2f} / {sum(POINTS[t] for t in points)}")

    print(f"\nYour usage: {USAGE.summary()}")
    print(f"Generator usage: {GENERATOR_USAGE.summary()}")
    print(f"Time: {total_seconds:.1f}s (index {index_seconds:.1f}s, retrieval {retrieval_seconds:.1f}s)")
    if total_seconds > TIME_LIMIT_SECONDS:
        print(f"WARNING: the run took longer than the {TIME_LIMIT_SECONDS // 60}-minute limit.")


if __name__ == "__main__":
    main()
