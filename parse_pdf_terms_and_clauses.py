#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract TWO kinds of records from a PDF (global scan):

1) TERMS (glossary-like):
   - "<n.n[.n]> <term>: <definition>"
   - or split start:
       "<n.n[.n]>"
       "<term>: <definition>"

2) CLAUSES / PARAGRAPHS (normative text):
   - "<n.n[.n]> <sentence...>"  (NO colon after the head)
   - Multiline continuation until next numbered start

Outputs:
  --out-terms: JSONL of terms
  --out-clauses: JSONL of numbered clauses

No section logic. No notes parsing.

Usage:
  python parse_pdf_terms_and_clauses.py \
      --pdf /path/to/file.pdf \
      --standard-id STD_ID \
      --out-terms terms.jsonl \
      --out-clauses clauses.jsonl
"""

import re
import json
import argparse
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict, Any

from pypdf import PdfReader


# ----------------------------
# Regex / heuristics
# ----------------------------

# Article number: 1.1, 1.1.1, 2.3.7, etc.
ARTICLE_ONLY_RE = re.compile(r"^(?P<article>\d+(?:\.\d+)+)\s*$")

# Term: "<article> <head>: <defn>"
TERM_START_SAME_LINE_RE = re.compile(
    r"^(?P<article>\d+(?:\.\d+)+)\s+(?P<head>.{2,400}?)\s*:\s*(?P<defn>.+?)\s*$"
)

# Term start without article (used when article was on previous line): "<head>: <defn>"
TERM_START_NO_ARTICLE_RE = re.compile(
    r"^(?P<head>.{2,400}?)\s*:\s*(?P<defn>.+?)\s*$"
)

# Clause / paragraph: "<article> <text...>" WITHOUT ":" close to start
# We intentionally do NOT require the text to end with '.' because it may continue.
CLAUSE_START_RE = re.compile(
    r"^(?P<article>\d+(?:\.\d+)+)\s+(?P<body>.+?)\s*$"
)

# Noise filters (light)
TOC_LIKE_RE = re.compile(r"\.{3,}\s*\d+\s*$")  # ".... 12"
PAGE_NUM_RE = re.compile(r"^\s*\d+\s*$")
BORING_PHRASES = ("издание официальное", "официальное издание")

# Hyphenation: soft hyphen and end-line '-'
SOFT_HYPHEN = "\u00ad"


def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def looks_like_noise(line: str) -> bool:
    t = line.strip()
    if not t:
        return True
    if PAGE_NUM_RE.match(t):
        return True
    if TOC_LIKE_RE.search(t):
        return True
    low = t.lower()
    if any(p in low for p in BORING_PHRASES):
        return True
    return False


def split_head_ru_en(head: str) -> Tuple[str, Optional[str]]:
    """
    Extract RU term + EN equivalent if the LAST parentheses contains Latin letters.
    """
    head = norm_space(head)
    parens = re.findall(r"\(([^()]*)\)", head)
    term_en = None
    if parens:
        last = parens[-1].strip()
        if re.search(r"[A-Za-z]", last):
            term_en = last

    if term_en:
        head_wo_last = re.sub(r"\([^()]*\)\s*$", "", head).strip()
        term_ru = head_wo_last
    else:
        term_ru = head

    return term_ru, term_en


def pdf_to_lines(pdf_path: str) -> List[str]:
    reader = PdfReader(pdf_path)
    out: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        for ln in text.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            if looks_like_noise(ln):
                continue
            out.append(ln)
    return out


def smart_append(buf: List[str], next_line: str) -> None:
    """
    Append next_line into buf with hyphenation handling.
    - If previous ends with soft hyphen or '-', join without space.
    - Else join as separate piece (later normalized).
    """
    if not buf:
        buf.append(next_line)
        return

    prev = buf[-1]
    if prev.endswith(SOFT_HYPHEN):
        buf[-1] = prev[:-1] + next_line
    elif prev.endswith("-") and len(prev) >= 2 and prev[-2].isalpha():
        buf[-1] = prev[:-1] + next_line
    else:
        buf.append(next_line)


# ----------------------------
# Output records
# ----------------------------

@dataclass
class TermRecord:
    standard_id: str
    source_file: str
    article: str
    term_ru: str
    term_en: Optional[str]
    definition: str
    raw_entry_start: str


@dataclass
class ClauseRecord:
    standard_id: str
    source_file: str
    article: str
    text: str
    raw_entry_start: str


# ----------------------------
# Main parsing logic
# ----------------------------

def parse_terms_and_clauses_global(pdf_path: str, standard_id: str) -> Tuple[List[TermRecord], List[ClauseRecord]]:
    lines = pdf_to_lines(pdf_path)
    source_file = pdf_path.split("/")[-1]

    terms: List[TermRecord] = []
    clauses: List[ClauseRecord] = []

    # Pending article when it's alone on its own line
    pending_article: Optional[str] = None

    # Current TERM state
    cur_term_article: Optional[str] = None
    cur_term_head: Optional[str] = None
    cur_term_start_line: Optional[str] = None
    cur_term_def_buf: List[str] = []

    # Current CLAUSE state
    cur_clause_article: Optional[str] = None
    cur_clause_start_line: Optional[str] = None
    cur_clause_buf: List[str] = []

    def flush_term():
        nonlocal cur_term_article, cur_term_head, cur_term_start_line, cur_term_def_buf
        if cur_term_article and cur_term_head and cur_term_def_buf:
            term_ru, term_en = split_head_ru_en(cur_term_head)
            definition = norm_space(" ".join(cur_term_def_buf)).replace(SOFT_HYPHEN, "").strip()
            terms.append(
                TermRecord(
                    standard_id=standard_id,
                    source_file=source_file,
                    article=cur_term_article,
                    term_ru=term_ru,
                    term_en=term_en,
                    definition=definition,
                    raw_entry_start=cur_term_start_line or "",
                )
            )
        cur_term_article = None
        cur_term_head = None
        cur_term_start_line = None
        cur_term_def_buf = []

    def flush_clause():
        nonlocal cur_clause_article, cur_clause_start_line, cur_clause_buf
        if cur_clause_article and cur_clause_buf:
            text = norm_space(" ".join(cur_clause_buf)).replace(SOFT_HYPHEN, "").strip()
            clauses.append(
                ClauseRecord(
                    standard_id=standard_id,
                    source_file=source_file,
                    article=cur_clause_article,
                    text=text,
                    raw_entry_start=cur_clause_start_line or "",
                )
            )
        cur_clause_article = None
        cur_clause_start_line = None
        cur_clause_buf = []

    for raw in lines:
        line = norm_space(raw)

        # 1) Article alone: "3.1"
        m_art_only = ARTICLE_ONLY_RE.match(line)
        if m_art_only:
            # A naked article likely starts a new entity => flush any open one
            flush_term()
            flush_clause()
            pending_article = m_art_only.group("article")
            continue

        # 2) Term starts in same line: "<article> <head>: <defn>"
        m_term_same = TERM_START_SAME_LINE_RE.match(line)
        if m_term_same:
            # new entity => flush others
            flush_term()
            flush_clause()
            pending_article = None

            cur_term_article = m_term_same.group("article")
            cur_term_head = m_term_same.group("head").strip()
            cur_term_start_line = line
            cur_term_def_buf = [m_term_same.group("defn").strip()]
            continue

        # 3) If we had pending_article, try term start without article: "<head>: <defn>"
        if pending_article is not None:
            m_term_no = TERM_START_NO_ARTICLE_RE.match(line)
            if m_term_no:
                flush_term()
                flush_clause()

                cur_term_article = pending_article
                pending_article = None
                cur_term_head = m_term_no.group("head").strip()
                cur_term_start_line = f"{cur_term_article} (split) + {line}"
                cur_term_def_buf = [m_term_no.group("defn").strip()]
                continue

            # If not a term, try clause start after a pending article:
            # example:
            #   "5.2"
            #   "Дистанционное наблюдение ..."
            # We'll treat the whole line as the body of that pending_article clause.
            flush_term()
            flush_clause()
            cur_clause_article = pending_article
            cur_clause_start_line = f"{cur_clause_article} (split) + {line}"
            pending_article = None
            cur_clause_buf = [line]
            continue

        # 4) Clause starts in same line: "<article> <body>" BUT avoid lines that are actually term starts
        # We already tested term-start above, so here ":" can still appear in the middle of text.
        # To avoid misclassifying terms, require that there is NO ":" early in the line.
        m_clause = CLAUSE_START_RE.match(line)
        if m_clause:
            art = m_clause.group("article")
            body = m_clause.group("body").strip()

            # If the body contains ":" very early, it's likely a term entry we missed; skip clause start.
            # (You can tune 60 if needed.)
            early_colon = (":" in body[:60])

            if not early_colon:
                flush_term()
                flush_clause()

                cur_clause_article = art
                cur_clause_start_line = line
                cur_clause_buf = [body]
                continue

        # 5) Continuation lines: attach to whichever entity is open
        if cur_term_article is not None:
            smart_append(cur_term_def_buf, line)
            continue

        if cur_clause_article is not None:
            smart_append(cur_clause_buf, line)
            continue

        # 6) Otherwise ignore line

    # End flush
    flush_term()
    flush_clause()

    return terms, clauses


# ----------------------------
# CLI
# ----------------------------

def write_jsonl(path: str, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to PDF")
    ap.add_argument("--standard-id", default="STD_PDF", help="Identifier for this standard")
    ap.add_argument("--out-terms", default="terms.jsonl", help="Output JSONL path for TERMS")
    ap.add_argument("--out-clauses", default="clauses.jsonl", help="Output JSONL path for numbered CLAUSES")
    args = ap.parse_args()

    terms, clauses = parse_terms_and_clauses_global(args.pdf, args.standard_id)

    write_jsonl(args.out_terms, terms)
    write_jsonl(args.out_clauses, clauses)

    print(f"Found {len(terms)} terms -> {args.out_terms}")
    print(f"Found {len(clauses)} clauses -> {args.out_clauses}")


if __name__ == "__main__":
    main()
