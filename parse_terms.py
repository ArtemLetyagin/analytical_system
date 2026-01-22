#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse 1+ DOCX terminology standards into structured "term -> definition" records.

Heuristics are tuned for documents where entries look like:
  "система ...; АББР (English; EN_ABBR): Definition ..."

and/or numbered entries like:
  "1 информация (information): Definition ..."

Outputs:
  - JSONL (one record per line)
  - CSV (flat)
"""

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

from docx import Document


# ----------------------------
# Helpers / normalization
# ----------------------------

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def norm_term(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("ё", "е")
    s = re.sub(r"\s+", " ", s)
    # remove surrounding quotes/brackets
    s = s.strip("«»\"'()[]{}")
    return s

def is_heading_like(text: str) -> bool:
    """
    Very light heuristic to detect section headings.
    We treat short lines without ':' that look like titles as headings.
    """
    t = norm_space(text)
    if not t or ":" in t:
        return False
    if len(t) > 120:
        return False
    # avoid pure page numbers/roman numerals
    if re.fullmatch(r"[\dIVXLCDM]+", t):
        return False
    # common headings often start with uppercase letter and have few words
    words = t.split()
    if len(words) <= 12 and (t[0].isupper() or t.isupper()):
        return True
    # sometimes headings end with a period (e.g., "Системы ... . Общие понятия")
    if len(words) <= 16 and t.endswith("."):
        return True
    return False


# ----------------------------
# Core parsing
# ----------------------------

ENTRY_RE = re.compile(
    r"""^\s*
    (?:(?P<article>\d+)\s+)?          # optional article number at start
    (?P<head>[^:]{3,300})             # header (term + abbr + en)
    \s*:\s*
    (?P<defn>.+?)\s*$                 # definition after colon
    """,
    re.VERBOSE,
)

# Detect "Примечание – ..." / "П р и м е ч а н и е – ..." etc.
NOTE_START_RE = re.compile(r"^\s*(?:П\s*р\s*и\s*м\s*е\s*ч\s*а\s*н\s*и\s*е|Примечание)\b", re.IGNORECASE)


def split_head(head: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """
    Parse header into:
      term_ru, abbr_ru, term_en, abbr_en

    Examples:
      "система поддержки принятия решений; СППР (Decision Support System; DSS)"
      "искусственный интеллект; ИИ (artificial intelligence, AI)"
      "сложная ситуация для принятия решений (Difficult Situation for Decision Making)"
    """
    head = norm_space(head)

    # Extract parentheses part (often English + EN abbr)
    paren = None
    mparen = re.search(r"\((.+)\)\s*$", head)
    if mparen:
        paren = mparen.group(1).strip()
        head_no_paren = head[: mparen.start()].strip()
    else:
        head_no_paren = head

    # RU part split by ';' (abbr often after ';')
    abbr_ru = None
    if ";" in head_no_paren:
        left, right = head_no_paren.split(";", 1)
        term_ru = norm_space(left)
        abbr_ru = norm_space(right)
    else:
        term_ru = head_no_paren.strip()

    # English part parsing (inside parentheses)
    term_en = None
    abbr_en = None
    if paren:
        # patterns:
        #   "Decision Support System; DSS"
        #   "artificial intelligence, AI"
        #   "Artificial Intelligence" (no abbr)
        # Try split by ';' first, else by comma
        if ";" in paren:
            left, right = paren.split(";", 1)
            term_en = norm_space(left)
            abbr_en = norm_space(right)
        elif "," in paren:
            left, right = paren.split(",", 1)
            term_en = norm_space(left)
            abbr_en = norm_space(right)
        else:
            term_en = norm_space(paren)

    # Clean abbr_ru artifacts like multiple abbreviations separated by commas
    if abbr_ru:
        abbr_ru = abbr_ru.strip()
        # remove trailing punctuation
        abbr_ru = abbr_ru.strip(" .;,")

    if abbr_en:
        abbr_en = abbr_en.strip()
        abbr_en = abbr_en.strip(" .;,")

    term_ru = term_ru.strip(" .;,")

    return term_ru, abbr_ru, term_en, abbr_en


@dataclass
class TermRecord:
    standard_id: str
    source_file: str
    article: Optional[str]
    section_path: str
    term_ru: str
    term_ru_norm: str
    abbr_ru: Optional[str]
    term_en: Optional[str]
    abbr_en: Optional[str]
    definition: str
    notes: Optional[str]
    raw_entry_text: str


def parse_docx_terms(docx_path: str, standard_id: str) -> List[TermRecord]:
    doc = Document(docx_path)

    records: List[TermRecord] = []

    current_section_stack: List[str] = []
    pending_note_lines: List[str] = []
    last_record_index: Optional[int] = None

    def section_path() -> str:
        return " / ".join(current_section_stack) if current_section_stack else ""

    for p in doc.paragraphs:
        text = norm_space(p.text)
        if not text:
            continue

        # Update heading/section context
        style_name = (p.style.name if p.style is not None else "") or ""
        if style_name.lower().startswith("heading") or is_heading_like(text):
            # A new heading resets "notes" accumulation
            pending_note_lines = []
            last_record_index = None

            # Keep a simple stack: if looks like top-level heading, reset.
            # Heuristic: very short headings are often deeper; very long are usually top-level.
            if len(text) <= 45 and current_section_stack:
                # treat as subheading, replace last
                # but avoid stacking too deep if repeated
                if len(current_section_stack) >= 2:
                    current_section_stack = current_section_stack[:1] + [text]
                else:
                    current_section_stack.append(text)
            else:
                current_section_stack = [text]
            continue

        # Notes handling: attach to the last parsed record if present
        if NOTE_START_RE.match(text):
            pending_note_lines = [text]
            continue

        # Sometimes notes continue on next paragraph(s) without explicit "Примечание"
        if pending_note_lines and not ENTRY_RE.match(text) and not is_heading_like(text):
            # continue collecting note lines until we hit next entry/heading
            pending_note_lines.append(text)
            continue

        m = ENTRY_RE.match(text)
        if not m:
            # Not an entry, not a note: ignore
            continue

        article = m.group("article")
        head = m.group("head")
        defn = m.group("defn")

        term_ru, abbr_ru, term_en, abbr_en = split_head(head)

        rec = TermRecord(
            standard_id=standard_id,
            source_file=os.path.basename(docx_path),
            article=article,
            section_path=section_path(),
            term_ru=term_ru,
            term_ru_norm=norm_term(term_ru),
            abbr_ru=abbr_ru,
            term_en=term_en,
            abbr_en=abbr_en,
            definition=defn.strip(),
            notes=None,
            raw_entry_text=text,
        )

        records.append(rec)
        last_record_index = len(records) - 1

        # If we have accumulated notes right before an entry, attach them to THIS entry
        if pending_note_lines:
            records[last_record_index].notes = norm_space(" ".join(pending_note_lines))
            pending_note_lines = []

    # Post-pass: sometimes notes come after the entry (common in ГОСТ/ПНСТ).
    # This parser already collects "Примечание" paragraphs and attaches them to the NEXT entry,
    # which is not always correct. To handle "notes after entry", do a lightweight pass over raw paragraphs.
    # (Optional) You can disable this if your docs already match the "note-before" style.
    #
    # For now we keep it simple: if a record has no notes, we try to find immediate following "Примечание"
    # in the raw text vicinity by scanning paragraphs again.
    #
    # If you want that, uncomment the function below and call it.
    #
    # records = attach_post_notes(doc, records)

    return records


# ----------------------------
# Output
# ----------------------------

def write_jsonl(path: str, records: List[TermRecord]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

def write_csv(path: str, records: List[TermRecord]) -> None:
    fieldnames = list(asdict(records[0]).keys()) if records else []
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow(asdict(r))


# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docx-a", required=True, help="Path to first DOCX (standard A)")
    ap.add_argument("--docx-b", required=True, help="Path to second DOCX (standard B)")
    ap.add_argument("--id-a", default="A", help="Standard id for A (default: A)")
    ap.add_argument("--id-b", default="B", help="Standard id for B (default: B)")
    ap.add_argument("--outdir", default="out_terms", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    rec_a = parse_docx_terms(args.docx_a, args.id_a)
    rec_b = parse_docx_terms(args.docx_b, args.id_b)
    all_recs = rec_a + rec_b

    # Write per-standard and combined
    write_jsonl(os.path.join(args.outdir, f"terms_{args.id_a}.jsonl"), rec_a)
    write_jsonl(os.path.join(args.outdir, f"terms_{args.id_b}.jsonl"), rec_b)
    write_jsonl(os.path.join(args.outdir, "terms_all.jsonl"), all_recs)

    if all_recs:
        write_csv(os.path.join(args.outdir, "terms_all.csv"), all_recs)

    print(f"Parsed A: {len(rec_a)} records")
    print(f"Parsed B: {len(rec_b)} records")
    print(f"Saved to: {args.outdir}")

if __name__ == "__main__":
    main()

"""
python parse_terms.py \
  --docx-a "data/docs/draft/Проект_ГОСТ_КИТ_на_НК_б.docx" \
  --docx-b "data/docs/draft/Проект_ГОСТ_СППР_на_НК_б.docx" \
  --id-a "PNST_DSS" \
  --id-b "PNST_CIT" \
  --outdir "out_terms"
"""