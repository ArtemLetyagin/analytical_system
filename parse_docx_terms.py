import re
import json
from pathlib import Path
from typing import Iterator, Union, List, Tuple, Optional, Dict

from docx import Document
from docx.document import Document as _Document
from docx.text.paragraph import Paragraph
from docx.table import Table, _Cell
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl

def get_num_props(p: Paragraph) -> Optional[Tuple[int, int]]:
    """
    Возвращает (numId, ilvl) если абзац — элемент нумерованного списка Word.
    """
    pPr = p._p.pPr
    if pPr is None or pPr.numPr is None:
        return None
    numId_elm = pPr.numPr.numId
    ilvl_elm = pPr.numPr.ilvl
    if numId_elm is None or ilvl_elm is None:
        return None
    try:
        numId = int(numId_elm.val)
        ilvl = int(ilvl_elm.val)
        return numId, ilvl
    except Exception:
        return None
def numbering_prefix(p: Paragraph, counters: Dict[Tuple[int, int], int]) -> str:
    """
    Возвращает строку-префикс номера ("8 " или "8. "), если номер есть как автонумерация.
    """
    props = get_num_props(p)
    if not props:
        return ""
    key = props
    counters[key] = counters.get(key, 0) + 1
    # формат можно менять: "8 " / "8." / "8. "
    return f"{counters[key]} "
# --- Регексы статьи термина ---
# "12 Термин : Определение" + допускаем многострочное определение
ENTRY_INLINE_RE = re.compile(r"^\s*(\d+)\s*\.?\s*(.+?)\s*[:：]\s*(.*)\s*$", re.DOTALL)

# маркер начала новой статьи ВНУТРИ строки
# ищем места вида: (начало или перевод строки) + число + пробел + что-то + двоеточие
ENTRY_START_FINDER = re.compile(
    r"(?=(^|\n)\s*\d+\s*\.?\s*.+?[:：])",
    re.DOTALL
)
NUMBER_ONLY_RE = re.compile(r"^\s*(\d+)\s*$")

# --- Разделение "термин_рус (term_en)" ---
TERM_RU_EN_RE = re.compile(
    r"^\s*(?P<ru>.+?)\s*\(\s*(?P<en>[A-Za-z][A-Za-z0-9\s\-/–—']*)\s*\)\s*$"
)

def split_term_ru_en(term: str) -> Tuple[str, Optional[str]]:
    term = (term or "").strip()
    m = TERM_RU_EN_RE.match(term)
    if not m:
        return term, None
    ru = m.group("ru").strip()
    en = re.sub(r"\s+", " ", m.group("en").strip())
    return ru, en or None


# --- Итератор по блокам документа (Paragraph/Table) в исходном порядке ---
def iter_block_items(parent: Union[_Document, _Cell]) -> Iterator[Union[Paragraph, Table]]:
    """
    Yield each paragraph and table child within *parent*, in document order.
    parent: docx.Document или docx.table._Cell
    """
    if isinstance(parent, _Document):
        parent_elm = parent.element.body
    else:
        parent_elm = parent._tc

    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def cell_text(cell: _Cell) -> str:
    """Текст ячейки, сохраняя переносы между абзацами."""
    parts = []
    for p in cell.paragraphs:
        t = (p.text or "").strip()
        if t:
            parts.append(t)
    return "\n".join(parts).strip()


# --- Шаг 1: превращаем docx в линейный список строк (из абзацев и таблиц) ---
def doc_to_lines(doc: Document):
    lines = []
    list_counters = {}  # (numId, ilvl) -> current_number

    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            text = (block.text or "").strip()
            if text:
                pref = numbering_prefix(block, list_counters)

                # если в тексте уже есть явный "8 ..." — не дублируем
                if pref and re.match(r"^\s*\d+\b", text):
                    pref = ""

                lines.append(("P", (pref + text).strip()))

        elif isinstance(block, Table):
            for row in block.rows:
                cells = [cell_text(c) for c in row.cells]
                cells = [c for c in cells if c]
                for c in cells:
                    lines.append(("T", c))

    return lines
# --- Шаг 2: парсим линейные строки в {number, term_ru, definition} ---
def normalize_for_parsing(text: str) -> str:
    # NBSP -> space, унификация переносов
    text = (text or "").replace("\xa0", " ")
    # убираем мусорные пробелы вокруг переносов
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    # схлопываем множественные пробелы (но переносы оставляем)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def split_into_entry_chunks(text: str) -> List[str]:
    """
    Если в одной строке/ячейке склеились несколько терминов (8 ... 9 ...),
    режем на куски по каждому старту статьи.
    """
    text = normalize_for_parsing(text)
    if not text:
        return []

    # Находим все старты статей
    starts = []
    for m in ENTRY_START_FINDER.finditer(text):
        # позиция фактического старта — начало захвата (^|\n) + ...
        # m.start() у lookahead — позиция совпадения (0..), ок
        starts.append(m.start())

    # Если старт только один (или ноль) — ничего резать
    # Ноль может быть если двоеточие отсутствует — тогда отдаём целиком (как продолжение определения)
    starts = sorted(set(starts))
    if len(starts) <= 1:
        return [text]

    # Режем по индексам стартов
    chunks = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(text)
        chunk = text[s:e].strip()
        if chunk:
            # если кусок начинается с '\n', уберём
            chunks.append(chunk.lstrip("\n").strip())
    return chunks


def parse_terms_from_lines(lines: List[Tuple[str, str]], source_file):
    entries = []
    current = None  # {"number": int, "term_ru": str, "definition": str}
    pending_number: Optional[str] = None

    def flush_current():
        nonlocal current
        if not current:
            return
        current["term_ru"] = re.sub(r"\s+", " ", current["term_ru"]).strip()

        if current.get("term_en"):
            current["term_en"] = re.sub(r"\s+", " ", current["term_en"]).strip()

        current["definition"] = re.sub(r"[ \t]+", " ", current["definition"]).strip()
        current["definition"] = re.sub(r"\n{3,}", "\n\n", current["definition"]).strip()

        # term_en НЕ обязателен
        if current.get("number") is not None and current["term_ru"] and current["definition"]:
            entries.append(current)
        current = None


    def start_entry(number: int, term: str, definition_start: str):
        nonlocal current
        flush_current()

        term_ru, term_en = split_term_ru_en(term)

        current = {
            "number": int(number),
            "term_ru": term_ru.strip(),
            "term_en": term_en,  # может быть None
            "definition": (definition_start or "").strip(),
            "source_file": source_file
        }

    for source, raw in lines:
        raw_text = normalize_for_parsing(raw)
        if not raw_text:
            if current and current["definition"]:
                current["definition"] += "\n"
            continue

        # ВАЖНО: сначала режем склеенные строки на чанки
        chunks = split_into_entry_chunks(raw_text)

        for text in chunks:
            # Для матчинга “в линию” превращаем внутренние переносы в пробелы
            text_for_match = re.sub(r"\s*\n\s*", " ", text).strip()

            # номер отдельной строкой (редко, но оставим)
            m_num = NUMBER_ONLY_RE.match(text_for_match)
            if m_num:
                pending_number = m_num.group(1)
                continue

            m = ENTRY_INLINE_RE.match(text_for_match)

            if not m and pending_number is not None:
                m = ENTRY_INLINE_RE.match(f"{pending_number} {text_for_match}")
                pending_number = None

            if m:
                num = int(m.group(1))
                term = m.group(2)
                definition_start = m.group(3)
                start_entry(num, term, definition_start)
                continue

            # иначе — продолжение определения (сохраняем переносы как есть)
            if current:
                if current["definition"]:
                    current["definition"] += "\n" + text.strip()
                else:
                    current["definition"] = text.strip()
            else:
                # вне термина
                continue

    flush_current()
    return entries


def save_jsonl(entries, out_path: str):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for obj in entries:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def parse_docx_terms(input_docx, output_jsonl):
    
    doc = Document(input_docx)
    lines = doc_to_lines(doc)

    # (опционально) дебаг: посмотреть что именно читается из блоков
    # for src, line in lines[:200]:
    #     print(f"{src}: {line}")

    source_file = input_docx.split("/")[-1]

    terms = parse_terms_from_lines(lines, source_file)
    save_jsonl(terms, output_jsonl)

    print(f"Extracted: {len(terms)} terms")
    print(f"Saved to: {output_jsonl}")

if __name__ == "__main__":
    input_docx = "data/docs/draft/Проект_ГОСТ_СППР_на_НК_б.docx"
    input_docx = "data/docs/draft/Проект_ГОСТ_КИТ_на_НК_б.docx"
    output_jsonl = "terms.jsonl"

    parse_docx_terms(
        input_docx=input_docx,
        output_jsonl=output_jsonl
    )
