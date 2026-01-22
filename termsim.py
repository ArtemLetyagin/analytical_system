from __future__ import annotations

import re
import difflib
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# ----------------------------
# Normalization & light stemming (no external deps)
# ----------------------------

_RU_ENDINGS = (
    # грубая нормализация окончаний для терминов
    "ами","ями","ого","его","ому","ему","ыми","ими",
    "ая","яя","ое","ее","ые","ие",
    "ой","ей","ом","ем",
    "ам","ям","ах","ях",
    "ов","ев","ей",
    "а","я","о","е","ы","и","у","ю",
)

_STOPWORDS = {
    "и","или","в","во","на","по","к","ко","с","со","для","при","об","о","от","до",
    "как","а","но","что","это","то","та","те","тд","тп"
}

_PAREN_EN_RE = re.compile(r"^(?P<ru>.*?)\s*\((?P<inside>[A-Za-z][^()]*)\)\s*$")


def norm_ru(s: str) -> str:
    s = (s or "").strip().lower().replace("ё", "е")
    s = s.strip("«»\"'()[]{}")
    # оставляем буквы/цифры/дефис/пробел
    s = re.sub(r"[^0-9a-zа-я\- ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def strip_en_paren(term: str) -> str:
    """
    "данные (data)" -> "данные" (если в скобках латиница)
    """
    t = (term or "").strip()
    m = _PAREN_EN_RE.match(t)
    if m:
        return m.group("ru").strip()
    return term


def tokenize(s: str) -> List[str]:
    s = norm_ru(s)
    if not s:
        return []
    toks = [t for t in s.split() if t and t not in _STOPWORDS]
    return toks


def light_stem(word: str) -> str:
    if len(word) <= 4:
        return word
    for end in _RU_ENDINGS:
        if word.endswith(end) and len(word) - len(end) >= 3:
            return word[:-len(end)]
    return word


def stem_phrase(s: str) -> str:
    toks = tokenize(s)
    stems = [light_stem(t) for t in toks]
    return " ".join(stems)


# ----------------------------
# Fuzzy similarity (token_set_ratio-like) using stdlib
# ----------------------------

def _seq_ratio(a: str, b: str) -> float:
    """0..100"""
    return 100.0 * difflib.SequenceMatcher(a=a, b=b).ratio()


def token_set_ratio(a: str, b: str) -> float:
    """
    Approximation of RapidFuzz/FuzzyWuzzy token_set_ratio using only stdlib.
    Good for multi-word terms with word reorderings.

    Returns 0..100
    """
    a_toks = set(tokenize(a))
    b_toks = set(tokenize(b))
    if not a_toks or not b_toks:
        return 0.0

    inter = a_toks & b_toks
    diff_a = a_toks - inter
    diff_b = b_toks - inter

    s_inter = " ".join(sorted(inter))
    s_a = " ".join(sorted(inter | diff_a))
    s_b = " ".join(sorted(inter | diff_b))

    # берём максимум из нескольких сравнений (как делают token_set_ratio)
    return max(
        _seq_ratio(s_inter, s_a),
        _seq_ratio(s_inter, s_b),
        _seq_ratio(s_a, s_b),
    )


# ----------------------------
# TermScore for a pair
# ----------------------------

def term_score_pair(a: str, b: str) -> float:
    """
    TermScore in [0..1] for two Russian terms (multi-word supported).
    Rules:
      - exact match (normalized) => 1.0
      - match after light stemming => 0.8
      - token_set_ratio thresholds => 0.5..0.7
      - else => 0.0
    """
    a = strip_en_paren(a)
    b = strip_en_paren(b)

    a_n = norm_ru(a)
    b_n = norm_ru(b)
    if not a_n or not b_n:
        return 0.0

    if a_n == b_n:
        return 1.0

    a_s = stem_phrase(a_n)
    b_s = stem_phrase(b_n)
    if a_s and b_s and a_s == b_s:
        return 0.8

    ts = token_set_ratio(a_n, b_n)  # 0..100
    if ts >= 90:
        return 0.7
    if ts >= 80:
        return 0.6
    if ts >= 70:
        return 0.5
    return 0.0


# ----------------------------
# Score list A against list B (best match per A)
# ----------------------------

def score_terms_against_list(
    terms_a: List[str],
    terms_b: List[str],
    min_score: float = 0.0
) -> List[Dict[str, Any]]:
    """
    For each term in A find best term in B by TermScore.
    Returns:
      [{"term_a":..., "best_term_b":..., "term_score":...}, ...]
    Complexity O(|A|*|B|) - fine for a few thousand terms.
    """
    # Precompute B normalized + stemmed for speed
    b_pre: List[Tuple[str, str, str]] = []
    for b in terms_b:
        b2 = strip_en_paren(b)
        b_n = norm_ru(b2)
        b_s = stem_phrase(b_n)
        b_pre.append((b, b_n, b_s))

    results: List[Dict[str, Any]] = []
    for a in terms_a:
        a2 = strip_en_paren(a)
        a_n = norm_ru(a2)
        a_s = stem_phrase(a_n)

        best_b: Optional[str] = None
        best_s: float = 0.0

        # 1) exact norm
        if a_n:
            for (b_orig, b_n, b_s) in b_pre:
                if a_n == b_n:
                    best_b, best_s = b_orig, 1.0
                    break

        # 2) stem equality
        if best_s < 1.0 and a_s:
            for (b_orig, b_n, b_s) in b_pre:
                if a_s == b_s and a_s:
                    if best_s < 0.8:
                        best_b, best_s = b_orig, 0.8

        # 3) fuzzy token_set_ratio
        if best_s < 0.8 and a_n:
            for (b_orig, b_n, b_s) in b_pre:
                if not b_n:
                    continue
                ts = token_set_ratio(a_n, b_n)
                if ts >= 90:
                    s = 0.7
                elif ts >= 80:
                    s = 0.6
                elif ts >= 70:
                    s = 0.5
                else:
                    s = 0.0

                if s > best_s:
                    best_s, best_b = s, b_orig

        if best_s < min_score:
            best_b, best_s = None, 0.0

        results.append({"term_a": a, "best_term_b": best_b, "term_score": float(best_s)})

    return results

# === TERM SYM ===
def TermSim(draft_terms, approved_terms):
    
    result = score_terms_against_list(draft_terms, approved_terms)
    
    return np.mean([r["term_score"] for r in result]), result


# ----------------------------
# Example
# ----------------------------
if __name__ == "__main__":
    A = ["данные", "обработка данных", "система поддержки принятия решений"]
    B = ["данные (data)", "обработка информации", "система поддержки принятия решения"]

    res = score_terms_against_list(A, B, min_score=0.5)
    for r in res:
        print(r)
