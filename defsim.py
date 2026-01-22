from typing import List, Dict, Any, Optional, Tuple
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


GOST_REF_BRACKET = re.compile(r"\[[^\]]*ГОСТ[^\]]*\]", re.IGNORECASE)
GOST_REF_PAREN   = re.compile(r"\([^\)]*ГОСТ[^\)]*\)", re.IGNORECASE)
GOST_INLINE      = re.compile(r"\bГОСТ\s*[РA-Z]*\s*\d+[^\s]*\s*[—-]\s*\d{4}\b", re.IGNORECASE)
NOTE_SPLIT       = re.compile(r"\bПримечание\s*[—-]", re.IGNORECASE)

def clean_definition(text: str, max_sentences: int = 2) -> str:
    t = (text or "").strip()

    # cut notes
    t = NOTE_SPLIT.split(t)[0].strip()

    # remove gost refs
    t = GOST_REF_BRACKET.sub("", t)
    t = GOST_REF_PAREN.sub("", t)
    t = GOST_INLINE.sub("", t)

    # normalize spaces
    t = re.sub(r"\s+", " ", t).strip()

    # keep first N sentences (simple splitter)
    if max_sentences is not None and max_sentences > 0:
        parts = re.split(r"(?<=[.!?])\s+", t)
        t = " ".join(parts[:max_sentences]).strip()

    return t

def DefSim(
    defs_a: List[str],
    defs_b: List[str],
    ollama_embed,
    batch_size: int = 32,
    tau: float = 0.6,
    clean: bool = True,
    top_k_pairs: int = 10,
    top_k_per_a: int = 0,
    top_k_per_b: int = 0,
) -> Tuple[float, Dict[str, Any]]:
    if not defs_a or not defs_b:
        return 0.0, {
            "mean_best_a": 0.0, "mean_best_b": 0.0,
            "coverageA": 0.0, "coverageB": 0.0,
            "f1_coverage": 0.0, "tau": float(tau),
            "score": 0.0,
            "closest_pairs": [],
            "best_for_a": [],
            "best_for_b": [],
        }

    if clean:
        defs_a = [clean_definition(x) for x in defs_a]
        defs_b = [clean_definition(x) for x in defs_b]

    def embed_all(texts: List[str]) -> np.ndarray:
        embs = []
        for i in range(0, len(texts), batch_size):
            chunk = [(t or "").strip() for t in texts[i:i + batch_size]]
            embs.extend(ollama_embed(chunk))
        return np.asarray(embs, dtype=np.float32)

    A = embed_all(defs_a)
    B = embed_all(defs_b)

    S = cosine_similarity(A, B)  # |A| x |B|

    best_a = S.max(axis=1)
    best_b = S.max(axis=0)

    mean_best_a = float(best_a.mean())
    mean_best_b = float(best_b.mean())

    coverageA = float((best_a >= tau).mean())
    coverageB = float((best_b >= tau).mean())

    f1cov = (2 * coverageA * coverageB) / (coverageA + coverageB + 1e-9)
    mean_best = 0.5 * (mean_best_a + mean_best_b)
    score = mean_best * f1cov

    # ---------- closest global pairs (top_k_pairs) ----------
    closest_pairs = []
    if top_k_pairs and top_k_pairs > 0:
        k = min(int(top_k_pairs), S.size)
        flat = S.ravel()
        # топ-k без полной сортировки
        idx = np.argpartition(flat, -k)[-k:]
        idx = idx[np.argsort(flat[idx])[::-1]]  # отсортировать убыванию
        ia, ib = np.unravel_index(idx, S.shape)

        closest_pairs = [
            {
                "a_idx": int(i),
                "b_idx": int(j),
                "a_text": defs_a[int(i)],
                "b_text": defs_b[int(j)],
                "sim": float(S[int(i), int(j)]),
            }
            for i, j in zip(ia, ib)
        ]

    # ---------- top matches for each A (optional) ----------
    best_for_a = []
    if top_k_per_a and top_k_per_a > 0:
        k = min(int(top_k_per_a), S.shape[1])
        for i in range(S.shape[0]):
            row = S[i]
            jj = np.argpartition(row, -k)[-k:]
            jj = jj[np.argsort(row[jj])[::-1]]
            best_for_a.append({
                "a_idx": int(i),
                "a_text": defs_a[int(i)],
                "matches": [
                    {"b_idx": int(j), "b_text": defs_b[int(j)], "sim": float(row[int(j)])}
                    for j in jj
                ],
            })

    # ---------- top matches for each B (optional) ----------
    best_for_b = []
    if top_k_per_b and top_k_per_b > 0:
        k = min(int(top_k_per_b), S.shape[0])
        for j in range(S.shape[1]):
            col = S[:, j]
            ii = np.argpartition(col, -k)[-k:]
            ii = ii[np.argsort(col[ii])[::-1]]
            best_for_b.append({
                "b_idx": int(j),
                "b_text": defs_b[int(j)],
                "matches": [
                    {"a_idx": int(i), "a_text": defs_a[int(i)], "sim": float(col[int(i)])}
                    for i in ii
                ],
            })

    stats: Dict[str, Any] = {
        "mean_best_a": mean_best_a,
        "mean_best_b": mean_best_b,
        "coverageA": coverageA,
        "coverageB": coverageB,
        "f1_coverage": float(f1cov),
        "tau": float(tau),
        "score": float(score),
        "closest_pairs": closest_pairs,
        "best_for_a": best_for_a,
        "best_for_b": best_for_b,
    }
    return float(score), stats

