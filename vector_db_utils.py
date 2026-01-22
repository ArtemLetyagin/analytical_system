import chromadb
from typing import Any, Dict, List, Optional
from OllamaEmbedder import OllamaEmbedder
import hashlib
import random

def truncate_collection(client, collection_name):
    collection = client.get_or_create_collection(
        name=collection_name
    )

    ids = collection.get()["ids"]

    if len(ids) > 0:
        collection.delete(
            ids=ids,
        )

def delete_collections(client, collection_name):
    if collection_name in [col.name for col in client.list_collections()]:
        client.delete_collection(collection_name)
        print(f"Коллекция '{collection_name}' удалена")
    else:
        print(f"Коллекция '{collection_name}' не существует")

def build_retrieval_text(r: Dict[str, Any]) -> str:
    parts = []
    parts.append(f"TERM_RU: {r.get('term_ru','')}")
    if r.get("term_en"):
        parts.append(f"TERM_EN: {r['term_en']}")
    abbr = []
    if r.get("abbr_ru"):
        abbr.append(r["abbr_ru"])
    if r.get("abbr_en"):
        abbr.append(r["abbr_en"])
    if abbr:
        parts.append(f"ABBR: {' / '.join(abbr)}")
    parts.append(f"DEF: {r.get('definition','')}")
    if r.get("section_path"):
        parts.append(f"CTX: {r['section_path']}")
    return "\n".join(parts)

def stable_id(standard_id: str, article: Optional[str], term_ru: str, suffix: str) -> str:
    base = f"{standard_id}|{article or ''}|{term_ru}|{suffix}|{random.random()}"
    h = hashlib.sha1(base.encode("utf-8")).hexdigest()
    return h

def to_metadata(r: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    md = {
        "standard_id": r.get("standard_id"),
        "source_file": r.get("source_file"),
        "article": r.get("article"),
        "term_ru": r.get("term_ru"),
        "term_ru_norm": r.get("term_ru_norm") or (r.get("term_ru") or "").lower().replace("ё", "е"),
        "term_en": r.get("term_en"),
        "abbr_ru": r.get("abbr_ru"),
        "abbr_en": r.get("abbr_en"),
        "section_path": r.get("section_path", ""),
        "def_len": len(r.get("definition", "") or ""),
        "doc_type": "term",
    }
    md.update(extra)
    # Chroma metadata values must be str/int/float/bool (no None)
    clean = {}
    for k, v in md.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean

def upsert_terms(
    client: chromadb.Client,
    collection_name: str,
    rows: List[Dict[str, Any]],
    embedder: OllamaEmbedder,
    text_builder,
    extra_md: Dict[str, Any],
    batch_size: int = 64,
):
    col = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    ids: List[str] = []
    docs: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for r in rows:
        term = r.get("term_ru", "")
        article = r.get("article")
        _id = stable_id(r.get("standard_id","STD"), article, term, suffix=collection_name)
        ids.append(_id)
        docs.append(text_builder(r))
        metadatas.append(to_metadata(r, extra_md))

    for bi in range(0, len(ids), batch_size):
        batch_ids = ids[bi:bi+batch_size]
        batch_docs = docs[bi:bi+batch_size]
        batch_mds = metadatas[bi:bi+batch_size]
        batch_emb = embedder.embed(batch_docs)

        col.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_mds,
            embeddings=batch_emb,
        )

def retrieve(query_text, collection, ollama_embed, where_rule, top_k):
    q_emb = ollama_embed([query_text], model="nomic-embed-text")[0]

    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        where=where_rule,  # опционально
        include=["documents", "metadatas", "distances"]
    )

    return res
