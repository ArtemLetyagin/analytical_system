import requests
from typing import List
import requests


class OllamaEmbedder:
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def embed(self, texts: List[str]) -> List[List[float]]:
        # Ollama supports embedding endpoint:
        # POST /api/embeddings  { "model": "...", "prompt": "text" }
        # We'll batch by looping (simple & reliable).
        out: List[List[float]] = []
        for t in texts:
            t = " ".join(t.split()[:256])
            resp = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": t},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            out.append(data["embedding"])
        return out

def ollama_embed(texts: List[str], model: str = "nomic-embed-text", base_url: str = "http://localhost:11434") -> List[List[float]]:
    """
    Возвращает список эмбеддингов (один embedding на один текст).
    """
    base_url = base_url.rstrip("/")
    embs = []
    for t in texts:
        t = " ".join(t.split()[:256])
        r = requests.post(
            f"{base_url}/api/embeddings",
            json={"model": model, "prompt": t},
            timeout=120,
        )
        r.raise_for_status()
        embs.append(r.json()["embedding"])
    return embs
