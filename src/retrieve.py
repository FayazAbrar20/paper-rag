import json
import sys
from functools import lru_cache
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX = ROOT / "index" / "bge-small_500"
QUERY_PREFIX = {"bge-small": "Represent this sentence for searching relevant passages: "}


@lru_cache(maxsize=None)
def load_model(model_id):
    return SentenceTransformer(model_id)


@lru_cache(maxsize=None)
def load_index(index_dir):
    config = json.loads((index_dir / "config.json").read_text(encoding="utf-8"))
    chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    index = faiss.read_index(str(index_dir / "faiss.index"))
    return config, chunks, index


def retrieve(question, k=4, index_dir=DEFAULT_INDEX):
    config, chunks, index = load_index(Path(index_dir).resolve())
    model = load_model(config["model_id"])
    vector = model.encode([QUERY_PREFIX.get(config["model"], "") + question], normalize_embeddings=True)
    scores, ids = index.search(vector, k)
    return [{**chunks[i], "score": float(s)} for s, i in zip(scores[0], ids[0])]


if __name__ == "__main__":
    for hit in retrieve(" ".join(sys.argv[1:])):
        print(f"{hit['score']:.3f}  {hit['paper']}, page {hit['page']}\n       {hit['text'][:160]}\n")
