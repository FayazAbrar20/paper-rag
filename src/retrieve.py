import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX = ROOT / "index" / "bge-small_300"
DEFAULT_MODE = "hybrid"
MODES = ("vector", "bm25", "hybrid")
QUERY_PREFIX = {"bge-small": "Represent this sentence for searching relevant passages: "}
CANDIDATES = 20
RRF_K = 60
TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def tokenize(text):
    return TOKEN.findall(text.lower())


@lru_cache(maxsize=None)
def load_model(model_id):
    return SentenceTransformer(model_id)


@lru_cache(maxsize=None)
def load_index(index_dir):
    config = json.loads((index_dir / "config.json").read_text(encoding="utf-8"))
    chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    index = faiss.read_index(str(index_dir / "faiss.index"))
    return config, chunks, index


@lru_cache(maxsize=None)
def load_bm25(index_dir):
    _, chunks, _ = load_index(index_dir)
    return BM25Okapi([tokenize(c["text"]) for c in chunks])


def embed_question(question, index_dir):
    config, _, _ = load_index(index_dir)
    model = load_model(config["model_id"])
    return model.encode([QUERY_PREFIX.get(config["model"], "") + question], normalize_embeddings=True)


def vector_ranking(vector, index_dir, n):
    _, _, index = load_index(index_dir)
    _, ids = index.search(vector, n)
    return [int(i) for i in ids[0] if i >= 0]


def bm25_ranking(question, index_dir, n):
    scores = load_bm25(index_dir).get_scores(tokenize(question))
    return [int(i) for i in np.argsort(-scores, kind="stable")[:n] if scores[i] > 0]


def reciprocal_rank_fusion(rankings, k=RRF_K):
    fused = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused, key=lambda chunk_id: (-fused[chunk_id], chunk_id))


def retrieve(question, k=4, index_dir=DEFAULT_INDEX, mode=DEFAULT_MODE):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    index_dir = Path(index_dir).resolve()
    _, chunks, index = load_index(index_dir)
    vector = embed_question(question, index_dir)
    if mode == "vector":
        ids = vector_ranking(vector, index_dir, k)
    elif mode == "bm25":
        ids = bm25_ranking(question, index_dir, k)
    else:
        ids = reciprocal_rank_fusion([vector_ranking(vector, index_dir, CANDIDATES), bm25_ranking(question, index_dir, CANDIDATES)])[:k]
    return [{**chunks[i], "score": float(index.reconstruct(i) @ vector[0])} for i in ids]


if __name__ == "__main__":
    for hit in retrieve(" ".join(sys.argv[1:])):
        print(f"{hit['score']:.3f}  {hit['paper']}, page {hit['page']}\n       {hit['text'][:160]}\n")
