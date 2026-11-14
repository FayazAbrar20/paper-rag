import argparse
import json
import re
from collections import Counter
from pathlib import Path

import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
PAPERS = {
    "IDS_final_project_group_5.pdf": "Sarcasm Detection in Social Media Text",
    "J G09 Paper.pdf": "Bitcoin Price Direction Prediction",
    "nlp_final_project_group_11.pdf": "Emotion Detection: Classical ML vs Transformers",
}
MODELS = {
    "bge-small": "BAAI/bge-small-en-v1.5",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}


def load_pages(papers_dir):
    pages = []
    for pdf in sorted(papers_dir.glob("*.pdf")):
        paper = PAPERS.get(pdf.name, pdf.stem)
        for number, page in enumerate(PdfReader(pdf).pages, start=1):
            text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
            if text:
                pages.append({"paper": paper, "page": number, "text": text})
    return pages


def chunk_page(text, tokenizer, size, overlap):
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)["offset_mapping"]
    chunks = []
    for start in range(0, len(offsets), size - overlap):
        end = min(start + size, len(offsets))
        chunks.append((text[offsets[start][0]:offsets[end - 1][1]], end - start))
        if end == len(offsets):
            break
    return chunks


def build(model_name, size, overlap, out_dir):
    model = SentenceTransformer(MODELS[model_name])
    chunks = []
    for page in load_pages(ROOT / "data" / "papers"):
        for text, tokens in chunk_page(page["text"], model.tokenizer, size, overlap):
            chunks.append({"chunk_id": len(chunks), "paper": page["paper"], "page": page["page"], "tokens": tokens, "text": text})
    embeddings = model.encode([c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "faiss.index"))
    (out_dir / "chunks.json").write_text(json.dumps(chunks, indent=1, ensure_ascii=False), encoding="utf-8")
    config = {"model": model_name, "model_id": MODELS[model_name], "chunk_size": size, "overlap": overlap, "max_seq_length": model.max_seq_length}
    (out_dir / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    return chunks, model.max_seq_length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="bge-small")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out_dir = args.out or ROOT / "index" / f"{args.model}_{args.chunk_size}"
    chunks, max_len = build(args.model, args.chunk_size, args.overlap, out_dir)
    for paper, count in Counter(c["paper"] for c in chunks).items():
        print(f"{count:4d}  {paper}")
    print(f"{len(chunks):4d}  total -> {out_dir}")
    truncated = sum(c["tokens"] + 2 > max_len for c in chunks)
    print(f"{truncated:4d}  chunks longer than the model limit ({max_len} tokens)")


if __name__ == "__main__":
    main()
